"""Small, explicit adapters over the supplied scientific implementation.

No original file is edited. Training losses, task generators and evaluation
functions remain the supplied implementations. Adapters only select routes,
cache completed cells, remove a behavioural dependency on successful topology,
and make qualification/missingness explicit.
"""
from __future__ import annotations
import argparse, functools, importlib, inspect, json, os, sys
from pathlib import Path
from types import SimpleNamespace
from .common import *


def quiet_plots(module):
    # Old per-condition auxiliary figures are not the requested manuscript set.
    for name in list(vars(module)):
        if name.startswith("plot_") and callable(getattr(module, name)):
            setattr(module, name, lambda *a, **k: None)


def call_main(module, cfg_path, extra=()):
    sys.argv = [module.__name__, "--config", str(cfg_path), *map(str, extra)]
    module.main()


def cached(module, name, cache_root: Path):
    original = getattr(module, name); signature = inspect.signature(original)
    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        bound = signature.bind(*args, **kwargs).arguments
        key = {k: bound.get(k) for k in ("arm", "source_ckpt", "source_type", "budget", "seed", "transform_spec")}
        path = cache_root / (name + "_" + digest(key) + ".json")
        if path.exists(): return read_json(path)
        result = original(*args, **kwargs)
        write_json(path, result)
        return result
    setattr(module, name, wrapped)


def retain_ctl_validator(module):
    """Allow intent-to-train comparison while preserving actual gate in manifest."""
    original = module.validate_primary_transfer_checkpoint
    def validate(cfg, checkpoint_path, transfer_mode):
        from gpm_cmap.experiments.exp8_symbolic_adapter import validate_pretrained_sources
        from gpm_cmap.utils.checkpointing import load_checkpoint
        payload = load_checkpoint(checkpoint_path, map_location="cpu")
        if payload.get("pretraining_source_type") != "non_navigation_delayed_reconstruction":
            return original(cfg, checkpoint_path, transfer_mode)
        entry = validate_pretrained_sources([dict(arm="ctl01_frozen", checkpoint=checkpoint_path,
                  source_type="sequence_control", require_gate=False)], cfg)[0]
        entry.update(qualification_policy="retain_all_and_flag", training_gate_required=False,
                     qualified=(payload.get("ctl01_gate") or {}).get("passed") is True,
                     note="Unqualified sources are retained, not described as qualified; report sensitivity separately.")
        return entry
    module.validate_primary_transfer_checkpoint = validate


def behaviour_source_link(module, cfg):
    """Hard/factorial TRAINING never calls a reduced model: only metadata is used.

    This is not a fake successful Source-v3 artifact. The explicit JSON identity
    has no model, no source-reduced hash and no torus claim. Genuine phase-based
    analyses still load and validate the original Source-v3 pickle themselves.
    """
    identity_path = Path(cfg["source_reduced_artifact"])
    identity = read_json(identity_path)
    if sha(identity["navigation_checkpoint"]) != identity["navigation_checkpoint_sha256"]:
        raise RuntimeError("Navigation identity checksum mismatch")
    def load(_path):
        if Path(_path) != identity_path: raise ValueError("Unexpected behavioural identity path")
        return SimpleNamespace(path=identity_path, sha256=None)
    def linkage(_source):
        return {"claim_scope": "conditional on this independent navigation source",
                "source_reference_kind": "navigation_checkpoint_only",
                "source_seed": identity["source_seed"],
                "navigation_checkpoint": identity["navigation_checkpoint"],
                "navigation_checkpoint_sha256": identity["navigation_checkpoint_sha256"],
                "source_reduced_artifact": None, "source_reduced_artifact_sha256": None,
                "source_classification": "not_required_for_behavioural_training"}
    module.load_source_reduced_artifact = load
    module.source_linkage_payload = linkage


def validate_mechanism_link(module, cfg):
    """Retain the original Source-v3 gates and additionally check nav source ID."""
    from gpm_cmap.utils.checkpointing import load_checkpoint
    expected = cfg["multinav_navigation_sha256"]
    source = module.load_source_reduced_artifact(cfg["source_reduced_artifact"])
    if source.provenance.get("checkpoint_sha256") != expected:
        raise RuntimeError("Source-v3 reduced model was fitted to a different navigation source")
    checkpoints = module._discover_checkpoints(cfg)
    if not checkpoints: raise ValueError("No target checkpoints match the mechanism request")
    for path in checkpoints:
        payload = load_checkpoint(path, map_location="cpu")
        observed = payload.get("source_linkage", {}).get("navigation_checkpoint_sha256")
        if observed != expected:
            raise RuntimeError(f"Mechanism/target source mismatch: {path}")


def mapping(cfg):
    import torch
    import gpm_cmap.experiments.exp5_abstract_pathint as m
    from gpm_cmap.utils import configure_runtime, save_config_snapshot
    from gpm_cmap.utils.logging_utils import get_logger
    quiet_plots(m); out=Path(cfg["output_dir"]); out.mkdir(parents=True, exist_ok=True)
    save_config_snapshot(cfg, out); device=configure_runtime(cfg)
    logger=get_logger("multinav_mapping", out/"experiment.log")
    specs={name:dict(transfer=True, freeze=False, freeze_mode="all",
                    checkpoint_path=cfg["transfer_checkpoint"], transfer_mode="full", action_map_mode=name)
           for name in ("shuffled", "mismatched_target")}
    _, _, raw=m.run_block("mapping_control", specs, cfg, 1, device, logger, out, main_block=False)
    write_json(out/"raw_per_seed.json", {"mapping_control":raw})


def target_loaders(cfg, seed, transform="none"):
    from gpm_cmap.data.graph_tasks.pathint_episodes import build_pathint_loaders
    d=cfg["data"]
    return build_pathint_loaders(task_name=d.get("task_name","social_hierarchy"),
        num_episodes=d["num_episodes"], seq_len=d["seq_len"], batch_size=d["batch_size"],
        split_ratio=d.get("split_ratio",.7), velocity_scale=d["velocity_scale"], seed=seed,
        n_levels=d.get("n_levels",4), n_items=d.get("n_items",7), action_map_mode="aligned",
        coord_strategy=d.get("coord_strategy"), coord_dim=d.get("coord_dim",2),
        normalize_displacements=d.get("normalize_displacements",True), coord_scale=d.get("coord_scale",.02),
        split_strategy=d.get("split_strategy","random"), velocity_transform="none",
        test_only_velocity_transform=transform, seq_len_range=d.get("seq_len_range"),
        validation_ratio=d.get("validation_ratio",.1), return_validation=True, num_workers=d.get("num_workers",0))


def test_transforms(cfg):
    from gpm_cmap.experiments.exp5_abstract_pathint import build_model
    from gpm_cmap.tasks.abstract_pathint import evaluate_pathint_basic
    from gpm_cmap.utils import configure_runtime
    from gpm_cmap.utils.checkpointing import load_checkpoint
    import torch
    out=Path(cfg["output_dir"]); out.mkdir(parents=True, exist_ok=True)
    raw=read_json(Path(cfg["multinav_main_output"])/"raw_per_seed.json")["main"]
    device=configure_runtime(cfg); rows=[]
    for arm in ("transfer_freeze", "random_frozen"):
        entry=raw[arm][0]
        checkpoint=entry.get("best_checkpoint") or entry.get("model_checkpoint")
        if not checkpoint:
            # Native Exp5's scalar extractor preserves best_checkpoint in this release.
            checkpoint=str(Path(cfg["multinav_main_output"])/f"seed_{cfg['seed']}"/arm/f"{arm}_model.pt")
        saved=load_checkpoint(checkpoint, map_location=device)
        for transform in ("none","rotate90","scale2x","signflip"):
            loaders=target_loaders(cfg,int(cfg["seed"]),transform)
            model=build_model(cfg,len(loaders[4]),device); model.load_state_dict(saved["model_state"])
            result=evaluate_pathint_basic(model,loaders[2],device)
            value=float(result["accuracy"])
            if transform == "none" and abs(value-float(entry["test_acc"])) > 1e-6:
                raise RuntimeError("Reused checkpoint does not reproduce the locked intact test score")
            rows.append(dict(arm=arm, transform=transform, seed=int(cfg["seed"]), test_accuracy=value,
                             reused_checkpoint=checkpoint, checkpoint_sha256=sha(checkpoint), retrained=False))
    write_json(out/"test_transform_rows.json",rows)


def run(kind, cfg_path):
    cfg=read_yaml(cfg_path)
    prefix="gpm_cmap.experiments."
    modules={"navigation":"exp1_nav_pretrain", "sequence_control":"pretrain_sequence_control",
             "exp5":"exp5_abstract_pathint", "exp5_ctl":"exp5_abstract_pathint",
             "adapter":"exp7_adapter_rescue", "symbolic":"exp8_symbolic_adapter",
             "topology":"analyze_schoyen_torus",
             "source_splits":"export_source_reduced_splits", "source_reduced":"analyze_source_reduced_model",
             "hard":"exp5_hard_benchmark", "factorial":"exp5_recurrent_factorial",
             "action":"analyze_target_action_causality", "path":"analyze_path_invariance"}
    if kind == "mapping": return mapping(cfg)
    if kind == "test_transforms": return test_transforms(cfg)
    if kind not in modules: raise ValueError(f"Unknown stage kind: {kind}")
    module=importlib.import_module(prefix+modules[kind]); quiet_plots(module)
    if kind == "navigation":
        original=module._analyse_navigation_checkpoint
        def without_integrated_topology(*args,**kwargs):
            kwargs["run_topology"]=False
            # Reuse scores/maps already computed by the scientific evaluation.
            # Saving them avoids another 4096-unit scoring pass in the report job.
            full=module._full_navigation_evaluation; captured={}
            def record(*a,**k):
                result=full(*a,**k); captured["stats"]=result; return result
            module._full_navigation_evaluation=record
            try: summary=original(*args,**kwargs)
            finally: module._full_navigation_evaluation=full
            import numpy as np
            grids=captured["stats"]["grid_results"]
            target=Path(kwargs["output_dir"])/"checkpoint_analysis"/kwargs["tag"]
            np.savez_compressed(target/"multinav_grid_display.npz",
                unit_ids=np.asarray([z["unit"] for z in grids]),
                grid_scores=np.asarray([z.get("grid_score",float("nan")) for z in grids]),
                conservative_scores=np.asarray([z.get("grid_score_conservative",float("nan")) for z in grids]),
                rate_maps=np.asarray([z["rate_map"] for z in grids]))
            return summary
        module._analyse_navigation_checkpoint=without_integrated_topology
        call_main(module,cfg_path)
        history=read_json(Path(cfg["output_dir"])/"history.json")
        if [x["epoch"] for x in history] != list(range(1,int(cfg["train"]["epochs"])+1)):
            raise RuntimeError("Incomplete source training history")
    elif kind == "exp5":
        call_main(module,cfg_path,["--conditions","transfer_freeze","transfer_finetune","random_init",
                                   "random_frozen","transfer_input_only","transfer_recurrent_only"])
    elif kind == "exp5_ctl":
        retain_ctl_validator(module); call_main(module,cfg_path,["--conditions","transfer_freeze"])
    elif kind in {"adapter","symbolic"}:
        cache_root=Path(cfg["multinav_cell_cache"])
        cached(module,"run_one",cache_root)
        if kind == "symbolic": cached(module,"run_simple_baseline",cache_root)
        call_main(module,cfg_path)
    elif kind == "topology":
        extra=["--checkpoint",cfg["multinav_checkpoint"],"--output_dir",cfg["output_dir"],
               "--seed",cfg["seed"],"--device",cfg.get("device","cuda")]
        if cfg.get("profile") == "smoke": extra.append("--smoke")
        call_main(module,cfg_path,extra)
    elif kind == "source_splits":
        call_main(module,cfg_path,["--checkpoint",cfg["multinav_checkpoint"],"--nav_analysis_dir",cfg["multinav_topology"],
            "--output_dir",cfg["output_dir"],"--fit_trajectories",cfg["multinav_fit_trajectories"],
            "--validation_trajectories",cfg["multinav_validation_trajectories"],
            "--test_trajectories",cfg["multinav_test_trajectories"],"--seq_len",cfg["multinav_long_length"],
            "--batch_size",cfg["multinav_export_batch_size"],"--no_non_grid_control"])
    elif kind in {"hard","factorial"}:
        behaviour_source_link(module,cfg); call_main(module,cfg_path,["--mode","confirmatory"])
    elif kind in {"action","path"}:
        validate_mechanism_link(module,cfg); call_main(module,cfg_path)
    else:
        call_main(module,cfg_path)


def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument("kind"); ap.add_argument("--config",required=True)
    args=ap.parse_args(); run(args.kind,args.config)

if __name__ == "__main__": main()
