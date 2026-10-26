"""Freeze a paper-only run before cluster submission (no GPU needed)."""
from __future__ import annotations
import argparse, copy, shutil
from pathlib import Path
from .common import *

FIGURE_MAP = {
    "Figure1": ["navigation", "source_lock", "navigation_display"],
    "Figure2": ["exp5_navigation", "exp5_ctl01"],
    "Figure3": ["exp5_mapping", "exp5_test_transforms", "adapter_rescue"],
    "Figure4": ["symbolic_adapter"],
    "FigureA1": ["torus_topology"],
    "FigureA2": ["hard_benchmark"],
    "FigureA3": ["source_splits", "source_reduced_v3", "factorial_confirmatory", "action_causality", "path_invariance"],
    "FigureA4": ["sequence_control", "control_qualification"],
}
EXCLUDED = ["nav_smoke (formal cohort)", "nav_512_ref", "nav_512_gridtuned", "nav_desaturation",
            "exp2_abstract_graph", "exp2b_comparison", "exp3_ablation", "constraint_ablation",
            "exp4_visual", "exp6_visual_concept", "toy_memory", "exp9_potent_null",
            "adapter_symbolic_heldout_pairs", "cross_graph", "checkpoint_sweep", "velocity_training_sweep",
            "hard_pilot", "factorial_screening", "shared_manifold_v2", "manifold_interventions"]


def validate(cfg: dict) -> None:
    if cfg.get("schema") != "multinav_v1": raise ValueError("Unknown config schema")
    safe_name(cfg["run_id"])
    source = cfg["source_seeds"]; control = cfg["control_source_seeds"]
    if len(source) != len(control) or not source: raise ValueError("Supply one independent CTL seed per navigation seed")
    if len(set(source)) != len(source) or len(set(control)) != len(control): raise ValueError("Duplicate source seeds")
    if set(source) & set(control): raise ValueError("Keep NAV and CTL seed blocks disjoint")
    if any(not isinstance(s, int) or s < 0 for s in source+control): raise ValueError("Source seeds must be nonnegative integers")
    for key in ["target_repeats", "hard_target_repeats", "factorial_target_repeats"]:
        if not 1 <= int(cfg[key]) <= 30: raise ValueError(f"{key} must be in 1..30")
    max_target = int(cfg["target_seed_base"]) + 100*(len(source)-1) + 60 + int(cfg["factorial_target_repeats"])-1
    if int(cfg["target_seed_base"]) < 0 or max_target*1000003 + 1000*101 >= 2**32:
        raise ValueError("Target seed block would overflow the original hard/factorial NumPy seed formula")
    if cfg["checkpoint_policy"] not in {"best_validation", "balanced_else_validation"}:
        raise ValueError("Unsupported source-only checkpoint policy")
    if cfg["control_qualification_policy"] != "retain_all_and_flag": raise ValueError("Do not filter source pairs by CTL success")
    if cfg.get("profile", "formal") not in {"formal", "smoke"}: raise ValueError("Unknown profile")
    topo = cfg.get("schoyen_topology") or {}
    for key in ["cluster_trajectories","tda_trajectories","min_cluster_size","top_band_clusters",
                "downsample_points","ripser_n_perm"]:
        if key not in topo or int(topo[key]) < 1:
            raise ValueError(f"schoyen_topology.{key} must be a positive integer")
    for key in ["h1_gap_threshold","h2_gap_threshold"]:
        if float(topo.get(key,0)) <= 1.0:
            raise ValueError(f"schoyen_topology.{key} must be > 1")
    if Path(cfg["output_root"]).is_absolute() or ".." in Path(cfg["output_root"]).parts:
        raise ValueError("output_root must be a new project-relative directory")
    if not str(cfg["output_root"]).startswith("outputs/multinav"):
        raise ValueError("Use outputs/multinav...; legacy result roots are protected")


def count_fits(cfg):
    m = int(cfg["target_repeats"]); h = int(cfg["hard_target_repeats"]); f = int(cfg["factorial_target_repeats"])
    per_source = {"Exp5_main_6_arms": 6*m, "Exp5_CTL_1_arm": m,
                  "Exp5_mapping_2_new_arms": 2*m, "Exp7_32_arms_transforms": 32*m,
                  "Exp8_38_arm_budget_cells": 38*m,
                  "hard_6_arms_x7_budgets": 42*h,
                  "factorial_8_cells_x7_budgets": 56*f}
    return {"source_trainings": 2*len(cfg["source_seeds"]), "target_fits_per_source": per_source,
            "total_target_fits_per_source": sum(per_source.values()),
            "total_target_fits": len(cfg["source_seeds"])*sum(per_source.values()),
            "note": "Formal schedule; test-time transforms, recurrence ablations, topology and reduced-model fits are additional evaluations, not counted as target neural-network trainings."}


def make_plan(root: Path, config_path: Path, *, run_id: str | None = None, smoke=False):
    cfg = read_yaml(config_path)
    if run_id: cfg["run_id"] = run_id
    if smoke:
        cfg.update(profile="smoke", source_seeds=cfg["source_seeds"][:2],
                   control_source_seeds=cfg["control_source_seeds"][:2], target_repeats=1,
                   hard_target_repeats=1, factorial_target_repeats=1, source_epochs=1,
                   source_steps_per_epoch=1, device="cpu", threads=1,
                   bootstrap_samples=200)
        cfg["schoyen_topology"] = dict(cfg["schoyen_topology"])
        cfg["schoyen_topology"].update(cluster_trajectories=400,tda_trajectories=300,
            rollout_batch_size=100,rotation_resolution=90,min_cluster_size=8,
            top_band_clusters=3,candidate_union_depth=2,downsample_neighbors=100,
            downsample_points=250,ripser_n_perm=100,umap_neighbors=10,dbscan_min_samples=5,
            max_missing_fraction_cluster=.25,max_missing_fraction_tda=.25)
    validate(cfg)
    run = root / cfg["output_root"] / cfg["run_id"]
    if (run / "plan.json").exists():
        old = read_json(run / "plan.json")
        if old["config"] != cfg: raise RuntimeError("Run ID already frozen with a different configuration; choose a new ID")
        verify_tree(root / old["code_root"], old["code_manifest"])
        return run / "plan.json"
    if run.exists() and any(run.iterdir()): raise RuntimeError(f"Nonempty unplanned run directory: {run}")
    run.mkdir(parents=True, exist_ok=True)
    code = run / "code"; code.mkdir()
    manifest = tree_hash(root)
    for name in manifest["files"]:
        dst = code / name; dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(root/name, dst)
    verify_tree(code, manifest)
    pairs=[]
    for i, (s, c) in enumerate(zip(cfg["source_seeds"], cfg["control_source_seeds"])):
        b = int(cfg["target_seed_base"]) + 100*i
        pairs.append(dict(source_seed=s, control_seed=c,
                          main_target_seeds=list(range(b, b+int(cfg["target_repeats"]))),
                          hard_target_seeds=list(range(b+30, b+30+int(cfg["hard_target_repeats"]))),
                          factorial_target_seeds=list(range(b+60, b+60+int(cfg["factorial_target_repeats"])))))
    # The paper mentions six contrasts, but the actual driver defines eight.
    # Freeze the executable list, disclose the discrepancy, do not invent six.
    import ast
    p = code / "gpm_cmap/experiments/exp5_recurrent_factorial.py"
    tree = ast.parse(p.read_text())
    contrasts = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == "DEFAULT_CONFIRMATORY_COMPARISONS" for t in n.targets))
    payload = dict(schema="multinav_plan_v1", created_utc=now(), config=cfg,
                   project_at_preparation=str(root), run_root=rel(run, root), code_root=rel(code, root),
                   code_manifest=manifest, source_pairs=pairs, figure_map=FIGURE_MAP,
                   excluded_routes=EXCLUDED, workload=count_fits(cfg),
                   factorial_contrasts=contrasts,
                   protocol_notes=[
                       "Navigation source is the highest sampling unit; paired CTL pretraining is part of each independent source block.",
                       "150-epoch/150000-update new cohort is not a numerical rerun of the archived 50-epoch source record.",
                       "No source is replaced after topology, transfer or qualification results are observed.",
                       "Behavioural hard/factorial tasks require source weights, not a successful torus reduction.",
                       "Legacy Source-v3 phase diagnostics are retired with the old topology route; the new topology analysis is representational and uses validation best.pt.",
                       "Original factorial code defines 8 comparisons, whereas manuscript Appendix A.2 states 6. This run explicitly locks 8 with multiplicity correction.",
                   ])
    payload["plan_sha256"] = digest(payload)
    write_json(run / "plan.json", payload); write_yaml(run / "config_frozen.yaml", cfg)
    write_json(run / "figure_experiment_map.json", FIGURE_MAP)
    write_json(run / "workload.json", payload["workload"])
    return run / "plan.json"


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="multinav/config.yaml"); ap.add_argument("--project")
    ap.add_argument("--run-id"); ap.add_argument("--smoke", action="store_true")
    args=ap.parse_args(); root=project_root(args.project)
    p=make_plan(root, root/args.config, run_id=args.run_id, smoke=args.smoke)
    print(rel(p, root)); print(read_json(p)["workload"])

if __name__ == "__main__": main()
