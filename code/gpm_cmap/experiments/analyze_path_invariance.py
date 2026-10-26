"""Stage I3C: evaluation-only same-start/same-end path independence tests."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from gpm_cmap.analysis.hard_pathint import load_target_checkpoint
from gpm_cmap.analysis.path_invariance import evaluate_path_invariance
from gpm_cmap.analysis.upgrade_contracts import (
    CLAIM_SCOPE, build_artifact_manifest, hash_jsonable,
    load_source_reduced_artifact, source_linkage_payload, write_csv_strict,
)
from gpm_cmap.data.graph_tasks.hard_pathint import build_hard_pathint_pools
from gpm_cmap.data.graph_tasks.path_invariance import build_path_invariance_dataset
from gpm_cmap.data.graph_tasks.pathint_episodes import episode_fingerprint
from gpm_cmap.utils import (
    configure_runtime, load_config, prepare_output_dir, save_config_snapshot,
    save_run_provenance, set_seed, write_json,
)
from gpm_cmap.utils.checkpointing import sha256_file
from gpm_cmap.utils.logging_utils import get_logger

SCHEMA_VERSION = "path_invariance_v1"


def _discover_checkpoints(cfg: dict[str, Any]) -> list[Path]:
    target = cfg["target"]
    paths = [Path(value) for value in target.get("checkpoints", [])]
    if target.get("checkpoint_manifest"):
        manifest_path = Path(target["checkpoint_manifest"])
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        conditions = set(target.get("conditions", []))
        budgets = {int(v) for v in target.get("episode_budgets", [])}
        for row in payload.get("checkpoints", []):
            condition = row.get("condition") or row.get("cell")
            budget = int(row.get("unique_training_episodes", -1))
            if conditions and condition not in conditions:
                continue
            if budgets and budget not in budgets:
                continue
            paths.append(Path(row["path"]))
    unique = []
    seen = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved); unique.append(path)
    if not unique:
        raise RuntimeError("No target checkpoints selected for path invariance")
    for path in unique:
        if not path.is_file():
            raise FileNotFoundError(path)
    return unique


def _condition(payload: dict[str, Any]) -> str:
    condition = payload.get("condition", {})
    return str(condition.get("name") or condition.get("cell") or "unknown")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    output_dir = prepare_output_dir(cfg["output_dir"])
    save_config_snapshot(cfg, output_dir)
    logger = get_logger("path_invariance", output_dir / "analysis.log")
    device = configure_runtime(cfg)
    set_seed(int(cfg.get("seed", 42)))
    source = load_source_reduced_artifact(cfg["source_reduced_artifact"])
    source_link = source_linkage_payload(source)
    checkpoints = _discover_checkpoints(cfg)
    per_seed = []
    group_rows_all = []
    variant_rows_all = []
    dataset_manifests = []
    checkpoint_rows = []
    for checkpoint_path in checkpoints:
        model, payload = load_target_checkpoint(checkpoint_path, device=device)
        linked_source = payload.get("source_linkage", {}).get("source_reduced_artifact_sha256")
        if linked_source and linked_source != source.sha256:
            raise RuntimeError("Target checkpoint and path analysis use different Source v3 artifacts")
        target_seed = int(payload["target_seed"])
        condition = _condition(payload)
        budget = int(payload["unique_training_episodes"])
        data_cfg = dict(cfg["data"])
        data_cfg["episode_budgets"] = sorted(set([budget] + [int(v) for v in data_cfg.get("episode_budgets", [budget])]))
        hard_pools = build_hard_pathint_pools(data_cfg, seed=target_seed)
        hard_hash = hash_jsonable(hard_pools.manifest)
        if hard_hash != payload.get("dataset_manifest_hash"):
            raise RuntimeError(
                f"Hard dataset fingerprint mismatch for {checkpoint_path}: "
                f"checkpoint={payload.get('dataset_manifest_hash')} regenerated={hard_hash}"
            )
        forbidden = {
            episode_fingerprint(ep)
            for name in ("train_pool", "validation", "id_test", "long_test", "composition_test")
            for ep in getattr(hard_pools, name)
        }
        path_cfg = dict(cfg["path_dataset"])
        path_cfg.setdefault("n_levels", data_cfg.get("n_levels", 8))
        path_cfg.setdefault("velocity_scale", data_cfg.get("velocity_scale", 0.02))
        path_cfg.setdefault("coord_scale", data_cfg.get("coord_scale", 0.02))
        dataset = build_path_invariance_dataset(
            path_cfg,
            seed=int(cfg.get("seed", 42)) + target_seed + int(path_cfg.get("seed_offset", 90001)),
            forbidden_fingerprints=forbidden,
        )
        dataset_path = output_dir / "dataset_manifests" / f"seed_{target_seed}_{condition}.json"
        write_json(dataset.manifest, dataset_path)
        dataset_hash = hash_jsonable(dataset.manifest)
        summary, group_rows, variant_rows = evaluate_path_invariance(
            model, source.model, dataset, device=device,
            seed=int(cfg.get("seed", 42)) + target_seed,
            decoder_pca_dim=int(cfg["analysis"].get("decoder_pca_dim", 64)),
        )
        gates = {
            "endpoint_accuracy_high": summary["endpoint_accuracy"] >= float(cfg["analysis"].get("min_endpoint_accuracy", 0.80)),
            "phase_dispersion_source_matched": summary["endpoint_phase_dispersion"] <= (
                float(cfg["analysis"].get("source_baseline_multiplier", 1.5))
                * summary["source_matched_endpoint_phase_dispersion"]
                + float(cfg["analysis"].get("phase_dispersion_floor", 0.05))
            ),
            "loop_holonomy_low": summary["loop_holonomy"] <= float(cfg["analysis"].get("max_loop_holonomy", 0.35)),
            "zero_padding_invariant": summary["zero_padding_phase_difference"] <= float(cfg["analysis"].get("max_padding_phase_difference", 0.25)),
            "path_identity_near_chance": summary["path_identity_decoder_accuracy"] <= (
                summary["path_identity_decoder_chance"] + float(cfg["analysis"].get("path_decoder_chance_tolerance", 0.10))
            ),
            "residual_time_decoding_low": summary["residual_time_decoder_r2"] <= float(cfg["analysis"].get("max_residual_time_r2", 0.20)),
        }
        per_seed.append({
            "schema_version": SCHEMA_VERSION, "claim_scope": CLAIM_SCOPE,
            "target_seed": target_seed, "condition": condition,
            "unique_training_episodes": budget,
            "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "hard_dataset_manifest_sha256": hard_hash,
            "path_dataset_manifest_sha256": dataset_hash,
            **summary,
            "gates": gates,
            "path_independence_gate_passed": all(gates.values()),
        })
        for row in group_rows:
            group_rows_all.append({
                "schema_version": SCHEMA_VERSION, "claim_scope": CLAIM_SCOPE,
                "target_seed": target_seed, "condition": condition,
                "unique_training_episodes": budget,
                "checkpoint_sha256": sha256_file(checkpoint_path), **row,
            })
        for row in variant_rows:
            variant_rows_all.append({
                "schema_version": SCHEMA_VERSION, "claim_scope": CLAIM_SCOPE,
                "target_seed": target_seed, "condition": condition,
                "unique_training_episodes": budget,
                "checkpoint_sha256": sha256_file(checkpoint_path), **row,
            })
        dataset_manifests.append({
            "target_seed": target_seed, "condition": condition,
            "path": str(dataset_path.resolve()), "sha256": sha256_file(dataset_path),
            "trajectory_fingerprint_hash": dataset.manifest["trajectory_fingerprint_hash"],
        })
        checkpoint_rows.append({
            "path": str(checkpoint_path.resolve()), "sha256": sha256_file(checkpoint_path),
            "target_seed": target_seed, "condition": condition, "unique_training_episodes": budget,
        })
        logger.info(
            "%s seed=%d endpoint=%.3f phase_disp=%.3f loop=%.3f path_decoder=%.3f gate=%s",
            condition, target_seed, summary["endpoint_accuracy"],
            summary["endpoint_phase_dispersion"], summary["loop_holonomy"],
            summary["path_identity_decoder_accuracy"], all(gates.values()),
        )
    write_csv_strict(output_dir / "path_invariance_per_seed.csv", per_seed)
    write_csv_strict(output_dir / "path_invariance_per_group.csv", group_rows_all)
    write_csv_strict(output_dir / "path_invariance_per_trajectory.csv", variant_rows_all)
    write_json({
        "schema_version": "path_invariance_dataset_collection_v1",
        "claim_scope": CLAIM_SCOPE,
        "datasets": dataset_manifests,
    }, output_dir / "path_invariance_dataset_manifest.json")
    write_json({"schema_version": "path_checkpoint_manifest_v1", "checkpoints": checkpoint_rows}, output_dir / "checkpoint_manifest.json")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "claim_scope": CLAIM_SCOPE,
        "source_linkage": source_link,
        "evaluation_only": True,
        "regularization_used": False,
        "dedicated_dataset": {
            "same_start_same_end": True,
            "alternative_paths": True,
            "inserted_zero_net_loops": True,
            "zero_action_padding": True,
            "same_action_multiset_reordered": True,
        },
        "primary_endpoints": [
            "endpoint_phase_dispersion", "loop_holonomy",
            "zero_padding_phase_difference", "path_identity_decoder_accuracy",
            "residual_time_decoder_r2",
        ],
        "per_seed_metrics": per_seed,
        "all_seed_gates_passed": bool(per_seed) and all(row["path_independence_gate_passed"] for row in per_seed),
    }
    write_json(summary, output_dir / "path_invariance_summary.json")
    provenance = save_run_provenance(
        output_dir, "path_invariance", cfg=cfg,
        extra={
            "schema_version": SCHEMA_VERSION, "claim_scope": CLAIM_SCOPE,
            "source_reduced_artifact_sha256": source.sha256,
            "target_checkpoint_hashes": {row["path"]: row["sha256"] for row in checkpoint_rows},
            "trajectory_fingerprint_hashes": [row["trajectory_fingerprint_hash"] for row in dataset_manifests],
            "config_path": str(Path(args.config).resolve()),
            "config_sha256": sha256_file(args.config),
        },
    )
    build_artifact_manifest(
        output_dir, schema_version=SCHEMA_VERSION,
        required_files=[
            "config_used.yaml", "analysis.log", "path_invariance_summary.json",
            "path_invariance_dataset_manifest.json", "path_invariance_per_seed.csv",
            "path_invariance_per_group.csv", "path_invariance_per_trajectory.csv",
            "checkpoint_manifest.json", "run_provenance.json",
        ],
        metadata={"source_tree_sha256": provenance["source_tree_sha256"]},
    )


if __name__ == "__main__":
    main()
