"""Stage I2: non-saturating hard abstract path-integration benchmark.

Formal analyses use coordinate-supervision off, nested low-data budgets,
validation-only checkpoint selection, train-short/test-long evaluation, and an
ordered held-out action-composition test. Existing Exp5 entry points and result
schemas are untouched.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from gpm_cmap.analysis.hard_pathint import (
    build_hard_model,
    evaluate_loader_detailed,
    save_target_checkpoint,
    threshold_sample_requirement,
    train_fixed_updates,
    trapezoid_learning_curve_auc,
)
from gpm_cmap.analysis.recurrent_factorial import (
    FactorialCell,
    install_factorial_weights,
    load_navigation_grid_weights,
    make_matched_random_weights,
)
from gpm_cmap.analysis.upgrade_contracts import (
    CLAIM_SCOPE,
    build_artifact_manifest,
    hash_jsonable,
    load_source_reduced_artifact,
    source_linkage_payload,
    write_csv_strict,
)
from gpm_cmap.data.graph_tasks.hard_pathint import build_hard_pathint_pools, pool_loaders
from gpm_cmap.utils import (
    configure_runtime,
    load_config,
    prepare_output_dir,
    save_config_snapshot,
    save_run_provenance,
    set_seed,
    write_json,
)
from gpm_cmap.utils.checkpointing import sha256_file
from gpm_cmap.utils.logging_utils import get_logger

SCHEMA_VERSION = "exp5_hard_v1"

CONDITIONS: dict[str, FactorialCell] = {
    "navigation_frozen": FactorialCell("navigation", "navigation", "frozen", "frozen"),
    "matched_random_frozen": FactorialCell("matched_random", "matched_random", "frozen", "frozen"),
    "navigation_trainable": FactorialCell("navigation", "navigation", "trainable", "trainable"),
    "matched_random_trainable": FactorialCell("matched_random", "matched_random", "trainable", "trainable"),
    "navigation_hh_only_frozen": FactorialCell("navigation", "matched_random", "frozen", "frozen"),
    "navigation_ih_only_frozen": FactorialCell("matched_random", "navigation", "frozen", "frozen"),
}


def _condition_names(cfg: Mapping[str, Any], mode: str) -> list[str]:
    section = cfg.get(mode, {})
    names = list(section.get("conditions", cfg.get("conditions", list(CONDITIONS))))
    unknown = sorted(set(names) - set(CONDITIONS))
    if unknown:
        raise ValueError(f"Unknown hard benchmark conditions: {unknown}")
    return names


def _run_protocol(
    cfg: dict[str, Any],
    *,
    mode: str,
    output_dir: Path,
    device: torch.device,
    logger,
    n_levels: int,
    target_seeds: list[int],
    budgets: list[int],
) -> dict[str, Any]:
    source = load_source_reduced_artifact(cfg["source_reduced_artifact"])
    source_link = source_linkage_payload(source)
    nav_hh, nav_ih, nav_meta = load_navigation_grid_weights(cfg["navigation_checkpoint"])
    model_cfg = dict(cfg["model"])
    if int(model_cfg["hidden_dim"]) != int(nav_hh.shape[0]):
        raise ValueError(
            f"model.hidden_dim={model_cfg['hidden_dim']} does not match navigation W_hh={nav_hh.shape[0]}"
        )
    data_template = deepcopy(cfg["data"])
    data_template["n_levels"] = int(n_levels)
    data_template["episode_budgets"] = list(budgets)
    batch_size = int(data_template.get("batch_size", 128))
    num_workers = int(data_template.get("num_workers", 0))
    condition_names = _condition_names(cfg, mode)
    train_cfg = dict(cfg["train"])
    if float(train_cfg.get("position_loss_weight", 0.0)) != 0.0:
        raise ValueError("Hard benchmark formal main analysis requires position_loss_weight=0")

    curve_rows: list[dict[str, Any]] = []
    generalization_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    update_rows: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    matching_rows: list[dict[str, Any]] = []
    per_seed_summaries: list[dict[str, Any]] = []

    for target_seed in target_seeds:
        logger.info("Hard benchmark: n_levels=%d target_seed=%d", n_levels, target_seed)
        pools = build_hard_pathint_pools(data_template, seed=int(target_seed))
        seed_manifest_dir = output_dir / "dataset_manifests"
        seed_manifest_dir.mkdir(parents=True, exist_ok=True)
        seed_manifest_path = seed_manifest_dir / f"seed_{target_seed}.json"
        write_json(pools.manifest, seed_manifest_path)
        dataset_hash = hash_jsonable(pools.manifest)
        random_hh, random_ih, random_diag = make_matched_random_weights(
            nav_hh,
            nav_ih,
            seed=int(target_seed) + int(cfg.get("matched_random_seed_offset", 70001)),
            iterations=int(cfg.get("matched_random", {}).get("iterations", 8)),
            singular_rank=int(cfg.get("matched_random", {}).get("singular_rank", 16)),
            probe_count=int(cfg.get("matched_random", {}).get("probe_count", 64)),
            tolerances=cfg.get("matched_random", {}).get("tolerances", {}),
            strict=bool(cfg.get("matched_random", {}).get("strict_matching_gate", True)),
        )
        matching_rows.append({
            "target_seed": target_seed,
            "n_levels": n_levels,
            "passed_matching_gate": random_diag["passed_matching_gate"],
            **{f"error_{k}": v for k, v in random_diag["matching_error"].items()},
        })
        write_json(random_diag, output_dir / "matched_random_diagnostics" / f"seed_{target_seed}.json")

        seed_curve: list[dict[str, Any]] = []
        for budget in budgets:
            loaders = pool_loaders(
                pools,
                unique_episodes=int(budget),
                batch_size=batch_size,
                seed=int(target_seed) + int(budget) * 17,
                num_workers=num_workers,
            )
            for condition_name in condition_names:
                cell = CONDITIONS[condition_name]
                generalization_start = len(generalization_rows)
                trajectory_start = len(trajectory_rows)
                update_start = len(update_rows)
                run_seed = int(target_seed) * 1000003 + int(budget) * 101
                set_seed(run_seed)
                model = build_hard_model(model_cfg, n_nodes=pools.n_nodes, device=device)
                install_factorial_weights(
                    model,
                    cell=cell,
                    nav_hh=nav_hh,
                    nav_ih=nav_ih,
                    random_hh=random_hh,
                    random_ih=random_ih,
                )
                training = train_fixed_updates(
                    model,
                    loaders["train"],
                    loaders["validation"],
                    device,
                    train_cfg,
                    seed=run_seed,
                )
                model.load_state_dict(training.state_dict, strict=True)
                validation, validation_traj = evaluate_loader_detailed(
                    model, loaders["validation"], device, split_name="validation",
                    calibration_bins=int(train_cfg.get("calibration_bins", 15)),
                )
                split_metrics: dict[str, dict[str, Any]] = {}
                for split_name in ("id_test", "long_test", "composition_test"):
                    metrics, rows = evaluate_loader_detailed(
                        model,
                        loaders[split_name],
                        device,
                        split_name=split_name,
                        calibration_bins=int(train_cfg.get("calibration_bins", 15)),
                    )
                    split_metrics[split_name] = metrics
                    for row in rows:
                        trajectory_rows.append({
                            "schema_version": SCHEMA_VERSION,
                            "claim_scope": CLAIM_SCOPE,
                            "mode": mode,
                            "n_levels": n_levels,
                            "target_seed": target_seed,
                            "run_seed": run_seed,
                            "condition": condition_name,
                            **cell.as_dict(),
                            "unique_training_episodes": budget,
                            **row,
                        })
                    generalization_rows.append({
                        "schema_version": SCHEMA_VERSION,
                        "claim_scope": CLAIM_SCOPE,
                        "mode": mode,
                        "n_levels": n_levels,
                        "target_seed": target_seed,
                        "run_seed": run_seed,
                        "condition": condition_name,
                        **cell.as_dict(),
                        "unique_training_episodes": budget,
                        "split": split_name,
                        **{key: value for key, value in metrics.items() if not isinstance(value, dict)},
                        "accuracy_by_path_length": metrics["accuracy_by_path_length"],
                        "accuracy_by_endpoint_distance": metrics["accuracy_by_endpoint_distance"],
                    })
                row = {
                    "schema_version": SCHEMA_VERSION,
                    "claim_scope": CLAIM_SCOPE,
                    "mode": mode,
                    "n_levels": n_levels,
                    "target_seed": target_seed,
                    "run_seed": run_seed,
                    "condition": condition_name,
                    **cell.as_dict(),
                    "unique_training_episodes": budget,
                    "best_update": training.best_update,
                    "stopped_update": training.stopped_update,
                    "validation_accuracy": validation["accuracy"],
                    "validation_nll": validation["nll"],
                    "id_test_accuracy": split_metrics["id_test"]["accuracy"],
                    "id_test_nll": split_metrics["id_test"]["nll"],
                    "id_test_ece": split_metrics["id_test"]["ece"],
                    "long_test_accuracy": split_metrics["long_test"]["accuracy"],
                    "long_test_nll": split_metrics["long_test"]["nll"],
                    "composition_test_accuracy": split_metrics["composition_test"]["accuracy"],
                    "composition_test_nll": split_metrics["composition_test"]["nll"],
                    "dataset_manifest_sha256": dataset_hash,
                }
                curve_rows.append(row)
                seed_curve.append(row)
                for hist in training.history:
                    update_rows.append({
                        "schema_version": SCHEMA_VERSION,
                        "claim_scope": CLAIM_SCOPE,
                        "mode": mode,
                        "n_levels": n_levels,
                        "target_seed": target_seed,
                        "run_seed": run_seed,
                        "condition": condition_name,
                        **cell.as_dict(),
                        "unique_training_episodes": budget,
                        **hist,
                    })
                ckpt_dir = output_dir / "checkpoints" / f"seed_{target_seed}" / condition_name / f"episodes_{budget}"
                ckpt_path = ckpt_dir / "best_validation.pt"
                save_target_checkpoint(
                    ckpt_path,
                    model=model,
                    state_dict=training.state_dict,
                    model_config=model_cfg,
                    condition={"name": condition_name, **cell.as_dict()},
                    seed=target_seed,
                    run_seed=run_seed,
                    unique_episodes=budget,
                    best_update=training.best_update,
                    validation_metrics=validation,
                    dataset_manifest_hash=dataset_hash,
                    source_linkage=source_link,
                    extra={
                        "mode": mode,
                        "n_levels": n_levels,
                        "navigation_checkpoint": str(Path(cfg["navigation_checkpoint"]).resolve()),
                        "navigation_checkpoint_sha256": nav_meta["checkpoint_sha256"],
                        "dataset_manifest_path": str(seed_manifest_path.resolve()),
                        "run_seed": run_seed,
                    },
                )
                checkpoint_hash = sha256_file(ckpt_path)
                row["checkpoint_sha256"] = checkpoint_hash
                for output_row in generalization_rows[generalization_start:]:
                    output_row["checkpoint_sha256"] = checkpoint_hash
                for output_row in trajectory_rows[trajectory_start:]:
                    output_row["checkpoint_sha256"] = checkpoint_hash
                for output_row in update_rows[update_start:]:
                    output_row["checkpoint_sha256"] = checkpoint_hash
                checkpoint_rows.append({
                    "target_seed": target_seed,
                    "run_seed": run_seed,
                    "condition": condition_name,
                    "unique_training_episodes": budget,
                    "path": str(ckpt_path.resolve()),
                    "sha256": checkpoint_hash,
                    "dataset_manifest_sha256": dataset_hash,
                    "best_update": training.best_update,
                })
                logger.info(
                    "  %s episodes=%d ID=%.3f long=%.3f composition=%.3f",
                    condition_name, budget,
                    split_metrics["id_test"]["accuracy"],
                    split_metrics["long_test"]["accuracy"],
                    split_metrics["composition_test"]["accuracy"],
                )
        for condition_name in condition_names:
            rows = [row for row in seed_curve if row["condition"] == condition_name]
            checkpoint_hashes = {
                str(row["unique_training_episodes"]): row["sha256"]
                for row in checkpoint_rows
                if int(row["target_seed"]) == target_seed
                and row["condition"] == condition_name
            }
            per_seed_summaries.append({
                "schema_version": SCHEMA_VERSION,
                "claim_scope": CLAIM_SCOPE,
                "target_seed": target_seed,
                "n_levels": n_levels,
                "condition": condition_name,
                "low_data_auc_id": trapezoid_learning_curve_auc(rows, metric="id_test_accuracy"),
                "low_data_auc_long": trapezoid_learning_curve_auc(rows, metric="long_test_accuracy"),
                "low_data_auc_composition": trapezoid_learning_curve_auc(rows, metric="composition_test_accuracy"),
                "episodes_to_80pct_id": threshold_sample_requirement(rows, metric="id_test_accuracy", threshold=0.80),
                "episodes_to_90pct_id": threshold_sample_requirement(rows, metric="id_test_accuracy", threshold=0.90),
                "episodes_to_95pct_id": threshold_sample_requirement(rows, metric="id_test_accuracy", threshold=0.95),
                "dataset_manifest_sha256": dataset_hash,
                "checkpoint_hashes_by_budget": checkpoint_hashes,
                "max_budget_checkpoint_sha256": checkpoint_hashes[str(max(budgets))],
            })

    write_csv_strict(output_dir / "hard_learning_curves.csv", curve_rows)
    write_csv_strict(output_dir / "hard_generalization_metrics.csv", generalization_rows)
    write_csv_strict(output_dir / "hard_per_trajectory_metrics.csv", trajectory_rows)
    write_csv_strict(output_dir / "hard_update_curves.csv", update_rows)
    write_csv_strict(output_dir / "hard_per_seed_metrics.csv", per_seed_summaries)
    write_csv_strict(output_dir / "matched_random_diagnostics.csv", matching_rows)
    write_json({"schema_version": "hard_checkpoint_manifest_v1", "checkpoints": checkpoint_rows}, output_dir / "checkpoint_manifest.json")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "claim_scope": CLAIM_SCOPE,
        "mode": mode,
        "source_linkage": source_link,
        "navigation_checkpoint": nav_meta,
        "n_levels": n_levels,
        "target_seeds": target_seeds,
        "episode_budgets": budgets,
        "conditions": condition_names,
        "primary_endpoints": [
            "low_data_auc_id",
            "long_test_accuracy",
            "composition_test_accuracy",
        ],
        "coordinate_supervision": "off",
        "selection": "validation only",
        "test_use": "once after checkpoint selection per run",
        "per_seed_metrics": per_seed_summaries,
        "checkpoint_count": len(checkpoint_rows),
    }
    write_json(summary, output_dir / "exp5_hard_summary.json")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--mode", choices=("pilot", "confirmatory"), default="confirmatory")
    args = parser.parse_args()
    cfg = load_config(args.config)
    cfg["mode"] = args.mode
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    output_dir = prepare_output_dir(cfg["output_dir"])
    save_config_snapshot(cfg, output_dir)
    logger = get_logger("exp5_hard_benchmark", output_dir / "experiment.log")
    device = configure_runtime(cfg)
    section = cfg[args.mode]
    target_seeds = [int(seed) for seed in section["target_seeds"]]
    budgets = [int(value) for value in section["episode_budgets"]]
    if args.mode == "pilot":
        level_summaries = []
        for n_levels in [int(v) for v in section.get("n_levels_candidates", [8, 12])]:
            level_dir = output_dir / f"n_levels_{n_levels}"
            level_dir.mkdir(parents=True, exist_ok=False)
            level_cfg = deepcopy(cfg)
            level_summary = _run_protocol(
                level_cfg,
                mode=args.mode,
                output_dir=level_dir,
                device=device,
                logger=logger,
                n_levels=n_levels,
                target_seeds=target_seeds,
                budgets=budgets,
            )
            level_summaries.append(level_summary)
        # Recommendation uses pilot-only matched-random trainable ID accuracy.
        candidates = []
        for summary in level_summaries:
            rows_path = output_dir / f"n_levels_{summary['n_levels']}" / "hard_learning_curves.csv"
            import csv
            with rows_path.open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            max_budget = max(budgets)
            values = [
                float(row["id_test_accuracy"])
                for row in rows
                if row["condition"] == "matched_random_trainable"
                and int(row["unique_training_episodes"]) == max_budget
            ]
            mean = float(np.mean(values)) if values else None
            candidates.append({"n_levels": summary["n_levels"], "random_init_id_accuracy": mean})
        in_band = [row for row in candidates if row["random_init_id_accuracy"] is not None and 0.60 <= row["random_init_id_accuracy"] <= 0.85]
        recommended = min(in_band, key=lambda row: row["n_levels"]) if in_band else max(candidates, key=lambda row: row["n_levels"])
        recommendation = {
            "schema_version": "hard_pilot_recommendation_v1",
            "claim_scope": CLAIM_SCOPE,
            "selection_band": [0.60, 0.85],
            "candidates": candidates,
            "recommended": recommended,
            "warning": "Pilot recommendation must be frozen before confirmatory target seeds are run.",
        }
        write_json(recommendation, output_dir / "pilot_recommendation.json")
        child_required = [
            "exp5_hard_summary.json", "hard_learning_curves.csv",
            "hard_generalization_metrics.csv", "hard_per_seed_metrics.csv",
            "hard_per_trajectory_metrics.csv", "hard_update_curves.csv",
            "checkpoint_manifest.json", "matched_random_diagnostics.csv",
        ]
        level_index = []
        for summary in level_summaries:
            level_dir = output_dir / f"n_levels_{summary['n_levels']}"
            artifacts = []
            for relative in child_required:
                path = level_dir / relative
                if not path.is_file() or path.stat().st_size <= 0:
                    raise RuntimeError(f"Incomplete pilot level output: {path}")
                artifacts.append({
                    "path": str(path.relative_to(output_dir)),
                    "sha256": sha256_file(path),
                    "size_bytes": int(path.stat().st_size),
                })
            level_index.append({"n_levels": summary["n_levels"], "artifacts": artifacts})
        write_json({
            "schema_version": "hard_pilot_results_index_v1",
            "claim_scope": CLAIM_SCOPE,
            "levels": level_index,
        }, output_dir / "pilot_results_index.json")
        write_json({
            "schema_version": SCHEMA_VERSION,
            "claim_scope": CLAIM_SCOPE,
            "mode": "pilot",
            "coordinate_supervision": "off",
            "selection_data": "independent pilot target seeds only",
            "level_summaries": level_summaries,
            "recommendation": recommendation,
        }, output_dir / "exp5_hard_summary.json")
        required = [
            "config_used.yaml", "pilot_recommendation.json",
            "pilot_results_index.json", "exp5_hard_summary.json",
            "experiment.log",
        ]
    else:
        n_levels = int(section.get("n_levels", cfg["data"].get("n_levels", 8)))
        _run_protocol(
            cfg,
            mode=args.mode,
            output_dir=output_dir,
            device=device,
            logger=logger,
            n_levels=n_levels,
            target_seeds=target_seeds,
            budgets=budgets,
        )
        required = [
            "config_used.yaml", "experiment.log", "exp5_hard_summary.json",
            "hard_learning_curves.csv", "hard_generalization_metrics.csv",
            "hard_per_seed_metrics.csv", "hard_per_trajectory_metrics.csv",
            "hard_update_curves.csv", "checkpoint_manifest.json",
            "matched_random_diagnostics.csv",
        ]
    # The artifact manifest is written last so hashes for experiment.log remain
    # stable.  No logger calls may follow build_artifact_manifest().
    logger.info("Completed hard benchmark: %s", output_dir)
    provenance = save_run_provenance(
        output_dir,
        experiment_name="exp5_hard_benchmark",
        cfg=cfg,
        extra={
            "schema_version": SCHEMA_VERSION,
            "claim_scope": CLAIM_SCOPE,
            "config_path": str(Path(args.config).resolve()),
            "config_sha256": sha256_file(args.config),
        },
    )
    build_artifact_manifest(
        output_dir,
        schema_version=SCHEMA_VERSION,
        required_files=required + ["run_provenance.json"],
        metadata={"mode": args.mode, "source_tree_sha256": provenance["source_tree_sha256"]},
    )


if __name__ == "__main__":
    main()
