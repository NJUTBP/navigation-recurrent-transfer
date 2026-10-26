"""Stage I3A: complete 2×2×2×2 W_hh/W_ih origin/trainability factorial."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from gpm_cmap.analysis.hard_pathint import (
    build_hard_model,
    evaluate_loader_detailed,
    save_target_checkpoint,
    train_fixed_updates,
    trapezoid_learning_curve_auc,
)
from gpm_cmap.analysis.recurrent_factorial import (
    all_factorial_cells,
    factorial_contrast_table,
    install_custom_weights,
    load_navigation_grid_weights,
    make_matched_random_weights,
    mismatched_input_basis,
    paired_condition_comparisons,
    parse_factorial_cell,
    spectrum_matched_shuffled_hh,
    weight_pair_diagnostics,
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
    configure_runtime, load_config, prepare_output_dir, save_config_snapshot,
    save_run_provenance, set_seed, write_json,
)
from gpm_cmap.utils.checkpointing import sha256_file
from gpm_cmap.utils.logging_utils import get_logger

SCHEMA_VERSION = "recurrent_factorial_v1"

DEFAULT_CONFIRMATORY_COMPARISONS = [
    {
        "name": "navigation_full_frozen_vs_matched_random_full_frozen",
        "left": "Ohh-navigation__Oih-navigation__Thh-frozen__Tih-frozen",
        "right": "Ohh-matched_random__Oih-matched_random__Thh-frozen__Tih-frozen",
    },
    {
        "name": "navigation_hh_effect_given_navigation_ih_frozen",
        "left": "Ohh-navigation__Oih-navigation__Thh-frozen__Tih-frozen",
        "right": "Ohh-matched_random__Oih-navigation__Thh-frozen__Tih-frozen",
    },
    {
        "name": "navigation_ih_effect_given_navigation_hh_frozen",
        "left": "Ohh-navigation__Oih-navigation__Thh-frozen__Tih-frozen",
        "right": "Ohh-navigation__Oih-matched_random__Thh-frozen__Tih-frozen",
    },
    {
        "name": "navigation_hh_effect_given_matched_random_ih_frozen",
        "left": "Ohh-navigation__Oih-matched_random__Thh-frozen__Tih-frozen",
        "right": "Ohh-matched_random__Oih-matched_random__Thh-frozen__Tih-frozen",
    },
    {
        "name": "navigation_ih_effect_given_matched_random_hh_frozen",
        "left": "Ohh-matched_random__Oih-navigation__Thh-frozen__Tih-frozen",
        "right": "Ohh-matched_random__Oih-matched_random__Thh-frozen__Tih-frozen",
    },
    {
        "name": "navigation_full_trainable_vs_matched_random_full_trainable",
        "left": "Ohh-navigation__Oih-navigation__Thh-trainable__Tih-trainable",
        "right": "Ohh-matched_random__Oih-matched_random__Thh-trainable__Tih-trainable",
    },
    {
        "name": "recurrent_trainability_with_navigation_origin",
        "left": "Ohh-navigation__Oih-navigation__Thh-trainable__Tih-frozen",
        "right": "Ohh-navigation__Oih-navigation__Thh-frozen__Tih-frozen",
    },
    {
        "name": "input_trainability_with_navigation_origin",
        "left": "Ohh-navigation__Oih-navigation__Thh-frozen__Tih-trainable",
        "right": "Ohh-navigation__Oih-navigation__Thh-frozen__Tih-frozen",
    },
]


def _cells(cfg: dict[str, Any], mode: str):
    selected = cfg[mode].get("cells", "all")
    if selected == "all":
        return list(all_factorial_cells())
    cells = [parse_factorial_cell(label) for label in selected]
    if len({cell.label for cell in cells}) != len(cells):
        raise ValueError("Duplicate factorial cells in configuration")
    return cells


def _extra_control_conditions(
    cfg: dict[str, Any],
    *,
    target_seed: int,
    nav_hh: torch.Tensor,
    nav_ih: torch.Tensor,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Construct optional non-factorial controls from the upgrade plan.

    These controls are deliberately excluded from balanced 2^4 contrasts.  They
    diagnose whether a positive navigation effect is specific to grid quality,
    the navigation spectrum, or W_hh/W_ih basis alignment.
    """
    section = cfg.get("extra_controls", {}) or {}
    if not bool(section.get("enabled", False)):
        return [], []
    controls = list(section.get("controls", []))
    if not controls:
        raise ValueError("extra_controls.enabled=true requires at least one control")
    output: list[dict[str, Any]] = []
    diagnostics_rows: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    singular_rank = int(cfg.get("matched_random", {}).get("singular_rank", 16))
    probe_count = int(cfg.get("matched_random", {}).get("probe_count", 64))
    for index, raw in enumerate(controls):
        spec = dict(raw)
        name = str(spec.get("name") or spec.get("kind") or f"extra_control_{index}")
        kind = str(spec.get("kind", name))
        if name in seen_names:
            raise ValueError(f"Duplicate extra-control name: {name}")
        seen_names.add(name)
        hh_trainability = str(spec.get("hh_trainability", "frozen"))
        ih_trainability = str(spec.get("ih_trainability", "frozen"))
        control_seed = int(target_seed) + int(spec.get("seed_offset", 81001 + index * 997))
        checkpoint_meta: dict[str, Any] | None = None
        if kind == "spectrum_matched_shuffled_hh":
            hh = spectrum_matched_shuffled_hh(nav_hh, seed=control_seed)
            ih = nav_ih.clone()
            hh_origin = "spectrum_matched_shuffled_navigation"
            ih_origin = "navigation"
        elif kind == "navigation_hh_mismatched_ih_basis":
            hh = nav_hh.clone()
            ih = mismatched_input_basis(nav_ih, seed=control_seed)
            hh_origin = "navigation"
            ih_origin = "navigation_mismatched_basis"
        elif kind in {"low_grid_checkpoint", "non_grid_checkpoint", "external_checkpoint"}:
            checkpoint = spec.get("checkpoint")
            if not checkpoint:
                raise ValueError(f"Extra control {name!r} requires checkpoint")
            hh, ih, checkpoint_meta = load_navigation_grid_weights(checkpoint)
            if tuple(hh.shape) != tuple(nav_hh.shape) or tuple(ih.shape) != tuple(nav_ih.shape):
                raise ValueError(
                    f"Extra-control checkpoint {checkpoint!r} has incompatible grid weights"
                )
            origin = {
                "low_grid_checkpoint": "low_grid_navigation_checkpoint",
                "non_grid_checkpoint": "non_grid_checkpoint",
                "external_checkpoint": str(spec.get("origin_label", "external_checkpoint")),
            }[kind]
            hh_origin = origin
            ih_origin = origin
        else:
            raise ValueError(f"Unknown extra-control kind: {kind}")
        diagnostics = weight_pair_diagnostics(
            hh,
            ih,
            nav_hh,
            nav_ih,
            seed=control_seed,
            singular_rank=singular_rank,
            probe_count=probe_count,
        )
        condition = {
            "hh_origin": hh_origin,
            "ih_origin": ih_origin,
            "hh_trainability": hh_trainability,
            "ih_trainability": ih_trainability,
            "cell": name,
            "is_factorial_cell": False,
            "extra_control_kind": kind,
        }
        output.append({"condition": condition, "hh": hh, "ih": ih})
        diagnostics_rows.append({
            "target_seed": int(target_seed),
            "control": name,
            "kind": kind,
            "control_seed": control_seed,
            "hh_trainability": hh_trainability,
            "ih_trainability": ih_trainability,
            "checkpoint": None if checkpoint_meta is None else checkpoint_meta["checkpoint"],
            "checkpoint_sha256": None if checkpoint_meta is None else checkpoint_meta["checkpoint_sha256"],
            **{f"error_{key}": value for key, value in diagnostics["matching_error"].items()},
            "diagnostics": diagnostics,
        })
    return output, diagnostics_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--mode", choices=("screening", "confirmatory"), default="screening")
    args = parser.parse_args()
    cfg = load_config(args.config)
    cfg["mode"] = args.mode
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    output_dir = prepare_output_dir(cfg["output_dir"])
    save_config_snapshot(cfg, output_dir)
    logger = get_logger("recurrent_factorial", output_dir / "experiment.log")
    device = configure_runtime(cfg)
    source = load_source_reduced_artifact(cfg["source_reduced_artifact"])
    source_link = source_linkage_payload(source)
    nav_hh, nav_ih, nav_meta = load_navigation_grid_weights(cfg["navigation_checkpoint"])
    if int(cfg["model"]["hidden_dim"]) != nav_hh.shape[0]:
        raise ValueError("model.hidden_dim must match the navigation checkpoint")
    train_cfg = dict(cfg["train"])
    if float(train_cfg.get("position_loss_weight", 0.0)) != 0:
        raise ValueError("Formal factorial requires coordinate supervision off")
    section = cfg[args.mode]
    seeds = [int(v) for v in section["target_seeds"]]
    budgets = [int(v) for v in section["episode_budgets"]]
    cells = _cells(cfg, args.mode)
    if args.mode == "screening" and len(cells) != 16:
        raise ValueError("Screening must contain the complete 16-cell factorial")
    data_cfg = dict(cfg["data"])
    data_cfg["episode_budgets"] = budgets
    batch_size = int(data_cfg.get("batch_size", 128))
    num_workers = int(data_cfg.get("num_workers", 0))

    curve_rows: list[dict[str, Any]] = []
    generalization_rows: list[dict[str, Any]] = []
    per_trajectory: list[dict[str, Any]] = []
    update_rows: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    per_seed_rows: list[dict[str, Any]] = []
    matching_rows: list[dict[str, Any]] = []
    extra_control_rows: list[dict[str, Any]] = []

    for target_seed in seeds:
        pools = build_hard_pathint_pools(data_cfg, seed=target_seed)
        dataset_dir = output_dir / "dataset_manifests"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        dataset_path = dataset_dir / f"seed_{target_seed}.json"
        write_json(pools.manifest, dataset_path)
        dataset_hash = hash_jsonable(pools.manifest)
        random_hh, random_ih, random_diag = make_matched_random_weights(
            nav_hh, nav_ih,
            seed=target_seed + int(cfg.get("matched_random_seed_offset", 70001)),
            iterations=int(cfg["matched_random"].get("iterations", 8)),
            singular_rank=int(cfg["matched_random"].get("singular_rank", 16)),
            probe_count=int(cfg["matched_random"].get("probe_count", 64)),
            tolerances=cfg["matched_random"].get("tolerances", {}),
            strict=bool(cfg["matched_random"].get("strict_matching_gate", True)),
        )
        write_json(random_diag, output_dir / "matched_random_diagnostics" / f"seed_{target_seed}.json")
        matching_rows.append({
            "target_seed": target_seed,
            "passed_matching_gate": random_diag["passed_matching_gate"],
            **{f"error_{name}": value for name, value in random_diag["matching_error"].items()},
        })
        condition_specs: list[dict[str, Any]] = []
        for cell in cells:
            condition_specs.append({
                "condition": {
                    **cell.as_dict(),
                    "is_factorial_cell": True,
                    "extra_control_kind": None,
                },
                "hh": nav_hh if cell.hh_origin == "navigation" else random_hh,
                "ih": nav_ih if cell.ih_origin == "navigation" else random_ih,
            })
        extra_specs, extra_diagnostics = _extra_control_conditions(
            cfg,
            target_seed=target_seed,
            nav_hh=nav_hh,
            nav_ih=nav_ih,
        )
        condition_specs.extend(extra_specs)
        for diagnostic in extra_diagnostics:
            detailed = diagnostic.pop("diagnostics")
            diagnostics_path = (
                output_dir / "extra_control_diagnostics" /
                f"seed_{target_seed}_{diagnostic['control']}.json"
            )
            write_json(detailed, diagnostics_path)
            diagnostic["diagnostics_path"] = str(diagnostics_path.resolve())
            diagnostic["diagnostics_sha256"] = sha256_file(diagnostics_path)
            extra_control_rows.append(diagnostic)
        seed_curves: list[dict[str, Any]] = []
        for budget in budgets:
            loaders = pool_loaders(
                pools, unique_episodes=budget, batch_size=batch_size,
                seed=target_seed + budget * 19, num_workers=num_workers,
            )
            for spec in condition_specs:
                condition = dict(spec["condition"])
                condition_name = str(condition["cell"])
                run_seed = target_seed * 1000003 + budget * 101
                set_seed(run_seed)
                model = build_hard_model(cfg["model"], n_nodes=pools.n_nodes, device=device)
                install_custom_weights(
                    model,
                    hh=spec["hh"],
                    ih=spec["ih"],
                    hh_trainability=str(condition["hh_trainability"]),
                    ih_trainability=str(condition["ih_trainability"]),
                )
                training = train_fixed_updates(
                    model, loaders["train"], loaders["validation"], device,
                    train_cfg, seed=run_seed,
                )
                model.load_state_dict(training.state_dict, strict=True)
                validation, _ = evaluate_loader_detailed(
                    model, loaders["validation"], device, split_name="validation",
                    calibration_bins=int(train_cfg.get("calibration_bins", 15)),
                )
                split_metrics = {}
                local_generalization: list[dict[str, Any]] = []
                local_trajectory: list[dict[str, Any]] = []
                for split in ("id_test", "long_test", "composition_test"):
                    metrics, rows = evaluate_loader_detailed(
                        model, loaders[split], device, split_name=split,
                        calibration_bins=int(train_cfg.get("calibration_bins", 15)),
                    )
                    split_metrics[split] = metrics
                    local_generalization.append({
                        "schema_version": SCHEMA_VERSION, "claim_scope": CLAIM_SCOPE,
                        "mode": args.mode, "target_seed": target_seed,
                        "run_seed": run_seed,
                        **condition, "unique_training_episodes": budget,
                        "split": split,
                        **{key: value for key, value in metrics.items() if not isinstance(value, dict)},
                        "accuracy_by_path_length": metrics["accuracy_by_path_length"],
                    })
                    for row in rows:
                        local_trajectory.append({
                            "schema_version": SCHEMA_VERSION, "claim_scope": CLAIM_SCOPE,
                            "mode": args.mode, "target_seed": target_seed,
                            "run_seed": run_seed,
                            **condition, "unique_training_episodes": budget,
                            **row,
                        })
                curve = {
                    "schema_version": SCHEMA_VERSION,
                    "claim_scope": CLAIM_SCOPE,
                    "mode": args.mode,
                    "target_seed": target_seed,
                    "run_seed": run_seed,
                    **condition,
                    "unique_training_episodes": budget,
                    "best_update": training.best_update,
                    "stopped_update": training.stopped_update,
                    "validation_accuracy": validation["accuracy"],
                    "validation_nll": validation["nll"],
                    "id_test_accuracy": split_metrics["id_test"]["accuracy"],
                    "id_test_nll": split_metrics["id_test"]["nll"],
                    "long_test_accuracy": split_metrics["long_test"]["accuracy"],
                    "composition_test_accuracy": split_metrics["composition_test"]["accuracy"],
                    "dataset_manifest_sha256": dataset_hash,
                }
                local_updates = []
                for update in training.history:
                    local_updates.append({
                        "schema_version": SCHEMA_VERSION, "claim_scope": CLAIM_SCOPE,
                        "mode": args.mode, "target_seed": target_seed,
                        "run_seed": run_seed,
                        **condition, "unique_training_episodes": budget,
                        **update,
                    })
                ckpt_path = (
                    output_dir / "checkpoints" / f"seed_{target_seed}" /
                    condition_name / f"episodes_{budget}" / "best_validation.pt"
                )
                save_target_checkpoint(
                    ckpt_path,
                    model=model,
                    state_dict=training.state_dict,
                    model_config=cfg["model"],
                    condition=condition,
                    seed=target_seed,
                    run_seed=run_seed,
                    unique_episodes=budget,
                    best_update=training.best_update,
                    validation_metrics=validation,
                    dataset_manifest_hash=dataset_hash,
                    source_linkage=source_link,
                    schema_version="recurrent_factorial_checkpoint_v1",
                    extra={
                        "mode": args.mode,
                        "dataset_manifest_path": str(dataset_path.resolve()),
                        "navigation_checkpoint_sha256": nav_meta["checkpoint_sha256"],
                        "run_seed": run_seed,
                    },
                )
                checkpoint_hash = sha256_file(ckpt_path)
                curve["checkpoint_sha256"] = checkpoint_hash
                for row in local_generalization:
                    row["checkpoint_sha256"] = checkpoint_hash
                for row in local_trajectory:
                    row["checkpoint_sha256"] = checkpoint_hash
                for row in local_updates:
                    row["checkpoint_sha256"] = checkpoint_hash
                curve_rows.append(curve)
                seed_curves.append(curve)
                generalization_rows.extend(local_generalization)
                per_trajectory.extend(local_trajectory)
                update_rows.extend(local_updates)
                checkpoint_rows.append({
                    "target_seed": target_seed, "run_seed": run_seed, **condition,
                    "unique_training_episodes": budget,
                    "path": str(ckpt_path.resolve()), "sha256": checkpoint_hash,
                    "best_update": training.best_update,
                })
                logger.info(
                    "seed=%d budget=%d %s ID=%.3f long=%.3f comp=%.3f",
                    target_seed, budget, condition_name,
                    split_metrics["id_test"]["accuracy"],
                    split_metrics["long_test"]["accuracy"],
                    split_metrics["composition_test"]["accuracy"],
                )
        for condition in [dict(spec["condition"]) for spec in condition_specs]:
            condition_name = str(condition["cell"])
            selected = [row for row in seed_curves if row["cell"] == condition_name]
            checkpoint_hashes = {
                str(row["unique_training_episodes"]): row["sha256"]
                for row in checkpoint_rows
                if int(row["target_seed"]) == target_seed and row["cell"] == condition_name
            }
            per_seed_rows.append({
                "schema_version": SCHEMA_VERSION,
                "claim_scope": CLAIM_SCOPE,
                "target_seed": target_seed,
                **condition,
                "auc_id": trapezoid_learning_curve_auc(selected, metric="id_test_accuracy"),
                "auc_long": trapezoid_learning_curve_auc(selected, metric="long_test_accuracy"),
                "auc_composition": trapezoid_learning_curve_auc(selected, metric="composition_test_accuracy"),
                "max_budget": max(budgets),
                "max_budget_id_accuracy": [row for row in selected if row["unique_training_episodes"] == max(budgets)][0]["id_test_accuracy"],
                "max_budget_long_accuracy": [row for row in selected if row["unique_training_episodes"] == max(budgets)][0]["long_test_accuracy"],
                "max_budget_composition_accuracy": [row for row in selected if row["unique_training_episodes"] == max(budgets)][0]["composition_test_accuracy"],
                "dataset_manifest_sha256": dataset_hash,
                "checkpoint_hashes_by_budget": checkpoint_hashes,
                "max_budget_checkpoint_sha256": checkpoint_hashes[str(max(budgets))],
            })

    contrasts = []
    confirmatory_comparison_rows: list[dict[str, Any]] = []
    if args.mode == "screening":
        for endpoint in (
            "auc_id", "auc_long", "auc_composition",
            "max_budget_id_accuracy", "max_budget_long_accuracy",
            "max_budget_composition_accuracy",
        ):
            contrasts.extend(factorial_contrast_table(
                [row for row in per_seed_rows if bool(row.get("is_factorial_cell"))],
                endpoint=endpoint,
                bootstrap_samples=int(cfg.get("statistics", {}).get("bootstrap_samples", 5000)),
                seed=int(cfg.get("seed", 42)) + len(contrasts),
            ))
    else:
        comparison_specs = list(
            cfg.get("statistics", {}).get(
                "confirmatory_comparisons", DEFAULT_CONFIRMATORY_COMPARISONS
            )
        )
        for endpoint in (
            "auc_id", "auc_long", "auc_composition",
            "max_budget_id_accuracy", "max_budget_long_accuracy",
            "max_budget_composition_accuracy",
        ):
            confirmatory_comparison_rows.extend(
                paired_condition_comparisons(
                    per_seed_rows,
                    endpoint=endpoint,
                    comparisons=comparison_specs,
                    bootstrap_samples=int(
                        cfg.get("statistics", {}).get("bootstrap_samples", 5000)
                    ),
                    seed=int(cfg.get("seed", 42))
                    + len(confirmatory_comparison_rows),
                )
            )
    write_csv_strict(output_dir / "factorial_learning_curves.csv", curve_rows)
    write_csv_strict(output_dir / "factorial_generalization_metrics.csv", generalization_rows)
    write_csv_strict(output_dir / "factorial_per_trajectory.csv", per_trajectory)
    write_csv_strict(output_dir / "factorial_update_curves.csv", update_rows)
    write_csv_strict(output_dir / "factorial_per_seed.csv", per_seed_rows)
    write_csv_strict(output_dir / "factorial_contrasts.csv", contrasts, required=args.mode == "screening")
    if args.mode == "confirmatory":
        write_csv_strict(
            output_dir / "factorial_confirmatory_comparisons.csv",
            confirmatory_comparison_rows,
        )
    write_csv_strict(output_dir / "matched_random_diagnostics.csv", matching_rows)
    extra_controls_enabled = bool((cfg.get("extra_controls", {}) or {}).get("enabled", False))
    write_json(
        {
            "schema_version": "recurrent_factorial_extra_controls_v1",
            "claim_scope": CLAIM_SCOPE,
            "enabled": extra_controls_enabled,
            "controls": list((cfg.get("extra_controls", {}) or {}).get("controls", [])),
            "diagnostics": extra_control_rows,
        },
        output_dir / "extra_control_diagnostics.json",
    )
    if extra_control_rows:
        write_csv_strict(output_dir / "extra_control_diagnostics.csv", extra_control_rows)
    write_json({"schema_version": "factorial_checkpoint_manifest_v1", "checkpoints": checkpoint_rows}, output_dir / "checkpoint_manifest.json")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "claim_scope": CLAIM_SCOPE,
        "mode": args.mode,
        "source_linkage": source_link,
        "navigation_checkpoint": nav_meta,
        "target_seeds": seeds,
        "episode_budgets": budgets,
        "cells": [cell.as_dict() for cell in cells],
        "extra_controls_enabled": extra_controls_enabled,
        "extra_controls": [
            row for row in per_seed_rows if not bool(row.get("is_factorial_cell"))
        ],
        "complete_16_cell_factorial": len(cells) == 16,
        "primary_endpoints": [
            "navigation W_hh main effect",
            "W_hh origin × W_ih origin interaction",
            "low-data AUC and long/composition accuracy",
        ],
        "coordinate_supervision": "off",
        "per_seed_metrics": per_seed_rows,
        "factorial_contrasts": contrasts,
        "confirmatory_comparisons": confirmatory_comparison_rows,
        "checkpoint_count": len(checkpoint_rows),
    }
    write_json(summary, output_dir / "factorial_summary.json")
    # The artifact manifest is written last so hashes for experiment.log remain
    # stable.  No logger calls may follow build_artifact_manifest().
    logger.info("Completed recurrent factorial: %s", output_dir)
    provenance = save_run_provenance(
        output_dir, "recurrent_factorial", cfg=cfg,
        extra={
            "schema_version": SCHEMA_VERSION, "claim_scope": CLAIM_SCOPE,
            "config_path": str(Path(args.config).resolve()),
            "config_sha256": sha256_file(args.config),
            "navigation_checkpoint_sha256": nav_meta["checkpoint_sha256"],
            "source_reduced_artifact_sha256": source.sha256,
        },
    )
    required = [
        "config_used.yaml", "experiment.log", "factorial_summary.json",
        "factorial_learning_curves.csv", "factorial_generalization_metrics.csv",
        "factorial_per_trajectory.csv", "factorial_update_curves.csv",
        "factorial_per_seed.csv", "checkpoint_manifest.json",
        "matched_random_diagnostics.csv", "extra_control_diagnostics.json",
        "run_provenance.json",
    ]
    if args.mode == "screening":
        required.append("factorial_contrasts.csv")
    else:
        required.append("factorial_confirmatory_comparisons.csv")
    if extra_control_rows:
        required.append("extra_control_diagnostics.csv")
    build_artifact_manifest(
        output_dir, schema_version=SCHEMA_VERSION, required_files=required,
        metadata={"mode": args.mode, "source_tree_sha256": provenance["source_tree_sha256"]},
    )


if __name__ == "__main__":
    main()
