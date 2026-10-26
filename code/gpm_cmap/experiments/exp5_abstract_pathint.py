"""Experiment 5 — Abstract Path-Integration Reasoning."""
from __future__ import annotations

import argparse
import copy
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from gpm_cmap.analysis.grid_score import batch_grid_scores
from gpm_cmap.analysis.repr_geometry import pca_projection
from gpm_cmap.data.graph_tasks.pathint_episodes import build_pathint_loaders, split_manifest
from gpm_cmap.models.abstract_pathint import AbstractPathIntModel
from gpm_cmap.tasks.abstract_pathint import evaluate_pathint, train_pathint_epoch
from gpm_cmap.utils import (
    save_run_provenance,
    configure_runtime,
    load_config,
    prepare_output_dir,
    save_config_snapshot,
    set_seed,
    warn_unconsumed_model_keys,
    write_json,
)
from gpm_cmap.utils.checkpointing import load_checkpoint, save_checkpoint, sha256_file
from gpm_cmap.utils.logging_utils import get_logger
from gpm_cmap.utils.metrics import compute_sample_efficiency


CONDITIONS = {
    "transfer_freeze": {"transfer": True, "freeze": True, "freeze_mode": "all"},
    "transfer_finetune": {"transfer": True, "freeze": False, "freeze_mode": "all"},
    "random_init": {"transfer": False, "freeze": False, "freeze_mode": "all"},
    # Frozen-random control: the untrained GridRNN is frozen while the start and
    # readout heads train. This separates transfer content from trainability of
    # the recurrent core.
    "random_frozen": {"transfer": False, "freeze": True, "freeze_mode": "all"},
    # Explicit source decomposition for the headline factorial experiment.
    "transfer_input_only": {
        "transfer": True,
        "freeze": False,
        "freeze_mode": "all",
        "transfer_mode": "input_only",
    },
    "transfer_recurrent_only": {
        "transfer": True,
        "freeze": False,
        "freeze_mode": "all",
        "transfer_mode": "recurrent_only",
    },
}


def get_primary_transfer_checkpoint(cfg: dict) -> str:
    if cfg.get("transfer_checkpoint"):
        return cfg["transfer_checkpoint"]
    transfer_cfg = cfg.get("transfer", {})
    candidates = transfer_cfg.get("checkpoint_candidates", [])
    if candidates:
        return candidates[-1]
    return ""


def validate_primary_transfer_checkpoint(
    cfg: dict, checkpoint_path: str, transfer_mode: str
) -> Dict[str, object]:
    """Preflight the configured transfer source before any target training.

    Full transfer must copy both two-dimensional input and recurrent tensors;
    accepting a partial shape match would silently mislabel the target arm.
    CTL-01 checkpoints additionally require a passing training gate.
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Transfer checkpoint not found: {path}")
    checkpoint = load_checkpoint(path, map_location="cpu")
    state = checkpoint.get("model_state")
    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint has no model_state mapping: {path}")
    hidden_dim = int(cfg["model"]["hidden_dim"])
    required = {
        "full": (
            ("grid_rnn.cell.in_proj.weight", [hidden_dim, 2]),
            ("grid_rnn.cell.h_proj.weight", [hidden_dim, hidden_dim]),
        ),
        "input_only": (("grid_rnn.cell.in_proj.weight", [hidden_dim, 2]),),
        "recurrent_only": (
            ("grid_rnn.cell.h_proj.weight", [hidden_dim, hidden_dim]),
        ),
    }
    if transfer_mode not in required:
        raise ValueError(f"Unsupported transfer mode: {transfer_mode!r}")
    shapes = {}
    for key, expected in required[transfer_mode]:
        if key not in state:
            raise ValueError(f"Transfer checkpoint is missing {key!r}: {path}")
        observed = list(state[key].shape)
        shapes[key] = observed
        if observed != expected:
            raise ValueError(
                f"Transfer checkpoint shape mismatch for {key}: "
                f"observed={observed}, expected={expected}"
            )

    source_type = checkpoint.get("pretraining_source_type")
    is_ctl01 = source_type == "non_navigation_delayed_reconstruction"
    require_gate = bool(
        cfg.get("transfer", {}).get("require_training_gate", False)
    ) or is_ctl01
    gate = checkpoint.get("ctl01_gate")
    if require_gate and (not isinstance(gate, dict) or gate.get("passed") is not True):
        raise RuntimeError(
            "The configured sequence-control source has not passed CTL-01's "
            f"training gate and must not be used for transfer: {path}"
        )
    return {
        "checkpoint": str(path),
        "sha256": sha256_file(path),
        "transfer_mode": transfer_mode,
        "validated_shapes": shapes,
        "pretraining_source_type": source_type,
        "ctl01_gate": gate,
        "training_gate_required": require_gate,
    }


def resolve_condition_names(requested: List[str], cfg: dict, logger) -> List[str]:
    """Drop transfer conditions when the configured source checkpoint is unavailable."""
    ckpt = get_primary_transfer_checkpoint(cfg)
    has_transfer = bool(ckpt) and Path(ckpt).exists()
    valid = []
    skipped = []
    for name in requested:
        if CONDITIONS[name].get("transfer", False) and not has_transfer:
            skipped.append(name)
        else:
            valid.append(name)

    if skipped:
        logger.warning(
            "Skipping transfer conditions %s because transfer_checkpoint is unavailable: %s",
            skipped,
            ckpt or "(none)",
        )
    if not valid:
        logger.warning("No runnable transfer condition remains; falling back to random_init.")
        valid = ["random_init"]
    return valid


# Keys under cfg["model"] that build_model actually forwards. Any other
# model key is silently ignored (rule #12). Audited once in main() where the
# logger is in scope (see warn_unconsumed_model_keys call after config snapshot).
_EXP5_CONSUMED_MODEL_KEYS = {
    "hidden_dim", "content_dim", "nonnegative", "cell_type", "use_bias",
    "dropout", "periodic_scales", "use_position_head", "start_init", "coord_dim",
    "periodic_encoder_kind", "rhc_moduli", "lattice_shape",
}


def build_model(cfg: dict, n_nodes: int, device: torch.device) -> AbstractPathIntModel:
    mcfg = cfg["model"]
    model = AbstractPathIntModel(
        n_nodes=n_nodes,
        hidden_dim=mcfg["hidden_dim"],
        content_dim=mcfg.get("content_dim", 64),
        nonnegative=mcfg.get("nonnegative", True),
        cell_type=mcfg.get("cell_type", "rnn"),
        use_bias=mcfg.get("use_bias", False),
        dropout=mcfg.get("dropout", 0.0),
        periodic_scales=mcfg.get("periodic_scales", 0),
        use_position_head=mcfg.get("use_position_head", True),
        start_init=mcfg.get("start_init", "embedding"),
        coord_dim=mcfg.get("coord_dim", 2),
        periodic_encoder_kind=mcfg.get("periodic_encoder_kind", "periodic_learned"),
        rhc_moduli=mcfg.get("rhc_moduli", None),
        lattice_shape=tuple(mcfg["lattice_shape"]) if mcfg.get("lattice_shape") else None,
    )
    return model.to(device)


def run_condition(
    name: str,
    opts: dict,
    cfg: dict,
    train_loader,
    validation_loader,
    test_loader,
    n_nodes: int,
    device: torch.device,
    logger,
    output_dir: Path,
) -> Tuple[List[dict], Dict[str, object]]:
    """Train one condition, return (history, final_eval)."""
    model = build_model(cfg, n_nodes, device)
    tcfg = cfg["train"]

    checkpoint_path = opts.get("checkpoint_path", "")
    transfer_mode = opts.get("transfer_mode", cfg.get("transfer", {}).get("mode", "full"))
    counts = {"copied": 0, "skipped": 0, "missing": 0}
    transfer_succeeded = False
    if opts.get("transfer", False):
        if not (checkpoint_path and Path(checkpoint_path).exists()):
            # Transfer was explicitly requested but the checkpoint is missing.
            # In the orchestrated path, resolve_condition_names() already drops
            # transfer conditions when the file is absent; this guard is
            # defense-in-depth for any caller that reaches run_condition()
            # directly. Silently continuing would mislabel a randomly-
            # initialised core as a "transfer" condition (rule #12).
            raise FileNotFoundError(
                f"Transfer condition '{name}' requested transfer "
                f"but checkpoint is unavailable: {checkpoint_path!r}. "
                "Run exp1_nav_pretrain to produce it, or point transfer_checkpoint "
                "at an existing .pt file."
            )
        counts = model.load_nav_checkpoint(
            checkpoint_path, transfer_mode=transfer_mode, strict=True
        )
        logger.info(
            "  Source transfer: mode=%s copied=%d skipped=%d missing=%d ckpt=%s",
            transfer_mode,
            counts["copied"],
            counts["skipped"],
            counts["missing"],
            checkpoint_path,
        )
        expected_copied = 2 if transfer_mode == "full" else 1
        if (
            counts["copied"] != expected_copied
            or counts["skipped"] != 0
            or counts["missing"] != 0
        ):
            raise RuntimeError(
                f"Transfer condition '{name}' did not copy the complete requested "
                f"tensor set from {checkpoint_path!r}: mode={transfer_mode}, "
                f"counts={counts}, expected_copied={expected_copied}."
            )
        transfer_succeeded = True

    if opts.get("freeze", False):
        if opts.get("transfer", False) and not transfer_succeeded:
            # Defensive: should be unreachable because failed transfer raises
            # above, but never freeze a core that was not actually transferred.
            raise RuntimeError(
                f"Refusing to freeze GridRNN for condition '{name}': "
                "transfer was requested but did not copy any parameters."
            )
        freeze_mode = opts.get("freeze_mode", "all")
        if freeze_mode == "all":
            model.freeze_grid_rnn_all()
        elif freeze_mode == "recurrent_only":
            model.freeze_grid_rnn_recurrent_only()
        else:
            raise ValueError(f"Unsupported freeze_mode: {freeze_mode}")
        logger.info("  GridRNN frozen (%s)", freeze_mode)

    trainable = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(
        trainable,
        lr=tcfg["lr"],
        weight_decay=tcfg.get("weight_decay", 0.0),
    )

    history: List[dict] = []
    # Allow a condition to override coordinate-supervision weights so we
    # can ablate position-head + RHC supervision (default falls back to the
    # global train config, i.e. unchanged behaviour).
    cond_position_w = opts.get("position_loss_weight", tcfg.get("position_loss_weight", 0.0))
    cond_rhc_w = opts.get("rhc_alignment_weight", tcfg.get("rhc_alignment_weight", 0.0))
    selection_metric = str(tcfg.get("selection_metric", "validation_accuracy"))
    if selection_metric not in {"validation_accuracy", "validation_loss"}:
        raise ValueError(
            "train.selection_metric must be 'validation_accuracy' or 'validation_loss'; "
            f"got {selection_metric!r}"
        )
    best_score = -float("inf") if selection_metric == "validation_accuracy" else float("inf")
    best_epoch = 0
    best_checkpoint_path = output_dir / "best_validation.pt"
    position_supervised = cond_position_w > 0.0

    for epoch in range(1, tcfg["epochs"] + 1):
        train_stats = train_pathint_epoch(
            model,
            train_loader,
            optimizer,
            device,
            grad_clip=tcfg.get("grad_clip", 1.0),
            whh_weight_decay=tcfg.get("whh_weight_decay", 0.0),
            position_loss_weight=cond_position_w,
            rhc_alignment_weight=cond_rhc_w,
        )
        entry = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "train_acc": train_stats["accuracy"],
        }

        eval_every = int(tcfg.get("eval_every", 5))
        if epoch % eval_every == 0 or epoch == tcfg["epochs"]:
            validation_stats = evaluate_pathint(
                model,
                validation_loader,
                device,
                position_supervised=position_supervised,
                evaluation_seed=cfg.get("seed", 0),
            )
            entry["validation_loss"] = validation_stats["loss"]
            entry["validation_acc"] = validation_stats["accuracy"]
            entry["validation_geometry_corr"] = validation_stats["geometry_corr"]
            entry["validation_sparsity_index"] = validation_stats["sparsity_index"]
            current_score = (
                validation_stats["accuracy"]
                if selection_metric == "validation_accuracy"
                else validation_stats["loss"]
            )
            improved = (
                current_score > best_score
                if selection_metric == "validation_accuracy"
                else current_score < best_score
            )
            if improved:
                best_score = float(current_score)
                best_epoch = epoch
                save_checkpoint(
                    {
                        "model_state": model.state_dict(),
                        "config": cfg,
                        "condition": name,
                        "epoch": epoch,
                        "selection_metric": selection_metric,
                        "selection_score": best_score,
                    },
                    best_checkpoint_path,
                )
            logger.info(
                "  [%s] epoch=%d train_acc=%.4f val_acc=%.4f val_loss=%.4f best_epoch=%d",
                name,
                epoch,
                train_stats["accuracy"],
                validation_stats["accuracy"],
                validation_stats["loss"],
                best_epoch,
            )
        history.append(entry)

    if not best_checkpoint_path.exists():
        raise RuntimeError(f"No validation checkpoint was produced for condition {name!r}")
    best_checkpoint = load_checkpoint(best_checkpoint_path, map_location=device)
    model.load_state_dict(best_checkpoint["model_state"])
    final_eval = evaluate_pathint(
        model,
        test_loader,
        device,
        position_supervised=position_supervised,
        evaluation_seed=cfg.get("seed", 0),
    )
    final_eval["selection_metric"] = selection_metric
    final_eval["best_validation_score"] = best_score
    final_eval["best_epoch"] = best_epoch
    final_eval["test_calls"] = 1
    final_eval["best_checkpoint"] = str(best_checkpoint_path)
    final_eval["best_checkpoint_sha256"] = sha256_file(best_checkpoint_path)
    final_eval["transfer_metadata"] = {
        "requested": bool(opts.get("transfer", False)),
        "applied": transfer_succeeded,
        "mode": transfer_mode if transfer_succeeded else "none",
        "copied_tensors": int(counts["copied"]),
        "skipped_tensors": int(counts["skipped"]),
        "missing_tensors": int(counts["missing"]),
        "checkpoint_path": str(checkpoint_path) if transfer_succeeded else None,
        "checkpoint_sha256": sha256_file(checkpoint_path) if transfer_succeeded else None,
        "resolved_condition_label": name,
    }

    # Optional cross-graph generalization: evaluate the trained model on
    # FOREIGN graphs using coordinate readout only (node classification can't
    # transfer across different node vocabularies). Reuses the in-place trained
    # model; results are attached under final_eval["cross_graph"].
    cross_graph_targets = opts.get("cross_graph_eval", None)
    if cross_graph_targets:
        from gpm_cmap.tasks.abstract_pathint import evaluate_cross_graph
        dcfg = cfg["data"]
        cg_results: Dict[str, object] = {}
        for tgt in cross_graph_targets:
            tgt_name = tgt["task_name"]
            fe_loader = build_pathint_loaders(
                task_name=tgt_name,
                num_episodes=tgt.get("num_episodes", dcfg.get("num_episodes", 2000)),
                seq_len=tgt.get("seq_len", dcfg.get("seq_len", 8)),
                batch_size=dcfg.get("batch_size", 64),
                split_ratio=dcfg.get("split_ratio", 0.8),
                velocity_scale=tgt.get("velocity_scale", dcfg.get("velocity_scale", 0.02)),
                seed=cfg.get("seed", 42),
                n_levels=tgt.get("n_levels", dcfg.get("n_levels", 4)),
                n_items=tgt.get("n_items", dcfg.get("n_items", 7)),
                coord_strategy=dcfg.get("coord_strategy", None),
                coord_dim=dcfg.get("coord_dim", 2),
                normalize_displacements=dcfg.get("normalize_displacements", True),
                coord_scale=tgt.get("coord_scale", dcfg.get("coord_scale", 1.0)),
                split_strategy=tgt.get("split_strategy", dcfg.get("split_strategy", "random")),
                seq_len_range=tgt.get("seq_len_range", dcfg.get("seq_len_range", None)),
                num_workers=dcfg.get("num_workers", 0),
            )[1]  # use the test loader of the foreign graph
            cg_results[tgt.get("label", tgt_name)] = evaluate_cross_graph(model, fe_loader, device)
            logger.info(
                "  [%s] cross-graph eval on %s: geometry_corr=%s position_error=%s",
                name, tgt_name,
                cg_results[tgt.get("label", tgt_name)].get("geometry_corr"),
                cg_results[tgt.get("label", tgt_name)].get("position_error"),
            )
        final_eval["cross_graph"] = cg_results

    # Optional behavioral cross-graph transfer: freeze the trained core and
    # re-fit a fresh head on each foreign graph (reports target accuracy +
    # sample-efficiency). Distinct from the T14 coordinate eval above.
    refit_targets = opts.get("cross_graph_refit", None)
    if refit_targets:
        from gpm_cmap.tasks.abstract_pathint import evaluate_cross_graph_refit
        dcfg = cfg["data"]
        refit_results: Dict[str, object] = {}
        for tgt in refit_targets:
            tgt_name = tgt["task_name"]
            (
                tr_loader,
                validation_loader,
                te_loader,
                _t,
                n2i_tgt,
                _a,
                _nc,
            ) = build_pathint_loaders(
                task_name=tgt_name,
                num_episodes=tgt.get("num_episodes", dcfg.get("num_episodes", 2000)),
                seq_len=tgt.get("seq_len", dcfg.get("seq_len", 8)),
                batch_size=dcfg.get("batch_size", 64),
                split_ratio=dcfg.get("split_ratio", 0.7),
                validation_ratio=tgt.get(
                    "validation_ratio", dcfg.get("validation_ratio", 0.1)
                ),
                return_validation=True,
                velocity_scale=tgt.get("velocity_scale", dcfg.get("velocity_scale", 0.02)),
                seed=cfg.get("seed", 42),
                n_levels=tgt.get("n_levels", dcfg.get("n_levels", 4)),
                n_items=tgt.get("n_items", dcfg.get("n_items", 7)),
                coord_strategy=dcfg.get("coord_strategy", None),
                coord_dim=dcfg.get("coord_dim", 2),
                normalize_displacements=dcfg.get("normalize_displacements", True),
                coord_scale=tgt.get("coord_scale", dcfg.get("coord_scale", 1.0)),
                split_strategy=tgt.get("split_strategy", dcfg.get("split_strategy", "random")),
                seq_len_range=tgt.get("seq_len_range", dcfg.get("seq_len_range", None)),
                num_workers=dcfg.get("num_workers", 0),
            )
            refit_results[tgt.get("label", tgt_name)] = evaluate_cross_graph_refit(
                model,
                tr_loader,
                validation_loader,
                te_loader,
                n_nodes_target=len(n2i_tgt),
                device=device,
                epochs=tgt.get("refit_epochs", 15),
            )
            logger.info(
                "  [%s] cross-graph REFIT on %s: final_acc=%s epochs_to_thr=%s",
                name, tgt_name,
                refit_results[tgt.get("label", tgt_name)].get("final_test_acc"),
                refit_results[tgt.get("label", tgt_name)].get("epochs_to_threshold"),
            )
        final_eval["cross_graph_refit"] = refit_results

    n_hidden = final_eval["hidden_all"].shape[0]
    hidden_dim = final_eval["hidden_all"].shape[1]
    analysis_cfg = cfg.get("analysis", {})
    max_units_for_grid = min(hidden_dim, analysis_cfg.get("max_grid_units", 64))

    unique_coords = np.unique(final_eval["coords_all"], axis=0)
    min_unique_coords_for_grid = analysis_cfg.get("min_unique_coords_for_grid", 25)
    force_discrete_grid_score = analysis_cfg.get("force_grid_score_on_discrete", False)

    if unique_coords.shape[0] < min_unique_coords_for_grid and not force_discrete_grid_score:
        # Sparse discrete graphs such as 4x4 social hierarchy do not provide a
        # dense rate map.  A grid-score of 0 in this setting is not evidence
        # against grid-like structure; it is an invalid measurement.
        final_eval["grid_scores"] = []
        final_eval["mean_grid_score"] = None
        final_eval["max_grid_score"] = None
        final_eval["frac_grid_above_03"] = None
        final_eval["mean_grid_score_conservative"] = None
        final_eval["max_grid_score_conservative"] = None
        final_eval["grid_score_applicable"] = False
        final_eval["grid_score_note"] = (
            f"N/A: only {unique_coords.shape[0]} unique discrete coordinates; "
            f"need at least {min_unique_coords_for_grid} for rate-map gridness."
        )
        logger.info("  [%s] Grid score skipped: %s", name, final_eval["grid_score_note"])
    elif n_hidden >= 100:
        n_sample = min(n_hidden, 10000)
        idx = np.random.choice(n_hidden, n_sample, replace=False)
        sampled_hidden = final_eval["hidden_all"][idx, :max_units_for_grid]
        sampled_coords = final_eval["coords_all"][idx]
        grid_results = batch_grid_scores(
            positions=sampled_coords,
            hidden_activations=sampled_hidden,
            bins=analysis_cfg.get("grid_bins", 20),
            arena_size=analysis_cfg.get("grid_arena_size", 4.0),
            smooth_sigma=analysis_cfg.get("grid_smooth_sigma", 0.0),
        )
        grid_scores = [float(result["grid_score"]) for result in grid_results]
        final_eval["grid_scores"] = grid_scores
        final_eval["mean_grid_score"] = float(np.mean(grid_scores))
        final_eval["max_grid_score"] = float(np.max(grid_scores))
        final_eval["frac_grid_above_03"] = float(np.mean([score > 0.3 for score in grid_scores]))
        # conservative (min-max) gridness, aggregated over finite values
        # (completes T8: per-unit dicts already carry grid_score_conservative).
        cons = [
            float(r.get("grid_score_conservative", float("nan"))) for r in grid_results
        ]
        finite_cons = [c for c in cons if np.isfinite(c)]
        final_eval["mean_grid_score_conservative"] = (
            float(np.mean(finite_cons)) if finite_cons else None
        )
        final_eval["max_grid_score_conservative"] = (
            float(np.max(finite_cons)) if finite_cons else None
        )
        final_eval["grid_score_applicable"] = True
        final_eval["grid_score_note"] = "computed"
        logger.info(
            "  [%s] Grid: mean=%.4f max=%.4f frac>0.3=%.2f%%",
            name,
            final_eval["mean_grid_score"],
            final_eval["max_grid_score"],
            final_eval["frac_grid_above_03"] * 100,
        )
    else:
        final_eval["grid_scores"] = []
        final_eval["mean_grid_score"] = None
        final_eval["max_grid_score"] = None
        final_eval["frac_grid_above_03"] = None
        final_eval["mean_grid_score_conservative"] = None
        final_eval["max_grid_score_conservative"] = None
        final_eval["grid_score_applicable"] = False
        final_eval["grid_score_note"] = f"N/A: too few hidden samples for gridness (n={n_hidden})."
        logger.info("  [%s] Grid score skipped: %s", name, final_eval["grid_score_note"])

    save_checkpoint(
        {
            "model_state": model.state_dict(),
            "config": cfg,
            "condition": name,
        },
        output_dir / f"{name}_model.pt",
    )
    final_eval["model_checkpoint"] = str(output_dir / f"{name}_model.pt")
    final_eval["model_checkpoint_sha256"] = sha256_file(output_dir / f"{name}_model.pt")
    return history, final_eval


def _extract_scalar_eval(final_eval: Dict[str, object], save_distance_breakdown: bool) -> Dict[str, object]:
    scalar_eval: Dict[str, object] = {
        "test_acc": float(final_eval["accuracy"]),
        "test_loss": float(final_eval["loss"]),
        "geometry_corr": float(final_eval["geometry_corr"]),
        "geometry_corr_all": float(final_eval["geometry_corr_all"]),
        "sparsity_index": float(final_eval["sparsity_index"]),
        "grid_score_applicable": bool(final_eval.get("grid_score_applicable", False)),
        "grid_score_note": str(final_eval.get("grid_score_note", "")),
    }

    for key in ["mean_grid_score", "max_grid_score", "frac_grid_above_03"]:
        value = final_eval.get(key, None)
        if value is not None:
            scalar_eval[key] = float(value)

    # Position metrics are intentionally unavailable when coordinate supervision
    # is disabled; preserve that as N/A instead of coercing None to float.
    position_mse = final_eval.get("position_mse")
    if position_mse is not None:
        scalar_eval["position_mse"] = float(position_mse)

    position_l2_error = final_eval.get("position_l2_error")
    if position_l2_error is not None:
        scalar_eval["position_l2_error"] = float(position_l2_error)
        scalar_eval["position_error"] = float(position_l2_error)

    if save_distance_breakdown:
        accuracy_breakdown = final_eval.get("accuracy_by_l1_distance")
        if isinstance(accuracy_breakdown, dict):
            scalar_eval["accuracy_by_l1_distance"] = {
                str(k): float(v) for k, v in accuracy_breakdown.items()
            }

        position_error_breakdown = final_eval.get("position_error_by_l1_distance")
        if isinstance(position_error_breakdown, dict):
            scalar_eval["position_error_by_l1_distance"] = {
                str(k): float(v) for k, v in position_error_breakdown.items()
            }

        position_mse_breakdown = final_eval.get("position_mse_by_l1_distance")
        if isinstance(position_mse_breakdown, dict):
            scalar_eval["position_mse_by_l1_distance"] = {
                str(k): float(v) for k, v in position_mse_breakdown.items()
            }

    for key in (
        "selection_metric",
        "best_validation_score",
        "best_epoch",
        "test_calls",
        "best_checkpoint",
        "best_checkpoint_sha256",
        "transfer_metadata",
        "model_checkpoint",
        "model_checkpoint_sha256",
    ):
        if key in final_eval:
            scalar_eval[key] = final_eval[key]

    # Cross-graph results are nested by target graph. Preserve them verbatim so
    # the aggregate step cannot substitute in-graph metrics for foreign-graph
    # metrics when producing the control summaries.
    for nested_key in ("cross_graph", "cross_graph_refit"):
        if nested_key in final_eval:
            scalar_eval[nested_key] = final_eval[nested_key]

    return scalar_eval


def _aggregate_distance_breakdown(evals: List[Dict[str, object]], key: str) -> Dict[str, Dict[str, float]]:
    all_distances = sorted({
        distance
        for eval_dict in evals
        for distance in eval_dict.get(key, {}).keys()
    }, key=lambda x: float(x))
    summary: Dict[str, Dict[str, float]] = {}
    for distance in all_distances:
        values = [
            float(eval_dict[key][distance])
            for eval_dict in evals
            if key in eval_dict and distance in eval_dict[key]
        ]
        if values:
            summary[distance] = {
                "mean": float(np.mean(values)),
                "sem": float(np.std(values) / max(1, np.sqrt(len(values)))),
            }
    return summary


def aggregate_block(
    histories_by_condition: Dict[str, List[List[dict]]],
    evals_by_condition: Dict[str, List[Dict[str, object]]],
) -> Dict[str, Dict[str, object]]:
    metric_keys = [
        "test_acc",
        "geometry_corr",
        "geometry_corr_all",
        "sparsity_index",
        "mean_grid_score",
        "max_grid_score",
        "frac_grid_above_03",
        "position_mse",
        "position_l2_error",
        "position_error",
    ]
    summary: Dict[str, Dict[str, object]] = {}
    for cond_name, evals in evals_by_condition.items():
        entry: Dict[str, object] = {}
        for metric_key in metric_keys:
            values = [float(e[metric_key]) for e in evals if metric_key in e and e[metric_key] is not None]
            if values:
                entry[f"{metric_key}_mean"] = float(np.mean(values))
                entry[f"{metric_key}_sem"] = float(np.std(values) / max(1, np.sqrt(len(values))))

        efficiencies = []
        for history in histories_by_condition.get(cond_name, []):
            validation_entries = [entry for entry in history if "validation_acc" in entry]
            efficiency = compute_sample_efficiency(
                validation_entries, metric_key="validation_acc", threshold=0.8
            )
            efficiencies.append(efficiency)
        valid_efficiencies = [eff for eff in efficiencies if eff > 0]
        entry["sample_efficiency_mean"] = float(np.mean(valid_efficiencies)) if valid_efficiencies else -1.0
        entry["sample_efficiency_sem"] = (
            float(np.std(valid_efficiencies) / max(1, np.sqrt(len(valid_efficiencies))))
            if valid_efficiencies else 0.0
        )
        entry["n_reached_80"] = len(valid_efficiencies)
        entry["n_seeds"] = len(evals)

        # Preserve convergence-speed metrics alongside final accuracy. These are
        # useful whenever endpoint accuracy alone does not separate conditions.
        histories_for_cond = histories_by_condition.get(cond_name, [])
        for thr in (0.5, 0.8, 0.95):
            effs = [
                compute_sample_efficiency(
                    [e for e in h if "validation_acc" in e],
                    metric_key="validation_acc",
                    threshold=thr,
                )
                for h in histories_for_cond
            ]
            reached = [e for e in effs if e > 0]
            entry[f"epochs_to_{int(thr*100)}_mean"] = float(np.mean(reached)) if reached else -1.0
            entry[f"epochs_to_{int(thr*100)}_sem"] = (
                float(np.std(reached) / max(1, np.sqrt(len(reached)))) if reached else 0.0
            )
            # n_reached makes "never reached" VISIBLE rather than silently
            # dropped from the mean (rule #12).
            entry[f"n_reached_{int(thr*100)}"] = len(reached)
        # Early validation accuracy (mean over the first up-to-3 evaluations):
        # a direct, saturation-free convergence-speed signal.
        early_vals = []
        for h in histories_for_cond:
            te = sorted(
                [e for e in h if "validation_acc" in e],
                key=lambda e: e.get("epoch", 0),
            )[:3]
            if te:
                early_vals.append(float(np.mean([e["validation_acc"] for e in te])))
        if early_vals:
            entry["early_epoch_acc_mean"] = float(np.mean(early_vals))
            entry["early_epoch_acc_sem"] = float(np.std(early_vals) / max(1, np.sqrt(len(early_vals))))

        accuracy_breakdown = _aggregate_distance_breakdown(evals, "accuracy_by_l1_distance")
        if accuracy_breakdown:
            entry["accuracy_by_l1_distance"] = accuracy_breakdown
        pos_breakdown = _aggregate_distance_breakdown(evals, "position_error_by_l1_distance")
        if pos_breakdown:
            entry["position_error_by_l1_distance"] = pos_breakdown
        pos_mse_breakdown = _aggregate_distance_breakdown(evals, "position_mse_by_l1_distance")
        if pos_mse_breakdown:
            entry["position_mse_by_l1_distance"] = pos_mse_breakdown

        transfer_metadata = [
            eval_dict.get("transfer_metadata")
            for eval_dict in evals
            if isinstance(eval_dict.get("transfer_metadata"), dict)
        ]
        if transfer_metadata:
            entry["resolved_condition_labels"] = sorted(
                {
                    str(metadata.get("resolved_condition_label", cond_name))
                    for metadata in transfer_metadata
                }
            )
            entry["transfer_applied_all_seeds"] = all(
                bool(metadata.get("applied", False)) for metadata in transfer_metadata
            )
            entry["transfer_checkpoint_sha256"] = sorted(
                {
                    str(metadata["checkpoint_sha256"])
                    for metadata in transfer_metadata
                    if metadata.get("checkpoint_sha256")
                }
            )

        test_calls = [
            int(eval_dict["test_calls"])
            for eval_dict in evals
            if isinstance(eval_dict.get("test_calls"), (int, float))
        ]
        if test_calls:
            entry["test_calls_per_seed"] = test_calls

        # Aggregate cross-graph blocks passed through above. Prefixes keep
        # foreign-graph values structurally separate from in-graph metrics.
        for nested_key, prefix in (("cross_graph", "cross_graph"),
                                   ("cross_graph_refit", "cross_graph_refit")):
            nested = [e[nested_key] for e in evals
                      if isinstance(e.get(nested_key), dict)]
            if not nested:
                continue
            graph_labels = sorted({lbl for d in nested for lbl in d.keys()})
            for lbl in graph_labels:
                per_graph = [d[lbl] for d in nested if isinstance(d.get(lbl), dict)]
                if not per_graph:
                    continue
                metric_names = sorted({
                    m for d in per_graph for m, v in d.items()
                    if isinstance(v, (int, float)) and not isinstance(v, bool)
                })
                for m in metric_names:
                    vals = [float(d[m]) for d in per_graph
                            if isinstance(d.get(m), (int, float))
                            and not isinstance(d.get(m), bool)
                            and np.isfinite(float(d[m]))]
                    if not vals:
                        continue
                    entry[f"{prefix}__{lbl}__{m}_mean"] = float(np.mean(vals))
                    entry[f"{prefix}__{lbl}__{m}_sem"] = float(
                        np.std(vals) / max(1, np.sqrt(len(vals)))
                    )
                    entry[f"{prefix}__{lbl}__{m}_n"] = len(vals)

        summary[cond_name] = entry
    return summary


def run_block(
    block_name: str,
    condition_specs: Dict[str, dict],
    cfg: dict,
    n_seeds: int,
    device: torch.device,
    logger,
    output_dir: Path,
    main_block: bool = False,
) -> Tuple[Dict[str, List[List[dict]]], Dict[str, List[Dict[str, object]]], Dict[str, List[Dict[str, object]]]]:
    histories_by_condition: Dict[str, List[List[dict]]] = defaultdict(list)
    evals_by_condition: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    raw_per_seed: Dict[str, List[Dict[str, object]]] = defaultdict(list)

    base_seed = cfg["seed"]
    dcfg = cfg["data"]
    save_distance_breakdown = cfg.get("analysis", {}).get("save_distance_breakdown", True)

    for seed_idx in range(n_seeds):
        seed = base_seed + seed_idx
        logger.info("\n=== %s: seed %d/%d (seed=%d) ===", block_name, seed_idx + 1, n_seeds, seed)

        for cond_name, opts in condition_specs.items():
            set_seed(seed)
            condition_cfg = copy.deepcopy(cfg)
            condition_cfg["seed"] = seed
            action_map_mode = opts.get("action_map_mode", cfg.get("controls", {}).get("action_map_mode", "aligned"))
            logger.info("\n--- %s / %s (seed=%d, action_map_mode=%s) ---", block_name, cond_name, seed, action_map_mode)

            train_loader, validation_loader, test_loader, _task, node_to_id, _action_to_id, _node_coords = build_pathint_loaders(
                task_name=dcfg.get("task_name", "social_hierarchy"),
                num_episodes=dcfg["num_episodes"],
                seq_len=dcfg["seq_len"],
                batch_size=dcfg["batch_size"],
                split_ratio=dcfg.get("split_ratio", 0.8),
                velocity_scale=dcfg["velocity_scale"],
                seed=seed,
                n_levels=dcfg.get("n_levels", 4),
                n_items=dcfg.get("n_items", 7),
                action_map_mode=action_map_mode,
                coord_strategy=dcfg.get("coord_strategy", None),
                coord_dim=dcfg.get("coord_dim", 2),
                normalize_displacements=dcfg.get("normalize_displacements", True),
                coord_scale=dcfg.get("coord_scale", 1.0),
                split_strategy=opts.get("split_strategy", dcfg.get("split_strategy", "random")),
                velocity_transform=opts.get("velocity_transform", dcfg.get("velocity_transform", "none")),
                # test-only perturbation (true OOD equivariance probe);
                # None keeps the default path byte-identical.
                test_only_velocity_transform=opts.get(
                    "test_only_velocity_transform",
                    dcfg.get("test_only_velocity_transform", None),
                ),
                seq_len_range=opts.get("seq_len_range", dcfg.get("seq_len_range", None)),
                validation_ratio=dcfg.get("validation_ratio", 0.1),
                return_validation=True,
                num_workers=dcfg.get("num_workers", 0),
            )
            n_nodes = len(node_to_id)

            if main_block:
                cond_dir = output_dir / f"seed_{seed}" / cond_name
            else:
                cond_dir = output_dir / block_name / f"seed_{seed}" / cond_name
            cond_dir.mkdir(parents=True, exist_ok=True)

            history, final_eval = run_condition(
                name=cond_name,
                opts=opts,
                cfg=condition_cfg,
                train_loader=train_loader,
                validation_loader=validation_loader,
                test_loader=test_loader,
                n_nodes=n_nodes,
                device=device,
                logger=logger,
                output_dir=cond_dir,
            )

            write_json(history, cond_dir / "history.json")
            write_json(
                split_manifest(
                    train_loader.dataset.episodes,
                    validation_loader.dataset.episodes,
                    test_loader.dataset.episodes,
                ),
                cond_dir / "split_manifest.json",
            )

            plot_geometry_scatter(
                final_eval["final_hidden"],
                final_eval["final_coords"],
                cond_dir / "geometry_scatter.png",
                title=f"{cond_name} (seed={seed})",
            )

            scalar_eval = _extract_scalar_eval(final_eval, save_distance_breakdown=save_distance_breakdown)
            scalar_eval["seed"] = seed
            scalar_eval["condition"] = cond_name
            scalar_eval["block"] = block_name
            scalar_eval["action_map_mode"] = action_map_mode
            scalar_eval["transfer_mode"] = opts.get("transfer_mode", cfg.get("transfer", {}).get("mode", "full"))
            scalar_eval["checkpoint_path"] = opts.get("checkpoint_path", "")

            histories_by_condition[cond_name].append(history)
            evals_by_condition[cond_name].append(scalar_eval)
            raw_per_seed[cond_name].append(scalar_eval)

    return histories_by_condition, evals_by_condition, raw_per_seed


def plot_learning_curves(
    all_histories: Dict[str, List[List[dict]]],
    output_path: Path,
    metric: str = "validation_acc",
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for cond_name, histories in all_histories.items():
        epoch_values = defaultdict(list)
        for history in histories:
            for entry in history:
                if metric in entry:
                    epoch_values[entry["epoch"]].append(entry[metric])
        if not epoch_values:
            continue
        epochs = sorted(epoch_values.keys())
        means = [np.mean(epoch_values[epoch]) for epoch in epochs]
        sems = [np.std(epoch_values[epoch]) / max(1, np.sqrt(len(epoch_values[epoch]))) for epoch in epochs]
        ax.plot(epochs, means, label=cond_name)
        ax.fill_between(epochs, np.array(means) - np.array(sems), np.array(means) + np.array(sems), alpha=0.2)

    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"Experiment 5 Learning Curves — {metric}")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_condition_bars(
    summary: Dict[str, Dict[str, object]],
    metric: str,
    ylabel: str,
    title: str,
    output_path: Path,
    ylim: Tuple[float, float] | None = None,
) -> None:
    names = list(summary.keys())
    means = [float(summary[name].get(f"{metric}_mean", 0.0)) for name in names]
    sems = [float(summary[name].get(f"{metric}_sem", 0.0)) for name in names]

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(names))
    ax.bar(x, means, yerr=sems, capsize=5, edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels([name.replace("_", "\n") for name in names], fontsize=10)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_geometry_scatter(
    final_hidden: np.ndarray,
    final_coords: np.ndarray,
    output_path: Path,
    title: str = "Hidden State Geometry",
) -> None:
    if final_hidden.shape[0] < 4:
        return
    proj, _ = pca_projection(final_hidden, n_components=2)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, dim, name in zip(axes, [0, 1], ["Competence (dim 0)", "Popularity (dim 1)"]):
        scatter = ax.scatter(proj[:, 0], proj[:, 1], c=final_coords[:, dim], cmap="viridis", s=8, alpha=0.7)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(f"Coloured by {name}")
        plt.colorbar(scatter, ax=ax)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_json(obj: object, path: Path) -> None:
    write_json(obj, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp5: Abstract Path-Integration")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--n_seeds", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument(
        "--conditions",
        type=str,
        nargs="*",
        default=None,
        help="Subset of main conditions to run (default: all main conditions)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.output_dir is not None:
        cfg["output_dir"] = args.output_dir
    n_seeds = args.n_seeds if args.n_seeds is not None else cfg.get("n_seeds", 5)
    device = configure_runtime(cfg)
    output_dir = prepare_output_dir(cfg["output_dir"])
    logger = get_logger("exp5_abstract_pathint", output_dir / "experiment.log")
    save_config_snapshot(cfg, output_dir)
    # [PROVENANCE] record code version / seed / environment alongside the
    # config so any figure or metric can be traced to the exact run.
    save_run_provenance(output_dir, "exp5_abstract_pathint", cfg=cfg)
    # Fail loud about model-config keys this experiment never consumes.
    warn_unconsumed_model_keys(
        cfg.get("model", {}), _EXP5_CONSUMED_MODEL_KEYS, logger,
        experiment="exp5_abstract_pathint",
    )

    primary_ckpt = get_primary_transfer_checkpoint(cfg)
    transfer_mode = cfg.get("transfer", {}).get("mode", "full")
    if primary_ckpt and Path(primary_ckpt).exists():
        transfer_manifest = validate_primary_transfer_checkpoint(
            cfg, primary_ckpt, transfer_mode
        )
        write_json(transfer_manifest, output_dir / "transfer_source_manifest.json")
    controls_cfg = cfg.get("controls", {})
    base_action_map_mode = controls_cfg.get("action_map_mode", "aligned")

    cond_names = args.conditions if args.conditions else list(CONDITIONS.keys())
    for cond_name in cond_names:
        if cond_name not in CONDITIONS:
            raise ValueError(f"Unknown condition: {cond_name}. Valid: {list(CONDITIONS.keys())}")
    cond_names = resolve_condition_names(cond_names, cfg, logger)

    logger.info("=" * 80)
    logger.info("Experiment 5: Abstract Path-Integration Reasoning")
    logger.info("  Device: %s", device)
    logger.info("  n_seeds: %d", n_seeds)
    logger.info("  Main conditions: %s", cond_names)
    logger.info("  Primary transfer checkpoint: %s", primary_ckpt or "(none)")
    logger.info("  Default transfer mode: %s", transfer_mode)
    logger.info("  Base action_map_mode: %s", base_action_map_mode)
    logger.info("=" * 80)

    raw_per_seed: Dict[str, Dict[str, List[Dict[str, object]]]] = {}

    main_specs: Dict[str, dict] = {}
    for cond_name in cond_names:
        spec = dict(CONDITIONS[cond_name])
        spec["checkpoint_path"] = primary_ckpt
        spec["transfer_mode"] = spec.get("transfer_mode", transfer_mode)
        spec["action_map_mode"] = base_action_map_mode
        main_specs[cond_name] = spec

    main_histories, main_evals, raw_main = run_block(
        block_name="main",
        condition_specs=main_specs,
        cfg=cfg,
        n_seeds=n_seeds,
        device=device,
        logger=logger,
        output_dir=output_dir,
        main_block=True,
    )
    summary_main = aggregate_block(main_histories, main_evals)
    raw_per_seed["main"] = raw_main

    save_json(summary_main, output_dir / "summary_main.json")
    save_json(summary_main, output_dir / "summary.json")
    plot_learning_curves(main_histories, output_dir / "learning_curves.png")

    for metric, ylabel, title, ylim, filename in [
        ("test_acc", "Test Accuracy", "Final Accuracy", (0, 1.05), "bar_test_acc.png"),
        ("geometry_corr", "Spearman ρ", "Geometry Correlation (final)", None, "bar_geometry_corr.png"),
        ("geometry_corr_all", "Spearman ρ", "Geometry Correlation (all steps)", None, "bar_geometry_corr_all.png"),
        ("sparsity_index", "Sparsity", "Sparsity Index", (0, 1.05), "bar_sparsity_index.png"),
        ("mean_grid_score", "Grid Score", "Mean Grid Score (aux)", None, "bar_mean_grid_score_aux.png"),
    ]:
        if any(f"{metric}_mean" in summary_main[name] for name in summary_main):
            plot_condition_bars(summary_main, metric, ylabel, title, output_dir / filename, ylim)

    if any("position_mse_mean" in summary_main[name] for name in summary_main):
        plot_condition_bars(
            summary_main,
            "position_mse",
            "Position MSE",
            "Position MSE (squared Euclidean error)",
            output_dir / "bar_position_mse.png",
            None,
        )

    if any("position_l2_error_mean" in summary_main[name] for name in summary_main):
        plot_condition_bars(
            summary_main,
            "position_l2_error",
            "Position L2 Error",
            "Position L2 Error",
            output_dir / "bar_position_l2_error.png",
            None,
        )

    if controls_cfg.get("run_checkpoint_compare", False):
        checkpoint_specs: Dict[str, dict] = {}
        for checkpoint_path in cfg.get("transfer", {}).get("checkpoint_candidates", []):
            if Path(checkpoint_path).exists():
                checkpoint_specs[Path(checkpoint_path).stem] = {
                    "transfer": True,
                    "freeze": False,
                    "freeze_mode": "all",
                    "checkpoint_path": checkpoint_path,
                    "transfer_mode": transfer_mode,
                    "action_map_mode": base_action_map_mode,
                }
            else:
                logger.warning("Skipping missing checkpoint candidate: %s", checkpoint_path)
        if checkpoint_specs:
            histories, evals, raw_block = run_block(
                block_name="checkpoint_control",
                condition_specs=checkpoint_specs,
                cfg=cfg,
                n_seeds=n_seeds,
                device=device,
                logger=logger,
                output_dir=output_dir,
                main_block=False,
            )
            summary_checkpoint = aggregate_block(histories, evals)
            save_json(summary_checkpoint, output_dir / "summary_checkpoint_control.json")
            raw_per_seed["checkpoint_control"] = raw_block
        else:
            logger.warning("Checkpoint control skipped because no candidate checkpoint is available.")

    if controls_cfg.get("run_action_mapping_control", False):
        if primary_ckpt and Path(primary_ckpt).exists():
            mapping_specs = {
                "aligned": {
                    "transfer": True,
                    "freeze": False,
                    "freeze_mode": "all",
                    "checkpoint_path": primary_ckpt,
                    "transfer_mode": transfer_mode,
                    "action_map_mode": "aligned",
                },
                "shuffled": {
                    "transfer": True,
                    "freeze": False,
                    "freeze_mode": "all",
                    "checkpoint_path": primary_ckpt,
                    "transfer_mode": transfer_mode,
                    "action_map_mode": "shuffled",
                },
                # stronger negative control: break the velocity->target
                # pairing across episodes (derangement). Immune to displacement
                # commutativity, unlike within-episode order permutation, and
                # not absorbable by lattice symmetry, unlike action-label
                # `shuffled`. If velocities truly drive the target prediction,
                # accuracy must collapse here.
                "mismatched_target": {
                    "transfer": True,
                    "freeze": False,
                    "freeze_mode": "all",
                    "checkpoint_path": primary_ckpt,
                    "transfer_mode": transfer_mode,
                    "action_map_mode": "mismatched_target",
                },
            }
            histories, evals, raw_block = run_block(
                block_name="mapping_control",
                condition_specs=mapping_specs,
                cfg=cfg,
                n_seeds=n_seeds,
                device=device,
                logger=logger,
                output_dir=output_dir,
                main_block=False,
            )
            summary_mapping = aggregate_block(histories, evals)
            save_json(summary_mapping, output_dir / "summary_mapping_control.json")
            raw_per_seed["mapping_control"] = raw_block
        else:
            logger.warning("Mapping control skipped because primary transfer checkpoint is unavailable.")

    # Generalization-split control: compare transfer vs random_init under
    # compositional / held-out splits. Saturated random-split accuracy hides
    # whether the prior helps; these splits make the comparison meaningful.
    if controls_cfg.get("run_generalization_split_control", False):
        if primary_ckpt and Path(primary_ckpt).exists():
            split_specs = {}
            for strat in ("held_out_pairs", "held_out_start_nodes", "held_out_target_nodes"):
                split_specs[f"transfer_{strat}"] = {
                    "transfer": True, "freeze": True, "freeze_mode": "all",
                    "checkpoint_path": primary_ckpt, "transfer_mode": transfer_mode,
                    "action_map_mode": base_action_map_mode, "split_strategy": strat,
                }
                split_specs[f"random_{strat}"] = {
                    "transfer": False, "freeze": False,
                    "action_map_mode": base_action_map_mode, "split_strategy": strat,
                }
                # Complete the 2x2 design over prior source and recurrent-core
                # trainability. This avoids attributing a freezing effect to the
                # navigation prior.
                split_specs[f"random_frozen_{strat}"] = {
                    "transfer": False, "freeze": True, "freeze_mode": "all",
                    "action_map_mode": base_action_map_mode, "split_strategy": strat,
                }
                split_specs[f"transfer_finetune_{strat}"] = {
                    "transfer": True, "freeze": False,
                    "checkpoint_path": primary_ckpt, "transfer_mode": transfer_mode,
                    "action_map_mode": base_action_map_mode, "split_strategy": strat,
                }
            histories, evals, raw_block = run_block(
                block_name="generalization_split_control",
                condition_specs=split_specs,
                cfg=cfg,
                n_seeds=n_seeds,
                device=device,
                logger=logger,
                output_dir=output_dir,
                main_block=False,
            )
            summary_split = aggregate_block(histories, evals)
            save_json(summary_split, output_dir / "summary_generalization_split_control.json")
            raw_per_seed["generalization_split_control"] = raw_block
        else:
            logger.warning("Generalization-split control skipped: primary transfer checkpoint unavailable.")

    # Held-out PATH-LENGTH control: train on short paths, test on longer
    # ones (variable-length episodes). Tests length extrapolation of the
    # integrator. seq_len_range default widens the fixed seq_len if unset.
    if controls_cfg.get("run_held_out_length_control", False):
        if primary_ckpt and Path(primary_ckpt).exists():
            base_seq = cfg.get("data", {}).get("seq_len", 8)
            length_range = controls_cfg.get("held_out_length_range", [max(2, base_seq // 2), base_seq * 2])
            len_specs = {
                "transfer_held_out_length": {
                    "transfer": True, "freeze": True, "freeze_mode": "all",
                    "checkpoint_path": primary_ckpt, "transfer_mode": transfer_mode,
                    "action_map_mode": base_action_map_mode,
                    "split_strategy": "held_out_length",
                    "seq_len_range": tuple(length_range),
                },
                "random_held_out_length": {
                    "transfer": False, "freeze": False,
                    "action_map_mode": base_action_map_mode,
                    "split_strategy": "held_out_length",
                    "seq_len_range": tuple(length_range),
                },
            }
            histories, evals, raw_block = run_block(
                block_name="held_out_length_control",
                condition_specs=len_specs,
                cfg=cfg,
                n_seeds=n_seeds,
                device=device,
                logger=logger,
                output_dir=output_dir,
                main_block=False,
            )
            summary_len = aggregate_block(histories, evals)
            save_json(summary_len, output_dir / "summary_held_out_length_control.json")
            raw_per_seed["held_out_length_control"] = raw_block
        else:
            logger.warning("Held-out-length control skipped: primary transfer checkpoint unavailable.")

    # Cross-graph generalization: train on the base graph, evaluate on
    # foreign graphs (different SIZE and different TOPOLOGY) via coordinate
    # readout. Tests whether the integration dynamics generalize across graph
    # structure, independent of the (graph-specific) node vocabulary.
    if controls_cfg.get("run_cross_graph_control", False):
        if primary_ckpt and Path(primary_ckpt).exists():
            base_n_levels = cfg.get("data", {}).get("n_levels", 4)
            coord_start = cfg.get("model", {}).get("start_init", "embedding") == "coordinate"
            # With start_init='coordinate' the model is node-count-
            # independent, so we can test LARGER graphs too. With the default
            # embedding start, foreign graphs must have <= base nodes (the
            # bounds guard fails loud otherwise), so we only test a SMALLER
            # graph + a different topology.
            if coord_start:
                bigger = base_n_levels + 2
                cross_targets = controls_cfg.get("cross_graph_targets", [
                    {"task_name": "social_hierarchy", "n_levels": bigger,
                     "label": f"social_{bigger}x{bigger}_LARGER"},
                    {"task_name": "transitive", "n_items": 7, "label": "transitive_chain"},
                ])
            else:
                smaller_levels = max(2, base_n_levels - 1)
                cross_targets = controls_cfg.get("cross_graph_targets", [
                    {"task_name": "social_hierarchy", "n_levels": smaller_levels,
                     "label": f"social_{smaller_levels}x{smaller_levels}"},
                    {"task_name": "transitive", "n_items": min(7, base_n_levels * base_n_levels),
                     "label": "transitive_chain"},
                ])
            cg_specs = {
                "transfer_cross_graph": {
                    "transfer": True, "freeze": True, "freeze_mode": "all",
                    "checkpoint_path": primary_ckpt, "transfer_mode": transfer_mode,
                    "action_map_mode": base_action_map_mode,
                    "cross_graph_eval": cross_targets,
                    "cross_graph_refit": cross_targets,
                },
                "random_cross_graph": {
                    "transfer": False, "freeze": False,
                    "action_map_mode": base_action_map_mode,
                    "cross_graph_eval": cross_targets,
                    "cross_graph_refit": cross_targets,
                },
            }
            histories, evals, raw_block = run_block(
                block_name="cross_graph_control",
                condition_specs=cg_specs,
                cfg=cfg,
                n_seeds=n_seeds,
                device=device,
                logger=logger,
                output_dir=output_dir,
                main_block=False,
            )
            summary_cg = aggregate_block(histories, evals)
            save_json(summary_cg, output_dir / "summary_cross_graph_control.json")
            raw_per_seed["cross_graph_control"] = raw_block
        else:
            logger.warning("Cross-graph control skipped: primary transfer checkpoint unavailable.")

    # Velocity-transform control: a frozen-transfer model whose readout was
    # trained on veridical velocities should degrade under rotation/scale/sign.
    if controls_cfg.get("run_velocity_transform_control", False):
        if primary_ckpt and Path(primary_ckpt).exists():
            # Train on veridical velocities and perturb only the test stream. A
            # transform applied to both train and test would merely relabel the
            # input space instead of probing out-of-distribution equivariance.
            vt_specs = {}
            for vt in ("none", "rotate90", "scale2x", "signflip"):
                vt_specs[f"transfer_{vt}"] = {
                    "transfer": True, "freeze": True, "freeze_mode": "all",
                    "checkpoint_path": primary_ckpt, "transfer_mode": transfer_mode,
                    "action_map_mode": base_action_map_mode,
                    "test_only_velocity_transform": vt,
                }
                # Matched frozen-random arm: if BOTH degrade identically under
                # rotation, the sensitivity is a property of the task/readout
                # rather than of the transferred navigation dynamics.
                vt_specs[f"random_frozen_{vt}"] = {
                    "transfer": False, "freeze": True, "freeze_mode": "all",
                    "action_map_mode": base_action_map_mode,
                    "test_only_velocity_transform": vt,
                }
            histories, evals, raw_block = run_block(
                block_name="velocity_transform_control",
                condition_specs=vt_specs,
                cfg=cfg,
                n_seeds=n_seeds,
                device=device,
                logger=logger,
                output_dir=output_dir,
                main_block=False,
            )
            summary_vt = aggregate_block(histories, evals)
            save_json(summary_vt, output_dir / "summary_velocity_transform_control.json")
            raw_per_seed["velocity_transform_control"] = raw_block
        else:
            logger.warning("Velocity-transform control skipped: primary transfer checkpoint unavailable.")

    # No-coordinate-supervision ablation: if geometry_corr stays high only
    # because the position head / RHC target inject coordinates into the loss,
    # then it is not evidence of a spontaneously-formed cognitive map. Compare
    # geometry with coord supervision ON vs OFF (position_loss_weight forced 0).
    if controls_cfg.get("run_no_coord_supervision_control", False):
        coord_specs = {
            "random_coord_sup_off": {
                "transfer": False,
                "freeze": False,
                "action_map_mode": base_action_map_mode,
                "position_loss_weight": 0.0,
                "rhc_alignment_weight": 0.0,
            },
        }
        if primary_ckpt and Path(primary_ckpt).exists():
            coord_specs = {
                "transfer_coord_sup_on": {
                    "transfer": True,
                    "freeze": True,
                    "freeze_mode": "all",
                    "checkpoint_path": primary_ckpt,
                    "transfer_mode": transfer_mode,
                    "action_map_mode": base_action_map_mode,
                    "position_loss_weight": cfg["train"].get("position_loss_weight", 0.0),
                },
                "transfer_coord_sup_off": {
                    "transfer": True,
                    "freeze": True,
                    "freeze_mode": "all",
                    "checkpoint_path": primary_ckpt,
                    "transfer_mode": transfer_mode,
                    "action_map_mode": base_action_map_mode,
                    "position_loss_weight": 0.0,
                    "rhc_alignment_weight": 0.0,
                },
                **coord_specs,
            }
        else:
            logger.warning(
                "No-coordinate control: transfer arms skipped because the checkpoint "
                "is unavailable; only the explicitly named random arm will run."
            )
        histories, evals, raw_block = run_block(
            block_name="no_coord_supervision_control",
            condition_specs=coord_specs,
            cfg=cfg,
            n_seeds=n_seeds,
            device=device,
            logger=logger,
            output_dir=output_dir,
            main_block=False,
        )
        summary_coord = aggregate_block(histories, evals)
        save_json(summary_coord, output_dir / "summary_no_coord_supervision_control.json")
        raw_per_seed["no_coord_supervision_control"] = raw_block

    if controls_cfg.get("run_transfer_mode_control", False):
        if primary_ckpt and Path(primary_ckpt).exists():
            transfer_mode_specs = {
                "full_transfer": {
                    "transfer": True,
                    "freeze": False,
                    "freeze_mode": "all",
                    "checkpoint_path": primary_ckpt,
                    "transfer_mode": "full",
                    "action_map_mode": base_action_map_mode,
                },
                "recurrent_only_transfer": {
                    "transfer": True,
                    "freeze": False,
                    "freeze_mode": "all",
                    "checkpoint_path": primary_ckpt,
                    "transfer_mode": "recurrent_only",
                    "action_map_mode": base_action_map_mode,
                },
                # Complete the transfer-source decomposition: full copies input
                # and recurrent projections, while the two partial conditions
                # isolate each projection against a matched no-transfer baseline.
                "input_only_transfer": {
                    "transfer": True,
                    "freeze": False,
                    "freeze_mode": "all",
                    "checkpoint_path": primary_ckpt,
                    "transfer_mode": "input_only",
                    "action_map_mode": base_action_map_mode,
                },
                "no_transfer_baseline": {
                    "transfer": False,
                    "freeze": False,
                    "freeze_mode": "all",
                    "action_map_mode": base_action_map_mode,
                },
            }
            histories, evals, raw_block = run_block(
                block_name="transfer_mode_control",
                condition_specs=transfer_mode_specs,
                cfg=cfg,
                n_seeds=n_seeds,
                device=device,
                logger=logger,
                output_dir=output_dir,
                main_block=False,
            )
            summary_transfer_mode = aggregate_block(histories, evals)
            save_json(summary_transfer_mode, output_dir / "summary_transfer_mode_control.json")
            raw_per_seed["transfer_mode_control"] = raw_block
        else:
            logger.warning("Transfer-mode control skipped because primary transfer checkpoint is unavailable.")

    save_json(raw_per_seed, output_dir / "raw_per_seed.json")

    logger.info("\n" + "=" * 80)
    logger.info("EXPERIMENT 5 — ABSTRACT PATH-INTEGRATION SUMMARY (MAIN BLOCK)")
    logger.info("=" * 80)
    header = (
        f"{'Condition':<22s} {'Acc':>7s} {'GeomCorr':>9s} "
        f"{'GridMean':>9s} {'Grid>0.3':>9s} {'Sparse':>8s} {'SampleEff':>10s}"
    )
    logger.info(header)
    logger.info("-" * 80)
    for cond_name in cond_names:
        if cond_name not in summary_main:
            continue
        entry = summary_main[cond_name]
        sample_eff = entry["sample_efficiency_mean"]
        sample_eff_str = f"{sample_eff:.0f}" if sample_eff > 0 else "N/A"
        grid_mean = entry.get("mean_grid_score_mean", None)
        grid_frac = entry.get("frac_grid_above_03_mean", None)
        grid_mean_str = f"{float(grid_mean):.4f}" if grid_mean is not None else "N/A"
        grid_frac_str = f"{float(grid_frac) * 100:.2f}%" if grid_frac is not None else "N/A"
        logger.info(
            "%-22s %6.3f  %8.4f  %8s  %8s  %7.4f  %10s",
            cond_name,
            entry.get("test_acc_mean", 0.0),
            entry.get("geometry_corr_mean", 0.0),
            grid_mean_str,
            grid_frac_str,
            entry.get("sparsity_index_mean", 0.0),
            sample_eff_str,
        )
    logger.info("=" * 80)
    model_checkpoints = sorted(
        set(output_dir.rglob("*_model.pt")) | set(output_dir.rglob("best_validation.pt")),
        key=lambda path: path.as_posix(),
    )
    save_json(
        {
            "experiment": "exp5_abstract_pathint",
            "selection_split": "validation",
            "test_evaluations_per_condition": 1,
            "summary": "summary.json",
            "raw_per_seed": "raw_per_seed.json",
            "model_checkpoints": [
                {"path": str(path.relative_to(output_dir)), "sha256": sha256_file(path)}
                for path in model_checkpoints
            ],
        },
        output_dir / "artifact_manifest.json",
    )
    logger.info("Results saved to %s", output_dir)


if __name__ == "__main__":
    main()
