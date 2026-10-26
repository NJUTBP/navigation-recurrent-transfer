"""Exp7: 2x2 linear-adapter rescue of coordinate-transformed transfer.

Motivation (from the current negative result): with the navigation core
frozen and relations mapped veridically, 4x4 endpoint accuracy is ~0.98, but
rotate90 -> ~0.15, signflip -> ~0.13, scale2x -> ~0.68. Rotation and sign
reversal do NOT change the relational algebra of the lattice — they only
change the coordinate frame. If the transferred computation is a
coordinate-equivalent 2-D integrator, a tiny learned remap A should invert T.

Design: apply a fixed transform T to ALL target relations (train + val +
test, implemented inside the model wrapper), keep the navigation core fully
frozen, and train only (adapter A, start encoder, endpoint readout).

PRE-REGISTERED pass criteria (fixed before running; do not tune after):
  C1  nav_adapter test acc >= recover_fraction (default 0.90) x its own
      T=identity baseline, per target seed (paired), for every orthogonal or
      pure-scale T within scale_range.
  C2  rel_identity_error(A.T) <= 0.15 for those T.
  C3  random_adapter does not exceed its own T=identity baseline + 0.05
      (the adapter cannot substitute for the navigation core).
  C4  nav_noadapter (A frozen at I, readout retrained under T) does NOT
      recover (< 0.5 x identity baseline for orthogonal T), ruling out
      "any retraining fixes it".
Positive claim if C1-C4 hold:
  "Misalignment disrupts direct transfer, but a low-dimensional remapping
   restores it" — the navigation prior is coordinate-equivalent, not tied to
   one absolute sensorimotor frame.

Supports multiple navigation source checkpoints (``nav_checkpoints`` list) so
the same run doubles as a source-level replication once more sources exist.

Usage:
    python -m gpm_cmap.experiments.exp7_adapter_rescue \
        --config configs/adapter_rescue.yaml
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from gpm_cmap.data.graph_tasks.pathint_episodes import build_pathint_loaders
from gpm_cmap.models.abstract_pathint import AbstractPathIntModel
from gpm_cmap.models.adapter_wrap import (
    AdapterPathIntModel,
    make_transform_matrix,
    transform_label,
)
from gpm_cmap.tasks.abstract_pathint import (
    evaluate_pathint_basic,
    train_pathint_epoch,
)
from gpm_cmap.utils import (
    load_config,
    prepare_output_dir,
    save_config_snapshot,
    set_seed,
)
from gpm_cmap.utils.runtime import configure_runtime, save_run_provenance
from gpm_cmap.utils.logging_utils import get_logger


def build_base_model(cfg: dict, n_nodes: int, device: torch.device) -> AbstractPathIntModel:
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
        use_position_head=mcfg.get("use_position_head", False),
        start_init=mcfg.get("start_init", "embedding"),
        coord_dim=mcfg.get("coord_dim", 2),
    )
    return model.to(device)


def build_loaders(cfg: dict, seed: int):
    dcfg = cfg["data"]
    if float(dcfg.get("validation_ratio", 0.0)) <= 0.0:
        raise ValueError("adapter_rescue requires data.validation_ratio > 0")
    return build_pathint_loaders(
        task_name=dcfg.get("task_name", "social_hierarchy"),
        num_episodes=dcfg["num_episodes"],
        seq_len=dcfg["seq_len"],
        batch_size=dcfg["batch_size"],
        split_ratio=dcfg.get("split_ratio", 0.7),
        velocity_scale=dcfg["velocity_scale"],
        seed=seed,
        n_levels=dcfg.get("n_levels", 4),
        n_items=dcfg.get("n_items", 7),
        action_map_mode="aligned",
        coord_strategy=dcfg.get("coord_strategy", None),
        coord_dim=dcfg.get("coord_dim", 2),
        normalize_displacements=dcfg.get("normalize_displacements", True),
        coord_scale=dcfg.get("coord_scale", 1.0),
        split_strategy=dcfg.get("split_strategy", "random"),
        velocity_transform="none",  # the transform lives inside the wrapper
        validation_ratio=dcfg.get("validation_ratio", 0.1),
        return_validation=True,
        num_workers=dcfg.get("num_workers", 0),
    )


def run_one(
    arm: str,
    source_ckpt: Optional[str],
    transform_spec,
    seed: int,
    cfg: dict,
    device: torch.device,
    logger,
) -> Dict[str, object]:
    """Train one (arm, transform, seed) cell and return a result row."""
    tcfg = cfg["train"]
    set_seed(seed)
    # Rebuild loaders inside the condition (after set_seed) so every arm of a
    # given seed sees byte-identical episode splits — required for the paired
    # C1/C3/C4 comparisons.
    (
        train_loader, validation_loader, test_loader,
        _task, node_to_id, _action_to_id, _node_coords,
    ) = build_loaders(cfg, seed)
    n_nodes = len(node_to_id)

    base = build_base_model(cfg, n_nodes, device)
    if source_ckpt is not None:
        counts = base.load_nav_checkpoint(
            source_ckpt, transfer_mode=cfg.get("transfer", {}).get("mode", "full")
        )
        if counts["copied"] == 0:
            raise RuntimeError(
                f"0 parameters copied from {source_ckpt!r}; check hidden_dim/cell_type."
            )

    T = make_transform_matrix(transform_spec)
    model = AdapterPathIntModel(base, in_dim=2, transform=T, adapter_init="identity")
    model.configure_trainable()
    if arm.endswith("noadapter"):
        model.freeze_adapter_at_identity()
    model.to(device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    optimizer = torch.optim.Adam(
        trainable, lr=tcfg["lr"], weight_decay=tcfg.get("weight_decay", 0.0)
    )

    best_val = -float("inf")
    best_state: Optional[dict] = None
    history: List[dict] = []
    eval_every = int(tcfg.get("eval_every", 5))
    for epoch in range(1, tcfg["epochs"] + 1):
        train_stats = train_pathint_epoch(
            model,
            train_loader,
            optimizer,
            device,
            grad_clip=tcfg.get("grad_clip", 1.0),
            whh_weight_decay=0.0,
            position_loss_weight=0.0,  # no coordinate supervision anywhere
            rhc_alignment_weight=0.0,
        )
        entry = {"epoch": epoch, "train_acc": train_stats["accuracy"]}
        if epoch % eval_every == 0 or epoch == tcfg["epochs"]:
            val_stats = evaluate_pathint_basic(model, validation_loader, device)
            entry["val_acc"] = val_stats["accuracy"]
            if val_stats["accuracy"] > best_val:
                best_val = val_stats["accuracy"]
                best_state = copy.deepcopy(model.state_dict())
        history.append(entry)

    if best_state is not None:
        model.load_state_dict(best_state)
    test_stats = evaluate_pathint_basic(model, test_loader, device)
    align = model.alignment_report()
    row: Dict[str, object] = {
        "arm": arm,
        "source": source_ckpt or "random",
        "transform": transform_label(transform_spec),
        "seed": seed,
        "n_trainable_params": int(n_trainable),
        "test_accuracy": float(test_stats["accuracy"]),
        "best_val_accuracy": float(best_val),
        **{k: align[k] for k in (
            "identity_error_fro",
            "rel_identity_error",
            "polar_angle_deg",
            "polar_det_sign",
            "rel_error_vs_T_inv",
        )},
        "adapter": align["adapter"],
        "history": history,
    }
    logger.info(
        "  [%s | %s | seed %d] test=%.4f relErr(AT,I)=%.3f angle=%.1f",
        arm,
        row["transform"],
        seed,
        row["test_accuracy"],
        row["rel_identity_error"],
        row["polar_angle_deg"],
    )
    return row


def aggregate(rows: List[Dict[str, object]], recover_fraction: float) -> Dict[str, object]:
    """Cell means/sds plus the pre-registered pass criteria per transform."""
    def cell(arm: str, transform: str, source: str) -> List[Dict[str, object]]:
        return [
            r for r in rows
            if r["arm"] == arm and r["transform"] == transform and r["source"] == source
        ]

    sources = sorted({r["source"] for r in rows if r["source"] != "random"})
    transforms = sorted({r["transform"] for r in rows})
    summary: Dict[str, object] = {"per_source": {}, "criteria": {}}
    for src in sources:
        src_summary: Dict[str, object] = {}
        base_rows = cell("nav_adapter", "none", src)
        base_by_seed = {r["seed"]: r["test_accuracy"] for r in base_rows}
        rnd_base = [r["test_accuracy"] for r in cell("random_adapter", "none", "random")]
        rnd_base_mean = float(np.mean(rnd_base)) if rnd_base else float("nan")
        for tf in transforms:
            nav = cell("nav_adapter", tf, src)
            noad = cell("nav_noadapter", tf, src)
            rnd = cell("random_adapter", tf, "random")
            paired_recover = [
                r["test_accuracy"] >= recover_fraction * base_by_seed.get(r["seed"], np.nan)
                for r in nav
                if r["seed"] in base_by_seed
            ]
            src_summary[tf] = {
                "nav_adapter_acc_mean": float(np.mean([r["test_accuracy"] for r in nav])) if nav else None,
                "nav_adapter_acc_sd": float(np.std([r["test_accuracy"] for r in nav])) if nav else None,
                "nav_noadapter_acc_mean": float(np.mean([r["test_accuracy"] for r in noad])) if noad else None,
                "random_adapter_acc_mean": float(np.mean([r["test_accuracy"] for r in rnd])) if rnd else None,
                "rel_identity_error_mean": float(np.mean([r["rel_identity_error"] for r in nav])) if nav else None,
                "C1_recovered_fraction_of_seeds": float(np.mean(paired_recover)) if paired_recover else None,
                "C2_identity_error_ok": bool(
                    nav and np.mean([r["rel_identity_error"] for r in nav]) <= 0.15
                ),
                "C3_random_not_rescued": bool(
                    (not rnd) or np.mean([r["test_accuracy"] for r in rnd]) <= rnd_base_mean + 0.05
                ),
                "C4_readout_alone_fails": None if not noad else bool(
                    np.mean([r["test_accuracy"] for r in noad])
                    < 0.5 * float(np.mean(list(base_by_seed.values()) or [np.nan]))
                ),
            }
        summary["per_source"][src] = src_summary
    return summary


def write_csv(rows: List[Dict[str, object]], path: Path) -> None:
    keys = [
        "arm", "source", "transform", "seed", "n_trainable_params",
        "test_accuracy", "best_val_accuracy", "identity_error_fro",
        "rel_identity_error", "polar_angle_deg", "polar_det_sign",
        "rel_error_vs_T_inv",
    ]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def plot_summary(rows: List[Dict[str, object]], output_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    # Rotation sweep: accuracy vs angle, with vs without adapter.
    def angle_of(label: str) -> Optional[float]:
        if label == "none":
            return 0.0
        if label == "signflip":
            return 180.0
        if label.startswith("rot"):
            try:
                return float(label[3:])
            except ValueError:
                return None
        return None

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for arm, color, marker in (
        ("nav_adapter", "#1f77b4", "o"),
        ("nav_noadapter", "#d62728", "s"),
        ("random_adapter", "#7f7f7f", "^"),
    ):
        pts: Dict[float, List[float]] = {}
        for r in rows:
            a = angle_of(str(r["transform"]))
            if a is None or r["arm"] != arm:
                continue
            pts.setdefault(a, []).append(float(r["test_accuracy"]))
        if not pts:
            continue
        xs = sorted(pts)
        means = [float(np.mean(pts[x])) for x in xs]
        sds = [float(np.std(pts[x])) for x in xs]
        ax.errorbar(xs, means, yerr=sds, label=arm, color=color, marker=marker, capsize=3)
    ax.set_xlabel("rotation of the relation frame (deg)")
    ax.set_ylabel("test accuracy")
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "rotation_sweep.png", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp7: 2x2 adapter rescue")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--n_seeds", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.output_dir is not None:
        cfg["output_dir"] = args.output_dir
    n_seeds = args.n_seeds if args.n_seeds is not None else cfg.get("n_seeds", 5)
    device = configure_runtime(cfg)
    output_dir = prepare_output_dir(cfg["output_dir"])
    logger = get_logger("exp7_adapter_rescue", output_dir / "experiment.log")
    save_config_snapshot(cfg, output_dir)
    save_run_provenance(output_dir, "exp7_adapter_rescue", cfg=cfg)

    nav_ckpts: List[str] = list(cfg.get("nav_checkpoints", []))
    if not nav_ckpts and cfg.get("transfer_checkpoint"):
        nav_ckpts = [cfg["transfer_checkpoint"]]
    nav_ckpts = [c for c in nav_ckpts if Path(c).exists()]
    if not nav_ckpts:
        raise FileNotFoundError("No existing navigation checkpoint in nav_checkpoints.")
    logger.info("Navigation sources: %s", nav_ckpts)

    transform_specs = cfg.get("transforms") or [
        "none", "signflip", {"name": "rotate", "degrees": 90},
        {"name": "rotate", "degrees": 30}, {"name": "rotate", "degrees": 135},
        {"name": "rotate", "degrees": 270}, {"name": "scale", "factor": 0.5},
        "scale2x", {"name": "scale", "factor": 4.0},
        {"name": "aniso", "fx": 2.0, "fy": 0.5}, {"name": "shear", "k": 0.5},
    ]
    recover_fraction = float(cfg.get("recover_fraction", 0.90))
    base_seed = int(cfg.get("seed", 42))

    rows: List[Dict[str, object]] = []
    for seed_idx in range(n_seeds):
        seed = base_seed + seed_idx
        for spec in transform_specs:
            label = transform_label(spec)
            for src_idx, ckpt in enumerate(nav_ckpts):
                logger.info(
                    "=== seed %d | transform %s | source %d/%d ===",
                    seed, label, src_idx + 1, len(nav_ckpts),
                )
                rows.append(run_one(
                    "nav_adapter", ckpt, spec, seed, cfg, device, logger,
                ))
                if cfg.get("run_noadapter_control", True) and label != "none":
                    rows.append(run_one(
                        "nav_noadapter", ckpt, spec, seed, cfg, device, logger,
                    ))
            if cfg.get("run_random_control", True):
                rows.append(run_one(
                    "random_adapter", None, spec, seed, cfg, device, logger,
                ))
        # persist incrementally so a crash keeps completed cells
        write_csv(rows, output_dir / "adapter_rescue_rows.csv")
        with (output_dir / "adapter_rescue_rows.json").open("w") as fh:
            json.dump(rows, fh, indent=2)

    summary = aggregate(rows, recover_fraction)
    summary["recover_fraction"] = recover_fraction
    with (output_dir / "summary_adapter_rescue.json").open("w") as fh:
        json.dump(summary, fh, indent=2)
    plot_summary(rows, output_dir)
    logger.info("Done. Summary: %s", output_dir / "summary_adapter_rescue.json")


if __name__ == "__main__":
    main()
