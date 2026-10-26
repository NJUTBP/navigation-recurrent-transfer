"""Exp8: symbolic (one-hot) relations + learned 2xK adapter into a frozen core.

The model sees arbitrary one-hot relation symbols rather than a supplied 2-D
relation-to-displacement map. A trainable linear adapter A (2 x K, no bias)
learns how to wire those symbols into a frozen recurrent system. This removes
a hard-coded input coordinate map, but it does *not* by itself establish that
recurrent navigation dynamics are required: in the current lattice generator,
the endpoint is a deterministic function of start identity and unordered
action counts. The simple controls below therefore form part of the primary
interpretation, not an optional afterthought.

Arms (all with identical adapter/start-encoder/readout parameter budgets):
  nav_frozen      frozen navigation core
  random_frozen   frozen matched-random core (same init distribution)
  scratch_full    fully trainable random model (upper bound; many more
                  trainable params — labelled as such, optional)

Optional task-characterisation controls:
  baseline_action_count_linear
      multinomial linear classifier on start identity plus unordered action
      counts; tests whether temporal order/recurrent state is unnecessary.
  baseline_linear_accumulator
      learned 2-D start and action vectors summed without a recurrent core;
      tests whether the adapter geometry is imposed by the task itself.
  post-hoc W_hh=0 ablation
      evaluates a trained adapter model after removing recurrence, while
      restoring the original weight immediately afterwards.

Historical S1/S2 criteria (retained for reproducibility):
  S1  Sample efficiency: at every training budget <= low_budget_max (default
      200 episodes), nav_frozen > random_frozen in paired mean test accuracy,
      and the log-budget AUC difference is positive across seeds.
  S2  Learned geometry (nav_frozen, full budget): the K adapter columns form
      two near-antipodal pairs (antipodality index <= 0.25) whose axes are
      near-orthogonal (90 +/- 20 deg), and align with the true displacement
      codebook up to one global rotation+scale (Procrustes residual <= 0.30).

S1+S2 are no longer treated as sufficient for a navigation-prior claim. A
source-specific recurrent interpretation additionally requires navigation to
outperform CTL-01 and the non-recurrent controls, plus a material loss when
recurrence is removed. Otherwise the experiment characterises learnability of
a symbolic additive endpoint task.

Usage:
    python -m gpm_cmap.experiments.exp8_symbolic_adapter \
        --config configs/symbolic_adapter.yaml
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from gpm_cmap.data.graph_tasks.pathint_episodes import (
    build_pathint_loaders,
    split_manifest,
)
from gpm_cmap.models.abstract_pathint import AbstractPathIntModel
from gpm_cmap.models.adapter_wrap import AdapterPathIntModel
from gpm_cmap.models.symbolic_controls import (
    ActionCountLinearBaseline,
    LinearAccumulatorBaseline,
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
    write_json,
)
from gpm_cmap.utils.runtime import configure_runtime, save_run_provenance
from gpm_cmap.utils.logging_utils import get_logger
from gpm_cmap.utils.checkpointing import load_checkpoint, sha256_file
from gpm_cmap.utils.seed import make_torch_generator, make_worker_init_fn


BASELINE_ARMS = {
    "baseline_action_count_linear",
    "baseline_linear_accumulator",
}


# ---------------------------------------------------------------------------
# One-hot symbolic view of an existing path-integration dataset
# ---------------------------------------------------------------------------


def build_action_codebook(dataset, decimals: int = 6) -> np.ndarray:
    """Unique per-step displacement vectors of a dataset -> [K,2] codebook."""
    seen: Dict[Tuple[float, float], np.ndarray] = {}
    for sample in dataset:
        vel = sample["velocities"].numpy()
        for step in vel:
            key = tuple(np.round(step, decimals).tolist())
            if key not in seen:
                seen[key] = step.astype(np.float32)
    codes = np.stack([seen[k] for k in sorted(seen.keys())], axis=0)
    if codes.shape[0] > 8:
        raise ValueError(
            f"Expected a small discrete action set; found {codes.shape[0]} unique "
            "step vectors. Is normalize_displacements on and seq_len fixed?"
        )
    return codes


def steps_to_ids(velocities: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    """Map [T,2] step vectors to nearest codebook ids [T]."""
    d = np.linalg.norm(velocities[:, None, :] - codebook[None, :, :], axis=-1)
    ids = d.argmin(axis=1)
    if not np.allclose(d[np.arange(len(ids)), ids], 0.0, atol=1e-4):
        raise ValueError("A step vector did not match the action codebook exactly.")
    return ids.astype(np.int64)


class OneHotActionDataset(Dataset):
    """Wraps a PathIntDataset, replacing 2-D velocities with one-hot codes."""

    def __init__(self, inner, codebook: np.ndarray) -> None:
        self.inner = inner
        self.codebook = codebook
        self.n_actions = int(codebook.shape[0])
        self.prevalidated = bool(getattr(inner, "prevalidated", False))
        self.min_node_id = int(getattr(inner, "min_node_id", 0))
        self.max_node_id = int(getattr(inner, "max_node_id", -1))

    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = dict(self.inner[idx])
        vel = sample["velocities"].numpy()
        ids = steps_to_ids(vel, self.codebook)
        onehot = np.zeros((vel.shape[0], self.n_actions), dtype=np.float32)
        onehot[np.arange(vel.shape[0]), ids] = 1.0
        sample["velocities"] = torch.from_numpy(onehot)
        return sample


def wrap_loader(inner_loader: DataLoader, codebook: np.ndarray, seed: int,
                shuffle: bool, batch_size: int, subset: Optional[int] = None) -> DataLoader:
    ds: Dataset = OneHotActionDataset(inner_loader.dataset, codebook)
    if subset is not None:
        subset = min(subset, len(ds))
        ds = Subset(ds, list(range(subset)))
    kwargs = dict(
        batch_size=batch_size,
        worker_init_fn=make_worker_init_fn(seed),
        collate_fn=inner_loader.collate_fn,
    )
    if shuffle:
        return DataLoader(ds, shuffle=True, generator=make_torch_generator(seed), **kwargs)
    return DataLoader(ds, shuffle=False, **kwargs)


# ---------------------------------------------------------------------------
# Adapter-geometry diagnostics
# ---------------------------------------------------------------------------


def adapter_geometry_report(adapter_cols: np.ndarray, codebook: np.ndarray) -> Dict[str, object]:
    """adapter_cols: [2,K] learned relation vectors; codebook: [K,2] truth."""
    K = adapter_cols.shape[1]
    cols = adapter_cols.astype(np.float64)
    norms = np.linalg.norm(cols, axis=0)
    mean_norm = float(np.mean(norms)) if np.all(norms > 0) else float("nan")

    report: Dict[str, object] = {
        "column_norms": norms.tolist(),
        "cosine_matrix": (
            (cols.T @ cols) / np.maximum(np.outer(norms, norms), 1e-12)
        ).tolist(),
    }

    if K == 4 and mean_norm > 0:
        best = None
        for pairing in (((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))):
            resid = sum(
                float(np.linalg.norm(cols[:, i] + cols[:, j])) for i, j in pairing
            )
            if best is None or resid < best[0]:
                best = (resid, pairing)
        resid, pairing = best
        report["antipodality_index"] = float(resid / (2.0 * 2.0 * mean_norm))
        axes = []
        for i, j in pairing:
            axis = cols[:, i] - cols[:, j]
            axes.append(axis / max(np.linalg.norm(axis), 1e-12))
        cosang = float(np.clip(abs(np.dot(axes[0], axes[1])), 0.0, 1.0))
        report["pair_axis_angle_deg"] = float(math.degrees(math.acos(cosang)))
        report["antipodal_pairing"] = [list(p) for p in pairing]

    # Orthogonal Procrustes (with scale) of learned columns onto the truth.
    D = codebook.T.astype(np.float64)  # [2,K]
    A = cols
    M = D @ A.T
    U, S, Vt = np.linalg.svd(M)
    R = U @ Vt
    denom = float(np.sum(A * A))
    s = float(np.sum(S) / max(denom, 1e-12))
    resid = float(np.linalg.norm(s * (R @ A) - D, ord="fro") / max(np.linalg.norm(D, ord="fro"), 1e-12))
    report["procrustes_residual"] = resid
    report["procrustes_scale"] = s
    report["procrustes_rotation_deg"] = float(math.degrees(math.atan2(R[1, 0], R[0, 0])))
    return report


# ---------------------------------------------------------------------------
# Named pretrained-source registry
# ---------------------------------------------------------------------------


def resolve_pretrained_sources(cfg: dict) -> List[Dict[str, object]]:
    """Resolve explicit named sources while preserving the legacy nav list.

    The old ``nav_checkpoints`` format hard-coded every checkpoint as
    ``nav_frozen``.  That is unsafe for CTL-01 because a sequence-control
    checkpoint would then be pooled with the navigation arm.  The new format
    is one named entry per scientific source:

    ``pretrained_sources: [{arm, checkpoint, source_type, require_gate}]``.
    """
    explicit = cfg.get("pretrained_sources")
    if explicit is None:
        checkpoints = cfg.get("nav_checkpoints") or [cfg.get("transfer_checkpoint", "")]
        resolved = [
            {
                "arm": "nav_frozen",
                "checkpoint": str(checkpoint),
                "source_type": "navigation",
                "require_gate": False,
            }
            for checkpoint in checkpoints
            if checkpoint
        ]
    else:
        if not isinstance(explicit, list) or not explicit:
            raise ValueError("pretrained_sources must be a non-empty list")
        resolved = []
        seen_arms = set()
        reserved = {"random_frozen", "scratch_full"}
        for index, raw in enumerate(explicit):
            if not isinstance(raw, dict):
                raise TypeError(
                    f"pretrained_sources[{index}] must be a mapping, got {type(raw).__name__}"
                )
            arm = str(raw.get("arm", "")).strip()
            checkpoint = str(raw.get("checkpoint", "")).strip()
            if not arm or not checkpoint:
                raise ValueError(
                    f"pretrained_sources[{index}] requires non-empty arm and checkpoint"
                )
            if arm in reserved:
                raise ValueError(f"Source arm {arm!r} is reserved")
            if arm in seen_arms:
                raise ValueError(
                    f"Duplicate explicit source arm {arm!r}; use one checkpoint per arm "
                    "so source identities are not pooled."
                )
            seen_arms.add(arm)
            resolved.append(
                {
                    "arm": arm,
                    "checkpoint": checkpoint,
                    "source_type": str(raw.get("source_type", "pretrained")),
                    "require_gate": bool(raw.get("require_gate", False)),
                }
            )

    if not resolved:
        raise FileNotFoundError("No pretrained source checkpoint was configured")
    return resolved


def validate_pretrained_sources(
    sources: List[Dict[str, object]], cfg: dict
) -> List[Dict[str, object]]:
    """Validate existence, transfer shapes and any required training gate."""
    hidden_dim = int(cfg["model"]["hidden_dim"])
    transfer_mode = str(cfg.get("transfer", {}).get("mode", "full"))
    required_keys = {
        "full": (
            ("grid_rnn.cell.in_proj.weight", [hidden_dim, 2]),
            ("grid_rnn.cell.h_proj.weight", [hidden_dim, hidden_dim]),
        ),
        "input_only": (("grid_rnn.cell.in_proj.weight", [hidden_dim, 2]),),
        "recurrent_only": (
            ("grid_rnn.cell.h_proj.weight", [hidden_dim, hidden_dim]),
        ),
    }
    if transfer_mode not in required_keys:
        raise ValueError(f"Unsupported transfer.mode={transfer_mode!r}")

    manifest: List[Dict[str, object]] = []
    for source in sources:
        path = Path(str(source["checkpoint"]))
        if not path.exists():
            raise FileNotFoundError(
                f"Configured source {source['arm']!r} checkpoint does not exist: {path}"
            )
        checkpoint = load_checkpoint(path, map_location="cpu")
        state = checkpoint.get("model_state")
        if not isinstance(state, dict):
            raise ValueError(f"Checkpoint has no model_state mapping: {path}")
        shapes: Dict[str, object] = {}
        for key, expected_shape in required_keys[transfer_mode]:
            if key not in state:
                raise ValueError(
                    f"Source {source['arm']!r} is missing required tensor {key!r}: {path}"
                )
            observed_shape = list(state[key].shape)
            shapes[key] = observed_shape
            if observed_shape != expected_shape:
                raise ValueError(
                    f"Source {source['arm']!r} has incompatible {key}: "
                    f"observed={observed_shape}, expected={expected_shape}"
                )

        gate = checkpoint.get("ctl01_gate")
        if bool(source.get("require_gate", False)):
            if not isinstance(gate, dict) or gate.get("passed") is not True:
                raise RuntimeError(
                    f"Source {source['arm']!r} requires a passing ctl01_gate, "
                    f"but the checkpoint is not qualified: {path}"
                )
        manifest.append(
            {
                **source,
                "checkpoint": str(path),
                "sha256": sha256_file(path),
                "transfer_mode": transfer_mode,
                "validated_shapes": shapes,
                "pretraining_source_type_in_checkpoint": checkpoint.get(
                    "pretraining_source_type"
                ),
                "ctl01_gate": gate,
            }
        )
    return manifest


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _fit_with_validation_selection(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: dict,
    device: torch.device,
    *,
    train_cfg: Optional[dict] = None,
) -> Tuple[float, dict]:
    """Train a model and select its state using validation accuracy only."""
    tcfg = cfg["train"] if train_cfg is None else train_cfg
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("The configured model has no trainable parameters")
    optimizer = torch.optim.Adam(
        trainable,
        lr=float(tcfg["lr"]),
        weight_decay=float(tcfg.get("weight_decay", 0.0)),
    )

    best_val = -float("inf")
    best_state: Optional[dict] = None
    eval_every = int(tcfg.get("eval_every", 5))
    for epoch in range(1, int(tcfg["epochs"]) + 1):
        train_pathint_epoch(
            model,
            train_loader,
            optimizer,
            device,
            grad_clip=float(tcfg.get("grad_clip", 1.0)),
            whh_weight_decay=0.0,
            position_loss_weight=0.0,
            rhc_alignment_weight=0.0,
        )
        if epoch % eval_every == 0 or epoch == int(tcfg["epochs"]):
            val_stats = evaluate_pathint_basic(model, val_loader, device)
            if float(val_stats["accuracy"]) > best_val:
                best_val = float(val_stats["accuracy"])
                best_state = copy.deepcopy(model.state_dict())

    if best_state is None:
        raise RuntimeError("Validation selection did not produce a checkpoint")
    model.load_state_dict(best_state)
    return best_val, best_state


@torch.no_grad()
def evaluate_without_recurrence(
    model: AdapterPathIntModel,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate a trained adapter model after setting W_hh to zero.

    This is a post-hoc mechanism ablation, not a separately optimised baseline.
    The original recurrent matrix is restored even when evaluation raises.
    """
    h_proj = getattr(getattr(model.base.grid_rnn, "cell", None), "h_proj", None)
    if h_proj is None or not hasattr(h_proj, "weight"):
        raise TypeError("Recurrence ablation currently requires a vanilla RNN h_proj")
    recurrent_weight = h_proj.weight
    backup = recurrent_weight.detach().clone()
    try:
        recurrent_weight.zero_()
        return evaluate_pathint_basic(model, loader, device)
    finally:
        recurrent_weight.copy_(backup)


def run_simple_baseline(
    arm: str,
    budget: int,
    seed: int,
    cfg: dict,
    inner_loaders,
    codebook: np.ndarray,
    n_nodes: int,
    device: torch.device,
    logger,
) -> Dict[str, object]:
    """Train a non-recurrent task-characterisation baseline."""
    inner_train, inner_val, inner_test = inner_loaders
    dcfg = cfg["data"]
    set_seed(seed)

    train_loader = wrap_loader(
        inner_train,
        codebook,
        seed,
        shuffle=True,
        batch_size=int(dcfg["batch_size"]),
        subset=budget,
    )
    val_loader = wrap_loader(
        inner_val,
        codebook,
        seed,
        shuffle=False,
        batch_size=int(dcfg["batch_size"]),
    )
    test_loader = wrap_loader(
        inner_test,
        codebook,
        seed,
        shuffle=False,
        batch_size=int(dcfg["batch_size"]),
    )

    n_actions = int(codebook.shape[0])
    controls_cfg = cfg.get("controls", {})
    if arm == "baseline_action_count_linear":
        model: torch.nn.Module = ActionCountLinearBaseline(
            n_nodes=n_nodes,
            n_actions=n_actions,
            use_bias=True,
        )
        geometry: Dict[str, object] = {}
    elif arm == "baseline_linear_accumulator":
        latent_dim = int(controls_cfg.get("linear_accumulator_dim", 2))
        model = LinearAccumulatorBaseline(
            n_nodes=n_nodes,
            n_actions=n_actions,
            latent_dim=latent_dim,
            init_scale=float(dcfg.get("velocity_scale", 0.02)),
            use_bias=True,
        )
        geometry = {}
    else:
        raise ValueError(f"Unknown simple baseline arm: {arm}")

    model.to(device)
    baseline_train_cfg = dict(cfg["train"])
    baseline_train_cfg.update(controls_cfg.get("baseline_train", {}))
    best_val, _ = _fit_with_validation_selection(
        model,
        train_loader,
        val_loader,
        cfg,
        device,
        train_cfg=baseline_train_cfg,
    )
    test_stats = evaluate_pathint_basic(model, test_loader, device)

    if isinstance(model, LinearAccumulatorBaseline) and model.latent_dim == 2:
        learned = model.action_matrix().detach().cpu().numpy().astype(np.float64)
        geometry = adapter_geometry_report(learned, codebook)

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    row = {
        "arm": arm,
        "arm_kind": "simple_nonrecurrent_baseline",
        "source": "none",
        "source_type": "simple_nonrecurrent_baseline",
        "budget": int(budget),
        "seed": int(seed),
        "n_trainable_params": int(sum(parameter.numel() for parameter in trainable)),
        "test_accuracy": float(test_stats["accuracy"]),
        "best_val_accuracy": float(best_val),
        "antipodality_index": geometry.get("antipodality_index"),
        "pair_axis_angle_deg": geometry.get("pair_axis_angle_deg"),
        "procrustes_residual": geometry.get("procrustes_residual"),
        "procrustes_rotation_deg": geometry.get("procrustes_rotation_deg"),
        "adapter_geometry": geometry or None,
        "test_accuracy_no_recurrence": None,
        "recurrence_ablation_drop": None,
    }
    logger.info(
        "  [%s | budget %d | seed %d] test=%.4f",
        arm,
        budget,
        seed,
        row["test_accuracy"],
    )
    return row


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


def run_one(
    arm: str,
    source_ckpt: Optional[str],
    source_type: str,
    budget: int,
    seed: int,
    cfg: dict,
    inner_loaders,
    codebook: np.ndarray,
    n_nodes: int,
    device: torch.device,
    logger,
    *,
    evaluate_recurrence_ablation: bool = False,
) -> Dict[str, object]:
    inner_train, inner_val, inner_test = inner_loaders
    dcfg = cfg["data"]
    set_seed(seed)

    train_loader = wrap_loader(
        inner_train, codebook, seed, shuffle=True,
        batch_size=dcfg["batch_size"], subset=budget,
    )
    val_loader = wrap_loader(inner_val, codebook, seed, shuffle=False, batch_size=dcfg["batch_size"])
    test_loader = wrap_loader(inner_test, codebook, seed, shuffle=False, batch_size=dcfg["batch_size"])

    base = build_base_model(cfg, n_nodes, device)
    if source_ckpt is not None:
        counts = base.load_nav_checkpoint(
            source_ckpt, transfer_mode=cfg.get("transfer", {}).get("mode", "full")
        )
        if counts["copied"] == 0:
            raise RuntimeError(f"0 parameters copied from {source_ckpt!r}.")

    model = AdapterPathIntModel(
        base,
        in_dim=int(codebook.shape[0]),
        transform=None,
        adapter_init="small",
        adapter_init_scale=float(dcfg.get("velocity_scale", 0.02)),
    )
    if arm == "scratch_full":
        # everything trainable (upper bound; far more parameters — labelled)
        for p in model.parameters():
            p.requires_grad = True
        if model.base.position_head is not None:
            for p in model.base.position_head.parameters():
                p.requires_grad = False
    else:
        model.configure_trainable()
    model.to(device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    best_val, _ = _fit_with_validation_selection(
        model, train_loader, val_loader, cfg, device
    )
    test_stats = evaluate_pathint_basic(model, test_loader, device)
    no_recurrence_accuracy: Optional[float] = None
    recurrence_drop: Optional[float] = None
    if evaluate_recurrence_ablation:
        ablated_stats = evaluate_without_recurrence(model, test_loader, device)
        no_recurrence_accuracy = float(ablated_stats["accuracy"])
        recurrence_drop = float(test_stats["accuracy"]) - no_recurrence_accuracy

    geometry = adapter_geometry_report(model.adapter_matrix(), codebook)
    row = {
        "arm": arm,
        "arm_kind": (
            "scratch_trainable_core"
            if arm == "scratch_full"
            else "matched_random_core"
            if arm == "random_frozen"
            else "pretrained_frozen_core"
        ),
        "source": source_ckpt or "random",
        "source_type": source_type,
        "budget": int(budget),
        "seed": seed,
        "n_trainable_params": int(sum(p.numel() for p in trainable)),
        "test_accuracy": float(test_stats["accuracy"]),
        "best_val_accuracy": float(best_val),
        "antipodality_index": geometry.get("antipodality_index"),
        "pair_axis_angle_deg": geometry.get("pair_axis_angle_deg"),
        "procrustes_residual": geometry.get("procrustes_residual"),
        "procrustes_rotation_deg": geometry.get("procrustes_rotation_deg"),
        "adapter_geometry": geometry,
        "test_accuracy_no_recurrence": no_recurrence_accuracy,
        "recurrence_ablation_drop": recurrence_drop,
    }
    logger.info(
        "  [%s | budget %d | seed %d] test=%.4f procrustes=%.3f%s",
        arm,
        budget,
        seed,
        row["test_accuracy"],
        geometry["procrustes_residual"],
        (
            f" no-recurrence={no_recurrence_accuracy:.4f}"
            if no_recurrence_accuracy is not None
            else ""
        ),
    )
    return row


def aggregate(
    rows: List[Dict[str, object]], budgets: List[int], low_budget_max: int
) -> Dict[str, object]:
    """Aggregate named sources against the same random-frozen baseline.

    AUC is intentionally restricted to budgets ``<= low_budget_max``.  The
    previous implementation accepted that argument but integrated every
    available budget, which made the reported "low-budget AUC" incorrect.
    """

    def values_by_seed(arm: str, budget: int) -> Dict[int, float]:
        grouped: Dict[int, List[float]] = {}
        for row in rows:
            if row["arm"] == arm and int(row["budget"]) == int(budget):
                grouped.setdefault(int(row["seed"]), []).append(
                    float(row["test_accuracy"])
                )
        return {seed: float(np.mean(values)) for seed, values in grouped.items()}

    def auc(arm: str, seed: int) -> Optional[float]:
        # Average duplicate rows at a budget before integrating.  Explicit
        # source configs disallow duplicate arm names, but this also makes
        # legacy multi-checkpoint files deterministic.
        grouped: Dict[int, List[float]] = {}
        for row in rows:
            budget = int(row["budget"])
            if (
                row["arm"] == arm
                and int(row["seed"]) == int(seed)
                and budget <= low_budget_max
            ):
                grouped.setdefault(budget, []).append(float(row["test_accuracy"]))
        pts = sorted((budget, float(np.mean(values))) for budget, values in grouped.items())
        if len(pts) < 2:
            return None
        xs = np.log(np.asarray([point[0] for point in pts], dtype=np.float64))
        ys = np.asarray([point[1] for point in pts], dtype=np.float64)
        return float(np.trapezoid(ys, xs) / (xs[-1] - xs[0]))

    arms = sorted({str(row["arm"]) for row in rows})
    arm_kinds: Dict[str, str] = {}
    for row in rows:
        arm = str(row["arm"])
        kind = row.get("arm_kind")
        if kind is not None:
            arm_kinds.setdefault(arm, str(kind))
    # Backward compatibility for previously generated rows that predate the
    # explicit arm-kind field.
    for arm in arms:
        if arm in arm_kinds:
            continue
        if arm in BASELINE_ARMS or arm.startswith("baseline_"):
            arm_kinds[arm] = "simple_nonrecurrent_baseline"
        elif arm == "random_frozen":
            arm_kinds[arm] = "matched_random_core"
        elif arm == "scratch_full":
            arm_kinds[arm] = "scratch_trainable_core"
        else:
            arm_kinds[arm] = "pretrained_frozen_core"

    source_arms = [
        arm for arm in arms if arm_kinds.get(arm) == "pretrained_frozen_core"
    ]
    baseline_arms = [
        arm
        for arm in arms
        if arm_kinds.get(arm) == "simple_nonrecurrent_baseline"
    ]
    seeds = sorted({int(row["seed"]) for row in rows})
    summary: Dict[str, object] = {
        "auc_budget_max": int(low_budget_max),
        "arms": arms,
        "arm_kinds": arm_kinds,
        "source_arms": source_arms,
        "simple_baseline_arms": baseline_arms,
        "per_budget": {},
        "criteria": {},
        "criteria_by_source": {},
        "task_characterization": {},
    }

    for budget in budgets:
        arm_stats: Dict[str, object] = {}
        for arm in arms:
            values = list(values_by_seed(arm, budget).values())
            no_recurrence_values = [
                float(row["test_accuracy_no_recurrence"])
                for row in rows
                if row["arm"] == arm
                and int(row["budget"]) == int(budget)
                and row.get("test_accuracy_no_recurrence") is not None
            ]
            recurrence_drops = [
                float(row["recurrence_ablation_drop"])
                for row in rows
                if row["arm"] == arm
                and int(row["budget"]) == int(budget)
                and row.get("recurrence_ablation_drop") is not None
            ]
            arm_stats[arm] = {
                "mean": float(np.mean(values)) if values else None,
                "sd": float(np.std(values)) if values else None,
                "n_seeds": len(values),
                "no_recurrence_mean": (
                    float(np.mean(no_recurrence_values))
                    if no_recurrence_values
                    else None
                ),
                "recurrence_ablation_drop_mean": (
                    float(np.mean(recurrence_drops)) if recurrence_drops else None
                ),
            }
        comparisons: Dict[str, object] = {}
        simple_baseline_comparisons: Dict[str, Dict[str, object]] = {}
        random_by_seed = values_by_seed("random_frozen", budget)
        for arm in source_arms:
            source_by_seed = values_by_seed(arm, budget)
            common = sorted(set(source_by_seed) & set(random_by_seed))
            diffs = [source_by_seed[seed] - random_by_seed[seed] for seed in common]
            comparisons[arm] = {
                "paired_diff_mean": float(np.mean(diffs)) if diffs else None,
                "paired_differences": diffs,
                "n_pairs": len(diffs),
            }
            simple_baseline_comparisons[arm] = {}
            for baseline_arm in baseline_arms:
                baseline_by_seed = values_by_seed(baseline_arm, budget)
                baseline_common = sorted(
                    set(source_by_seed) & set(baseline_by_seed)
                )
                baseline_diffs = [
                    source_by_seed[seed] - baseline_by_seed[seed]
                    for seed in baseline_common
                ]
                simple_baseline_comparisons[arm][baseline_arm] = {
                    "paired_diff_mean": (
                        float(np.mean(baseline_diffs)) if baseline_diffs else None
                    ),
                    "paired_differences": baseline_diffs,
                    "n_pairs": len(baseline_diffs),
                }

        nav_stats = arm_stats.get("nav_frozen", {})
        random_stats = arm_stats.get("random_frozen", {})
        nav_comparison = comparisons.get("nav_frozen", {})
        summary["per_budget"][str(budget)] = {
            "arms": arm_stats,
            "comparisons_to_random_frozen": comparisons,
            "comparisons_to_simple_baselines": simple_baseline_comparisons,
            # Legacy fields retained for existing figure/report readers.
            "nav_frozen_mean": nav_stats.get("mean"),
            "nav_frozen_sd": nav_stats.get("sd"),
            "random_frozen_mean": random_stats.get("mean"),
            "random_frozen_sd": random_stats.get("sd"),
            "paired_diff_mean": nav_comparison.get("paired_diff_mean"),
        }

    for arm in source_arms:
        auc_diffs = []
        for seed in seeds:
            source_auc = auc(arm, seed)
            random_auc = auc("random_frozen", seed)
            if source_auc is not None and random_auc is not None:
                auc_diffs.append(source_auc - random_auc)
        baseline_auc_diffs: Dict[str, List[float]] = {}
        for baseline_arm in baseline_arms:
            diffs: List[float] = []
            for seed in seeds:
                source_auc = auc(arm, seed)
                baseline_auc = auc(baseline_arm, seed)
                if source_auc is not None and baseline_auc is not None:
                    diffs.append(source_auc - baseline_auc)
            baseline_auc_diffs[baseline_arm] = diffs
        low_budget_diffs = [
            summary["per_budget"][str(budget)]["comparisons_to_random_frozen"]
            .get(arm, {})
            .get("paired_diff_mean")
            for budget in budgets
            if budget <= low_budget_max
        ]
        low_budget_diffs = [
            float(value) for value in low_budget_diffs if value is not None
        ]
        geometry_rows = [
            row
            for row in rows
            if row["arm"] == arm
            and int(row["budget"]) == max(budgets)
            and row.get("procrustes_residual") is not None
        ]
        summary["criteria_by_source"][arm] = {
            "low_budget_all_positive": bool(
                low_budget_diffs and all(value > 0 for value in low_budget_diffs)
            ),
            "low_budget_auc_diff_mean": (
                float(np.mean(auc_diffs)) if auc_diffs else None
            ),
            "low_budget_auc_diff_positive_fraction": (
                float(np.mean([value > 0 for value in auc_diffs]))
                if auc_diffs
                else None
            ),
            "low_budget_auc_paired_differences": auc_diffs,
            "low_budget_auc_differences_vs_simple_baselines": {
                baseline_arm: {
                    "mean": float(np.mean(diffs)) if diffs else None,
                    "positive_fraction": (
                        float(np.mean([value > 0 for value in diffs]))
                        if diffs
                        else None
                    ),
                    "paired_differences": diffs,
                }
                for baseline_arm, diffs in baseline_auc_diffs.items()
            },
            "procrustes_residual_mean_at_max_budget": (
                float(
                    np.mean(
                        [float(row["procrustes_residual"]) for row in geometry_rows]
                    )
                )
                if geometry_rows
                else None
            ),
            "antipodality_mean_at_max_budget": (
                float(
                    np.mean(
                        [
                            float(row["antipodality_index"])
                            for row in geometry_rows
                            if row.get("antipodality_index") is not None
                        ]
                    )
                )
                if any(
                    row.get("antipodality_index") is not None
                    for row in geometry_rows
                )
                else None
            ),
            "axis_angle_mean_deg_at_max_budget": (
                float(
                    np.mean(
                        [
                            float(row["pair_axis_angle_deg"])
                            for row in geometry_rows
                            if row.get("pair_axis_angle_deg") is not None
                        ]
                    )
                )
                if any(
                    row.get("pair_axis_angle_deg") is not None
                    for row in geometry_rows
                )
                else None
            ),
        }

    baseline_observed_budgets = sorted(
        {
            int(row["budget"])
            for row in rows
            if str(row["arm"]) in baseline_arms
        }
    )
    max_budget = max(budgets)
    baseline_max_budget = (
        max(baseline_observed_budgets) if baseline_observed_budgets else None
    )
    summary["task_characterization"] = {
        "max_budget": (
            int(baseline_max_budget) if baseline_max_budget is not None else None
        ),
        "simple_baselines": {
            arm: {
                "mean_accuracy_at_max_budget": (
                    summary["per_budget"][str(baseline_max_budget)]["arms"]
                    .get(arm, {})
                    .get("mean")
                    if baseline_max_budget is not None
                    else None
                ),
                "procrustes_residual_mean_at_max_budget": (
                    float(
                        np.mean(
                            [
                                float(row["procrustes_residual"])
                                for row in rows
                                if row["arm"] == arm
                                and baseline_max_budget is not None
                                and int(row["budget"]) == int(baseline_max_budget)
                                and row.get("procrustes_residual") is not None
                            ]
                        )
                    )
                    if any(
                        row["arm"] == arm
                        and baseline_max_budget is not None
                        and int(row["budget"]) == int(baseline_max_budget)
                        and row.get("procrustes_residual") is not None
                        for row in rows
                    )
                    else None
                ),
            }
            for arm in baseline_arms
        },
        "interpretation_rule": (
            "If a non-recurrent baseline matches or exceeds a pretrained-core arm, "
            "the protocol does not establish that recurrent navigation dynamics are "
            "necessary. Adapter geometry alone is then task-characterisation evidence."
        ),
    }

    nav = summary["criteria_by_source"].get("nav_frozen", {})
    summary["criteria"] = {
        # Legacy names, now correctly restricted to <= low_budget_max.
        "S1_low_budget_all_positive": nav.get("low_budget_all_positive"),
        "S1_auc_diff_mean": nav.get("low_budget_auc_diff_mean"),
        "S1_auc_diff_positive_fraction": nav.get(
            "low_budget_auc_diff_positive_fraction"
        ),
        "S2_procrustes_residual_mean": nav.get(
            "procrustes_residual_mean_at_max_budget"
        ),
        "S2_antipodality_mean": nav.get("antipodality_mean_at_max_budget"),
        "S2_axis_angle_mean_deg": nav.get(
            "axis_angle_mean_deg_at_max_budget"
        ),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp8: symbolic one-hot adapter")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--n_seeds", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument(
        "--controls-only",
        action="store_true",
        help=(
            "Run only the non-recurrent action-count and linear-accumulator "
            "controls. No pretrained checkpoint is required."
        ),
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.output_dir is not None:
        cfg["output_dir"] = args.output_dir
    n_seeds = args.n_seeds if args.n_seeds is not None else cfg.get("n_seeds", 5)
    device = configure_runtime(cfg)
    output_dir = prepare_output_dir(cfg["output_dir"])
    logger = get_logger("exp8_symbolic_adapter", output_dir / "experiment.log")
    save_config_snapshot(cfg, output_dir)
    save_run_provenance(output_dir, "exp8_symbolic_adapter", cfg=cfg)

    if args.controls_only:
        source_manifest: List[Dict[str, object]] = []
    else:
        configured_sources = resolve_pretrained_sources(cfg)
        source_manifest = validate_pretrained_sources(configured_sources, cfg)
    write_json(source_manifest, output_dir / "pretrained_source_manifest.json")
    logger.info(
        "Validated pretrained sources: %s",
        [(source["arm"], source["source_type"]) for source in source_manifest],
    )

    dcfg = cfg["data"]
    budgets = [int(b) for b in cfg.get("episode_budgets", [10, 20, 50, 100, 200, 500, 1000])]
    low_budget_max = int(cfg.get("low_budget_max", 200))
    base_seed = int(cfg.get("seed", 42))
    controls_cfg = cfg.get("controls", {})
    include_action_count = bool(
        controls_cfg.get("include_action_count_linear", False)
    )
    include_linear_accumulator = bool(
        controls_cfg.get("include_linear_accumulator", False)
    )
    if args.controls_only and not (include_action_count or include_linear_accumulator):
        raise ValueError(
            "--controls-only requires at least one simple control to be enabled"
        )
    recurrence_budgets_raw = controls_cfg.get(
        "recurrence_ablation_budgets", []
    )
    if recurrence_budgets_raw == "max":
        recurrence_budgets = {max(budgets)}
    else:
        recurrence_budgets = {int(value) for value in recurrence_budgets_raw}
    baseline_budgets_raw = controls_cfg.get("baseline_budgets", budgets)
    if baseline_budgets_raw == "all":
        baseline_budgets = set(budgets)
    else:
        baseline_budgets = {int(value) for value in baseline_budgets_raw}

    protocol_manifest: Dict[str, object] = {
        "split_strategy": dcfg.get("split_strategy", "random"),
        "seq_len": dcfg.get("seq_len"),
        "seq_len_range": dcfg.get("seq_len_range"),
        "episode_budgets": budgets,
        "low_budget_max": low_budget_max,
        "controls": {
            "controls_only": bool(args.controls_only),
            "include_action_count_linear": include_action_count,
            "include_linear_accumulator": include_linear_accumulator,
            "linear_accumulator_dim": int(
                controls_cfg.get("linear_accumulator_dim", 2)
            ),
            "baseline_train": {
                **dict(cfg["train"]),
                **dict(controls_cfg.get("baseline_train", {})),
            },
            "baseline_budgets": sorted(baseline_budgets),
            "recurrence_ablation_budgets": sorted(recurrence_budgets),
        },
        "seeds": [],
    }
    splits_dir = output_dir / "split_audits"
    splits_dir.mkdir(parents=True, exist_ok=False)

    rows: List[Dict[str, object]] = []
    for seed_idx in range(n_seeds):
        seed = base_seed + seed_idx
        (
            inner_train, inner_val, inner_test,
            _task, node_to_id, _action_to_id, _node_coords,
        ) = build_pathint_loaders(
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
            velocity_transform="none",
            seq_len_range=(
                tuple(int(value) for value in dcfg["seq_len_range"])
                if dcfg.get("seq_len_range") is not None
                else None
            ),
            validation_ratio=dcfg.get("validation_ratio", 0.1),
            return_validation=True,
            num_workers=int(dcfg.get("num_workers", 0)),
        )
        n_nodes = len(node_to_id)
        codebook = build_action_codebook(inner_train.dataset)
        logger.info("seed %d: action codebook K=%d", seed, codebook.shape[0])
        inner_loaders = (inner_train, inner_val, inner_test)

        split_info = split_manifest(
            list(inner_train.dataset.episodes),
            list(inner_val.dataset.episodes),
            list(inner_test.dataset.episodes),
        )
        split_info.update(
            {
                "seed": int(seed),
                "split_strategy": dcfg.get("split_strategy", "random"),
                "n_nodes": int(n_nodes),
                "n_actions": int(codebook.shape[0]),
                "action_codebook": codebook.tolist(),
            }
        )
        write_json(split_info, splits_dir / f"seed_{seed}.json")
        protocol_manifest["seeds"].append(split_info)

        for budget in budgets:
            evaluate_recurrence = int(budget) in recurrence_budgets
            if not args.controls_only:
                for source in source_manifest:
                    rows.append(run_one(
                        str(source["arm"]),
                        str(source["checkpoint"]),
                        str(source.get("source_type", "pretrained")),
                        budget,
                        seed,
                        cfg,
                        inner_loaders,
                        codebook,
                        n_nodes,
                        device,
                        logger,
                        evaluate_recurrence_ablation=evaluate_recurrence,
                    ))
                rows.append(run_one(
                    "random_frozen",
                    None,
                    "matched_random",
                    budget,
                    seed,
                    cfg,
                    inner_loaders,
                    codebook,
                    n_nodes,
                    device,
                    logger,
                    evaluate_recurrence_ablation=evaluate_recurrence,
                ))
                if cfg.get("include_scratch", False):
                    rows.append(run_one(
                        "scratch_full",
                        None,
                        "scratch_random",
                        budget,
                        seed,
                        cfg,
                        inner_loaders,
                        codebook,
                        n_nodes,
                        device,
                        logger,
                        evaluate_recurrence_ablation=evaluate_recurrence,
                    ))
            if include_action_count and int(budget) in baseline_budgets:
                rows.append(
                    run_simple_baseline(
                        "baseline_action_count_linear",
                        budget,
                        seed,
                        cfg,
                        inner_loaders,
                        codebook,
                        n_nodes,
                        device,
                        logger,
                    )
                )
            if include_linear_accumulator and int(budget) in baseline_budgets:
                rows.append(
                    run_simple_baseline(
                        "baseline_linear_accumulator",
                        budget,
                        seed,
                        cfg,
                        inner_loaders,
                        codebook,
                        n_nodes,
                        device,
                        logger,
                    )
                )
        with (output_dir / "symbolic_adapter_rows.json").open("w") as fh:
            json.dump(rows, fh, indent=2)
        keys = [
            "arm", "arm_kind", "source", "source_type", "budget", "seed", "n_trainable_params",
            "test_accuracy", "best_val_accuracy", "antipodality_index",
            "pair_axis_angle_deg", "procrustes_residual", "procrustes_rotation_deg",
            "test_accuracy_no_recurrence", "recurrence_ablation_drop",
        ]
        with (output_dir / "symbolic_adapter_rows.csv").open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            for r in rows:
                writer.writerow(r)

    write_json(protocol_manifest, output_dir / "protocol_manifest.json")
    summary = aggregate(rows, budgets, low_budget_max)
    summary["protocol"] = {
        "split_strategy": dcfg.get("split_strategy", "random"),
        "seq_len": dcfg.get("seq_len"),
        "seq_len_range": dcfg.get("seq_len_range"),
    }
    summary["controls_config"] = protocol_manifest["controls"]
    with (output_dir / "summary_symbolic_adapter.json").open("w") as fh:
        json.dump(summary, fh, indent=2)
    logger.info("Done. Summary: %s", output_dir / "summary_symbolic_adapter.json")


if __name__ == "__main__":
    main()
