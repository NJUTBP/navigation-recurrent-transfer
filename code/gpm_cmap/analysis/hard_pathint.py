"""Training and evaluation primitives for hard abstract path integration."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import io
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from gpm_cmap.analysis.upgrade_contracts import CLAIM_SCOPE, hash_jsonable
from gpm_cmap.data.graph_tasks.pathint_episodes import episode_fingerprint
from gpm_cmap.models.abstract_pathint import AbstractPathIntModel
from gpm_cmap.utils.checkpointing import save_checkpoint


@dataclass
class BudgetTrainingResult:
    best_update: int
    best_validation_accuracy: float
    best_validation_nll: float
    stopped_update: int
    history: list[dict[str, Any]]
    state_dict: dict[str, torch.Tensor]


def build_hard_model(model_cfg: Mapping[str, Any], *, n_nodes: int, device: torch.device) -> AbstractPathIntModel:
    model = AbstractPathIntModel(
        n_nodes=int(n_nodes),
        hidden_dim=int(model_cfg["hidden_dim"]),
        content_dim=int(model_cfg.get("content_dim", 64)),
        nonnegative=bool(model_cfg.get("nonnegative", True)),
        cell_type=str(model_cfg.get("cell_type", "rnn")),
        use_bias=bool(model_cfg.get("use_bias", False)),
        dropout=float(model_cfg.get("dropout", 0.0)),
        periodic_scales=int(model_cfg.get("periodic_scales", 0)),
        use_position_head=bool(model_cfg.get("use_position_head", False)),
        start_init=str(model_cfg.get("start_init", "embedding")),
        coord_dim=int(model_cfg.get("coord_dim", 2)),
    )
    return model.to(device)


def _batch_to_device(batch: Mapping[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: value.to(device) if torch.is_tensor(value) else value for name, value in batch.items()}


def _forward_batch(model: AbstractPathIntModel, batch: Mapping[str, torch.Tensor]) -> Mapping[str, torch.Tensor]:
    start_coords = batch.get("start_coords") if model.start_init == "coordinate" else None
    return model(
        batch["start_node"],
        batch["velocities"],
        lengths=batch.get("lengths"),
        start_coords=start_coords,
        validate_inputs=False,
    )


def expected_calibration_error(probabilities: np.ndarray, targets: np.ndarray, *, bins: int = 15) -> float:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    if probabilities.ndim != 2 or probabilities.shape[0] != targets.size:
        raise ValueError("probabilities/targets shape mismatch")
    confidence = probabilities.max(axis=1)
    prediction = probabilities.argmax(axis=1)
    correct = (prediction == targets).astype(np.float64)
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    ece = 0.0
    for index in range(int(bins)):
        lo, hi = edges[index], edges[index + 1]
        mask = (confidence >= lo) & (confidence < hi if index + 1 < bins else confidence <= hi)
        if np.any(mask):
            ece += float(mask.mean()) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return float(ece)


@torch.no_grad()
def evaluate_loader_detailed(
    model: AbstractPathIntModel,
    loader,
    device: torch.device,
    *,
    split_name: str,
    calibration_bins: int = 15,
    collect_hidden: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    losses: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    lengths_all: list[np.ndarray] = []
    starts: list[np.ndarray] = []
    l1_all: list[np.ndarray] = []
    margins: list[np.ndarray] = []
    hidden: list[np.ndarray] = []
    offset = 0
    rows: list[dict[str, Any]] = []
    episodes = getattr(loader.dataset, "episodes", None)
    for batch in loader:
        cpu_lengths = batch.get("lengths")
        batch_d = _batch_to_device(batch, device)
        output = _forward_batch(model, batch_d)
        logits = output["node_logits"]
        target = batch_d["target_node"]
        nll = F.cross_entropy(logits, target, reduction="none")
        prob = torch.softmax(logits, dim=-1)
        pred = prob.argmax(dim=-1)
        sorted_prob = torch.topk(prob, k=min(2, prob.shape[1]), dim=-1).values
        margin = sorted_prob[:, 0] - (sorted_prob[:, 1] if sorted_prob.shape[1] > 1 else 0.0)
        batch_size = target.shape[0]
        loss_np = nll.cpu().numpy()
        prob_np = prob.cpu().numpy()
        target_np = target.cpu().numpy()
        pred_np = pred.cpu().numpy()
        lengths_np = (
            cpu_lengths.numpy()
            if cpu_lengths is not None
            else np.full(batch_size, batch["velocities"].shape[1], dtype=np.int64)
        )
        l1_np = batch["net_l1_distance"].numpy()
        start_np = batch["start_node"].numpy()
        margin_np = margin.cpu().numpy()
        losses.append(loss_np)
        probabilities.append(prob_np)
        targets.append(target_np)
        predictions.append(pred_np)
        lengths_all.append(lengths_np)
        starts.append(start_np)
        l1_all.append(l1_np)
        margins.append(margin_np)
        if collect_hidden:
            hidden.append(output["hidden_final"].cpu().numpy())
        for local in range(batch_size):
            episode = episodes[offset + local] if episodes is not None else None
            rows.append({
                "split": split_name,
                "trajectory_index": offset + local,
                "trajectory_fingerprint": episode_fingerprint(episode) if episode is not None else None,
                "start_node": int(start_np[local]),
                "target_node": int(target_np[local]),
                "predicted_node": int(pred_np[local]),
                "path_length": int(lengths_np[local]),
                "net_l1_distance": float(l1_np[local]),
                "correct": int(pred_np[local] == target_np[local]),
                "nll": float(loss_np[local]),
                "confidence": float(prob_np[local].max()),
                "readout_margin": float(margin_np[local]),
            })
        offset += batch_size
    if not rows:
        raise RuntimeError(f"Evaluation split {split_name!r} is empty")
    loss_v = np.concatenate(losses)
    prob_v = np.concatenate(probabilities)
    target_v = np.concatenate(targets)
    pred_v = np.concatenate(predictions)
    length_v = np.concatenate(lengths_all)
    l1_v = np.concatenate(l1_all)
    correct = (pred_v == target_v).astype(np.float64)
    summary: dict[str, Any] = {
        "split": split_name,
        "n_trajectories": int(target_v.size),
        "accuracy": float(correct.mean()),
        "nll": float(loss_v.mean()),
        "ece": expected_calibration_error(prob_v, target_v, bins=calibration_bins),
        "mean_confidence": float(prob_v.max(axis=1).mean()),
        "mean_readout_margin": float(np.concatenate(margins).mean()),
        "accuracy_by_path_length": {
            str(int(length)): float(correct[length_v == length].mean())
            for length in sorted(np.unique(length_v))
        },
        "accuracy_by_endpoint_distance": {
            np.format_float_positional(float(distance), trim="-"): float(correct[l1_v == distance].mean())
            for distance in sorted(np.unique(l1_v))
        },
    }
    if collect_hidden:
        summary["hidden_final"] = np.concatenate(hidden)
    return summary, rows


def _infinite_batches(loader) -> Iterable[Mapping[str, torch.Tensor]]:
    while True:
        for batch in loader:
            yield batch


def train_fixed_updates(
    model: AbstractPathIntModel,
    train_loader,
    validation_loader,
    device: torch.device,
    train_cfg: Mapping[str, Any],
    *,
    seed: int,
) -> BudgetTrainingResult:
    """Train with a fixed update budget and select checkpoints on validation only."""
    if float(train_cfg.get("position_loss_weight", 0.0)) != 0.0:
        raise ValueError("Formal hard-task main analysis requires position_loss_weight=0")
    max_updates = int(train_cfg.get("max_gradient_updates", 2000))
    eval_every = int(train_cfg.get("eval_every_updates", 100))
    patience = int(train_cfg.get("early_stopping_patience_evals", 10))
    min_updates = int(train_cfg.get("minimum_updates", eval_every))
    if max_updates < 1 or eval_every < 1:
        raise ValueError("max_gradient_updates and eval_every_updates must be positive")
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not params:
        raise RuntimeError("No trainable parameters remain after applying the condition")
    optimizer = torch.optim.Adam(
        params,
        lr=float(train_cfg.get("lr", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )
    best_accuracy = -np.inf
    best_nll = np.inf
    best_update = 0
    best_state: dict[str, torch.Tensor] | None = None
    no_improve = 0
    history: list[dict[str, Any]] = []
    iterator = iter(_infinite_batches(train_loader))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    for update in range(1, max_updates + 1):
        model.train()
        batch = _batch_to_device(next(iterator), device)
        output = _forward_batch(model, batch)
        loss = F.cross_entropy(output["node_logits"], batch["target_node"])
        whh_decay = float(train_cfg.get("whh_weight_decay", 0.0))
        if whh_decay:
            loss = loss + whh_decay * model.recurrent_l2()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite training loss at update {update}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(params, grad_clip, error_if_nonfinite=True)
        optimizer.step()
        if update % eval_every == 0 or update == max_updates:
            validation, _rows = evaluate_loader_detailed(
                model, validation_loader, device, split_name="validation",
                calibration_bins=int(train_cfg.get("calibration_bins", 15)),
            )
            entry = {
                "update": update,
                "train_batch_nll": float(loss.detach().cpu().item()),
                "validation_accuracy": validation["accuracy"],
                "validation_nll": validation["nll"],
                "validation_ece": validation["ece"],
            }
            history.append(entry)
            improved = (
                validation["accuracy"] > best_accuracy + 1e-12
                or (
                    abs(validation["accuracy"] - best_accuracy) <= 1e-12
                    and validation["nll"] < best_nll
                )
            )
            if improved:
                best_accuracy = float(validation["accuracy"])
                best_nll = float(validation["nll"])
                best_update = update
                best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            if update >= min_updates and patience > 0 and no_improve >= patience:
                break
    if best_state is None:
        raise RuntimeError("Validation selection produced no checkpoint")
    return BudgetTrainingResult(
        best_update=best_update,
        best_validation_accuracy=float(best_accuracy),
        best_validation_nll=float(best_nll),
        stopped_update=history[-1]["update"],
        history=history,
        state_dict=best_state,
    )


def state_dict_sha256(state_dict: Mapping[str, torch.Tensor]) -> str:
    buffer = io.BytesIO()
    torch.save({name: tensor.detach().cpu() for name, tensor in state_dict.items()}, buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def save_target_checkpoint(
    path: str | Path,
    *,
    model: AbstractPathIntModel,
    state_dict: Mapping[str, torch.Tensor],
    model_config: Mapping[str, Any],
    condition: Mapping[str, Any],
    seed: int,
    run_seed: int | None = None,
    unique_episodes: int,
    best_update: int,
    validation_metrics: Mapping[str, Any],
    dataset_manifest_hash: str,
    source_linkage: Mapping[str, Any],
    schema_version: str = "hard_pathint_checkpoint_v1",
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Atomically save a new-stage target checkpoint.

    ``target_seed`` identifies the dataset/training replicate shared across
    conditions. ``run_seed`` records the fully resolved RNG seed for a specific
    budget/condition cell.  Existing project checkpoint formats are untouched;
    this function only writes the explicitly versioned I2--I4 formats.
    """
    path = Path(path)
    if schema_version not in {
        "hard_pathint_checkpoint_v1",
        "recurrent_factorial_checkpoint_v1",
    }:
        raise ValueError(f"Unsupported target checkpoint schema: {schema_version!r}")
    resolved_run_seed = int(seed if run_seed is None else run_seed)
    payload = {
        "schema_version": schema_version,
        "claim_scope": CLAIM_SCOPE,
        "model_state": {name: tensor.detach().cpu() for name, tensor in state_dict.items()},
        "model_config": dict(model_config),
        "n_nodes": int(model.n_nodes),
        "condition": dict(condition),
        "target_seed": int(seed),
        "run_seed": resolved_run_seed,
        "unique_training_episodes": int(unique_episodes),
        "best_validation_update": int(best_update),
        "validation_metrics": dict(validation_metrics),
        "dataset_manifest_hash": str(dataset_manifest_hash),
        "source_linkage": dict(source_linkage),
        "state_dict_sha256": state_dict_sha256(state_dict),
        "extra": dict(extra or {}),
    }
    save_checkpoint(payload, path)
    return path


def load_target_checkpoint(path: str | Path, *, device: torch.device) -> tuple[AbstractPathIntModel, dict[str, Any]]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Target checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") not in {"hard_pathint_checkpoint_v1", "recurrent_factorial_checkpoint_v1"}:
        raise ValueError(f"Unsupported target checkpoint schema: {payload.get('schema_version')!r}")
    model_cfg = payload["model_config"]
    model = build_hard_model(model_cfg, n_nodes=int(payload["n_nodes"]), device=device)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    return model, payload


def trapezoid_learning_curve_auc(rows: Sequence[Mapping[str, Any]], *, metric: str) -> float:
    if not rows:
        raise ValueError("Cannot compute learning-curve AUC from no rows")
    ordered = sorted(rows, key=lambda row: int(row["unique_training_episodes"]))
    x = np.asarray([float(row["unique_training_episodes"]) for row in ordered])
    y = np.asarray([float(row[metric]) for row in ordered])
    if x.size == 1:
        return float(y[0])
    log_x = np.log10(x)
    area = np.trapz(y, log_x)
    return float(area / max(log_x[-1] - log_x[0], 1e-12))


def threshold_sample_requirement(
    rows: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    threshold: float,
) -> int | None:
    for row in sorted(rows, key=lambda item: int(item["unique_training_episodes"])):
        if float(row[metric]) >= float(threshold):
            return int(row["unique_training_episodes"])
    return None
