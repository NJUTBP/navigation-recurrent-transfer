"""Training and evaluation for abstract path-integration reasoning."""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from gpm_cmap.analysis.repr_geometry import pairwise_distance_correlation
from gpm_cmap.utils.metrics import AverageMeter


def _require_finite(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all():
        raise FloatingPointError(f"{name} contains NaN or Inf")


def sequence_mask(lengths: torch.Tensor, max_steps: int) -> torch.Tensor:
    """Return ``[B,T]`` validity mask for padded sequence tensors."""
    if lengths.ndim != 1:
        raise ValueError(f"lengths must have shape [B]; got {tuple(lengths.shape)}")
    if max_steps < 1:
        raise ValueError(f"max_steps must be positive; got {max_steps}")
    if torch.any(lengths < 1) or torch.any(lengths > max_steps):
        raise ValueError(
            f"lengths must lie in [1,{max_steps}]; got min={int(lengths.min())}, "
            f"max={int(lengths.max())}"
        )
    return torch.arange(max_steps, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)


def gather_final_step(sequence: torch.Tensor, lengths: torch.Tensor | None) -> torch.Tensor:
    """Gather each row's final valid step, or the last step if unpadded."""
    if sequence.ndim < 2:
        raise ValueError(f"sequence must have shape [B,T,...]; got {tuple(sequence.shape)}")
    if lengths is None:
        return sequence[:, -1]
    if lengths.shape[0] != sequence.shape[0]:
        raise ValueError("lengths batch dimension does not match sequence")
    sequence_mask(lengths, sequence.shape[1])  # validates bounds
    index = (lengths.to(sequence.device) - 1).view(-1, 1)
    trailing = sequence.shape[2:]
    gather_index = index.view(sequence.shape[0], 1, *([1] * len(trailing))).expand(
        sequence.shape[0], 1, *trailing
    )
    return sequence.gather(1, gather_index).squeeze(1)


def masked_sequence_rows(sequence: torch.Tensor, lengths: torch.Tensor | None) -> torch.Tensor:
    """Flatten only valid sequence steps to ``[N_valid,...]``."""
    if lengths is None:
        return sequence.reshape(-1, *sequence.shape[2:])
    mask = sequence_mask(lengths.to(sequence.device), sequence.shape[1])
    return sequence[mask]


def masked_mse(prediction: torch.Tensor, target: torch.Tensor, lengths: torch.Tensor | None) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError(
            f"masked_mse shape mismatch: prediction={tuple(prediction.shape)} "
            f"target={tuple(target.shape)}"
        )
    if lengths is None:
        return F.mse_loss(prediction, target)
    valid_pred = masked_sequence_rows(prediction, lengths)
    valid_target = masked_sequence_rows(target, lengths)
    return F.mse_loss(valid_pred, valid_target)


def _distance_key(distance: float) -> str:
    """Preserve physically scaled distances instead of rounding them to zero."""
    return np.format_float_positional(float(distance), precision=7, trim="-")


def _loader_has_prevalidated_batches(loader: object, model: torch.nn.Module) -> bool:
    dataset = getattr(loader, "dataset", None)
    if not bool(getattr(dataset, "prevalidated", False)):
        return False

    n_nodes = getattr(model, "n_nodes", None)
    if n_nodes is not None:
        min_node_id = int(getattr(dataset, "min_node_id", 0))
        max_node_id = int(getattr(dataset, "max_node_id", -1))
        if min_node_id < 0 or max_node_id >= int(n_nodes):
            raise IndexError(
                f"dataset node ids must lie in [0,{int(n_nodes) - 1}]; "
                f"got min={min_node_id}, max={max_node_id}"
            )
    return bool(getattr(model, "supports_prevalidated_inputs", False))


def _pathint_forward(
    model: torch.nn.Module,
    *,
    start_node: torch.Tensor,
    velocities: torch.Tensor,
    lengths: torch.Tensor | None,
    start_coords: torch.Tensor | None,
    prevalidated: bool,
) -> Dict[str, torch.Tensor]:
    kwargs = {
        "start_node": start_node,
        "velocities": velocities,
        "lengths": lengths,
        "start_coords": start_coords,
    }
    if prevalidated:
        kwargs["validate_inputs"] = False
    return model(**kwargs)


def _uses_coordinate_start(model: torch.nn.Module) -> bool:
    return getattr(model, "start_init", None) == "coordinate"


def _update_metric_meters(
    metric_rows: List[torch.Tensor],
    batch_sizes: List[int],
    meters: List[AverageMeter],
) -> None:
    if not metric_rows:
        return
    values = torch.stack(metric_rows).cpu().tolist()
    for row, batch_size in zip(values, batch_sizes):
        for meter, value in zip(meters, row):
            meter.update(value, batch_size)


def train_pathint_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float = 1.0,
    whh_weight_decay: float = 0.0,
    position_loss_weight: float = 0.0,
    rhc_alignment_weight: float = 0.0,
) -> Dict[str, float]:
    """Train one epoch on the path-integration task."""
    model.train()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    pos_meter = AverageMeter()
    rhc_meter = AverageMeter()
    metric_rows: List[torch.Tensor] = []
    batch_sizes: List[int] = []
    prevalidated = _loader_has_prevalidated_batches(loader, model)
    coordinate_start = _uses_coordinate_start(model)

    for batch in loader:
        start_node = batch["start_node"].to(device)
        velocities = batch["velocities"].to(device)
        target_node = batch["target_node"].to(device)
        target_coords = (
            batch["target_coords"].to(device) if position_loss_weight > 0 else None
        )

        lengths = batch["lengths"].to(device) if "lengths" in batch else None
        start_coords = batch["start_coords"].to(device) if coordinate_start else None
        outputs = _pathint_forward(
            model,
            start_node=start_node,
            velocities=velocities,
            lengths=lengths,
            start_coords=start_coords,
            prevalidated=prevalidated,
        )
        logits = outputs["node_logits"]
        loss = F.cross_entropy(logits, target_node)

        pos_loss_metric = loss.new_zeros(())
        if position_loss_weight > 0 and "position_pred" in outputs:
            assert target_coords is not None
            pos_loss = F.mse_loss(outputs["position_pred"], target_coords)
            loss = loss + position_loss_weight * pos_loss
            pos_loss_metric = pos_loss.detach()

        rhc_loss_metric = loss.new_zeros(())
        if (
            rhc_alignment_weight > 0
            and "periodic_seq" in outputs
            and "path_coords" in batch
            and hasattr(model, "compute_rhc_target")
        ):
            path_coords = batch["path_coords"].to(device)
            z_target = model.compute_rhc_target(path_coords)
            if z_target is not None:
                z_pred = outputs["periodic_seq"]
                rhc_loss = masked_mse(z_pred, z_target.detach(), lengths)
                loss = loss + rhc_alignment_weight * rhc_loss
                rhc_loss_metric = rhc_loss.detach()

        if whh_weight_decay > 0 and hasattr(model, "recurrent_l2"):
            loss = loss + whh_weight_decay * model.recurrent_l2()

        _require_finite("training loss", loss)
        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            try:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    grad_clip,
                    error_if_nonfinite=True,
                )
            except RuntimeError as exc:
                if "non-finite" not in str(exc):
                    raise
                raise FloatingPointError("gradient norm contains NaN or Inf") from exc
        else:
            for name, parameter in model.named_parameters():
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                    raise FloatingPointError(f"gradient for {name} contains NaN or Inf")
        optimizer.step()

        preds = logits.argmax(dim=-1)
        acc = (preds == target_node).float().mean()
        metric_rows.append(
            torch.stack(
                (
                    loss.detach(),
                    acc.detach(),
                    pos_loss_metric,
                    rhc_loss_metric,
                )
            )
        )
        batch_sizes.append(start_node.size(0))

    _update_metric_meters(
        metric_rows,
        batch_sizes,
        [loss_meter, acc_meter, pos_meter, rhc_meter],
    )

    return {
        "loss": loss_meter.avg,
        "accuracy": acc_meter.avg,
        "position_loss": pos_meter.avg,
        "rhc_alignment_loss": rhc_meter.avg,
    }


def _compute_l1_distances(batch: Dict[str, torch.Tensor]) -> np.ndarray:
    if "net_l1_distance" in batch:
        return batch["net_l1_distance"].cpu().numpy()
    if "start_coords" in batch and "target_coords" in batch:
        return (batch["target_coords"] - batch["start_coords"]).abs().sum(dim=-1).cpu().numpy()
    return np.zeros(batch["start_node"].shape[0], dtype=np.float32)


def _group_metric_by_l1(values: np.ndarray, l1_distances: np.ndarray) -> Dict[str, float]:
    grouped: Dict[str, float] = {}
    if values.size == 0 or l1_distances.size == 0:
        return grouped
    unique_distances = sorted(np.unique(l1_distances))
    for dist in unique_distances:
        mask = l1_distances == dist
        key = _distance_key(float(dist))
        grouped[key] = float(values[mask].mean())
    return grouped


@torch.no_grad()
def evaluate_pathint_basic(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate only cross-entropy loss and node accuracy.

    This is the selection/diagnostic path. It intentionally skips hidden-state
    collection, NumPy transfers, pairwise-distance analysis, and sparsity
    statistics when callers do not consume those outputs.
    """
    model.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    metric_rows: List[torch.Tensor] = []
    batch_sizes: List[int] = []
    prevalidated = _loader_has_prevalidated_batches(loader, model)
    coordinate_start = _uses_coordinate_start(model)

    for batch in loader:
        start_node = batch["start_node"].to(device)
        velocities = batch["velocities"].to(device)
        target_node = batch["target_node"].to(device)
        lengths = batch["lengths"].to(device) if "lengths" in batch else None
        start_coords = batch["start_coords"].to(device) if coordinate_start else None
        outputs = _pathint_forward(
            model,
            start_node=start_node,
            velocities=velocities,
            lengths=lengths,
            start_coords=start_coords,
            prevalidated=prevalidated,
        )
        logits = outputs["node_logits"]
        loss = F.cross_entropy(logits, target_node)
        preds = logits.argmax(dim=-1)
        acc = (preds == target_node).float().mean()
        metric_rows.append(torch.stack((loss.detach(), acc.detach())))
        batch_sizes.append(start_node.size(0))

    _update_metric_meters(metric_rows, batch_sizes, [loss_meter, acc_meter])
    if not np.isfinite(loss_meter.avg):
        raise FloatingPointError("evaluation loss contains NaN or Inf")
    return {"loss": loss_meter.avg, "accuracy": acc_meter.avg}


@torch.no_grad()
def evaluate_pathint(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    position_supervised: bool = True,
    evaluation_seed: int = 0,
) -> Dict[str, object]:
    """Evaluate on the path-integration task with full analysis."""
    model.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    all_hidden_seq: List[np.ndarray] = []
    all_path_coords: List[np.ndarray] = []
    all_final_hidden: List[np.ndarray] = []
    all_final_coords: List[np.ndarray] = []
    all_pred_nodes: List[np.ndarray] = []
    all_target_nodes: List[np.ndarray] = []
    all_start_nodes: List[np.ndarray] = []
    all_l1_distances: List[np.ndarray] = []
    # Keep squared Euclidean error and Euclidean/L2 error separate.
    # Previous code stored an L2 error under the misleading key
    # ``position_mse``.  Downstream summaries now receive both metrics.
    all_pos_sq_errors_per_sample: List[np.ndarray] = []
    all_pos_l2_errors_per_sample: List[np.ndarray] = []
    prevalidated = _loader_has_prevalidated_batches(loader, model)
    coordinate_start = _uses_coordinate_start(model)

    for batch in loader:
        start_node = batch["start_node"].to(device)
        velocities = batch["velocities"].to(device)
        target_node = batch["target_node"].to(device)
        target_coords = batch["target_coords"]
        path_coords = batch["path_coords"]

        lengths_cpu = batch.get("lengths")
        lengths = lengths_cpu.to(device) if lengths_cpu is not None else None
        start_coords = batch["start_coords"].to(device) if coordinate_start else None
        outputs = _pathint_forward(
            model,
            start_node=start_node,
            velocities=velocities,
            lengths=lengths,
            start_coords=start_coords,
            prevalidated=prevalidated,
        )
        logits = outputs["node_logits"]
        loss = F.cross_entropy(logits, target_node)
        preds = logits.argmax(dim=-1)
        acc = (preds == target_node).float().mean().item()
        loss_meter.update(loss.item(), start_node.size(0))
        acc_meter.update(acc, start_node.size(0))

        hidden_cpu = outputs["hidden_seq"].cpu()
        coords_cpu = path_coords[:, 1:, :]
        valid_hidden = masked_sequence_rows(hidden_cpu, lengths_cpu)
        valid_coords = masked_sequence_rows(coords_cpu, lengths_cpu)
        all_hidden_seq.append(valid_hidden.numpy())
        all_path_coords.append(valid_coords.numpy())

        all_final_hidden.append(outputs["hidden_final"].cpu().numpy())
        all_final_coords.append(target_coords.numpy())
        all_pred_nodes.append(preds.cpu().numpy())
        all_target_nodes.append(target_node.cpu().numpy())
        all_start_nodes.append(batch["start_node"].cpu().numpy())
        all_l1_distances.append(_compute_l1_distances(batch))

        if position_supervised and "position_pred" in outputs:
            pos_pred = outputs["position_pred"].cpu()
            sq_err_per_sample = (pos_pred - target_coords).pow(2).sum(dim=-1).numpy()
            l2_err_per_sample = np.sqrt(sq_err_per_sample)
            all_pos_sq_errors_per_sample.append(sq_err_per_sample)
            all_pos_l2_errors_per_sample.append(l2_err_per_sample)

    hidden_all = np.concatenate(all_hidden_seq, axis=0)
    coords_all = np.concatenate(all_path_coords, axis=0)
    final_hidden = np.concatenate(all_final_hidden, axis=0)
    final_coords = np.concatenate(all_final_coords, axis=0)
    pred_nodes = np.concatenate(all_pred_nodes, axis=0)
    target_nodes = np.concatenate(all_target_nodes, axis=0)
    start_nodes = np.concatenate(all_start_nodes, axis=0)
    l1_distances = np.concatenate(all_l1_distances, axis=0)

    geometry_corr = 0.0
    if final_hidden.shape[0] >= 4:
        geometry_corr = pairwise_distance_correlation(final_hidden, final_coords)

    geometry_corr_all = 0.0
    if hidden_all.shape[0] >= 4:
        n = hidden_all.shape[0]
        if n > 5000:
            idx = np.random.default_rng(evaluation_seed).choice(n, 5000, replace=False)
            geometry_corr_all = pairwise_distance_correlation(hidden_all[idx], coords_all[idx])
        else:
            geometry_corr_all = pairwise_distance_correlation(hidden_all, coords_all)

    sparsity_index = float((np.abs(hidden_all) < 0.01).mean())
    accuracy_per_sample = (pred_nodes == target_nodes).astype(np.float32)
    accuracy_by_l1_distance = _group_metric_by_l1(accuracy_per_sample, l1_distances)

    result = {
        "loss": loss_meter.avg,
        "accuracy": acc_meter.avg,
        "hidden_all": hidden_all,
        "coords_all": coords_all,
        "final_hidden": final_hidden,
        "final_coords": final_coords,
        "geometry_corr": geometry_corr,
        "geometry_corr_all": geometry_corr_all,
        "sparsity_index": sparsity_index,
        "pred_nodes": pred_nodes,
        "target_nodes": target_nodes,
        "start_nodes": start_nodes,
        "l1_distances": l1_distances,
        "accuracy_by_l1_distance": accuracy_by_l1_distance,
    }
    if all_pos_sq_errors_per_sample:
        pos_sq_errors = np.concatenate(all_pos_sq_errors_per_sample, axis=0)
        pos_l2_errors = np.concatenate(all_pos_l2_errors_per_sample, axis=0)
        # ``position_mse`` is now a true squared-error metric.  The legacy
        # "position error" interpretation is exposed explicitly as L2 error.
        result["position_mse"] = float(np.mean(pos_sq_errors))
        result["position_l2_error"] = float(np.mean(pos_l2_errors))
        result["position_error"] = result["position_l2_error"]  # readable alias
        result["position_mse_by_l1_distance"] = _group_metric_by_l1(pos_sq_errors, l1_distances)
        result["position_error_by_l1_distance"] = _group_metric_by_l1(pos_l2_errors, l1_distances)
    elif getattr(model, "position_head", None) is not None and not position_supervised:
        result["position_mse"] = None
        result["position_l2_error"] = None
        result["position_error"] = None
        result["position_note"] = "position head was not supervised; error is intentionally not reported"

    return result


@torch.no_grad()
def evaluate_cross_graph(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, object]:
    """Evaluate a trained model on a FOREIGN graph via coordinate readout (T14).

    The node-classification head is graph-specific (Linear(hidden, n_nodes)) and
    cannot transfer across graphs with different node counts. cross_entropy on
    foreign target_node ids would also be out-of-range. We therefore measure
    only architecture-invariant, coordinate-based quantities:
      - position_l2_error / position_mse (requires the model's position head),
      - geometry_corr (final-hidden vs final-coord distance correlation),
      - geometry_corr_all (per-step hidden vs per-step coord).
    These test whether the INTEGRATION DYNAMICS generalize across graph
    structure, independent of the node vocabulary. Node accuracy is deliberately
    NOT reported here because it is undefined across mismatched vocabularies.
    """
    model.eval()
    # The model's start_embed is sized to the TRAINING graph's node count. A
    # foreign graph with more nodes would index out of range; fail loud with a
    # clear message rather than a cryptic CUDA/index error (rule #12). The
    # node_head mismatch is handled by simply not using node_logits below.
    embed_n = getattr(getattr(model, "start_embed", None), "num_embeddings", None)
    all_final_hidden: List[np.ndarray] = []
    all_final_coords: List[np.ndarray] = []
    all_hidden_seq: List[np.ndarray] = []
    all_path_coords: List[np.ndarray] = []
    all_pos_sq: List[np.ndarray] = []
    all_pos_l2: List[np.ndarray] = []
    has_position_head = False
    # The bounds guard only applies to embedding-based start init (which
    # indexes start_node). A coordinate-init model has no per-node start lookup,
    # so foreign graphs of ANY size are allowed.
    uses_node_embedding = getattr(model, "start_coord_proj", None) is None

    for batch in loader:
        start_node = batch["start_node"].to(device)
        if uses_node_embedding and embed_n is not None and int(batch["start_node"].max().item()) >= embed_n:
            raise RuntimeError(
                f"Cross-graph eval: foreign start_node id "
                f"{int(batch['start_node'].max().item())} exceeds the model's start "
                f"embedding size ({embed_n}). The foreign graph has more nodes than "
                "the training graph. Either build the model with "
                "start_init='coordinate' (node-count-independent), use a smaller "
                "foreign graph, or restrict start nodes to the trained range."
            )
        velocities = batch["velocities"].to(device)
        target_coords = batch["target_coords"]
        path_coords = batch["path_coords"]
        lengths = batch["lengths"].to(device) if "lengths" in batch else None

        outputs = model(start_node=start_node, velocities=velocities, lengths=lengths,
                        start_coords=batch["start_coords"].to(device))

        all_final_hidden.append(outputs["hidden_final"].cpu().numpy())
        all_final_coords.append(target_coords.numpy())
        hidden_cpu = outputs["hidden_seq"].cpu()
        valid_lengths = lengths.cpu() if lengths is not None else None
        all_hidden_seq.append(masked_sequence_rows(hidden_cpu, valid_lengths).numpy())
        all_path_coords.append(masked_sequence_rows(path_coords[:, 1:, :], valid_lengths).numpy())

        if "position_pred" in outputs:
            has_position_head = True
            pos_pred = outputs["position_pred"].cpu()
            sq = (pos_pred - target_coords).pow(2).sum(dim=-1).numpy()
            all_pos_sq.append(sq)
            all_pos_l2.append(np.sqrt(sq))

    final_hidden = np.concatenate(all_final_hidden, axis=0)
    final_coords = np.concatenate(all_final_coords, axis=0)
    hidden_all = np.concatenate(all_hidden_seq, axis=0)
    coords_all = np.concatenate(all_path_coords, axis=0)

    result: Dict[str, object] = {"cross_graph": True}
    if final_hidden.shape[0] >= 2:
        result["geometry_corr"] = pairwise_distance_correlation(final_hidden, final_coords)
        if hidden_all.shape[0] > 5000:
            idx = np.random.default_rng(0).choice(hidden_all.shape[0], 5000, replace=False)
            result["geometry_corr_all"] = pairwise_distance_correlation(hidden_all[idx], coords_all[idx])
        else:
            result["geometry_corr_all"] = pairwise_distance_correlation(hidden_all, coords_all)
    else:
        result["geometry_corr"] = None
        result["geometry_corr_all"] = None

    if has_position_head:
        pos_sq = np.concatenate(all_pos_sq, axis=0)
        pos_l2 = np.concatenate(all_pos_l2, axis=0)
        result["position_mse"] = float(np.mean(pos_sq))
        result["position_l2_error"] = float(np.mean(pos_l2))
        result["position_error"] = result["position_l2_error"]
    else:
        # No position head -> coordinate error is unavailable; say so (rule #12).
        result["position_error"] = None
        result["position_note"] = "model has no position head; cross-graph coord error unavailable."
    return result


def evaluate_cross_graph_refit(
    model: torch.nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    test_loader: DataLoader,
    n_nodes_target: int,
    device: torch.device,
    epochs: int = 15,
    lr: float = 1e-3,
    threshold: float = 0.8,
) -> Dict[str, object]:
    """Freeze the transferred core and select a fresh target head on validation.

    Target-graph training and model selection use separate loaders. The locked
    head is evaluated on ``test_loader`` exactly once, so test data cannot select
    an epoch or the sample-efficiency threshold crossing.
    """
    import copy

    from gpm_cmap.utils.metrics import compute_sample_efficiency

    if epochs < 1:
        raise ValueError(f"epochs must be positive; got {epochs}")
    uses_node_embedding = getattr(model, "start_coord_proj", None) is None
    embed_n = getattr(getattr(model, "start_embed", None), "num_embeddings", None)

    frozen = copy.deepcopy(model).to(device)
    for parameter in frozen.parameters():
        parameter.requires_grad = False
    frozen.eval()

    head = torch.nn.Linear(model.hidden_dim, n_nodes_target).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)

    def _final_hidden(batch):
        start_node = batch["start_node"].to(device)
        if uses_node_embedding and embed_n is not None and int(start_node.max().item()) >= embed_n:
            raise RuntimeError(
                f"Cross-graph refit: foreign start_node id {int(start_node.max().item())} "
                f"exceeds the model's start embedding size ({embed_n}). Build the "
                "model with start_init='coordinate' for larger foreign graphs."
            )
        velocities = batch["velocities"].to(device)
        lengths = batch["lengths"].to(device) if "lengths" in batch else None
        out = frozen(
            start_node=start_node,
            velocities=velocities,
            lengths=lengths,
            start_coords=batch["start_coords"].to(device),
        )
        return out["hidden_final"]

    def _accuracy(loader: DataLoader) -> float:
        head.eval()
        correct = total = 0
        with torch.no_grad():
            for batch in loader:
                target_node = batch["target_node"].to(device)
                logits = head(_final_hidden(batch))
                correct += (logits.argmax(dim=-1) == target_node).sum().item()
                total += target_node.size(0)
        return correct / max(1, total)

    history: List[dict] = []
    best_validation_acc = -float("inf")
    best_epoch = 0
    best_head_state = None
    for epoch in range(1, epochs + 1):
        head.train()
        for batch in train_loader:
            target_node = batch["target_node"].to(device)
            with torch.no_grad():
                hidden = _final_hidden(batch)
            logits = head(hidden)
            loss = F.cross_entropy(logits, target_node)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        validation_acc = _accuracy(validation_loader)
        history.append({"epoch": epoch, "validation_acc": validation_acc})
        if validation_acc > best_validation_acc:
            best_validation_acc = validation_acc
            best_epoch = epoch
            best_head_state = copy.deepcopy(head.state_dict())

    if best_head_state is None:
        raise RuntimeError("Cross-graph refit produced no validation-selected head")
    head.load_state_dict(best_head_state)
    final_test_acc = _accuracy(test_loader)
    efficiency = compute_sample_efficiency(
        history, metric_key="validation_acc", threshold=threshold
    )
    return {
        "cross_graph_refit": True,
        "best_validation_acc": best_validation_acc,
        "best_epoch": best_epoch,
        "final_test_acc": final_test_acc,
        "epochs_to_threshold": efficiency,
        "threshold": threshold,
        "history": history,
        "selection_split": "validation",
        "test_calls": 1,
    }


def _prepare_rhc_coordinate_codebooks(
    target_encoder: torch.nn.Module,
    node_coordinates,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Build axis codebooks from the actual node-coordinate map.

    Coordinates may be scaled and need not form a complete rectangular product.
    Node IDs are defined by row order (or integer dictionary keys), never by an
    inferred row-major lattice formula.
    """
    if isinstance(node_coordinates, dict):
        expected = list(range(len(node_coordinates)))
        if sorted(node_coordinates) != expected:
            raise ValueError(
                "node_coordinates dictionary keys must be contiguous node ids 0..N-1"
            )
        coordinates = torch.as_tensor(
            [node_coordinates[node_id] for node_id in expected],
            dtype=torch.float32,
            device=device,
        )
    else:
        coordinates = torch.as_tensor(node_coordinates, dtype=torch.float32, device=device)

    if coordinates.ndim != 2 or coordinates.shape[1] != 2 or coordinates.shape[0] < 1:
        raise ValueError(
            "node_coordinates must have shape [n_nodes, 2]; "
            f"got {tuple(coordinates.shape)}"
        )
    if not torch.isfinite(coordinates).all():
        raise ValueError("node_coordinates contain NaN or Inf")
    if torch.unique(coordinates, dim=0).shape[0] != coordinates.shape[0]:
        raise ValueError("node_coordinates must uniquely identify every node")

    axis0_values = torch.unique(coordinates[:, 0], sorted=True)
    axis1_values = torch.unique(coordinates[:, 1], sorted=True)
    axis0_coords = torch.zeros(axis0_values.shape[0], 2, device=device)
    axis1_coords = torch.zeros(axis1_values.shape[0], 2, device=device)
    axis0_coords[:, 0] = axis0_values
    axis1_coords[:, 1] = axis1_values
    return {
        "node_coordinates": coordinates,
        "axis0_values": axis0_values,
        "axis1_values": axis1_values,
        "axis0_codebook": target_encoder(axis0_coords),
        "axis1_codebook": target_encoder(axis1_coords),
    }


def _match_rhc_node_coordinate(
    predicted_coordinate: torch.Tensor,
    node_coordinates: torch.Tensor,
    tolerance: float,
) -> int:
    """Return the node id for an exact coordinate match, or -1 for a lattice hole."""
    distances = torch.linalg.vector_norm(
        node_coordinates - predicted_coordinate.unsqueeze(0), dim=1
    )
    minimum_distance, predicted_node = distances.min(dim=0)
    if float(minimum_distance.item()) > float(tolerance):
        return -1
    return int(predicted_node.item())


@torch.no_grad()
def evaluate_pathint_rhc(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    node_coordinates,
    rhc_moduli: list | None = None,
) -> Dict[str, object]:
    """Evaluate path integration with an RHC decoder tied to real node coordinates."""
    from gpm_cmap.models.resonator_network import ResonatorNetwork
    from gpm_cmap.models.rhc_encoder import MultiDimResidueEncoder

    model.eval()
    if rhc_moduli is None:
        rhc_moduli = [3, 5, 7]
    if model.periodic_encoder is None:
        return {"error": "model has no periodic_encoder — cannot run RHC evaluation"}

    phasor_dim = model.periodic_encoder.output_dim // 2
    if model.target_rhc_encoder is not None:
        target_encoder = model.target_rhc_encoder.to(device)
    else:
        target_encoder = MultiDimResidueEncoder(
            n_dims=2,
            phasor_dim=phasor_dim,
            moduli=rhc_moduli,
        ).to(device)

    codebooks = _prepare_rhc_coordinate_codebooks(
        target_encoder, node_coordinates, device
    )
    coordinates = codebooks["node_coordinates"]
    if coordinates.shape[0] != int(model.n_nodes):
        raise ValueError(
            f"node coordinate count ({coordinates.shape[0]}) does not match "
            f"model.n_nodes ({model.n_nodes})"
        )
    axis0_values = codebooks["axis0_values"]
    axis1_values = codebooks["axis1_values"]
    axis0_codebook = codebooks["axis0_codebook"]
    axis1_codebook = codebooks["axis1_codebook"]
    resonator = ResonatorNetwork(max_iterations=50, convergence_threshold=0.95)

    res_acc = AverageMeter()
    lin_acc = AverageMeter()
    ax0_acc = AverageMeter()
    ax1_acc = AverageMeter()
    invalid_coord = AverageMeter()
    conv_meter = AverageMeter()
    iter_meter = AverageMeter()
    align_meter = AverageMeter()

    coordinate_scale = max(1.0, float(coordinates.abs().max().item()))
    tolerance = 1e-6 * coordinate_scale
    for batch in loader:
        start_node = batch["start_node"].to(device)
        velocities = batch["velocities"].to(device)
        target_node = batch["target_node"].to(device)
        target_coords = batch["target_coords"].to(device)
        expected_coords = coordinates[target_node]
        if not torch.allclose(target_coords, expected_coords, atol=tolerance, rtol=1e-5):
            raise ValueError(
                "Batch target_coords do not match node_coordinates for target_node; "
                "coordinate scale or node-id mapping is inconsistent."
            )

        lengths = batch["lengths"].to(device) if "lengths" in batch else None
        outputs = model(
            start_node=start_node,
            velocities=velocities,
            lengths=lengths,
            start_coords=batch["start_coords"].to(device),
        )
        linear_predictions = outputs["node_logits"].argmax(dim=-1)
        lin_acc.update(
            (linear_predictions == target_node).float().mean().item(), start_node.size(0)
        )
        if "periodic_seq" not in outputs:
            continue

        z_final = gather_final_step(outputs["periodic_seq"], lengths)
        if "path_coords" in batch:
            path_coords = batch["path_coords"].to(device)
            z_target = model.compute_rhc_target(path_coords)
            if z_target is not None:
                alignment = masked_mse(outputs["periodic_seq"], z_target, lengths).item()
                align_meter.update(alignment, start_node.size(0))

        for batch_index in range(z_final.shape[0]):
            result = resonator.factorize(
                z_final[batch_index], [axis0_codebook, axis1_codebook]
            )
            pred_axis0_index = int(result["factor_indices"][0].item())
            pred_axis1_index = int(result["factor_indices"][1].item())
            predicted_coordinate = torch.stack([
                axis0_values[pred_axis0_index],
                axis1_values[pred_axis1_index],
            ])
            predicted_node = _match_rhc_node_coordinate(
                predicted_coordinate, coordinates, tolerance
            )
            is_valid_coordinate = predicted_node >= 0

            true_coordinate = target_coords[batch_index]
            true_axis0_index = int(
                torch.argmin(torch.abs(axis0_values - true_coordinate[0])).item()
            )
            true_axis1_index = int(
                torch.argmin(torch.abs(axis1_values - true_coordinate[1])).item()
            )
            true_node = int(target_node[batch_index].item())

            res_acc.update(float(predicted_node == true_node))
            ax0_acc.update(float(pred_axis0_index == true_axis0_index))
            ax1_acc.update(float(pred_axis1_index == true_axis1_index))
            invalid_coord.update(float(not is_valid_coordinate))
            conv_meter.update(float(bool(result["converged"].item())))
            iter_meter.update(float(result["n_iterations"]))

    return {
        "resonator_accuracy": res_acc.avg,
        "resonator_axis0_acc": ax0_acc.avg,
        "resonator_axis1_acc": ax1_acc.avg,
        "invalid_coordinate_rate": invalid_coord.avg,
        "linear_accuracy": lin_acc.avg,
        "convergence_rate": conv_meter.avg,
        "avg_iterations": iter_meter.avg,
        "rhc_alignment": align_meter.avg,
        "coordinate_codebook": "actual_node_coordinates",
        "axis0_values": axis0_values.detach().cpu().tolist(),
        "axis1_values": axis1_values.detach().cpu().tolist(),
    }
