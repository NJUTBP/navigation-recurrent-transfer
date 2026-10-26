"""Shared training / evaluation for abstract graph-walk tasks.

Changes vs original
-------------------
- evaluate_abstract computes **hex modulation** on hidden units using
  movement directions in abstract coordinate space.
- geometry_corr now remains backward-compatible and equals the periodic-space
  correlation, while geometry_corr_hidden is reported in parallel.
"""
from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Dict, List
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from gpm_cmap.analysis.hex_modulation import batch_hex_modulation
from gpm_cmap.analysis.repr_geometry import pairwise_distance_correlation
from gpm_cmap.models.inhibitory import decorrelation_loss, sparsity_loss
from gpm_cmap.utils.metrics import AverageMeter



def _sample_fingerprint(sample: Dict[str, torch.Tensor]) -> str:
    """Stable identity for an entire graph-walk episode."""
    digest = hashlib.sha256()
    for key in ("observations", "actions", "targets", "valid_mask"):
        value = sample.get(key)
        if value is None:
            continue
        array = value.detach().cpu().contiguous().numpy()
        digest.update(key.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _grouped_partition_indices(dataset, ratios: List[float], seed: int) -> List[List[int]]:
    if not ratios or any(r <= 0 for r in ratios):
        raise ValueError(f"All split ratios must be positive; got {ratios}")
    if not np.isclose(sum(ratios), 1.0):
        raise ValueError(f"Split ratios must sum to 1; got {ratios}")
    groups: Dict[str, List[int]] = defaultdict(list)
    for idx in range(len(dataset)):
        groups[_sample_fingerprint(dataset[idx])].append(idx)
    rng = np.random.default_rng(seed)
    grouped = list(groups.values())
    rng.shuffle(grouped)
    targets = np.asarray(ratios, dtype=float) * len(dataset)
    partitions: List[List[int]] = [[] for _ in ratios]
    counts = np.zeros(len(ratios), dtype=float)
    for group in sorted(grouped, key=len, reverse=True):
        deficits = targets - counts
        candidates = np.flatnonzero(deficits == deficits.max())
        chosen = int(rng.choice(candidates))
        partitions[chosen].extend(group)
        counts[chosen] += len(group)
    if any(not part for part in partitions):
        raise RuntimeError(
            f"Grouped split produced an empty partition with sizes {[len(p) for p in partitions]}"
        )
    fingerprints = [
        {_sample_fingerprint(dataset[idx]) for idx in partition}
        for partition in partitions
    ]
    for i in range(len(fingerprints)):
        for j in range(i + 1, len(fingerprints)):
            if fingerprints[i] & fingerprints[j]:
                raise RuntimeError("Exact graph-walk episode leakage across splits")
    return partitions


def split_dataset(dataset, split_ratio: float = 0.8, seed: int = 42):
    if not 0.0 < split_ratio < 1.0:
        raise ValueError(f"split_ratio must lie in (0,1); got {split_ratio}")
    train_idx, test_idx = _grouped_partition_indices(
        dataset, [float(split_ratio), 1.0 - float(split_ratio)], seed
    )
    return Subset(dataset, train_idx), Subset(dataset, test_idx)


def split_dataset_three_way(
    dataset,
    train_ratio: float = 0.7,
    validation_ratio: float = 0.1,
    seed: int = 42,
):
    if train_ratio <= 0 or validation_ratio <= 0 or train_ratio + validation_ratio >= 1:
        raise ValueError(
            "train_ratio and validation_ratio must be positive and sum to less than 1"
        )
    train_idx, validation_idx, test_idx = _grouped_partition_indices(
        dataset,
        [train_ratio, validation_ratio, 1.0 - train_ratio - validation_ratio],
        seed,
    )
    return (
        Subset(dataset, train_idx),
        Subset(dataset, validation_idx),
        Subset(dataset, test_idx),
    )


def _valid_mask(batch: Dict[str, torch.Tensor], reference: torch.Tensor) -> torch.Tensor:
    mask = batch.get("valid_mask")
    if mask is None:
        return torch.ones(reference.shape[:2], dtype=torch.bool, device=reference.device)
    mask = mask.to(reference.device, dtype=torch.bool)
    if mask.shape != reference.shape[:2]:
        raise ValueError(
            f"valid_mask shape {tuple(mask.shape)} does not match sequence "
            f"shape {tuple(reference.shape[:2])}"
        )
    if not mask.any():
        raise ValueError("valid_mask contains no supervised steps")
    return mask



def train_abstract_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    decorr_weight: float = 0.01,
    sparse_weight: float = 0.001,
    grad_clip: float = 1.0,
    nav_loss_weight: float = 0.0,
    periodic_loss_weight: float = 0.0,
    epoch: int = 0,
    use_plasticity_schedule: bool = False,
) -> Dict[str, float]:
    model.train()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    periodic_meter = AverageMeter()

    if use_plasticity_schedule:
        if epoch < 50:
            eff_decorr, eff_sparse = 0.0, 0.0
        elif epoch < 100:
            frac = (epoch - 50) / 50.0
            eff_decorr = decorr_weight * frac
            eff_sparse = sparse_weight * frac
        else:
            eff_decorr, eff_sparse = decorr_weight, sparse_weight
    else:
        eff_decorr, eff_sparse = decorr_weight, sparse_weight

    for batch in loader:
        actions = batch["actions"].to(device)
        observations = batch["observations"].to(device)
        targets = batch["targets"].to(device)
        outputs = model(actions=actions, observations=observations, reset_memory=True)
        logits = outputs["node_logits"]
        mask = _valid_mask(batch, logits)
        task_loss = F.cross_entropy(logits[mask], targets[mask])
        rec_loss = F.mse_loss(
            outputs["content_reconstruction"][mask], outputs["content_seq"].detach()[mask]
        )
        loss = task_loss + 0.1 * rec_loss
        if eff_decorr > 0:
            loss = loss + eff_decorr * decorrelation_loss(outputs["hidden_seq"][mask])
        if eff_sparse > 0:
            # Differentiable "activation probability" surrogate.
            # The previous code used a hard threshold (hidden > 0.01).float(),
            # whose boolean->float cast has zero gradient almost everywhere, so
            # the KL sparsity penalty added a constant to the loss WITHOUT ever
            # pressuring hidden_seq toward sparsity. We instead map each ReLU
            # activation into (0, 1) with tanh, giving a smooth per-unit "on"
            # probability that (a) is differentiable everywhere and (b) lives in
            # the (0,1) domain expected by sparsity_loss's KL term.
            rates = outputs["hidden_seq"][mask].clamp_min(0.0)
            hidden_active = torch.tanh(rates)
            # [C10] The tanh(relu(h)) surrogate has vanishing gradient for
            # strongly-firing units (d/dh = 1 - tanh(h)^2 -> 0 as h grows), so
            # the KL term exerts almost no pressure on exactly the high-firing
            # units. Add a small non-saturating L1-on-rates term so sparsity
            # pressure also reaches those units. Coefficient 0.1 keeps the KL
            # term dominant; expose as a config knob if finer control is needed.
            loss = loss + eff_sparse * (sparsity_loss(hidden_active) + 0.1 * rates.mean())

        periodic_loss_val = 0.0
        if periodic_loss_weight > 0 and "periodic_target" in outputs:
            periodic_loss = F.mse_loss(
                outputs["periodic_seq"][mask], outputs["periodic_target"].detach()[mask]
            )
            loss = loss + periodic_loss_weight * periodic_loss
            periodic_loss_val = periodic_loss.item()

        if nav_loss_weight > 0 and "coordinates" in batch:
            coords = batch["coordinates"].to(device)
            periodic = outputs["periodic_seq"]
            # The aux head must be created at setup time (model.build_nav_aux_head)
            # so the optimizer captures its parameters. Creating it here — after
            # the optimizer was built — was the original bug: the head trained
            # nothing. Fail loud rather than silently re-introduce that (rule #12).
            if not hasattr(model, "_nav_aux_head"):
                raise RuntimeError(
                    "nav_loss_weight > 0 but model._nav_aux_head is missing. "
                    "Call model.build_nav_aux_head(coord_dim=...) during setup, "
                    "before constructing the optimizer."
                )
            coord_pred = model._nav_aux_head(periodic)
            loss = loss + nav_loss_weight * F.mse_loss(coord_pred[mask], coords[mask])

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        preds = logits.argmax(dim=-1)
        acc = (preds[mask] == targets[mask]).float().mean().item()
        valid_count = int(mask.sum().item())
        loss_meter.update(loss.item(), valid_count)
        acc_meter.update(acc, valid_count)
        periodic_meter.update(periodic_loss_val, actions.size(0))
    return {
        "loss": loss_meter.avg,
        "accuracy": acc_meter.avg,
        "periodic_loss": periodic_meter.avg,
    }


@torch.no_grad()
def evaluate_abstract(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, object]:
    model.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    rec_meter = AverageMeter()
    all_periodic = []
    all_coords = []
    all_hidden = []
    all_hidden_flat = []
    all_directions = []

    for batch in loader:
        actions = batch["actions"].to(device)
        observations = batch["observations"].to(device)
        targets = batch["targets"].to(device)
        coordinates = batch["coordinates"].cpu().numpy()

        outputs = model(actions=actions, observations=observations, reset_memory=True)
        logits = outputs["node_logits"]
        mask = _valid_mask(batch, logits)
        task_loss = F.cross_entropy(logits[mask], targets[mask])
        preds = logits.argmax(dim=-1)
        acc = (preds[mask] == targets[mask]).float().mean().item()
        valid_count = int(mask.sum().item())
        loss_meter.update(task_loss.item(), valid_count)
        acc_meter.update(acc, valid_count)

        rec_mse = F.mse_loss(
            outputs["content_reconstruction"][mask], outputs["content_seq"][mask]
        ).item()
        rec_meter.update(rec_mse, valid_count)

        mask_cpu = mask.cpu()
        all_periodic.append(outputs["periodic_seq"][mask].cpu().numpy())
        all_coords.append(coordinates[mask_cpu.numpy()])

        hidden_cpu = outputs["hidden_seq"].cpu()
        hidden_np = hidden_cpu.numpy()
        all_hidden_flat.append(hidden_cpu[mask_cpu].numpy())
        B, S, H = hidden_np.shape
        if S >= 2 and coordinates.shape[1] >= 2:
            transition_mask = mask_cpu[:, :-1] & mask_cpu[:, 1:]
            dx = coordinates[:, 1:, 0] - coordinates[:, :-1, 0]
            dy = coordinates[:, 1:, 1] - coordinates[:, :-1, 1]
            dirs = np.arctan2(dy, dx)
            hid_aligned = hidden_np[:, :-1, :]
            if transition_mask.any():
                all_directions.append(dirs[transition_mask.numpy()])
                all_hidden.append(hid_aligned[transition_mask.numpy()])

    periodic_features = np.concatenate(all_periodic, axis=0)
    coords = np.concatenate(all_coords, axis=0)
    hidden_flat = np.concatenate(all_hidden_flat, axis=0)
    geometry_corr_periodic = pairwise_distance_correlation(periodic_features, coords)
    geometry_corr_hidden = pairwise_distance_correlation(hidden_flat, coords)

    sparsity_index = float((np.abs(hidden_flat) < 0.01).mean())

    hex_results: Dict[str, object] = {}
    if all_directions:
        directions_cat = np.concatenate(all_directions, axis=0)
        hidden_cat = np.concatenate(all_hidden, axis=0)
        if np.unique(directions_cat).size >= 4:
            hex_results = batch_hex_modulation(directions_cat, hidden_cat, top_k=20)

    return {
        "loss": loss_meter.avg,
        "accuracy": acc_meter.avg,
        "features": periodic_features,
        "coords": coords,
        "hidden": hidden_flat,
        "geometry_corr": geometry_corr_periodic,
        "geometry_corr_periodic": geometry_corr_periodic,
        "geometry_corr_hidden": geometry_corr_hidden,
        "reconstruction_mse": rec_meter.avg,
        "sparsity_index": sparsity_index,
        "hex_modulation": hex_results,
    }
