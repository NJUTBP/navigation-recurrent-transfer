"""Few-shot visual concept evaluation.

Tests whether GPM-CMap's Hebbian memory enables rapid binding of visual
patterns to abstract positions, outperforming standard baselines.

The protocol:
1. Support phase: present K examples per class, write to Hebb memory.
2. Query phase:   classify new examples by reading from memory.
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from gpm_cmap.utils.metrics import AverageMeter


def _episode_tensor(x: torch.Tensor) -> torch.Tensor:
    """Convert episode tensors from DataLoader(batch_size=1) to per-episode tensors.

    Expected inputs are either:
      - already episode-shaped: (N, C, H, W) or (N,)
      - dataloader batched:      (1, N, C, H, W) or (1, N)
    """
    if x.ndim >= 2 and x.size(0) == 1:
        return x.squeeze(0)
    return x


def _episode_class_keys(
    n_classes: int,
    key_dim: int,
    device: torch.device,
) -> torch.Tensor:
    """Create deterministic per-class keys for episodic Hebbian binding."""
    if key_dim >= n_classes:
        keys = torch.zeros(n_classes, key_dim, device=device)
        keys[torch.arange(n_classes, device=device), torch.arange(n_classes, device=device)] = 1.0
        return keys

    generator = torch.Generator(device=device if device.type != "mps" else "cpu")
    generator.manual_seed(0)
    keys = torch.randn(n_classes, key_dim, generator=generator, device=device)
    return F.normalize(keys, dim=-1)


@torch.no_grad()
def evaluate_few_shot_gpm(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    n_way: int = 5,
) -> Dict[str, float]:
    """Evaluate GPM-CMap on few-shot episodes.

    For real GPMCMap models we bind support prototypes directly into the
    Hebbian memory using deterministic episode-specific class keys. This
    avoids the old dummy-action path, whose identical action stream made
    the memory keys depend on time index rather than class identity.

    A legacy fallback is kept for lightweight mock models used in tests.
    """
    model.eval()
    acc_meter = AverageMeter()

    for batch in loader:
        support_imgs = _episode_tensor(batch["support_images"]).to(device)
        support_labels = _episode_tensor(batch["support_labels"]).to(device)
        query_imgs = _episode_tensor(batch["query_images"]).to(device)
        query_labels = _episode_tensor(batch["query_labels"]).to(device)
        n_query = query_imgs.size(0)

        can_use_direct_memory = all(
            hasattr(model, attr) for attr in ["content_encoder", "memory", "reset_memory"]
        )

        if can_use_direct_memory:
            support_emb = model.content_encoder(support_imgs)
            query_emb = model.content_encoder(query_imgs)

            unique_labels = torch.unique(support_labels, sorted=True)
            class_keys = _episode_class_keys(
                n_classes=unique_labels.numel(),
                key_dim=model.memory.key_dim,
                device=device,
            )

            model.reset_memory(1, device)
            for idx, cls in enumerate(unique_labels):
                prototype = support_emb[support_labels == cls].mean(dim=0, keepdim=True)
                model.memory.write(class_keys[idx:idx + 1], prototype)

            retrieved = torch.cat(
                [model.memory.read(class_keys[idx:idx + 1]) for idx in range(unique_labels.numel())],
                dim=0,
            )
            dists = torch.cdist(query_emb, retrieved, p=2)
            preds = dists.argmin(dim=-1)
            pred_labels = unique_labels[preds]
        else:
            # Backward-compatible fallback for simplified fake models used in tests.
            n_support = support_imgs.size(0)
            support_emb = model.content_encoder(support_imgs)
            unique_labels = torch.unique(support_labels)
            prototypes = []
            for c in unique_labels:
                mask = support_labels == c
                prototypes.append(support_emb[mask].mean(dim=0))
            prototypes = torch.stack(prototypes)

            dummy_actions = torch.zeros(1, n_support, dtype=torch.long, device=device)
            support_obs = support_imgs.unsqueeze(0)
            model.reset_memory(1, device)
            model(actions=dummy_actions, observations=support_obs, reset_memory=False)

            dummy_q_actions = torch.zeros(1, n_query, dtype=torch.long, device=device)
            query_obs = query_imgs.unsqueeze(0)
            q_outputs = model(actions=dummy_q_actions, observations=query_obs, reset_memory=False)
            retrievals = q_outputs["retrievals"].squeeze(0)

            dists = torch.cdist(retrievals, prototypes, p=2)
            preds = dists.argmin(dim=-1)
            pred_labels = unique_labels[preds]

        acc = (pred_labels == query_labels).float().mean().item()
        acc_meter.update(acc, n_query)

    return {
        "accuracy": acc_meter.avg,
        "one_shot_binding_acc": acc_meter.avg,
    }


@torch.no_grad()
def evaluate_few_shot_gpm_no_memory(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate the same GPM content encoder without Hebb memory.

    Support and query images are embedded with the model's content encoder,
    then classified by nearest centroid in embedding space.
    """
    model.eval()
    acc_meter = AverageMeter()

    for batch in loader:
        support_imgs = _episode_tensor(batch["support_images"]).to(device)
        support_labels = _episode_tensor(batch["support_labels"]).to(device)
        query_imgs = _episode_tensor(batch["query_images"]).to(device)
        query_labels = _episode_tensor(batch["query_labels"]).to(device)

        support_emb = model.content_encoder(support_imgs)
        query_emb = model.content_encoder(query_imgs)

        unique_labels = torch.unique(support_labels, sorted=True)
        prototypes = []
        for cls in unique_labels:
            prototype = support_emb[support_labels == cls].mean(dim=0)
            prototypes.append(prototype)
        prototypes = torch.stack(prototypes, dim=0)

        dists = torch.cdist(query_emb, prototypes, p=2)
        preds = dists.argmin(dim=-1)
        pred_labels = unique_labels[preds]

        acc = (pred_labels == query_labels).float().mean().item()
        acc_meter.update(acc, query_imgs.size(0))

    return {"accuracy": acc_meter.avg}


@torch.no_grad()
def evaluate_few_shot_protonet(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate Prototypical Network on few-shot episodes."""
    model.eval()
    acc_meter = AverageMeter()

    for batch in loader:
        support_imgs = _episode_tensor(batch["support_images"]).to(device)
        support_labels = _episode_tensor(batch["support_labels"]).to(device)
        query_imgs = _episode_tensor(batch["query_images"]).to(device)
        query_labels = _episode_tensor(batch["query_labels"]).to(device)

        logits = model(support_imgs, support_labels, query_imgs)
        preds = logits.argmax(dim=-1)
        unique_labels = torch.unique(support_labels)
        pred_labels = unique_labels[preds]
        acc = (pred_labels == query_labels).float().mean().item()
        acc_meter.update(acc, query_imgs.size(0))

    return {"accuracy": acc_meter.avg}


def evaluate_few_shot_cnn(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    n_way: int = 5,
) -> Dict[str, float]:
    """Evaluate CNN baseline on few-shot episodes via nearest-centroid."""
    model.eval()
    acc_meter = AverageMeter()

    with torch.no_grad():
        for batch in loader:
            support_imgs = _episode_tensor(batch["support_images"]).to(device)
            support_labels = _episode_tensor(batch["support_labels"]).to(device)
            query_imgs = _episode_tensor(batch["query_images"]).to(device)
            query_labels = _episode_tensor(batch["query_labels"]).to(device)

            support_feat = model.get_features(support_imgs)
            query_feat = model.get_features(query_imgs)

            unique_labels = torch.unique(support_labels)
            prototypes = []
            for c in unique_labels:
                mask = support_labels == c
                prototypes.append(support_feat[mask].mean(dim=0))
            prototypes = torch.stack(prototypes)

            dists = torch.cdist(query_feat, prototypes, p=2)
            preds = dists.argmin(dim=-1)
            pred_labels = unique_labels[preds]
            acc = (pred_labels == query_labels).float().mean().item()
            acc_meter.update(acc, query_imgs.size(0))

    return {"accuracy": acc_meter.avg}
