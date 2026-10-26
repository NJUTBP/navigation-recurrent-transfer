"""Training and evaluation loops for pairwise comparison tasks.

Works with GPMCMap **and** all baseline models – the only requirement is that
``model.forward(actions, observations)`` returns a dict containing
``features`` of shape ``(B, S, D)``.
"""
from __future__ import annotations
from collections import defaultdict
from typing import Dict, List
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from gpm_cmap.utils.metrics import AverageMeter


# -----------------------------------------------------------------------
#  Comparison head  (lives outside the main model so it can be shared)
# -----------------------------------------------------------------------

class ComparisonHead(nn.Module):
    """Binary classifier on a pair of feature vectors.

    forward(feat_a, feat_b) -> logit (scalar per sample).
    """

    def __init__(self, feature_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, feat_a: torch.Tensor, feat_b: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([feat_a, feat_b], dim=-1)).squeeze(-1)


# -----------------------------------------------------------------------
#  Training
# -----------------------------------------------------------------------

def train_comparison_epoch(
    model: nn.Module,
    comparison_head: ComparisonHead,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float = 1.0,
) -> Dict[str, float]:
    model.train()
    comparison_head.train()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    for batch in loader:
        actions = batch["actions"].to(device)
        observations = batch["observations"].to(device)
        labels = batch["label"].to(device)

        outputs = model(actions=actions, observations=observations, reset_memory=True)
        feat_a = outputs["features"][:, 0]   # representation of node A
        feat_b = outputs["features"][:, 1]   # representation of node B
        logit = comparison_head(feat_a, feat_b)
        loss = F.binary_cross_entropy_with_logits(logit, labels)

        optimizer.zero_grad()
        loss.backward()
        all_params = list(model.parameters()) + list(comparison_head.parameters())
        torch.nn.utils.clip_grad_norm_(all_params, grad_clip)
        optimizer.step()

        preds = (logit > 0.0).float()
        acc = (preds == labels).float().mean().item()
        loss_meter.update(loss.item(), actions.size(0))
        acc_meter.update(acc, actions.size(0))

    return {"loss": loss_meter.avg, "accuracy": acc_meter.avg}


# -----------------------------------------------------------------------
#  Evaluation  (also computes per-distance accuracy for symbolic distance)
# -----------------------------------------------------------------------

@torch.no_grad()
def evaluate_comparison(
    model: nn.Module,
    comparison_head: ComparisonHead,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, object]:
    model.eval()
    comparison_head.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    # per-distance bookkeeping (for symbolic distance effect)
    dist_correct: Dict[float, List[float]] = defaultdict(list)
    all_features_a, all_features_b = [], []

    for batch in loader:
        actions = batch["actions"].to(device)
        observations = batch["observations"].to(device)
        labels = batch["label"].to(device)
        distances = batch["distance"].cpu().numpy()

        outputs = model(actions=actions, observations=observations, reset_memory=True)
        feat_a = outputs["features"][:, 0]
        feat_b = outputs["features"][:, 1]
        logit = comparison_head(feat_a, feat_b)
        loss = F.binary_cross_entropy_with_logits(logit, labels)

        preds = (logit > 0.0).float()
        correct = (preds == labels).cpu().numpy()

        loss_meter.update(loss.item(), actions.size(0))
        acc_meter.update(correct.mean(), actions.size(0))

        for d, c in zip(distances, correct):
            dist_correct[round(float(d), 2)].append(float(c))

        all_features_a.append(feat_a.cpu().numpy())
        all_features_b.append(feat_b.cpu().numpy())

    # symbolic distance curve: distance -> accuracy
    distance_accuracy = {
        d: float(np.mean(vals)) for d, vals in sorted(dist_correct.items())
    }

    return {
        "loss": loss_meter.avg,
        "accuracy": acc_meter.avg,
        "distance_accuracy": distance_accuracy,
        "features_a": np.concatenate(all_features_a, axis=0),
        "features_b": np.concatenate(all_features_b, axis=0),
    }
