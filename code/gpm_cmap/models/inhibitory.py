from __future__ import annotations

import torch
from torch import nn


def decorrelation_loss(hidden_activations: torch.Tensor) -> torch.Tensor:
    if hidden_activations.ndim == 3:
        hidden_activations = hidden_activations.reshape(-1, hidden_activations.shape[-1])
    h = hidden_activations - hidden_activations.mean(dim=0, keepdim=True)
    cov = (h.t() @ h) / max(1, h.shape[0] - 1)
    eye = torch.eye(cov.shape[0], device=cov.device, dtype=cov.dtype)
    return ((cov * (1 - eye)) ** 2).mean()


def sparsity_loss(hidden_activations: torch.Tensor, target_sparsity: float = 0.05) -> torch.Tensor:
    if hidden_activations.ndim == 3:
        hidden_activations = hidden_activations.reshape(-1, hidden_activations.shape[-1])
    avg_activation = hidden_activations.mean(dim=0).clamp(1e-6, 1 - 1e-6)
    p = torch.full_like(avg_activation, fill_value=target_sparsity)
    kl = p * torch.log(p / avg_activation) + (1 - p) * torch.log((1 - p) / (1 - avg_activation))
    return kl.mean()


class LateralInhibition(nn.Module):
    def __init__(self, k_winners: int = 10) -> None:
        super().__init__()
        self.k_winners = k_winners

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        topk_vals, topk_idx = h.topk(self.k_winners, dim=-1)
        mask = torch.zeros_like(h)
        mask.scatter_(-1, topk_idx, 1.0)
        return h * mask
