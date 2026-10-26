from __future__ import annotations

import torch
from torch import nn


class ConjunctiveLayer(nn.Module):
    def __init__(self, grid_dim: int, content_dim: int, output_dim: int, mode: str = "concat") -> None:
        super().__init__()
        self.mode = mode
        if mode == "concat":
            self.proj = nn.Sequential(
                nn.Linear(grid_dim + content_dim, output_dim),
                nn.ReLU(),
            )
        elif mode == "tensor":
            self.proj = nn.Sequential(
                nn.Linear(grid_dim * content_dim, output_dim),
                nn.ReLU(),
            )
        else:
            raise ValueError(f"Unsupported conjunction mode: {mode}")

    def forward(self, g_t: torch.Tensor, e_t: torch.Tensor) -> torch.Tensor:
        if self.mode == "concat":
            x = torch.cat([g_t, e_t], dim=-1)
        else:
            x = torch.einsum("bi,bj->bij", g_t, e_t).reshape(g_t.size(0), -1)
        return self.proj(x)
