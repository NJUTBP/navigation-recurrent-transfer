from __future__ import annotations

from typing import Any

import torch
from torch import nn


class CategoricalContentEncoder(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x)


class VectorMLPEncoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SmallCNNEncoder(nn.Module):
    def __init__(self, output_dim: int, in_channels: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
            nn.Linear(32 * 2 * 2, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class IdentityEncoder(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def build_content_encoder(kind: str, **kwargs: Any) -> nn.Module:
    if kind == "categorical":
        return CategoricalContentEncoder(kwargs["vocab_size"], kwargs["embedding_dim"])
    if kind == "vector":
        return VectorMLPEncoder(kwargs["input_dim"], kwargs["output_dim"], kwargs.get("hidden_dim", 128))
    if kind == "image":
        return SmallCNNEncoder(kwargs["output_dim"], kwargs.get("in_channels", 1))
    if kind == "identity":
        return IdentityEncoder()
    raise ValueError(f"Unknown content encoder kind: {kind}")
