"""Baseline models for comparison with GPM-CMap.

Every baseline exposes the **same forward signature and output dict keys** as
:class:`GPMCMap` so that the training / evaluation loops in
``tasks.abstract_common`` and ``tasks.comparison_common`` work unchanged.

Required output keys
--------------------
- ``hidden_seq``   : (B, S, hidden_dim)
- ``features``     : (B, S, hidden_dim)   ← used by ComparisonHead
- ``node_logits``  : (B, S, node_vocab)   ← used by graph-walk CE loss
- ``content_seq``  : (B, S, content_dim)  ← used by reconstruction loss
- ``content_reconstruction`` : (B, S, content_dim)
- ``periodic_seq`` : (B, S, hidden_dim)   ← used by geometry analysis
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn


# ---------------------------------------------------------------------------
#  Plain GRU baseline
# ---------------------------------------------------------------------------

class PlainGRUBaseline(nn.Module):
    """Vanilla GRU – no periodic encoding, no Hebb memory, no conjunctive."""

    def __init__(
        self,
        action_vocab: int,
        action_embed_dim: int,
        hidden_dim: int,
        content_dim: int,
        observation_vocab: int,
        node_vocab: int,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.content_dim = content_dim
        self.action_encoder = nn.Embedding(action_vocab, action_embed_dim)
        self.content_encoder = nn.Embedding(observation_vocab, content_dim)
        self.gru = nn.GRU(action_embed_dim + content_dim, hidden_dim, batch_first=True)
        self.node_head = nn.Linear(hidden_dim, node_vocab)
        self.reconstruction_head = nn.Linear(hidden_dim, content_dim)

    # -- public helpers expected by experiment scripts ----------------------
    def freeze_grid_rnn(self) -> None:  # no-op for baselines
        pass

    def reset_memory(self, batch_size: int, device) -> None:
        pass

    def forward(
        self,
        actions: torch.Tensor,
        observations: torch.Tensor,
        h0: Optional[torch.Tensor] = None,
        reset_memory: bool = True,
    ) -> Dict[str, torch.Tensor]:
        act_emb = self.action_encoder(actions)
        obs_emb = self.content_encoder(observations)
        inp = torch.cat([act_emb, obs_emb], dim=-1)
        hidden_seq, _ = self.gru(inp, h0.unsqueeze(0) if h0 is not None else None)
        return {
            "hidden_seq": hidden_seq,
            "features": hidden_seq,
            "periodic_seq": hidden_seq,           # geometry analysis placeholder
            "content_seq": obs_emb,
            "content_reconstruction": self.reconstruction_head(hidden_seq),
            "node_logits": self.node_head(hidden_seq),
        }


# ---------------------------------------------------------------------------
#  Transformer baseline
# ---------------------------------------------------------------------------

class _PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: d_model // 2])  # handle odd d_model
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerBaseline(nn.Module):
    """Small causal Transformer – attends to full context."""

    def __init__(
        self,
        action_vocab: int,
        action_embed_dim: int,
        hidden_dim: int,
        content_dim: int,
        observation_vocab: int,
        node_vocab: int,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.content_dim = content_dim
        self.action_encoder = nn.Embedding(action_vocab, action_embed_dim)
        self.content_encoder = nn.Embedding(observation_vocab, content_dim)
        self.input_proj = nn.Linear(action_embed_dim + content_dim, hidden_dim)
        self.pos_enc = _PositionalEncoding(hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim * 4,
            dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.node_head = nn.Linear(hidden_dim, node_vocab)
        self.reconstruction_head = nn.Linear(hidden_dim, content_dim)

    def freeze_grid_rnn(self) -> None:
        pass

    def reset_memory(self, batch_size: int, device) -> None:
        pass

    def forward(
        self,
        actions: torch.Tensor,
        observations: torch.Tensor,
        h0: Optional[torch.Tensor] = None,
        reset_memory: bool = True,
    ) -> Dict[str, torch.Tensor]:
        act_emb = self.action_encoder(actions)
        obs_emb = self.content_encoder(observations)
        x = self.input_proj(torch.cat([act_emb, obs_emb], dim=-1))
        x = self.pos_enc(x)
        seq_len = x.size(1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=x.device)
        hidden_seq = self.transformer(x, mask=causal_mask, is_causal=True)
        return {
            "hidden_seq": hidden_seq,
            "features": hidden_seq,
            "periodic_seq": hidden_seq,
            "content_seq": obs_emb,
            "content_reconstruction": self.reconstruction_head(hidden_seq),
            "node_logits": self.node_head(hidden_seq),
        }


# ---------------------------------------------------------------------------
#  MLP baseline (non-sequential)
# ---------------------------------------------------------------------------

class MLPBaseline(nn.Module):
    """Per-step MLP – no temporal modelling at all."""

    def __init__(
        self,
        action_vocab: int,
        action_embed_dim: int,
        hidden_dim: int,
        content_dim: int,
        observation_vocab: int,
        node_vocab: int,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.content_dim = content_dim
        self.action_encoder = nn.Embedding(action_vocab, action_embed_dim)
        self.content_encoder = nn.Embedding(observation_vocab, content_dim)
        self.net = nn.Sequential(
            nn.Linear(action_embed_dim + content_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.node_head = nn.Linear(hidden_dim, node_vocab)
        self.reconstruction_head = nn.Linear(hidden_dim, content_dim)

    def freeze_grid_rnn(self) -> None:
        pass

    def reset_memory(self, batch_size: int, device) -> None:
        pass

    def forward(
        self,
        actions: torch.Tensor,
        observations: torch.Tensor,
        h0: Optional[torch.Tensor] = None,
        reset_memory: bool = True,
    ) -> Dict[str, torch.Tensor]:
        act_emb = self.action_encoder(actions)
        obs_emb = self.content_encoder(observations)
        x = torch.cat([act_emb, obs_emb], dim=-1)
        # apply MLP independently per time-step
        B, S, D = x.shape
        hidden_seq = self.net(x.reshape(-1, D)).reshape(B, S, self.hidden_dim)
        return {
            "hidden_seq": hidden_seq,
            "features": hidden_seq,
            "periodic_seq": hidden_seq,
            "content_seq": obs_emb,
            "content_reconstruction": self.reconstruction_head(hidden_seq),
            "node_logits": self.node_head(hidden_seq),
        }


# ---------------------------------------------------------------------------
#  Factory
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
#  Plain vanilla RNN baseline
# ---------------------------------------------------------------------------

class PlainRNNBaseline(nn.Module):
    """Vanilla RNN (not GRU) – simplest recurrent baseline."""

    def __init__(
        self,
        action_vocab: int,
        action_embed_dim: int,
        hidden_dim: int,
        content_dim: int,
        observation_vocab: int,
        node_vocab: int,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.content_dim = content_dim
        self.action_encoder = nn.Embedding(action_vocab, action_embed_dim)
        self.content_encoder = nn.Embedding(observation_vocab, content_dim)
        self.rnn = nn.RNN(action_embed_dim + content_dim, hidden_dim, batch_first=True, nonlinearity="relu")
        self.node_head = nn.Linear(hidden_dim, node_vocab)
        self.reconstruction_head = nn.Linear(hidden_dim, content_dim)

    def freeze_grid_rnn(self) -> None:
        pass

    def reset_memory(self, batch_size: int, device) -> None:
        pass

    def forward(
        self,
        actions: torch.Tensor,
        observations: torch.Tensor,
        h0: Optional[torch.Tensor] = None,
        reset_memory: bool = True,
    ) -> Dict[str, torch.Tensor]:
        act_emb = self.action_encoder(actions)
        obs_emb = self.content_encoder(observations)
        inp = torch.cat([act_emb, obs_emb], dim=-1)
        hidden_seq, _ = self.rnn(inp, h0.unsqueeze(0) if h0 is not None else None)
        return {
            "hidden_seq": hidden_seq,
            "features": hidden_seq,
            "periodic_seq": hidden_seq,
            "content_seq": obs_emb,
            "content_reconstruction": self.reconstruction_head(hidden_seq),
            "node_logits": self.node_head(hidden_seq),
        }


# ---------------------------------------------------------------------------
#  CNN + FC baseline for visual tasks
# ---------------------------------------------------------------------------

class CNNClassifier(nn.Module):
    """Standard CNN classifier baseline for visual concept tasks."""

    def __init__(
        self,
        n_classes: int = 10,
        in_channels: int = 1,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 2 * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        feat = self.cnn(images)
        return self.fc(feat)

    def get_features(self, images: torch.Tensor) -> torch.Tensor:
        return self.cnn(images)


# ---------------------------------------------------------------------------
#  Prototypical Network baseline for few-shot visual tasks
# ---------------------------------------------------------------------------

class PrototypicalNetwork(nn.Module):
    """Prototypical Network (Snell et al., 2017) for few-shot classification.

    Computes class prototypes from support embeddings, classifies queries by
    nearest-prototype distance.
    """

    def __init__(self, in_channels: int = 1, embed_dim: int = 64) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
            nn.Linear(64 * 2 * 2, embed_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def compute_prototypes(
        self, support_images: torch.Tensor, support_labels: torch.Tensor
    ) -> torch.Tensor:
        embeddings = self.encode(support_images)
        classes = torch.unique(support_labels)
        prototypes = []
        for c in classes:
            mask = support_labels == c
            prototypes.append(embeddings[mask].mean(dim=0))
        return torch.stack(prototypes)

    def forward(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
    ) -> torch.Tensor:
        prototypes = self.compute_prototypes(support_images, support_labels)
        query_emb = self.encode(query_images)
        return -torch.cdist(query_emb, prototypes, p=2)


# ---------------------------------------------------------------------------
#  Factory
# ---------------------------------------------------------------------------

def build_baseline(kind: str, **kwargs) -> nn.Module:
    """Build a baseline model by name.

    Parameters
    ----------
    kind : str
        One of ``"plain_gru"``, ``"transformer"``, ``"mlp"``.
    **kwargs
        Forwarded to the constructor.  Required keys at minimum:
        ``action_vocab``, ``action_embed_dim``, ``hidden_dim``,
        ``content_dim``, ``observation_vocab``, ``node_vocab``.
    """
    classes = {
        "plain_rnn": PlainRNNBaseline,
        "plain_gru": PlainGRUBaseline,
        "transformer": TransformerBaseline,
        "mlp": MLPBaseline,
    }
    if kind not in classes:
        raise ValueError(f"Unknown baseline kind: {kind}. Choose from {list(classes)}")
    return classes[kind](**kwargs)
