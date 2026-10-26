"""Low-capacity controls for symbolic endpoint decoding.

These models are deliberately simpler than the recurrent transfer model.  They
characterise whether the symbolic task can be solved from the start identity
and the multiset of relation symbols alone, without a learned recurrent
reservoir.

The controls use the same call signature as ``AbstractPathIntModel`` so the
existing training and validation-only checkpoint-selection code can be reused.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn


def _validate_symbolic_inputs(
    *,
    start_node: torch.Tensor,
    velocities: torch.Tensor,
    lengths: Optional[torch.Tensor],
    n_nodes: int,
    n_actions: int,
    validate_inputs: bool,
) -> torch.Tensor:
    """Validate a symbolic batch and return a [B,T] validity mask."""
    if start_node.ndim != 1:
        raise ValueError(f"start_node must have shape [B]; got {tuple(start_node.shape)}")
    if velocities.ndim != 3 or velocities.shape[-1] != n_actions:
        raise ValueError(
            f"velocities must have shape [B,T,{n_actions}]; got {tuple(velocities.shape)}"
        )
    if velocities.shape[0] != start_node.shape[0]:
        raise ValueError("start_node and velocities batch dimensions do not match")
    if validate_inputs:
        if torch.any(start_node < 0) or torch.any(start_node >= n_nodes):
            raise IndexError(f"start_node ids must lie in [0,{n_nodes - 1}]")
        if not torch.isfinite(velocities).all():
            raise FloatingPointError("velocities contain NaN or Inf")

    batch_size, max_steps, _ = velocities.shape
    if lengths is None:
        return torch.ones(
            (batch_size, max_steps), dtype=torch.bool, device=velocities.device
        )
    if lengths.ndim != 1 or lengths.shape[0] != batch_size:
        raise ValueError("lengths must have shape [B] matching start_node")
    if validate_inputs and (
        torch.any(lengths < 1) or torch.any(lengths > max_steps)
    ):
        raise ValueError(f"lengths must lie in [1,{max_steps}]")
    return (
        torch.arange(max_steps, device=velocities.device).unsqueeze(0)
        < lengths.to(velocities.device).unsqueeze(1)
    )


class ActionCountLinearBaseline(nn.Module):
    """Multinomial linear classifier on start identity plus action counts.

    This is the strongest direct test of the "action-count shortcut".  It has no
    hidden recurrence and no ordered sequence processing: all temporal order is
    discarded before classification.
    """

    supports_prevalidated_inputs = True
    start_init = "embedding"

    def __init__(self, n_nodes: int, n_actions: int, use_bias: bool = True) -> None:
        super().__init__()
        if n_nodes < 2 or n_actions < 1:
            raise ValueError("n_nodes must be >=2 and n_actions must be positive")
        self.n_nodes = int(n_nodes)
        self.n_actions = int(n_actions)
        self.classifier = nn.Linear(
            self.n_nodes + self.n_actions, self.n_nodes, bias=use_bias
        )

    def forward(
        self,
        start_node: torch.Tensor,
        velocities: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        start_coords: Optional[torch.Tensor] = None,
        *,
        validate_inputs: bool = True,
    ) -> Dict[str, torch.Tensor]:
        del start_coords
        mask = _validate_symbolic_inputs(
            start_node=start_node,
            velocities=velocities,
            lengths=lengths,
            n_nodes=self.n_nodes,
            n_actions=self.n_actions,
            validate_inputs=validate_inputs,
        )
        counts = (velocities * mask.unsqueeze(-1).to(velocities.dtype)).sum(dim=1)
        start_onehot = F.one_hot(start_node, num_classes=self.n_nodes).to(
            velocities.dtype
        )
        features = torch.cat((start_onehot, counts), dim=-1)
        logits = self.classifier(features)
        return {"node_logits": logits, "features": features}


class LinearAccumulatorBaseline(nn.Module):
    """Learned two-dimensional additive state without recurrent dynamics.

    The model learns a 2-D vector for every start node and every relation symbol,
    sums the relation vectors, and applies a linear endpoint readout.  It is a
    direct non-recurrent alternative to the claim that the frozen high-
    dimensional reservoir is needed to organise the symbols geometrically.
    """

    supports_prevalidated_inputs = True
    start_init = "embedding"

    def __init__(
        self,
        n_nodes: int,
        n_actions: int,
        latent_dim: int = 2,
        init_scale: float = 0.02,
        use_bias: bool = True,
    ) -> None:
        super().__init__()
        if n_nodes < 2 or n_actions < 1 or latent_dim < 1:
            raise ValueError(
                "n_nodes must be >=2; n_actions and latent_dim must be positive"
            )
        if init_scale <= 0:
            raise ValueError("init_scale must be positive")
        self.n_nodes = int(n_nodes)
        self.n_actions = int(n_actions)
        self.latent_dim = int(latent_dim)
        self.start_embed = nn.Embedding(self.n_nodes, self.latent_dim)
        self.action_adapter = nn.Linear(
            self.n_actions, self.latent_dim, bias=False
        )
        self.node_head = nn.Linear(self.latent_dim, self.n_nodes, bias=use_bias)
        with torch.no_grad():
            nn.init.uniform_(self.start_embed.weight, -init_scale, init_scale)
            nn.init.uniform_(self.action_adapter.weight, -init_scale, init_scale)

    def forward(
        self,
        start_node: torch.Tensor,
        velocities: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        start_coords: Optional[torch.Tensor] = None,
        *,
        validate_inputs: bool = True,
    ) -> Dict[str, torch.Tensor]:
        del start_coords
        mask = _validate_symbolic_inputs(
            start_node=start_node,
            velocities=velocities,
            lengths=lengths,
            n_nodes=self.n_nodes,
            n_actions=self.n_actions,
            validate_inputs=validate_inputs,
        )
        action_steps = self.action_adapter(velocities)
        action_sum = (action_steps * mask.unsqueeze(-1).to(action_steps.dtype)).sum(
            dim=1
        )
        h0 = self.start_embed(start_node)
        state = h0 + action_sum
        logits = self.node_head(state)
        return {
            "node_logits": logits,
            "hidden_final": state,
            "h0": h0,
            "action_sum": action_sum,
        }

    def action_matrix(self) -> torch.Tensor:
        """Return the learned [latent_dim, n_actions] action matrix."""
        return self.action_adapter.weight
