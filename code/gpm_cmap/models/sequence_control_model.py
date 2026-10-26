"""GridRNN wrapper for CTL-01 non-navigation sequence pretraining.

The transferable submodule is deliberately named ``grid_rnn`` and retains
input dimension 2.  Its state-dict keys therefore match
``AbstractPathIntModel.load_nav_checkpoint`` exactly.  The primary head
reconstructs the input from ``primary_delay`` steps ago.  Optional, separate
auxiliary heads reconstruct shorter lags; they provide dense learning signals
for the same recurrent memory trace and are discarded during downstream
transfer.  Checkpoint compatibility depends only on the unchanged
``grid_rnn`` tensors.
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Dict, Optional

import torch
import torch.nn as nn

from gpm_cmap.models.grid_rnn import GridRNN


class SequenceControlModel(nn.Module):
    """Delayed-reconstruction model with an architecture-matched GridRNN core."""

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 4096,
        activation: str = "relu",
        nonnegative: bool = True,
        cell_type: str = "rnn",
        dropout: float = 0.0,
        use_bias: bool = False,
        primary_delay: int = 10,
        auxiliary_lags: Iterable[int] = (),
    ) -> None:
        super().__init__()
        if input_dim != 2:
            raise ValueError(
                "input_dim must stay 2 for full checkpoint transfer into "
                "AbstractPathIntModel"
            )
        if primary_delay < 1:
            raise ValueError("primary_delay must be positive")

        lags = sorted({int(lag) for lag in auxiliary_lags})
        invalid = [lag for lag in lags if lag < 1 or lag >= primary_delay]
        if invalid:
            raise ValueError(
                "auxiliary lags must lie in [1, primary_delay); invalid values: "
                f"{invalid}"
            )

        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.primary_delay = int(primary_delay)
        self.auxiliary_lags = tuple(lags)
        self.grid_rnn = GridRNN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            activation=activation,
            nonnegative=nonnegative,
            cell_type=cell_type,
            dropout=dropout,
            use_bias=use_bias,
        )
        # Keep this historical name for the formal delay-10 readout.
        self.copy_head = nn.Linear(hidden_dim, input_dim, bias=use_bias)
        self.aux_copy_heads = nn.ModuleDict(
            {
                f"lag_{lag}": nn.Linear(hidden_dim, input_dim, bias=use_bias)
                for lag in self.auxiliary_lags
            }
        )

    def forward(
        self, input_seq: torch.Tensor, h0: Optional[torch.Tensor] = None
    ) -> Dict[str, object]:
        hidden_seq, h_last = self.grid_rnn(input_seq, h0=h0)
        primary_pred = self.copy_head(hidden_seq)
        auxiliary_pred = {
            lag: self.aux_copy_heads[f"lag_{lag}"](hidden_seq)
            for lag in self.auxiliary_lags
        }
        return {
            "copy_pred": primary_pred,
            "aux_copy_pred": auxiliary_pred,
            "hidden_seq": hidden_seq,
            "hidden_last": h_last,
        }

    def predictions_by_lag(self, outputs: Dict[str, object]) -> Dict[int, torch.Tensor]:
        """Return all full-sequence prediction tensors keyed by lag."""
        primary = outputs["copy_pred"]
        auxiliary = outputs["aux_copy_pred"]
        if not isinstance(primary, torch.Tensor) or not isinstance(auxiliary, dict):
            raise TypeError("Unexpected model output structure")
        predictions: Dict[int, torch.Tensor] = {self.primary_delay: primary}
        for lag, value in auxiliary.items():
            if not isinstance(lag, int) or not isinstance(value, torch.Tensor):
                raise TypeError("Unexpected auxiliary prediction structure")
            predictions[lag] = value
        return predictions

    def recurrent_l2(self) -> torch.Tensor:
        return self.grid_rnn.recurrent_l2()
