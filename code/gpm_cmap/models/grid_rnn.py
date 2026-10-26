"""Grid RNN for path integration.

Upgrades vs original (learning from reference code)
----------------------------------------------------
1. **Fixed double-ReLU bug** — original GridRNNCell applied activation()
   then ReLU again if nonnegative=True.  When activation='relu' this
   wasted a computation and when activation='tanh' it silently clipped
   negative outputs.  Now: if nonnegative=True, always use ReLU as the
   activation (override whatever was requested); if False, use the
   requested activation.
2. **Bias-free option (use_bias=False)** — reference code uses bias=False
   on all projections.  Zero bias simplifies the dynamics and lets the
   grid pattern emerge from weight structure alone.
3. **Dedicated W_hh regularisation** — ``recurrent_l2()`` exposes the
   squared Frobenius norm of the recurrent weight so the training loop
   can penalise it *separately* from other parameters (reference does
   ``loss += weight_decay * (W_hh**2).sum()``).  This is critical:
   regularising only W_hh encourages grid formation; applying weight_decay
   to ALL parameters via the optimizer does not have the same effect.
4. **Reference-aligned recurrent initialisation** — custom cells now use the
   same hidden-size-based initialisation scale as torch.nn.RNN/GRU. This is
   crucial for grid emergence when input_dim is very small (e.g. velocity_dim=2).
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import math
import torch
from torch import nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
#  Vanilla RNN cell
# ---------------------------------------------------------------------------

class GridRNNCell(nn.Module):
    """Custom RNN cell for path integration.

    Parameters
    ----------
    input_dim, hidden_dim : int
    activation : str   "relu" or "tanh".
    nonnegative : bool  If True, activation is forced to ReLU regardless
        of the ``activation`` argument (avoids the double-ReLU trap).
    dropout : float
    use_bias : bool   [NEW] set False to match reference (bias=False).
    """

    def __init__(self, input_dim: int, hidden_dim: int, activation: str = "relu",
                 nonnegative: bool = True, dropout: float = 0.0,
                 use_bias: bool = False) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.nonnegative = nonnegative
        self.in_proj = nn.Linear(input_dim, hidden_dim, bias=use_bias)
        self.h_proj  = nn.Linear(hidden_dim, hidden_dim, bias=use_bias)

        # [FIX] Resolve double-ReLU: if nonneg requested, just use ReLU
        if nonnegative:
            self.activation = torch.relu
        elif activation == "relu":
            self.activation = torch.relu
        elif activation == "tanh":
            self.activation = torch.tanh
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Match the built-in torch.nn.RNN initialisation scale.

        Important: nn.Linear(input_dim, hidden_dim) would initialise W_ih with
        a fan-in-dependent bound ~1/sqrt(input_dim). For velocity_dim=2 that
        yields weights around ±0.7, whereas torch.nn.RNN uses ±1/sqrt(hidden_dim)
        (~±0.0156 for hidden_dim=4096). The reference grid code relies on the
        latter much smaller scale.
        """
        bound = 1.0 / math.sqrt(self.hidden_dim)
        nn.init.uniform_(self.in_proj.weight, -bound, bound)
        nn.init.uniform_(self.h_proj.weight, -bound, bound)
        if self.in_proj.bias is not None:
            nn.init.uniform_(self.in_proj.bias, -bound, bound)
        if self.h_proj.bias is not None:
            nn.init.uniform_(self.h_proj.bias, -bound, bound)

    def forward(self, x_t: torch.Tensor, h_t: torch.Tensor) -> torch.Tensor:
        h_new = self.activation(self.in_proj(x_t) + self.h_proj(h_t))
        # No extra F.relu — activation already handles non-negativity
        if self.dropout is not None and self.training:
            h_new = self.dropout(h_new)
        return h_new


# ---------------------------------------------------------------------------
#  GRU cell variant
# ---------------------------------------------------------------------------

class GridGRUCell(nn.Module):
    """GRU cell variant for path integration."""

    def __init__(self, input_dim: int, hidden_dim: int,
                 nonnegative: bool = True, dropout: float = 0.0,
                 use_bias: bool = False) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.nonnegative = nonnegative
        self.W_ir = nn.Linear(input_dim, hidden_dim, bias=use_bias)
        self.W_hr = nn.Linear(hidden_dim, hidden_dim, bias=use_bias)
        self.W_iz = nn.Linear(input_dim, hidden_dim, bias=use_bias)
        self.W_hz = nn.Linear(hidden_dim, hidden_dim, bias=use_bias)
        self.W_in = nn.Linear(input_dim, hidden_dim, bias=use_bias)
        self.W_hn = nn.Linear(hidden_dim, hidden_dim, bias=use_bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Match torch.nn.GRU hidden-size-based initialisation."""
        bound = 1.0 / math.sqrt(self.hidden_dim)
        for layer in [self.W_ir, self.W_hr, self.W_iz, self.W_hz, self.W_in, self.W_hn]:
            nn.init.uniform_(layer.weight, -bound, bound)
            if layer.bias is not None:
                nn.init.uniform_(layer.bias, -bound, bound)

    def forward(self, x_t: torch.Tensor, h_t: torch.Tensor) -> torch.Tensor:
        r_t = torch.sigmoid(self.W_ir(x_t) + self.W_hr(h_t))
        z_t = torch.sigmoid(self.W_iz(x_t) + self.W_hz(h_t))
        n_t = torch.tanh(self.W_in(x_t) + r_t * self.W_hn(h_t))
        h_new = (1 - z_t) * n_t + z_t * h_t
        if self.nonnegative:
            h_new = F.relu(h_new)
        if self.dropout is not None and self.training:
            h_new = self.dropout(h_new)
        return h_new


# ---------------------------------------------------------------------------
#  Recurrent module (unrolls either cell type)
# ---------------------------------------------------------------------------

class GridRNN(nn.Module):
    """Recurrent module for path integration.

    [NEW] ``recurrent_l2()`` method returns ||W_hh||^2 for separate
    regularisation in the loss function (reference approach).
    """

    def __init__(self, input_dim: int, hidden_dim: int, activation: str = "relu",
                 nonnegative: bool = True, cell_type: str = "rnn",
                 dropout: float = 0.0, use_bias: bool = False) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.cell_type = cell_type
        if cell_type == "rnn":
            self.cell = GridRNNCell(input_dim, hidden_dim, activation=activation,
                                    nonnegative=nonnegative, dropout=dropout,
                                    use_bias=use_bias)
        elif cell_type == "gru":
            self.cell = GridGRUCell(input_dim, hidden_dim, nonnegative=nonnegative,
                                    dropout=dropout, use_bias=use_bias)
        else:
            raise ValueError(f"Unsupported cell_type: {cell_type!r}")

    def forward(self, inputs: torch.Tensor, h0: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        if inputs.ndim != 3:
            raise ValueError(f"Expected inputs [batch, seq, dim], got {inputs.shape}")
        batch_size, seq_len, _ = inputs.shape
        h_t = (torch.zeros(batch_size, self.hidden_dim,
                           device=inputs.device, dtype=inputs.dtype)
               if h0 is None else h0)
        hidden_seq = []
        for t in range(seq_len):
            h_t = self.cell(inputs[:, t], h_t)
            hidden_seq.append(h_t)
        hidden_seq = torch.stack(hidden_seq, dim=1)
        return hidden_seq, h_t

    def recurrent_l2(self) -> torch.Tensor:
        """[NEW] Return sum of squared recurrent (h→h) weights.

        This lets the training loop add ``weight_decay * model.recurrent_l2()``
        to the loss — penalising only W_hh, not all parameters.
        Reference: ``loss += self.weight_decay * (self.RNN.weight_hh_l0**2).sum()``
        """
        total = torch.tensor(0.0, device=next(self.parameters()).device)
        if self.cell_type == "rnn":
            total = total + (self.cell.h_proj.weight ** 2).sum()
        elif self.cell_type == "gru":
            for name in ["W_hr", "W_hz", "W_hn"]:
                total = total + (getattr(self.cell, name).weight ** 2).sum()
        return total


# ---------------------------------------------------------------------------
#  Navigation wrapper with place-cell supervision
# ---------------------------------------------------------------------------

class GridNavigationModel(nn.Module):
    """Navigation wrapper with place-cell supervision.

    [UPGRADE] init_encoder can optionally apply ReLU to respect the
    nonnegative constraint, and use_bias is forwarded to GridRNN.
    """

    def __init__(self, velocity_dim: int, hidden_dim: int, place_dim: int,
                 activation: str = "relu", nonnegative: bool = True,
                 cell_type: str = "rnn", dropout: float = 0.0,
                 use_bias: bool = False,
                 init_relu: bool = False) -> None:                
        super().__init__()
        self.nonnegative = nonnegative
        self.init_relu = init_relu
        self.grid_rnn = GridRNN(
            input_dim=velocity_dim, hidden_dim=hidden_dim,
            activation=activation, nonnegative=nonnegative,
            cell_type=cell_type, dropout=dropout, use_bias=use_bias,
        )
        self.place_head = nn.Linear(hidden_dim, place_dim, bias=use_bias)
        self.init_encoder = nn.Linear(place_dim, hidden_dim, bias=use_bias)

    def forward(self, velocity_seq: torch.Tensor,
                init_place: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        if init_place is not None:
            h0 = self.init_encoder(init_place)
            # Keep the initial place-code projection linear by default to match
            # the reference implementation. An extra ReLU here tends to create
            # an early static spatial code that can inflate grid scores before
            # the recurrent dynamics have learned path integration.
            if self.nonnegative and self.init_relu:            
                h0 = F.relu(h0)
        else:
            h0 = None
        hidden_seq, h_last = self.grid_rnn(velocity_seq, h0=h0)
        place_pred = self.place_head(hidden_seq)
        return {"place_pred": place_pred, "hidden_seq": hidden_seq, "hidden_last": h_last}

    def recurrent_l2(self) -> torch.Tensor:
        """[NEW] Expose W_hh regularisation from inner GridRNN."""
        return self.grid_rnn.recurrent_l2()
