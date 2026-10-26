"""Position encoding modules — full comparison matrix for Phase 2.

This module implements **five** position encoding strategies as required
by the roadmap's ablation design (§2.4):

====================  =================================================
Encoder               Description
====================  =================================================
``identity``          **Baseline** — pass-through (no encoding); input
                      is zero-padded to ``output_dim``.
``rbf``               **RBF** — Gaussian radial basis functions centred
                      at learnable or fixed anchor points.
``mlp``               **MLP** — two-layer feedforward network.
``periodic_fixed``    **Periodic-fixed** — multi-scale sin/cos with
                      geometrically-spaced, *frozen* frequencies.
``periodic_learned``  **Periodic-learned** — same architecture, but ω
                      and φ are *learnable* parameters.
====================  =================================================

Every encoder exposes:
  - ``__init__(input_dim, output_dim=None, ...)``
  - ``output_dim`` property (or attribute)
  - ``forward(xi) → z``   where ``xi: (..., input_dim)``

The factory ``build_position_encoder(kind, ...)`` is the single entry
point used by :class:`GPMCMap`.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn


# -----------------------------------------------------------------------
#  1. Baseline / Identity
# -----------------------------------------------------------------------

class IdentityPositionEncoder(nn.Module):
    """No encoding — zero-pad input to *output_dim*."""

    def __init__(self, input_dim: int, output_dim: int = 16) -> None:
        super().__init__()
        self.input_dim = input_dim
        self._output_dim = max(output_dim, input_dim)

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, xi: torch.Tensor) -> torch.Tensor:
        if self._output_dim == self.input_dim:
            return xi
        pad_size = self._output_dim - self.input_dim
        return torch.nn.functional.pad(xi, (0, pad_size))


# -----------------------------------------------------------------------
#  2. RBF (Radial Basis Function)
# -----------------------------------------------------------------------

class RBFPositionEncoder(nn.Module):
    """Gaussian RBF encoding with learnable centres and widths.

    z_k(ξ) = exp( -||ξ - c_k||² / (2 σ_k²) )

    ``n_centres`` controls the number of basis functions; ``output_dim``
    is always equal to ``n_centres``.
    """

    def __init__(
        self,
        input_dim: int,
        n_centres: int = 16,
        learnable: bool = True,
        init_spread: float = 2.0,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self._output_dim = n_centres

        centres = torch.randn(n_centres, input_dim) * init_spread
        log_sigmas = torch.zeros(n_centres)  # σ = 1 initially

        if learnable:
            self.centres = nn.Parameter(centres)
            self.log_sigmas = nn.Parameter(log_sigmas)
        else:
            self.register_buffer("centres", centres)
            self.register_buffer("log_sigmas", log_sigmas)

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, xi: torch.Tensor) -> torch.Tensor:
        # xi: (..., input_dim)  centres: (K, input_dim)
        diff = xi.unsqueeze(-2) - self.centres                     # (..., K, D)
        sq_dist = (diff ** 2).sum(dim=-1)                          # (..., K)
        sigma_sq = (2 * self.log_sigmas).exp().clamp(min=1e-6)     # (K,)
        return torch.exp(-sq_dist / (2 * sigma_sq))                # (..., K)


# -----------------------------------------------------------------------
#  3. MLP
# -----------------------------------------------------------------------

class MLPPositionEncoder(nn.Module):
    """Two-layer MLP position encoder."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 16,
        hidden_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self._output_dim = output_dim
        h = hidden_dim or max(output_dim, input_dim * 4)
        self.net = nn.Sequential(
            nn.Linear(input_dim, h),
            nn.ReLU(),
            nn.Linear(h, output_dim),
        )

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, xi: torch.Tensor) -> torch.Tensor:
        return self.net(xi)


# -----------------------------------------------------------------------
#  4 & 5. Multi-Scale Periodic Encoder (fixed / learned)
# -----------------------------------------------------------------------

class MultiScalePeriodicEncoder(nn.Module):
    """Multi-scale sin/cos encoding: z(ξ) = [cos(ω·ξ+φ), sin(ω·ξ+φ)].

    Parameters
    ----------
    input_dim : int
        Dimensionality of the structural coordinate ξ.
    n_scales : int
        Number of frequency modules K.  ``output_dim = 2 * n_scales``.
    learnable : bool
        If True → ``periodic_learned``; if False → ``periodic_fixed``.
    freq_init : str
        ``"logspace"`` (default) — frequencies log-spaced in [0.1, 10].
        ``"geometric"`` — ω_k = 2^{k/2} for k = 0 … K-1.
    """

    def __init__(
        self,
        input_dim: int,
        n_scales: int = 8,
        learnable: bool = True,
        freq_init: str = "logspace",
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.n_scales = n_scales
        self._learnable = learnable

        omegas = torch.randn(n_scales, input_dim)
        phis = torch.zeros(n_scales)

        # Deterministic initialisation
        with torch.no_grad():
            if freq_init == "logspace":
                freqs = torch.logspace(-1, 1, n_scales)
            elif freq_init == "geometric":
                freqs = torch.pow(2.0, torch.arange(n_scales).float() / 2.0)
            else:
                raise ValueError(f"Unknown freq_init: {freq_init}")

            if input_dim >= 2:
                angles = torch.linspace(0, math.pi, n_scales)
                omegas[:, 0] = freqs * torch.cos(angles)
                omegas[:, 1] = freqs * torch.sin(angles)
            else:
                omegas[:, 0] = freqs

        if learnable:
            self.omegas = nn.Parameter(omegas)
            self.phis = nn.Parameter(phis)
        else:
            self.register_buffer("omegas", omegas)
            self.register_buffer("phis", phis)

    @property
    def output_dim(self) -> int:
        return 2 * self.n_scales

    def forward(self, xi: torch.Tensor) -> torch.Tensor:
        # xi: (..., input_dim)
        proj = xi @ self.omegas.t() + self.phis          # (..., n_scales)
        return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)


# -----------------------------------------------------------------------
#  Backward-compat alias (used by existing LinearPositionEncoder imports)
# -----------------------------------------------------------------------

class LinearPositionEncoder(nn.Module):
    """Simple linear + ReLU encoder (kept for backward compatibility)."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self._output_dim = output_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
        )

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# -----------------------------------------------------------------------
#  Factory
# -----------------------------------------------------------------------

def build_position_encoder(
    kind: str,
    input_dim: int,
    n_scales: int = 8,
    output_dim: Optional[int] = None,
    freq_init: str = "logspace",
    rbf_n_centres: Optional[int] = None,
    mlp_hidden_dim: Optional[int] = None,
) -> nn.Module:
    """Build a position encoder by name.

    Parameters
    ----------
    kind : str
        One of ``"identity"``, ``"rbf"``, ``"mlp"``,
        ``"periodic_fixed"``, ``"periodic_learned"``.
    input_dim : int
        Dimensionality of the input structural coordinate ξ(s).
    n_scales : int
        Number of frequency modules for periodic encoders.
    output_dim : int, optional
        Desired output dimensionality.  For periodic encoders this is
        *ignored* — ``output_dim = 2 * n_scales``.  For others it
        defaults to ``2 * n_scales`` if not given.
    freq_init : str
        ``"logspace"`` or ``"geometric"`` (periodic encoders only).
    rbf_n_centres : int, optional
        Number of centres for RBF (defaults to ``output_dim``).
    mlp_hidden_dim : int, optional
        Hidden layer size for MLP.

    Returns
    -------
    nn.Module
        Position encoder with a ``.output_dim`` attribute.
    """
    default_out = output_dim or (2 * n_scales)

    if kind == "identity":
        return IdentityPositionEncoder(input_dim, default_out)

    elif kind == "rbf":
        nc = rbf_n_centres or default_out
        return RBFPositionEncoder(input_dim, n_centres=nc, learnable=True)

    elif kind == "mlp":
        return MLPPositionEncoder(input_dim, default_out, hidden_dim=mlp_hidden_dim)

    elif kind == "periodic_fixed":
        return MultiScalePeriodicEncoder(
            input_dim, n_scales=n_scales, learnable=False, freq_init=freq_init,
        )

    elif kind == "periodic_learned":
        return MultiScalePeriodicEncoder(
            input_dim, n_scales=n_scales, learnable=True, freq_init=freq_init,
        )

    elif kind == "rhc_fixed":
        from .rhc_encoder import LearnableRHCEncoder
        return LearnableRHCEncoder(
            input_dim, n_scales=n_scales, phasor_dim=n_scales, learnable=False,
        )

    elif kind == "rhc_learned":
        from .rhc_encoder import LearnableRHCEncoder
        return LearnableRHCEncoder(
            input_dim, n_scales=n_scales, phasor_dim=n_scales, learnable=True,
        )

    else:
        raise ValueError(
            f"Unknown position encoder kind: {kind!r}.  "
            f"Choose from: identity, rbf, mlp, periodic_fixed, periodic_learned, "
            f"rhc_fixed, rhc_learned."
        )
