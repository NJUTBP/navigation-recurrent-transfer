"""Residue Hyperdimensional Computing (RHC) encoder.

Implements the Fractional Power Encoding (FPE) and Residue Number System
framework from Kymn et al. (2023), "Computing with Residue Numbers in
High-Dimensional Representation".

Key concepts
------------
1. **Fractional Power Encoding (FPE)**: Maps integer x to a phasor vector
   z(x) = z^x, where z is a random base vector of unit-magnitude complex
   numbers.  When phases are drawn from the m-th roots of unity, congruent
   values map to the same vector: z_m(x+m) = z_m(x).

2. **Residue HD encoding**: Combines K modular FPE vectors via Hadamard
   product: z(x) = z_{m1}(x) ⊙ z_{m2}(x) ⊙ ... ⊙ z_{mK}(x).
   This gives quasi-orthogonal representations for different remainders
   with dynamic range M = ∏ m_k, using only ∑ m_k codebook vectors.

3. **Hexagonal coordinates**: For 2D spaces, projects into a 3-axis
   Mercedes-Benz frame with 120° separation, achieving higher spatial
   resolution than Cartesian grids (like grid cells in entorhinal cortex).

4. **Algebraic operations**: Addition via Hadamard product z(x1+x2) =
   z(x1) ⊙ z(x2); this enables path integration in the HD space.

Integration with GPM-CMap
-------------------------
- Replaces / augments ``MultiScalePeriodicEncoder`` for concept-space tasks.
- The phasor encoding is analogous to grid-cell firing patterns.
- The Hadamard-product addition implements the same position-update logic
  that GridRNN learns, but in a principled algebraic framework.
- Can be used with the ResonatorNetwork for efficient factorization of
  composite visual scenes into identity + position.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


# -----------------------------------------------------------------------
#  Fractional Power Encoding, modulo m  (Definition 2.2.1)
# -----------------------------------------------------------------------

class FractionalPowerEncoder(nn.Module):
    """Encode integers via phasors drawn from m-th roots of unity.

    Each dimension j has a random phase ϕ_j that is a multiple of 2π/m.
    The encoding of integer x is:  z_m(x) = [e^{iϕ_1 x}, ..., e^{iϕ_D x}]

    Since each ϕ_j = 2πk_j/m for integer k_j, we have z_m(x+m) = z_m(x),
    implementing modular arithmetic.

    Parameters
    ----------
    dim : int
        Output dimension D of the phasor vector.
    modulus : int
        The modulus m.  Phases are drawn from the m-th roots of unity.
    seed : int
        Random seed for reproducible base vector generation.
    """

    def __init__(self, dim: int, modulus: int, seed: int = 0) -> None:
        super().__init__()
        self.dim = dim
        self.modulus = modulus

        rng = np.random.default_rng(seed)
        # Draw random integers k_j ∈ {0, 1, ..., m-1} for each dimension
        k = rng.integers(0, modulus, size=dim)
        # Phases: ϕ_j = 2π k_j / m
        phases = 2.0 * np.pi * k.astype(np.float32) / modulus
        self.register_buffer("base_phases", torch.from_numpy(phases))  # (D,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode scalar or batch of integers.

        Parameters
        ----------
        x : (...,) int or float tensor — values to encode.

        Returns
        -------
        z_real : (..., 2*D) float tensor — [cos(ϕx), sin(ϕx)] concatenated.
            We use real-valued representation for compatibility with PyTorch.
        """
        # x: (...) -> (..., 1) for broadcasting with phases (D,)
        x_expanded = x.unsqueeze(-1).float()
        angles = x_expanded * self.base_phases  # (..., D)
        return torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)

    def encode_complex(self, x: torch.Tensor) -> torch.Tensor:
        """Encode as complex phasors (for Hadamard product operations).

        Returns (..., D) complex tensor.
        """
        x_expanded = x.unsqueeze(-1).float()
        angles = x_expanded * self.base_phases
        return torch.complex(torch.cos(angles), torch.sin(angles))

    @property
    def output_dim(self) -> int:
        return 2 * self.dim


# -----------------------------------------------------------------------
#  Residue HD Encoder  (Definition 2.2.2)
# -----------------------------------------------------------------------

class ResidueHDEncoder(nn.Module):
    """Residue Hyperdimensional Computing encoder.

    Combines K modular FPE encoders via Hadamard product (in complex domain)
    to create a residue number representation with dynamic range M = ∏ m_k.

    The kernel induced by the resulting vector is ≈1 when Δx = 0 (mod M)
    and ≈0 for other integer differences — quasi-orthogonal codes.

    Parameters
    ----------
    dim : int
        Phasor dimension D per modulus (output_dim = 2*D in real representation).
    moduli : list of int
        List of pairwise co-prime moduli [m1, m2, ..., mK].
    seed : int
        Base seed (each modulus uses seed + k).
    """

    def __init__(self, dim: int, moduli: List[int], seed: int = 42) -> None:
        super().__init__()
        self.dim = dim
        self.moduli = list(moduli)
        self.K = len(moduli)
        self.M = 1
        for m in moduli:
            self.M *= m

        self.encoders = nn.ModuleList([
            FractionalPowerEncoder(dim, m, seed=seed + k)
            for k, m in enumerate(moduli)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode integer(s) as real-valued residue HD vector.

        Parameters
        ----------
        x : (...,) tensor of integer values.

        Returns
        -------
        z : (..., 2*D) real tensor — the residue HD encoding.
        """
        # Encode in complex domain, combine via Hadamard product, convert to real
        z_complex = self.encode_complex(x)
        return torch.cat([z_complex.real, z_complex.imag], dim=-1)

    def encode_complex(self, x: torch.Tensor) -> torch.Tensor:
        """Encode as complex phasor vector, combining all moduli.

        z(x) = z_{m1}(x) ⊙ z_{m2}(x) ⊙ ... ⊙ z_{mK}(x)

        Returns (..., D) complex tensor.
        """
        result = self.encoders[0].encode_complex(x)
        for enc in self.encoders[1:]:
            result = result * enc.encode_complex(x)  # Hadamard product
        return result

    def addition(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """Perform addition in HD space: z(x1+x2) = z(x1) ⊙ z(x2).

        Works on real-valued representations by converting to complex,
        multiplying, and converting back.

        Parameters
        ----------
        z1, z2 : (..., 2*D) real tensors.

        Returns
        -------
        z_sum : (..., 2*D) real tensor representing z(x1 + x2).
        """
        D = self.dim
        c1 = torch.complex(z1[..., :D], z1[..., D:])
        c2 = torch.complex(z2[..., :D], z2[..., D:])
        c_sum = c1 * c2
        return torch.cat([c_sum.real, c_sum.imag], dim=-1)

    def get_codebook(self, modulus_idx: int) -> torch.Tensor:
        """Get codebook vectors for a specific modulus.

        Returns (m_k, 2*D) tensor of all possible residue encodings.
        """
        m = self.moduli[modulus_idx]
        enc = self.encoders[modulus_idx]
        indices = torch.arange(m, device=enc.base_phases.device)
        return enc(indices)

    def get_all_codebooks(self) -> List[torch.Tensor]:
        """Get codebook vectors for all moduli.

        Returns list of (m_k, 2*D) tensors.
        """
        return [self.get_codebook(k) for k in range(self.K)]

    def similarity(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """Compute kernel similarity K(x1, x2) = Re{z1† z2} / D.

        Parameters
        ----------
        z1, z2 : (..., 2*D) real tensors.

        Returns
        -------
        K : (...,) similarity values.
        """
        D = self.dim
        c1 = torch.complex(z1[..., :D], z1[..., D:])
        c2 = torch.complex(z2[..., :D], z2[..., D:])
        return (c1.conj() * c2).real.mean(dim=-1)

    @property
    def output_dim(self) -> int:
        return 2 * self.dim

    @property
    def effective_range(self) -> int:
        return self.M

    @property
    def codebook_size(self) -> int:
        """Total number of codebook vectors needed (∑ m_k, not ∏ m_k)."""
        return sum(self.moduli)


# -----------------------------------------------------------------------
#  Multi-dimensional RHC Encoder  (Section 2.4.1)
# -----------------------------------------------------------------------

class MultiDimResidueEncoder(nn.Module):
    """RHC encoder for multi-dimensional integer vectors.

    For x ∈ Z^n, encodes as:
        z(x) = z_1(x_1) ⊙ z_2(x_2) ⊙ ... ⊙ z_n(x_n)

    Each z_i is an independent ResidueHDEncoder for dimension i.
    The resulting kernel factorizes: K(Δx) = ∏_i K_i(Δx_i).

    Parameters
    ----------
    n_dims : int
        Number of spatial dimensions (e.g., 2 for a concept lattice).
    phasor_dim : int
        Phasor dimension D per axis.
    moduli : list of int
        Moduli shared across all axes.
    seed : int
    """

    def __init__(
        self,
        n_dims: int = 2,
        phasor_dim: int = 128,
        moduli: Optional[List[int]] = None,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.n_dims = n_dims
        if moduli is None:
            moduli = [3, 5, 7]  # default co-prime moduli
        self.moduli = moduli
        self.phasor_dim = phasor_dim

        self.axis_encoders = nn.ModuleList([
            ResidueHDEncoder(phasor_dim, moduli, seed=seed + i * 1000)
            for i in range(n_dims)
        ])

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Encode multi-dimensional coordinates.

        Parameters
        ----------
        coords : (..., n_dims) float or int tensor.

        Returns
        -------
        z : (..., 2*D) real tensor.
        """
        # Combine axes via Hadamard product in complex domain
        D = self.phasor_dim
        z_complex = self.axis_encoders[0].encode_complex(coords[..., 0])
        for i in range(1, self.n_dims):
            z_complex = z_complex * self.axis_encoders[i].encode_complex(coords[..., i])
        return torch.cat([z_complex.real, z_complex.imag], dim=-1)

    def encode_per_axis(self, coords: torch.Tensor) -> List[torch.Tensor]:
        """Encode each axis separately (for resonator network factorization).

        Returns list of (..., D) complex tensors, one per axis.
        """
        return [
            self.axis_encoders[i].encode_complex(coords[..., i])
            for i in range(self.n_dims)
        ]

    @property
    def output_dim(self) -> int:
        return 2 * self.phasor_dim


# -----------------------------------------------------------------------
#  Hexagonal RHC Encoder  (Section 2.4.2)
# -----------------------------------------------------------------------

class HexagonalResidueEncoder(nn.Module):
    """Hexagonal coordinate RHC encoder for 2D spaces.

    Projects 2D positions into a 3-axis Mercedes-Benz frame (120° separation),
    then encodes each axis with a modular FPE.  Enforces the constraint
    z([1,1,1]) = z([0,0,0]) so that paths returning to the same 2D position
    produce identical encodings.

    This mirrors the hexagonal grid cell firing pattern in entorhinal cortex.

    Parameters
    ----------
    phasor_dim : int
        Phasor dimension D.
    moduli : list of int
        Moduli for the residue system.
    seed : int
    """

    # Mercedes-Benz projection matrix: 2D → 3D
    PSI = np.array([
        [-np.sqrt(3) / 2, -0.5],
        [ np.sqrt(3) / 2, -0.5],
        [0.0,              1.0],
    ], dtype=np.float32)

    def __init__(
        self,
        phasor_dim: int = 128,
        moduli: Optional[List[int]] = None,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.phasor_dim = phasor_dim
        if moduli is None:
            moduli = [3, 5, 7]
        self.moduli = moduli

        self.register_buffer(
            "projection_matrix",
            torch.from_numpy(self.PSI),  # (3, 2)
        )

        # 3-axis encoder with phase constraint: sum of 3 phases = 0 (mod 2π)
        self.axis_encoders = nn.ModuleList()
        for axis_idx in range(3):
            self.axis_encoders.append(
                ResidueHDEncoder(phasor_dim, moduli, seed=seed + axis_idx * 1000)
            )

        # Enforce hex constraint on base phases:
        # For each modulus encoder, the 3 axis phases must sum to 0 (mod 2π)
        self._enforce_hex_constraint()

    def _enforce_hex_constraint(self) -> None:
        """Enforce z([1,1,1]) = z([0,0,0]) by constraining phase triplets.

        For each phasor dimension j and modulus k:
            ϕ_{1,j,k} + ϕ_{2,j,k} + ϕ_{3,j,k} ≡ 0  (mod 2π)

        We achieve this by setting axis 2's phases to -(ϕ_0 + ϕ_1) mod 2π.
        """
        for mod_idx in range(len(self.moduli)):
            m = self.moduli[mod_idx]
            # Get base phases from axis 0 and 1
            p0 = self.axis_encoders[0].encoders[mod_idx].base_phases
            p1 = self.axis_encoders[1].encoders[mod_idx].base_phases
            # Set axis 2 phases so sum = 0 (mod 2π)
            p2_target = -(p0 + p1)
            # Snap to nearest m-th root of unity
            k2 = torch.round(p2_target * m / (2.0 * math.pi)) % m
            p2_snapped = 2.0 * math.pi * k2 / m
            self.axis_encoders[2].encoders[mod_idx].base_phases.copy_(p2_snapped)

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        """Encode 2D positions using hexagonal coordinates.

        Parameters
        ----------
        xy : (..., 2) float tensor — 2D positions.

        Returns
        -------
        z : (..., 2*D) real tensor.
        """
        # Project to 3D hex coordinates: y = Ψ x
        hex_coords = xy @ self.projection_matrix.t()  # (..., 3)

        # Encode each hex axis and combine via Hadamard product
        D = self.phasor_dim
        z_complex = self.axis_encoders[0].encode_complex(hex_coords[..., 0])
        for i in range(1, 3):
            z_complex = z_complex * self.axis_encoders[i].encode_complex(hex_coords[..., i])
        return torch.cat([z_complex.real, z_complex.imag], dim=-1)

    @property
    def output_dim(self) -> int:
        return 2 * self.phasor_dim


# -----------------------------------------------------------------------
#  Learnable RHC-inspired position encoder  (for neural network training)
# -----------------------------------------------------------------------

class LearnableRHCEncoder(nn.Module):
    """Learnable position encoder inspired by RHC.

    Unlike the pure algebraic RHC (fixed random phases), this version
    makes the base phases learnable parameters, initialized from the
    RHC structure.  This allows end-to-end training while retaining
    the residue number system inductive bias.

    Can replace ``MultiScalePeriodicEncoder`` in GPMCMap.

    Parameters
    ----------
    input_dim : int
        Dimensionality of input (hidden_dim for model encoder, or
        coord_dim for target encoder).
    n_scales : int
        Number of moduli / frequency scales.
    phasor_dim : int
        Phasor dimension D (output_dim = 2*D).
    moduli : list of int or None
        If None, auto-generates co-prime moduli starting from 3.
    learnable : bool
        If True, phases are nn.Parameter; if False, frozen buffer.
    """

    def __init__(
        self,
        input_dim: int,
        n_scales: int = 8,
        phasor_dim: int = 0,
        moduli: Optional[List[int]] = None,
        learnable: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.n_scales = n_scales

        # Auto-generate co-prime moduli if not specified
        if moduli is None:
            moduli = self._generate_coprimes(n_scales)
        self.moduli = moduli[:n_scales]

        # Determine output dim
        if phasor_dim <= 0:
            phasor_dim = n_scales  # default: 1 phasor per scale
        self._output_dim = 2 * phasor_dim
        self.phasor_dim = phasor_dim

        # Input projection: map input_dim → n_scales scalar "coordinates"
        self.input_proj = nn.Linear(input_dim, n_scales, bias=False)

        # Phase parameters: (n_scales, phasor_dim)
        # Initialized from m-th roots of unity
        phases = torch.zeros(n_scales, phasor_dim)
        for k, m in enumerate(self.moduli):
            roots = torch.randint(0, m, (phasor_dim,))
            phases[k] = 2.0 * math.pi * roots.float() / m

        if learnable:
            self.phases = nn.Parameter(phases)
        else:
            self.register_buffer("phases", phases)

    @staticmethod
    def _generate_coprimes(n: int) -> List[int]:
        """Generate n pairwise co-prime integers starting from small primes."""
        primes = []
        candidate = 2
        while len(primes) < n:
            is_prime = all(candidate % p != 0 for p in primes)
            if is_prime:
                primes.append(candidate)
            candidate += 1
        return primes

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input vectors.

        Parameters
        ----------
        x : (..., input_dim) tensor.

        Returns
        -------
        z : (..., 2*phasor_dim) tensor — [cos, sin] concatenation.
        """
        # Project input to n_scales "coordinates"
        coords = self.input_proj(x)  # (..., n_scales)

        # Compute angles: coords_k * phases_k for each scale
        # coords: (..., n_scales), phases: (n_scales, phasor_dim)
        # We want: sum over scales of coords_k * phases[k, :]
        angles = torch.einsum("...k,kd->...d", coords, self.phases)  # (..., phasor_dim)

        return torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)
