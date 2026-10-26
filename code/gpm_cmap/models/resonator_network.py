"""Resonator Network for efficient factorization of HD vectors.

Implements the resonator network from Frady et al. (2020) as used in
Kymn et al. (2023) for decoding Residue HD Computing representations.

The resonator network factorizes a composite HD vector z into its
constituent factors {z_1, z_2, ..., z_K} where z = z_1 ⊙ z_2 ⊙ ... ⊙ z_K.
Each z_j is constrained to come from a codebook Z_j of candidate vectors.

Compared to brute-force codebook search (O(∏ |Z_j|) comparisons), the
resonator network converges in far fewer iterations, especially when
combined with residue number systems that decompose the search space.

Usage in GPM-CMap
-----------------
The resonator network enables efficient disentangling of visual scenes
into object identity + position factors, following Section 2.6.1 of
the Kymn et al. paper.  It can also decode the concept lattice position
(digit, style) from a composite representation.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from .rhc_encoder import ResidueHDEncoder


class ResonatorNetwork(nn.Module):
    """Resonator network for HD vector factorization.

    Given a composite vector z = z_1 ⊙ z_2 ⊙ ... ⊙ z_K and codebooks
    {Z_1, Z_2, ..., Z_K}, iteratively recovers the constituent factors.

    The update rule (Eq. 6 in Kymn et al.) for each factor estimate is:
        ẑ_j(t+1) = g(Z_j Z_j† (z ⊙ ∏_{i≠j} ẑ_i†(t)))

    where g normalizes each complex component to unit magnitude.

    Parameters
    ----------
    max_iterations : int
        Maximum resonator iterations before declaring failure.
    convergence_threshold : float
        Cosine similarity threshold for convergence detection.
    use_async_update : bool
        If True, update factors one at a time (asynchronous, usually better).
        If False, update all factors simultaneously (synchronous).
    """

    def __init__(
        self,
        max_iterations: int = 100,
        convergence_threshold: float = 0.95,
        use_async_update: bool = True,
    ) -> None:
        super().__init__()
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.use_async_update = use_async_update

    @staticmethod
    def _to_complex(z_real: torch.Tensor) -> torch.Tensor:
        """Convert [cos, sin] representation to complex."""
        D = z_real.shape[-1] // 2
        return torch.complex(z_real[..., :D], z_real[..., D:])

    @staticmethod
    def _to_real(z_complex: torch.Tensor) -> torch.Tensor:
        """Convert complex representation to [cos, sin]."""
        return torch.cat([z_complex.real, z_complex.imag], dim=-1)

    @staticmethod
    def _normalize_phase(z: torch.Tensor) -> torch.Tensor:
        """Normalize complex vector to unit magnitude (preserve phase only).

        This is the g() nonlinearity in the resonator update rule.
        """
        mag = z.abs().clamp(min=1e-8)
        return z / mag

    def factorize(
        self,
        z: torch.Tensor,
        codebooks: List[torch.Tensor],
        return_indices: bool = True,
    ) -> Dict[str, object]:
        """Factorize a composite HD vector into its constituents.

        Parameters
        ----------
        z : (2*D,) or (B, 2*D) real tensor — the composite vector to factorize.
        codebooks : list of (m_k, 2*D) real tensors — candidate vectors per factor.
        return_indices : bool
            If True, return decoded indices (argmax similarity per codebook).

        Returns
        -------
        dict with:
            factor_estimates : list of (2*D,) or (B, 2*D) tensors
            factor_indices : list of int or (B,) long tensors (if return_indices)
            converged : bool or (B,) bool tensor
            n_iterations : int
            similarities : list of (m_k,) or (B, m_k) tensors — final similarities
        """
        is_batched = z.ndim == 2
        if not is_batched:
            z = z.unsqueeze(0)

        B = z.shape[0]
        D = z.shape[-1] // 2
        K = len(codebooks)
        device = z.device

        # Convert to complex
        z_c = self._to_complex(z)  # (B, D)
        cb_c = [self._to_complex(cb) for cb in codebooks]  # list of (m_k, D)

        # Initialize factor estimates randomly
        estimates = []
        for k in range(K):
            # Random initialization from codebook
            idx = torch.randint(0, cb_c[k].shape[0], (B,), device=device)
            estimates.append(cb_c[k][idx])  # (B, D)

        # Resonator dynamics
        converged = torch.zeros(B, dtype=torch.bool, device=device)
        n_iters = 0

        for iteration in range(self.max_iterations):
            old_estimates = [e.clone() for e in estimates]

            if self.use_async_update:
                # Asynchronous: update one factor at a time
                for j in range(K):
                    estimates[j] = self._update_factor(
                        z_c, estimates, cb_c[j], j
                    )
            else:
                # Synchronous: compute all updates then apply
                new_estimates = []
                for j in range(K):
                    new_estimates.append(
                        self._update_factor(z_c, estimates, cb_c[j], j)
                    )
                estimates = new_estimates

            n_iters = iteration + 1

            # Check convergence
            sim_sum = torch.zeros(B, device=device)
            for j in range(K):
                sim = (old_estimates[j].conj() * estimates[j]).real.mean(dim=-1)
                sim_sum += sim
            avg_sim = sim_sum / K
            converged = avg_sim > self.convergence_threshold

            if converged.all():
                break

        # Decode: find best-matching codebook entry for each factor
        result: Dict[str, object] = {
            "factor_estimates": [self._to_real(e) for e in estimates],
            "converged": converged.squeeze(0) if not is_batched else converged,
            "n_iterations": n_iters,
        }

        if return_indices:
            indices = []
            similarities = []
            for j in range(K):
                # Inner product with codebook: (B, D) × (m_k, D)^H → (B, m_k)
                sim = torch.einsum(
                    "bd,md->bm", estimates[j].conj(), cb_c[j]
                ).real / D
                indices.append(sim.argmax(dim=-1))  # (B,)
                similarities.append(sim)

            result["factor_indices"] = [
                idx.squeeze(0) if not is_batched else idx for idx in indices
            ]
            result["similarities"] = [
                s.squeeze(0) if not is_batched else s for s in similarities
            ]

        return result

    def _update_factor(
        self,
        z: torch.Tensor,
        estimates: List[torch.Tensor],
        codebook: torch.Tensor,
        factor_idx: int,
    ) -> torch.Tensor:
        """One resonator update step for factor j.

        ẑ_j(t+1) = g(Z_j Z_j† (z ⊙ ∏_{i≠j} ẑ_i†(t)))

        Parameters
        ----------
        z : (B, D) complex — target composite vector.
        estimates : list of (B, D) complex — current factor estimates.
        codebook : (m_j, D) complex — candidates for factor j.
        factor_idx : int — which factor to update.

        Returns
        -------
        (B, D) complex — updated estimate for factor j.
        """
        B, D = z.shape
        m = codebook.shape[0]

        # Compute residual: z ⊙ ∏_{i≠j} ẑ_i†
        residual = z.clone()
        for i, est in enumerate(estimates):
            if i != factor_idx:
                residual = residual * est.conj()

        # Project onto codebook: Z_j Z_j† residual
        # = Z_j @ (Z_j^H @ residual)
        # sim = Z_j^H @ residual: (m, D)^H × (B, D)^T → (m, B) → transpose → (B, m)
        sim = torch.einsum("md,bd->bm", codebook.conj(), residual)  # (B, m)

        # Project back: Z_j @ sim^T: (m, D)^T × (B, m)^T → (B, D)
        projected = torch.einsum("md,bm->bd", codebook, sim)  # (B, D)

        # Normalize phase
        return self._normalize_phase(projected)


# -----------------------------------------------------------------------
#  Visual Scene Encoder using RHC  (Section 2.6.1)
# -----------------------------------------------------------------------

class RHCSceneEncoder(nn.Module):
    """Encode visual features at positions into a single HD scene vector.

    Following Eq. 10 of Kymn et al.:
        s = Σ_{j,x,y}  h(x) ⊙ v(y) ⊙ d_j · A_j(x,y)

    where h(x), v(y) are RHC position encodings and d_j are random
    feature identity vectors.

    Parameters
    ----------
    n_features : int
        Number of feature types (e.g., number of dictionary elements).
    spatial_size : tuple of int
        (height, width) of the feature map.
    phasor_dim : int
        HD vector dimension.
    moduli : list of int
        RHC moduli for spatial encoding.
    seed : int
    """

    def __init__(
        self,
        n_features: int,
        spatial_size: Tuple[int, int],
        phasor_dim: int = 512,
        moduli: Optional[List[int]] = None,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.spatial_size = spatial_size
        self.phasor_dim = phasor_dim

        if moduli is None:
            moduli = [3, 5, 7]

        # Spatial encoders
        self.h_encoder = ResidueHDEncoder(phasor_dim, moduli, seed=seed)
        self.v_encoder = ResidueHDEncoder(phasor_dim, moduli, seed=seed + 500)

        # Feature identity vectors: random complex unit vectors
        rng = torch.Generator().manual_seed(seed + 999)
        phases = torch.rand(n_features, phasor_dim, generator=rng) * 2 * math.pi
        feat_vectors = torch.complex(torch.cos(phases), torch.sin(phases))
        self.register_buffer("feature_vectors", feat_vectors)  # (n_features, D)

        # Pre-compute position encodings for the grid
        self._precompute_position_codes()

    def _precompute_position_codes(self) -> None:
        """Pre-compute h(x) ⊙ v(y) for all grid positions."""
        H, W = self.spatial_size
        h_codes = []
        for x in range(W):
            h_codes.append(self.h_encoder.encode_complex(torch.tensor(x)))
        h_codes = torch.stack(h_codes)  # (W, D)

        v_codes = []
        for y in range(H):
            v_codes.append(self.v_encoder.encode_complex(torch.tensor(y)))
        v_codes = torch.stack(v_codes)  # (H, D)

        # Combined: (H, W, D)
        pos_codes = v_codes.unsqueeze(1) * h_codes.unsqueeze(0)
        self.register_buffer("pos_codes", pos_codes)

    def encode_features(
        self,
        feature_maps: torch.Tensor,
    ) -> torch.Tensor:
        """Encode sparse feature maps into a single scene vector.

        Parameters
        ----------
        feature_maps : (B, n_features, H, W) float tensor — sparse feature activations.

        Returns
        -------
        scene : (B, 2*D) real tensor — the composite scene vector.
        """
        B, F, H, W = feature_maps.shape
        D = self.phasor_dim

        scene = torch.zeros(B, D, dtype=torch.cfloat, device=feature_maps.device)

        for j in range(F):
            # d_j ⊙ pos_codes: (H, W, D)
            feat_pos = self.feature_vectors[j].unsqueeze(0).unsqueeze(0) * self.pos_codes
            # Weight by activation and sum: (B, H, W) × (H, W, D) → (B, D)
            # PyTorch does not implicitly promote a real activation map in
            # einsum against a complex phasor codebook.  Cast only for the
            # contraction; gradients still propagate to the real input.
            weighted = torch.einsum(
                "bhw,hwd->bd",
                feature_maps[:, j].to(dtype=feat_pos.dtype),
                feat_pos,
            )
            scene = scene + weighted

        # Normalize
        scene = scene / scene.abs().clamp(min=1e-8)

        return torch.cat([scene.real, scene.imag], dim=-1)

    @property
    def output_dim(self) -> int:
        return 2 * self.phasor_dim
