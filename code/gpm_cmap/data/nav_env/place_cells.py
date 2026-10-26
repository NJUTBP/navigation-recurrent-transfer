"""Place cell encoder / decoder.

Upgrades vs original (learning from reference code)
----------------------------------------------------
1. **Difference of Gaussians (DoG)** — centre-surround antagonism produces
   sharper place fields and helps the RNN learn cleaner grid patterns.
   Controlled by ``use_dog`` and ``surround_scale``.
2. **Softmax normalisation** — reference uses softmax (not sum-division),
   which is more numerically stable and produces a proper probability
   distribution for cross-entropy training.
3. **GPU-friendly batch computation** — ``encode`` now handles arbitrary
   leading dimensions without explicit loops.
4. **Top-k decode** — ``decode`` averages the centres of the top-k most
   active place cells (reference uses k=3) instead of a weighted sum,
   giving more robust position readout.
5. **Periodic boundary support** — handles torus-wrapped distances.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class PlaceCellEncoder:
    n_cells: int
    arena_size: float
    sigma: float
    seed: int = 42
    use_dog: bool = True              # [NEW] Difference of Gaussians
    surround_scale: float = 2.0       # [NEW] surround scale parameter
    surround_scale_mode: str = "variance"  # "variance" matches Sorscher/Banino; "sigma" keeps old GPM behavior
    periodic: bool = False            # [NEW] periodic boundaries
    decode_top_k: int = 3             # [NEW] top-k decoding

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        self.centers = rng.uniform(
            low=-self.arena_size / 2.0,
            high=self.arena_size / 2.0,
            size=(self.n_cells, 2),
        ).astype(np.float32)

    def encode(self, positions: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Encode positions to place cell activations.

        Parameters
        ----------
        positions : (..., 2)   arbitrary leading dims

        Returns
        -------
        activations : (..., n_cells)  probability distribution per position
        """
        if isinstance(positions, np.ndarray):
            pos = torch.from_numpy(positions).float()
        else:
            pos = positions.float()

        centers = torch.from_numpy(self.centers).to(pos.device)

        # distance: (..., 1, 2) - (n_cells, 2) -> (..., n_cells, 2)
        diff = pos.unsqueeze(-2) - centers

        # [UPGRADE] periodic boundary handling
        if self.periodic:
            diff = torch.remainder(diff + self.arena_size / 2, self.arena_size) - self.arena_size / 2

        sq_dist = (diff ** 2).sum(dim=-1)  # (..., n_cells)

        # [UPGRADE] softmax normalisation (reference approach, more stable)
        activations = F.softmax(-sq_dist / (2 * self.sigma ** 2), dim=-1)

        # [UPGRADE] Difference of Gaussians
        if self.use_dog:
            # The Sorscher/Banino reference code defines ``surround_scale`` as a
            # VARIANCE ratio: denominator = 2 * surround_scale * sigma**2.
            # A previous GPM-CMap version treated it as a SIGMA ratio:
            # denominator = 2 * (surround_scale * sigma)**2.  For
            # surround_scale=2 this changes the surround variance by 2x and
            # is not a neutral refactor.  Keep both modes explicit.
            mode = str(self.surround_scale_mode).lower()
            if mode == "variance":
                surround_denom = 2.0 * self.surround_scale * (self.sigma ** 2)
            elif mode == "sigma":
                surround_denom = 2.0 * ((self.surround_scale * self.sigma) ** 2)
            else:
                raise ValueError(
                    "surround_scale_mode must be 'variance' or 'sigma', "
                    f"got {self.surround_scale_mode!r}"
                )
            surround = F.softmax(-sq_dist / surround_denom, dim=-1)
            activations = activations - surround
            # shift to non-negative and re-normalise
            min_val = activations.min(dim=-1, keepdim=True).values
            activations = activations - min_val
            activations = activations / activations.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        return activations

    def decode(self, activities: torch.Tensor) -> torch.Tensor:
        """Decode place cell activities back to (x, y).

        [UPGRADE] Uses top-k averaging (reference style) instead of
        full weighted sum.  More robust to noisy predictions.
        """
        centers = torch.from_numpy(self.centers).to(activities.device)
        _, topk_idx = torch.topk(activities, k=min(self.decode_top_k, self.n_cells), dim=-1)
        return centers[topk_idx].mean(dim=-2)
