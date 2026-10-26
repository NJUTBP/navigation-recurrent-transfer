"""Hebbian associative memory with three write modes.

Write modes (per roadmap §3.1.2)
---------------------------------
- ``place_to_content``  (mode 2): M += η · g_t ⊗ e_t   — query by position, retrieve content
- ``content_to_place``  (mode 3): M += η · e_t ⊗ g_t   — query by content, retrieve position
- ``autoassociative``   (mode 1): M += η · p_t ⊗ p_t   — pattern completion on conjunctive repr

The default mode is ``place_to_content`` (backward-compatible with the
original single-mode implementation).

Read helpers
------------
- ``read(query)``              → M_pc^T @ query  (position → content)
- ``read_reverse(query)``      → M_cp^T @ query  (content → position)
- ``read_auto(query)``         → M_auto^T @ query (autoassociative completion)

All three matrices live in ``self.M_pc``, ``self.M_cp``, ``self.M_auto``
so they can be independently reset / inspected.
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import nn


class HebbianMemory(nn.Module):
    """Fast Hebbian memory supporting three biologically-motivated write modes.

    Parameters
    ----------
    key_dim : int
        Dimensionality of the *position* (grid/periodic) key vector.
    value_dim : int
        Dimensionality of the *content* value vector.
    auto_dim : int or None
        Dimensionality of the conjunctive representation used by the
        autoassociative mode.  If ``None``, defaults to ``key_dim``.
    decay : float
        Exponential decay applied to the memory matrix before each write.
    lr_hebb : float
        Learning rate for the outer-product update.
    differentiable : bool
        If True, the memory matrix stays in the computation graph.
    write_modes : list of str
        Which modes to activate during ``write()``.  Subset of
        ``{"place_to_content", "content_to_place", "autoassociative"}``.
        Default: ``["place_to_content"]`` (backward-compatible).
    """

    VALID_MODES = frozenset({"place_to_content", "content_to_place", "autoassociative"})

    def __init__(
        self,
        key_dim: int,
        value_dim: int,
        auto_dim: Optional[int] = None,
        decay: float = 0.99,
        lr_hebb: float = 0.05,
        differentiable: bool = False,
        write_modes: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.auto_dim = auto_dim if auto_dim is not None else key_dim
        self.decay = decay
        self.lr_hebb = lr_hebb
        self.differentiable = differentiable

        if write_modes is None:
            write_modes = ["place_to_content"]
        for m in write_modes:
            if m not in self.VALID_MODES:
                raise ValueError(f"Unknown write mode '{m}'. Valid: {self.VALID_MODES}")
        self.write_modes = list(write_modes)

        # Memory matrices — lazily initialised per batch
        self.M_pc: Optional[torch.Tensor] = None    # (B, key_dim, value_dim)
        self.M_cp: Optional[torch.Tensor] = None    # (B, value_dim, key_dim)
        self.M_auto: Optional[torch.Tensor] = None  # (B, auto_dim, auto_dim)

    # ------------------------------------------------------------------
    #  Backward-compatible property: ``self.memory`` → ``self.M_pc``
    # ------------------------------------------------------------------
    @property
    def memory(self) -> Optional[torch.Tensor]:
        return self.M_pc

    @memory.setter
    def memory(self, value: Optional[torch.Tensor]) -> None:
        self.M_pc = value

    # ------------------------------------------------------------------
    #  Reset / ensure
    # ------------------------------------------------------------------

    def reset(self, batch_size: int, device: torch.device | str) -> None:
        """Zero-initialise all memory matrices."""
        self.M_pc = torch.zeros(batch_size, self.key_dim, self.value_dim, device=device)
        self.M_cp = torch.zeros(batch_size, self.value_dim, self.key_dim, device=device)
        self.M_auto = torch.zeros(batch_size, self.auto_dim, self.auto_dim, device=device)

    def _ensure(self, batch_size: int, device: torch.device | str) -> None:
        target_device = torch.device(device)
        if (
            self.M_pc is None
            or self.M_pc.size(0) != batch_size
            or self.M_pc.device.type != target_device.type
        ):
            self.reset(batch_size, device)

    # ------------------------------------------------------------------
    #  Write
    # ------------------------------------------------------------------

    def write(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        conjunctive: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Write to all active memory matrices.

        Parameters
        ----------
        key : (B, key_dim)   — position / periodic representation z_t.
        value : (B, value_dim)  — content representation e_t.
        conjunctive : (B, auto_dim) or None
            Conjunctive representation p_t; required if autoassociative is active.

        Returns
        -------
        M_pc : (B, key_dim, value_dim)  — the place→content matrix (for compat).
        """
        B = key.size(0)
        self._ensure(B, key.device)
        key_n = F.normalize(key, dim=-1)
        value_n = F.normalize(value, dim=-1)

        if "place_to_content" in self.write_modes:
            outer_pc = key_n.unsqueeze(-1) * value_n.unsqueeze(-2)       # (B, K, V)
            updated = self.decay * self.M_pc + self.lr_hebb * outer_pc
            self.M_pc = updated if self.differentiable else updated.detach()

        if "content_to_place" in self.write_modes:
            outer_cp = value_n.unsqueeze(-1) * key_n.unsqueeze(-2)       # (B, V, K)
            updated = self.decay * self.M_cp + self.lr_hebb * outer_cp
            self.M_cp = updated if self.differentiable else updated.detach()

        if "autoassociative" in self.write_modes:
            if conjunctive is None:
                raise ValueError(
                    "conjunctive (p_t) must be provided for autoassociative write"
                )
            p_n = F.normalize(conjunctive, dim=-1)
            outer_aa = p_n.unsqueeze(-1) * p_n.unsqueeze(-2)            # (B, A, A)
            updated = self.decay * self.M_auto + self.lr_hebb * outer_aa
            self.M_auto = updated if self.differentiable else updated.detach()

        return self.M_pc

    # ------------------------------------------------------------------
    #  Read
    # ------------------------------------------------------------------

    def read(self, query: torch.Tensor) -> torch.Tensor:
        """Position → content:  out = M_pc^T @ query.  (B, key_dim) → (B, value_dim)"""
        self._ensure(query.size(0), query.device)
        q = F.normalize(query, dim=-1)
        return torch.einsum("bkv,bk->bv", self.M_pc, q)

    def read_reverse(self, query: torch.Tensor) -> torch.Tensor:
        """Content → position:  out = M_cp^T @ query.  (B, value_dim) → (B, key_dim)"""
        self._ensure(query.size(0), query.device)
        q = F.normalize(query, dim=-1)
        return torch.einsum("bvk,bv->bk", self.M_cp, q)

    def read_auto(self, query: torch.Tensor) -> torch.Tensor:
        """Autoassociative completion.  (B, auto_dim) → (B, auto_dim)"""
        self._ensure(query.size(0), query.device)
        q = F.normalize(query, dim=-1)
        return torch.einsum("bij,bi->bj", self.M_auto, q)

    # ------------------------------------------------------------------
    #  Diagnostics
    # ------------------------------------------------------------------

    def memory_norm(self) -> dict:
        """Return Frobenius norms of each active matrix for monitoring."""
        norms = {}
        if self.M_pc is not None:
            norms["M_pc"] = float(self.M_pc.norm().item())
        if self.M_cp is not None:
            norms["M_cp"] = float(self.M_cp.norm().item())
        if self.M_auto is not None:
            norms["M_auto"] = float(self.M_auto.norm().item())
        return norms
