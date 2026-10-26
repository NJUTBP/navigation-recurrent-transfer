"""Linear input adapters between target relation inputs and a frozen navigation core.

Implements the "adapter rescue" experiments (低维坐标重映射拯救实验):

    exp7 (2x2 rescue):    u_nav = A · (T · u_target),   A ∈ R^{2x2} trainable, T fixed
    exp8 (symbolic 2xK):  u_nav = A · onehot(action),   A ∈ R^{2xK} trainable

Only the adapter, start encoder and endpoint readout are trainable; the
GridRNN core (W_ih, W_hh) stays frozen. Pre-registered predictions:

  P1 (exp7): for orthogonal/scale transforms T, a trained adapter satisfies
     A·T ≈ I and restores accuracy to near the untransformed baseline, while
     the same adapter on a frozen RANDOM core does not exceed its own
     untransformed baseline. Interpretation: the transferred computation is
     coordinate-equivalent, not tied to one absolute input frame.
  P2 (exp8): with purely symbolic (one-hot) relations, a frozen navigation
     core lets the 2xK adapter organise arbitrary symbols into a 2-D action
     system (two near-antipodal pairs on near-orthogonal axes) faster / from
     fewer episodes than a matched frozen random core.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Union

import numpy as np
import torch
from torch import nn

from .abstract_pathint import AbstractPathIntModel

TransformSpec = Union[str, Dict[str, object]]


def make_transform_matrix(spec: TransformSpec) -> torch.Tensor:
    """Build a fixed 2x2 transform T from a spec.

    Accepted specs (dict with ``name`` plus params, or a bare string):
      - ``none`` / ``identity``
      - ``rotate`` (``degrees``), ``rotate90``
      - ``signflip``  (= rotate 180)
      - ``scale`` (``factor``), ``scale2x``
      - ``aniso`` (``fx``, ``fy``)
      - ``shear`` (``k``)
    """
    if isinstance(spec, str):
        spec = {"name": spec}
    name = str(spec["name"])
    if name in {"none", "identity"}:
        return torch.eye(2)
    if name in {"rotate", "rotate90"}:
        degrees = 90.0 if name == "rotate90" else float(spec.get("degrees", 90.0))
        r = math.radians(degrees)
        c, s = math.cos(r), math.sin(r)
        return torch.tensor([[c, -s], [s, c]], dtype=torch.float32)
    if name == "signflip":
        return -torch.eye(2)
    if name in {"scale", "scale2x"}:
        factor = 2.0 if name == "scale2x" else float(spec.get("factor", 2.0))
        return factor * torch.eye(2)
    if name == "aniso":
        fx = float(spec.get("fx", 2.0))
        fy = float(spec.get("fy", 0.5))
        return torch.tensor([[fx, 0.0], [0.0, fy]], dtype=torch.float32)
    if name == "shear":
        k = float(spec.get("k", 0.5))
        return torch.tensor([[1.0, k], [0.0, 1.0]], dtype=torch.float32)
    raise ValueError(f"Unknown transform spec: {spec!r}")


def transform_label(spec: TransformSpec) -> str:
    """Short, filename-safe label for a transform spec."""
    if isinstance(spec, str):
        spec = {"name": spec}
    name = str(spec["name"])
    if name in {"rotate"}:
        return f"rot{float(spec.get('degrees', 90.0)):g}"
    if name in {"scale"}:
        return f"scale{float(spec.get('factor', 2.0)):g}x"
    if name == "aniso":
        return f"aniso{float(spec.get('fx', 2.0)):g}_{float(spec.get('fy', 0.5)):g}"
    if name == "shear":
        return f"shear{float(spec.get('k', 0.5)):g}"
    return name


class AdapterPathIntModel(nn.Module):
    """AbstractPathIntModel plus a fixed transform T and a trainable adapter A.

    Forward (2-D mode, ``in_dim == 2``):
        u = A @ (T @ v_t)  for every step t, then the frozen base model runs.
    Forward (symbolic mode, ``in_dim == K``):
        velocities are one-hot action codes [B,T,K]; u = A @ onehot.

    The wrapper keeps the same call signature the task code uses
    (``start_node, velocities, lengths, start_coords, validate_inputs``), so
    ``train_pathint_epoch`` / ``evaluate_pathint*`` work unchanged.
    """

    supports_prevalidated_inputs = True

    def __init__(
        self,
        base: AbstractPathIntModel,
        in_dim: int = 2,
        transform: Optional[torch.Tensor] = None,
        adapter_init: str = "identity",
        adapter_init_scale: float = 0.02,
    ) -> None:
        super().__init__()
        if in_dim < 1:
            raise ValueError(f"in_dim must be positive; got {in_dim}")
        self.base = base
        self.in_dim = int(in_dim)

        T = torch.eye(2) if transform is None else transform.detach().clone().float()
        if T.shape != (2, 2):
            raise ValueError(f"transform must be 2x2; got {tuple(T.shape)}")
        if self.in_dim != 2 and not torch.allclose(T, torch.eye(2)):
            raise ValueError("A non-identity transform is only defined for in_dim == 2")
        self.register_buffer("transform_T", T)

        self.adapter = nn.Linear(self.in_dim, 2, bias=False)
        with torch.no_grad():
            if adapter_init == "identity":
                if self.in_dim != 2:
                    raise ValueError("adapter_init='identity' requires in_dim == 2")
                self.adapter.weight.copy_(torch.eye(2))
            elif adapter_init == "small":
                # Match the frozen core's expected input magnitude
                # (velocity_scale, default 0.02); a badly scaled init can park
                # the frozen ReLU net far outside its trained operating range.
                nn.init.uniform_(self.adapter.weight, -adapter_init_scale, adapter_init_scale)
            else:
                raise ValueError(f"Unsupported adapter_init: {adapter_init!r}")

    # -- passthroughs the task code reads ---------------------------------
    @property
    def n_nodes(self) -> int:
        return self.base.n_nodes

    @property
    def start_init(self) -> str:
        return self.base.start_init

    @property
    def grid_rnn(self):
        return self.base.grid_rnn

    @property
    def node_head(self):
        return self.base.node_head

    def recurrent_l2(self):
        return self.base.recurrent_l2()

    # -- training setup ----------------------------------------------------
    def configure_trainable(self) -> None:
        """Freeze the GridRNN core and every analysis-only head.

        Trainable afterwards: adapter, start encoder (embedding + projection,
        or coordinate projection), endpoint readout. Position head and
        periodic encoder are frozen so no coordinate supervision can leak in
        (use ``position_loss_weight: 0.0`` in the config as well).
        """
        self.base.freeze_grid_rnn_all()
        if self.base.position_head is not None:
            for p in self.base.position_head.parameters():
                p.requires_grad = False
        if self.base.periodic_encoder is not None:
            for p in self.base.periodic_encoder.parameters():
                p.requires_grad = False
        if getattr(self.base, "target_rhc_encoder", None) is not None:
            for p in self.base.target_rhc_encoder.parameters():
                p.requires_grad = False

    def freeze_adapter_at_identity(self) -> None:
        """Control arm: keep A = I fixed (readout retraining alone)."""
        if self.in_dim != 2:
            raise ValueError("identity-frozen adapter requires in_dim == 2")
        with torch.no_grad():
            self.adapter.weight.copy_(torch.eye(2))
        self.adapter.weight.requires_grad = False

    # -- forward -------------------------------------------------------------
    def forward(
        self,
        start_node: torch.Tensor,
        velocities: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        start_coords: Optional[torch.Tensor] = None,
        *,
        validate_inputs: bool = True,
    ) -> Dict[str, torch.Tensor]:
        if velocities.ndim != 3 or velocities.shape[-1] != self.in_dim:
            raise ValueError(
                f"velocities must have shape [B,T,{self.in_dim}]; got {tuple(velocities.shape)}"
            )
        v = velocities
        if self.in_dim == 2:
            v = torch.matmul(v, self.transform_T.transpose(0, 1))
        u = self.adapter(v)
        return self.base(
            start_node=start_node,
            velocities=u,
            lengths=lengths,
            start_coords=start_coords,
            validate_inputs=validate_inputs,
        )

    # -- diagnostics -----------------------------------------------------------
    def adapter_matrix(self) -> np.ndarray:
        return self.adapter.weight.detach().cpu().numpy().astype(np.float64)

    def alignment_report(self) -> Dict[str, object]:
        """Quantify whether A inverted T (only meaningful for in_dim == 2).

        Returns Frobenius errors of M = A·T against I, the polar-decomposition
        rotation angle / determinant sign / singular values of M, and the
        relative error of A against T^{-1}.
        """
        A = self.adapter_matrix()
        T = self.transform_T.detach().cpu().numpy().astype(np.float64)
        M = A @ T
        eye = np.eye(2)
        identity_error = float(np.linalg.norm(M - eye, ord="fro"))
        rel_identity_error = identity_error / float(np.linalg.norm(eye, ord="fro"))
        U, S, Vt = np.linalg.svd(M)
        R = U @ Vt
        angle_deg = float(math.degrees(math.atan2(R[1, 0], R[0, 0])))
        det_sign = float(np.sign(np.linalg.det(R)))
        T_inv = np.linalg.inv(T)
        rel_error_vs_T_inv = float(
            np.linalg.norm(A - T_inv, ord="fro") / max(np.linalg.norm(T_inv, ord="fro"), 1e-12)
        )
        return {
            "adapter": A.tolist(),
            "transform": T.tolist(),
            "product_AT": M.tolist(),
            "identity_error_fro": identity_error,
            "rel_identity_error": rel_identity_error,
            "polar_angle_deg": angle_deg,
            "polar_det_sign": det_sign,
            "singular_values_AT": [float(s) for s in S],
            "rel_error_vs_T_inv": rel_error_vs_T_inv,
        }
