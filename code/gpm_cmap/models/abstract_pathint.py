"""Abstract path-integration model for relational reasoning.

This model directly reuses the navigation GridRNN architecture
(``input_dim=2``, ``hidden_dim=H``) so that weights trained on
physical path integration can be transferred *without any dimension
mismatch*.

Architecture
------------
::

    start_node ──► Embedding ──► Linear ──► h₀
                                             │
    velocity_seq ──────────────► GridRNN ◄───┘
            (T, 2)               │ (unroll T steps, no observations)
                                 ▼
                              h_T  ──► node_head ──► target prediction
                                  └──► periodic_encoder (optional, analysis)

The ``start_encoder`` and ``node_head`` are always trained from scratch.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn

from .grid_rnn import GridRNN
from .multiscale_periodic import build_position_encoder


class AbstractPathIntModel(nn.Module):
    """Model for abstract path-integration reasoning."""

    supports_prevalidated_inputs = True

    def __init__(
        self,
        n_nodes: int,
        hidden_dim: int = 4096,
        content_dim: int = 64,
        nonnegative: bool = True,
        cell_type: str = "rnn",
        use_bias: bool = False,
        dropout: float = 0.0,
        periodic_scales: int = 0,
        use_position_head: bool = True,
        periodic_encoder_kind: str = "periodic_learned",
        rhc_moduli: Optional[list] = None,
        lattice_shape: Optional[tuple] = None,
        start_init: str = "embedding",
        coord_dim: int = 2,
    ) -> None:
        super().__init__()
        self.n_nodes = n_nodes
        self.hidden_dim = hidden_dim
        self.nonnegative = nonnegative
        self.lattice_shape = lattice_shape
        if start_init not in {"embedding", "coordinate"}:
            raise ValueError(f"Unsupported start_init={start_init!r}; use 'embedding' or 'coordinate'.")
        self.start_init = start_init
        self.coord_dim = coord_dim

        self.grid_rnn = GridRNN(
            input_dim=2,
            hidden_dim=hidden_dim,
            activation="relu",
            nonnegative=nonnegative,
            cell_type=cell_type,
            dropout=dropout,
            use_bias=use_bias,
        )

        self.start_embed = nn.Embedding(n_nodes, content_dim)
        self.start_proj = nn.Linear(content_dim, hidden_dim, bias=use_bias)
        # node-count-independent start: initialise h0 from the start
        # COORDINATE rather than a per-node embedding. This removes the
        # "foreign graph must have <= base node count" limitation (the embedding
        # is node-indexed; the coordinate projection is not), enabling
        # cross-graph eval on LARGER graphs.
        self.start_coord_proj = None
        if start_init == "coordinate":
            self.start_coord_proj = nn.Linear(coord_dim, hidden_dim, bias=use_bias)
        self.node_head = nn.Linear(hidden_dim, n_nodes)

        self.periodic_encoder = None
        self._periodic_encoder_kind = periodic_encoder_kind
        if periodic_scales > 0:
            self.periodic_encoder = build_position_encoder(
                kind=periodic_encoder_kind,
                input_dim=hidden_dim,
                n_scales=periodic_scales,
            )

        self.target_rhc_encoder = None
        self.rhc_moduli = rhc_moduli
        if rhc_moduli is not None and self.periodic_encoder is not None:
            from .rhc_encoder import MultiDimResidueEncoder

            phasor_dim = self.periodic_encoder.output_dim // 2
            self.target_rhc_encoder = MultiDimResidueEncoder(
                n_dims=2,
                phasor_dim=phasor_dim,
                moduli=rhc_moduli,
            )

        self.position_head = None
        if use_position_head:
            self.position_head = nn.Linear(hidden_dim, 2)

    def forward(
        self,
        start_node: torch.Tensor,
        velocities: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        start_coords: Optional[torch.Tensor] = None,
        *,
        validate_inputs: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass: blind walk from start_node using velocity sequence.

        ``lengths`` (optional, shape [B]) gives the true number of velocity
        steps per row for ZERO-PADDED variable-length batches. Because the RNN
        keeps evolving on zero input (h_t = ReLU(W_hh h_{t-1}) != h_{t-1}), the
        final hidden state must be read at each row's true length, NOT at the
        padded end. When lengths is None, behaviour is unchanged (uses the last
        timestep for every row).

        ``start_coords`` (optional, shape [B, coord_dim]) is required when this
        model was built with start_init='coordinate': h0 is then initialised
        from the start coordinate, not the per-node embedding, so the model is
        independent of the (graph-specific) node count.
        """
        if start_node.ndim != 1:
            raise ValueError(f"start_node must have shape [B]; got {tuple(start_node.shape)}")
        if velocities.ndim != 3 or velocities.shape[-1] != 2:
            raise ValueError(f"velocities must have shape [B,T,2]; got {tuple(velocities.shape)}")
        if velocities.shape[0] != start_node.shape[0]:
            raise ValueError("start_node and velocities batch dimensions do not match")
        if validate_inputs and not torch.isfinite(velocities).all():
            raise FloatingPointError("velocities contain NaN or Inf")
        if self.start_coord_proj is None:
            if validate_inputs and (
                torch.any(start_node < 0) or torch.any(start_node >= self.n_nodes)
            ):
                raise IndexError(
                    f"start_node ids must lie in [0,{self.n_nodes - 1}] for embedding init"
                )
        if lengths is not None:
            if lengths.ndim != 1 or lengths.shape[0] != start_node.shape[0]:
                raise ValueError("lengths must have shape [B] matching start_node")
            if validate_inputs and (
                torch.any(lengths < 1) or torch.any(lengths > velocities.shape[1])
            ):
                raise ValueError(
                    f"lengths must lie in [1,{velocities.shape[1]}] for the padded batch"
                )

        batch_size = start_node.size(0)

        if self.start_coord_proj is not None:
            if start_coords is None:
                raise ValueError(
                    "start_init='coordinate' requires start_coords in the forward "
                    "call, but none was provided."
                )
            if start_coords.ndim != 2 or start_coords.shape != (batch_size, self.coord_dim):
                raise ValueError(
                    f"start_coords must have shape {(batch_size, self.coord_dim)}; "
                    f"got {tuple(start_coords.shape)}"
                )
            if validate_inputs and not torch.isfinite(start_coords).all():
                raise FloatingPointError("start_coords contain NaN or Inf")
            h0 = self.start_coord_proj(start_coords)
        else:
            start_emb = self.start_embed(start_node)
            h0 = self.start_proj(start_emb)
        if self.nonnegative:
            h0 = F.relu(h0)

        hidden_seq, h_final = self.grid_rnn(velocities, h0=h0)
        if validate_inputs and not torch.isfinite(hidden_seq).all():
            raise FloatingPointError("recurrent hidden state contains NaN or Inf")
        if lengths is not None:
            # Gather the hidden state at each row's true final step (index L-1).
            idx = (lengths.clamp(min=1) - 1).to(hidden_seq.device)
            gather_idx = idx.view(-1, 1, 1).expand(-1, 1, hidden_seq.size(-1))
            h_final = hidden_seq.gather(1, gather_idx).squeeze(1)
        node_logits = self.node_head(h_final)

        outputs: Dict[str, torch.Tensor] = {
            "node_logits": node_logits,
            "hidden_seq": hidden_seq,
            "hidden_final": h_final,
            "h0": h0,
        }

        if self.periodic_encoder is not None:
            periodic_seq = self.periodic_encoder(
                hidden_seq.reshape(-1, self.hidden_dim)
            ).reshape(batch_size, velocities.size(1), -1)
            outputs["periodic_seq"] = periodic_seq

        if self.position_head is not None:
            outputs["position_pred"] = self.position_head(h_final)

        return outputs

    def compute_rhc_target(self, path_coords: torch.Tensor) -> Optional[torch.Tensor]:
        """Compute RHC target encoding for alignment loss."""
        if self.target_rhc_encoder is None:
            return None
        batch_size = path_coords.shape[0]
        coords = path_coords[:, 1:, :]
        steps = coords.shape[1]
        z_target = self.target_rhc_encoder(coords.reshape(-1, 2)).reshape(batch_size, steps, -1)
        return z_target

    def load_nav_checkpoint(
        self,
        checkpoint_path: str | Path,
        transfer_mode: str = "full",
        strict: bool = False,
    ) -> Dict[str, int]:
        """Transfer GridRNN weights from a navigation checkpoint.

        Parameters
        ----------
        checkpoint_path : str | Path
            Path to the ``.pt`` file.
        transfer_mode : str
            ``"full"`` copies both input and recurrent weights.
            ``"recurrent_only"`` copies only the recurrent weight.
        strict : bool
            If True, raise on shape mismatch.
        """
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Nav checkpoint not found: {ckpt_path}")
        # ``input_only`` isolates the velocity-input projection from the
        # recurrent dynamics, completing the full/recurrent/input/none design.
        if transfer_mode not in {"full", "recurrent_only", "input_only"}:
            raise ValueError(
                f"Unsupported transfer_mode: {transfer_mode}. "
                "Choose from: full, recurrent_only, input_only."
            )

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        src_state = ckpt["model_state"]
        dst_state = self.grid_rnn.state_dict()

        if transfer_mode == "full":
            target_keys = ["cell.in_proj.weight", "cell.h_proj.weight"]
        elif transfer_mode == "input_only":
            target_keys = ["cell.in_proj.weight"]
        else:  # recurrent_only
            target_keys = ["cell.h_proj.weight"]

        counts = {"copied": 0, "skipped": 0, "missing": 0}
        for key in target_keys:
            src_key = f"grid_rnn.{key}"
            if src_key not in src_state:
                counts["missing"] += 1
                continue
            if src_state[src_key].shape != dst_state[key].shape:
                if strict:
                    raise ValueError(
                        f"Shape mismatch for {key}: "
                        f"src={src_state[src_key].shape} dst={dst_state[key].shape}"
                    )
                counts["skipped"] += 1
                continue
            dst_state[key] = src_state[src_key]
            counts["copied"] += 1

        self.grid_rnn.load_state_dict(dst_state)
        return counts

    def freeze_grid_rnn_all(self) -> None:
        """Freeze all GridRNN parameters."""
        for param in self.grid_rnn.parameters():
            param.requires_grad = False

    def freeze_grid_rnn_recurrent_only(self) -> None:
        """Freeze only the recurrent GridRNN weight."""
        for name, param in self.grid_rnn.named_parameters():
            if name == "cell.h_proj.weight":
                param.requires_grad = False
            else:
                param.requires_grad = True

    def set_grid_trainability(
        self,
        *,
        input_trainable: bool,
        recurrent_trainable: bool,
    ) -> None:
        """Set W_ih and W_hh trainability independently.

        This is the four-factor mechanism API used by the I3 factorial. It
        changes only ``requires_grad`` flags and therefore preserves all legacy
        checkpoint tensors and key names.
        """
        expected = {"cell.in_proj.weight", "cell.h_proj.weight"}
        seen = set()
        for name, parameter in self.grid_rnn.named_parameters():
            if name == "cell.in_proj.weight":
                parameter.requires_grad = bool(input_trainable)
                seen.add(name)
            elif name == "cell.h_proj.weight":
                parameter.requires_grad = bool(recurrent_trainable)
                seen.add(name)
            else:
                # Biases/dropout-related trainable tensors, when enabled, follow
                # the corresponding matrix where possible and otherwise remain
                # trainable. The formal default uses a bias-free vanilla RNN.
                parameter.requires_grad = True
        missing = expected - seen
        if missing:
            raise RuntimeError(f"GridRNN lacks required factorial parameters: {sorted(missing)}")

    def grid_trainability(self) -> Dict[str, bool]:
        params = dict(self.grid_rnn.named_parameters())
        return {
            "input_trainable": bool(params["cell.in_proj.weight"].requires_grad),
            "recurrent_trainable": bool(params["cell.h_proj.weight"].requires_grad),
        }

    def freeze_grid_rnn_input_only(self) -> None:
        """Freeze only W_ih while leaving W_hh trainable."""
        self.set_grid_trainability(input_trainable=False, recurrent_trainable=True)

    def freeze_grid_rnn(self) -> None:
        """Backward-compatible alias for freezing the full GridRNN."""
        self.freeze_grid_rnn_all()

    def unfreeze_grid_rnn(self) -> None:
        """Unfreeze all GridRNN parameters."""
        for param in self.grid_rnn.parameters():
            param.requires_grad = True

    def recurrent_l2(self) -> torch.Tensor:
        """W_hh regularisation from inner GridRNN."""
        return self.grid_rnn.recurrent_l2()
