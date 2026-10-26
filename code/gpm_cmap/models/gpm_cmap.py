"""GPM-CMap: Generalized Planning Model with Cognitive Map.

Phase-2 upgrades
-----------------
1. **Position encoder ablation** — ``periodic_encoder`` is now built by
   ``build_position_encoder(kind, ...)``, supporting identity / rbf / mlp /
   periodic_fixed / periodic_learned comparisons via a single config key.

2. **Target periodic encoder + structural coordinates ξ(s)** — a second
   encoder ``target_encoder`` maps *true* structural coordinates (looked
   up from a per-node table) into the same periodic space.  The training
   loop uses this to compute the auxiliary loss:

       loss_periodic = ||z_t − z_target_t||²

   where ``z_t = periodic_encoder(hidden_t)`` and
   ``z_target_t = target_encoder(ξ(s_t))``.

3. The structural coordinate table ``structural_coords`` is registered as
   a non-learnable buffer (shape ``[num_nodes, coord_dim]``) and populated
   once at experiment setup via ``register_structural_coords()``.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn

from .conjunctive import ConjunctiveLayer
from .content_encoder import build_content_encoder
from .grid_rnn import GridRNN
from .hebb_memory import HebbianMemory
from .inhibitory import LateralInhibition
from .multiscale_periodic import (
    MultiScalePeriodicEncoder,
    build_position_encoder,
)


class GPMCMap(nn.Module):
    """Full GPM-CMap model with Phase-2 periodic-encoding upgrades.

    New constructor parameters (all optional, backward-compatible)
    ---------------------------------------------------------------
    position_encoder_kind : str
        Which encoder to use for ``periodic_encoder`` (the model's
        representation).  Default ``"periodic_learned"``.
    target_encoder_kind : str
        Which encoder to use for ``target_encoder`` (the ground-truth
        coordinate encoder).  Default ``"periodic_learned"``.
    coord_dim : int
        Dimensionality of structural coordinates ξ(s).  Needed by
        ``target_encoder``.  Default 2.
    freq_init : str
        ``"logspace"`` or ``"geometric"`` for periodic encoders.
    """

    def __init__(
        self,
        action_vocab: int,
        action_embed_dim: int,
        hidden_dim: int,
        content_dim: int,
        periodic_scales: int,
        conjunctive_dim: int,
        observation_mode: str = "categorical",
        observation_vocab: Optional[int] = None,
        image_channels: int = 1,
        use_memory: bool = True,
        memory_decay: float = 0.99,
        hebb_lr: float = 0.05,
        nonnegative: bool = True,
        node_vocab: Optional[int] = None,
        use_lateral_inhibition: bool = False,
        lateral_k_winners: int = 20,
        learnable_hebb: bool = False,
        memory_write_modes: Optional[list] = None,
        cell_type: str = "rnn",
        dropout: float = 0.0,
        use_bias: bool = False,
        # --- Phase 2 additions ---
        position_encoder_kind: str = "periodic_learned",
        target_encoder_kind: str = "periodic_learned",
        coord_dim: int = 2,
        freq_init: str = "logspace",
        # --- T4: extended (reverse + autoassociative) memory retrieval ---
        fuse_extended_retrievals: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.coord_dim = coord_dim
        self.fuse_extended_retrievals = fuse_extended_retrievals

        # Action encoder
        self.action_encoder = nn.Embedding(action_vocab, action_embed_dim)

        # Grid RNN (path integration backbone)
        self.grid_rnn = GridRNN(
            action_embed_dim, hidden_dim, activation="relu",
            nonnegative=nonnegative, cell_type=cell_type, dropout=dropout,
            use_bias=use_bias,
        )

        # ------------------------------------------------------------------
        #  Position encoder: hidden_state → periodic representation z_t
        #  (configurable for ablation: identity / rbf / mlp / periodic_*)
        # ------------------------------------------------------------------
        self.position_encoder_kind = position_encoder_kind
        self.periodic_encoder = build_position_encoder(
            kind=position_encoder_kind,
            input_dim=hidden_dim,
            n_scales=periodic_scales,
            freq_init=freq_init,
        )
        periodic_dim = self.periodic_encoder.output_dim

        # ------------------------------------------------------------------
        #  Target encoder: ξ(s) → z_target  (for auxiliary periodic loss)
        #  This is a SEPARATE encoder that encodes true structural coords.
        #  Its output_dim must match periodic_dim.
        # ------------------------------------------------------------------
        self.target_encoder_kind = target_encoder_kind
        self.target_encoder = build_position_encoder(
            kind=target_encoder_kind,
            input_dim=coord_dim,
            n_scales=periodic_scales,
            output_dim=periodic_dim,       # force same dim as model encoder
            freq_init=freq_init,
        )
        # [T12] The target_encoder is a FIXED teacher that produces ground-truth
        # structural-coordinate targets; every loss that consumes its output
        # detaches it. Freeze it explicitly so this correctness no longer relies
        # on each caller remembering to .detach() — a future non-detached use
        # would otherwise silently start training the teacher. Freezing is
        # behaviourally inert today (these params already receive zero gradient).
        for _p in self.target_encoder.parameters():
            _p.requires_grad = False
        # Shape: (num_nodes, coord_dim) — registered as buffer so it
        # moves with the model to GPU and is saved in state_dict.
        self._has_structural_coords = False
        self.register_buffer(
            "structural_coords",
            torch.zeros(1, coord_dim),       # placeholder; resized later
        )

        # Lateral inhibition (winner-take-all) on hidden states
        self.use_lateral_inhibition = use_lateral_inhibition
        self.lateral_inhibition = (
            LateralInhibition(k_winners=lateral_k_winners)
            if use_lateral_inhibition else None
        )

        # Content encoder
        if observation_mode == "categorical":
            if observation_vocab is None:
                raise ValueError("observation_vocab is required for categorical mode")
            self.content_encoder = build_content_encoder(
                "categorical",
                vocab_size=observation_vocab,
                embedding_dim=content_dim,
            )
        elif observation_mode == "image":
            self.content_encoder = build_content_encoder(
                "image",
                output_dim=content_dim,
                in_channels=image_channels,
            )
        elif observation_mode == "vector":
            raise ValueError(
                "Use build_content_encoder directly for vector mode "
                "in custom experiments."
            )
        else:
            raise ValueError(f"Unknown observation mode: {observation_mode}")

        # Conjunctive layer
        self.conjunctive = ConjunctiveLayer(
            periodic_dim, content_dim, conjunctive_dim, mode="concat",
        )

        # Hebbian memory
        self.use_memory = use_memory
        # T4: reverse (content->place) and autoassociative reads only carry
        # signal if the corresponding write modes populate M_cp / M_auto.
        # When extended-retrieval fusion is requested but the caller did not
        # pin write_modes, enable all three modes so the fused reverse/auto
        # inputs are not constant-zero (which would make the ablation vacuous).
        if memory_write_modes is None and fuse_extended_retrievals:
            memory_write_modes = [
                "place_to_content",
                "content_to_place",
                "autoassociative",
            ]
        self.memory_write_modes = memory_write_modes
        self.memory = HebbianMemory(
            key_dim=periodic_dim,
            value_dim=content_dim,
            auto_dim=conjunctive_dim,
            decay=memory_decay,
            lr_hebb=hebb_lr,
            differentiable=learnable_hebb,
            write_modes=memory_write_modes,
        )

        # Feature fuser
        # Base inputs: [z_t, e_t, retrieved (place->content), conj].
        # T4 (optional): also fuse [retrieved_rev (content->place, dim=periodic),
        # retrieved_auto (autoassociative, dim=conjunctive)].
        fuser_in_dim = periodic_dim + content_dim + content_dim + conjunctive_dim
        if fuse_extended_retrievals:
            fuser_in_dim += periodic_dim + conjunctive_dim
        self.feature_fuser = nn.Sequential(
            nn.Linear(fuser_in_dim, hidden_dim),
            nn.ReLU(),
        )

        # Task heads
        self.node_head = (
            None if node_vocab is None
            else nn.Linear(hidden_dim, node_vocab)
        )
        self.content_reconstruction_head = nn.Linear(hidden_dim, content_dim)

    # ------------------------------------------------------------------
    #  Structural coordinates management
    # ------------------------------------------------------------------

    def register_structural_coords(self, coords_tensor: torch.Tensor) -> None:
        """Register structural coordinates ξ(s) for all graph nodes.

        Parameters
        ----------
        coords_tensor : Tensor of shape (num_nodes, coord_dim)
            Row *i* = ξ for node with id *i*.  Produced by
            ``graph_positional.coords_dict_to_tensor()``.
        """
        assert coords_tensor.ndim == 2, (
            f"Expected (num_nodes, coord_dim), got {coords_tensor.shape}"
        )
        # Replace the placeholder buffer
        self.structural_coords = coords_tensor.to(
            self.structural_coords.device
        )
        self._has_structural_coords = True

    @property
    def has_structural_coords(self) -> bool:
        return self._has_structural_coords

    # ------------------------------------------------------------------
    #  Navigation auxiliary coordinate head (T5)
    # ------------------------------------------------------------------

    def build_nav_aux_head(self, coord_dim: Optional[int] = None) -> None:
        """Create the periodic->coordinate auxiliary head used by the optional
        ``nav_loss_weight`` regression term.

        This MUST be called at model-setup time (before the optimizer is built)
        so the head's parameters are captured by the optimizer. The previous
        implementation created this head lazily inside the training loop, after
        the optimizer already existed, so it never received gradients (rule #5:
        deterministic module construction belongs in setup, not the train step).
        """
        cdim = coord_dim if coord_dim is not None else self.coord_dim
        periodic_dim = self.periodic_encoder.output_dim
        # Place the head on the same device as the rest of the model. The model
        # may already have been moved to GPU before this is called (build_model
        # returns model.to(device)), so infer device from an existing parameter.
        device = next(self.parameters()).device
        self._nav_aux_head = nn.Linear(periodic_dim, cdim).to(device)

    @property
    def has_nav_aux_head(self) -> bool:
        return hasattr(self, "_nav_aux_head")

    # ------------------------------------------------------------------
    #  Memory management
    # ------------------------------------------------------------------

    def reset_memory(self, batch_size: int, device: torch.device | str) -> None:
        self.memory.reset(batch_size, device)

    # ------------------------------------------------------------------
    #  Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        actions: torch.Tensor,
        observations: torch.Tensor,
        h0: Optional[torch.Tensor] = None,
        reset_memory: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Run one episode through the full GPM-CMap pipeline.

        Returns
        -------
        dict with keys:
            hidden_seq, hidden_last, periodic_seq, content_seq,
            retrievals, conjunctions, features, content_reconstruction,
            node_logits (if node_head exists),
            **periodic_target** (if structural coords registered and
            observations are node IDs).
        """
        if actions.ndim != 2:
            raise ValueError("actions must have shape [batch, seq]")
        batch_size, seq_len = actions.shape
        if reset_memory:
            self.reset_memory(batch_size, actions.device)

        # 1. Encode actions and run RNN
        action_vec = self.action_encoder(actions)
        hidden_seq, hidden_last = self.grid_rnn(action_vec, h0=h0)

        # Apply lateral inhibition if enabled
        if self.use_lateral_inhibition and self.lateral_inhibition is not None:
            B, S, D = hidden_seq.shape
            hidden_seq = self.lateral_inhibition(
                hidden_seq.reshape(-1, D)
            ).reshape(B, S, D)

        # 2. Periodic encoding of hidden states → z_t
        periodic_seq = self.periodic_encoder(
            hidden_seq.reshape(-1, hidden_seq.shape[-1])
        ).reshape(batch_size, seq_len, -1)

        # 3. Content encoding
        if observations.ndim == 2:
            content_seq = self.content_encoder(observations)
        elif observations.ndim == 5:
            flat = observations.reshape(
                batch_size * seq_len, *observations.shape[2:]
            )
            content_seq = self.content_encoder(flat).reshape(
                batch_size, seq_len, -1
            )
        else:
            raise ValueError(
                f"Unsupported observation shape: {observations.shape}"
            )

        # 4. Sequential processing: memory read/write + conjunctive binding
        retrievals = []
        retrievals_reverse = []
        retrievals_auto = []
        conjunctions = []
        fused_features = []
        for t in range(seq_len):
            z_t = periodic_seq[:, t]
            e_t = content_seq[:, t]

            # Read BEFORE write (test recall of previously stored items)
            if self.use_memory:
                retrieved = self.memory.read(z_t)                   # position → content
                retrieved_rev = self.memory.read_reverse(e_t)       # content → position
            else:
                retrieved = torch.zeros_like(e_t)
                retrieved_rev = torch.zeros_like(z_t)

            conj = self.conjunctive(z_t, e_t)

            if self.use_memory:
                retrieved_auto = self.memory.read_auto(conj)        # autoassociative
            else:
                retrieved_auto = torch.zeros_like(conj)

            # Write AFTER read
            if self.use_memory:
                self.memory.write(z_t, e_t, conjunctive=conj)

            if self.fuse_extended_retrievals:
                fuser_input = torch.cat(
                    [z_t, e_t, retrieved, conj, retrieved_rev, retrieved_auto],
                    dim=-1,
                )
            else:
                fuser_input = torch.cat([z_t, e_t, retrieved, conj], dim=-1)
            fused = self.feature_fuser(fuser_input)
            retrievals.append(retrieved)
            retrievals_reverse.append(retrieved_rev)
            retrievals_auto.append(retrieved_auto)
            conjunctions.append(conj)
            fused_features.append(fused)

        retrievals_t = torch.stack(retrievals, dim=1)
        retrievals_reverse_t = torch.stack(retrievals_reverse, dim=1)
        retrievals_auto_t = torch.stack(retrievals_auto, dim=1)
        conjunctions_t = torch.stack(conjunctions, dim=1)
        fused_t = torch.stack(fused_features, dim=1)

        # 5. Build output dict
        outputs: Dict[str, torch.Tensor] = {
            "hidden_seq": hidden_seq,
            "hidden_last": hidden_last,
            "periodic_seq": periodic_seq,
            "content_seq": content_seq,
            "retrievals": retrievals_t,
            "retrievals_reverse": retrievals_reverse_t,
            "retrievals_auto": retrievals_auto_t,
            "conjunctions": conjunctions_t,
            "features": fused_t,
            "content_reconstruction": self.content_reconstruction_head(fused_t),
        }
        if self.node_head is not None:
            outputs["node_logits"] = self.node_head(fused_t)

        # 6. [Phase 2] Compute periodic target z_target from true ξ(s)
        #    Only possible when structural coords are registered AND
        #    observations are integer node IDs (categorical mode).
        if self._has_structural_coords and observations.ndim == 2:
            # observations: (B, S) long — node IDs
            # structural_coords: (num_nodes, coord_dim)
            xi = self.structural_coords[observations]      # (B, S, coord_dim)
            z_target = self.target_encoder(
                xi.reshape(-1, self.coord_dim)
            ).reshape(batch_size, seq_len, -1)
            outputs["periodic_target"] = z_target

        return outputs

    # ------------------------------------------------------------------
    #  Navigation transfer utilities
    # ------------------------------------------------------------------

    def load_grid_rnn_state(
        self, state_dict: Dict[str, torch.Tensor], strict: bool = False,
    ) -> None:
        self.grid_rnn.load_state_dict(state_dict, strict=strict)

    def freeze_grid_rnn(self) -> None:
        for param in self.grid_rnn.parameters():
            param.requires_grad = False

    def recurrent_l2(self) -> torch.Tensor:
        """Expose W_hh regularisation from inner GridRNN."""
        return self.grid_rnn.recurrent_l2()
