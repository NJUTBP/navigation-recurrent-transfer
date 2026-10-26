"""Non-navigation sequence-control inputs for CTL-01.

The control model must see the same two-dimensional input *operating range* as
navigation pretraining without inheriting a navigational trajectory.  The
formal sampler therefore obtains the actual post-boundary-correction step
magnitudes from :class:`OpenFieldSampler`, shuffles those magnitudes across
batch and time, and combines them with independent uniformly sampled angles.
For every generated batch, the multiset of input magnitudes is consequently
identical to a navigation batch generated with the same seed and open-field
configuration, while heading continuity, position, wall identity and spatial
path structure are absent from the vectors shown to the model.

Two dataset forms mirror ``gpm_cmap.data.nav_env.dataset``:

* ``SequenceControlStaticDataset`` creates a reproducible validation set.
* ``SequenceControlOnlineDataset`` yields one already-batched sample per
  optimization step and is used with ``DataLoader(..., batch_size=None)``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset

from .open_field import OpenFieldConfig, OpenFieldSampler


@dataclass(frozen=True)
class SequenceControlConfig:
    """Configuration for the delayed-reconstruction control input stream."""

    seq_len: int = 20
    dt: float = 0.02
    speed_scale: float = 0.8168140899
    sigma_theta: float = 11.52
    min_speed: float = 0.0
    max_speed: float = 100.0
    arena_size: float = 2.2
    border_region: float = 0.03
    boundary_mode: str = "avoid"
    delay: int = 10
    magnitude_mode: str = "navigation_matched"
    shuffle_magnitudes: bool = True

    def __post_init__(self) -> None:
        if self.seq_len < 2:
            raise ValueError("seq_len must be at least 2")
        if not 1 <= self.delay < self.seq_len:
            raise ValueError(
                f"delay must satisfy 1 <= delay < seq_len; got delay={self.delay}, "
                f"seq_len={self.seq_len}"
            )
        if self.dt <= 0 or self.speed_scale <= 0:
            raise ValueError("dt and speed_scale must be positive")
        if self.min_speed < 0 or self.max_speed < self.min_speed:
            raise ValueError("speed bounds are invalid")
        if self.magnitude_mode not in {"navigation_matched", "rayleigh"}:
            raise ValueError(
                "magnitude_mode must be 'navigation_matched' or 'rayleigh'; "
                f"got {self.magnitude_mode!r}"
            )
        if self.magnitude_mode == "navigation_matched" and not self.shuffle_magnitudes:
            raise ValueError(
                "The formal navigation-matched control requires shuffle_magnitudes=true "
                "to remove wall/trajectory correlations."
            )

    def open_field_config(self) -> OpenFieldConfig:
        """Return the exact open-field settings used to source magnitudes."""
        return OpenFieldConfig(
            arena_size=self.arena_size,
            dt=self.dt,
            speed_scale=self.speed_scale,
            sigma_theta=self.sigma_theta,
            min_speed=self.min_speed,
            max_speed=self.max_speed,
            border_region=self.border_region,
            boundary_mode=self.boundary_mode,
        )


class SequenceControlSampler:
    """Generate i.i.d.-direction two-dimensional control vectors.

    ``navigation_matched`` is the formal mode.  ``rayleigh`` is retained only
    as a diagnostic/backward-compatible mode; it matches the pre-boundary
    Rayleigh recipe, not the actual navigation inputs saved after slowdown and
    boundary correction.
    """

    _DIRECTION_SEED_OFFSET = 1_000_003

    def __init__(self, config: SequenceControlConfig, seed: int = 42) -> None:
        self.config = config
        self.seed = int(seed)
        # Keep the navigation-magnitude RNG on exactly the requested seed so a
        # separate OpenFieldSampler(seed) produces the same magnitude multiset.
        self._navigation_sampler = (
            OpenFieldSampler(config.open_field_config(), seed=self.seed)
            if config.magnitude_mode == "navigation_matched"
            else None
        )
        # Direction/permutation draws must not advance the navigation RNG.
        self._direction_rng = np.random.default_rng(
            self.seed + self._DIRECTION_SEED_OFFSET
        )

    def _sample_magnitudes(self, batch_size: int, seq_len: int) -> np.ndarray:
        cfg = self.config
        if self._navigation_sampler is not None:
            nav_batch = self._navigation_sampler.sample_batch(batch_size, seq_len)
            magnitudes = np.linalg.norm(nav_batch["velocities"], axis=-1)
            if cfg.shuffle_magnitudes:
                flat = magnitudes.reshape(-1).copy()
                self._direction_rng.shuffle(flat)
                magnitudes = flat.reshape(batch_size, seq_len)
            return magnitudes.astype(np.float32, copy=False)

        speed = self._direction_rng.rayleigh(
            cfg.speed_scale, size=(batch_size, seq_len)
        )
        speed = np.clip(speed, cfg.min_speed, cfg.max_speed)
        return (speed * cfg.dt).astype(np.float32)

    def sample_batch(self, batch_size: int, seq_len: int) -> np.ndarray:
        """Return ``float32`` vectors with shape ``[batch, sequence, 2]``."""
        if batch_size < 1 or seq_len < 1:
            raise ValueError("batch_size and seq_len must be positive")
        magnitudes = self._sample_magnitudes(batch_size, seq_len)
        angles = self._direction_rng.uniform(
            0.0, 2.0 * np.pi, size=(batch_size, seq_len)
        )
        vectors = np.stack(
            [magnitudes * np.cos(angles), magnitudes * np.sin(angles)], axis=-1
        )
        return vectors.astype(np.float32)


def make_delay_targets(inputs: torch.Tensor, delay: int) -> tuple[torch.Tensor, slice]:
    """Return the target and prediction window for one reconstruction lag."""
    if inputs.ndim != 3:
        raise ValueError(f"inputs must have shape [batch, sequence, dim], got {inputs.shape}")
    seq_len = int(inputs.shape[1])
    if not 1 <= delay < seq_len:
        raise ValueError(
            f"delay must satisfy 1 <= delay < sequence length; got {delay} and {seq_len}"
        )
    return inputs[:, : seq_len - delay, :], slice(delay, seq_len)


class SequenceControlStaticDataset(Dataset):
    """Pre-generated, reproducible validation dataset."""

    def __init__(
        self, config: SequenceControlConfig, num_sequences: int, seed: int
    ) -> None:
        if num_sequences < 1:
            raise ValueError("num_sequences must be positive")
        self.config = config
        sampler = SequenceControlSampler(config, seed=seed)
        batch = sampler.sample_batch(num_sequences, config.seq_len)
        self.inputs = torch.from_numpy(batch).float()

    def __len__(self) -> int:
        return int(self.inputs.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {"inputs": self.inputs[idx]}


class SequenceControlOnlineDataset(IterableDataset):
    """Fresh already-batched control sequences for each optimization step."""

    def __init__(
        self,
        config: SequenceControlConfig,
        steps_per_epoch: int,
        batch_size: int,
        seed: int,
    ) -> None:
        super().__init__()
        if steps_per_epoch < 1 or batch_size < 1:
            raise ValueError("steps_per_epoch and batch_size must be positive")
        self.config = config
        self.steps_per_epoch = int(steps_per_epoch)
        self.batch_size = int(batch_size)
        self.sampler = SequenceControlSampler(config, seed=seed)

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        for _ in range(self.steps_per_epoch):
            batch = self.sampler.sample_batch(
                batch_size=self.batch_size, seq_len=self.config.seq_len
            )
            yield {"inputs": torch.from_numpy(batch).float()}
