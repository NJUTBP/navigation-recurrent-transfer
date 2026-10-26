"""Navigation dataset.

Upgrades vs original
---------------------
1. **Online generator mode** — ``NavOnlineDataLoader`` generates fresh
   trajectories each epoch (like the reference infinite generator).
   Avoids storing thousands of trajectories in RAM and gives the
   model more diverse training data.
2. **Static dataset preserved** — ``NavTrajectoryDataset`` still works
   for reproducible experiments and validation sets.
3. **Batch-level generation** — uses ``OpenFieldSampler.sample_batch``
   for much faster data creation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset

from .open_field import OpenFieldConfig, OpenFieldSampler
from .place_cells import PlaceCellEncoder


@dataclass
class NavDatasetConfig:
    num_trajectories: int
    seq_len: int
    open_field: OpenFieldConfig
    place_cells: int
    place_sigma: float
    arena_size: float
    seed: int = 42
    trajectory_seed: int | None = None
    place_seed: int | None = None
    use_dog: bool = True          # [NEW] pass to PlaceCellEncoder
    surround_scale: float = 2.0   # [NEW]
    surround_scale_mode: str = "variance"  # reference-compatible DoG semantics


# ---------------------------------------------------------------------------
#  Static dataset (original, kept for validation / reproducibility)
# ---------------------------------------------------------------------------

class NavTrajectoryDataset(Dataset):
    """Pre-generated static dataset.  Good for val/test sets."""

    def __init__(self, config: NavDatasetConfig) -> None:
        self.config = config
        traj_seed = config.trajectory_seed if config.trajectory_seed is not None else config.seed
        place_seed = config.place_seed if config.place_seed is not None else config.seed
        self.sampler = OpenFieldSampler(config.open_field, seed=traj_seed)
        self.place_encoder = PlaceCellEncoder(
            n_cells=config.place_cells,
            arena_size=config.arena_size,
            sigma=config.place_sigma,
            seed=place_seed,
            use_dog=config.use_dog,
            surround_scale=config.surround_scale,
            surround_scale_mode=config.surround_scale_mode,
        )
        # [UPGRADE] use batch generation for speed
        batch = self.sampler.sample_batch(
            batch_size=config.num_trajectories,
            seq_len=config.seq_len,
        )
        positions = torch.from_numpy(batch["positions"]).float()
        velocities = torch.from_numpy(batch["velocities"]).float()
        headings = torch.from_numpy(batch["headings"]).float()
        init_pos = torch.from_numpy(batch["init_pos"]).float()  # (N, 2) ← pos[1]
        # encode all positions at once
        place_targets = self.place_encoder.encode(positions)  # (N, T, Np) ← PC(pos[2:])
        init_place = self.place_encoder.encode(init_pos)      # (N, Np)   ← PC(pos[1])

        self.samples = []
        for i in range(config.num_trajectories):
            self.samples.append({
                "positions": positions[i],
                "velocities": velocities[i],
                "headings": headings[i],
                "place_targets": place_targets[i],
                "init_place": init_place[i],    # [FIX] PC(pos[1]), NOT PC(pos[2])!
            })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.samples[idx]


# ---------------------------------------------------------------------------
#  Online (infinite) dataset — generates fresh trajectories each iteration
# ---------------------------------------------------------------------------

class NavOnlineDataset(IterableDataset):
    """[NEW] Infinite-stream dataset like the reference get_generator().

    Each iteration yields a freshly sampled batch of trajectories.
    Use with ``DataLoader(dataset, batch_size=None)`` (batch is built
    inside the dataset).
    """

    def __init__(self, config: NavDatasetConfig, steps_per_epoch: int = 1000,
                 batch_size: int = 200) -> None:
        super().__init__()
        self.config = config
        self.steps_per_epoch = steps_per_epoch
        traj_seed = config.trajectory_seed if config.trajectory_seed is not None else config.seed
        place_seed = config.place_seed if config.place_seed is not None else config.seed
        self.sampler = OpenFieldSampler(config.open_field, seed=traj_seed)
        self.place_encoder = PlaceCellEncoder(
            n_cells=config.place_cells,
            arena_size=config.arena_size,
            sigma=config.place_sigma,
            seed=place_seed,
            use_dog=config.use_dog,
            surround_scale=config.surround_scale,
            surround_scale_mode=config.surround_scale_mode,
        )
        self.batch_size = batch_size  # [FIX] was hardcoded to 64!

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        for _ in range(self.steps_per_epoch):
            batch = self.sampler.sample_batch(
                batch_size=self.batch_size,
                seq_len=self.config.seq_len,
            )
            positions  = torch.from_numpy(batch["positions"]).float()
            velocities = torch.from_numpy(batch["velocities"]).float()
            headings   = torch.from_numpy(batch["headings"]).float()
            init_pos   = torch.from_numpy(batch["init_pos"]).float()  # (B, 2)
            place_targets = self.place_encoder.encode(positions)       # PC(pos[2:])
            init_place    = self.place_encoder.encode(init_pos)        # PC(pos[1])
            yield {
                "positions": positions,
                "velocities": velocities,
                "headings": headings,
                "place_targets": place_targets,
                "init_place": init_place,    # [FIX] PC(starting position)
            }
