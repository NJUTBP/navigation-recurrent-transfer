"""Non-saturating social-hierarchy benchmark datasets for upgrade stage I2.

The generator creates mutually disjoint pools for nested low-data training
budgets, validation-only model selection, in-distribution testing,
train-short/test-long generalization, and ordered held-out action composition.
No target test pool is used to choose graph size, optimization settings, or
checkpoints.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
from torch.utils.data import DataLoader

from gpm_cmap.analysis.upgrade_contracts import (
    CLAIM_SCOPE,
    assert_disjoint_fingerprint_sets,
    hash_jsonable,
)
from gpm_cmap.data.graph_tasks.pathint_episodes import (
    PathIntDataset,
    PathIntEpisode,
    episode_fingerprint,
    generate_pathint_episodes,
    pad_collate,
)
from gpm_cmap.data.graph_tasks.social_hierarchy import SocialHierarchyTask
from gpm_cmap.utils.seed import make_torch_generator, make_worker_init_fn

ACTION_LABELS = {0: "+x", 1: "-x", 2: "+y", 3: "-y", 4: "zero"}
DEFAULT_HELD_OUT_BIGRAMS = ((0, 2), (1, 3), (2, 1), (3, 0))


@dataclass(frozen=True)
class HardPathIntPools:
    train_pool: tuple[PathIntEpisode, ...]
    validation: tuple[PathIntEpisode, ...]
    id_test: tuple[PathIntEpisode, ...]
    long_test: tuple[PathIntEpisode, ...]
    composition_test: tuple[PathIntEpisode, ...]
    node_to_id: Mapping[str, int]
    node_coords: Mapping[str, np.ndarray]
    action_to_id: Mapping[str, int]
    manifest: Mapping[str, Any]

    @property
    def n_nodes(self) -> int:
        return len(self.node_to_id)

    def training_subset(self, unique_episodes: int) -> tuple[PathIntEpisode, ...]:
        count = int(unique_episodes)
        if count < 1 or count > len(self.train_pool):
            raise ValueError(
                f"unique_episodes must lie in [1,{len(self.train_pool)}]; got {count}"
            )
        return self.train_pool[:count]


def velocity_action_labels(velocities: np.ndarray, *, atol: float = 1e-10) -> tuple[int, ...]:
    velocity = np.asarray(velocities, dtype=np.float64)
    if velocity.ndim != 2 or velocity.shape[1] != 2:
        raise ValueError(f"velocities must have shape [T,2]; got {velocity.shape}")
    labels: list[int] = []
    for row in velocity:
        if float(np.linalg.norm(row)) <= atol:
            labels.append(4)
            continue
        axis = int(np.argmax(np.abs(row)))
        sign_positive = bool(row[axis] > 0)
        labels.append(axis * 2 + (0 if sign_positive else 1))
    return tuple(labels)


def ordered_bigrams(episode: PathIntEpisode) -> frozenset[tuple[int, int]]:
    labels = velocity_action_labels(episode.velocities)
    return frozenset(zip(labels[:-1], labels[1:]))


def composition_signature(episode: PathIntEpisode) -> str:
    labels = velocity_action_labels(episode.velocities)
    bigrams = sorted(f"{a}>{b}" for a, b in zip(labels[:-1], labels[1:]))
    return hash_jsonable({"labels": labels, "ordered_bigrams": bigrams})


def _filter_composition(
    episodes: Iterable[PathIntEpisode],
    held_out_bigrams: set[tuple[int, int]],
    *,
    require_held_out: bool,
) -> list[PathIntEpisode]:
    result = []
    for episode in episodes:
        has = bool(ordered_bigrams(episode) & held_out_bigrams)
        if has == require_held_out:
            result.append(episode)
    return result


def _unique_extend(
    target: list[PathIntEpisode],
    candidates: Iterable[PathIntEpisode],
    seen: set[str],
    *,
    forbidden: set[str],
    needed: int,
) -> None:
    for episode in candidates:
        fingerprint = episode_fingerprint(episode)
        if fingerprint in seen or fingerprint in forbidden:
            continue
        seen.add(fingerprint)
        target.append(episode)
        if len(target) >= needed:
            return


def _generate_filtered_pool(
    *,
    graph,
    node_to_id,
    action_to_id,
    node_coords,
    count: int,
    length_range: tuple[int, int],
    velocity_scale: float,
    seed: int,
    held_out_bigrams: set[tuple[int, int]],
    require_held_out: bool,
    forbidden: set[str],
    oversample_factor: int = 8,
) -> tuple[PathIntEpisode, ...]:
    result: list[PathIntEpisode] = []
    seen: set[str] = set()
    round_index = 0
    while len(result) < count:
        remaining = count - len(result)
        batch_count = max(64, remaining * int(oversample_factor))
        candidates = generate_pathint_episodes(
            graph=graph,
            node_to_id=node_to_id,
            action_to_id=action_to_id,
            node_coords=node_coords,
            num_episodes=batch_count,
            seq_len=length_range[1],
            seq_len_range=length_range,
            velocity_scale=velocity_scale,
            seed=int(seed) + round_index * 7919,
            action_map_mode="aligned",
            normalize_displacements=True,
        )
        candidates = _filter_composition(
            candidates,
            held_out_bigrams,
            require_held_out=require_held_out,
        )
        _unique_extend(
            result,
            candidates,
            seen,
            forbidden=forbidden,
            needed=count,
        )
        round_index += 1
        if round_index > 200:
            raise RuntimeError(
                "Could not generate enough unique hard-path episodes under the "
                f"composition constraint: obtained {len(result)}/{count}."
            )
    return tuple(result)


def _scale_node_coords(raw: Mapping[str, np.ndarray], scale: float) -> dict[str, np.ndarray]:
    if scale <= 0:
        raise ValueError("coord_scale must be positive")
    return {name: (np.asarray(value, dtype=np.float32) * scale).astype(np.float32) for name, value in raw.items()}


def build_hard_pathint_pools(config: Mapping[str, Any], *, seed: int) -> HardPathIntPools:
    """Create all formal I2/I3 target pools from independent random streams."""
    if str(config.get("task_name", "social_hierarchy")) != "social_hierarchy":
        raise ValueError("The formal I2 benchmark is restricted to social_hierarchy")
    n_levels = int(config.get("n_levels", 8))
    if n_levels < 4:
        raise ValueError("Hard benchmark n_levels must be at least 4")
    velocity_scale = float(config.get("velocity_scale", 0.02))
    coord_scale = float(config.get("coord_scale", velocity_scale))
    if not np.isclose(coord_scale, velocity_scale):
        raise ValueError(
            "Formal path integration requires coord_scale == velocity_scale so "
            "target-start equals the summed velocity on the unit lattice"
        )
    train_range = tuple(int(x) for x in config.get("train_path_length", [1, 5]))
    id_range = tuple(int(x) for x in config.get("id_test_path_length", train_range))
    long_range = tuple(int(x) for x in config.get("long_test_path_length", [6, 12]))
    if len(train_range) != 2 or len(id_range) != 2 or len(long_range) != 2:
        raise ValueError("Each path-length range must contain exactly [min,max]")
    if long_range[0] <= train_range[1]:
        raise ValueError("long_test_path_length must begin above the maximum train length")
    max_budget = max(int(x) for x in config.get("episode_budgets", [10, 20, 50, 100, 200, 500, 1000]))
    validation_count = int(config.get("validation_episodes", 1000))
    id_count = int(config.get("id_test_episodes", 2000))
    long_count = int(config.get("long_test_episodes", 2000))
    comp_count = int(config.get("composition_test_episodes", 2000))
    held_out = {
        tuple(int(v) for v in pair)
        for pair in config.get("held_out_action_bigrams", DEFAULT_HELD_OUT_BIGRAMS)
    }
    if not held_out:
        raise ValueError("held_out_action_bigrams cannot be empty")
    if any(len(pair) != 2 or min(pair) < 0 or max(pair) > 3 for pair in held_out):
        raise ValueError("held_out_action_bigrams must contain pairs from the four cardinal actions")

    task = SocialHierarchyTask(n_levels=n_levels, seed=int(seed))
    graph, node_to_id, action_to_id, raw_coords = task.build_graph()
    node_coords = _scale_node_coords(raw_coords, coord_scale)
    offsets = dict(config.get("seed_offsets", {}))
    streams = {
        "train": int(seed) + int(offsets.get("train", 1001)),
        "validation": int(seed) + int(offsets.get("validation", 2001)),
        "id_test": int(seed) + int(offsets.get("id_test", 3001)),
        "long_test": int(seed) + int(offsets.get("long_test", 4001)),
        "composition_test": int(seed) + int(offsets.get("composition_test", 5001)),
    }
    forbidden: set[str] = set()
    train = _generate_filtered_pool(
        graph=graph, node_to_id=node_to_id, action_to_id=action_to_id,
        node_coords=node_coords, count=max_budget, length_range=train_range,
        velocity_scale=velocity_scale, seed=streams["train"],
        held_out_bigrams=held_out, require_held_out=False, forbidden=forbidden,
    )
    forbidden.update(episode_fingerprint(ep) for ep in train)
    validation = _generate_filtered_pool(
        graph=graph, node_to_id=node_to_id, action_to_id=action_to_id,
        node_coords=node_coords, count=validation_count, length_range=train_range,
        velocity_scale=velocity_scale, seed=streams["validation"],
        held_out_bigrams=held_out, require_held_out=False, forbidden=forbidden,
    )
    forbidden.update(episode_fingerprint(ep) for ep in validation)
    id_test = _generate_filtered_pool(
        graph=graph, node_to_id=node_to_id, action_to_id=action_to_id,
        node_coords=node_coords, count=id_count, length_range=id_range,
        velocity_scale=velocity_scale, seed=streams["id_test"],
        held_out_bigrams=held_out, require_held_out=False, forbidden=forbidden,
    )
    forbidden.update(episode_fingerprint(ep) for ep in id_test)
    long_test = _generate_filtered_pool(
        graph=graph, node_to_id=node_to_id, action_to_id=action_to_id,
        node_coords=node_coords, count=long_count, length_range=long_range,
        velocity_scale=velocity_scale, seed=streams["long_test"],
        held_out_bigrams=held_out, require_held_out=False, forbidden=forbidden,
    )
    forbidden.update(episode_fingerprint(ep) for ep in long_test)
    composition_test = _generate_filtered_pool(
        graph=graph, node_to_id=node_to_id, action_to_id=action_to_id,
        node_coords=node_coords, count=comp_count, length_range=train_range,
        velocity_scale=velocity_scale, seed=streams["composition_test"],
        held_out_bigrams=held_out, require_held_out=True, forbidden=forbidden,
    )

    named = {
        "train_pool": [episode_fingerprint(ep) for ep in train],
        "validation": [episode_fingerprint(ep) for ep in validation],
        "id_test": [episode_fingerprint(ep) for ep in id_test],
        "long_test": [episode_fingerprint(ep) for ep in long_test],
        "composition_test": [episode_fingerprint(ep) for ep in composition_test],
    }
    overlaps = assert_disjoint_fingerprint_sets(named)
    train_bigrams = set().union(*(ordered_bigrams(ep) for ep in train)) if train else set()
    comp_bigrams = set().union(*(ordered_bigrams(ep) for ep in composition_test)) if composition_test else set()
    if train_bigrams & held_out:
        raise RuntimeError("Held-out ordered action compositions leaked into the training pool")
    if not (comp_bigrams & held_out):
        raise RuntimeError("Composition test contains no held-out ordered action composition")
    manifest = {
        "schema_version": "hard_pathint_dataset_v1",
        "claim_scope": CLAIM_SCOPE,
        "task_name": "social_hierarchy",
        "n_levels": n_levels,
        "n_nodes": len(node_to_id),
        "velocity_scale": velocity_scale,
        "coord_scale": coord_scale,
        "path_length_ranges": {
            "train": list(train_range), "validation": list(train_range),
            "id_test": list(id_range), "long_test": list(long_range),
            "composition_test": list(train_range),
        },
        "counts": {name: len(values) for name, values in named.items()},
        "seeds": streams,
        "held_out_action_bigrams": [list(pair) for pair in sorted(held_out)],
        "held_out_action_bigram_labels": [
            [ACTION_LABELS[a], ACTION_LABELS[b]] for a, b in sorted(held_out)
        ],
        "training_contains_held_out_bigrams": False,
        "composition_test_contains_held_out_bigrams": True,
        "fingerprints": named,
        "fingerprint_set_hashes": {name: hash_jsonable(sorted(values)) for name, values in named.items()},
        "overlaps": overlaps,
        "nested_training_budgets": sorted(int(x) for x in config.get("episode_budgets", [10, 20, 50, 100, 200, 500, 1000])),
        "selection_protocol": "validation only; all test pools consumed after checkpoint selection",
    }
    return HardPathIntPools(
        train_pool=train,
        validation=validation,
        id_test=id_test,
        long_test=long_test,
        composition_test=composition_test,
        node_to_id=node_to_id,
        node_coords=node_coords,
        action_to_id=action_to_id,
        manifest=manifest,
    )


def make_loader(
    episodes: Sequence[PathIntEpisode],
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int = 0,
) -> DataLoader:
    if not episodes:
        raise ValueError("Cannot build a DataLoader from an empty episode sequence")
    return DataLoader(
        PathIntDataset(list(episodes)),
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        collate_fn=pad_collate,
        num_workers=int(num_workers),
        worker_init_fn=make_worker_init_fn(int(seed)),
        generator=make_torch_generator(int(seed)) if shuffle else None,
    )


def pool_loaders(
    pools: HardPathIntPools,
    *,
    unique_episodes: int,
    batch_size: int,
    seed: int,
    num_workers: int = 0,
) -> dict[str, DataLoader]:
    return {
        "train": make_loader(
            pools.training_subset(unique_episodes), batch_size=batch_size,
            shuffle=True, seed=seed, num_workers=num_workers,
        ),
        "validation": make_loader(
            pools.validation, batch_size=batch_size, shuffle=False,
            seed=seed + 1, num_workers=num_workers,
        ),
        "id_test": make_loader(
            pools.id_test, batch_size=batch_size, shuffle=False,
            seed=seed + 2, num_workers=num_workers,
        ),
        "long_test": make_loader(
            pools.long_test, batch_size=batch_size, shuffle=False,
            seed=seed + 3, num_workers=num_workers,
        ),
        "composition_test": make_loader(
            pools.composition_test, batch_size=batch_size, shuffle=False,
            seed=seed + 4, num_workers=num_workers,
        ),
    }
