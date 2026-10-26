"""Dedicated same-start/same-end path groups for stage I3C."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from gpm_cmap.analysis.upgrade_contracts import CLAIM_SCOPE, hash_jsonable, trajectory_fingerprint
from gpm_cmap.data.graph_tasks.pathint_episodes import PathIntEpisode, episode_fingerprint
from gpm_cmap.data.graph_tasks.social_hierarchy import SocialHierarchyTask

ACTIONS = {
    "+x": np.asarray([1, 0], dtype=np.int64),
    "-x": np.asarray([-1, 0], dtype=np.int64),
    "+y": np.asarray([0, 1], dtype=np.int64),
    "-y": np.asarray([0, -1], dtype=np.int64),
    "zero": np.asarray([0, 0], dtype=np.int64),
}
VARIANT_NAMES = (
    "shortest_a",
    "alternative_b",
    "a_plus_loop",
    "b_plus_different_loop",
    "a_plus_zero_padding",
    "same_multiset_reordered",
)


@dataclass(frozen=True)
class PathInvariantVariant:
    group_id: int
    variant_name: str
    episode: PathIntEpisode
    action_labels: tuple[str, ...]
    loop_before_step: int | None = None
    loop_after_step: int | None = None


@dataclass(frozen=True)
class PathInvariantGroup:
    group_id: int
    start_xy: tuple[int, int]
    end_xy: tuple[int, int]
    variants: tuple[PathInvariantVariant, ...]


@dataclass(frozen=True)
class PathInvariantDataset:
    groups: tuple[PathInvariantGroup, ...]
    manifest: Mapping[str, Any]

    @property
    def variants(self) -> tuple[PathInvariantVariant, ...]:
        return tuple(variant for group in self.groups for variant in group.variants)


def _axis_actions(delta: int, positive: str, negative: str) -> list[str]:
    return [positive if delta > 0 else negative] * abs(int(delta))


def _valid_sequence(start: tuple[int, int], labels: Sequence[str], n_levels: int) -> tuple[bool, list[tuple[int, int]]]:
    current = np.asarray(start, dtype=np.int64)
    coords = [tuple(int(x) for x in current)]
    for label in labels:
        current = current + ACTIONS[label]
        if np.any(current < 0) or np.any(current >= n_levels):
            return False, coords
        coords.append(tuple(int(x) for x in current))
    return True, coords


def _loop_for_coordinate(coord: tuple[int, int], n_levels: int, *, prefer_axis: int) -> list[str]:
    x, y = coord
    candidates = (
        [("+x", "-x"), ("-x", "+x")] if prefer_axis == 0
        else [("+y", "-y"), ("-y", "+y")]
    )
    for first, second in candidates:
        valid, _ = _valid_sequence(coord, [first, second], n_levels)
        if valid:
            return [first, second]
    raise RuntimeError(f"No valid two-step loop from coordinate {coord}")


def _insert_loop(
    start: tuple[int, int],
    base: Sequence[str],
    n_levels: int,
    *,
    prefer_axis: int,
) -> tuple[list[str], int, int]:
    # Insert after the first recurrent step whenever possible so holonomy is
    # measured around a recurrent-generated state rather than the start encoder.
    insertion = 1 if len(base) >= 1 else 0
    valid, coords = _valid_sequence(start, base[:insertion], n_levels)
    if not valid:
        raise RuntimeError("Invalid base prefix")
    loop = _loop_for_coordinate(coords[-1], n_levels, prefer_axis=prefer_axis)
    labels = list(base[:insertion]) + loop + list(base[insertion:])
    valid, _ = _valid_sequence(start, labels, n_levels)
    if not valid:
        raise RuntimeError("Loop insertion produced an invalid path")
    # hidden_seq index: after prefix is insertion-1; after loop is insertion+1.
    return labels, insertion, insertion + len(loop)


def _interleave(x_actions: Sequence[str], y_actions: Sequence[str]) -> list[str]:
    result: list[str] = []
    for index in range(max(len(x_actions), len(y_actions))):
        if index < len(x_actions):
            result.append(x_actions[index])
        if index < len(y_actions):
            result.append(y_actions[index])
    return result


def _episode_from_labels(
    *,
    start: tuple[int, int],
    end: tuple[int, int],
    labels: Sequence[str],
    n_levels: int,
    velocity_scale: float,
    coord_scale: float,
) -> PathIntEpisode:
    valid, coords_int = _valid_sequence(start, labels, n_levels)
    if not valid or coords_int[-1] != end:
        raise RuntimeError(f"Variant path does not terminate at the requested endpoint: {start}->{end}, {labels}")
    coords = np.asarray(coords_int, dtype=np.float32) * float(coord_scale)
    velocities = np.asarray([ACTIONS[label] for label in labels], dtype=np.float32) * float(velocity_scale)
    start_node = start[0] * n_levels + start[1]
    target_node = end[0] * n_levels + end[1]
    return PathIntEpisode(
        start_node=int(start_node),
        velocities=velocities,
        target_node=int(target_node),
        start_coords=coords[0].copy(),
        target_coords=coords[-1].copy(),
        path_coords=coords,
        net_l1_distance=float(np.abs(coords[-1] - coords[0]).sum()),
        path_unique_nodes=len(set(coords_int)),
    )


def build_path_invariance_dataset(
    config: Mapping[str, Any],
    *,
    seed: int,
    forbidden_fingerprints: Iterable[str] = (),
) -> PathInvariantDataset:
    n_levels = int(config.get("n_levels", 8))
    n_groups = int(config.get("n_groups", 512))
    velocity_scale = float(config.get("velocity_scale", 0.02))
    coord_scale = float(config.get("coord_scale", velocity_scale))
    padding_steps = int(config.get("zero_padding_steps", 4))
    if not np.isclose(velocity_scale, coord_scale):
        raise ValueError("path-invariance coord_scale must equal velocity_scale")
    if n_levels < 4 or n_groups < 2 or padding_steps < 1:
        raise ValueError("Invalid path-invariance dataset size")
    rng = np.random.default_rng(int(seed))
    forbidden = set(forbidden_fingerprints)
    groups: list[PathInvariantGroup] = []
    fingerprints: set[str] = set()
    attempts = 0
    while len(groups) < n_groups:
        attempts += 1
        if attempts > n_groups * 1000:
            raise RuntimeError(f"Could not construct {n_groups} unique path-invariance groups")
        start = tuple(int(x) for x in rng.integers(1, n_levels - 1, size=2))
        end = tuple(int(x) for x in rng.integers(1, n_levels - 1, size=2))
        dx, dy = end[0] - start[0], end[1] - start[1]
        if dx == 0 or dy == 0:
            continue
        x_actions = _axis_actions(dx, "+x", "-x")
        y_actions = _axis_actions(dy, "+y", "-y")
        shortest_a = x_actions + y_actions
        alternative_b = y_actions + x_actions
        reordered = _interleave(x_actions, y_actions)
        if reordered in (shortest_a, alternative_b):
            # Reverse interleaving order for a distinct same-multiset path.
            reordered = _interleave(y_actions, x_actions)
        a_loop, a_before, a_after = _insert_loop(start, shortest_a, n_levels, prefer_axis=1)
        b_loop, b_before, b_after = _insert_loop(start, alternative_b, n_levels, prefer_axis=0)
        padded = list(shortest_a[:1]) + ["zero"] * padding_steps + list(shortest_a[1:])
        specs = (
            ("shortest_a", shortest_a, None, None),
            ("alternative_b", alternative_b, None, None),
            ("a_plus_loop", a_loop, a_before, a_after),
            ("b_plus_different_loop", b_loop, b_before, b_after),
            ("a_plus_zero_padding", padded, None, None),
            ("same_multiset_reordered", reordered, None, None),
        )
        variants = []
        local_fingerprints = []
        try:
            for variant_name, labels, before, after in specs:
                episode = _episode_from_labels(
                    start=start, end=end, labels=labels, n_levels=n_levels,
                    velocity_scale=velocity_scale, coord_scale=coord_scale,
                )
                fp = episode_fingerprint(episode)
                if fp in forbidden or fp in fingerprints or fp in local_fingerprints:
                    raise ValueError("duplicate")
                local_fingerprints.append(fp)
                variants.append(PathInvariantVariant(
                    group_id=len(groups), variant_name=variant_name,
                    episode=episode, action_labels=tuple(labels),
                    loop_before_step=before, loop_after_step=after,
                ))
        except (RuntimeError, ValueError):
            continue
        fingerprints.update(local_fingerprints)
        groups.append(PathInvariantGroup(
            group_id=len(groups), start_xy=start, end_xy=end,
            variants=tuple(variants),
        ))
    rows = []
    for group in groups:
        for variant in group.variants:
            rows.append({
                "group_id": group.group_id,
                "variant_name": variant.variant_name,
                "start_xy": list(group.start_xy),
                "end_xy": list(group.end_xy),
                "path_length": int(variant.episode.velocities.shape[0]),
                "action_labels": list(variant.action_labels),
                "loop_before_step": variant.loop_before_step,
                "loop_after_step": variant.loop_after_step,
                "trajectory_fingerprint": episode_fingerprint(variant.episode),
            })
    manifest = {
        "schema_version": "path_invariance_dataset_v1",
        "claim_scope": CLAIM_SCOPE,
        "seed": int(seed),
        "n_levels": n_levels,
        "n_groups": n_groups,
        "variants_per_group": list(VARIANT_NAMES),
        "velocity_scale": velocity_scale,
        "coord_scale": coord_scale,
        "zero_padding_steps": padding_steps,
        "same_start_same_end_enforced": True,
        "start_encoder_protocol": "one h0 per group, repeated across all variants",
        "rows": rows,
        "trajectory_fingerprint_hash": hash_jsonable(sorted(fingerprints)),
    }
    return PathInvariantDataset(groups=tuple(groups), manifest=manifest)
