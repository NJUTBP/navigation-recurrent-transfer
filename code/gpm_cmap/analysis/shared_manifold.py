"""Cross-domain tests for reuse of a navigation-learned neural manifold.

This module implements an operational, falsifiable version of the statement
"navigation and an abstract task use the same manifold".  All source charts,
local tangent spaces, circular coordinates and reduced dynamics are fitted on
navigation data only.  Target-domain hidden trajectories are then evaluated
without refitting the source model.

The two primary quantities mirror the SSM terminology used by Marraffa et al.:

``cross_mfe``
    Distance of a target trajectory from a manifold fitted only on navigation
    trajectories (cross-domain manifold fitting error).

``cross_nmte``
    Error of a navigation-fitted reduced map rolled out on target-domain input
    sequences and lifted back to hidden activity (cross-domain normalized mean
    trajectory error).

The implementation is deliberately discrete-time and controlled because the
GPM-CMap core evolves as a ReLU map h[t+1] = Phi(h[t], u[t]), not as a smooth
continuous-time autonomous ODE.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import json
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import PolynomialFeatures

from .toroidal_models import (
    ResidualFiberModel,
    StructuredPhaseMap,
    TorusDecoderSuite,
    angles_from_features as torus_angles_from_features,
    features_from_angles as torus_features_from_angles,
)


_EPS = 1e-12


def _finite_2d(array: np.ndarray, name: str) -> np.ndarray:
    out = np.asarray(array, dtype=np.float64)
    if out.ndim != 2:
        raise ValueError(f"{name} must have shape [N,D]; got {out.shape}")
    if out.shape[0] < 1 or out.shape[1] < 1:
        raise ValueError(f"{name} is empty: {out.shape}")
    if not np.isfinite(out).all():
        raise FloatingPointError(f"{name} contains NaN or Inf")
    return out


def _finite_1d(array: np.ndarray, name: str, n: Optional[int] = None) -> np.ndarray:
    out = np.asarray(array).reshape(-1)
    if n is not None and out.shape[0] != int(n):
        raise ValueError(f"{name} has {out.shape[0]} values; expected {n}")
    if np.issubdtype(out.dtype, np.number) and not np.isfinite(out).all():
        raise FloatingPointError(f"{name} contains NaN or Inf")
    return out


def wrap_angles(angle: np.ndarray) -> np.ndarray:
    """Wrap angles to [-pi, pi)."""
    value = np.asarray(angle, dtype=np.float64)
    return (value + np.pi) % (2.0 * np.pi) - np.pi


def circular_features_from_turns(turns: np.ndarray) -> np.ndarray:
    turns = _finite_2d(turns, "turns")
    if turns.shape[1] != 2:
        raise ValueError(f"Expected two circular coordinates; got {turns.shape}")
    theta = 2.0 * np.pi * turns
    return np.column_stack(
        [np.cos(theta[:, 0]), np.sin(theta[:, 0]),
         np.cos(theta[:, 1]), np.sin(theta[:, 1])]
    )


def normalize_circular_features(features: np.ndarray) -> np.ndarray:
    out = _finite_2d(features, "circular_features").copy()
    if out.shape[1] != 4:
        raise ValueError(f"Circular features must have four columns; got {out.shape}")
    for start in (0, 2):
        radius = np.linalg.norm(out[:, start:start + 2], axis=1, keepdims=True)
        bad = radius[:, 0] <= _EPS
        radius[bad] = 1.0
        out[:, start:start + 2] /= radius
        if np.any(bad):
            out[bad, start] = 1.0
            out[bad, start + 1] = 0.0
    return out


def angles_from_circular_features(features: np.ndarray) -> np.ndarray:
    f = normalize_circular_features(features)
    return np.column_stack([np.arctan2(f[:, 1], f[:, 0]), np.arctan2(f[:, 3], f[:, 2])])


def _safe_relative_error(numerator: np.ndarray, denominator_vectors: np.ndarray) -> float:
    numerator = np.asarray(numerator, dtype=np.float64).reshape(-1)
    denominator_vectors = _finite_2d(denominator_vectors, "denominator_vectors")
    denominator = float(np.max(np.linalg.norm(denominator_vectors, axis=1)))
    return float(np.mean(numerator) / max(denominator, _EPS))


def _group_train_validation_indices(
    groups: np.ndarray,
    *,
    validation_fraction: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    groups = _finite_1d(groups, "groups")
    unique = np.unique(groups)
    if unique.size < 2:
        n = groups.shape[0]
        split = max(1, min(n - 1, int(round(n * (1.0 - validation_fraction)))))
        order = np.random.default_rng(seed).permutation(n)
        return np.sort(order[:split]), np.sort(order[split:])
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=float(validation_fraction),
        random_state=int(seed),
    )
    train, validation = next(splitter.split(np.zeros(groups.shape[0]), groups=groups))
    return np.asarray(train, dtype=int), np.asarray(validation, dtype=int)


@dataclass
class TrajectoryData:
    """Flattened states plus explicit within-trajectory transition indices."""

    states: np.ndarray
    coords: np.ndarray
    trajectory_ids: np.ndarray
    time_indices: np.ndarray
    transition_prev: np.ndarray
    transition_next: np.ndarray
    transition_velocities: np.ndarray
    is_initial: Optional[np.ndarray] = None
    trajectory_correct: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.states = _finite_2d(self.states, "states")
        self.coords = _finite_2d(self.coords, "coords")
        if self.coords.shape != (self.states.shape[0], 2):
            raise ValueError(
                f"coords must have shape {(self.states.shape[0], 2)}; got {self.coords.shape}"
            )
        n = self.states.shape[0]
        self.trajectory_ids = _finite_1d(self.trajectory_ids, "trajectory_ids", n).astype(np.int64)
        self.time_indices = _finite_1d(self.time_indices, "time_indices", n).astype(np.int64)
        self.transition_prev = _finite_1d(self.transition_prev, "transition_prev").astype(np.int64)
        self.transition_next = _finite_1d(self.transition_next, "transition_next").astype(np.int64)
        self.transition_velocities = _finite_2d(
            self.transition_velocities, "transition_velocities"
        )
        if self.transition_velocities.shape != (self.transition_prev.shape[0], 2):
            raise ValueError("transition_velocities and transition index counts do not match")
        if self.transition_next.shape != self.transition_prev.shape:
            raise ValueError("transition_prev and transition_next must have the same shape")
        if self.transition_prev.size:
            if self.transition_prev.min() < 0 or self.transition_next.min() < 0:
                raise IndexError("transition indices must be non-negative")
            if self.transition_prev.max() >= n or self.transition_next.max() >= n:
                raise IndexError("transition index exceeds state count")
            if not np.all(self.trajectory_ids[self.transition_prev] == self.trajectory_ids[self.transition_next]):
                raise ValueError("A transition crosses trajectory boundaries")
            if not np.all(self.time_indices[self.transition_next] > self.time_indices[self.transition_prev]):
                raise ValueError("Transition time must increase")
        if self.is_initial is None:
            self.is_initial = np.zeros(n, dtype=bool)
            for trajectory_id in np.unique(self.trajectory_ids):
                idx = np.flatnonzero(self.trajectory_ids == trajectory_id)
                first = idx[np.argmin(self.time_indices[idx])]
                self.is_initial[first] = True
        else:
            self.is_initial = _finite_1d(self.is_initial, "is_initial", n).astype(bool)
        if self.trajectory_correct is not None:
            self.trajectory_correct = _finite_1d(
                self.trajectory_correct, "trajectory_correct"
            ).astype(bool)
            if self.trajectory_correct.shape[0] != np.unique(self.trajectory_ids).size:
                raise ValueError(
                    "trajectory_correct must contain one value per unique trajectory id"
                )

    @property
    def n_states(self) -> int:
        return int(self.states.shape[0])

    @property
    def n_units(self) -> int:
        return int(self.states.shape[1])

    @property
    def n_transitions(self) -> int:
        return int(self.transition_prev.shape[0])

    @property
    def unique_trajectories(self) -> np.ndarray:
        return np.unique(self.trajectory_ids)

    def trajectory_state_indices(self, trajectory_id: int) -> np.ndarray:
        idx = np.flatnonzero(self.trajectory_ids == int(trajectory_id))
        return idx[np.argsort(self.time_indices[idx], kind="stable")]

    def trajectory_transition_rows(self, trajectory_id: int) -> np.ndarray:
        rows = np.flatnonzero(self.trajectory_ids[self.transition_prev] == int(trajectory_id))
        if rows.size == 0:
            return rows
        return rows[np.argsort(self.time_indices[self.transition_prev[rows]], kind="stable")]

    def select_units(self, unit_indices: Sequence[int]) -> "TrajectoryData":
        indices = np.asarray(unit_indices, dtype=int).reshape(-1)
        if indices.size < 2:
            raise ValueError("At least two units are required")
        if indices.min() < 0 or indices.max() >= self.n_units:
            raise IndexError("unit_indices exceed the available hidden dimension")
        return TrajectoryData(
            states=self.states[:, indices],
            coords=self.coords,
            trajectory_ids=self.trajectory_ids,
            time_indices=self.time_indices,
            transition_prev=self.transition_prev,
            transition_next=self.transition_next,
            transition_velocities=self.transition_velocities,
            is_initial=self.is_initial,
            trajectory_correct=self.trajectory_correct,
            metadata={**self.metadata, "selected_unit_indices": indices.tolist()},
        )

    def previous_velocity_per_state(self) -> np.ndarray:
        """Return the control that produced each state.

        Initial states receive the zero vector.  For a strictly consecutive
        trajectory, state ``t`` receives ``u[t-1]``.  This context is required
        for diagnosing a toroidal base with velocity-dependent neural fibres.
        """
        previous = np.zeros((self.n_states, 2), dtype=np.float64)
        if self.n_transitions:
            previous[self.transition_next] = self.transition_velocities
        return previous

    def take_states(
        self,
        state_indices: Sequence[int],
        *,
        require_consecutive: bool = True,
    ) -> "TrajectoryData":
        """Subset and reindex states without creating transitions across gaps."""
        indices = np.asarray(state_indices, dtype=np.int64).reshape(-1)
        if indices.size == 0:
            raise ValueError("state_indices is empty")
        if indices.min() < 0 or indices.max() >= self.n_states:
            raise IndexError("state_indices exceed available states")
        indices = np.unique(indices)
        states = self.states[indices]
        coords = self.coords[indices]
        trajectory_ids = self.trajectory_ids[indices]
        time_indices = self.time_indices[indices]
        prev, nxt, velocity = build_consecutive_transitions(
            trajectory_ids,
            time_indices,
            coords,
            require_consecutive=require_consecutive,
        )
        initial = np.zeros(indices.size, dtype=bool)
        for trajectory_id in np.unique(trajectory_ids):
            rows = np.flatnonzero(trajectory_ids == trajectory_id)
            initial[rows[np.argmin(time_indices[rows])]] = True
        return TrajectoryData(
            states=states,
            coords=coords,
            trajectory_ids=trajectory_ids,
            time_indices=time_indices,
            transition_prev=prev,
            transition_next=nxt,
            transition_velocities=velocity,
            is_initial=initial,
            trajectory_correct=None,
            metadata={
                **self.metadata,
                "selected_original_state_indices": indices.tolist(),
                "require_consecutive_transitions": bool(require_consecutive),
            },
        )


def build_consecutive_transitions(
    trajectory_ids: np.ndarray,
    time_indices: np.ndarray,
    coords: np.ndarray,
    *,
    require_consecutive: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    trajectory_ids = _finite_1d(trajectory_ids, "trajectory_ids").astype(np.int64)
    time_indices = _finite_1d(time_indices, "time_indices", trajectory_ids.shape[0]).astype(np.int64)
    coords = _finite_2d(coords, "coords")
    if coords.shape != (trajectory_ids.shape[0], 2):
        raise ValueError("coords shape does not match metadata")
    prev: List[int] = []
    nxt: List[int] = []
    velocity: List[np.ndarray] = []
    for trajectory_id in np.unique(trajectory_ids):
        idx = np.flatnonzero(trajectory_ids == trajectory_id)
        idx = idx[np.argsort(time_indices[idx], kind="stable")]
        if idx.size < 2:
            continue
        for a, b in zip(idx[:-1], idx[1:]):
            delta_t = int(time_indices[b] - time_indices[a])
            if delta_t <= 0:
                continue
            if require_consecutive and delta_t != 1:
                continue
            prev.append(int(a))
            nxt.append(int(b))
            velocity.append(coords[b] - coords[a])
    if not prev:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty((0, 2), dtype=np.float64),
        )
    return (
        np.asarray(prev, dtype=np.int64),
        np.asarray(nxt, dtype=np.int64),
        np.asarray(velocity, dtype=np.float64),
    )


def load_navigation_trajectory_data(
    path: str | Path,
    *,
    unit_indices: Optional[Sequence[int]] = None,
    state_indices: Optional[Sequence[int]] = None,
    burn_in_steps: Optional[int] = None,
    min_step_displacement: Optional[float] = None,
    require_consecutive: bool = True,
) -> TrajectoryData:
    """Load navigation trajectories under the exact source-analysis filter.

    Parameters
    ----------
    state_indices:
        Exact original-state indices, normally
        ``torus_analysis_arrays.npz::analysis_input_indices``.  Supplying these
        prevents the shared-manifold reference from silently reintroducing
        burn-in and immobility samples excluded by ``analyze_torus.py``.
    burn_in_steps, min_step_displacement:
        Rule-based filtering for an independent calibration array whose row
        indices do not match the topology array.
    require_consecutive:
        When filtering removes an intermediate state, no transition is created
        across that gap.  This is essential for one-step controlled dynamics.
    """
    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        if "hidden" not in archive or "positions" not in archive:
            raise KeyError(f"{path} must contain hidden and positions")
        if "trajectory_ids" not in archive or "time_indices" not in archive:
            raise KeyError(
                f"{path} must contain trajectory_ids and time_indices for strict dynamics analysis"
            )
        hidden = np.asarray(archive["hidden"])
        coords_all = np.asarray(archive["positions"], dtype=np.float64)
        trajectory_all = np.asarray(archive["trajectory_ids"], dtype=np.int64)
        time_all = np.asarray(archive["time_indices"], dtype=np.int64)
        step_all = (
            np.asarray(archive["step_displacements"], dtype=np.float64)
            if "step_displacements" in archive else None
        )
        stored_unit_indices = (
            np.asarray(archive["unit_indices"], dtype=np.int64).reshape(-1)
            if "unit_indices" in archive else None
        )

    n = hidden.shape[0]
    if coords_all.shape != (n, 2):
        raise ValueError(f"positions must have shape {(n, 2)}; got {coords_all.shape}")
    if trajectory_all.shape[0] != n or time_all.shape[0] != n:
        raise ValueError("navigation metadata row count differs from hidden")

    finite_mask = np.isfinite(coords_all).all(axis=1) & np.isfinite(hidden).all(axis=1)
    mask = finite_mask.copy()
    exact_indices = None
    if state_indices is not None:
        exact_indices = np.asarray(state_indices, dtype=np.int64).reshape(-1)
        if exact_indices.size == 0:
            raise ValueError("state_indices is empty")
        if exact_indices.min() < 0 or exact_indices.max() >= n:
            raise IndexError("state_indices exceed the navigation array")
        exact_mask = np.zeros(n, dtype=bool)
        exact_mask[np.unique(exact_indices)] = True
        mask &= exact_mask
    if burn_in_steps is not None:
        mask &= time_all >= int(burn_in_steps)
    if min_step_displacement is not None and float(min_step_displacement) > 0:
        if step_all is None:
            raise KeyError(
                f"{path} lacks step_displacements required by min_step_displacement"
            )
        mask &= step_all > float(min_step_displacement)
    selected_rows = np.flatnonzero(mask)
    if selected_rows.size < 2:
        time_min = int(time_all.min()) if time_all.size else None
        time_max = int(time_all.max()) if time_all.size else None
        finite_count = int(finite_mask.sum())
        burn_count = (
            int((finite_mask & (time_all >= int(burn_in_steps))).sum())
            if burn_in_steps is not None else finite_count
        )
        movement_count = (
            int((finite_mask & (step_all > float(min_step_displacement))).sum())
            if step_all is not None and min_step_displacement is not None
            and float(min_step_displacement) > 0 else finite_count
        )
        raise ValueError(
            "Too few navigation states remain after source filtering: "
            f"path={path}, original_states={n}, finite_states={finite_count}, "
            f"time_range=[{time_min}, {time_max}], burn_in_steps={burn_in_steps}, "
            f"states_after_burn_in={burn_count}, "
            f"min_step_displacement={min_step_displacement}, "
            f"states_passing_movement_filter={movement_count}, "
            f"final_states={selected_rows.size}. "
            "A common cause is using the 20-step nav_eval_arrays.npz with a "
            "50-step calibration burn-in. Use an independent long-rollout "
            "calibration NPZ, or set calibration_burn_in_steps below the array's "
            "maximum time index for explicitly exploratory analysis."
        )

    if unit_indices is not None:
        indices = np.asarray(unit_indices, dtype=int).reshape(-1)
        if indices.size < 2:
            raise ValueError("At least two selected units are required")
        if stored_unit_indices is not None:
            if stored_unit_indices.size != hidden.shape[1]:
                raise ValueError(
                    "Stored unit_indices length does not match hidden columns: "
                    f"{stored_unit_indices.size} vs {hidden.shape[1]}"
                )
            global_to_local = {int(global_idx): int(local_idx)
                               for local_idx, global_idx in enumerate(stored_unit_indices)}
            missing = [int(idx) for idx in indices if int(idx) not in global_to_local]
            if missing:
                raise IndexError(
                    "Requested selected units are absent from the compact navigation "
                    f"array; first missing indices={missing[:10]}"
                )
            local_indices = np.asarray([global_to_local[int(idx)] for idx in indices], dtype=int)
            states = np.asarray(hidden[selected_rows][:, local_indices], dtype=np.float64)
        else:
            if indices.min() < 0 or indices.max() >= hidden.shape[1]:
                raise IndexError("unit_indices exceed hidden dimension")
            states = np.asarray(hidden[selected_rows][:, indices], dtype=np.float64)
    else:
        states = np.asarray(hidden[selected_rows], dtype=np.float64)
    coords = coords_all[selected_rows]
    trajectory_ids = trajectory_all[selected_rows]
    time_indices = time_all[selected_rows]
    prev, nxt, velocity = build_consecutive_transitions(
        trajectory_ids,
        time_indices,
        coords,
        require_consecutive=require_consecutive,
    )
    if prev.size < 10:
        raise ValueError(
            "Fewer than ten strict consecutive transitions remain after source filtering"
        )
    initial = np.zeros(states.shape[0], dtype=bool)
    for trajectory_id in np.unique(trajectory_ids):
        rows = np.flatnonzero(trajectory_ids == trajectory_id)
        initial[rows[np.argmin(time_indices[rows])]] = True
    return TrajectoryData(
        states=states,
        coords=coords,
        trajectory_ids=trajectory_ids,
        time_indices=time_indices,
        transition_prev=prev,
        transition_next=nxt,
        transition_velocities=velocity,
        is_initial=initial,
        metadata={
            "source_path": str(path.resolve()),
            "domain": "navigation",
            "n_original_states": int(n),
            "n_filtered_states": int(selected_rows.size),
            "original_state_indices": selected_rows.tolist(),
            "exact_state_indices_supplied": exact_indices is not None,
            "burn_in_steps": None if burn_in_steps is None else int(burn_in_steps),
            "min_step_displacement": (
                None if min_step_displacement is None else float(min_step_displacement)
            ),
            "require_consecutive_transitions": bool(require_consecutive),
        },
    )


def save_trajectory_data(path: str | Path, data: TrajectoryData) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, np.ndarray] = {
        "states": data.states.astype(np.float32),
        "coords": data.coords.astype(np.float32),
        "trajectory_ids": data.trajectory_ids.astype(np.int64),
        "time_indices": data.time_indices.astype(np.int64),
        "transition_prev": data.transition_prev.astype(np.int64),
        "transition_next": data.transition_next.astype(np.int64),
        "transition_velocities": data.transition_velocities.astype(np.float32),
        "is_initial": data.is_initial.astype(np.bool_),
    }
    if data.trajectory_correct is not None:
        payload["trajectory_correct"] = data.trajectory_correct.astype(np.bool_)
    payload["metadata_json"] = np.asarray(
        json.dumps(data.metadata, ensure_ascii=False, default=str)
    )
    np.savez_compressed(path, **payload)
    return path


def load_trajectory_data(path: str | Path) -> TrajectoryData:
    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "states", "coords", "trajectory_ids", "time_indices",
            "transition_prev", "transition_next", "transition_velocities",
        }
        missing = sorted(required - set(archive.files))
        if missing:
            raise KeyError(f"{path} is missing {missing}")
        metadata: Dict[str, Any] = {"source_path": str(path.resolve())}
        if "metadata_json" in archive:
            try:
                stored = json.loads(str(np.asarray(archive["metadata_json"]).item()))
                if isinstance(stored, dict):
                    metadata = {**stored, "source_path": str(path.resolve())}
            except (TypeError, ValueError, json.JSONDecodeError):
                metadata["metadata_parse_warning"] = "invalid metadata_json"
        return TrajectoryData(
            states=np.asarray(archive["states"]),
            coords=np.asarray(archive["coords"]),
            trajectory_ids=np.asarray(archive["trajectory_ids"]),
            time_indices=np.asarray(archive["time_indices"]),
            transition_prev=np.asarray(archive["transition_prev"]),
            transition_next=np.asarray(archive["transition_next"]),
            transition_velocities=np.asarray(archive["transition_velocities"]),
            is_initial=(np.asarray(archive["is_initial"]) if "is_initial" in archive else None),
            trajectory_correct=(
                np.asarray(archive["trajectory_correct"])
                if "trajectory_correct" in archive else None
            ),
            metadata=metadata,
        )


def sample_navigation_windows_like_target(
    navigation: TrajectoryData,
    target: TrajectoryData,
    *,
    seed: int = 42,
    max_windows_per_source_trajectory: int = 1,
) -> TrajectoryData:
    """Draw navigation windows with the target horizon distribution.

    Cross-domain rollout errors grow with horizon. Calibrating an eight-step
    abstract path against an unrelated fifty-step navigation rollout is not a
    valid equivalence design. This function uses only target trajectory lengths
    (never target hidden activity) to draw contiguous navigation windows. By
    default, each independent source trajectory contributes at most one window,
    preventing overlapping windows from becoming pseudo-replicates.
    """
    rng = np.random.default_rng(int(seed))
    target_lengths = np.asarray([
        int(target.trajectory_transition_rows(int(t)).size)
        for t in target.unique_trajectories
    ], dtype=int)
    if target_lengths.size == 0 or int(target_lengths.max()) < 1:
        raise ValueError("Target data contain no non-empty trajectories")
    max_per_source = max(1, int(max_windows_per_source_trajectory))

    source_sequences: List[Tuple[int, np.ndarray]] = []
    for trajectory_id in navigation.unique_trajectories:
        idx = navigation.trajectory_state_indices(int(trajectory_id))
        if idx.size >= 2:
            source_sequences.append((int(trajectory_id), idx))
    if not source_sequences:
        raise ValueError("Navigation calibration data contain no trajectories")

    # Each source trajectory may appear at most max_per_source times. The source
    # identity is recorded so downstream analyses can block-bootstrap if needed.
    source_slots = [
        (source_id, sequence)
        for source_id, sequence in source_sequences
        for _ in range(max_per_source)
    ]
    rng.shuffle(source_slots)
    n_windows = min(len(source_slots), int(target_lengths.size))
    if n_windows < 2:
        raise ValueError("At least two matched navigation windows are required")
    sampled_target_lengths = target_lengths[
        rng.choice(target_lengths.size, size=n_windows, replace=False)
        if n_windows <= target_lengths.size
        else rng.integers(0, target_lengths.size, size=n_windows)
    ]
    rng.shuffle(sampled_target_lengths)

    state_blocks: List[np.ndarray] = []
    coord_blocks: List[np.ndarray] = []
    trajectory_blocks: List[np.ndarray] = []
    time_blocks: List[np.ndarray] = []
    initial_blocks: List[np.ndarray] = []
    prev: List[int] = []
    nxt: List[int] = []
    velocities: List[np.ndarray] = []
    source_ids_used: List[int] = []
    actual_lengths: List[int] = []
    offset = 0

    unused_slots = list(source_slots)
    for new_id, requested_length in enumerate(sampled_target_lengths.tolist()):
        eligible_rows = [
            slot_idx for slot_idx, (_source_id, sequence) in enumerate(unused_slots)
            if sequence.size >= requested_length + 1
        ]
        if not eligible_rows:
            # Fall back to any remaining target horizon that fits one remaining
            # source trajectory. This only changes finite-sample matching when
            # source rollouts are shorter than some target paths.
            feasible = [
                (slot_idx, int(length))
                for slot_idx, (_source_id, sequence) in enumerate(unused_slots)
                for length in target_lengths
                if sequence.size >= int(length) + 1
            ]
            if not feasible:
                longest = max(sequence.size - 1 for _source_id, sequence in source_slots)
                raise ValueError(
                    f"Navigation calibration trajectories are too short for target "
                    f"horizons; longest available horizon is {longest}"
                )
            slot_idx, requested_length = feasible[int(rng.integers(0, len(feasible)))]
        else:
            slot_idx = eligible_rows[int(rng.integers(0, len(eligible_rows)))]
        source_id, sequence = unused_slots.pop(slot_idx)
        start_max = sequence.size - (requested_length + 1)
        start = int(rng.integers(0, start_max + 1)) if start_max > 0 else 0
        window = sequence[start:start + requested_length + 1]

        state_blocks.append(navigation.states[window])
        coord_blocks.append(navigation.coords[window])
        trajectory_blocks.append(
            np.full(requested_length + 1, new_id, dtype=np.int64)
        )
        time_blocks.append(np.arange(requested_length + 1, dtype=np.int64))
        initial = np.zeros(requested_length + 1, dtype=bool)
        initial[0] = True
        initial_blocks.append(initial)
        for step in range(requested_length):
            prev.append(offset + step)
            nxt.append(offset + step + 1)
            velocities.append(
                navigation.coords[window[step + 1]]
                - navigation.coords[window[step]]
            )
        source_ids_used.append(source_id)
        actual_lengths.append(int(requested_length))
        offset += requested_length + 1

    return TrajectoryData(
        states=np.concatenate(state_blocks, axis=0),
        coords=np.concatenate(coord_blocks, axis=0),
        trajectory_ids=np.concatenate(trajectory_blocks, axis=0),
        time_indices=np.concatenate(time_blocks, axis=0),
        transition_prev=np.asarray(prev, dtype=np.int64),
        transition_next=np.asarray(nxt, dtype=np.int64),
        transition_velocities=np.asarray(velocities, dtype=np.float64),
        is_initial=np.concatenate(initial_blocks, axis=0),
        metadata={
            **navigation.metadata,
            "matched_to_target_trajectories": int(target_lengths.size),
            "n_matched_navigation_windows": int(len(actual_lengths)),
            "target_transition_lengths": target_lengths.tolist(),
            "matched_transition_lengths": actual_lengths,
            "source_trajectory_ids_used": source_ids_used,
            "max_windows_per_source_trajectory": max_per_source,
            "matching_seed": int(seed),
        },
    )


@dataclass
class LocalTangentAtlas:
    """Piecewise-affine approximation of a source manifold and its tangents."""

    centers: np.ndarray
    local_means: np.ndarray
    local_bases: np.ndarray
    local_normal_scales: np.ndarray
    support_points: np.ndarray
    center_index: NearestNeighbors
    support_index: NearestNeighbors
    manifold_dim: int

    @classmethod
    def fit(
        cls,
        ambient: np.ndarray,
        chart: np.ndarray,
        *,
        manifold_dim: int = 2,
        n_atlas_centers: int = 256,
        neighbors_per_center: int = 128,
        seed: int = 42,
    ) -> "LocalTangentAtlas":
        ambient = _finite_2d(ambient, "ambient")
        chart = _finite_2d(chart, "chart")
        if ambient.shape[0] != chart.shape[0]:
            raise ValueError("ambient and chart sample counts differ")
        if manifold_dim < 1 or manifold_dim >= ambient.shape[1]:
            raise ValueError("manifold_dim must be between 1 and ambient_dim-1")
        n_samples = ambient.shape[0]
        n_centers = min(
            max(4, int(n_atlas_centers)),
            max(4, n_samples // max(8, int(neighbors_per_center) // 2)),
            n_samples,
        )
        k = min(max(manifold_dim + 3, int(neighbors_per_center)), n_samples)
        kmeans = MiniBatchKMeans(
            n_clusters=n_centers,
            random_state=int(seed),
            batch_size=min(4096, max(256, n_samples)),
            n_init=10,
        )
        kmeans.fit(chart)
        centers = np.asarray(kmeans.cluster_centers_, dtype=np.float64)
        support_index = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(chart)
        neighbor_idx = support_index.kneighbors(centers, return_distance=False)

        local_means = np.empty((n_centers, ambient.shape[1]), dtype=np.float64)
        local_bases = np.empty((n_centers, ambient.shape[1], manifold_dim), dtype=np.float64)
        local_scales = np.empty(n_centers, dtype=np.float64)
        for center_id, idx in enumerate(neighbor_idx):
            local = ambient[idx]
            mean = local.mean(axis=0)
            centered = local - mean
            _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
            basis = vt[:manifold_dim].T
            residual = centered - (centered @ basis) @ basis.T
            residual_norm = np.linalg.norm(residual, axis=1)
            positive = residual_norm[residual_norm > _EPS]
            scale = float(np.median(positive)) if positive.size else 1.0
            local_means[center_id] = mean
            local_bases[center_id] = basis
            local_scales[center_id] = max(scale, _EPS)

        center_index = NearestNeighbors(n_neighbors=1, metric="euclidean").fit(centers)
        # A separate one-neighbour index is used for support-distance diagnostics.
        support_one = NearestNeighbors(n_neighbors=1, metric="euclidean").fit(chart)
        return cls(
            centers=centers,
            local_means=local_means,
            local_bases=local_bases,
            local_normal_scales=local_scales,
            support_points=chart,
            center_index=center_index,
            support_index=support_one,
            manifold_dim=int(manifold_dim),
        )

    def assign(self, chart: np.ndarray) -> np.ndarray:
        chart = _finite_2d(chart, "chart")
        return self.center_index.kneighbors(chart, return_distance=False)[:, 0].astype(int)

    def reconstruct(
        self,
        ambient: np.ndarray,
        chart: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        ambient = _finite_2d(ambient, "ambient")
        chart = _finite_2d(chart, "chart")
        if ambient.shape[0] != chart.shape[0]:
            raise ValueError("ambient and chart sample counts differ")
        assignment = self.assign(chart)
        reconstruction = np.empty_like(ambient)
        residual_norm = np.empty(ambient.shape[0], dtype=np.float64)
        residual_z = np.empty(ambient.shape[0], dtype=np.float64)
        for atlas_id in np.unique(assignment):
            rows = np.flatnonzero(assignment == atlas_id)
            mean = self.local_means[atlas_id]
            basis = self.local_bases[atlas_id]
            centered = ambient[rows] - mean
            reconstructed = mean + (centered @ basis) @ basis.T
            residual = ambient[rows] - reconstructed
            norm = np.linalg.norm(residual, axis=1)
            reconstruction[rows] = reconstructed
            residual_norm[rows] = norm
            residual_z[rows] = norm / self.local_normal_scales[atlas_id]
        support_distance = self.support_index.kneighbors(chart, return_distance=True)[0][:, 0]
        return {
            "reconstruction": reconstruction,
            "residual_norm": residual_norm,
            "residual_z": residual_z,
            "support_distance": support_distance,
            "atlas_assignment": assignment,
        }

    def tangent_fraction(
        self,
        deltas: np.ndarray,
        current_chart: np.ndarray,
    ) -> np.ndarray:
        deltas = _finite_2d(deltas, "deltas")
        current_chart = _finite_2d(current_chart, "current_chart")
        if deltas.shape[0] != current_chart.shape[0]:
            raise ValueError("deltas and current_chart sample counts differ")
        assignment = self.assign(current_chart)
        fraction = np.empty(deltas.shape[0], dtype=np.float64)
        for atlas_id in np.unique(assignment):
            rows = np.flatnonzero(assignment == atlas_id)
            basis = self.local_bases[atlas_id]
            projected = (deltas[rows] @ basis) @ basis.T
            num = np.sum(projected * projected, axis=1)
            den = np.sum(deltas[rows] * deltas[rows], axis=1)
            fraction[rows] = np.divide(num, den, out=np.ones_like(num), where=den > _EPS)
        return np.clip(fraction, 0.0, 1.0)


@dataclass
class PolynomialRegressor:
    degree: int
    alpha: float
    polynomial: PolynomialFeatures
    ridge: Ridge

    @classmethod
    def fit_selected(
        cls,
        x: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
        *,
        degrees: Sequence[int],
        alphas: Sequence[float],
        validation_fraction: float,
        seed: int,
        output_normalizer: Optional[Any] = None,
    ) -> Tuple["PolynomialRegressor", Dict[str, Any]]:
        x = _finite_2d(x, "regression_x")
        y = _finite_2d(y, "regression_y")
        if x.shape[0] != y.shape[0]:
            raise ValueError("Regression x/y sample counts differ")
        groups = _finite_1d(groups, "regression_groups", x.shape[0])
        train_idx, val_idx = _group_train_validation_indices(
            groups, validation_fraction=validation_fraction, seed=seed
        )
        best: Optional[Tuple[float, int, float]] = None
        rows: List[Dict[str, float]] = []
        for degree in degrees:
            polynomial = PolynomialFeatures(degree=int(degree), include_bias=False)
            x_train = polynomial.fit_transform(x[train_idx])
            x_val = polynomial.transform(x[val_idx])
            for alpha in alphas:
                ridge = Ridge(alpha=float(alpha), fit_intercept=True)
                ridge.fit(x_train, y[train_idx])
                prediction = ridge.predict(x_val)
                if output_normalizer is not None:
                    prediction = output_normalizer(prediction)
                mse = float(np.mean((prediction - y[val_idx]) ** 2))
                rows.append({"degree": int(degree), "alpha": float(alpha), "validation_mse": mse})
                candidate = (mse, int(degree), float(alpha))
                if best is None or candidate < best:
                    best = candidate
        if best is None:
            raise RuntimeError("No polynomial regression candidate was fitted")
        _, degree, alpha = best
        polynomial = PolynomialFeatures(degree=degree, include_bias=False)
        transformed = polynomial.fit_transform(x)
        ridge = Ridge(alpha=alpha, fit_intercept=True)
        ridge.fit(transformed, y)
        model = cls(degree=degree, alpha=alpha, polynomial=polynomial, ridge=ridge)
        return model, {
            "selected_degree": degree,
            "selected_alpha": alpha,
            "validation_rows": rows,
            "n_train": int(train_idx.size),
            "n_validation": int(val_idx.size),
        }

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = _finite_2d(x, "regression_x")
        return np.asarray(self.ridge.predict(self.polynomial.transform(x)), dtype=np.float64)


@dataclass
class CircularTorusChart:
    """Navigation-only encoder/decoder for two cohomological circle coordinates."""

    encoder: PolynomialRegressor
    decoder: PolynomialRegressor
    fit_diagnostics: Dict[str, Any]

    @classmethod
    def fit(
        cls,
        chart_landmarks: np.ndarray,
        ambient_landmarks: np.ndarray,
        coordinate_turns: np.ndarray,
        groups: np.ndarray,
        *,
        seed: int = 42,
        validation_fraction: float = 0.2,
    ) -> "CircularTorusChart":
        chart_landmarks = _finite_2d(chart_landmarks, "chart_landmarks")
        ambient_landmarks = _finite_2d(ambient_landmarks, "ambient_landmarks")
        coordinate_turns = _finite_2d(coordinate_turns, "coordinate_turns")
        groups = _finite_1d(groups, "landmark_groups", chart_landmarks.shape[0])
        if not (
            chart_landmarks.shape[0]
            == ambient_landmarks.shape[0]
            == coordinate_turns.shape[0]
        ):
            raise ValueError("Landmark sample counts differ")
        circular = circular_features_from_turns(coordinate_turns)
        encoder, encoder_diag = PolynomialRegressor.fit_selected(
            chart_landmarks,
            circular,
            groups,
            degrees=(1, 2, 3),
            alphas=(1e-6, 1e-4, 1e-2, 1.0, 100.0),
            validation_fraction=validation_fraction,
            seed=seed,
            output_normalizer=normalize_circular_features,
        )
        # Decode from ideal circular features, not from encoder predictions, so
        # the learned surface is parameterized by the topological coordinates.
        decoder, decoder_diag = PolynomialRegressor.fit_selected(
            circular,
            ambient_landmarks,
            groups,
            degrees=(1, 2, 3),
            alphas=(1e-6, 1e-4, 1e-2, 1.0, 100.0),
            validation_fraction=validation_fraction,
            seed=seed + 1,
        )
        return cls(
            encoder=encoder,
            decoder=decoder,
            fit_diagnostics={"encoder": encoder_diag, "decoder": decoder_diag},
        )

    def encode(self, chart: np.ndarray) -> np.ndarray:
        return normalize_circular_features(self.encoder.predict(chart))

    def angles(self, chart: np.ndarray) -> np.ndarray:
        return angles_from_circular_features(self.encode(chart))

    def decode(self, latent: np.ndarray) -> np.ndarray:
        return self.decoder.predict(normalize_circular_features(latent))


@dataclass
class ControlledPolynomialMap:
    """Discrete controlled reduced map z[t+1] = R(z[t], u[t])."""

    regressor: PolynomialRegressor
    latent_kind: str
    fit_diagnostics: Dict[str, Any]

    @classmethod
    def fit(
        cls,
        latent_states: np.ndarray,
        data: TrajectoryData,
        *,
        latent_kind: str,
        seed: int = 42,
        validation_fraction: float = 0.2,
    ) -> "ControlledPolynomialMap":
        latent_states = _finite_2d(latent_states, "latent_states")
        if latent_states.shape[0] != data.n_states:
            raise ValueError("latent_states and data state counts differ")
        if data.n_transitions < 10:
            raise ValueError("At least ten transitions are needed for a reduced map")
        x = np.column_stack([
            latent_states[data.transition_prev],
            data.transition_velocities,
        ])
        y = latent_states[data.transition_next]
        groups = data.trajectory_ids[data.transition_prev]
        normalizer = normalize_circular_features if latent_kind == "torus" else None
        regressor, diagnostics = PolynomialRegressor.fit_selected(
            x,
            y,
            groups,
            degrees=(1, 2, 3),
            alphas=(1e-7, 1e-5, 1e-3, 1e-1, 10.0),
            validation_fraction=validation_fraction,
            seed=seed,
            output_normalizer=normalizer,
        )
        return cls(regressor=regressor, latent_kind=latent_kind, fit_diagnostics=diagnostics)

    def predict(self, latent: np.ndarray, velocity: np.ndarray) -> np.ndarray:
        latent = _finite_2d(latent, "latent")
        velocity = _finite_2d(velocity, "velocity")
        if velocity.shape != (latent.shape[0], 2):
            raise ValueError("velocity must have shape [N,2]")
        prediction = self.regressor.predict(np.column_stack([latent, velocity]))
        return normalize_circular_features(prediction) if self.latent_kind == "torus" else prediction

    def rollout(self, initial: np.ndarray, velocities: np.ndarray) -> np.ndarray:
        initial = np.asarray(initial, dtype=np.float64).reshape(1, -1)
        velocities = _finite_2d(velocities, "rollout_velocities")
        if velocities.shape[1] != 2:
            raise ValueError("rollout_velocities must have two columns")
        states = [initial[0].copy()]
        current = initial
        for velocity in velocities:
            current = self.predict(current, velocity.reshape(1, 2))
            states.append(current[0].copy())
        return np.asarray(states, dtype=np.float64)


@dataclass
class NavigationManifoldReference:
    """All source-only objects required for zero-refit cross-domain tests.

    ``torus_chart`` extracts two cohomological circle coordinates.  The newer
    source model then (i) aligns their ``GL(2,Z)`` basis to navigation control,
    (ii) fits an explicit Fourier torus decoder, and (iii) fits a structured
    additive-plus-residual phase map.  The legacy polynomial decoder/reduced map
    is retained as a secondary diagnostic and for backwards-compatible pickles.
    """

    selected_unit_indices: np.ndarray
    retained_unit_mask: np.ndarray
    unit_mean: np.ndarray
    unit_std: np.ndarray
    pca: PCA
    atlas: LocalTangentAtlas
    torus_chart: Optional[CircularTorusChart]
    reduced_map: ControlledPolynomialMap
    action_matrix: Optional[np.ndarray]
    source_support_chart: np.ndarray
    diagnostics: Dict[str, Any]
    structured_phase_map: Optional[StructuredPhaseMap] = None
    torus_decoders: Optional[TorusDecoderSuite] = None
    fiber_model: Optional[ResidualFiberModel] = None
    # Mean of every selected source unit, including constant/dead units that
    # are excluded from the standardized chart.  Keeping this separately avoids
    # zeroing those units during causal projection interventions.
    full_unit_mean: Optional[np.ndarray] = None

    @classmethod
    def fit(
        cls,
        source: TrajectoryData,
        *,
        selected_unit_indices: Sequence[int],
        chart_dim: int = 6,
        manifold_dim: int = 2,
        n_atlas_centers: int = 256,
        atlas_neighbors: int = 128,
        landmark_indices: Optional[np.ndarray] = None,
        landmark_torus_turns: Optional[np.ndarray] = None,
        fourier_orders: Sequence[int] = (1, 2, 3, 4),
        fourier_alphas: Sequence[float] = (1e-6, 1e-4, 1e-2, 1.0, 100.0),
        conditioned_decoder_min_improvement: float = 0.05,
        fiber_candidate_dims: Sequence[int] = (0, 1, 2, 4, 6, 8),
        max_gl2z_entry: int = 2,
        seed: int = 42,
    ) -> "NavigationManifoldReference":
        selected = np.asarray(selected_unit_indices, dtype=int).reshape(-1)
        if source.n_units != selected.size:
            raise ValueError(
                "source states must already be restricted to selected_unit_indices; "
                f"got {source.n_units} units and {selected.size} selected indices"
            )
        mean_all = source.states.mean(axis=0)
        std_all = source.states.std(axis=0)
        retained = np.isfinite(mean_all) & np.isfinite(std_all) & (std_all > 1e-8)
        if int(retained.sum()) <= max(manifold_dim, 2):
            raise ValueError("Too few non-constant selected navigation units")
        mean = mean_all[retained]
        std = std_all[retained]
        standardized = (source.states[:, retained] - mean) / std

        n_components = min(int(chart_dim), standardized.shape[1], standardized.shape[0] - 1)
        if n_components < max(2, manifold_dim):
            raise ValueError("Insufficient PCA rank for the requested source chart")
        pca = PCA(n_components=n_components, random_state=int(seed))
        chart = pca.fit_transform(standardized)
        atlas = LocalTangentAtlas.fit(
            standardized,
            chart,
            manifold_dim=manifold_dim,
            n_atlas_centers=n_atlas_centers,
            neighbors_per_center=atlas_neighbors,
            seed=seed,
        )

        torus_chart: Optional[CircularTorusChart] = None
        torus_note = "cohomological coordinates unavailable"
        n_landmarks_mapped = 0
        if landmark_indices is not None and landmark_torus_turns is not None:
            original_landmarks = np.asarray(landmark_indices, dtype=int).reshape(-1)
            turns_all = _finite_2d(landmark_torus_turns, "landmark_torus_turns")
            if turns_all.shape[0] != original_landmarks.size:
                raise ValueError("landmark indices and torus coordinate counts differ")

            # ``analyze_torus`` stores indices in the original NPZ coordinate
            # system.  A strictly filtered source reference has a new local row
            # system, so map exact original indices rather than comparing them
            # directly with source.n_states.
            original_source = np.asarray(
                source.metadata.get("original_state_indices", np.arange(source.n_states)),
                dtype=np.int64,
            ).reshape(-1)
            if original_source.size != source.n_states:
                raise ValueError("source original_state_indices metadata is malformed")
            original_to_local = {int(original): int(local) for local, original in enumerate(original_source)}
            mapped_local: List[int] = []
            mapped_turns: List[np.ndarray] = []
            for original, turn in zip(original_landmarks.tolist(), turns_all):
                local = original_to_local.get(int(original))
                if local is not None:
                    mapped_local.append(local)
                    mapped_turns.append(turn)
            local_landmarks = np.asarray(mapped_local, dtype=int)
            turns = (
                np.asarray(mapped_turns, dtype=np.float64)
                if mapped_turns else np.empty((0, 2), dtype=np.float64)
            )
            n_landmarks_mapped = int(local_landmarks.size)
            if local_landmarks.size >= 30 and turns.shape == (local_landmarks.size, 2):
                torus_chart = CircularTorusChart.fit(
                    chart[local_landmarks],
                    standardized[local_landmarks],
                    turns,
                    source.trajectory_ids[local_landmarks],
                    seed=seed,
                )
                torus_note = (
                    f"fit from {local_landmarks.size}/{original_landmarks.size} "
                    "navigation landmarks after exact source filtering"
                )
            else:
                torus_note = (
                    f"only {local_landmarks.size}/{original_landmarks.size} torus landmarks "
                    "survived exact source filtering"
                )

        structured_phase_map: Optional[StructuredPhaseMap] = None
        torus_decoders: Optional[TorusDecoderSuite] = None
        fiber_model: Optional[ResidualFiberModel] = None
        if torus_chart is not None:
            raw_latent = torus_chart.encode(chart)
            raw_theta = angles_from_circular_features(raw_latent)
            structured_phase_map = StructuredPhaseMap.fit(
                raw_theta,
                source.transition_prev,
                source.transition_next,
                source.transition_velocities,
                source.trajectory_ids[source.transition_prev],
                max_gl2z_entry=max_gl2z_entry,
                seed=seed + 11,
            )
            theta = structured_phase_map.align_angles(raw_theta)
            latent = torus_features_from_angles(theta)
            latent_kind = "torus"

            previous_velocity = source.previous_velocity_per_state()
            torus_decoders = TorusDecoderSuite.fit(
                theta,
                standardized,
                source.trajectory_ids,
                previous_velocity,
                orders=fourier_orders,
                alphas=fourier_alphas,
                conditioned_min_relative_improvement=(
                    conditioned_decoder_min_improvement
                ),
                seed=seed + 23,
            )
            selected_prediction = torus_decoders.predict(
                theta,
                previous_velocity,
                mode="selected",
            )
            fiber_model = ResidualFiberModel.fit(
                standardized - selected_prediction,
                source.trajectory_ids,
                candidate_dims=fiber_candidate_dims,
                seed=seed + 31,
            )
        else:
            latent = chart
            latent_kind = "pca"

        reduced_map = ControlledPolynomialMap.fit(
            latent,
            source,
            latent_kind=latent_kind,
            seed=seed + 17,
        )

        action_matrix = (
            np.asarray(structured_phase_map.action_matrix, dtype=np.float64)
            if structured_phase_map is not None else None
        )
        action_fit_rmse = (
            structured_phase_map.diagnostics.get("heldout_additive_rmse")
            if structured_phase_map is not None else None
        )
        diagnostics = {
            "reference_version": 2,
            "n_source_states": source.n_states,
            "n_source_transitions": source.n_transitions,
            "source_filter": {
                key: source.metadata.get(key)
                for key in [
                    "n_original_states",
                    "n_filtered_states",
                    "exact_state_indices_supplied",
                    "burn_in_steps",
                    "min_step_displacement",
                    "require_consecutive_transitions",
                ]
            },
            "n_selected_units": int(selected.size),
            "n_retained_units": int(retained.sum()),
            "chart_dim": int(n_components),
            "manifold_dim": int(manifold_dim),
            "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "latent_kind": latent_kind,
            "torus_chart_note": torus_note,
            "n_torus_landmarks_mapped": n_landmarks_mapped,
            "torus_chart_fit": torus_chart.fit_diagnostics if torus_chart is not None else None,
            "structured_phase_map": (
                structured_phase_map.diagnostics
                if structured_phase_map is not None else None
            ),
            "torus_decoders": (
                torus_decoders.diagnostics if torus_decoders is not None else None
            ),
            "fiber_model": fiber_model.diagnostics if fiber_model is not None else None,
            "reduced_map_fit": reduced_map.fit_diagnostics,
            "action_fit_rmse": action_fit_rmse,
        }
        return cls(
            selected_unit_indices=selected,
            retained_unit_mask=retained,
            unit_mean=mean,
            unit_std=std,
            pca=pca,
            atlas=atlas,
            torus_chart=torus_chart,
            reduced_map=reduced_map,
            action_matrix=action_matrix,
            source_support_chart=chart,
            diagnostics=diagnostics,
            structured_phase_map=structured_phase_map,
            torus_decoders=torus_decoders,
            fiber_model=fiber_model,
            full_unit_mean=np.asarray(mean_all, dtype=np.float64),
        )

    @property
    def latent_kind(self) -> str:
        return "torus" if self.torus_chart is not None else "pca"

    @property
    def reference_version(self) -> int:
        return int(self.diagnostics.get("reference_version", 1))

    def standardize(self, selected_states: np.ndarray) -> np.ndarray:
        states = _finite_2d(selected_states, "selected_states")
        if states.shape[1] != self.selected_unit_indices.size:
            raise ValueError(
                f"Expected {self.selected_unit_indices.size} selected units; got {states.shape[1]}"
            )
        return (states[:, self.retained_unit_mask] - self.unit_mean) / self.unit_std

    def destandardize(self, ambient: np.ndarray) -> np.ndarray:
        ambient = _finite_2d(ambient, "ambient")
        if ambient.shape[1] != int(np.sum(self.retained_unit_mask)):
            raise ValueError("ambient dimension differs from retained source units")
        selected = np.zeros(
            (ambient.shape[0], self.selected_unit_indices.size), dtype=np.float64
        )
        selected[:, self.retained_unit_mask] = ambient * self.unit_std + self.unit_mean
        # Constant/dead units are restored to their actual source mean.  Older
        # v1 pickles do not contain ``full_unit_mean``; in that compatibility
        # case we retain the historical zero fallback and expose the limitation
        # through the reference version rather than silently changing the chart.
        if np.any(~self.retained_unit_mask):
            full_mean = getattr(self, "full_unit_mean", None)
            if full_mean is not None:
                full_mean = np.asarray(full_mean, dtype=np.float64).reshape(-1)
            if full_mean is None or full_mean.size != self.selected_unit_indices.size:
                full_mean = np.zeros(
                    self.selected_unit_indices.size, dtype=np.float64
                )
                full_mean[self.retained_unit_mask] = self.unit_mean
            selected[:, ~self.retained_unit_mask] = full_mean[
                ~self.retained_unit_mask
            ]
        return selected

    def chart(self, selected_states: np.ndarray) -> np.ndarray:
        return self.pca.transform(self.standardize(selected_states))

    def raw_torus_latent_from_chart(self, chart: np.ndarray) -> np.ndarray:
        if self.torus_chart is None:
            raise RuntimeError("No cohomological torus chart is available")
        return self.torus_chart.encode(chart)

    def latent_from_chart(self, chart: np.ndarray) -> np.ndarray:
        chart = _finite_2d(chart, "chart")
        if self.torus_chart is None:
            return chart
        raw = self.torus_chart.encode(chart)
        phase_map = getattr(self, "structured_phase_map", None)
        if phase_map is not None:
            return phase_map.align_features(raw)
        return raw

    def latent(self, selected_states: np.ndarray) -> np.ndarray:
        return self.latent_from_chart(self.chart(selected_states))

    def decode_latent(
        self,
        latent: np.ndarray,
        previous_velocity: Optional[np.ndarray] = None,
        *,
        decoder_mode: str = "base",
    ) -> np.ndarray:
        latent = _finite_2d(latent, "latent")
        decoders = getattr(self, "torus_decoders", None)
        if self.torus_chart is not None and decoders is not None:
            theta = torus_angles_from_features(latent)
            return decoders.predict(
                theta,
                previous_velocity,
                mode=decoder_mode,
            )
        if self.torus_chart is not None:
            # Backwards-compatible reference: latent uses the original basis.
            return self.torus_chart.decode(latent)
        return self.pca.inverse_transform(latent)

    def reconstruct_torus(
        self,
        selected_states: np.ndarray,
        previous_velocity: Optional[np.ndarray] = None,
        *,
        decoder_mode: str = "base",
    ) -> Optional[np.ndarray]:
        if self.torus_chart is None:
            return None
        latent = self.latent(selected_states)
        return self.decode_latent(
            latent,
            previous_velocity,
            decoder_mode=decoder_mode,
        )

    def reconstruct_with_fiber(
        self,
        selected_states: np.ndarray,
        previous_velocity: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        decoders = getattr(self, "torus_decoders", None)
        fiber = getattr(self, "fiber_model", None)
        if self.torus_chart is None or decoders is None or fiber is None:
            return None
        ambient = self.standardize(selected_states)
        latent = self.latent(selected_states)
        theta = torus_angles_from_features(latent)
        base = decoders.predict(theta, previous_velocity, mode="selected")
        return base + fiber.reconstruct(ambient - base)

    def torus_tangent_bases(
        self,
        latent: np.ndarray,
        previous_velocity: Optional[np.ndarray] = None,
        *,
        decoder_mode: str = "base",
    ) -> Optional[np.ndarray]:
        decoders = getattr(self, "torus_decoders", None)
        if decoders is None or self.torus_chart is None:
            return None
        theta = torus_angles_from_features(latent)
        jacobian = decoders.jacobian(
            theta,
            previous_velocity,
            mode=decoder_mode,
        )
        bases = np.zeros_like(jacobian)
        for row in range(jacobian.shape[0]):
            q, _r = np.linalg.qr(jacobian[row], mode="reduced")
            bases[row] = q[:, :2]
        return bases


def _circular_variance_by_keys(
    keys: np.ndarray,
    angles: np.ndarray,
    *,
    minimum_group_size: int = 2,
) -> Dict[str, Any]:
    keys = np.asarray(keys)
    if keys.ndim == 1:
        keys = keys[:, None]
    angles = _finite_2d(angles, "angles")
    if keys.shape[0] != angles.shape[0] or angles.shape[1] != 2:
        raise ValueError("keys/angles shape mismatch")
    unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    variances: List[np.ndarray] = []
    counts: List[int] = []
    for group in range(unique.shape[0]):
        rows = np.flatnonzero(inverse == group)
        if rows.size < int(minimum_group_size):
            continue
        resultant = np.abs(np.mean(np.exp(1j * angles[rows]), axis=0))
        variances.append(1.0 - resultant)
        counts.append(int(rows.size))
    if not variances:
        return {
            "n_repeated_groups": 0,
            "n_rows_in_repeated_groups": 0,
            "mean_circular_variance": None,
            "mean_circular_variance_over_axes": None,
        }
    values = np.vstack(variances)
    weights = np.asarray(counts, dtype=np.float64)
    weighted = np.average(values, axis=0, weights=weights)
    return {
        "n_repeated_groups": int(len(variances)),
        "n_rows_in_repeated_groups": int(np.sum(counts)),
        "mean_circular_variance": weighted.tolist(),
        "mean_circular_variance_over_axes": float(np.mean(weighted)),
    }


def _coordinate_phase_dispersion(
    coords: np.ndarray,
    angles: np.ndarray,
    *,
    time_indices: Optional[np.ndarray] = None,
    trajectory_ids: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Separate a path-independent coordinate code from a coordinate-clock code."""
    coords = _finite_2d(coords, "coords")
    angles = _finite_2d(angles, "angles")
    if coords.shape[0] != angles.shape[0] or angles.shape[1] != 2:
        raise ValueError("coords/angles shape mismatch")
    rounded = np.round(coords, decimals=8)
    by_coordinate = _circular_variance_by_keys(rounded, angles)
    result: Dict[str, Any] = {"by_coordinate": by_coordinate}
    # Backwards-compatible aliases.
    result["n_repeated_coordinates"] = by_coordinate["n_repeated_groups"]
    result["mean_circular_variance"] = by_coordinate["mean_circular_variance"]
    result["mean_circular_variance_over_axes"] = by_coordinate[
        "mean_circular_variance_over_axes"
    ]

    if time_indices is not None:
        time = _finite_1d(time_indices, "time_indices", coords.shape[0]).astype(np.int64)
        coord_time = np.column_stack([rounded, time])
        by_coordinate_time = _circular_variance_by_keys(coord_time, angles)
        result["by_coordinate_and_time"] = by_coordinate_time
        coord_var = by_coordinate.get("mean_circular_variance_over_axes")
        coord_time_var = by_coordinate_time.get("mean_circular_variance_over_axes")
        result["time_dependence_fraction"] = (
            None
            if coord_var is None or coord_time_var is None
            else float(
                np.clip(
                    1.0 - float(coord_time_var) / max(float(coord_var), _EPS),
                    -1.0,
                    1.0,
                )
            )
        )
        max_time = np.empty(coords.shape[0], dtype=bool)
        max_time[:] = False
        if trajectory_ids is not None:
            tids = _finite_1d(
                trajectory_ids, "trajectory_ids", coords.shape[0]
            ).astype(np.int64)
            for trajectory_id in np.unique(tids):
                rows = np.flatnonzero(tids == trajectory_id)
                max_time[rows[np.argmax(time[rows])]] = True
        else:
            max_time = time == np.max(time)
        result["final_state_by_coordinate"] = _circular_variance_by_keys(
            rounded[max_time], angles[max_time]
        )
    return result


def _fraction_in_bases(deltas: np.ndarray, bases: np.ndarray) -> np.ndarray:
    deltas = _finite_2d(deltas, "deltas")
    bases = np.asarray(bases, dtype=np.float64)
    if bases.ndim != 3 or bases.shape[0] != deltas.shape[0] or bases.shape[1] != deltas.shape[1]:
        raise ValueError("bases must have shape [N, ambient_dim, tangent_dim]")
    projected_coeff = np.einsum("ndk,nd->nk", bases, deltas)
    projected = np.einsum("ndk,nk->nd", bases, projected_coeff)
    numerator = np.sum(projected ** 2, axis=1)
    denominator = np.sum(deltas ** 2, axis=1)
    return np.clip(
        np.divide(
            numerator,
            denominator,
            out=np.ones_like(numerator),
            where=denominator > _EPS,
        ),
        0.0,
        1.0,
    )


def _endpoint_path_invariance(
    data: TrajectoryData,
    ambient: np.ndarray,
    angles: Optional[np.ndarray],
) -> Dict[str, Any]:
    """Compare final states reached by different paths with the same endpoints."""
    records: Dict[Tuple[float, ...], List[Tuple[int, np.ndarray, Optional[np.ndarray], bytes]]] = {}
    for trajectory_id in data.unique_trajectories:
        state_idx = data.trajectory_state_indices(int(trajectory_id))
        trans_rows = data.trajectory_transition_rows(int(trajectory_id))
        if state_idx.size < 2:
            continue
        key = tuple(
            np.round(
                np.concatenate([data.coords[state_idx[0]], data.coords[state_idx[-1]]]),
                8,
            ).tolist()
        )
        velocity_signature = np.round(
            data.transition_velocities[trans_rows], 8
        ).tobytes()
        records.setdefault(key, []).append(
            (
                int(trajectory_id),
                ambient[state_idx[-1]],
                None if angles is None else angles[state_idx[-1]],
                velocity_signature,
            )
        )
    hidden_distances: List[float] = []
    phase_distances: List[float] = []
    n_endpoint_groups = 0
    n_pairs = 0
    for values in records.values():
        if len(values) < 2:
            continue
        used_group = False
        for i in range(len(values)):
            for j in range(i + 1, len(values)):
                if values[i][3] == values[j][3]:
                    continue
                used_group = True
                n_pairs += 1
                hidden_distances.append(
                    float(np.linalg.norm(values[i][1] - values[j][1]))
                )
                if values[i][2] is not None and values[j][2] is not None:
                    phase_distances.append(
                        float(np.linalg.norm(wrap_angles(values[i][2] - values[j][2])))
                    )
        if used_group:
            n_endpoint_groups += 1
    return {
        "n_endpoint_groups": int(n_endpoint_groups),
        "n_path_pairs": int(n_pairs),
        "mean_final_hidden_distance": (
            float(np.mean(hidden_distances)) if hidden_distances else None
        ),
        "median_final_hidden_distance": (
            float(np.median(hidden_distances)) if hidden_distances else None
        ),
        "mean_final_phase_distance": (
            float(np.mean(phase_distances)) if phase_distances else None
        ),
        "median_final_phase_distance": (
            float(np.median(phase_distances)) if phase_distances else None
        ),
    }


def evaluate_domain_against_navigation(
    reference: NavigationManifoldReference,
    data: TrajectoryData,
    *,
    label: str,
    loop_coordinate_tolerance: float = 1e-8,
    support_distance_threshold: Optional[float] = None,
    normal_residual_z_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """Evaluate one domain with a completely frozen navigation reference.

    Version-2 references expose two distinct geometric objects:

    * the two-dimensional toroidal *base* decoded by an explicit Fourier map;
    * an optional velocity-conditioned neural fibre and residual PCA fibre.

    The base metrics are the confirmatory tests.  Conditioned/fibre metrics are
    diagnostics that determine whether the full hidden state is better described
    as ``T^2 x R^k`` than as an exact two-dimensional embedded torus.
    """
    if data.n_units != reference.selected_unit_indices.size:
        raise ValueError(
            "Target data must contain the same source-selected units in the same order"
        )
    ambient = reference.standardize(data.states)
    chart = reference.pca.transform(ambient)
    atlas_result = reference.atlas.reconstruct(ambient, chart)
    local_residual_norm = atlas_result["residual_norm"]
    previous_velocity = data.previous_velocity_per_state()

    latent = reference.latent_from_chart(chart)
    base_reconstruction = reference.decode_latent(
        latent, previous_velocity, decoder_mode="base"
    )
    surface_residual_norm = np.linalg.norm(ambient - base_reconstruction, axis=1)

    conditioned_reconstruction = None
    conditioned_residual_norm = None
    decoders = getattr(reference, "torus_decoders", None)
    if decoders is not None and decoders.conditioned is not None:
        conditioned_reconstruction = reference.decode_latent(
            latent, previous_velocity, decoder_mode="previous_velocity"
        )
        conditioned_residual_norm = np.linalg.norm(
            ambient - conditioned_reconstruction, axis=1
        )
    fiber_reconstruction = reference.reconstruct_with_fiber(
        data.states, previous_velocity
    )
    fiber_residual_norm = (
        None
        if fiber_reconstruction is None
        else np.linalg.norm(ambient - fiber_reconstruction, axis=1)
    )

    prev = data.transition_prev
    nxt = data.transition_next
    velocity = data.transition_velocities
    current_latent = latent[prev]
    next_latent = latent[nxt]

    # Legacy flexible polynomial map: retained as a secondary diagnostic.
    generic_one_step_latent = reference.reduced_map.predict(current_latent, velocity)
    generic_one_step_hidden = reference.decode_latent(
        generic_one_step_latent,
        velocity,
        decoder_mode=(
            "selected" if getattr(reference, "torus_decoders", None) is not None else "base"
        ),
    )
    generic_one_step_hidden_error = np.linalg.norm(
        ambient[nxt] - generic_one_step_hidden, axis=1
    )
    generic_one_step_latent_error = np.linalg.norm(
        next_latent - generic_one_step_latent, axis=1
    )

    local_tangent_fraction = reference.atlas.tangent_fraction(
        ambient[nxt] - ambient[prev], chart[prev]
    )
    torus_tangent_fraction = None
    torus_bases = reference.torus_tangent_bases(
        current_latent,
        previous_velocity[prev],
        decoder_mode="base",
    )
    if torus_bases is not None:
        torus_tangent_fraction = _fraction_in_bases(
            ambient[nxt] - ambient[prev], torus_bases
        )
    primary_tangent_fraction = (
        torus_tangent_fraction
        if torus_tangent_fraction is not None else local_tangent_fraction
    )

    phase_map = getattr(reference, "structured_phase_map", None)
    angles = None
    actual_delta = None
    phase_action_error = None
    phase_action_normalized = None
    structured_additive_latent = None
    structured_nonlinear_latent = None
    structured_additive_hidden_error = None
    structured_nonlinear_hidden_error = None
    if reference.torus_chart is not None:
        angles = torus_angles_from_features(latent)
    if angles is not None and reference.action_matrix is not None:
        actual_delta = wrap_angles(angles[nxt] - angles[prev])
        predicted_delta = velocity @ reference.action_matrix.T
        residual = wrap_angles(actual_delta - predicted_delta)
        phase_action_error = np.linalg.norm(residual, axis=1)
        source_phase_scale = (
            reference.diagnostics.get("structured_phase_map", {})
            .get("heldout_phase_step_rms")
        )
        if source_phase_scale is None or float(source_phase_scale) <= _EPS:
            source_phase_scale = float(
                np.sqrt(np.mean(actual_delta ** 2))
            )
        phase_action_normalized = (
            phase_action_error / max(float(source_phase_scale), 1e-6)
        )
    if phase_map is not None:
        structured_additive_latent = phase_map.predict_features(
            current_latent, velocity, mode="additive"
        )
        structured_nonlinear_latent = phase_map.predict_features(
            current_latent, velocity, mode="nonlinear"
        )
        structured_additive_hidden = reference.decode_latent(
            structured_additive_latent,
            velocity,
            decoder_mode=("selected" if decoders is not None else "base"),
        )
        structured_nonlinear_hidden = reference.decode_latent(
            structured_nonlinear_latent,
            velocity,
            decoder_mode=("selected" if decoders is not None else "base"),
        )
        structured_additive_hidden_error = np.linalg.norm(
            ambient[nxt] - structured_additive_hidden, axis=1
        )
        structured_nonlinear_hidden_error = np.linalg.norm(
            ambient[nxt] - structured_nonlinear_hidden, axis=1
        )

    unique_trajectories = data.unique_trajectories
    if data.trajectory_correct is not None:
        trajectory_correct_map = {
            int(trajectory_id): bool(correct)
            for trajectory_id, correct in zip(
                unique_trajectories, data.trajectory_correct
            )
        }
    else:
        trajectory_correct_map = {int(t): None for t in unique_trajectories}

    rows: List[Dict[str, Any]] = []
    time_resolved: Dict[int, List[float]] = {}
    all_internal_loop_hidden: List[float] = []
    all_internal_loop_phase: List[float] = []

    for trajectory_id in unique_trajectories:
        state_idx = data.trajectory_state_indices(int(trajectory_id))
        trans_rows = data.trajectory_transition_rows(int(trajectory_id))
        if state_idx.size == 0:
            continue
        post_idx = state_idx[~data.is_initial[state_idx]]
        if post_idx.size == 0:
            post_idx = state_idx
        denominator_states = ambient[post_idx]
        local_cross_mfe = _safe_relative_error(
            local_residual_norm[post_idx], denominator_states
        )
        surface_cross_mfe = _safe_relative_error(
            surface_residual_norm[post_idx], denominator_states
        )
        conditioned_surface_cross_mfe = (
            None
            if conditioned_residual_norm is None else
            _safe_relative_error(conditioned_residual_norm[post_idx], denominator_states)
        )
        fiber_surface_cross_mfe = (
            None
            if fiber_residual_norm is None else
            _safe_relative_error(fiber_residual_norm[post_idx], denominator_states)
        )
        initial_idx = state_idx[data.is_initial[state_idx]]
        initial_cross_mfe = (
            _safe_relative_error(surface_residual_norm[initial_idx], ambient[initial_idx])
            if initial_idx.size else None
        )

        for idx in state_idx:
            time_resolved.setdefault(int(data.time_indices[idx]), []).append(
                float(
                    surface_residual_norm[idx]
                    / max(np.linalg.norm(ambient[idx]), _EPS)
                )
            )

        support_values = atlas_result["support_distance"][post_idx]
        normal_values = atlas_result["residual_z"][post_idx]
        support_ood_fraction = (
            None
            if support_distance_threshold is None else
            float(np.mean(support_values > float(support_distance_threshold)))
        )
        normal_ood_fraction = (
            None
            if normal_residual_z_threshold is None else
            float(np.mean(normal_values > float(normal_residual_z_threshold)))
        )

        generic_one_step_nmte = None
        generic_one_step_latent_rmse = None
        structured_additive_one_step_nmte = None
        structured_nonlinear_one_step_nmte = None
        local_tangent_mean = None
        torus_tangent_mean = None
        tangent_mean = None
        phase_rmse = None
        phase_normalized_rmse = None
        path_cocycle_error = None

        generic_rollout_nmte = None
        generic_rollout_full_nmte = None
        generic_rollout_latent_rmse = None
        structured_additive_rollout_nmte = None
        structured_additive_rollout_full_nmte = None
        structured_nonlinear_rollout_nmte = None
        structured_nonlinear_rollout_full_nmte = None
        structured_additive_phase_rmse = None
        structured_nonlinear_phase_rmse = None

        if trans_rows.size:
            actual_next = ambient[nxt[trans_rows]]
            generic_one_step_nmte = _safe_relative_error(
                generic_one_step_hidden_error[trans_rows], actual_next
            )
            generic_one_step_latent_rmse = float(
                np.sqrt(np.mean(generic_one_step_latent_error[trans_rows] ** 2))
            )
            if structured_additive_hidden_error is not None:
                structured_additive_one_step_nmte = _safe_relative_error(
                    structured_additive_hidden_error[trans_rows], actual_next
                )
                structured_nonlinear_one_step_nmte = _safe_relative_error(
                    structured_nonlinear_hidden_error[trans_rows], actual_next
                )
            local_tangent_mean = float(
                np.mean(local_tangent_fraction[trans_rows])
            )
            torus_tangent_mean = (
                None
                if torus_tangent_fraction is None else
                float(np.mean(torus_tangent_fraction[trans_rows]))
            )
            tangent_mean = float(
                np.mean(primary_tangent_fraction[trans_rows])
            )
            if phase_action_error is not None:
                phase_rmse = float(
                    np.sqrt(np.mean(phase_action_error[trans_rows] ** 2))
                )
                phase_normalized_rmse = float(
                    np.sqrt(np.mean(phase_action_normalized[trans_rows] ** 2))
                )

            first_state = int(data.transition_prev[trans_rows[0]])
            velocities = velocity[trans_rows]
            actual_indices = np.concatenate([
                np.asarray([first_state], dtype=int),
                data.transition_next[trans_rows],
            ])
            actual_latent = latent[actual_indices]
            actual_hidden = ambient[actual_indices]
            decoder_mode = "selected" if decoders is not None else "base"
            decode_context = np.vstack([
                previous_velocity[first_state],
                velocities,
            ])

            generic_rolled_latent = reference.reduced_map.rollout(
                latent[first_state], velocities
            )
            generic_rolled_hidden = reference.decode_latent(
                generic_rolled_latent,
                decode_context,
                decoder_mode=decoder_mode,
            )
            generic_hidden_error = np.linalg.norm(
                actual_hidden - generic_rolled_hidden, axis=1
            )
            generic_rollout_full_nmte = _safe_relative_error(
                generic_hidden_error, actual_hidden
            )
            generic_rollout_nmte = _safe_relative_error(
                generic_hidden_error[1:], actual_hidden[1:]
            )
            generic_rollout_latent_rmse = float(
                np.sqrt(np.mean((actual_latent[1:] - generic_rolled_latent[1:]) ** 2))
            )

            if phase_map is not None:
                additive_latent = phase_map.rollout_features(
                    latent[first_state], velocities, mode="additive"
                )
                nonlinear_latent = phase_map.rollout_features(
                    latent[first_state], velocities, mode="nonlinear"
                )
                additive_hidden = reference.decode_latent(
                    additive_latent, decode_context, decoder_mode=decoder_mode
                )
                nonlinear_hidden = reference.decode_latent(
                    nonlinear_latent, decode_context, decoder_mode=decoder_mode
                )
                additive_error = np.linalg.norm(actual_hidden - additive_hidden, axis=1)
                nonlinear_error = np.linalg.norm(actual_hidden - nonlinear_hidden, axis=1)
                structured_additive_rollout_full_nmte = _safe_relative_error(
                    additive_error, actual_hidden
                )
                structured_additive_rollout_nmte = _safe_relative_error(
                    additive_error[1:], actual_hidden[1:]
                )
                structured_nonlinear_rollout_full_nmte = _safe_relative_error(
                    nonlinear_error, actual_hidden
                )
                structured_nonlinear_rollout_nmte = _safe_relative_error(
                    nonlinear_error[1:], actual_hidden[1:]
                )
                actual_theta = torus_angles_from_features(actual_latent)
                additive_theta = torus_angles_from_features(additive_latent)
                nonlinear_theta = torus_angles_from_features(nonlinear_latent)
                structured_additive_phase_rmse = _rmse_phase_path(
                    actual_theta[1:], additive_theta[1:]
                )
                structured_nonlinear_phase_rmse = _rmse_phase_path(
                    actual_theta[1:], nonlinear_theta[1:]
                )

            if angles is not None and reference.action_matrix is not None:
                start_angle = angles[actual_indices[0]]
                end_angle = angles[actual_indices[-1]]
                predicted_total = reference.action_matrix @ np.sum(velocities, axis=0)
                path_cocycle_error = float(
                    np.linalg.norm(
                        wrap_angles(end_angle - start_angle - predicted_total)
                    )
                )

        # External loop includes the learned h0 anchor; internal loops compare
        # recurrent states generated by the same mechanism and are the primary
        # path-independence diagnostic.
        start_coord = data.coords[state_idx[0]]
        end_coord = data.coords[state_idx[-1]]
        closed_loop = bool(
            np.linalg.norm(end_coord - start_coord) <= loop_coordinate_tolerance
        )
        closed_loop_hidden_return = None
        closed_loop_phase_return = None
        if closed_loop and state_idx.size >= 2:
            closed_loop_hidden_return = float(
                np.linalg.norm(ambient[state_idx[-1]] - ambient[state_idx[0]])
                / max(np.linalg.norm(ambient[state_idx[0]]), _EPS)
            )
            if angles is not None:
                closed_loop_phase_return = float(
                    np.linalg.norm(
                        wrap_angles(angles[state_idx[-1]] - angles[state_idx[0]])
                    )
                )

        internal_hidden: List[float] = []
        internal_phase: List[float] = []
        rounded_coords = np.round(data.coords[state_idx], 8)
        _, inverse = np.unique(rounded_coords, axis=0, return_inverse=True)
        for coordinate_group in np.unique(inverse):
            local_rows = np.flatnonzero(inverse == coordinate_group)
            if local_rows.size < 2:
                continue
            for i in range(local_rows.size - 1):
                for j in range(i + 1, local_rows.size):
                    a = int(state_idx[local_rows[i]])
                    b = int(state_idx[local_rows[j]])
                    # Exclude comparisons involving the explicit start encoder.
                    if data.is_initial[a] or data.is_initial[b]:
                        continue
                    if data.time_indices[b] <= data.time_indices[a]:
                        continue
                    internal_hidden.append(
                        float(
                            np.linalg.norm(ambient[b] - ambient[a])
                            / max(np.linalg.norm(ambient[a]), _EPS)
                        )
                    )
                    if angles is not None:
                        internal_phase.append(
                            float(np.linalg.norm(wrap_angles(angles[b] - angles[a])))
                        )
        all_internal_loop_hidden.extend(internal_hidden)
        all_internal_loop_phase.extend(internal_phase)

        # For version 2 the structured nonlinear phase map is the primary
        # controlled rollout.  The generic map remains available explicitly.
        primary_rollout_nmte = (
            structured_nonlinear_rollout_nmte
            if structured_nonlinear_rollout_nmte is not None else generic_rollout_nmte
        )
        primary_rollout_full_nmte = (
            structured_nonlinear_rollout_full_nmte
            if structured_nonlinear_rollout_full_nmte is not None
            else generic_rollout_full_nmte
        )
        primary_one_step_nmte = (
            structured_nonlinear_one_step_nmte
            if structured_nonlinear_one_step_nmte is not None
            else generic_one_step_nmte
        )

        rows.append({
            "label": label,
            "trajectory_id": int(trajectory_id),
            "correct": trajectory_correct_map.get(int(trajectory_id)),
            "n_states": int(state_idx.size),
            "n_transitions": int(trans_rows.size),
            "local_cross_mfe": local_cross_mfe,
            "surface_cross_mfe": surface_cross_mfe,
            "conditioned_surface_cross_mfe": conditioned_surface_cross_mfe,
            "fiber_surface_cross_mfe": fiber_surface_cross_mfe,
            "initial_surface_cross_mfe": initial_cross_mfe,
            "support_distance_mean": float(np.mean(support_values)),
            "support_distance_q95": float(np.quantile(support_values, 0.95)),
            "support_ood_fraction": support_ood_fraction,
            "normal_residual_z_mean": float(np.mean(normal_values)),
            "normal_residual_z_q95": float(np.quantile(normal_values, 0.95)),
            "normal_ood_fraction": normal_ood_fraction,
            "one_step_nmte": primary_one_step_nmte,
            "generic_one_step_nmte": generic_one_step_nmte,
            "generic_one_step_latent_rmse": generic_one_step_latent_rmse,
            "structured_additive_one_step_nmte": structured_additive_one_step_nmte,
            "structured_nonlinear_one_step_nmte": structured_nonlinear_one_step_nmte,
            "rollout_cross_nmte": primary_rollout_nmte,
            "rollout_full_cross_nmte": primary_rollout_full_nmte,
            "generic_rollout_cross_nmte": generic_rollout_nmte,
            "generic_rollout_full_cross_nmte": generic_rollout_full_nmte,
            "generic_rollout_latent_rmse": generic_rollout_latent_rmse,
            "structured_additive_rollout_cross_nmte": structured_additive_rollout_nmte,
            "structured_additive_rollout_full_cross_nmte": structured_additive_rollout_full_nmte,
            "structured_nonlinear_rollout_cross_nmte": structured_nonlinear_rollout_nmte,
            "structured_nonlinear_rollout_full_cross_nmte": structured_nonlinear_rollout_full_nmte,
            "structured_additive_phase_rollout_rmse": structured_additive_phase_rmse,
            "structured_nonlinear_phase_rollout_rmse": structured_nonlinear_phase_rmse,
            "local_tangent_fraction": local_tangent_mean,
            "torus_tangent_fraction": torus_tangent_mean,
            "tangent_fraction": tangent_mean,
            "phase_action_rmse": phase_rmse,
            "phase_action_normalized_rmse": phase_normalized_rmse,
            "path_cocycle_error": path_cocycle_error,
            "closed_loop": closed_loop,
            "closed_loop_hidden_return": closed_loop_hidden_return,
            "closed_loop_phase_return": closed_loop_phase_return,
            "n_internal_closed_loops": int(len(internal_hidden)),
            "internal_closed_loop_hidden_return": (
                float(np.mean(internal_hidden)) if internal_hidden else None
            ),
            "internal_closed_loop_phase_return": (
                float(np.mean(internal_phase)) if internal_phase else None
            ),
        })

    summary = summarize_trajectory_metrics(rows)
    phase_dispersion = (
        _coordinate_phase_dispersion(
            data.coords,
            angles,
            time_indices=data.time_indices,
            trajectory_ids=data.trajectory_ids,
        )
        if angles is not None else None
    )
    endpoint_invariance = _endpoint_path_invariance(data, ambient, angles)
    state_diagnostics = {
        "support_distance_q95": float(np.quantile(atlas_result["support_distance"], 0.95)),
        "normal_residual_z_q95": float(np.quantile(atlas_result["residual_z"], 0.95)),
        "support_ood_fraction": (
            None
            if support_distance_threshold is None else
            float(np.mean(atlas_result["support_distance"] > support_distance_threshold))
        ),
        "normal_ood_fraction": (
            None
            if normal_residual_z_threshold is None else
            float(np.mean(atlas_result["residual_z"] > normal_residual_z_threshold))
        ),
        "n_internal_closed_loops": int(len(all_internal_loop_hidden)),
        "mean_internal_closed_loop_hidden_return": (
            float(np.mean(all_internal_loop_hidden))
            if all_internal_loop_hidden else None
        ),
        "mean_internal_closed_loop_phase_return": (
            float(np.mean(all_internal_loop_phase))
            if all_internal_loop_phase else None
        ),
    }
    return {
        "label": label,
        "latent_kind": reference.latent_kind,
        "reference_version": reference.reference_version,
        "decoder_mode_selected": (
            decoders.selected_mode if decoders is not None else None
        ),
        "fiber_dimension": (
            getattr(reference, "fiber_model", None).n_components
            if getattr(reference, "fiber_model", None) is not None else None
        ),
        "trajectory_metrics": rows,
        "summary": summary,
        "phase_dispersion": phase_dispersion,
        "endpoint_path_invariance": endpoint_invariance,
        "state_diagnostics": state_diagnostics,
        "time_resolved_surface_error": {
            str(time): {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "n": int(len(values)),
            }
            for time, values in sorted(time_resolved.items())
        },
    }


def _rmse_phase_path(actual_theta: np.ndarray, predicted_theta: np.ndarray) -> float:
    actual_theta = _finite_2d(actual_theta, "actual_theta")
    predicted_theta = _finite_2d(predicted_theta, "predicted_theta")
    if actual_theta.shape != predicted_theta.shape:
        raise ValueError("actual/predicted phase path shapes differ")
    return float(np.sqrt(np.mean(wrap_angles(actual_theta - predicted_theta) ** 2)))


def _bootstrap_mean_ci(
    values: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
    ci: float,
) -> Tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    if values.size == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, values.size, size=(int(n_bootstrap), values.size))
    means = values[draws].mean(axis=1)
    alpha = (1.0 - float(ci)) / 2.0
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1.0 - alpha))


def summarize_trajectory_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    n_bootstrap: int = 2000,
    seed: int = 42,
    ci: float = 0.95,
) -> Dict[str, Any]:
    if not rows:
        return {"n_trajectories": 0, "metrics": {}}
    excluded = {"label", "trajectory_id", "correct", "closed_loop"}
    metric_names = sorted({
        key
        for row in rows
        for key, value in row.items()
        if key not in excluded
        and isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, bool)
    })
    summary: Dict[str, Any] = {}
    for metric_idx, metric in enumerate(metric_names):
        values = np.asarray([
            float(row[metric])
            for row in rows
            if row.get(metric) is not None and np.isfinite(float(row[metric]))
        ], dtype=np.float64)
        if values.size == 0:
            continue
        low, high = _bootstrap_mean_ci(
            values,
            n_bootstrap=n_bootstrap,
            seed=seed + metric_idx,
            ci=ci,
        )
        summary[metric] = {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
            "ci_low": low,
            "ci_high": high,
            "n": int(values.size),
        }
    correct_count = sum(row.get("correct") is True for row in rows)
    incorrect_count = sum(row.get("correct") is False for row in rows)
    return {
        "n_trajectories": int(len(rows)),
        "n_correct": int(correct_count),
        "n_incorrect": int(incorrect_count),
        "metrics": summary,
    }


def _metric_values(
    evaluation: Mapping[str, Any], metric: str
) -> np.ndarray:
    values = [
        float(row[metric])
        for row in evaluation.get("trajectory_metrics", [])
        if row.get(metric) is not None and np.isfinite(float(row[metric]))
    ]
    return np.asarray(values, dtype=np.float64)


def _bootstrap_ratio_ci(
    target: np.ndarray,
    source: np.ndarray,
    *,
    seed: int,
    n_bootstrap: int,
    ci: float = 0.95,
    statistic: str = "median",
) -> Dict[str, Any]:
    target = np.asarray(target, dtype=np.float64)
    source = np.asarray(source, dtype=np.float64)
    target = target[np.isfinite(target)]
    source = source[np.isfinite(source)]
    if target.size == 0 or source.size == 0:
        return {
            "available": False,
            "ratio": None,
            "ci_low": None,
            "ci_high": None,
            "n_target": int(target.size),
            "n_source": int(source.size),
        }
    reducer = np.median if statistic == "median" else np.mean
    target_stat = float(reducer(target))
    source_stat = float(reducer(source))
    if abs(source_stat) <= _EPS:
        return {
            "available": False,
            "reason": "source statistic is zero",
            "ratio": None,
            "ci_low": None,
            "ci_high": None,
            "n_target": int(target.size),
            "n_source": int(source.size),
        }
    rng = np.random.default_rng(int(seed))
    n_bootstrap = max(1000, int(n_bootstrap))
    ratios = np.empty(n_bootstrap, dtype=np.float64)
    for idx in range(n_bootstrap):
        target_sample = target[rng.integers(0, target.size, size=target.size)]
        source_sample = source[rng.integers(0, source.size, size=source.size)]
        denominator = float(reducer(source_sample))
        ratios[idx] = (
            float(reducer(target_sample)) / denominator
            if abs(denominator) > _EPS else np.nan
        )
    ratios = ratios[np.isfinite(ratios)]
    alpha = 1.0 - float(ci)
    return {
        "available": bool(ratios.size),
        "statistic": statistic,
        "target_statistic": target_stat,
        "source_statistic": source_stat,
        "ratio": float(target_stat / source_stat),
        "ci_low": float(np.quantile(ratios, alpha / 2.0)),
        "ci_high": float(np.quantile(ratios, 1.0 - alpha / 2.0)),
        "n_target": int(target.size),
        "n_source": int(source.size),
        "n_bootstrap": int(ratios.size),
    }


def calibrate_equivalence_thresholds(
    navigation_evaluation: Mapping[str, Any],
    *,
    quantile: float = 0.95,
    tolerance_multiplier: float = 1.25,
    ratio_margin: Optional[float] = None,
    max_support_ood_fraction: float = 0.10,
    max_normal_ood_fraction: float = 0.10,
    max_time_dependence_fraction: float = 0.50,
    max_final_phase_variance: float = 0.10,
    max_internal_loop_phase_return: float = 0.35,
    reference_diagnostics: Optional[Mapping[str, Any]] = None,
    source_min_velocity_specificity_ratio: float = 1.25,
    source_max_normalized_additive_rmse: float = 0.75,
    source_max_surface_mfe: float = 0.50,
    source_max_rollout_nmte: float = 0.50,
    source_min_torus_tangent_fraction: float = 0.50,
) -> Dict[str, Any]:
    """Store source distributions for ratio-based operational equivalence.

    The previous implementation used ``1.25 * source_q95``.  That combines an
    extreme source tail with an additional multiplier and can produce very wide
    bands.  Version 2 bootstraps the target/source median ratio and requires its
    confidence interval to lie inside a preregistered multiplicative margin.
    """
    rows = navigation_evaluation.get("trajectory_metrics", [])
    if not rows:
        raise ValueError("navigation_evaluation contains no trajectory metrics")
    margin = float(tolerance_multiplier if ratio_margin is None else ratio_margin)
    lower_metrics = [
        "surface_cross_mfe",
        "local_cross_mfe",
        "conditioned_surface_cross_mfe",
        "fiber_surface_cross_mfe",
        "initial_surface_cross_mfe",
        "rollout_cross_nmte",
        "generic_rollout_cross_nmte",
        "structured_additive_rollout_cross_nmte",
        "structured_nonlinear_rollout_cross_nmte",
        "one_step_nmte",
        "phase_action_rmse",
        "phase_action_normalized_rmse",
        "path_cocycle_error",
        "internal_closed_loop_phase_return",
    ]
    higher_metrics = [
        "tangent_fraction",
        "local_tangent_fraction",
        "torus_tangent_fraction",
    ]
    source_values: Dict[str, List[float]] = {}
    for metric in lower_metrics + higher_metrics:
        values = _metric_values(navigation_evaluation, metric)
        if values.size:
            source_values[metric] = values.tolist()

    # Legacy thresholds are still emitted for auditability, but are not used by
    # the version-2 verdict.
    legacy_upper: Dict[str, float] = {}
    for metric in lower_metrics:
        values = np.asarray(source_values.get(metric, []), dtype=np.float64)
        if values.size:
            legacy_upper[metric] = float(
                np.quantile(values, quantile) * margin
            )
    legacy_lower: Dict[str, float] = {}
    for metric in higher_metrics:
        values = np.asarray(source_values.get(metric, []), dtype=np.float64)
        if values.size:
            legacy_lower[metric] = float(
                np.quantile(values, 1.0 - quantile) / margin
            )

    state_diag = navigation_evaluation.get("state_diagnostics", {})
    support_threshold = state_diag.get("support_distance_q95")
    normal_threshold = state_diag.get("normal_residual_z_q95")

    phase_diagnostics = (
        (reference_diagnostics or {}).get("structured_phase_map") or {}
    )
    diagnostics_supplied = reference_diagnostics is not None
    specificity = phase_diagnostics.get("velocity_specificity_ratio")
    normalized_rmse = phase_diagnostics.get("normalized_additive_rmse")
    def _source_median(metric: str) -> Optional[float]:
        values = np.asarray(source_values.get(metric, []), dtype=np.float64)
        values = values[np.isfinite(values)]
        return float(np.median(values)) if values.size else None

    source_surface = _source_median("surface_cross_mfe")
    source_rollout = _source_median("rollout_cross_nmte")
    tangent_metric = (
        "torus_tangent_fraction"
        if source_values.get("torus_tangent_fraction") else "tangent_fraction"
    )
    source_tangent = _source_median(tangent_metric)
    source_quality_checks = {
        "structured_phase_map_available": {
            "value": bool(phase_diagnostics),
            "passed": bool(phase_diagnostics),
        },
        "heldout_surface_mfe": {
            "value": source_surface,
            "threshold": float(source_max_surface_mfe),
            "direction": "less_equal",
            "passed": bool(
                source_surface is not None
                and source_surface <= float(source_max_surface_mfe)
            ),
        },
        "heldout_rollout_nmte": {
            "value": source_rollout,
            "threshold": float(source_max_rollout_nmte),
            "direction": "less_equal",
            "passed": bool(
                source_rollout is not None
                and source_rollout <= float(source_max_rollout_nmte)
            ),
        },
        "heldout_torus_tangent_fraction": {
            "metric": tangent_metric,
            "value": source_tangent,
            "threshold": float(source_min_torus_tangent_fraction),
            "direction": "greater_equal",
            "passed": bool(
                source_tangent is not None
                and source_tangent >= float(source_min_torus_tangent_fraction)
            ),
        },
        "velocity_specificity_ratio": {
            "value": specificity,
            "threshold": float(source_min_velocity_specificity_ratio),
            "direction": "greater_equal",
            "passed": bool(
                specificity is not None
                and float(specificity) >= float(source_min_velocity_specificity_ratio)
            ),
        },
        "normalized_additive_rmse": {
            "value": normalized_rmse,
            "threshold": float(source_max_normalized_additive_rmse),
            "direction": "less_equal",
            "passed": bool(
                normalized_rmse is not None
                and float(normalized_rmse) <= float(source_max_normalized_additive_rmse)
            ),
        },
    }
    source_quality_supported = (
        all(item["passed"] for item in source_quality_checks.values())
        if diagnostics_supplied else True
    )
    if not diagnostics_supplied:
        source_quality_checks["assessment_status"] = {
            "value": "not_assessed_no_reference_diagnostics",
            "passed": True,
        }
    return {
        "method": "bootstrap_median_ratio_v2",
        "quantile": float(quantile),
        "ratio_margin": margin,
        "source_values": source_values,
        "lower_is_better_metrics": lower_metrics,
        "higher_is_better_metrics": higher_metrics,
        "support_distance_threshold": support_threshold,
        "normal_residual_z_threshold": normal_threshold,
        "max_support_ood_fraction": float(max_support_ood_fraction),
        "max_normal_ood_fraction": float(max_normal_ood_fraction),
        "max_time_dependence_fraction": float(max_time_dependence_fraction),
        "max_final_phase_variance": float(max_final_phase_variance),
        "max_internal_loop_phase_return": float(max_internal_loop_phase_return),
        "source_quality_supported": bool(source_quality_supported),
        "source_quality_checks": source_quality_checks,
        "legacy_upper": legacy_upper,
        "legacy_lower": legacy_lower,
        # Backwards-compatible keys.
        "tolerance_multiplier": margin,
        "upper": legacy_upper,
        "lower": legacy_lower,
    }


def operational_equivalence_verdict(
    target_evaluation: Mapping[str, Any],
    thresholds: Mapping[str, Any],
    *,
    require_phase: bool = True,
    require_anchor: bool = True,
    require_support: bool = True,
    require_holonomy: bool = True,
    require_path_independence: bool = True,
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Apply nested source-normalized shared-manifold criteria."""
    latent_kind = str(target_evaluation.get("latent_kind", "pca"))
    reference_version = int(target_evaluation.get("reference_version", 1))
    ratio_margin = float(
        thresholds.get("ratio_margin", thresholds.get("tolerance_multiplier", 1.25))
    )
    geometry_metric = "surface_cross_mfe" if latent_kind == "torus" else "local_cross_mfe"
    rollout_metric = "rollout_cross_nmte"
    tangent_metric = (
        "torus_tangent_fraction"
        if _metric_values(target_evaluation, "torus_tangent_fraction").size
        else "tangent_fraction"
    )

    checks: Dict[str, Any] = {}

    def ratio_check(metric: str, *, direction: str, seed_offset: int) -> None:
        target_values = _metric_values(target_evaluation, metric)
        source_values = np.asarray(
            thresholds.get("source_values", {}).get(metric, []), dtype=np.float64
        )
        ratio = _bootstrap_ratio_ci(
            target_values,
            source_values,
            seed=seed + seed_offset,
            n_bootstrap=n_bootstrap,
            statistic="median",
        )
        if direction == "lower":
            passed = bool(
                ratio.get("available")
                and float(ratio["ci_high"]) <= ratio_margin
            )
            bound = ratio.get("ci_high")
            threshold = ratio_margin
        else:
            lower_bound = 1.0 / ratio_margin
            passed = bool(
                ratio.get("available")
                and float(ratio["ci_low"]) >= lower_bound
            )
            bound = ratio.get("ci_low")
            threshold = lower_bound
        checks[metric] = {
            "kind": "source_normalized_ratio",
            "direction": direction,
            "ci_bound": bound,
            "threshold": threshold,
            "passed": passed,
            **ratio,
        }

    for idx, metric in enumerate([
        geometry_metric,
        rollout_metric,
        "initial_surface_cross_mfe",
        "phase_action_normalized_rmse",
    ]):
        ratio_check(metric, direction="lower", seed_offset=idx * 17)
    ratio_check(tangent_metric, direction="higher", seed_offset=101)

    state_diag = target_evaluation.get("state_diagnostics", {})
    support_ood = state_diag.get("support_ood_fraction")
    normal_ood = state_diag.get("normal_ood_fraction")
    if support_ood is None and thresholds.get("support_distance_threshold") is not None:
        support_values = _metric_values(target_evaluation, "support_distance_mean")
        if support_values.size:
            support_ood = float(
                np.mean(
                    support_values
                    > float(thresholds["support_distance_threshold"])
                )
            )
    if normal_ood is None and thresholds.get("normal_residual_z_threshold") is not None:
        normal_values = _metric_values(target_evaluation, "normal_residual_z_mean")
        if normal_values.size:
            normal_ood = float(
                np.mean(
                    normal_values
                    > float(thresholds["normal_residual_z_threshold"])
                )
            )
    checks["support_ood_fraction"] = {
        "kind": "absolute_fraction",
        "value": support_ood,
        "threshold": thresholds.get("max_support_ood_fraction", 0.10),
        "direction": "lower",
        "passed": bool(
            support_ood is not None
            and float(support_ood)
            <= float(thresholds.get("max_support_ood_fraction", 0.10))
        ),
    }
    checks["normal_ood_fraction"] = {
        "kind": "absolute_fraction",
        "value": normal_ood,
        "threshold": thresholds.get("max_normal_ood_fraction", 0.10),
        "direction": "lower",
        "passed": bool(
            normal_ood is not None
            and float(normal_ood)
            <= float(thresholds.get("max_normal_ood_fraction", 0.10))
        ),
    }

    phase_dispersion = target_evaluation.get("phase_dispersion") or {}
    time_dependence = phase_dispersion.get("time_dependence_fraction")
    final_phase_variance = (
        phase_dispersion.get("final_state_by_coordinate", {})
        .get("mean_circular_variance_over_axes")
    )
    checks["time_dependence_fraction"] = {
        "kind": "absolute",
        "value": time_dependence,
        "threshold": thresholds.get("max_time_dependence_fraction", 0.50),
        "direction": "lower",
        "passed": bool(
            time_dependence is not None
            and float(time_dependence)
            <= float(thresholds.get("max_time_dependence_fraction", 0.50))
        ),
    }
    checks["final_coordinate_phase_variance"] = {
        "kind": "absolute",
        "value": final_phase_variance,
        "threshold": thresholds.get("max_final_phase_variance", 0.10),
        "direction": "lower",
        "passed": bool(
            final_phase_variance is not None
            and float(final_phase_variance)
            <= float(thresholds.get("max_final_phase_variance", 0.10))
        ),
    }

    internal_loop_phase = state_diag.get("mean_internal_closed_loop_phase_return")
    checks["internal_closed_loop_phase_return"] = {
        "kind": "absolute",
        "value": internal_loop_phase,
        "threshold": thresholds.get("max_internal_loop_phase_return", 0.35),
        "direction": "lower",
        "available": internal_loop_phase is not None,
        "passed": bool(
            internal_loop_phase is not None
            and float(internal_loop_phase)
            <= float(thresholds.get("max_internal_loop_phase_return", 0.35))
        ),
    }

    endpoint_phase = (
        target_evaluation.get("endpoint_path_invariance", {})
        .get("mean_final_phase_distance")
    )
    checks["same_endpoint_phase_distance"] = {
        "kind": "absolute",
        "value": endpoint_phase,
        "threshold": thresholds.get("max_internal_loop_phase_return", 0.35),
        "direction": "lower",
        "available": endpoint_phase is not None,
        "passed": bool(
            endpoint_phase is not None
            and float(endpoint_phase)
            <= float(thresholds.get("max_internal_loop_phase_return", 0.35))
        ),
    }

    source_quality_supported = bool(
        thresholds.get("source_quality_supported", reference_version < 2)
    )
    checks["source_reference_quality"] = {
        "kind": "source_gate",
        "passed": source_quality_supported,
        "details": thresholds.get("source_quality_checks"),
    }

    geometry_required = [geometry_metric, tangent_metric]
    if require_support:
        geometry_required += ["support_ood_fraction", "normal_ood_fraction"]
    geometric_scaffold_supported = all(
        checks[name]["passed"] for name in geometry_required
    )

    dynamics_required = list(geometry_required) + [rollout_metric]
    if require_phase:
        dynamics_required.append("phase_action_normalized_rmse")
    controlled_dynamics_supported = bool(
        source_quality_supported
        and all(checks[name]["passed"] for name in dynamics_required)
    )

    full_required = list(dynamics_required)
    if require_anchor:
        full_required.append("initial_surface_cross_mfe")
    full_system_supported = bool(
        source_quality_supported
        and all(checks[name]["passed"] for name in full_required)
    )

    path_required = list(full_required)
    if require_holonomy:
        path_required.append("internal_closed_loop_phase_return")
    if require_path_independence:
        path_required += [
            "time_dependence_fraction",
            "final_coordinate_phase_variance",
        ]
        # Same-endpoint pairs are highly design-dependent. Require them only
        # when at least one pair is available.
        if checks["same_endpoint_phase_distance"]["available"]:
            path_required.append("same_endpoint_phase_distance")
    path_independent_supported = bool(
        latent_kind == "torus"
        and source_quality_supported
        and all(checks[name]["passed"] for name in path_required)
    )

    return {
        "reference_version": reference_version,
        "source_reference_quality_supported": source_quality_supported,
        "geometric_scaffold_supported": bool(geometric_scaffold_supported),
        "controlled_dynamics_reuse_supported": bool(controlled_dynamics_supported),
        "path_independent_cognitive_map_supported": bool(path_independent_supported),
        # Backwards-compatible nested labels.
        "dynamics_reuse_supported": bool(controlled_dynamics_supported),
        "full_system_reuse_supported": bool(full_system_supported),
        "torus_test_available": bool(latent_kind == "torus" and require_phase),
        "exact_shared_torus_supported": bool(path_independent_supported),
        "operational_exact_shared_manifold_supported": bool(
            path_independent_supported
        ),
        "geometry_metric_used": geometry_metric,
        "rollout_metric_used": rollout_metric,
        "tangent_metric_used": tangent_metric,
        "global_source_manifold_chart_available": bool(latent_kind == "torus"),
        "required_metrics": {
            "geometric_scaffold": geometry_required,
            "controlled_dynamics": dynamics_required,
            "full_system": full_required,
            "path_independent_map": path_required,
        },
        "checks": checks,
        "interpretation": {
            "geometry": (
                "Target trajectories reuse the navigation-specific local/toroidal geometry"
                if geometric_scaffold_supported else
                "At least one source-normalized geometry/support criterion failed"
            ),
            "dynamics": (
                "The source structured controlled dynamics predict target trajectories"
                if controlled_dynamics_supported else
                "Geometric proximity does not pass the controlled-dynamics criterion"
            ),
            "path_independence": (
                "Phase-action, loop holonomy and coordinate/path invariance jointly pass"
                if path_independent_supported else
                "The strongest same-torus cognitive-map criterion is not satisfied"
            ),
        },
    }


def exact_sign_flip_pvalue(differences: Sequence[float], *, alternative: str) -> float:
    """Exact paired sign-flip test over seed-level differences.

    ``alternative='less'`` tests mean(differences) < 0 and ``'greater'`` tests
    mean(differences) > 0.
    """
    diff = np.asarray(differences, dtype=np.float64)
    diff = diff[np.isfinite(diff)]
    if diff.size == 0:
        return float("nan")
    if alternative not in {"less", "greater", "two-sided"}:
        raise ValueError("alternative must be less, greater or two-sided")
    observed = float(np.mean(diff))
    if diff.size <= 20:
        signs = ((np.arange(1 << diff.size)[:, None] >> np.arange(diff.size)) & 1) * 2 - 1
        null = np.mean(signs * diff[None, :], axis=1)
    else:
        rng = np.random.default_rng(0)
        signs = rng.choice([-1.0, 1.0], size=(200_000, diff.size))
        null = np.mean(signs * diff[None, :], axis=1)
    if alternative == "less":
        exceedances = int(np.sum(null <= observed))
    elif alternative == "greater":
        exceedances = int(np.sum(null >= observed))
    else:
        exceedances = int(np.sum(np.abs(null) >= abs(observed)))
    # Complete enumeration already contains the observed assignment and needs
    # no Monte-Carlo add-one correction. Use add-one only for sampled nulls.
    if diff.size <= 20:
        return float(exceedances / null.size)
    return float((exceedances + 1) / (null.size + 1))


def torus_artifacts_from_analysis_dir(
    analysis_dir: str | Path,
) -> Dict[str, Any]:
    """Load source unit selection and optional cohomological coordinates."""
    analysis_dir = Path(analysis_dir)
    arrays_path = analysis_dir / "torus_analysis_arrays.npz"
    summary_path = analysis_dir / "torus_analysis_summary.json"
    if not arrays_path.exists():
        raise FileNotFoundError(
            f"Missing {arrays_path}; run analyze_torus.py or Exp1 topology analysis first"
        )
    with np.load(arrays_path, allow_pickle=False) as archive:
        if "selected_unit_indices" not in archive:
            raise KeyError(f"{arrays_path} lacks selected_unit_indices")
        selected = np.asarray(archive["selected_unit_indices"], dtype=int)
        landmark_indices = (
            np.asarray(archive["ph_landmark_input_indices_original"], dtype=int)
            if "ph_landmark_input_indices_original" in archive else None
        )
        turns = (
            np.asarray(archive["decoded_torus_coordinates_turns"], dtype=np.float64)
            if "decoded_torus_coordinates_turns" in archive else None
        )
        analysis_input_indices = (
            np.asarray(archive["analysis_input_indices"], dtype=np.int64)
            if "analysis_input_indices" in archive else None
        )
        module_selection_input_indices = (
            np.asarray(archive["module_selection_input_indices"], dtype=np.int64)
            if "module_selection_input_indices" in archive else None
        )
        topology_evaluation_input_indices = (
            np.asarray(archive["topology_evaluation_input_indices"], dtype=np.int64)
            if "topology_evaluation_input_indices" in archive else None
        )
    summary: Dict[str, Any] = {}
    if summary_path.exists():
        import json
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
    persistent = summary.get("persistent_cohomology", {}) if summary else {}
    return {
        "selected_unit_indices": selected,
        "landmark_indices": landmark_indices,
        "landmark_torus_turns": turns,
        "analysis_input_indices": analysis_input_indices,
        "module_selection_input_indices": module_selection_input_indices,
        "topology_evaluation_input_indices": topology_evaluation_input_indices,
        "summary": summary,
        "source_torus_confirmed": persistent.get("is_torus_like") is True,
        "source_torus_exploratory": persistent.get("is_torus_like_exploratory") is True,
        "n_shuffles": persistent.get("n_shuffles"),
    }
