"""Leakage-controlled source reduced models for one analyzed navigation source.

This module implements the first stage of the GPM-CMap upgrade plan.  It keeps
all model selection strictly outside the final navigation test split and
separates the toroidal base from residual neural fibres.  The formal scope of
all conclusions produced here is *conditional on the analyzed navigation
source*; this module does not implement a multi-source cohort.

The topology-derived phase encoder is fitted only from the already formal,
held-out torus analysis artifact.  Source-fit trajectories estimate dynamics
and candidate models, Source-validation trajectories select hyperparameters,
and Source-test trajectories are consumed exactly once by :func:`evaluate_test`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures

from gpm_cmap.analysis.shared_manifold import (
    CircularTorusChart,
    TrajectoryData,
    summarize_trajectory_metrics,
)
from gpm_cmap.analysis.toroidal_models import (
    FourierFeatureMap,
    generate_unimodular_matrices,
)

SCHEMA_VERSION = "source_reduced_v3"
CLAIM_SCOPE = "conditional on the analyzed navigation source"
_EPS = 1e-12


class NumericalRolloutError(RuntimeError):
    """Raised when a reduced-model rollout becomes numerically non-finite.

    A flexible candidate can fit finite one-step training targets yet diverge
    recursively on held-out trajectories.  Such divergence is a model-selection
    outcome, not a reason to terminate evaluation of the remaining candidates.
    """


def _numerical_instability_reason(exc: BaseException) -> Optional[str]:
    """Return a diagnostic string for numerical rollout failures, else ``None``."""
    if isinstance(exc, NumericalRolloutError):
        return str(exc)
    if isinstance(exc, (FloatingPointError, OverflowError, RuntimeWarning)):
        return f"{type(exc).__name__}: {exc}"
    if isinstance(exc, ValueError):
        message = str(exc)
        lowered = message.lower()
        markers = (
            "contains infinity",
            "contains nan",
            "nan or inf",
            "too large for dtype",
            "array must not contain infs or nans",
        )
        if any(marker in lowered for marker in markers):
            return f"ValueError: {message}"
    return None


def _finite_2d(value: np.ndarray, name: str, n_cols: Optional[int] = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must be 2-D; got {array.shape}")
    if array.shape[0] < 1 or array.shape[1] < 1:
        raise ValueError(f"{name} is empty: {array.shape}")
    if n_cols is not None and array.shape[1] != int(n_cols):
        raise ValueError(f"{name} must have {n_cols} columns; got {array.shape}")
    if not np.isfinite(array).all():
        raise FloatingPointError(f"{name} contains NaN or Inf")
    return array


def _finite_1d(value: np.ndarray, name: str, n: Optional[int] = None) -> np.ndarray:
    array = np.asarray(value).reshape(-1)
    if n is not None and array.shape[0] != int(n):
        raise ValueError(f"{name} has {array.shape[0]} rows; expected {n}")
    if np.issubdtype(array.dtype, np.number) and not np.isfinite(array).all():
        raise FloatingPointError(f"{name} contains NaN or Inf")
    return array


def wrap_angles(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    return (value + np.pi) % (2.0 * np.pi) - np.pi


def circular_features(theta: np.ndarray) -> np.ndarray:
    theta = _finite_2d(theta, "theta", 2)
    return np.column_stack(
        [
            np.cos(theta[:, 0]),
            np.sin(theta[:, 0]),
            np.cos(theta[:, 1]),
            np.sin(theta[:, 1]),
        ]
    )


def circular_angles(features: np.ndarray) -> np.ndarray:
    features = _finite_2d(features, "circular_features", 4)
    normalized = features.copy()
    for start in (0, 2):
        norm = np.linalg.norm(normalized[:, start : start + 2], axis=1, keepdims=True)
        norm = np.maximum(norm, _EPS)
        normalized[:, start : start + 2] /= norm
    return np.column_stack(
        [
            np.arctan2(normalized[:, 1], normalized[:, 0]),
            np.arctan2(normalized[:, 3], normalized[:, 2]),
        ]
    )


def _relative_error(actual: np.ndarray, predicted: np.ndarray) -> float:
    actual = _finite_2d(actual, "actual")
    predicted = _finite_2d(predicted, "predicted")
    if actual.shape != predicted.shape:
        raise ValueError("actual and predicted shapes differ")
    numerator = np.linalg.norm(actual - predicted, axis=1)
    denominator = float(np.max(np.linalg.norm(actual, axis=1)))
    return float(np.mean(numerator) / max(denominator, _EPS))


def _wrapped_rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    actual = _finite_2d(actual, "actual_phase", 2)
    predicted = _finite_2d(predicted, "predicted_phase", 2)
    if actual.shape != predicted.shape:
        raise ValueError("actual and predicted phase shapes differ")
    return float(np.sqrt(np.mean(wrap_angles(actual - predicted) ** 2)))


def _mean_se(values: Sequence[float]) -> Tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return float("inf"), float("inf")
    mean = float(np.mean(array))
    se = float(np.std(array, ddof=1) / math.sqrt(array.size)) if array.size > 1 else 0.0
    return mean, se


def _one_standard_error_choice(
    rows: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    se_metric: str,
    complexity_key,
) -> Mapping[str, Any]:
    usable = []
    for row in rows:
        value = row.get(metric)
        if value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(numeric):
            usable.append(row)
    if not usable:
        raise RuntimeError(f"No finite candidates for {metric}")
    best = min(usable, key=lambda row: (float(row[metric]), complexity_key(row)))
    threshold = float(best[metric]) + max(0.0, float(best.get(se_metric, 0.0)))
    eligible = [row for row in usable if float(row[metric]) <= threshold + _EPS]
    selected = min(eligible, key=complexity_key)
    return {
        **selected,
        "one_se_best_metric": float(best[metric]),
        "one_se_best_se": float(best.get(se_metric, 0.0)),
        "one_se_threshold": float(threshold),
        "one_se_best_candidate": dict(best),
    }


def trajectory_fingerprints(data: TrajectoryData) -> Dict[str, List[int]]:
    """Fingerprint coordinate/time trajectories independently of neural activity."""
    fingerprints: Dict[str, List[int]] = {}
    for trajectory_id in data.unique_trajectories:
        rows = data.trajectory_state_indices(int(trajectory_id))
        if rows.size == 0:
            continue
        relative_time = (
            data.time_indices[rows] - int(data.time_indices[rows][0])
        ).astype(np.int64)
        coordinates = np.round(data.coords[rows], decimals=10).astype(np.float64)
        digest = hashlib.sha256()
        digest.update(np.asarray([rows.size], dtype=np.int64).tobytes())
        digest.update(relative_time.tobytes())
        digest.update(coordinates.tobytes())
        fingerprints.setdefault(digest.hexdigest(), []).append(int(trajectory_id))
    return fingerprints


def audit_disjoint_trajectory_sets(
    datasets: Mapping[str, TrajectoryData],
    *,
    fail_on_overlap: bool = True,
) -> Dict[str, Any]:
    """Audit pairwise trajectory overlap and optionally fail loudly."""
    fingerprints = {name: trajectory_fingerprints(data) for name, data in datasets.items()}
    overlaps: List[Dict[str, Any]] = []
    names = list(datasets)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            shared = sorted(set(fingerprints[left]).intersection(fingerprints[right]))
            if shared:
                first = shared[0]
                overlaps.append(
                    {
                        "left": left,
                        "right": right,
                        "n_shared": len(shared),
                        "example_fingerprint": first,
                        "left_trajectory_ids": fingerprints[left][first],
                        "right_trajectory_ids": fingerprints[right][first],
                    }
                )
    audit = {
        "algorithm": "sha256(n_states, relative_time, rounded_coordinates)",
        "sets": {
            name: {
                "n_trajectories": int(data.unique_trajectories.size),
                "n_unique_fingerprints": int(len(fingerprints[name])),
                "fingerprints": fingerprints[name],
            }
            for name, data in datasets.items()
        },
        "pairwise_overlaps": overlaps,
        "all_disjoint": not overlaps,
    }
    if fail_on_overlap and overlaps:
        example = overlaps[0]
        raise RuntimeError(
            "Navigation trajectory leakage detected between "
            f"{example['left']} and {example['right']}: "
            f"{example['n_shared']} shared fingerprint(s); example IDs "
            f"{example['left_trajectory_ids']} vs {example['right_trajectory_ids']}."
        )
    return audit


@dataclass(frozen=True)
class ContiguousSegment:
    trajectory_id: int
    segment_id: int
    state_indices: np.ndarray
    transition_rows: np.ndarray


def contiguous_segments(
    data: TrajectoryData,
    *,
    min_transitions: int = 1,
) -> List[ContiguousSegment]:
    """Return strict transition-connected segments after burn-in/movement filtering."""
    segments: List[ContiguousSegment] = []
    segment_id = 0
    for trajectory_id in data.unique_trajectories:
        transition_rows = data.trajectory_transition_rows(int(trajectory_id))
        if transition_rows.size == 0:
            continue
        run: List[int] = []
        for row in transition_rows.tolist():
            if run and int(data.transition_next[run[-1]]) != int(data.transition_prev[row]):
                if len(run) >= int(min_transitions):
                    rows = np.asarray(run, dtype=np.int64)
                    states = np.concatenate(
                        [data.transition_prev[rows[:1]], data.transition_next[rows]]
                    )
                    segments.append(
                        ContiguousSegment(
                            trajectory_id=int(trajectory_id),
                            segment_id=segment_id,
                            state_indices=states,
                            transition_rows=rows,
                        )
                    )
                    segment_id += 1
                run = []
            run.append(int(row))
        if len(run) >= int(min_transitions):
            rows = np.asarray(run, dtype=np.int64)
            states = np.concatenate(
                [data.transition_prev[rows[:1]], data.transition_next[rows]]
            )
            segments.append(
                ContiguousSegment(
                    trajectory_id=int(trajectory_id),
                    segment_id=segment_id,
                    state_indices=states,
                    transition_rows=rows,
                )
            )
            segment_id += 1
    return segments


def combine_trajectory_data(*datasets: TrajectoryData) -> TrajectoryData:
    """Concatenate independent splits while making state/trajectory IDs unique."""
    if not datasets:
        raise ValueError("At least one TrajectoryData object is required")
    n_units = datasets[0].n_units
    state_parts: List[np.ndarray] = []
    coord_parts: List[np.ndarray] = []
    trajectory_parts: List[np.ndarray] = []
    time_parts: List[np.ndarray] = []
    prev_parts: List[np.ndarray] = []
    next_parts: List[np.ndarray] = []
    velocity_parts: List[np.ndarray] = []
    initial_parts: List[np.ndarray] = []
    state_offset = 0
    trajectory_offset = 0
    for data in datasets:
        if data.n_units != n_units:
            raise ValueError("All datasets must have the same hidden dimension")
        unique = data.unique_trajectories
        mapping = {int(old): trajectory_offset + index for index, old in enumerate(unique)}
        state_parts.append(data.states)
        coord_parts.append(data.coords)
        trajectory_parts.append(
            np.asarray([mapping[int(value)] for value in data.trajectory_ids], dtype=np.int64)
        )
        time_parts.append(data.time_indices)
        prev_parts.append(data.transition_prev + state_offset)
        next_parts.append(data.transition_next + state_offset)
        velocity_parts.append(data.transition_velocities)
        initial_parts.append(data.is_initial)
        state_offset += data.n_states
        trajectory_offset += unique.size
    return TrajectoryData(
        states=np.concatenate(state_parts, axis=0),
        coords=np.concatenate(coord_parts, axis=0),
        trajectory_ids=np.concatenate(trajectory_parts),
        time_indices=np.concatenate(time_parts),
        transition_prev=np.concatenate(prev_parts),
        transition_next=np.concatenate(next_parts),
        transition_velocities=np.concatenate(velocity_parts, axis=0),
        is_initial=np.concatenate(initial_parts),
        metadata={
            "combined_independent_splits": True,
            "n_parts": len(datasets),
            "part_metadata": [data.metadata for data in datasets],
        },
    )


def previous_velocity_per_state(data: TrajectoryData) -> np.ndarray:
    previous = np.zeros((data.n_states, 2), dtype=np.float64)
    if data.n_transitions:
        previous[data.transition_next] = data.transition_velocities
    return previous


def current_velocity_per_state(data: TrajectoryData) -> np.ndarray:
    current = np.zeros((data.n_states, 2), dtype=np.float64)
    if data.n_transitions:
        current[data.transition_prev] = data.transition_velocities
    return current


def decoder_context_per_state(data: TrajectoryData, mode: str) -> Optional[np.ndarray]:
    if mode == "none":
        return None
    previous = previous_velocity_per_state(data)
    current = current_velocity_per_state(data)
    if mode == "previous_velocity":
        return previous
    if mode == "current_velocity":
        return current
    if mode == "previous_current_velocity":
        return np.column_stack([previous, current])
    raise ValueError(f"Unknown decoder context mode: {mode}")


def decoder_context_for_segment(velocities: np.ndarray, mode: str) -> Optional[np.ndarray]:
    velocities = np.asarray(velocities, dtype=np.float64).reshape(-1, 2)
    n_states = velocities.shape[0] + 1
    if mode == "none":
        return None
    previous = np.zeros((n_states, 2), dtype=np.float64)
    current = np.zeros((n_states, 2), dtype=np.float64)
    if velocities.size:
        previous[1:] = velocities
        current[:-1] = velocities
    if mode == "previous_velocity":
        return previous
    if mode == "current_velocity":
        return current
    if mode == "previous_current_velocity":
        return np.column_stack([previous, current])
    raise ValueError(f"Unknown decoder context mode: {mode}")


@dataclass
class AmbientStandardizer:
    selected_unit_indices: np.ndarray
    retained_unit_mask: np.ndarray
    unit_mean: np.ndarray
    unit_std: np.ndarray
    full_unit_mean: np.ndarray

    @classmethod
    def fit(
        cls,
        source_fit: TrajectoryData,
        selected_unit_indices: Sequence[int],
    ) -> "AmbientStandardizer":
        selected = np.asarray(selected_unit_indices, dtype=np.int64).reshape(-1)
        if source_fit.n_units != selected.size:
            raise ValueError(
                "source_fit must already be restricted to selected_unit_indices"
            )
        full_mean = source_fit.states.mean(axis=0)
        full_std = source_fit.states.std(axis=0)
        retained = np.isfinite(full_mean) & np.isfinite(full_std) & (full_std > 1e-8)
        if int(retained.sum()) < 3:
            raise ValueError("Too few non-constant source units remain")
        return cls(
            selected_unit_indices=selected,
            retained_unit_mask=retained,
            unit_mean=np.asarray(full_mean[retained], dtype=np.float64),
            unit_std=np.asarray(full_std[retained], dtype=np.float64),
            full_unit_mean=np.asarray(full_mean, dtype=np.float64),
        )

    @property
    def ambient_dim(self) -> int:
        return int(np.sum(self.retained_unit_mask))

    def transform(self, states: np.ndarray) -> np.ndarray:
        states = _finite_2d(states, "selected_states")
        if states.shape[1] != self.selected_unit_indices.size:
            raise ValueError("Selected hidden dimension differs from source standardizer")
        return (
            states[:, self.retained_unit_mask] - self.unit_mean
        ) / self.unit_std

    def inverse(self, ambient: np.ndarray) -> np.ndarray:
        ambient = _finite_2d(ambient, "ambient")
        if ambient.shape[1] != self.ambient_dim:
            raise ValueError("Ambient dimension differs from source standardizer")
        states = np.tile(self.full_unit_mean, (ambient.shape[0], 1))
        states[:, self.retained_unit_mask] = ambient * self.unit_std + self.unit_mean
        return states


@dataclass
class FrozenTorusPhaseEncoder:
    """Topology-derived, frozen phase encoder independent of fit/val/test."""

    selected_unit_indices: np.ndarray
    retained_unit_mask: np.ndarray
    unit_mean: np.ndarray
    unit_std: np.ndarray
    pca: PCA
    chart: CircularTorusChart
    diagnostics: Dict[str, Any]

    @classmethod
    def fit_from_formal_topology(
        cls,
        topology_data: TrajectoryData,
        *,
        selected_unit_indices: Sequence[int],
        landmark_indices: np.ndarray,
        landmark_torus_turns: np.ndarray,
        chart_dim: int = 8,
        seed: int = 42,
    ) -> "FrozenTorusPhaseEncoder":
        selected = np.asarray(selected_unit_indices, dtype=np.int64).reshape(-1)
        if topology_data.n_units != selected.size:
            raise ValueError("Topology data must already contain exactly the selected units")
        full_mean = topology_data.states.mean(axis=0)
        full_std = topology_data.states.std(axis=0)
        retained = np.isfinite(full_mean) & np.isfinite(full_std) & (full_std > 1e-8)
        if int(retained.sum()) < 3:
            raise ValueError("Too few non-constant topology units for phase encoding")
        mean = full_mean[retained]
        std = full_std[retained]
        standardized = (topology_data.states[:, retained] - mean) / std
        n_components = min(int(chart_dim), standardized.shape[1], standardized.shape[0] - 1)
        if n_components < 2:
            raise ValueError("Insufficient topology rank for a two-circle phase chart")
        pca = PCA(n_components=n_components, random_state=int(seed))
        chart_states = pca.fit_transform(standardized)

        original_rows = np.asarray(
            topology_data.metadata.get(
                "original_state_indices", np.arange(topology_data.n_states)
            ),
            dtype=np.int64,
        ).reshape(-1)
        if original_rows.size != topology_data.n_states:
            raise ValueError("Malformed topology original_state_indices metadata")
        original_to_local = {
            int(original): int(local) for local, original in enumerate(original_rows)
        }
        landmarks = np.asarray(landmark_indices, dtype=np.int64).reshape(-1)
        turns = _finite_2d(landmark_torus_turns, "landmark_torus_turns", 2)
        if turns.shape[0] != landmarks.size:
            raise ValueError("Torus landmark indices and coordinates differ in length")
        local_rows: List[int] = []
        local_turns: List[np.ndarray] = []
        for original, turn in zip(landmarks.tolist(), turns):
            local = original_to_local.get(int(original))
            if local is not None:
                local_rows.append(local)
                local_turns.append(turn)
        if len(local_rows) < 30:
            raise RuntimeError(
                "Fewer than 30 cohomological landmarks map into the formal topology data"
            )
        local = np.asarray(local_rows, dtype=np.int64)
        mapped_turns = np.asarray(local_turns, dtype=np.float64)
        chart = CircularTorusChart.fit(
            chart_states[local],
            standardized[local],
            mapped_turns,
            topology_data.trajectory_ids[local],
            seed=int(seed),
        )
        return cls(
            selected_unit_indices=selected,
            retained_unit_mask=retained,
            unit_mean=np.asarray(mean, dtype=np.float64),
            unit_std=np.asarray(std, dtype=np.float64),
            pca=pca,
            chart=chart,
            diagnostics={
                "fit_domain": "formal_topology_evaluation_trajectories",
                "n_topology_states": topology_data.n_states,
                "n_topology_trajectories": int(topology_data.unique_trajectories.size),
                "n_landmarks_available": int(landmarks.size),
                "n_landmarks_mapped": int(local.size),
                "chart_dim": int(n_components),
                "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
                "chart_fit": chart.fit_diagnostics,
            },
        )

    def raw_angles(self, selected_states: np.ndarray) -> np.ndarray:
        states = _finite_2d(selected_states, "selected_states")
        if states.shape[1] != self.selected_unit_indices.size:
            raise ValueError("Selected hidden dimension differs from phase encoder")
        standardized = (
            states[:, self.retained_unit_mask] - self.unit_mean
        ) / self.unit_std
        chart_states = self.pca.transform(standardized)
        return circular_angles(self.chart.encode(chart_states))


@dataclass
class PeriodicPhaseResidual:
    feature_map: FourierFeatureMap
    alpha: float
    ridge: Ridge
    velocity_mean: np.ndarray
    velocity_std: np.ndarray

    @staticmethod
    def design(
        feature_map: FourierFeatureMap,
        theta: np.ndarray,
        velocity: np.ndarray,
        velocity_mean: np.ndarray,
        velocity_std: np.ndarray,
    ) -> np.ndarray:
        base = feature_map.transform(theta)
        scaled = (velocity - velocity_mean) / np.maximum(velocity_std, 1e-8)
        interactions = np.concatenate(
            [base * scaled[:, index : index + 1] for index in range(2)], axis=1
        )
        return np.column_stack([base, scaled, interactions])

    @classmethod
    def fit(
        cls,
        theta: np.ndarray,
        velocity: np.ndarray,
        residual_delta: np.ndarray,
        *,
        order: int,
        alpha: float,
    ) -> "PeriodicPhaseResidual":
        theta = _finite_2d(theta, "theta", 2)
        velocity = _finite_2d(velocity, "velocity", 2)
        residual_delta = _finite_2d(residual_delta, "residual_delta", 2)
        mean = velocity.mean(axis=0)
        std = velocity.std(axis=0)
        std = np.where(std > 1e-8, std, 1.0)
        feature_map = FourierFeatureMap.create(int(order))
        x = cls.design(feature_map, theta, velocity, mean, std)
        ridge = Ridge(alpha=float(alpha), fit_intercept=True, solver="lsqr")
        ridge.fit(x, residual_delta)
        return cls(feature_map, float(alpha), ridge, mean, std)

    def predict(self, theta: np.ndarray, velocity: np.ndarray) -> np.ndarray:
        theta = _finite_2d(theta, "theta", 2)
        velocity = _finite_2d(velocity, "velocity", 2)
        return np.asarray(
            self.ridge.predict(
                self.design(
                    self.feature_map,
                    theta,
                    velocity,
                    self.velocity_mean,
                    self.velocity_std,
                )
            ),
            dtype=np.float64,
        )


@dataclass
class PhaseDynamicsV3:
    basis_matrix: np.ndarray
    action_alpha: float
    action_matrix: np.ndarray
    residual_model: Optional[PeriodicPhaseResidual]
    state_only_feature_map: FourierFeatureMap
    state_only_ridge: Ridge
    diagnostics: Dict[str, Any]

    def align(self, raw_theta: np.ndarray) -> np.ndarray:
        raw_theta = _finite_2d(raw_theta, "raw_theta", 2)
        return wrap_angles(raw_theta @ self.basis_matrix.T)

    def predict_delta(
        self,
        theta: np.ndarray,
        velocity: np.ndarray,
        *,
        use_residual: bool = True,
    ) -> np.ndarray:
        theta = _finite_2d(theta, "theta", 2)
        velocity = _finite_2d(velocity, "velocity", 2)
        delta = velocity @ self.action_matrix.T
        if use_residual and self.residual_model is not None:
            delta = delta + self.residual_model.predict(theta, velocity)
        return wrap_angles(delta)

    def predict_state_only_delta(self, theta: np.ndarray) -> np.ndarray:
        theta = _finite_2d(theta, "theta", 2)
        return wrap_angles(
            np.asarray(
                self.state_only_ridge.predict(self.state_only_feature_map.transform(theta)),
                dtype=np.float64,
            )
        )

    def rollout(
        self,
        initial_theta: np.ndarray,
        velocities: np.ndarray,
        *,
        mode: str = "correct",
    ) -> np.ndarray:
        current = np.asarray(initial_theta, dtype=np.float64).reshape(1, 2)
        velocities = np.asarray(velocities, dtype=np.float64).reshape(-1, 2)
        result = [current[0].copy()]
        for velocity in velocities:
            velocity_row = velocity.reshape(1, 2)
            if mode == "zero_delta":
                delta = np.zeros((1, 2), dtype=np.float64)
            elif mode == "state_only":
                delta = self.predict_state_only_delta(current)
            else:
                delta = self.predict_delta(current, velocity_row, use_residual=(mode != "additive"))
            current = wrap_angles(current + delta)
            result.append(current[0].copy())
        return np.asarray(result, dtype=np.float64)


@dataclass
class FourierDecoderV3:
    feature_map: FourierFeatureMap
    alpha: float
    context_mode: str
    context_mean: Optional[np.ndarray]
    context_std: Optional[np.ndarray]
    ridge: Ridge

    @staticmethod
    def _design_from_parts(
        feature_map: FourierFeatureMap,
        theta: np.ndarray,
        context: Optional[np.ndarray],
        *,
        context_mode: str,
        context_mean: Optional[np.ndarray],
        context_std: Optional[np.ndarray],
    ) -> np.ndarray:
        base = feature_map.transform(theta)
        if context_mode == "none":
            return base
        if context is None:
            raise ValueError(f"Decoder mode {context_mode} requires context")
        context = _finite_2d(context, "decoder_context")
        if context.shape[0] != base.shape[0]:
            raise ValueError("Decoder context and theta row counts differ")
        mean = np.zeros(context.shape[1]) if context_mean is None else context_mean
        std = np.ones(context.shape[1]) if context_std is None else context_std
        scaled = (context - mean) / np.maximum(std, 1e-8)
        interactions = np.concatenate(
            [base * scaled[:, index : index + 1] for index in range(scaled.shape[1])],
            axis=1,
        )
        return np.column_stack([base, scaled, scaled**2, interactions])

    @classmethod
    def fit(
        cls,
        theta: np.ndarray,
        ambient: np.ndarray,
        context: Optional[np.ndarray],
        *,
        order: int,
        alpha: float,
        context_mode: str,
    ) -> "FourierDecoderV3":
        theta = _finite_2d(theta, "theta", 2)
        ambient = _finite_2d(ambient, "ambient")
        if theta.shape[0] != ambient.shape[0]:
            raise ValueError("Decoder theta and ambient row counts differ")
        context_mean = None
        context_std = None
        if context_mode != "none":
            context = _finite_2d(context, "decoder_context")
            if context.shape[0] != theta.shape[0]:
                raise ValueError("Decoder context and theta row counts differ")
            context_mean = context.mean(axis=0)
            context_std = context.std(axis=0)
            context_std = np.where(context_std > 1e-8, context_std, 1.0)
        feature_map = FourierFeatureMap.create(int(order))
        design = cls._design_from_parts(
            feature_map,
            theta,
            context,
            context_mode=context_mode,
            context_mean=context_mean,
            context_std=context_std,
        )
        ridge = Ridge(alpha=float(alpha), fit_intercept=True, solver="lsqr")
        ridge.fit(design, ambient)
        return cls(
            feature_map=feature_map,
            alpha=float(alpha),
            context_mode=context_mode,
            context_mean=context_mean,
            context_std=context_std,
            ridge=ridge,
        )

    def predict(self, theta: np.ndarray, context: Optional[np.ndarray] = None) -> np.ndarray:
        theta = _finite_2d(theta, "theta", 2)
        design = self._design_from_parts(
            self.feature_map,
            theta,
            context,
            context_mode=self.context_mode,
            context_mean=self.context_mean,
            context_std=self.context_std,
        )
        return np.asarray(self.ridge.predict(design), dtype=np.float64)

    def jacobian(self, theta: np.ndarray, context: Optional[np.ndarray] = None) -> np.ndarray:
        theta = _finite_2d(theta, "theta", 2)
        base = self.feature_map.transform(theta)
        derivative = self.feature_map.jacobian(theta)
        if self.context_mode == "none":
            design_derivative = derivative
        else:
            if context is None:
                raise ValueError("Conditioned decoder Jacobian requires context")
            context = _finite_2d(context, "decoder_context")
            scaled = (context - self.context_mean) / np.maximum(self.context_std, 1e-8)
            zeros = np.zeros(
                (theta.shape[0], 2 * scaled.shape[1], 2), dtype=np.float64
            )
            interactions = np.concatenate(
                [derivative * scaled[:, index : index + 1, None]
                 for index in range(scaled.shape[1])],
                axis=1,
            )
            design_derivative = np.concatenate([derivative, zeros, interactions], axis=1)
        return np.einsum(
            "od,ndk->nok", np.asarray(self.ridge.coef_), design_derivative
        )


@dataclass
class FibrePCASelection:
    pca: Optional[PCA]
    n_components: int
    diagnostics: Dict[str, Any]

    @property
    def basis(self) -> np.ndarray:
        if self.pca is None:
            return np.empty((0, 0), dtype=np.float64)
        return np.asarray(self.pca.components_.T, dtype=np.float64)

    def encode(self, residual: np.ndarray) -> np.ndarray:
        residual = _finite_2d(residual, "residual")
        if self.pca is None:
            return np.empty((residual.shape[0], 0), dtype=np.float64)
        return np.asarray(self.pca.transform(residual), dtype=np.float64)

    def decode(self, fibre: np.ndarray, n_rows: Optional[int] = None) -> np.ndarray:
        if self.pca is None:
            if n_rows is None:
                n_rows = np.asarray(fibre).shape[0]
            ambient_dim = int(self.diagnostics["ambient_dim"])
            return np.zeros((int(n_rows), ambient_dim), dtype=np.float64)
        fibre = np.asarray(fibre, dtype=np.float64).reshape(-1, self.n_components)
        return np.asarray(self.pca.inverse_transform(fibre), dtype=np.float64)

    def reconstruct(self, residual: np.ndarray) -> np.ndarray:
        residual = _finite_2d(residual, "residual")
        if self.pca is None:
            return np.zeros_like(residual)
        return self.decode(self.encode(residual))


@dataclass
class FibreDynamicsV3:
    kind: str
    alpha: float
    degree: int = 1
    n_clusters: int = 0
    polynomial: Optional[PolynomialFeatures] = None
    global_ridge: Optional[Ridge] = None
    clusterer: Optional[KMeans] = None
    local_ridges: Dict[int, Ridge] = field(default_factory=dict)
    active_unit_indices: Optional[np.ndarray] = None
    active_signatures: Dict[Tuple[int, ...], Ridge] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def _checked_prediction(
        self,
        value: np.ndarray,
        *,
        n_rows: int,
        stage: str,
    ) -> np.ndarray:
        output = np.asarray(value, dtype=np.float64)
        if output.ndim == 1:
            output = output.reshape(1, -1)
        if output.ndim != 2 or output.shape[0] != int(n_rows):
            raise RuntimeError(
                f"Invalid {self.kind} fibre prediction shape at {stage}: "
                f"{output.shape}; expected ({int(n_rows)}, k)"
            )
        if not np.isfinite(output).all():
            raise NumericalRolloutError(
                f"{self.kind} fibre prediction became non-finite at {stage}"
            )
        return output

    @staticmethod
    def base_features(xi: np.ndarray, theta: np.ndarray, velocity: np.ndarray) -> np.ndarray:
        xi = np.asarray(xi, dtype=np.float64)
        if xi.ndim != 2:
            raise ValueError("xi must be 2-D")
        theta = _finite_2d(theta, "theta", 2)
        velocity = _finite_2d(velocity, "velocity", 2)
        if not (xi.shape[0] == theta.shape[0] == velocity.shape[0]):
            raise ValueError("Fibre feature row counts differ")
        return np.column_stack([xi, circular_features(theta), velocity])

    @staticmethod
    def _active_signature(states: np.ndarray, active_units: np.ndarray) -> List[Tuple[int, ...]]:
        states = _finite_2d(states, "active_states")
        bits = states[:, active_units] > 0.0
        return [tuple(int(value) for value in row.tolist()) for row in bits]

    @classmethod
    def fit(
        cls,
        xi: np.ndarray,
        theta: np.ndarray,
        data: TrajectoryData,
        *,
        kind: str,
        alpha: float,
        degree: int = 2,
        n_clusters: int = 8,
        raw_states: Optional[np.ndarray] = None,
        active_max_units: int = 16,
        active_max_regimes: int = 64,
        active_min_samples: int = 32,
        max_polynomial_features: int = 2048,
        seed: int = 42,
    ) -> "FibreDynamicsV3":
        xi = np.asarray(xi, dtype=np.float64)
        if xi.ndim != 2 or xi.shape[0] != data.n_states:
            raise ValueError("xi must have one row per state")
        if xi.shape[1] < 1:
            raise ValueError("Fibre dynamics require at least one fibre component")
        theta = _finite_2d(theta, "theta", 2)
        if theta.shape[0] != data.n_states:
            raise ValueError("theta must have one row per state")
        prev = data.transition_prev
        nxt = data.transition_next
        x_base = cls.base_features(xi[prev], theta[prev], data.transition_velocities)
        y = xi[nxt]
        global_ridge = Ridge(alpha=float(alpha), fit_intercept=True, solver="lsqr")
        global_ridge.fit(x_base, y)
        model = cls(kind=kind, alpha=float(alpha), global_ridge=global_ridge)

        if kind == "linear_ar":
            pass
        elif kind == "ridge_polynomial":
            output_features = math.comb(
                int(x_base.shape[1]) + int(degree), int(degree)
            ) - 1
            if output_features > int(max_polynomial_features):
                raise ValueError(
                    "Polynomial fibre candidate would create "
                    f"{output_features} dense features, exceeding "
                    f"max_polynomial_features={int(max_polynomial_features)}"
                )
            polynomial = PolynomialFeatures(degree=int(degree), include_bias=False)
            x_poly = polynomial.fit_transform(x_base)
            ridge = Ridge(alpha=float(alpha), fit_intercept=True, solver="lsqr")
            ridge.fit(x_poly, y)
            model.degree = int(degree)
            model.polynomial = polynomial
            model.global_ridge = ridge
        elif kind == "local_piecewise_affine":
            cluster_count = max(2, min(int(n_clusters), max(2, prev.size // 20)))
            cluster_input = np.column_stack([circular_features(theta[prev]), xi[prev]])
            clusterer = KMeans(
                n_clusters=cluster_count,
                random_state=int(seed),
                n_init=10,
            )
            labels = clusterer.fit_predict(cluster_input)
            local: Dict[int, Ridge] = {}
            for label in range(cluster_count):
                rows = np.flatnonzero(labels == label)
                if rows.size < max(8, xi.shape[1] + 2):
                    continue
                ridge = Ridge(alpha=float(alpha), fit_intercept=True, solver="lsqr")
                ridge.fit(x_base[rows], y[rows])
                local[int(label)] = ridge
            model.n_clusters = cluster_count
            model.clusterer = clusterer
            model.local_ridges = local
        elif kind == "active_set_conditioned":
            if raw_states is None:
                raise ValueError("active_set_conditioned requires raw selected states")
            raw_states = _finite_2d(raw_states, "raw_states")
            if raw_states.shape[0] != data.n_states:
                raise ValueError("raw_states must have one row per state")
            activity_rate = np.mean(raw_states[prev] > 0.0, axis=0)
            variability = activity_rate * (1.0 - activity_rate)
            n_active = min(int(active_max_units), raw_states.shape[1])
            active_units = np.argsort(variability)[-n_active:]
            signatures = cls._active_signature(raw_states[prev], active_units)
            counts: Dict[Tuple[int, ...], List[int]] = {}
            for row, signature in enumerate(signatures):
                counts.setdefault(signature, []).append(row)
            ranked = sorted(counts.items(), key=lambda item: (-len(item[1]), item[0]))
            local_signatures: Dict[Tuple[int, ...], Ridge] = {}
            for signature, rows_list in ranked[: int(active_max_regimes)]:
                rows = np.asarray(rows_list, dtype=np.int64)
                if rows.size < int(active_min_samples):
                    continue
                ridge = Ridge(alpha=float(alpha), fit_intercept=True, solver="lsqr")
                ridge.fit(x_base[rows], y[rows])
                local_signatures[signature] = ridge
            model.active_unit_indices = np.asarray(active_units, dtype=np.int64)
            model.active_signatures = local_signatures
        else:
            raise ValueError(f"Unknown fibre dynamics kind: {kind}")

        model.diagnostics = {
            "kind": kind,
            "alpha": float(alpha),
            "degree": int(model.degree),
            "n_clusters": int(model.n_clusters),
            "n_local_models": int(len(model.local_ridges)),
            "n_active_units": (
                0 if model.active_unit_indices is None else int(model.active_unit_indices.size)
            ),
            "n_active_regimes": int(len(model.active_signatures)),
            "n_transitions": int(prev.size),
            "max_polynomial_features": int(max_polynomial_features),
        }
        return model

    def predict(
        self,
        xi: np.ndarray,
        theta: np.ndarray,
        velocity: np.ndarray,
        *,
        raw_states: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        x_base = self.base_features(xi, theta, velocity)
        if self.global_ridge is None:
            raise RuntimeError("Fibre dynamics model is not fitted")
        if self.kind == "linear_ar":
            try:
                with np.errstate(over="raise", invalid="raise", divide="raise"):
                    prediction = self.global_ridge.predict(x_base)
            except (FloatingPointError, OverflowError) as exc:
                raise NumericalRolloutError(
                    "linear_ar fibre prediction overflowed"
                ) from exc
            return self._checked_prediction(
                prediction, n_rows=x_base.shape[0], stage="linear prediction"
            )
        if self.kind == "ridge_polynomial":
            if self.polynomial is None:
                raise RuntimeError("Polynomial fibre model is incomplete")
            if (
                hasattr(self.polynomial, "n_features_in_")
                and int(self.polynomial.n_features_in_) != int(x_base.shape[1])
            ):
                raise RuntimeError(
                    "Polynomial fibre feature dimension differs from fitted model"
                )
            try:
                with np.errstate(over="raise", invalid="raise", divide="raise"):
                    transformed = self.polynomial.transform(x_base)
            except (FloatingPointError, OverflowError) as exc:
                raise NumericalRolloutError(
                    "ridge_polynomial feature expansion overflowed during rollout"
                ) from exc
            if not np.isfinite(transformed).all():
                raise NumericalRolloutError(
                    "ridge_polynomial feature expansion became non-finite during rollout"
                )
            try:
                with np.errstate(over="raise", invalid="raise", divide="raise"):
                    prediction = self.global_ridge.predict(transformed)
            except (FloatingPointError, OverflowError, ValueError) as exc:
                reason = _numerical_instability_reason(exc)
                if reason is None:
                    raise
                raise NumericalRolloutError(
                    f"ridge_polynomial prediction failed numerically: {reason}"
                ) from exc
            return self._checked_prediction(
                prediction, n_rows=x_base.shape[0], stage="polynomial prediction"
            )
        if self.kind == "local_piecewise_affine":
            if self.clusterer is None:
                raise RuntimeError("Local fibre model is incomplete")
            cluster_input = np.column_stack([circular_features(theta), xi])
            try:
                with np.errstate(over="raise", invalid="raise", divide="raise"):
                    labels = self.clusterer.predict(cluster_input)
                    output = self.global_ridge.predict(x_base)
            except (FloatingPointError, OverflowError, ValueError) as exc:
                reason = _numerical_instability_reason(exc)
                if reason is None:
                    raise
                raise NumericalRolloutError(
                    f"local_piecewise_affine prediction failed numerically: {reason}"
                ) from exc
            output = self._checked_prediction(
                output, n_rows=x_base.shape[0], stage="global fallback prediction"
            )
            for label, ridge in self.local_ridges.items():
                rows = np.flatnonzero(labels == label)
                if rows.size:
                    try:
                        with np.errstate(
                            over="raise", invalid="raise", divide="raise"
                        ):
                            local_prediction = ridge.predict(x_base[rows])
                    except (FloatingPointError, OverflowError, ValueError) as exc:
                        reason = _numerical_instability_reason(exc)
                        if reason is None:
                            raise
                        raise NumericalRolloutError(
                            "local_piecewise_affine local prediction failed "
                            f"numerically: {reason}"
                        ) from exc
                    output[rows] = self._checked_prediction(
                        local_prediction,
                        n_rows=rows.size,
                        stage=f"local regime {int(label)} prediction",
                    )
            return self._checked_prediction(
                output, n_rows=x_base.shape[0], stage="piecewise prediction"
            )
        if self.kind == "active_set_conditioned":
            if raw_states is None or self.active_unit_indices is None:
                try:
                    with np.errstate(over="raise", invalid="raise", divide="raise"):
                        prediction = self.global_ridge.predict(x_base)
                except (FloatingPointError, OverflowError, ValueError) as exc:
                    reason = _numerical_instability_reason(exc)
                    if reason is None:
                        raise
                    raise NumericalRolloutError(
                        f"active_set fallback prediction failed numerically: {reason}"
                    ) from exc
                return self._checked_prediction(
                    prediction,
                    n_rows=x_base.shape[0],
                    stage="active-set fallback prediction",
                )
            signatures = self._active_signature(raw_states, self.active_unit_indices)
            try:
                with np.errstate(over="raise", invalid="raise", divide="raise"):
                    output = self.global_ridge.predict(x_base)
            except (FloatingPointError, OverflowError, ValueError) as exc:
                reason = _numerical_instability_reason(exc)
                if reason is None:
                    raise
                raise NumericalRolloutError(
                    f"active_set global prediction failed numerically: {reason}"
                ) from exc
            output = self._checked_prediction(
                output,
                n_rows=x_base.shape[0],
                stage="active-set global prediction",
            )
            for row, signature in enumerate(signatures):
                ridge = self.active_signatures.get(signature)
                if ridge is not None:
                    try:
                        with np.errstate(
                            over="raise", invalid="raise", divide="raise"
                        ):
                            local_prediction = ridge.predict(x_base[row : row + 1])
                    except (FloatingPointError, OverflowError, ValueError) as exc:
                        reason = _numerical_instability_reason(exc)
                        if reason is None:
                            raise
                        raise NumericalRolloutError(
                            "active_set local prediction failed numerically: "
                            f"{reason}"
                        ) from exc
                    output[row] = self._checked_prediction(
                        local_prediction,
                        n_rows=1,
                        stage="active-set local prediction",
                    )[0]
            return self._checked_prediction(
                output, n_rows=x_base.shape[0], stage="active-set prediction"
            )
        raise ValueError(f"Unknown fibre dynamics kind: {self.kind}")


def _phase_transition_arrays(
    theta: np.ndarray,
    data: TrajectoryData,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    theta = _finite_2d(theta, "theta", 2)
    if theta.shape[0] != data.n_states:
        raise ValueError("theta must have one row per state")
    prev = data.transition_prev
    nxt = data.transition_next
    delta = wrap_angles(theta[nxt] - theta[prev])
    velocity = data.transition_velocities
    groups = data.trajectory_ids[prev]
    return theta[prev], delta, velocity, groups


def _phase_error_by_trajectory(
    actual_delta: np.ndarray,
    predicted_delta: np.ndarray,
    groups: np.ndarray,
) -> List[float]:
    actual_delta = _finite_2d(actual_delta, "actual_delta", 2)
    predicted_delta = _finite_2d(predicted_delta, "predicted_delta", 2)
    groups = _finite_1d(groups, "groups", actual_delta.shape[0])
    values: List[float] = []
    for group in np.unique(groups):
        rows = np.flatnonzero(groups == group)
        if rows.size:
            values.append(_wrapped_rmse(actual_delta[rows], predicted_delta[rows]))
    return values


def fit_phase_dynamics_v3(
    raw_theta_fit: np.ndarray,
    source_fit: TrajectoryData,
    raw_theta_validation: np.ndarray,
    source_validation: TrajectoryData,
    *,
    max_gl2z_entry: int = 2,
    basis_search_alpha: float = 1e-6,
    action_alphas: Sequence[float] = (1e-8, 1e-6, 1e-4, 1e-2, 1.0),
    residual_orders: Sequence[int] = (1, 2),
    residual_alphas: Sequence[float] = (1e-6, 1e-4, 1e-2, 1.0),
    residual_min_relative_improvement: float = 0.01,
    state_only_order: int = 2,
    state_only_alpha: float = 1e-3,
) -> Tuple[PhaseDynamicsV3, PhaseDynamicsV3, Dict[str, Any]]:
    """Select phase dynamics on validation and return fit-only/final models."""
    raw_theta_fit = _finite_2d(raw_theta_fit, "raw_theta_fit", 2)
    raw_theta_validation = _finite_2d(raw_theta_validation, "raw_theta_validation", 2)
    if raw_theta_fit.shape[0] != source_fit.n_states:
        raise ValueError("raw_theta_fit state count differs")
    if raw_theta_validation.shape[0] != source_validation.n_states:
        raise ValueError("raw_theta_validation state count differs")

    basis_rows: List[Dict[str, Any]] = []
    matrices = generate_unimodular_matrices(int(max_gl2z_entry))
    for matrix in matrices:
        theta_fit = wrap_angles(raw_theta_fit @ matrix.T)
        theta_val = wrap_angles(raw_theta_validation @ matrix.T)
        _fit_state, fit_delta, fit_velocity, _fit_groups = _phase_transition_arrays(
            theta_fit, source_fit
        )
        _val_state, val_delta, val_velocity, val_groups = _phase_transition_arrays(
            theta_val, source_validation
        )
        ridge = Ridge(
            alpha=float(basis_search_alpha), fit_intercept=False, solver="lsqr"
        )
        ridge.fit(fit_velocity, fit_delta)
        prediction = val_velocity @ np.asarray(ridge.coef_).T
        per_trajectory = _phase_error_by_trajectory(val_delta, prediction, val_groups)
        mean, se = _mean_se(per_trajectory)
        basis_rows.append(
            {
                "basis_matrix": np.asarray(matrix, dtype=int).tolist(),
                "validation_phase_rmse": mean,
                "validation_phase_rmse_se": se,
                "complexity": int(np.sum(np.abs(matrix))),
                "identity": bool(np.array_equal(matrix, np.eye(2, dtype=int))),
                "n_validation_trajectories": len(per_trajectory),
            }
        )
    basis_choice = _one_standard_error_choice(
        basis_rows,
        metric="validation_phase_rmse",
        se_metric="validation_phase_rmse_se",
        complexity_key=lambda row: (
            int(row["complexity"]),
            0 if bool(row["identity"]) else 1,
            tuple(np.asarray(row["basis_matrix"]).reshape(-1).tolist()),
        ),
    )
    basis_matrix = np.asarray(basis_choice["basis_matrix"], dtype=np.int64)
    theta_fit = wrap_angles(raw_theta_fit @ basis_matrix.T)
    theta_val = wrap_angles(raw_theta_validation @ basis_matrix.T)
    fit_state, fit_delta, fit_velocity, _fit_groups = _phase_transition_arrays(
        theta_fit, source_fit
    )
    val_state, val_delta, val_velocity, val_groups = _phase_transition_arrays(
        theta_val, source_validation
    )

    action_rows: List[Dict[str, Any]] = []
    action_models: Dict[float, Ridge] = {}
    for alpha in action_alphas:
        ridge = Ridge(alpha=float(alpha), fit_intercept=False, solver="lsqr")
        ridge.fit(fit_velocity, fit_delta)
        prediction = val_velocity @ np.asarray(ridge.coef_).T
        per_trajectory = _phase_error_by_trajectory(val_delta, prediction, val_groups)
        mean, se = _mean_se(per_trajectory)
        action_rows.append(
            {
                "alpha": float(alpha),
                "validation_phase_rmse": mean,
                "validation_phase_rmse_se": se,
                "n_validation_trajectories": len(per_trajectory),
            }
        )
        action_models[float(alpha)] = ridge
    action_choice = _one_standard_error_choice(
        action_rows,
        metric="validation_phase_rmse",
        se_metric="validation_phase_rmse_se",
        # Larger alpha is the simpler/more strongly regularized eligible model.
        complexity_key=lambda row: (-float(row["alpha"]),),
    )
    action_alpha = float(action_choice["alpha"])
    action_cv = action_models[action_alpha]
    action_matrix_cv = np.asarray(action_cv.coef_, dtype=np.float64)
    additive_fit_residual = wrap_angles(fit_delta - fit_velocity @ action_matrix_cv.T)
    additive_val_prediction = val_velocity @ action_matrix_cv.T
    additive_per_trajectory = _phase_error_by_trajectory(
        val_delta, additive_val_prediction, val_groups
    )
    additive_mean, additive_se = _mean_se(additive_per_trajectory)

    residual_rows: List[Dict[str, Any]] = [
        {
            "kind": "none",
            "order": 0,
            "alpha": None,
            "validation_phase_rmse": additive_mean,
            "validation_phase_rmse_se": additive_se,
            "n_validation_trajectories": len(additive_per_trajectory),
        }
    ]
    residual_models: Dict[Tuple[int, float], PeriodicPhaseResidual] = {}
    for order in residual_orders:
        for alpha in residual_alphas:
            model = PeriodicPhaseResidual.fit(
                fit_state,
                fit_velocity,
                additive_fit_residual,
                order=int(order),
                alpha=float(alpha),
            )
            prediction = additive_val_prediction + model.predict(val_state, val_velocity)
            per_trajectory = _phase_error_by_trajectory(val_delta, prediction, val_groups)
            mean, se = _mean_se(per_trajectory)
            residual_rows.append(
                {
                    "kind": "periodic_residual",
                    "order": int(order),
                    "alpha": float(alpha),
                    "validation_phase_rmse": mean,
                    "validation_phase_rmse_se": se,
                    "n_validation_trajectories": len(per_trajectory),
                }
            )
            residual_models[(int(order), float(alpha))] = model
    residual_choice = _one_standard_error_choice(
        residual_rows,
        metric="validation_phase_rmse",
        se_metric="validation_phase_rmse_se",
        complexity_key=lambda row: (
            0 if row["kind"] == "none" else 1,
            int(row["order"]),
            -float(row["alpha"] or 0.0),
        ),
    )
    best_residual = min(
        residual_rows, key=lambda row: float(row["validation_phase_rmse"])
    )
    relative_improvement = (
        additive_mean - float(best_residual["validation_phase_rmse"])
    ) / max(additive_mean, _EPS)
    if relative_improvement < float(residual_min_relative_improvement):
        residual_choice = dict(residual_rows[0])
        residual_choice["selection_override"] = (
            "best nonlinear residual did not meet the configured relative-improvement floor"
        )
    residual_cv = None
    if residual_choice["kind"] != "none":
        residual_cv = residual_models[
            (int(residual_choice["order"]), float(residual_choice["alpha"]))
        ]

    state_fmap = FourierFeatureMap.create(int(state_only_order))
    state_ridge_cv = Ridge(
        alpha=float(state_only_alpha), fit_intercept=True, solver="lsqr"
    )
    state_ridge_cv.fit(state_fmap.transform(fit_state), fit_delta)

    combined = combine_trajectory_data(source_fit, source_validation)
    raw_combined = np.concatenate([raw_theta_fit, raw_theta_validation], axis=0)
    theta_combined = wrap_angles(raw_combined @ basis_matrix.T)
    combined_state, combined_delta, combined_velocity, _combined_groups = (
        _phase_transition_arrays(theta_combined, combined)
    )
    action_final = Ridge(
        alpha=action_alpha, fit_intercept=False, solver="lsqr"
    )
    action_final.fit(combined_velocity, combined_delta)
    action_matrix_final = np.asarray(action_final.coef_, dtype=np.float64)
    residual_final = None
    if residual_choice["kind"] != "none":
        residual_target = wrap_angles(
            combined_delta - combined_velocity @ action_matrix_final.T
        )
        residual_final = PeriodicPhaseResidual.fit(
            combined_state,
            combined_velocity,
            residual_target,
            order=int(residual_choice["order"]),
            alpha=float(residual_choice["alpha"]),
        )
    state_ridge_final = Ridge(
        alpha=float(state_only_alpha), fit_intercept=True, solver="lsqr"
    )
    state_ridge_final.fit(state_fmap.transform(combined_state), combined_delta)

    diagnostics = {
        "selection_protocol": (
            "fit candidates on source-fit; select on source-validation; "
            "refit selected hyperparameters on fit+validation"
        ),
        "basis_candidates": basis_rows,
        "selected_basis": dict(basis_choice),
        "action_candidates": action_rows,
        "selected_action": dict(action_choice),
        "residual_candidates": residual_rows,
        "selected_residual": dict(residual_choice),
        "best_residual_relative_improvement": float(relative_improvement),
        "residual_min_relative_improvement": float(
            residual_min_relative_improvement
        ),
        "state_only_order": int(state_only_order),
        "state_only_alpha": float(state_only_alpha),
    }
    cv_model = PhaseDynamicsV3(
        basis_matrix=basis_matrix,
        action_alpha=action_alpha,
        action_matrix=action_matrix_cv,
        residual_model=residual_cv,
        state_only_feature_map=state_fmap,
        state_only_ridge=state_ridge_cv,
        diagnostics={**diagnostics, "fit_scope": "source_fit_only"},
    )
    final_model = PhaseDynamicsV3(
        basis_matrix=basis_matrix,
        action_alpha=action_alpha,
        action_matrix=action_matrix_final,
        residual_model=residual_final,
        state_only_feature_map=state_fmap,
        state_only_ridge=state_ridge_final,
        diagnostics={**diagnostics, "fit_scope": "source_fit_plus_validation"},
    )
    return cv_model, final_model, diagnostics


def _trajectory_surface_rows(
    actual: np.ndarray,
    predicted: np.ndarray,
    data: TrajectoryData,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for trajectory_id in data.unique_trajectories:
        state_rows = data.trajectory_state_indices(int(trajectory_id))
        if state_rows.size:
            rows.append(
                {
                    "trajectory_id": int(trajectory_id),
                    "surface_mfe": _relative_error(
                        actual[state_rows], predicted[state_rows]
                    ),
                }
            )
    return rows


def _base_rollout_rows(
    decoder: FourierDecoderV3,
    phase_model: PhaseDynamicsV3,
    theta_actual: np.ndarray,
    ambient_actual: np.ndarray,
    data: TrajectoryData,
    *,
    min_rollout_steps: int = 2,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for segment in contiguous_segments(data, min_transitions=min_rollout_steps):
        velocities = data.transition_velocities[segment.transition_rows]
        predicted_theta = phase_model.rollout(theta_actual[segment.state_indices[0]], velocities)
        context = decoder_context_for_segment(velocities, decoder.context_mode)
        predicted = decoder.predict(predicted_theta, context)
        actual = ambient_actual[segment.state_indices]
        rows.append(
            {
                "trajectory_id": int(segment.trajectory_id),
                "segment_id": int(segment.segment_id),
                "rollout_nmte": _relative_error(actual, predicted),
                "n_transitions": int(segment.transition_rows.size),
            }
        )
    return rows


def _aggregate_segment_metric_by_trajectory(
    segment_rows: Sequence[Mapping[str, Any]],
    metric: str,
) -> List[float]:
    grouped: Dict[int, List[float]] = {}
    for row in segment_rows:
        grouped.setdefault(int(row["trajectory_id"]), []).append(float(row[metric]))
    return [float(np.mean(grouped[key])) for key in sorted(grouped)]


def select_fourier_decoders_v3(
    theta_fit: np.ndarray,
    ambient_fit: np.ndarray,
    source_fit: TrajectoryData,
    theta_validation: np.ndarray,
    ambient_validation: np.ndarray,
    source_validation: TrajectoryData,
    phase_cv: PhaseDynamicsV3,
    *,
    orders: Sequence[int] = (1, 2, 3, 4, 5, 6),
    alphas: Sequence[float] = (1e-6, 1e-4, 1e-2, 1.0, 100.0),
    context_modes: Sequence[str] = (
        "none",
        "previous_velocity",
        "current_velocity",
        "previous_current_velocity",
    ),
    conditioned_min_relative_rollout_improvement: float = 0.0,
    min_rollout_steps: int = 2,
) -> Tuple[FourierDecoderV3, FourierDecoderV3, FourierDecoderV3, FourierDecoderV3, Dict[str, Any]]:
    """Select S0 and S1 decoders without exposing source-test."""
    theta_fit = _finite_2d(theta_fit, "theta_fit", 2)
    theta_validation = _finite_2d(theta_validation, "theta_validation", 2)
    ambient_fit = _finite_2d(ambient_fit, "ambient_fit")
    ambient_validation = _finite_2d(ambient_validation, "ambient_validation")
    if theta_fit.shape[0] != source_fit.n_states or ambient_fit.shape[0] != source_fit.n_states:
        raise ValueError("Fit decoder arrays must have one row per fit state")
    if (
        theta_validation.shape[0] != source_validation.n_states
        or ambient_validation.shape[0] != source_validation.n_states
    ):
        raise ValueError("Validation decoder arrays must have one row per validation state")

    candidate_rows: List[Dict[str, Any]] = []
    candidate_models: Dict[Tuple[str, int, float], FourierDecoderV3] = {}
    for mode in context_modes:
        fit_context = decoder_context_per_state(source_fit, mode)
        val_context = decoder_context_per_state(source_validation, mode)
        for order in orders:
            for alpha in alphas:
                model = FourierDecoderV3.fit(
                    theta_fit,
                    ambient_fit,
                    fit_context,
                    order=int(order),
                    alpha=float(alpha),
                    context_mode=str(mode),
                )
                prediction = model.predict(theta_validation, val_context)
                surface_rows = _trajectory_surface_rows(
                    ambient_validation, prediction, source_validation
                )
                surface_values = [float(row["surface_mfe"]) for row in surface_rows]
                surface_mean, surface_se = _mean_se(surface_values)
                rollout_rows = _base_rollout_rows(
                    model,
                    phase_cv,
                    theta_validation,
                    ambient_validation,
                    source_validation,
                    min_rollout_steps=min_rollout_steps,
                )
                rollout_values = _aggregate_segment_metric_by_trajectory(
                    rollout_rows, "rollout_nmte"
                )
                rollout_mean, rollout_se = _mean_se(rollout_values)
                key = (str(mode), int(order), float(alpha))
                candidate_models[key] = model
                candidate_rows.append(
                    {
                        "context_mode": str(mode),
                        "order": int(order),
                        "alpha": float(alpha),
                        "validation_surface_mfe": surface_mean,
                        "validation_surface_mfe_se": surface_se,
                        "validation_rollout_nmte": rollout_mean,
                        "validation_rollout_nmte_se": rollout_se,
                        "n_validation_trajectories": len(rollout_values),
                    }
                )

    context_complexity = {
        "none": 0,
        "previous_velocity": 1,
        "current_velocity": 1,
        "previous_current_velocity": 2,
    }
    complexity = lambda row: (
        context_complexity[str(row["context_mode"])],
        int(row["order"]),
        -float(row["alpha"]),
    )
    base_rows = [row for row in candidate_rows if row["context_mode"] == "none"]
    base_choice = _one_standard_error_choice(
        base_rows,
        metric="validation_rollout_nmte",
        se_metric="validation_rollout_nmte_se",
        complexity_key=complexity,
    )
    selected_choice = _one_standard_error_choice(
        candidate_rows,
        metric="validation_rollout_nmte",
        se_metric="validation_rollout_nmte_se",
        complexity_key=complexity,
    )
    best_base_rollout = float(base_choice["validation_rollout_nmte"])
    selected_rollout = float(selected_choice["validation_rollout_nmte"])
    conditioned_improvement = (
        best_base_rollout - selected_rollout
    ) / max(best_base_rollout, _EPS)
    if (
        selected_choice["context_mode"] != "none"
        and conditioned_improvement
        < float(conditioned_min_relative_rollout_improvement)
    ):
        selected_choice = dict(base_choice)
        selected_choice["selection_override"] = (
            "conditioned decoder did not meet validation rollout improvement floor"
        )

    def _model_for(choice: Mapping[str, Any]) -> FourierDecoderV3:
        return candidate_models[
            (
                str(choice["context_mode"]),
                int(choice["order"]),
                float(choice["alpha"]),
            )
        ]

    base_cv = _model_for(base_choice)
    selected_cv = _model_for(selected_choice)
    combined = combine_trajectory_data(source_fit, source_validation)
    theta_combined = np.concatenate([theta_fit, theta_validation], axis=0)
    ambient_combined = np.concatenate([ambient_fit, ambient_validation], axis=0)
    base_final = FourierDecoderV3.fit(
        theta_combined,
        ambient_combined,
        decoder_context_per_state(combined, "none"),
        order=int(base_choice["order"]),
        alpha=float(base_choice["alpha"]),
        context_mode="none",
    )
    selected_final = FourierDecoderV3.fit(
        theta_combined,
        ambient_combined,
        decoder_context_per_state(combined, str(selected_choice["context_mode"])),
        order=int(selected_choice["order"]),
        alpha=float(selected_choice["alpha"]),
        context_mode=str(selected_choice["context_mode"]),
    )
    diagnostics = {
        "selection_protocol": (
            "candidate fit on source-fit; validation rollout is primary; "
            "one-standard-error simplicity rule"
        ),
        "candidate_rows": candidate_rows,
        "selected_s0_base": dict(base_choice),
        "selected_s1": dict(selected_choice),
        "conditioned_relative_rollout_improvement": float(conditioned_improvement),
        "conditioned_min_relative_rollout_improvement": float(
            conditioned_min_relative_rollout_improvement
        ),
    }
    return base_cv, selected_cv, base_final, selected_final, diagnostics


def select_fibre_pca_v3(
    residual_fit: np.ndarray,
    source_fit: TrajectoryData,
    residual_validation: np.ndarray,
    source_validation: TrajectoryData,
    *,
    candidate_dims: Sequence[int] = (0, 1, 2, 4, 6, 8, 12, 16, 24, 32),
    plateau_relative_gain: float = 0.02,
    seed: int = 42,
) -> Tuple[FibrePCASelection, int, Dict[str, Any]]:
    """Select the smallest sufficient fibre dimension by a trajectory-level 1-SE rule."""
    residual_fit = _finite_2d(residual_fit, "residual_fit")
    residual_validation = _finite_2d(residual_validation, "residual_validation")
    if residual_fit.shape[1] != residual_validation.shape[1]:
        raise ValueError("Fit and validation residual dimensions differ")
    max_rank = min(
        residual_fit.shape[1],
        max(0, residual_fit.shape[0] - 1),
    )
    dims = sorted({max(0, min(int(value), max_rank)) for value in candidate_dims})
    if not dims:
        dims = [0]
    rows: List[Dict[str, Any]] = []
    models: Dict[int, Optional[PCA]] = {0: None}
    for dimension in dims:
        if dimension == 0:
            prediction = np.zeros_like(residual_validation)
        else:
            pca = PCA(n_components=int(dimension), random_state=int(seed))
            pca.fit(residual_fit)
            prediction = pca.inverse_transform(pca.transform(residual_validation))
            models[int(dimension)] = pca
        trajectory_values: List[float] = []
        for trajectory_id in source_validation.unique_trajectories:
            state_rows = source_validation.trajectory_state_indices(int(trajectory_id))
            if state_rows.size:
                trajectory_values.append(
                    _relative_error(
                        residual_validation[state_rows], prediction[state_rows]
                    )
                )
        mean, se = _mean_se(trajectory_values)
        rows.append(
            {
                "n_components": int(dimension),
                "validation_residual_mfe": mean,
                "validation_residual_mfe_se": se,
                "n_validation_trajectories": len(trajectory_values),
            }
        )
    choice = _one_standard_error_choice(
        rows,
        metric="validation_residual_mfe",
        se_metric="validation_residual_mfe_se",
        complexity_key=lambda row: (int(row["n_components"]),),
    )
    selected = int(choice["n_components"])
    best_metric = min(float(row["validation_residual_mfe"]) for row in rows)
    numerical_tolerance = max(_EPS, abs(best_metric) * 1e-8)
    best_dimension = min(
        int(row["n_components"])
        for row in rows
        if float(row["validation_residual_mfe"])
        <= best_metric + numerical_tolerance
    )
    cap = int(max(dims))
    selected_metric = next(
        float(row["validation_residual_mfe"])
        for row in rows
        if int(row["n_components"]) == selected
    )
    cap_metric = next(
        float(row["validation_residual_mfe"])
        for row in rows
        if int(row["n_components"]) == cap
    )
    tail_relative_gain = (
        selected_metric - cap_metric
    ) / max(selected_metric, _EPS)
    plateau_detected = bool(
        selected < cap
        and best_dimension < cap
        and tail_relative_gain <= float(plateau_relative_gain)
    )
    model = FibrePCASelection(
        pca=models.get(selected),
        n_components=selected,
        diagnostics={
            "ambient_dim": int(residual_fit.shape[1]),
            "candidate_rows": rows,
            "selected": dict(choice),
            "best_dimension": best_dimension,
            "candidate_cap": cap,
            "selected_at_cap": selected == cap,
            "best_at_cap": best_dimension == cap,
            "tail_relative_gain": float(tail_relative_gain),
            "plateau_relative_gain_threshold": float(plateau_relative_gain),
            "clear_plateau_detected": plateau_detected,
            "low_dimensional_fibre_supported": plateau_detected,
            "selection_protocol": (
                "PCA fit on source-fit residuals; trajectory-level validation "
                "one-standard-error rule"
            ),
        },
    )
    return model, selected, model.diagnostics


def fit_final_fibre_pca(
    residual: np.ndarray,
    *,
    n_components: int,
    seed: int,
    selection_diagnostics: Mapping[str, Any],
) -> FibrePCASelection:
    residual = _finite_2d(residual, "combined_residual")
    dimension = int(n_components)
    pca = None
    if dimension > 0:
        if dimension > min(residual.shape[1], residual.shape[0] - 1):
            raise ValueError("Selected fibre dimension exceeds final residual rank")
        pca = PCA(n_components=dimension, random_state=int(seed))
        pca.fit(residual)
    return FibrePCASelection(
        pca=pca,
        n_components=dimension,
        diagnostics={
            **dict(selection_diagnostics),
            "ambient_dim": int(residual.shape[1]),
            "final_fit_scope": "source_fit_plus_validation",
        },
    )


def _rollout_full_segment(
    *,
    decoder: FourierDecoderV3,
    phase_model: PhaseDynamicsV3,
    fibre_model: FibrePCASelection,
    fibre_dynamics: Optional[FibreDynamicsV3],
    standardizer: AmbientStandardizer,
    initial_theta: np.ndarray,
    initial_xi: np.ndarray,
    velocities: np.ndarray,
    phase_mode: str = "correct",
    fibre_mode: str = "dynamic",
    phase_velocities: Optional[np.ndarray] = None,
    fibre_velocities: Optional[np.ndarray] = None,
    context_velocities: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    velocities = np.asarray(velocities, dtype=np.float64).reshape(-1, 2)
    phase_velocities = (
        velocities
        if phase_velocities is None
        else np.asarray(phase_velocities, dtype=np.float64).reshape(-1, 2)
    )
    fibre_velocities = (
        velocities
        if fibre_velocities is None
        else np.asarray(fibre_velocities, dtype=np.float64).reshape(-1, 2)
    )
    context_velocities = (
        velocities
        if context_velocities is None
        else np.asarray(context_velocities, dtype=np.float64).reshape(-1, 2)
    )
    if not (
        phase_velocities.shape == velocities.shape
        and fibre_velocities.shape == velocities.shape
        and context_velocities.shape == velocities.shape
    ):
        raise ValueError("Counterfactual velocity sequence shapes differ")
    contexts = decoder_context_for_segment(context_velocities, decoder.context_mode)
    theta = np.asarray(initial_theta, dtype=np.float64).reshape(1, 2)
    xi = np.asarray(initial_xi, dtype=np.float64).reshape(1, fibre_model.n_components)
    theta_rows: List[np.ndarray] = []
    xi_rows: List[np.ndarray] = []
    ambient_rows: List[np.ndarray] = []
    for state_index in range(velocities.shape[0] + 1):
        context_row = None if contexts is None else contexts[state_index : state_index + 1]
        base = decoder.predict(theta, context_row)
        residual = fibre_model.decode(xi, n_rows=1)
        ambient = base + residual
        theta_rows.append(theta[0].copy())
        xi_rows.append(xi[0].copy())
        ambient_rows.append(ambient[0].copy())
        if state_index == velocities.shape[0]:
            break
        phase_velocity = phase_velocities[state_index : state_index + 1]
        fibre_velocity = fibre_velocities[state_index : state_index + 1]
        if phase_mode == "zero_delta":
            phase_delta = np.zeros((1, 2), dtype=np.float64)
        elif phase_mode == "state_only":
            phase_delta = phase_model.predict_state_only_delta(theta)
        elif phase_mode == "additive":
            phase_delta = phase_model.predict_delta(
                theta, phase_velocity, use_residual=False
            )
        elif phase_mode == "correct":
            phase_delta = phase_model.predict_delta(
                theta, phase_velocity, use_residual=True
            )
        else:
            raise ValueError(f"Unknown phase rollout mode: {phase_mode}")
        if fibre_model.n_components > 0:
            if fibre_mode == "hold_initial":
                pass
            elif fibre_mode == "zero":
                xi = np.zeros_like(xi)
            elif fibre_mode == "dynamic":
                if fibre_dynamics is None:
                    raise RuntimeError(
                        "A non-zero dynamic fibre requires a fitted fibre dynamics model"
                    )
                raw_current = standardizer.inverse(ambient)
                xi = fibre_dynamics.predict(
                    xi,
                    theta,
                    fibre_velocity,
                    raw_states=raw_current,
                )
            else:
                raise ValueError(f"Unknown fibre rollout mode: {fibre_mode}")
        theta = wrap_angles(theta + phase_delta)
    return (
        np.asarray(ambient_rows, dtype=np.float64),
        np.asarray(theta_rows, dtype=np.float64),
        np.asarray(xi_rows, dtype=np.float64),
    )


def _full_rollout_rows(
    *,
    decoder: FourierDecoderV3,
    phase_model: PhaseDynamicsV3,
    fibre_model: FibrePCASelection,
    fibre_dynamics: Optional[FibreDynamicsV3],
    standardizer: AmbientStandardizer,
    theta_actual: np.ndarray,
    xi_actual: np.ndarray,
    ambient_actual: np.ndarray,
    data: TrajectoryData,
    min_rollout_steps: int = 2,
    phase_mode: str = "correct",
    fibre_mode: str = "dynamic",
    transformed_velocities: Optional[np.ndarray] = None,
    phase_only_null: bool = False,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    transformed = None
    if transformed_velocities is not None:
        transformed = _finite_2d(
            transformed_velocities, "transformed_velocities", 2
        )
        if transformed.shape[0] != data.n_transitions:
            raise ValueError("transformed_velocities must have one row per transition")
    for segment in contiguous_segments(data, min_transitions=min_rollout_steps):
        correct_velocity = data.transition_velocities[segment.transition_rows]
        counterfactual = (
            correct_velocity
            if transformed is None
            else transformed[segment.transition_rows]
        )
        phase_velocity = correct_velocity if phase_only_null else counterfactual
        fibre_velocity = correct_velocity if phase_only_null else counterfactual
        context_velocity = correct_velocity if phase_only_null else counterfactual
        predicted, _theta, _xi = _rollout_full_segment(
            decoder=decoder,
            phase_model=phase_model,
            fibre_model=fibre_model,
            fibre_dynamics=fibre_dynamics,
            standardizer=standardizer,
            initial_theta=theta_actual[segment.state_indices[0]],
            initial_xi=xi_actual[segment.state_indices[0]],
            velocities=correct_velocity,
            phase_mode=phase_mode,
            fibre_mode=fibre_mode,
            phase_velocities=phase_velocity,
            fibre_velocities=fibre_velocity,
            context_velocities=context_velocity,
        )
        actual = ambient_actual[segment.state_indices]
        rows.append(
            {
                "trajectory_id": int(segment.trajectory_id),
                "segment_id": int(segment.segment_id),
                "rollout_nmte": _relative_error(actual, predicted),
                "n_transitions": int(segment.transition_rows.size),
            }
        )
    return rows


def _fibre_candidate_specs(config: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for alpha in config.get("linear_alphas", [1e-6, 1e-4, 1e-2, 1.0]):
        rows.append({"kind": "linear_ar", "alpha": float(alpha)})
    for degree in config.get("polynomial_degrees", [2]):
        for alpha in config.get("polynomial_alphas", [1e-5, 1e-3, 1e-1, 10.0]):
            rows.append(
                {
                    "kind": "ridge_polynomial",
                    "degree": int(degree),
                    "alpha": float(alpha),
                }
            )
    for n_clusters in config.get("local_clusters", [4, 8, 16]):
        for alpha in config.get("local_alphas", [1e-5, 1e-3, 1e-1]):
            rows.append(
                {
                    "kind": "local_piecewise_affine",
                    "n_clusters": int(n_clusters),
                    "alpha": float(alpha),
                }
            )
    if bool(config.get("enable_active_set_model", True)):
        for alpha in config.get("active_set_alphas", [1e-5, 1e-3, 1e-1]):
            rows.append({"kind": "active_set_conditioned", "alpha": float(alpha)})
    return rows


def select_fibre_dynamics_v3(
    *,
    fibre_cv: FibrePCASelection,
    theta_fit: np.ndarray,
    xi_fit: np.ndarray,
    source_fit: TrajectoryData,
    theta_validation: np.ndarray,
    xi_validation: np.ndarray,
    source_validation: TrajectoryData,
    ambient_validation: np.ndarray,
    decoder_cv: FourierDecoderV3,
    phase_cv: PhaseDynamicsV3,
    standardizer: AmbientStandardizer,
    config: Mapping[str, Any],
    seed: int,
) -> Tuple[Optional[FibreDynamicsV3], Optional[Dict[str, Any]], Dict[str, Any]]:
    """Select S3 fibre dynamics by held-out multi-step validation rollout."""
    if fibre_cv.n_components == 0:
        return None, None, {
            "selected_kind": "none",
            "reason": "selected fibre dimension is zero",
            "candidate_rows": [],
        }
    candidate_rows: List[Dict[str, Any]] = []
    candidate_models: Dict[int, FibreDynamicsV3] = {}
    specs = _fibre_candidate_specs(config)
    for candidate_index, spec in enumerate(specs):
        try:
            model = FibreDynamicsV3.fit(
                xi_fit,
                theta_fit,
                source_fit,
                kind=str(spec["kind"]),
                alpha=float(spec["alpha"]),
                degree=int(spec.get("degree", 2)),
                n_clusters=int(spec.get("n_clusters", 8)),
                raw_states=source_fit.states,
                active_max_units=int(config.get("active_max_units", 16)),
                active_max_regimes=int(config.get("active_max_regimes", 64)),
                active_min_samples=int(config.get("active_min_samples", 32)),
                max_polynomial_features=int(
                    config.get("max_polynomial_features", 2048)
                ),
                seed=int(seed) + candidate_index,
            )
        except ValueError as exc:
            candidate_rows.append(
                {
                    **spec,
                    "candidate_index": candidate_index,
                    "status": "skipped",
                    "skip_reason": str(exc),
                    "validation_full_rollout_nmte": None,
                    "validation_full_rollout_nmte_se": None,
                    "n_validation_trajectories": 0,
                }
            )
            continue
        try:
            rollout_rows = _full_rollout_rows(
                decoder=decoder_cv,
                phase_model=phase_cv,
                fibre_model=fibre_cv,
                fibre_dynamics=model,
                standardizer=standardizer,
                theta_actual=theta_validation,
                xi_actual=xi_validation,
                ambient_actual=ambient_validation,
                data=source_validation,
                min_rollout_steps=int(config.get("min_rollout_steps", 2)),
            )
        except Exception as exc:
            instability_reason = _numerical_instability_reason(exc)
            if instability_reason is None:
                raise
            candidate_rows.append(
                {
                    **spec,
                    "candidate_index": candidate_index,
                    "status": "unstable_validation_rollout",
                    "skip_reason": instability_reason,
                    "validation_full_rollout_nmte": None,
                    "validation_full_rollout_nmte_se": None,
                    "n_validation_trajectories": 0,
                    "model_diagnostics": model.diagnostics,
                }
            )
            continue
        trajectory_values = _aggregate_segment_metric_by_trajectory(
            rollout_rows, "rollout_nmte"
        )
        mean, se = _mean_se(trajectory_values)
        row = {
            **spec,
            "candidate_index": candidate_index,
            "validation_full_rollout_nmte": mean,
            "validation_full_rollout_nmte_se": se,
            "n_validation_trajectories": len(trajectory_values),
            "model_diagnostics": model.diagnostics,
            "status": "evaluated",
        }
        candidate_rows.append(row)
        candidate_models[candidate_index] = model
    kind_rank = {
        "linear_ar": 0,
        "ridge_polynomial": 1,
        "local_piecewise_affine": 2,
        "active_set_conditioned": 3,
    }
    choice = _one_standard_error_choice(
        candidate_rows,
        metric="validation_full_rollout_nmte",
        se_metric="validation_full_rollout_nmte_se",
        complexity_key=lambda row: (
            kind_rank[str(row["kind"])],
            int(row.get("degree", 1)),
            int(row.get("n_clusters", 0)),
            -float(row["alpha"]),
        ),
    )
    selected_index = int(choice["candidate_index"])
    return candidate_models[selected_index], dict(choice), {
        "selection_protocol": (
            "candidate fibre dynamics fitted on source-fit and selected by "
            "source-validation full hidden rollout NMTE"
        ),
        "candidate_rows": candidate_rows,
        "selected": dict(choice),
    }


def refit_selected_fibre_dynamics(
    *,
    choice: Optional[Mapping[str, Any]],
    fibre: FibrePCASelection,
    xi: np.ndarray,
    theta: np.ndarray,
    data: TrajectoryData,
    config: Mapping[str, Any],
    seed: int,
) -> Optional[FibreDynamicsV3]:
    if fibre.n_components == 0:
        return None
    if choice is None:
        raise RuntimeError("A non-zero fibre has no selected dynamics candidate")
    return FibreDynamicsV3.fit(
        xi,
        theta,
        data,
        kind=str(choice["kind"]),
        alpha=float(choice["alpha"]),
        degree=int(choice.get("degree", 2)),
        n_clusters=int(choice.get("n_clusters", 8)),
        raw_states=data.states,
        active_max_units=int(config.get("active_max_units", 16)),
        active_max_regimes=int(config.get("active_max_regimes", 64)),
        active_min_samples=int(config.get("active_min_samples", 32)),
        max_polynomial_features=int(config.get("max_polynomial_features", 2048)),
        seed=int(seed),
    )


@dataclass
class MatchedLinearBaseline:
    pca: PCA
    ridge: Ridge
    latent_dim: int
    diagnostics: Dict[str, Any]

    @classmethod
    def fit(
        cls,
        ambient: np.ndarray,
        data: TrajectoryData,
        *,
        latent_dim: int,
        alpha: float = 1e-4,
        seed: int = 42,
    ) -> "MatchedLinearBaseline":
        ambient = _finite_2d(ambient, "ambient")
        dimension = min(int(latent_dim), ambient.shape[1], ambient.shape[0] - 1)
        if dimension < 1:
            raise ValueError("Matched linear baseline requires at least one dimension")
        pca = PCA(n_components=dimension, random_state=int(seed))
        latent = pca.fit_transform(ambient)
        x = np.column_stack(
            [latent[data.transition_prev], data.transition_velocities]
        )
        y = latent[data.transition_next]
        ridge = Ridge(alpha=float(alpha), fit_intercept=True, solver="lsqr")
        ridge.fit(x, y)
        return cls(
            pca=pca,
            ridge=ridge,
            latent_dim=dimension,
            diagnostics={
                "latent_dim": int(dimension),
                "alpha": float(alpha),
                "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
                "fit_scope": "source_fit_plus_validation",
            },
        )

    def reconstruct(self, ambient: np.ndarray) -> np.ndarray:
        ambient = _finite_2d(ambient, "ambient")
        return np.asarray(
            self.pca.inverse_transform(self.pca.transform(ambient)),
            dtype=np.float64,
        )

    @property
    def basis(self) -> np.ndarray:
        return np.asarray(self.pca.components_.T, dtype=np.float64)

    def rollout(self, initial_ambient: np.ndarray, velocities: np.ndarray) -> np.ndarray:
        current = self.pca.transform(
            np.asarray(initial_ambient, dtype=np.float64).reshape(1, -1)
        )
        velocities = np.asarray(velocities, dtype=np.float64).reshape(-1, 2)
        result = [self.pca.inverse_transform(current)[0]]
        for velocity in velocities:
            current = np.asarray(
                self.ridge.predict(
                    np.column_stack([current, velocity.reshape(1, 2)])
                ),
                dtype=np.float64,
            )
            result.append(self.pca.inverse_transform(current)[0])
        return np.asarray(result, dtype=np.float64)


def _orthonormal_columns(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("Basis matrix must be 2-D")
    if matrix.shape[1] == 0:
        return np.empty((matrix.shape[0], 0), dtype=np.float64)
    u, singular, _vh = np.linalg.svd(matrix, full_matrices=False)
    rank = int(np.sum(singular > max(matrix.shape) * np.finfo(float).eps * singular[0]))
    return u[:, :rank]


def _projection_fractions(deltas: np.ndarray, bases: Sequence[np.ndarray]) -> np.ndarray:
    deltas = _finite_2d(deltas, "deltas")
    if len(bases) != deltas.shape[0]:
        raise ValueError("One tangent basis is required per delta")
    fractions = np.zeros(deltas.shape[0], dtype=np.float64)
    for row, basis in enumerate(bases):
        q = _orthonormal_columns(basis)
        denominator = float(np.dot(deltas[row], deltas[row]))
        if denominator <= _EPS or q.shape[1] == 0:
            fractions[row] = 0.0
        else:
            projected = q @ (q.T @ deltas[row])
            fractions[row] = float(
                np.dot(projected, projected) / max(denominator, _EPS)
            )
    return np.clip(fractions, 0.0, 1.0)


def _assign_transition_metric(
    rows_by_trajectory: Dict[int, Dict[str, Any]],
    values: np.ndarray,
    data: TrajectoryData,
    metric_name: str,
) -> None:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.shape[0] != data.n_transitions:
        raise ValueError(f"{metric_name} must have one value per transition")
    groups = data.trajectory_ids[data.transition_prev]
    for trajectory_id in data.unique_trajectories:
        selected = values[groups == trajectory_id]
        if selected.size:
            rows_by_trajectory[int(trajectory_id)][metric_name] = float(
                np.mean(selected)
            )


def _assign_segment_metric(
    rows_by_trajectory: Dict[int, Dict[str, Any]],
    segment_rows: Sequence[Mapping[str, Any]],
    source_metric: str,
    target_metric: str,
) -> None:
    grouped: Dict[int, List[float]] = {}
    for row in segment_rows:
        grouped.setdefault(int(row["trajectory_id"]), []).append(
            float(row[source_metric])
        )
    for trajectory_id, values in grouped.items():
        rows_by_trajectory[trajectory_id][target_metric] = float(np.mean(values))


def _counterfactual_transition_velocities(
    data: TrajectoryData,
    *,
    seed: int,
) -> Dict[str, np.ndarray]:
    velocity = np.asarray(data.transition_velocities, dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    rotation = np.asarray([[0.0, -1.0], [1.0, 0.0]], dtype=np.float64)
    return {
        "correct": velocity.copy(),
        "velocity_shuffle": velocity[rng.permutation(velocity.shape[0])],
        "sign_flip": -velocity,
        "rotate_90": velocity @ rotation.T,
        "axis_swap": velocity[:, ::-1],
    }


def _action_direction_labels(velocity: np.ndarray) -> np.ndarray:
    velocity = _finite_2d(velocity, "velocity", 2)
    absolute = np.abs(velocity)
    axis = np.argmax(absolute, axis=1)
    selected = velocity[np.arange(velocity.shape[0]), axis]
    sign = (selected >= 0.0).astype(np.int64)
    labels = axis * 2 + sign
    stationary = np.linalg.norm(velocity, axis=1) <= 1e-10
    labels[stationary] = 4
    return labels


def _phase_action_classification(
    phase_delta: np.ndarray,
    actual_velocity: np.ndarray,
    action_matrix: np.ndarray,
) -> np.ndarray:
    phase_delta = _finite_2d(phase_delta, "phase_delta", 2)
    actual_velocity = _finite_2d(actual_velocity, "actual_velocity", 2)
    inverse = np.linalg.pinv(np.asarray(action_matrix, dtype=np.float64))
    decoded_velocity = phase_delta @ inverse.T
    return (
        _action_direction_labels(decoded_velocity)
        == _action_direction_labels(actual_velocity)
    ).astype(np.float64)


def _square_loop_metrics(
    model: "SourceReducedModelV3",
    theta: np.ndarray,
    xi: np.ndarray,
    ambient: np.ndarray,
    data: TrajectoryData,
) -> Dict[int, Dict[str, float]]:
    nonzero = np.abs(data.transition_velocities)
    positive = nonzero[nonzero > 1e-10]
    step = float(np.median(positive)) if positive.size else 1.0
    u = np.asarray([step, 0.0], dtype=np.float64)
    v = np.asarray([0.0, step], dtype=np.float64)
    loop = np.asarray([u, v, -u, -v], dtype=np.float64)
    result: Dict[int, Dict[str, float]] = {}
    for trajectory_id in data.unique_trajectories:
        states = data.trajectory_state_indices(int(trajectory_id))
        if states.size == 0:
            continue
        initial = int(states[0])
        predicted, theta_path, _xi_path = _rollout_full_segment(
            decoder=model.selected_decoder,
            phase_model=model.phase_dynamics,
            fibre_model=model.fibre_model,
            fibre_dynamics=model.fibre_dynamics,
            standardizer=model.standardizer,
            initial_theta=theta[initial],
            initial_xi=xi[initial],
            velocities=loop,
        )
        phase_return = float(
            np.linalg.norm(wrap_angles(theta_path[-1] - theta_path[0]))
        )
        hidden_scale = float(np.max(np.linalg.norm(ambient, axis=1)))
        hidden_return = float(
            np.linalg.norm(predicted[-1] - predicted[0]) / max(hidden_scale, _EPS)
        )
        result[int(trajectory_id)] = {
            "square_loop_phase_return": phase_return,
            "square_loop_hidden_return": hidden_return,
        }
    return result


def _select_active_set_probe_units(
    states: np.ndarray,
    *,
    max_units: int,
) -> np.ndarray:
    """Choose a fixed ReLU switching probe on source-fit only."""
    states = _finite_2d(states, "active_set_probe_states")
    activity_rate = np.mean(states > 0.0, axis=0)
    variability = activity_rate * (1.0 - activity_rate)
    n_units = max(1, min(int(max_units), states.shape[1]))
    return np.sort(np.argsort(variability)[-n_units:].astype(np.int64))


def _active_set_test_diagnostics(
    *,
    data: TrajectoryData,
    theta: np.ndarray,
    probe_units: np.ndarray,
    base_fraction: np.ndarray,
    full_fraction: np.ndarray,
    phase_bins_per_axis: int = 8,
) -> Tuple[Dict[int, Dict[str, float]], Dict[str, Any]]:
    """Diagnose whether ReLU switches explain low torus-base tangency.

    Probe units are fixed on source-fit before this function is called.  The
    source-test split is used only to estimate switching rates, run lengths and
    tangent fractions conditional on stable versus switching active sets.
    """
    units = np.asarray(probe_units, dtype=np.int64).reshape(-1)
    if units.size == 0:
        return {}, {"available": False, "reason": "empty fit-selected probe"}
    if units.min() < 0 or units.max() >= data.n_units:
        raise IndexError("Fit-selected active-set probe units exceed test dimension")
    theta = _finite_2d(theta, "active_set_theta", 2)
    if theta.shape[0] != data.n_states:
        raise ValueError("active-set theta must have one row per state")
    base_fraction = np.asarray(base_fraction, dtype=np.float64).reshape(-1)
    full_fraction = np.asarray(full_fraction, dtype=np.float64).reshape(-1)
    if base_fraction.size != data.n_transitions or full_fraction.size != data.n_transitions:
        raise ValueError("active-set tangent arrays must have one value per transition")

    bits = data.states[:, units] > 0.0
    switch = np.any(
        bits[data.transition_prev] != bits[data.transition_next], axis=1
    )
    result: Dict[int, Dict[str, float]] = {}
    for trajectory_id in data.unique_trajectories:
        transition_rows = data.trajectory_transition_rows(int(trajectory_id))
        if transition_rows.size == 0:
            continue
        switches = switch[transition_rows]
        stable_lengths: List[int] = []
        current = 0
        for changed in switches.tolist():
            current += 1
            if changed:
                stable_lengths.append(current)
                current = 0
        if current:
            stable_lengths.append(current)
        row: Dict[str, float] = {
            "active_set_switch_fraction": float(np.mean(switches)),
            "active_set_mean_stable_run_length": float(
                np.mean(stable_lengths) if stable_lengths else 0.0
            ),
            "active_set_unique_transition_regimes": float(
                len(
                    {
                        tuple(int(value) for value in bits[int(state)].tolist())
                        for state in data.transition_prev[transition_rows]
                    }
                )
            ),
        }
        stable_rows = transition_rows[~switches]
        switching_rows = transition_rows[switches]
        if stable_rows.size:
            row["torus_base_tangent_fraction_stable_active_set"] = float(
                np.mean(base_fraction[stable_rows])
            )
            row["base_plus_fibre_tangent_fraction_stable_active_set"] = float(
                np.mean(full_fraction[stable_rows])
            )
        if switching_rows.size:
            row["torus_base_tangent_fraction_switching_active_set"] = float(
                np.mean(base_fraction[switching_rows])
            )
            row["base_plus_fibre_tangent_fraction_switching_active_set"] = float(
                np.mean(full_fraction[switching_rows])
            )
        if stable_rows.size and switching_rows.size:
            row["active_set_switch_torus_tangent_gap"] = float(
                row["torus_base_tangent_fraction_stable_active_set"]
                - row["torus_base_tangent_fraction_switching_active_set"]
            )

        state_rows = data.trajectory_state_indices(int(trajectory_id))
        scaled = (wrap_angles(theta[state_rows]) + np.pi) / (2.0 * np.pi)
        bins = np.floor(scaled * int(phase_bins_per_axis)).astype(np.int64)
        bins %= int(phase_bins_per_axis)
        regimes_by_bin: Dict[Tuple[int, int], set[Tuple[int, ...]]] = {}
        for state_row, phase_bin in zip(state_rows.tolist(), bins.tolist()):
            key = (int(phase_bin[0]), int(phase_bin[1]))
            signature = tuple(int(value) for value in bits[int(state_row)].tolist())
            regimes_by_bin.setdefault(key, set()).add(signature)
        if regimes_by_bin:
            multiplicity = [len(values) for values in regimes_by_bin.values()]
            row["active_set_regimes_per_occupied_phase_bin"] = float(
                np.mean(multiplicity)
            )
            row["active_set_fraction_phase_bins_with_multiple_regimes"] = float(
                np.mean(np.asarray(multiplicity) > 1)
            )
        result[int(trajectory_id)] = row

    return result, {
        "available": True,
        "selection_scope": "source_fit_only",
        "probe_unit_indices_local": units.tolist(),
        "n_probe_units": int(units.size),
        "phase_bins_per_axis": int(phase_bins_per_axis),
        "n_test_transitions": int(data.n_transitions),
        "global_switch_fraction": float(np.mean(switch)) if switch.size else None,
        "n_global_transition_regimes": int(
            len(
                {
                    tuple(int(value) for value in row.tolist())
                    for row in bits[data.transition_prev]
                }
            )
        ),
    }


@dataclass
class SourceReducedModelV3:
    """Frozen source reduced model selected without source-test leakage."""

    schema_version: str
    claim_scope: str
    selected_unit_indices: np.ndarray
    phase_encoder: FrozenTorusPhaseEncoder
    standardizer: AmbientStandardizer
    phase_dynamics: PhaseDynamicsV3
    base_decoder: FourierDecoderV3
    selected_decoder: FourierDecoderV3
    fibre_model: FibrePCASelection
    fibre_dynamics: Optional[FibreDynamicsV3]
    matched_linear_baseline: MatchedLinearBaseline
    active_set_probe_unit_indices: np.ndarray
    diagnostics: Dict[str, Any]

    @classmethod
    def fit(
        cls,
        *,
        topology_data: TrajectoryData,
        source_fit: TrajectoryData,
        source_validation: TrajectoryData,
        selected_unit_indices: Sequence[int],
        landmark_indices: Optional[np.ndarray] = None,
        landmark_torus_turns: Optional[np.ndarray] = None,
        phase_encoder: Optional[FrozenTorusPhaseEncoder] = None,
        config: Mapping[str, Any],
        seed: int = 42,
    ) -> "SourceReducedModelV3":
        """Fit and select every component using topology/fit/validation only."""
        selected = np.asarray(selected_unit_indices, dtype=np.int64).reshape(-1)
        for name, data in (
            ("topology", topology_data),
            ("fit", source_fit),
            ("validation", source_validation),
        ):
            if data.n_units != selected.size:
                raise ValueError(
                    f"{name} data contain {data.n_units} units; expected {selected.size}"
                )
        phase_cfg = dict(config.get("phase", {}))
        decoder_cfg = dict(config.get("decoder", {}))
        fibre_cfg = dict(config.get("fibre", {}))
        dynamics_cfg = dict(config.get("fibre_dynamics", {}))
        baseline_cfg = dict(config.get("baselines", {}))

        if phase_encoder is None:
            if landmark_indices is None or landmark_torus_turns is None:
                raise ValueError(
                    "Either an explicit phase_encoder or both legacy "
                    "landmark_indices/landmark_torus_turns must be supplied"
                )
            phase_encoder = FrozenTorusPhaseEncoder.fit_from_formal_topology(
                topology_data,
                selected_unit_indices=selected,
                landmark_indices=landmark_indices,
                landmark_torus_turns=landmark_torus_turns,
                chart_dim=int(config.get("chart_dim", 8)),
                seed=int(seed),
            )
        else:
            encoder_units = np.asarray(
                phase_encoder.selected_unit_indices, dtype=np.int64
            ).reshape(-1)
            if not np.array_equal(encoder_units, selected):
                raise ValueError(
                    "Explicit phase_encoder selected units differ from "
                    "selected_unit_indices"
                )
        standardizer = AmbientStandardizer.fit(source_fit, selected)
        ambient_fit = standardizer.transform(source_fit.states)
        ambient_validation = standardizer.transform(source_validation.states)
        raw_theta_fit = phase_encoder.raw_angles(source_fit.states)
        raw_theta_validation = phase_encoder.raw_angles(source_validation.states)

        phase_cv, phase_final, phase_diagnostics = fit_phase_dynamics_v3(
            raw_theta_fit,
            source_fit,
            raw_theta_validation,
            source_validation,
            max_gl2z_entry=int(phase_cfg.get("max_gl2z_entry", 2)),
            basis_search_alpha=float(phase_cfg.get("basis_search_alpha", 1e-6)),
            action_alphas=tuple(
                phase_cfg.get("action_alphas", [1e-8, 1e-6, 1e-4, 1e-2, 1.0])
            ),
            residual_orders=tuple(phase_cfg.get("residual_orders", [1, 2])),
            residual_alphas=tuple(
                phase_cfg.get("residual_alphas", [1e-6, 1e-4, 1e-2, 1.0])
            ),
            residual_min_relative_improvement=float(
                phase_cfg.get("residual_min_relative_improvement", 0.01)
            ),
            state_only_order=int(phase_cfg.get("state_only_order", 2)),
            state_only_alpha=float(phase_cfg.get("state_only_alpha", 1e-3)),
        )
        theta_fit = phase_cv.align(raw_theta_fit)
        theta_validation = phase_cv.align(raw_theta_validation)

        base_cv, decoder_cv, base_final, decoder_final, decoder_diagnostics = (
            select_fourier_decoders_v3(
                theta_fit,
                ambient_fit,
                source_fit,
                theta_validation,
                ambient_validation,
                source_validation,
                phase_cv,
                orders=tuple(decoder_cfg.get("orders", [1, 2, 3, 4, 5, 6])),
                alphas=tuple(
                    decoder_cfg.get("alphas", [1e-6, 1e-4, 1e-2, 1.0, 100.0])
                ),
                context_modes=tuple(
                    decoder_cfg.get(
                        "context_modes",
                        [
                            "none",
                            "previous_velocity",
                            "current_velocity",
                            "previous_current_velocity",
                        ],
                    )
                ),
                conditioned_min_relative_rollout_improvement=float(
                    decoder_cfg.get(
                        "conditioned_min_relative_rollout_improvement", 0.0
                    )
                ),
                min_rollout_steps=int(decoder_cfg.get("min_rollout_steps", 2)),
            )
        )

        fit_context = decoder_context_per_state(source_fit, decoder_cv.context_mode)
        validation_context = decoder_context_per_state(
            source_validation, decoder_cv.context_mode
        )
        residual_fit = ambient_fit - decoder_cv.predict(theta_fit, fit_context)
        residual_validation = ambient_validation - decoder_cv.predict(
            theta_validation, validation_context
        )
        fibre_cv, selected_fibre_dim, fibre_diagnostics = select_fibre_pca_v3(
            residual_fit,
            source_fit,
            residual_validation,
            source_validation,
            candidate_dims=tuple(
                fibre_cfg.get(
                    "candidate_dims", [0, 1, 2, 4, 6, 8, 12, 16, 24, 32]
                )
            ),
            plateau_relative_gain=float(
                fibre_cfg.get("plateau_relative_gain", 0.02)
            ),
            seed=int(seed) + 31,
        )
        xi_fit = fibre_cv.encode(residual_fit)
        xi_validation = fibre_cv.encode(residual_validation)
        _dynamics_cv, dynamics_choice, dynamics_diagnostics = (
            select_fibre_dynamics_v3(
                fibre_cv=fibre_cv,
                theta_fit=theta_fit,
                xi_fit=xi_fit,
                source_fit=source_fit,
                theta_validation=theta_validation,
                xi_validation=xi_validation,
                source_validation=source_validation,
                ambient_validation=ambient_validation,
                decoder_cv=decoder_cv,
                phase_cv=phase_cv,
                standardizer=standardizer,
                config=dynamics_cfg,
                seed=int(seed) + 47,
            )
        )

        combined = combine_trajectory_data(source_fit, source_validation)
        raw_theta_combined = np.concatenate(
            [raw_theta_fit, raw_theta_validation], axis=0
        )
        theta_combined = phase_final.align(raw_theta_combined)
        ambient_combined = np.concatenate(
            [ambient_fit, ambient_validation], axis=0
        )
        combined_context = decoder_context_per_state(
            combined, decoder_final.context_mode
        )
        combined_residual = ambient_combined - decoder_final.predict(
            theta_combined, combined_context
        )
        fibre_final = fit_final_fibre_pca(
            combined_residual,
            n_components=selected_fibre_dim,
            seed=int(seed) + 53,
            selection_diagnostics=fibre_diagnostics,
        )
        xi_combined = fibre_final.encode(combined_residual)
        fibre_dynamics_final = refit_selected_fibre_dynamics(
            choice=dynamics_choice,
            fibre=fibre_final,
            xi=xi_combined,
            theta=theta_combined,
            data=combined,
            config=dynamics_cfg,
            seed=int(seed) + 59,
        )
        matched_dim = 2 + selected_fibre_dim
        matched_linear = MatchedLinearBaseline.fit(
            ambient_combined,
            combined,
            latent_dim=matched_dim,
            alpha=float(baseline_cfg.get("linear_state_space_alpha", 1e-4)),
            seed=int(seed) + 67,
        )
        active_set_probe_units = _select_active_set_probe_units(
            source_fit.states,
            max_units=int(dynamics_cfg.get("active_max_units", 16)),
        )
        diagnostics = {
            "schema_version": SCHEMA_VERSION,
            "claim_scope": CLAIM_SCOPE,
            "selection_leakage_guard": {
                "topology_phase_encoder": "formal topology trajectories only",
                "normalization": "source-fit only",
                "hyperparameter_fit": "source-fit",
                "hyperparameter_selection": "source-validation",
                "selected_model_refit": "source-fit plus source-validation",
                "source_test_used_during_fit": False,
            },
            "n_selected_units": int(selected.size),
            "n_retained_units": int(standardizer.ambient_dim),
            "phase_encoder": phase_encoder.diagnostics,
            "phase_dynamics": phase_diagnostics,
            "decoder": decoder_diagnostics,
            "fibre": fibre_final.diagnostics,
            "fibre_dynamics_selection": dynamics_diagnostics,
            "fibre_dynamics_final": (
                None
                if fibre_dynamics_final is None
                else fibre_dynamics_final.diagnostics
            ),
            "matched_linear_baseline": matched_linear.diagnostics,
            "active_set_probe": {
                "selection_scope": "source_fit_only",
                "unit_indices_local": active_set_probe_units.tolist(),
                "n_units": int(active_set_probe_units.size),
            },
            "fit_split": {
                "n_states": source_fit.n_states,
                "n_transitions": source_fit.n_transitions,
                "n_trajectories": int(source_fit.unique_trajectories.size),
            },
            "validation_split": {
                "n_states": source_validation.n_states,
                "n_transitions": source_validation.n_transitions,
                "n_trajectories": int(source_validation.unique_trajectories.size),
            },
        }
        return cls(
            schema_version=SCHEMA_VERSION,
            claim_scope=CLAIM_SCOPE,
            selected_unit_indices=selected,
            phase_encoder=phase_encoder,
            standardizer=standardizer,
            phase_dynamics=phase_final,
            base_decoder=base_final,
            selected_decoder=decoder_final,
            fibre_model=fibre_final,
            fibre_dynamics=fibre_dynamics_final,
            matched_linear_baseline=matched_linear,
            active_set_probe_unit_indices=active_set_probe_units,
            diagnostics=diagnostics,
        )

    def phase(self, selected_states: np.ndarray) -> np.ndarray:
        return self.phase_dynamics.align(
            self.phase_encoder.raw_angles(selected_states)
        )

    def standardized(self, selected_states: np.ndarray) -> np.ndarray:
        return self.standardizer.transform(selected_states)

    def reconstruct_base(
        self,
        data: TrajectoryData,
        *,
        selected_context: bool = False,
    ) -> np.ndarray:
        theta = self.phase(data.states)
        decoder = self.selected_decoder if selected_context else self.base_decoder
        context = decoder_context_per_state(data, decoder.context_mode)
        return decoder.predict(theta, context)

    def reconstruct_full(self, data: TrajectoryData) -> np.ndarray:
        ambient = self.standardized(data.states)
        theta = self.phase(data.states)
        context = decoder_context_per_state(data, self.selected_decoder.context_mode)
        base = self.selected_decoder.predict(theta, context)
        return base + self.fibre_model.reconstruct(ambient - base)

    def evaluate_test(
        self,
        source_test: TrajectoryData,
        *,
        seed: int = 42,
        n_bootstrap: int = 2000,
        min_rollout_steps: int = 2,
    ) -> Dict[str, Any]:
        """Consume a frozen source-test split without fitting or selecting anything."""
        if source_test.n_units != self.selected_unit_indices.size:
            raise ValueError("source_test hidden dimension differs from fitted source model")
        ambient = self.standardized(source_test.states)
        theta = self.phase(source_test.states)
        base_context = decoder_context_per_state(
            source_test, self.base_decoder.context_mode
        )
        selected_context = decoder_context_per_state(
            source_test, self.selected_decoder.context_mode
        )
        base_prediction = self.base_decoder.predict(theta, base_context)
        selected_base_prediction = self.selected_decoder.predict(
            theta, selected_context
        )
        residual = ambient - selected_base_prediction
        xi = self.fibre_model.encode(residual)
        full_prediction = selected_base_prediction + self.fibre_model.reconstruct(
            residual
        )
        matched_prediction = self.matched_linear_baseline.reconstruct(ambient)

        rows_by_trajectory: Dict[int, Dict[str, Any]] = {
            int(trajectory_id): {
                "trajectory_id": int(trajectory_id),
                "claim_scope": CLAIM_SCOPE,
            }
            for trajectory_id in source_test.unique_trajectories
        }
        for trajectory_id in source_test.unique_trajectories:
            state_rows = source_test.trajectory_state_indices(int(trajectory_id))
            if state_rows.size == 0:
                continue
            row = rows_by_trajectory[int(trajectory_id)]
            row["surface_mfe_torus_base"] = _relative_error(
                ambient[state_rows], base_prediction[state_rows]
            )
            row["surface_mfe_selected_base"] = _relative_error(
                ambient[state_rows], selected_base_prediction[state_rows]
            )
            row["surface_mfe_base_plus_fibre"] = _relative_error(
                ambient[state_rows], full_prediction[state_rows]
            )
            row["surface_mfe_matched_pca"] = _relative_error(
                ambient[state_rows], matched_prediction[state_rows]
            )
            # Backward-readable alias used by v2 source summaries.
            row["surface_cross_mfe"] = row["surface_mfe_torus_base"]

        prev = source_test.transition_prev
        nxt = source_test.transition_next
        delta_h = ambient[nxt] - ambient[prev]
        base_jacobian = self.base_decoder.jacobian(
            theta[prev], None if base_context is None else base_context[prev]
        )
        selected_jacobian = self.selected_decoder.jacobian(
            theta[prev],
            None if selected_context is None else selected_context[prev],
        )
        base_bases = [base_jacobian[index] for index in range(base_jacobian.shape[0])]
        full_bases = []
        fibre_basis = self.fibre_model.basis
        for index in range(selected_jacobian.shape[0]):
            if fibre_basis.size:
                full_bases.append(
                    np.column_stack([selected_jacobian[index], fibre_basis])
                )
            else:
                full_bases.append(selected_jacobian[index])
        base_fraction = _projection_fractions(delta_h, base_bases)
        full_fraction = _projection_fractions(delta_h, full_bases)
        matched_basis = self.matched_linear_baseline.basis
        matched_fraction = _projection_fractions(
            delta_h, [matched_basis] * delta_h.shape[0]
        )
        _assign_transition_metric(
            rows_by_trajectory,
            base_fraction,
            source_test,
            "torus_base_tangent_fraction",
        )
        _assign_transition_metric(
            rows_by_trajectory,
            full_fraction,
            source_test,
            "base_plus_fibre_tangent_fraction",
        )
        _assign_transition_metric(
            rows_by_trajectory,
            matched_fraction,
            source_test,
            "matched_pca_tangent_fraction",
        )
        active_set_rows, active_set_diagnostics = _active_set_test_diagnostics(
            data=source_test,
            theta=theta,
            probe_units=self.active_set_probe_unit_indices,
            base_fraction=base_fraction,
            full_fraction=full_fraction,
            phase_bins_per_axis=8,
        )
        for trajectory_id, metrics in active_set_rows.items():
            rows_by_trajectory[trajectory_id].update(metrics)
        for row in rows_by_trajectory.values():
            if "torus_base_tangent_fraction" in row:
                row["torus_tangent_fraction"] = row[
                    "torus_base_tangent_fraction"
                ]

        actual_delta = wrap_angles(theta[nxt] - theta[prev])
        counterfactuals = _counterfactual_transition_velocities(
            source_test, seed=int(seed) + 101
        )
        additive_delta = self.phase_dynamics.predict_delta(
            theta[prev], counterfactuals["correct"], use_residual=False
        )
        nonlinear_delta = self.phase_dynamics.predict_delta(
            theta[prev], counterfactuals["correct"], use_residual=True
        )
        phase_predictions: Dict[str, np.ndarray] = {
            "additive": additive_delta,
            "nonlinear": nonlinear_delta,
            "zero_delta": np.zeros_like(actual_delta),
            "state_only": self.phase_dynamics.predict_state_only_delta(theta[prev]),
        }
        for mode in ("velocity_shuffle", "sign_flip", "rotate_90", "axis_swap"):
            phase_predictions[mode] = self.phase_dynamics.predict_delta(
                theta[prev], counterfactuals[mode], use_residual=True
            )
        transition_groups = source_test.trajectory_ids[prev]
        phase_metric_names = {
            "additive": "phase_additive_rmse",
            "nonlinear": "phase_nonlinear_rmse",
            "zero_delta": "phase_zero_delta_rmse",
            "state_only": "phase_state_only_rmse",
            "velocity_shuffle": "phase_velocity_shuffle_rmse",
            "sign_flip": "phase_sign_flip_rmse",
            "rotate_90": "phase_rotate_90_rmse",
            "axis_swap": "phase_axis_swap_rmse",
        }
        classification_arrays = {
            mode: _phase_action_classification(
                prediction,
                source_test.transition_velocities,
                self.phase_dynamics.action_matrix,
            )
            for mode, prediction in phase_predictions.items()
        }
        for trajectory_id in source_test.unique_trajectories:
            transition_rows = np.flatnonzero(transition_groups == trajectory_id)
            if transition_rows.size == 0:
                continue
            row = rows_by_trajectory[int(trajectory_id)]
            for mode, prediction in phase_predictions.items():
                row[phase_metric_names[mode]] = _wrapped_rmse(
                    actual_delta[transition_rows], prediction[transition_rows]
                )
                row[f"action_classification_{mode}_accuracy"] = float(
                    np.mean(classification_arrays[mode][transition_rows])
                )
            step_rms = float(
                np.sqrt(np.mean(actual_delta[transition_rows] ** 2))
            )
            row["phase_step_rms"] = step_rms
            row["normalized_additive_phase_rmse"] = float(
                row["phase_additive_rmse"] / max(step_rms, _EPS)
            )
            row["velocity_specificity_ratio"] = float(
                row["phase_velocity_shuffle_rmse"]
                / max(row["phase_additive_rmse"], _EPS)
            )
            row["minimum_confirmatory_phase_specificity_ratio"] = float(
                min(
                    row["phase_sign_flip_rmse"],
                    row["phase_rotate_90_rmse"],
                    row["phase_axis_swap_rmse"],
                )
                / max(row["phase_nonlinear_rmse"], _EPS)
            )

        base_rollouts = _base_rollout_rows(
            self.base_decoder,
            self.phase_dynamics,
            theta,
            ambient,
            source_test,
            min_rollout_steps=min_rollout_steps,
        )
        selected_base_rollouts = _base_rollout_rows(
            self.selected_decoder,
            self.phase_dynamics,
            theta,
            ambient,
            source_test,
            min_rollout_steps=min_rollout_steps,
        )
        static_fibre_rollouts = _full_rollout_rows(
            decoder=self.selected_decoder,
            phase_model=self.phase_dynamics,
            fibre_model=self.fibre_model,
            fibre_dynamics=None,
            standardizer=self.standardizer,
            theta_actual=theta,
            xi_actual=xi,
            ambient_actual=ambient,
            data=source_test,
            min_rollout_steps=min_rollout_steps,
            fibre_mode="hold_initial",
        )
        full_rollouts = _full_rollout_rows(
            decoder=self.selected_decoder,
            phase_model=self.phase_dynamics,
            fibre_model=self.fibre_model,
            fibre_dynamics=self.fibre_dynamics,
            standardizer=self.standardizer,
            theta_actual=theta,
            xi_actual=xi,
            ambient_actual=ambient,
            data=source_test,
            min_rollout_steps=min_rollout_steps,
        )
        phase_null_rollouts = _full_rollout_rows(
            decoder=self.selected_decoder,
            phase_model=self.phase_dynamics,
            fibre_model=self.fibre_model,
            fibre_dynamics=self.fibre_dynamics,
            standardizer=self.standardizer,
            theta_actual=theta,
            xi_actual=xi,
            ambient_actual=ambient,
            data=source_test,
            min_rollout_steps=min_rollout_steps,
            phase_mode="zero_delta",
            phase_only_null=True,
        )
        _assign_segment_metric(
            rows_by_trajectory, base_rollouts, "rollout_nmte", "rollout_nmte_torus_base"
        )
        _assign_segment_metric(
            rows_by_trajectory,
            selected_base_rollouts,
            "rollout_nmte",
            "rollout_nmte_selected_base",
        )
        _assign_segment_metric(
            rows_by_trajectory,
            static_fibre_rollouts,
            "rollout_nmte",
            "rollout_nmte_base_plus_static_fibre",
        )
        _assign_segment_metric(
            rows_by_trajectory,
            full_rollouts,
            "rollout_nmte",
            "rollout_nmte_base_plus_fibre",
        )
        _assign_segment_metric(
            rows_by_trajectory,
            phase_null_rollouts,
            "rollout_nmte",
            "rollout_nmte_phase_null",
        )
        for mode in ("velocity_shuffle", "sign_flip", "rotate_90", "axis_swap"):
            null_rows = _full_rollout_rows(
                decoder=self.selected_decoder,
                phase_model=self.phase_dynamics,
                fibre_model=self.fibre_model,
                fibre_dynamics=self.fibre_dynamics,
                standardizer=self.standardizer,
                theta_actual=theta,
                xi_actual=xi,
                ambient_actual=ambient,
                data=source_test,
                min_rollout_steps=min_rollout_steps,
                transformed_velocities=counterfactuals[mode],
            )
            _assign_segment_metric(
                rows_by_trajectory,
                null_rows,
                "rollout_nmte",
                f"rollout_nmte_{mode}",
            )

        matched_rows: List[Dict[str, Any]] = []
        for segment in contiguous_segments(
            source_test, min_transitions=min_rollout_steps
        ):
            actual = ambient[segment.state_indices]
            predicted = self.matched_linear_baseline.rollout(
                actual[0], source_test.transition_velocities[segment.transition_rows]
            )
            matched_rows.append(
                {
                    "trajectory_id": int(segment.trajectory_id),
                    "rollout_nmte": _relative_error(actual, predicted),
                }
            )
        _assign_segment_metric(
            rows_by_trajectory,
            matched_rows,
            "rollout_nmte",
            "rollout_nmte_matched_linear_state_space",
        )

        square_loops = _square_loop_metrics(
            self, theta, xi, ambient, source_test
        )
        for trajectory_id, metrics in square_loops.items():
            rows_by_trajectory[trajectory_id].update(metrics)

        for row in rows_by_trajectory.values():
            if "rollout_nmte_torus_base" in row:
                row["rollout_cross_nmte"] = row["rollout_nmte_torus_base"]
            if (
                "rollout_nmte_phase_null" in row
                and "rollout_nmte_base_plus_fibre" in row
            ):
                row["phase_null_to_full_rollout_ratio"] = float(
                    row["rollout_nmte_phase_null"]
                    / max(row["rollout_nmte_base_plus_fibre"], _EPS)
                )
            null_metrics = [
                row.get("rollout_nmte_velocity_shuffle"),
                row.get("rollout_nmte_sign_flip"),
                row.get("rollout_nmte_rotate_90"),
                row.get("rollout_nmte_axis_swap"),
            ]
            null_metrics = [float(value) for value in null_metrics if value is not None]
            if null_metrics and "rollout_nmte_base_plus_fibre" in row:
                row["minimum_rollout_specificity_ratio"] = float(
                    min(null_metrics)
                    / max(row["rollout_nmte_base_plus_fibre"], _EPS)
                )

        trajectory_rows = [
            rows_by_trajectory[key] for key in sorted(rows_by_trajectory)
        ]
        summary = summarize_trajectory_metrics(
            trajectory_rows,
            n_bootstrap=int(n_bootstrap),
            seed=int(seed),
        )
        hierarchy = {
            "S0_pure_fourier_torus": {
                "surface_metric": "surface_mfe_torus_base",
                "rollout_metric": "rollout_nmte_torus_base",
                "surface_mean": summary.get("metrics", {})
                .get("surface_mfe_torus_base", {})
                .get("mean"),
                "rollout_mean": summary.get("metrics", {})
                .get("rollout_nmte_torus_base", {})
                .get("mean"),
            },
            "S1_velocity_conditioned_torus": {
                "context_mode": self.selected_decoder.context_mode,
                "surface_metric": "surface_mfe_selected_base",
                "rollout_metric": "rollout_nmte_selected_base",
                "surface_mean": summary.get("metrics", {})
                .get("surface_mfe_selected_base", {})
                .get("mean"),
                "rollout_mean": summary.get("metrics", {})
                .get("rollout_nmte_selected_base", {})
                .get("mean"),
            },
            "S2_torus_base_plus_static_linear_fibre": {
                "fibre_dimension": int(self.fibre_model.n_components),
                "surface_metric": "surface_mfe_base_plus_fibre",
                "rollout_metric": "rollout_nmte_base_plus_static_fibre",
                "surface_mean": summary.get("metrics", {})
                .get("surface_mfe_base_plus_fibre", {})
                .get("mean"),
                "rollout_mean": summary.get("metrics", {})
                .get("rollout_nmte_base_plus_static_fibre", {})
                .get("mean"),
            },
            "S3_controlled_fibre_dynamics": {
                "fibre_dimension": int(self.fibre_model.n_components),
                "dynamics_kind": (
                    "none" if self.fibre_dynamics is None else self.fibre_dynamics.kind
                ),
                "surface_metric": "surface_mfe_base_plus_fibre",
                "rollout_metric": "rollout_nmte_base_plus_fibre",
                "surface_mean": summary.get("metrics", {})
                .get("surface_mfe_base_plus_fibre", {})
                .get("mean"),
                "rollout_mean": summary.get("metrics", {})
                .get("rollout_nmte_base_plus_fibre", {})
                .get("mean"),
            },
        }
        return {
            "schema_version": SCHEMA_VERSION,
            "claim_scope": CLAIM_SCOPE,
            "test_consumption": {
                "used_for_fitting": False,
                "used_for_hyperparameter_selection": False,
                "used_for_quality_gate_only": True,
                "n_states": source_test.n_states,
                "n_transitions": source_test.n_transitions,
                "n_trajectories": int(source_test.unique_trajectories.size),
            },
            "trajectory_metrics": trajectory_rows,
            "summary": summary,
            "model_hierarchy": hierarchy,
            "active_set_diagnostics": active_set_diagnostics,
            "counterfactual_action_modes": sorted(counterfactuals),
        }


def _summary_metric_mean(evaluation: Mapping[str, Any], metric: str) -> Optional[float]:
    value = (
        evaluation.get("summary", {})
        .get("metrics", {})
        .get(metric, {})
        .get("mean")
    )
    return None if value is None else float(value)


def evaluate_source_quality_gates(
    model: SourceReducedModelV3,
    test_evaluation: Mapping[str, Any],
    *,
    thresholds: Optional[Mapping[str, float]] = None,
) -> Dict[str, Any]:
    """Apply preregistered exact-torus and torus-plus-fibre gates."""
    limits = {
        "exact_max_surface_mfe": 0.50,
        "exact_max_rollout_nmte": 0.50,
        "exact_min_torus_tangent_fraction": 0.50,
        "max_normalized_additive_phase_rmse": 0.75,
        "min_velocity_specificity_ratio": 1.25,
        "full_max_surface_mfe": 0.50,
        "full_max_rollout_nmte": 0.50,
        "full_min_tangent_fraction": 0.70,
        "min_phase_null_to_full_rollout_ratio": 1.10,
    }
    if thresholds:
        limits.update({key: float(value) for key, value in thresholds.items()})

    metrics = {
        "surface_mfe_torus_base": _summary_metric_mean(
            test_evaluation, "surface_mfe_torus_base"
        ),
        "rollout_nmte_torus_base": _summary_metric_mean(
            test_evaluation, "rollout_nmte_torus_base"
        ),
        "torus_base_tangent_fraction": _summary_metric_mean(
            test_evaluation, "torus_base_tangent_fraction"
        ),
        "normalized_additive_phase_rmse": _summary_metric_mean(
            test_evaluation, "normalized_additive_phase_rmse"
        ),
        "velocity_specificity_ratio": _summary_metric_mean(
            test_evaluation, "velocity_specificity_ratio"
        ),
        "surface_mfe_base_plus_fibre": _summary_metric_mean(
            test_evaluation, "surface_mfe_base_plus_fibre"
        ),
        "rollout_nmte_base_plus_fibre": _summary_metric_mean(
            test_evaluation, "rollout_nmte_base_plus_fibre"
        ),
        "base_plus_fibre_tangent_fraction": _summary_metric_mean(
            test_evaluation, "base_plus_fibre_tangent_fraction"
        ),
        "phase_null_to_full_rollout_ratio": _summary_metric_mean(
            test_evaluation, "phase_null_to_full_rollout_ratio"
        ),
    }

    def _le(metric: str, threshold: str) -> Dict[str, Any]:
        value = metrics[metric]
        return {
            "metric": metric,
            "value": value,
            "operator": "<=",
            "threshold": limits[threshold],
            "passed": bool(value is not None and value <= limits[threshold]),
        }

    def _ge(metric: str, threshold: str) -> Dict[str, Any]:
        value = metrics[metric]
        return {
            "metric": metric,
            "value": value,
            "operator": ">=",
            "threshold": limits[threshold],
            "passed": bool(value is not None and value >= limits[threshold]),
        }

    exact_checks = {
        "surface_mfe": _le("surface_mfe_torus_base", "exact_max_surface_mfe"),
        "rollout_nmte": _le("rollout_nmte_torus_base", "exact_max_rollout_nmte"),
        "torus_tangent_fraction": _ge(
            "torus_base_tangent_fraction", "exact_min_torus_tangent_fraction"
        ),
        "normalized_additive_phase_rmse": _le(
            "normalized_additive_phase_rmse",
            "max_normalized_additive_phase_rmse",
        ),
        "velocity_specificity": _ge(
            "velocity_specificity_ratio", "min_velocity_specificity_ratio"
        ),
    }
    fibre_diagnostics = model.fibre_model.diagnostics
    low_dimensional_fibre_supported = bool(
        fibre_diagnostics.get("low_dimensional_fibre_supported", False)
    )
    full_checks = {
        "surface_mfe": _le(
            "surface_mfe_base_plus_fibre", "full_max_surface_mfe"
        ),
        "rollout_nmte": _le(
            "rollout_nmte_base_plus_fibre", "full_max_rollout_nmte"
        ),
        "base_plus_fibre_tangent_fraction": _ge(
            "base_plus_fibre_tangent_fraction", "full_min_tangent_fraction"
        ),
        "velocity_specificity": _ge(
            "velocity_specificity_ratio", "min_velocity_specificity_ratio"
        ),
        "phase_null_ratio": _ge(
            "phase_null_to_full_rollout_ratio",
            "min_phase_null_to_full_rollout_ratio",
        ),
        "validated_low_dimensional_fibre": {
            "metric": "validated_low_dimensional_fibre",
            "value": low_dimensional_fibre_supported,
            "operator": "is",
            "threshold": True,
            "passed": low_dimensional_fibre_supported,
            "selected_dimension": int(model.fibre_model.n_components),
            "candidate_cap": fibre_diagnostics.get("candidate_cap"),
            "best_at_cap": fibre_diagnostics.get("best_at_cap"),
            "clear_plateau_detected": fibre_diagnostics.get(
                "clear_plateau_detected"
            ),
        },
    }
    exact_passed = all(check["passed"] for check in exact_checks.values())
    full_passed = all(check["passed"] for check in full_checks.values())
    if exact_passed:
        classification = "exact_2d_controlled_torus"
        permitted_claim = (
            "Conditional on the analyzed navigation source, the held-out source "
            "quality gate supports an exact two-dimensional controlled torus model."
        )
        downstream_scope = "exact shared navigation torus may be tested"
    elif full_passed:
        classification = "toroidal_base_plus_neural_fibre"
        permitted_claim = (
            "Conditional on the analyzed navigation source, the held-out source "
            "quality gate supports a toroidal base with a validated low-dimensional "
            "neural fibre; it does not support an exact two-dimensional neural-state torus."
        )
        downstream_scope = "only shared toroidal base / shared fibre dynamics may be tested"
    else:
        classification = "torus_like_predictive_scaffold"
        permitted_claim = (
            "Conditional on the analyzed navigation source, neither preregistered "
            "held-out reduced-model gate passed; only torus-like topology or a "
            "predictive scaffold may be discussed."
        )
        downstream_scope = "stop exact/shared-manifold claims"
    return {
        "schema_version": SCHEMA_VERSION,
        "claim_scope": CLAIM_SCOPE,
        "thresholds": limits,
        "test_metric_means": metrics,
        "exact_2d_gate": {
            "passed": exact_passed,
            "checks": exact_checks,
        },
        "toroidal_base_plus_fibre_gate": {
            "passed": full_passed,
            "checks": full_checks,
        },
        "classification": classification,
        "permitted_claim": permitted_claim,
        "downstream_scope": downstream_scope,
    }


def evaluate_non_grid_control_v3(
    model: SourceReducedModelV3,
    *,
    grid_fit: TrajectoryData,
    grid_validation: TrajectoryData,
    grid_test: TrajectoryData,
    non_grid_fit: TrajectoryData,
    non_grid_validation: TrajectoryData,
    non_grid_test: TrajectoryData,
    seed: int = 42,
    n_bootstrap: int = 2000,
    min_rollout_steps: int = 2,
) -> Dict[str, Any]:
    """Fit a size-matched non-grid control using the frozen grid-derived phase.

    The control uses the same selected decoder hyperparameters and the same phase
    law.  It asks whether those source phases explain a same-sized non-grid module
    equally well; no non-grid test trajectory is used for fitting or selection.
    """
    for split_name, grid, non_grid in (
        ("fit", grid_fit, non_grid_fit),
        ("validation", grid_validation, non_grid_validation),
        ("test", grid_test, non_grid_test),
    ):
        if grid.n_states != non_grid.n_states:
            raise ValueError(f"{split_name} grid/non-grid state counts differ")
        if not (
            np.array_equal(grid.trajectory_ids, non_grid.trajectory_ids)
            and np.array_equal(grid.time_indices, non_grid.time_indices)
            and np.allclose(grid.coords, non_grid.coords)
        ):
            raise RuntimeError(
                f"{split_name} grid/non-grid arrays do not describe identical trajectories"
            )
    non_grid_indices = np.arange(non_grid_fit.n_units, dtype=np.int64)
    standardizer = AmbientStandardizer.fit(non_grid_fit, non_grid_indices)
    combined_grid = combine_trajectory_data(grid_fit, grid_validation)
    combined_non_grid = combine_trajectory_data(non_grid_fit, non_grid_validation)
    theta_combined = model.phase(combined_grid.states)
    ambient_combined = standardizer.transform(combined_non_grid.states)
    context_combined = decoder_context_per_state(
        combined_non_grid, model.selected_decoder.context_mode
    )
    decoder = FourierDecoderV3.fit(
        theta_combined,
        ambient_combined,
        context_combined,
        order=int(model.selected_decoder.feature_map.order),
        alpha=float(model.selected_decoder.alpha),
        context_mode=model.selected_decoder.context_mode,
    )
    residual_combined = ambient_combined - decoder.predict(
        theta_combined, context_combined
    )
    fibre_dimension = min(
        int(model.fibre_model.n_components),
        residual_combined.shape[1],
        residual_combined.shape[0] - 1,
    )
    fibre = fit_final_fibre_pca(
        residual_combined,
        n_components=fibre_dimension,
        seed=int(seed) + 1,
        selection_diagnostics={
            "control": "matched_non_grid_module",
            "selected_from_grid_source": int(model.fibre_model.n_components),
            "low_dimensional_fibre_supported": False,
        },
    )
    xi_combined = fibre.encode(residual_combined)
    dynamics = None
    if fibre_dimension > 0:
        source_dynamics = model.fibre_dynamics
        if source_dynamics is None:
            raise RuntimeError("Grid model has non-zero fibre but no dynamics")
        dynamics = FibreDynamicsV3.fit(
            xi_combined,
            theta_combined,
            combined_non_grid,
            kind=source_dynamics.kind,
            alpha=source_dynamics.alpha,
            degree=source_dynamics.degree,
            n_clusters=max(2, source_dynamics.n_clusters or 8),
            raw_states=combined_non_grid.states,
            active_max_units=(
                16
                if source_dynamics.active_unit_indices is None
                else int(source_dynamics.active_unit_indices.size)
            ),
            active_max_regimes=max(64, len(source_dynamics.active_signatures)),
            active_min_samples=16,
            max_polynomial_features=int(
                source_dynamics.diagnostics.get("max_polynomial_features", 2048)
            ),
            seed=int(seed) + 2,
        )

    theta_test = model.phase(grid_test.states)
    ambient_test = standardizer.transform(non_grid_test.states)
    context_test = decoder_context_per_state(
        non_grid_test, decoder.context_mode
    )
    base_test = decoder.predict(theta_test, context_test)
    residual_test = ambient_test - base_test
    xi_test = fibre.encode(residual_test)
    full_test = base_test + fibre.reconstruct(residual_test)
    rows: Dict[int, Dict[str, Any]] = {
        int(trajectory_id): {
            "trajectory_id": int(trajectory_id),
            "claim_scope": CLAIM_SCOPE,
        }
        for trajectory_id in non_grid_test.unique_trajectories
    }
    for trajectory_id in non_grid_test.unique_trajectories:
        state_rows = non_grid_test.trajectory_state_indices(int(trajectory_id))
        if state_rows.size:
            rows[int(trajectory_id)]["non_grid_surface_mfe_base"] = _relative_error(
                ambient_test[state_rows], base_test[state_rows]
            )
            rows[int(trajectory_id)][
                "non_grid_surface_mfe_base_plus_fibre"
            ] = _relative_error(ambient_test[state_rows], full_test[state_rows])
    rollout_rows = _full_rollout_rows(
        decoder=decoder,
        phase_model=model.phase_dynamics,
        fibre_model=fibre,
        fibre_dynamics=dynamics,
        standardizer=standardizer,
        theta_actual=theta_test,
        xi_actual=xi_test,
        ambient_actual=ambient_test,
        data=non_grid_test,
        min_rollout_steps=min_rollout_steps,
    )
    _assign_segment_metric(
        rows,
        rollout_rows,
        "rollout_nmte",
        "non_grid_rollout_nmte_base_plus_fibre",
    )
    trajectory_rows = [rows[key] for key in sorted(rows)]
    return {
        "schema_version": SCHEMA_VERSION,
        "claim_scope": CLAIM_SCOPE,
        "control": "matched_non_grid_module",
        "test_used_for_fit_or_selection": False,
        "n_units": non_grid_test.n_units,
        "trajectory_metrics": trajectory_rows,
        "summary": summarize_trajectory_metrics(
            trajectory_rows,
            n_bootstrap=int(n_bootstrap),
            seed=int(seed),
        ),
    }


def flatten_test_summary_for_seed(
    test_evaluation: Mapping[str, Any],
    quality_gates: Mapping[str, Any],
    *,
    seed: int,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "claim_scope": CLAIM_SCOPE,
        "source_seed": int(seed),
        "classification": quality_gates.get("classification"),
        "exact_2d_gate_passed": bool(
            quality_gates.get("exact_2d_gate", {}).get("passed", False)
        ),
        "base_plus_fibre_gate_passed": bool(
            quality_gates.get("toroidal_base_plus_fibre_gate", {}).get(
                "passed", False
            )
        ),
    }
    for metric, summary in test_evaluation.get("summary", {}).get("metrics", {}).items():
        row[metric] = summary.get("mean")
        row[f"{metric}_ci_low"] = summary.get("ci_low")
        row[f"{metric}_ci_high"] = summary.get("ci_high")
    return row
