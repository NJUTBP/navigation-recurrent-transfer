"""Source-only toroidal decoders and controlled phase dynamics.

The objects in this module are deliberately fitted only on navigation data.
They provide three complementary models:

* an explicit Fourier decoder ``h = W(theta)`` for the toroidal base;
* an input-conditioned decoder ``h = W(theta, u_prev)`` used to diagnose
  velocity-dependent neural fibres;
* a structured phase map ``theta[t+1] = theta[t] + B u[t] + g(theta[t],u[t])``.

The phase map searches only over small unimodular integer changes of the two
cohomology generators.  This respects the fact that persistent-cohomology
circle coordinates are identifiable only up to a ``GL(2,Z)`` basis change,
axis sign/exchange and phase offsets.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupShuffleSplit

_EPS = 1e-12


def _as_2d(value: np.ndarray, name: str, n_cols: Optional[int] = None) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2-D; got {arr.shape}")
    if arr.shape[0] < 1 or arr.shape[1] < 1:
        raise ValueError(f"{name} is empty: {arr.shape}")
    if n_cols is not None and arr.shape[1] != int(n_cols):
        raise ValueError(f"{name} must have {n_cols} columns; got {arr.shape}")
    if not np.isfinite(arr).all():
        raise FloatingPointError(f"{name} contains NaN or Inf")
    return arr


def _as_1d(value: np.ndarray, name: str, n: Optional[int] = None) -> np.ndarray:
    arr = np.asarray(value).reshape(-1)
    if n is not None and arr.shape[0] != int(n):
        raise ValueError(f"{name} has {arr.shape[0]} rows; expected {n}")
    if np.issubdtype(arr.dtype, np.number) and not np.isfinite(arr).all():
        raise FloatingPointError(f"{name} contains NaN or Inf")
    return arr


def wrap_angles(value: np.ndarray) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    return (arr + np.pi) % (2.0 * np.pi) - np.pi


def angles_from_features(features: np.ndarray) -> np.ndarray:
    features = _as_2d(features, "circular_features", 4)
    return np.column_stack([
        np.arctan2(features[:, 1], features[:, 0]),
        np.arctan2(features[:, 3], features[:, 2]),
    ])


def features_from_angles(theta: np.ndarray) -> np.ndarray:
    theta = _as_2d(theta, "theta", 2)
    return np.column_stack([
        np.cos(theta[:, 0]), np.sin(theta[:, 0]),
        np.cos(theta[:, 1]), np.sin(theta[:, 1]),
    ])


def normalize_circular_features(features: np.ndarray) -> np.ndarray:
    out = _as_2d(features, "circular_features", 4).copy()
    for start in (0, 2):
        radius = np.linalg.norm(out[:, start:start + 2], axis=1, keepdims=True)
        bad = radius[:, 0] <= _EPS
        radius[bad] = 1.0
        out[:, start:start + 2] /= radius
        if np.any(bad):
            out[bad, start] = 1.0
            out[bad, start + 1] = 0.0
    return out


def _group_split(
    groups: np.ndarray,
    *,
    validation_fraction: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    groups = _as_1d(groups, "groups")
    unique = np.unique(groups)
    if unique.size < 2:
        n = groups.size
        if n < 2:
            raise ValueError("At least two rows are required for validation")
        rng = np.random.default_rng(int(seed))
        order = rng.permutation(n)
        n_val = max(1, min(n - 1, int(round(n * validation_fraction))))
        return np.sort(order[n_val:]), np.sort(order[:n_val])
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=float(validation_fraction),
        random_state=int(seed),
    )
    train, validation = next(splitter.split(np.zeros(groups.size), groups=groups))
    return np.asarray(train, dtype=int), np.asarray(validation, dtype=int)


def _rmse_wrapped(error: np.ndarray) -> float:
    return float(np.sqrt(np.mean(wrap_angles(error) ** 2)))


def _canonical_wavevectors(order: int) -> np.ndarray:
    """Return one representative of each +/- Fourier wavevector pair."""
    order = int(order)
    if order < 1:
        raise ValueError("Fourier order must be >= 1")
    vectors: List[Tuple[int, int]] = []
    for k1 in range(-order, order + 1):
        for k2 in range(-order, order + 1):
            if k1 == 0 and k2 == 0:
                continue
            if max(abs(k1), abs(k2)) > order:
                continue
            if k1 > 0 or (k1 == 0 and k2 > 0):
                vectors.append((k1, k2))
    return np.asarray(vectors, dtype=np.int64)


@dataclass(frozen=True)
class FourierFeatureMap:
    order: int
    wavevectors: np.ndarray

    @classmethod
    def create(cls, order: int) -> "FourierFeatureMap":
        return cls(order=int(order), wavevectors=_canonical_wavevectors(int(order)))

    @property
    def n_features(self) -> int:
        return int(2 * self.wavevectors.shape[0])

    def transform(self, theta: np.ndarray) -> np.ndarray:
        theta = _as_2d(theta, "theta", 2)
        phase = theta @ self.wavevectors.T
        return np.column_stack([np.cos(phase), np.sin(phase)])

    def jacobian(self, theta: np.ndarray) -> np.ndarray:
        """Derivative of Fourier features with respect to theta.

        Returns
        -------
        ndarray, shape [N, n_features, 2]
        """
        theta = _as_2d(theta, "theta", 2)
        phase = theta @ self.wavevectors.T
        k = self.wavevectors.astype(np.float64)
        cos_derivative = -np.sin(phase)[:, :, None] * k[None, :, :]
        sin_derivative = np.cos(phase)[:, :, None] * k[None, :, :]
        return np.concatenate([cos_derivative, sin_derivative], axis=1)


@dataclass
class FourierRidgeDecoder:
    """Periodic ridge decoder with optional previous-velocity conditioning."""

    feature_map: FourierFeatureMap
    alpha: float
    ridge: Ridge
    mode: str
    context_mean: Optional[np.ndarray]
    context_std: Optional[np.ndarray]
    validation_mse: float

    @staticmethod
    def _context_features(
        base: np.ndarray,
        context: Optional[np.ndarray],
        *,
        mode: str,
        context_mean: Optional[np.ndarray],
        context_std: Optional[np.ndarray],
    ) -> np.ndarray:
        if mode == "base":
            return base
        if mode != "previous_velocity":
            raise ValueError(f"Unknown decoder mode: {mode}")
        if context is None:
            raise ValueError("previous_velocity decoder requires context")
        context = _as_2d(context, "previous_velocity", 2)
        if context.shape[0] != base.shape[0]:
            raise ValueError("context and theta row counts differ")
        mean = np.zeros(2) if context_mean is None else np.asarray(context_mean)
        std = np.ones(2) if context_std is None else np.asarray(context_std)
        scaled = (context - mean) / np.maximum(std, 1e-8)
        interactions = np.concatenate(
            [base * scaled[:, j:j + 1] for j in range(2)], axis=1
        )
        return np.column_stack([base, scaled, scaled ** 2, interactions])

    @classmethod
    def fit(
        cls,
        theta: np.ndarray,
        ambient: np.ndarray,
        groups: np.ndarray,
        *,
        previous_velocity: Optional[np.ndarray],
        mode: str,
        orders: Sequence[int],
        alphas: Sequence[float],
        validation_fraction: float,
        seed: int,
    ) -> Tuple["FourierRidgeDecoder", Dict[str, Any]]:
        theta = _as_2d(theta, "theta", 2)
        ambient = _as_2d(ambient, "ambient")
        groups = _as_1d(groups, "groups", theta.shape[0])
        if ambient.shape[0] != theta.shape[0]:
            raise ValueError("theta and ambient row counts differ")
        context = None
        context_mean = None
        context_std = None
        if mode == "previous_velocity":
            context = _as_2d(previous_velocity, "previous_velocity", 2)
            if context.shape[0] != theta.shape[0]:
                raise ValueError("previous_velocity row count differs")
        train, validation = _group_split(
            groups, validation_fraction=validation_fraction, seed=seed
        )
        if context is not None:
            context_mean = context[train].mean(axis=0)
            context_std = context[train].std(axis=0)
            context_std = np.where(context_std > 1e-8, context_std, 1.0)

        candidates: List[Dict[str, Any]] = []
        best: Optional[Tuple[float, int, float]] = None
        for order in orders:
            fmap = FourierFeatureMap.create(int(order))
            base = fmap.transform(theta)
            x = cls._context_features(
                base,
                context,
                mode=mode,
                context_mean=context_mean,
                context_std=context_std,
            )
            for alpha in alphas:
                model = Ridge(alpha=float(alpha), fit_intercept=True)
                model.fit(x[train], ambient[train])
                prediction = model.predict(x[validation])
                mse = float(np.mean((prediction - ambient[validation]) ** 2))
                row = {
                    "mode": mode,
                    "order": int(order),
                    "alpha": float(alpha),
                    "validation_mse": mse,
                }
                candidates.append(row)
                key = (mse, int(order), float(alpha))
                if best is None or key < best:
                    best = key
        if best is None:
            raise RuntimeError("No Fourier decoder candidate was fitted")
        validation_mse, order, alpha = best
        fmap = FourierFeatureMap.create(order)
        base = fmap.transform(theta)
        # Re-estimate context statistics on all source data for the final model.
        if context is not None:
            context_mean = context.mean(axis=0)
            context_std = context.std(axis=0)
            context_std = np.where(context_std > 1e-8, context_std, 1.0)
        x = cls._context_features(
            base,
            context,
            mode=mode,
            context_mean=context_mean,
            context_std=context_std,
        )
        ridge = Ridge(alpha=alpha, fit_intercept=True)
        ridge.fit(x, ambient)
        fitted = cls(
            feature_map=fmap,
            alpha=float(alpha),
            ridge=ridge,
            mode=mode,
            context_mean=(None if context_mean is None else np.asarray(context_mean)),
            context_std=(None if context_std is None else np.asarray(context_std)),
            validation_mse=float(validation_mse),
        )
        return fitted, {
            "mode": mode,
            "selected_order": int(order),
            "selected_alpha": float(alpha),
            "validation_mse": float(validation_mse),
            "n_train": int(train.size),
            "n_validation": int(validation.size),
            "train_group_ids": np.unique(groups[train]).tolist(),
            "validation_group_ids": np.unique(groups[validation]).tolist(),
            "candidates": candidates,
        }

    def _design(self, theta: np.ndarray, previous_velocity: Optional[np.ndarray]) -> np.ndarray:
        theta = _as_2d(theta, "theta", 2)
        base = self.feature_map.transform(theta)
        return self._context_features(
            base,
            previous_velocity,
            mode=self.mode,
            context_mean=self.context_mean,
            context_std=self.context_std,
        )

    def predict(self, theta: np.ndarray, previous_velocity: Optional[np.ndarray] = None) -> np.ndarray:
        return np.asarray(
            self.ridge.predict(self._design(theta, previous_velocity)),
            dtype=np.float64,
        )

    def jacobian(
        self,
        theta: np.ndarray,
        previous_velocity: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Return d h / d theta with shape [N, ambient_dim, 2]."""
        theta = _as_2d(theta, "theta", 2)
        base = self.feature_map.transform(theta)
        dbase = self.feature_map.jacobian(theta)
        if self.mode == "base":
            ddesign = dbase
        else:
            if previous_velocity is None:
                raise ValueError("previous_velocity is required for conditioned Jacobian")
            context = _as_2d(previous_velocity, "previous_velocity", 2)
            scaled = (context - self.context_mean) / np.maximum(self.context_std, 1e-8)
            zeros_context = np.zeros((theta.shape[0], 4, 2), dtype=np.float64)
            interactions = np.concatenate(
                [dbase * scaled[:, j:j + 1, None] for j in range(2)], axis=1
            )
            ddesign = np.concatenate([dbase, zeros_context, interactions], axis=1)
        # Ridge coef_ is [ambient_dim, design_dim].
        return np.einsum("od,ndk->nok", np.asarray(self.ridge.coef_), ddesign)


@dataclass
class TorusDecoderSuite:
    base: FourierRidgeDecoder
    conditioned: Optional[FourierRidgeDecoder]
    selected_mode: str
    diagnostics: Dict[str, Any]

    @classmethod
    def fit(
        cls,
        theta: np.ndarray,
        ambient: np.ndarray,
        groups: np.ndarray,
        previous_velocity: np.ndarray,
        *,
        orders: Sequence[int] = (1, 2, 3, 4),
        alphas: Sequence[float] = (1e-6, 1e-4, 1e-2, 1.0, 100.0),
        validation_fraction: float = 0.2,
        conditioned_min_relative_improvement: float = 0.05,
        seed: int = 42,
    ) -> "TorusDecoderSuite":
        base, base_diag = FourierRidgeDecoder.fit(
            theta,
            ambient,
            groups,
            previous_velocity=None,
            mode="base",
            orders=orders,
            alphas=alphas,
            validation_fraction=validation_fraction,
            seed=seed,
        )
        conditioned, cond_diag = FourierRidgeDecoder.fit(
            theta,
            ambient,
            groups,
            previous_velocity=previous_velocity,
            mode="previous_velocity",
            orders=orders,
            alphas=alphas,
            validation_fraction=validation_fraction,
            # The base and conditioned models must be compared on the same
            # held-out trajectories.  Using a different split for the richer
            # model can turn split difficulty into an apparent fibre benefit.
            seed=seed,
        )
        relative_improvement = (
            (base.validation_mse - conditioned.validation_mse)
            / max(base.validation_mse, _EPS)
        )
        selected_mode = (
            "previous_velocity"
            if relative_improvement >= float(conditioned_min_relative_improvement)
            else "base"
        )
        diagnostics = {
            "base": base_diag,
            "conditioned": cond_diag,
            "conditioned_relative_improvement": float(relative_improvement),
            "conditioned_min_relative_improvement": float(
                conditioned_min_relative_improvement
            ),
            "selected_mode": selected_mode,
        }
        return cls(
            base=base,
            conditioned=conditioned,
            selected_mode=selected_mode,
            diagnostics=diagnostics,
        )

    def model(self, mode: str = "selected") -> FourierRidgeDecoder:
        resolved = self.selected_mode if mode == "selected" else mode
        if resolved == "base":
            return self.base
        if resolved == "previous_velocity" and self.conditioned is not None:
            return self.conditioned
        raise ValueError(f"Decoder mode {mode!r} is unavailable")

    def predict(
        self,
        theta: np.ndarray,
        previous_velocity: Optional[np.ndarray] = None,
        *,
        mode: str = "selected",
    ) -> np.ndarray:
        model = self.model(mode)
        context = previous_velocity if model.mode == "previous_velocity" else None
        return model.predict(theta, context)

    def jacobian(
        self,
        theta: np.ndarray,
        previous_velocity: Optional[np.ndarray] = None,
        *,
        mode: str = "base",
    ) -> np.ndarray:
        model = self.model(mode)
        context = previous_velocity if model.mode == "previous_velocity" else None
        return model.jacobian(theta, context)


@dataclass
class ResidualFiberModel:
    """PCA model of neural residuals normal to the toroidal base decoder."""

    pca: Optional[PCA]
    n_components: int
    diagnostics: Dict[str, Any]

    @classmethod
    def fit(
        cls,
        residual: np.ndarray,
        groups: np.ndarray,
        *,
        candidate_dims: Sequence[int] = (0, 1, 2, 4, 6, 8),
        validation_fraction: float = 0.2,
        best_tolerance: float = 0.01,
        seed: int = 42,
    ) -> "ResidualFiberModel":
        residual = _as_2d(residual, "residual")
        groups = _as_1d(groups, "groups", residual.shape[0])
        train, validation = _group_split(
            groups, validation_fraction=validation_fraction, seed=seed
        )
        max_dim = min(residual.shape[1], max(1, train.size - 1))
        dims = sorted({max(0, min(int(k), max_dim)) for k in candidate_dims})
        rows: List[Dict[str, Any]] = []
        models: Dict[int, Optional[PCA]] = {0: None}
        for k in dims:
            if k == 0:
                prediction = np.zeros_like(residual[validation])
                mse = float(np.mean((prediction - residual[validation]) ** 2))
            else:
                pca = PCA(n_components=k, random_state=int(seed))
                pca.fit(residual[train])
                prediction = pca.inverse_transform(pca.transform(residual[validation]))
                mse = float(np.mean((prediction - residual[validation]) ** 2))
                models[k] = pca
            rows.append({"n_components": int(k), "validation_mse": mse})
        best_mse = min(row["validation_mse"] for row in rows)
        # Select the smallest fibre dimension statistically indistinguishable at
        # this operational tolerance from the best candidate.
        eligible = [
            row["n_components"]
            for row in rows
            if row["validation_mse"] <= best_mse * (1.0 + float(best_tolerance)) + _EPS
        ]
        selected = int(min(eligible))
        final_pca: Optional[PCA]
        if selected == 0:
            final_pca = None
        else:
            final_pca = PCA(n_components=selected, random_state=int(seed))
            final_pca.fit(residual)
        return cls(
            pca=final_pca,
            n_components=selected,
            diagnostics={
                "candidate_rows": rows,
                "selected_n_components": selected,
                "best_validation_mse": float(best_mse),
                "best_tolerance": float(best_tolerance),
                "n_train": int(train.size),
                "n_validation": int(validation.size),
            },
        )

    def encode(self, residual: np.ndarray) -> np.ndarray:
        residual = _as_2d(residual, "residual")
        if self.pca is None:
            return np.empty((residual.shape[0], 0), dtype=np.float64)
        return np.asarray(self.pca.transform(residual), dtype=np.float64)

    def reconstruct(self, residual: np.ndarray) -> np.ndarray:
        residual = _as_2d(residual, "residual")
        if self.pca is None:
            return np.zeros_like(residual)
        return np.asarray(
            self.pca.inverse_transform(self.pca.transform(residual)),
            dtype=np.float64,
        )

    @property
    def basis(self) -> np.ndarray:
        if self.pca is None:
            return np.empty((0, 0), dtype=np.float64)
        return np.asarray(self.pca.components_.T, dtype=np.float64)


def generate_unimodular_matrices(max_entry: int = 2) -> List[np.ndarray]:
    max_entry = int(max_entry)
    matrices: List[np.ndarray] = []
    for a in range(-max_entry, max_entry + 1):
        for b in range(-max_entry, max_entry + 1):
            for c in range(-max_entry, max_entry + 1):
                for d in range(-max_entry, max_entry + 1):
                    det = a * d - b * c
                    if abs(det) != 1:
                        continue
                    matrices.append(np.asarray([[a, b], [c, d]], dtype=np.int64))
    matrices.sort(
        key=lambda m: (
            int(np.sum(np.abs(m))),
            0 if np.array_equal(m, np.eye(2, dtype=int)) else 1,
            tuple(m.reshape(-1).tolist()),
        )
    )
    return matrices


def _integer_inverse(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.int64).reshape(2, 2)
    det = int(round(np.linalg.det(matrix)))
    if abs(det) != 1:
        raise ValueError("matrix is not unimodular")
    inverse = det * np.asarray(
        [[matrix[1, 1], -matrix[0, 1]], [-matrix[1, 0], matrix[0, 0]]],
        dtype=np.int64,
    )
    return inverse


@dataclass
class PeriodicResidualRegressor:
    feature_map: FourierFeatureMap
    alpha: float
    ridge: Ridge
    velocity_mean: np.ndarray
    velocity_std: np.ndarray
    validation_rmse: float

    @staticmethod
    def _design(
        fmap: FourierFeatureMap,
        theta: np.ndarray,
        velocity: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
    ) -> np.ndarray:
        base = fmap.transform(theta)
        scaled = (velocity - mean) / np.maximum(std, 1e-8)
        interactions = np.concatenate(
            [base * scaled[:, j:j + 1] for j in range(2)], axis=1
        )
        return np.column_stack([base, scaled, interactions])

    @classmethod
    def fit_selected(
        cls,
        theta: np.ndarray,
        velocity: np.ndarray,
        residual_delta: np.ndarray,
        groups: np.ndarray,
        *,
        orders: Sequence[int] = (1, 2),
        alphas: Sequence[float] = (1e-6, 1e-4, 1e-2, 1.0),
        validation_fraction: float = 0.2,
        seed: int = 42,
        train_indices: Optional[np.ndarray] = None,
        validation_indices: Optional[np.ndarray] = None,
        final_residual_delta: Optional[np.ndarray] = None,
    ) -> Tuple["PeriodicResidualRegressor", Dict[str, Any]]:
        theta = _as_2d(theta, "theta", 2)
        velocity = _as_2d(velocity, "velocity", 2)
        residual_delta = _as_2d(residual_delta, "residual_delta", 2)
        groups = _as_1d(groups, "groups", theta.shape[0])
        if velocity.shape[0] != theta.shape[0] or residual_delta.shape[0] != theta.shape[0]:
            raise ValueError("theta, velocity and residual_delta row counts differ")
        if (train_indices is None) != (validation_indices is None):
            raise ValueError(
                "train_indices and validation_indices must be supplied together"
            )
        if train_indices is None:
            train, validation = _group_split(
                groups, validation_fraction=validation_fraction, seed=seed
            )
        else:
            train = np.asarray(train_indices, dtype=np.int64).reshape(-1)
            validation = np.asarray(validation_indices, dtype=np.int64).reshape(-1)
            if train.size == 0 or validation.size == 0:
                raise ValueError("Explicit train/validation indices must be non-empty")
            if train.min() < 0 or validation.min() < 0:
                raise IndexError("Explicit split indices must be non-negative")
            if train.max() >= theta.shape[0] or validation.max() >= theta.shape[0]:
                raise IndexError("Explicit split indices exceed the sample count")
            if np.intersect1d(train, validation).size:
                raise ValueError("Explicit train/validation indices overlap")
        final_target = (
            residual_delta
            if final_residual_delta is None
            else _as_2d(final_residual_delta, "final_residual_delta", 2)
        )
        if final_target.shape[0] != theta.shape[0]:
            raise ValueError("final_residual_delta row count differs from theta")
        mean = velocity[train].mean(axis=0)
        std = velocity[train].std(axis=0)
        std = np.where(std > 1e-8, std, 1.0)
        rows: List[Dict[str, Any]] = []
        best: Optional[Tuple[float, int, float]] = None
        for order in orders:
            fmap = FourierFeatureMap.create(int(order))
            x = cls._design(fmap, theta, velocity, mean, std)
            for alpha in alphas:
                ridge = Ridge(alpha=float(alpha), fit_intercept=True)
                ridge.fit(x[train], residual_delta[train])
                prediction = ridge.predict(x[validation])
                rmse = _rmse_wrapped(residual_delta[validation] - prediction)
                rows.append({
                    "order": int(order),
                    "alpha": float(alpha),
                    "validation_rmse": rmse,
                })
                key = (rmse, int(order), float(alpha))
                if best is None or key < best:
                    best = key
        if best is None:
            raise RuntimeError("No periodic residual candidate was fitted")
        rmse, order, alpha = best
        mean = velocity.mean(axis=0)
        std = velocity.std(axis=0)
        std = np.where(std > 1e-8, std, 1.0)
        fmap = FourierFeatureMap.create(order)
        x = cls._design(fmap, theta, velocity, mean, std)
        ridge = Ridge(alpha=alpha, fit_intercept=True)
        ridge.fit(x, final_target)
        return cls(
            feature_map=fmap,
            alpha=float(alpha),
            ridge=ridge,
            velocity_mean=mean,
            velocity_std=std,
            validation_rmse=float(rmse),
        ), {
            "selected_order": int(order),
            "selected_alpha": float(alpha),
            "validation_rmse": float(rmse),
            "candidates": rows,
            "n_train": int(train.size),
            "n_validation": int(validation.size),
            "explicit_split": bool(train_indices is not None),
        }

    def predict(self, theta: np.ndarray, velocity: np.ndarray) -> np.ndarray:
        theta = _as_2d(theta, "theta", 2)
        velocity = _as_2d(velocity, "velocity", 2)
        x = self._design(
            self.feature_map,
            theta,
            velocity,
            self.velocity_mean,
            self.velocity_std,
        )
        return np.asarray(self.ridge.predict(x), dtype=np.float64)


@dataclass
class StructuredPhaseMap:
    """Additive and additive-plus-residual dynamics on a two-torus."""

    basis_matrix: np.ndarray
    basis_inverse: np.ndarray
    action_matrix: np.ndarray
    residual_model: Optional[PeriodicResidualRegressor]
    diagnostics: Dict[str, Any]

    @classmethod
    def fit(
        cls,
        raw_theta: np.ndarray,
        transition_prev: np.ndarray,
        transition_next: np.ndarray,
        velocities: np.ndarray,
        trajectory_groups: np.ndarray,
        *,
        max_gl2z_entry: int = 2,
        ridge_alpha: float = 1e-7,
        validation_fraction: float = 0.2,
        fit_nonlinear_residual: bool = True,
        seed: int = 42,
    ) -> "StructuredPhaseMap":
        raw_theta = _as_2d(raw_theta, "raw_theta", 2)
        prev = _as_1d(transition_prev, "transition_prev").astype(int)
        nxt = _as_1d(transition_next, "transition_next", prev.size).astype(int)
        velocity = _as_2d(velocities, "velocities", 2)
        groups = _as_1d(trajectory_groups, "trajectory_groups", prev.size)
        if velocity.shape[0] != prev.size:
            raise ValueError("velocity and transition counts differ")
        train, validation = _group_split(
            groups, validation_fraction=validation_fraction, seed=seed
        )
        candidates: List[Dict[str, Any]] = []
        best: Optional[Tuple[float, int, Tuple[int, ...], np.ndarray]] = None
        matrices = generate_unimodular_matrices(max_gl2z_entry)
        for matrix in matrices:
            theta = wrap_angles(raw_theta @ matrix.T)
            delta = wrap_angles(theta[nxt] - theta[prev])
            ridge = Ridge(alpha=float(ridge_alpha), fit_intercept=False)
            ridge.fit(velocity[train], delta[train])
            predicted = velocity[validation] @ np.asarray(ridge.coef_).T
            rmse = _rmse_wrapped(delta[validation] - predicted)
            complexity = int(np.sum(np.abs(matrix)))
            key_tuple = tuple(int(x) for x in matrix.reshape(-1))
            candidates.append({
                "matrix": matrix.tolist(),
                "validation_rmse": rmse,
                "complexity": complexity,
            })
            key = (rmse, complexity, key_tuple, matrix)
            if best is None or key[:3] < best[:3]:
                best = key
        if best is None:
            raise RuntimeError("No GL(2,Z) phase basis candidate was fitted")
        _, _, _, matrix = best
        theta = wrap_angles(raw_theta @ matrix.T)
        delta = wrap_angles(theta[nxt] - theta[prev])
        action = Ridge(alpha=float(ridge_alpha), fit_intercept=False)
        action.fit(velocity, delta)
        action_matrix = np.asarray(action.coef_, dtype=np.float64)

        # Validation diagnostics are recomputed with the selected basis using a
        # group-held-out action fit, avoiding optimistic all-data scores.
        action_val = Ridge(alpha=float(ridge_alpha), fit_intercept=False)
        action_val.fit(velocity[train], delta[train])
        additive_prediction = velocity[validation] @ np.asarray(action_val.coef_).T
        additive_rmse = _rmse_wrapped(delta[validation] - additive_prediction)
        zero_rmse = _rmse_wrapped(delta[validation])
        phase_step_rms = float(np.sqrt(np.mean(delta[validation] ** 2)))

        rng = np.random.default_rng(int(seed) + 91)
        shuffled_velocity = velocity[validation][rng.permutation(validation.size)]
        shuffled_prediction = shuffled_velocity @ np.asarray(action_val.coef_).T
        velocity_shuffle_rmse = _rmse_wrapped(delta[validation] - shuffled_prediction)
        sign_flip_rmse = _rmse_wrapped(delta[validation] + additive_prediction)
        rotation = np.asarray([[0.0, -1.0], [1.0, 0.0]])
        rotated_prediction = (velocity[validation] @ rotation.T) @ np.asarray(action_val.coef_).T
        rotation_rmse = _rmse_wrapped(delta[validation] - rotated_prediction)

        # A state-only periodic null predicts phase increments without velocity.
        state_fmap = FourierFeatureMap.create(2)
        state_x = state_fmap.transform(theta[prev])
        state_model = Ridge(alpha=1e-3, fit_intercept=True)
        state_model.fit(state_x[train], delta[train])
        state_only_prediction = state_model.predict(state_x[validation])
        state_only_rmse = _rmse_wrapped(delta[validation] - state_only_prediction)

        residual_model = None
        residual_diag = None
        if fit_nonlinear_residual:
            # Select the nonlinear correction without leaking the validation
            # trajectories through the additive baseline.  Hyperparameters are
            # chosen against residuals from the train-only additive fit, then
            # the final correction is refitted against the all-source additive
            # model used at inference time.
            train_action_matrix = np.asarray(action_val.coef_, dtype=np.float64)
            residual_target_cv = wrap_angles(
                delta - velocity @ train_action_matrix.T
            )
            residual_target_final = wrap_angles(
                delta - velocity @ action_matrix.T
            )
            residual_model, residual_diag = PeriodicResidualRegressor.fit_selected(
                theta[prev],
                velocity,
                residual_target_cv,
                groups,
                validation_fraction=validation_fraction,
                seed=seed + 7,
                train_indices=train,
                validation_indices=validation,
                final_residual_delta=residual_target_final,
            )

        inverse = _integer_inverse(matrix)
        diagnostics = {
            "basis_matrix": matrix.tolist(),
            "basis_inverse": inverse.tolist(),
            "basis_candidates": candidates,
            "action_matrix": action_matrix.tolist(),
            "heldout_additive_rmse": additive_rmse,
            "heldout_zero_delta_rmse": zero_rmse,
            "heldout_phase_step_rms": phase_step_rms,
            "heldout_velocity_shuffle_rmse": velocity_shuffle_rmse,
            "heldout_sign_flip_rmse": sign_flip_rmse,
            "heldout_rotation_90_rmse": rotation_rmse,
            "heldout_state_only_rmse": state_only_rmse,
            "velocity_specificity_ratio": float(
                velocity_shuffle_rmse / max(additive_rmse, _EPS)
            ),
            "state_only_specificity_ratio": float(
                state_only_rmse / max(additive_rmse, _EPS)
            ),
            "normalized_additive_rmse": float(
                additive_rmse / max(phase_step_rms, _EPS)
            ),
            "heldout_nonlinear_rmse": (
                None
                if residual_diag is None
                else float(residual_diag["validation_rmse"])
            ),
            "nonlinear_relative_improvement": (
                None
                if residual_diag is None
                else float(
                    (additive_rmse - float(residual_diag["validation_rmse"]))
                    / max(additive_rmse, _EPS)
                )
            ),
            "residual_model": residual_diag,
            "n_train": int(train.size),
            "n_validation": int(validation.size),
        }
        return cls(
            basis_matrix=np.asarray(matrix, dtype=np.int64),
            basis_inverse=np.asarray(inverse, dtype=np.int64),
            action_matrix=action_matrix,
            residual_model=residual_model,
            diagnostics=diagnostics,
        )

    def align_angles(self, raw_theta: np.ndarray) -> np.ndarray:
        raw_theta = _as_2d(raw_theta, "raw_theta", 2)
        return wrap_angles(raw_theta @ self.basis_matrix.T)

    def unalign_angles(self, theta: np.ndarray) -> np.ndarray:
        theta = _as_2d(theta, "theta", 2)
        return wrap_angles(theta @ self.basis_inverse.T)

    def align_features(self, raw_features: np.ndarray) -> np.ndarray:
        raw_features = normalize_circular_features(raw_features)
        return features_from_angles(self.align_angles(angles_from_features(raw_features)))

    def predict_delta(
        self,
        theta: np.ndarray,
        velocity: np.ndarray,
        *,
        mode: str = "nonlinear",
    ) -> np.ndarray:
        theta = _as_2d(theta, "theta", 2)
        velocity = _as_2d(velocity, "velocity", 2)
        if theta.shape[0] != velocity.shape[0]:
            raise ValueError("theta and velocity row counts differ")
        delta = velocity @ self.action_matrix.T
        if mode == "nonlinear" and self.residual_model is not None:
            delta = delta + self.residual_model.predict(theta, velocity)
        elif mode not in {"additive", "nonlinear"}:
            raise ValueError("mode must be 'additive' or 'nonlinear'")
        return wrap_angles(delta)

    def predict_features(
        self,
        latent: np.ndarray,
        velocity: np.ndarray,
        *,
        mode: str = "nonlinear",
    ) -> np.ndarray:
        latent = normalize_circular_features(latent)
        theta = angles_from_features(latent)
        next_theta = wrap_angles(theta + self.predict_delta(theta, velocity, mode=mode))
        return features_from_angles(next_theta)

    def rollout_features(
        self,
        initial: np.ndarray,
        velocities: np.ndarray,
        *,
        mode: str = "nonlinear",
    ) -> np.ndarray:
        initial = normalize_circular_features(np.asarray(initial).reshape(1, 4))
        velocities = _as_2d(velocities, "velocities", 2)
        result = [initial[0].copy()]
        current = initial
        for velocity in velocities:
            current = self.predict_features(
                current, velocity.reshape(1, 2), mode=mode
            )
            result.append(current[0].copy())
        return np.asarray(result, dtype=np.float64)
