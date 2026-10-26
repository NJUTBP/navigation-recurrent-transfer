"""Gardner-style toroidal-manifold analysis for population activity.

The implementation separates two operations that are often conflated:

1. **Visualisation**: unit-wise standardisation -> PCA -> 3-D UMAP.
2. **Topology quantification**: unit-wise standardisation -> six PCs ->
   density-aware landmark sampling -> fuzzy-neighbourhood distance ->
   persistent cohomology.

This follows the analysis logic in Gardner et al. (2022) while remaining
usable for continuous RNN activations rather than spike trains.  Temporal
Gaussian smoothing is therefore not imposed here: neural spike counts need
smoothing to estimate firing rates, whereas deterministic hidden states are
already continuous-valued.  Callers can smooth within trajectory boundaries
before using this module when their activations are noisy.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from joblib import Parallel, delayed, parallel_config
from scipy import sparse
from scipy.sparse.linalg import lsmr
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

try:  # numba is optional; the Python fallback is correct but slower.
    from numba import njit
except Exception:  # pragma: no cover - depends on optional environment
    njit = None


_EPS = 1e-12


@dataclass(frozen=True)
class ModuleSelection:
    """Description of the grid-cell module chosen for torus analysis."""

    unit_indices: np.ndarray
    candidate_indices: np.ndarray
    excluded_head_direction_indices: np.ndarray
    score_key: str
    score_threshold: float
    median_score: float
    median_spacing: float
    mean_orientation_deg: float
    n_units: int
    n_candidates: int
    sufficient_for_topology: bool
    note: str

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        for key in (
            "unit_indices",
            "candidate_indices",
            "excluded_head_direction_indices",
        ):
            out[key] = np.asarray(out[key], dtype=int).tolist()
        return out


def _require_2d_finite(x: np.ndarray, name: str = "activations") -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{name} must have shape [samples, units], got {arr.shape}")
    if arr.shape[0] < 3 or arr.shape[1] < 2:
        raise ValueError(f"{name} is too small for manifold analysis: {arr.shape}")
    return arr


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def standardize_units(
    activations: np.ndarray,
    min_unit_std: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Remove invalid/constant units and z-score each retained unit.

    Returns
    -------
    z : ndarray
        Standardised activity with finite rows and non-constant units.
    row_mask : ndarray[bool]
        Rows retained from the input.
    unit_mask : ndarray[bool]
        Units retained from the input.
    mean, std : ndarray
        Per-unit moments for the retained units.
    """
    x = _require_2d_finite(activations)
    row_mask = np.isfinite(x).all(axis=1)
    x = x[row_mask]
    if x.shape[0] < 3:
        raise ValueError("Fewer than three finite population vectors remain")

    unit_mean_all = x.mean(axis=0)
    unit_std_all = x.std(axis=0)
    unit_mask = (
        np.isfinite(unit_mean_all)
        & np.isfinite(unit_std_all)
        & (unit_std_all > min_unit_std)
    )
    if int(unit_mask.sum()) < 2:
        raise ValueError("Fewer than two non-constant units remain")

    mean = unit_mean_all[unit_mask]
    std = unit_std_all[unit_mask]
    z = (x[:, unit_mask] - mean[None, :]) / std[None, :]
    return z, row_mask, unit_mask, mean, std


def population_activity_score(
    activations: np.ndarray,
    mode: str = "auto",
) -> np.ndarray:
    """Score each population vector for active-sample selection.

    Gardner et al. ranked non-negative firing-rate vectors by mean population
    firing rate.  For signed model states, the mean can cancel genuine
    activity; ``mode='auto'`` therefore uses RMS magnitude when appreciable
    negative values are present.
    """
    x = _require_2d_finite(activations)
    mode_norm = mode.lower()
    if mode_norm == "auto":
        finite = x[np.isfinite(x)]
        nonnegative = finite.size == 0 or np.quantile(finite, 0.001) >= -1e-8
        mode_norm = "mean" if nonnegative else "rms"
    if mode_norm == "mean":
        return np.nanmean(x, axis=1)
    if mode_norm in {"rms", "l2"}:
        return np.sqrt(np.nanmean(np.square(x), axis=1))
    if mode_norm in {"mean_abs", "abs"}:
        return np.nanmean(np.abs(x), axis=1)
    raise ValueError(f"Unsupported activity score mode: {mode!r}")


def select_active_population_vectors(
    activations: np.ndarray,
    max_points: Optional[int] = 15_000,
    mode: str = "auto",
) -> np.ndarray:
    """Return time indices of the most active population vectors.

    The returned indices are sorted, preserving temporal order for downstream
    inspection.  When ``max_points`` is ``None`` or exceeds the sample count,
    all points are retained.
    """
    x = _require_2d_finite(activations)
    n = x.shape[0]
    if max_points is None or max_points >= n:
        return np.arange(n, dtype=int)
    if max_points < 3:
        raise ValueError("max_points must be at least 3")
    scores = population_activity_score(x, mode=mode)
    scores = np.nan_to_num(scores, nan=-np.inf)
    chosen = np.argpartition(scores, n - int(max_points))[n - int(max_points):]
    return np.sort(chosen.astype(int))


def pca_population_activity(
    activations: np.ndarray,
    n_components: int = 6,
    min_unit_std: float = 1e-8,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Unit-wise z-score followed by PCA.

    Unlike the former project implementation, this standardises *each unit*
    independently before PCA instead of dividing the entire matrix by one
    global scalar.
    """
    z, row_mask, unit_mask, mean, std = standardize_units(
        activations, min_unit_std=min_unit_std,
    )
    n_actual = min(int(n_components), z.shape[1], z.shape[0] - 1)
    if n_actual < 2:
        raise ValueError("At least two principal components are required")
    pca = PCA(n_components=n_actual, random_state=random_state)
    projection = pca.fit_transform(z)
    return {
        "projection": projection,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "pca": pca,
        "standardized": z,
        "row_mask": row_mask,
        "unit_mask": unit_mask,
        "unit_mean": mean,
        "unit_std": std,
    }


# -------------------------------------------------------------------------
# Fuzzy-neighbourhood graph and density-aware landmark sampling
# -------------------------------------------------------------------------


def _smooth_knn_sigmas_python(distances: np.ndarray, target: float) -> np.ndarray:
    n = distances.shape[0]
    sigmas = np.empty(n, dtype=np.float64)
    mean_distance = float(np.mean(distances))
    for i in range(n):
        row = distances[i]
        lo, hi, mid = 0.0, np.inf, 1.0
        for _ in range(64):
            # The first neighbour is normally the point itself and is omitted.
            psum = float(np.exp(-row[1:] / max(mid, _EPS)).sum())
            if abs(psum - target) < 1e-5:
                break
            if psum > target:
                hi = mid
                mid = (lo + hi) / 2.0
            else:
                lo = mid
                mid = mid * 2.0 if np.isinf(hi) else (lo + hi) / 2.0
        floor = 1e-3 * (float(np.mean(row)) if np.mean(row) > 0 else mean_distance)
        sigmas[i] = max(mid, floor, _EPS)
    return sigmas


if njit is not None:  # pragma: no branch - definition depends on environment

    @njit(cache=True)
    def _smooth_knn_sigmas_numba(distances: np.ndarray, target: float) -> np.ndarray:
        n = distances.shape[0]
        k = distances.shape[1]
        sigmas = np.empty(n, dtype=np.float64)
        mean_distance = np.mean(distances)
        for i in range(n):
            lo = 0.0
            hi = np.inf
            mid = 1.0
            for _ in range(64):
                psum = 0.0
                for j in range(1, k):
                    psum += np.exp(-distances[i, j] / max(mid, 1e-12))
                if abs(psum - target) < 1e-5:
                    break
                if psum > target:
                    hi = mid
                    mid = (lo + hi) / 2.0
                else:
                    lo = mid
                    if np.isinf(hi):
                        mid *= 2.0
                    else:
                        mid = (lo + hi) / 2.0
            row_mean = np.mean(distances[i])
            floor = 1e-3 * (row_mean if row_mean > 0 else mean_distance)
            if mid < floor:
                mid = floor
            if mid < 1e-12:
                mid = 1e-12
            sigmas[i] = mid
        return sigmas


def _solve_sigmas(distances: np.ndarray) -> np.ndarray:
    target = float(np.log2(max(2, distances.shape[1])))
    if njit is not None:
        return _smooth_knn_sigmas_numba(np.asarray(distances, dtype=np.float64), target)
    return _smooth_knn_sigmas_python(distances, target)


def _nearest_neighbours(
    data: np.ndarray,
    n_neighbors: int,
    metric: str,
    n_jobs: int = -1,
) -> Tuple[np.ndarray, np.ndarray]:
    x = _require_2d_finite(data, name="data")
    k = max(2, min(int(n_neighbors), x.shape[0]))
    model = NearestNeighbors(
        n_neighbors=k,
        metric=metric,
        algorithm="brute" if metric in {"cosine", "correlation"} else "auto",
        n_jobs=n_jobs,
    )
    distances, indices = model.fit(x).kneighbors(x, return_distance=True)

    # Ensure self is the first neighbour, matching the reference construction.
    for i in range(x.shape[0]):
        matches = np.flatnonzero(indices[i] == i)
        if matches.size == 0:
            indices[i, -1] = i
            distances[i, -1] = 0.0
            order = np.argsort(distances[i])
            indices[i] = indices[i, order]
            distances[i] = distances[i, order]
        elif matches[0] != 0:
            j = int(matches[0])
            indices[i, 0], indices[i, j] = indices[i, j], indices[i, 0]
            distances[i, 0], distances[i, j] = distances[i, j], distances[i, 0]
    return indices.astype(np.int64), distances.astype(np.float64)


def fuzzy_neighbourhood_graph(
    data: np.ndarray,
    n_neighbors: int,
    metric: str = "cosine",
    n_jobs: int = -1,
) -> sparse.csr_matrix:
    """Build a symmetric fuzzy k-neighbourhood graph.

    This is the neighbourhood-normalisation stage used by UMAP/UMA.  Local
    distance scales are solved so each point has comparable total membership;
    directed memberships are then combined by fuzzy union.
    """
    indices, distances = _nearest_neighbours(
        data, n_neighbors=n_neighbors, metric=metric, n_jobs=n_jobs,
    )
    sigmas = _solve_sigmas(distances)
    n, k = indices.shape

    rows = np.repeat(np.arange(n, dtype=np.int64), k)
    cols = indices.ravel()
    vals = np.exp(-distances.ravel() / np.repeat(sigmas, k))
    vals[rows == cols] = 0.0
    directed = sparse.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    directed.eliminate_zeros()
    transposed = directed.T.tocsr()
    graph = directed + transposed - directed.multiply(transposed)
    graph.eliminate_zeros()
    return graph.tocsr()


def density_landmark_sample(
    data: np.ndarray,
    n_landmarks: int = 1_200,
    n_neighbors: int = 1_500,
    omega: float = 1.0,
    metric: str = "cosine",
    n_jobs: int = -1,
) -> Dict[str, Any]:
    """Density-aware landmark sampling used before persistent cohomology.

    At each step the point with the strongest remaining fuzzy-neighbourhood
    support is selected, then nearby support is suppressed.  This retains
    dense manifold structure without letting repeatedly sampled regions
    dominate the point cloud.
    """
    x = _require_2d_finite(data, name="data")
    m = min(max(3, int(n_landmarks)), x.shape[0])
    graph = fuzzy_neighbourhood_graph(
        x,
        n_neighbors=min(int(n_neighbors), x.shape[0]),
        metric=metric,
        n_jobs=n_jobs,
    )
    density = np.asarray(graph.sum(axis=1)).ravel().astype(np.float64)
    selected = np.empty(m, dtype=np.int64)
    support = np.empty(m, dtype=np.float64)

    for t in range(m):
        idx = int(np.argmax(density))
        selected[t] = idx
        support[t] = density[idx]
        start, end = graph.indptr[idx], graph.indptr[idx + 1]
        neighbours = graph.indices[start:end]
        weights = graph.data[start:end]
        density[neighbours] -= float(omega) * weights
        density[idx] = -np.inf

    return {
        "indices": selected,
        "support": support,
        "graph": graph,
        "sampled": x[selected],
    }


def fuzzy_distance_matrix(
    data: np.ndarray,
    n_neighbors: int = 800,
    metric: str = "cosine",
    n_jobs: int = -1,
) -> Dict[str, Any]:
    """Convert fuzzy memberships to the distance matrix used by Ripser."""
    x = _require_2d_finite(data, name="data")
    graph = fuzzy_neighbourhood_graph(
        x,
        n_neighbors=min(int(n_neighbors), x.shape[0]),
        metric=metric,
        n_jobs=n_jobs,
    )
    membership = graph.toarray()
    with np.errstate(divide="ignore"):
        distance = -np.log(membership)
    np.fill_diagonal(distance, 0.0)
    finite_positive = distance[np.isfinite(distance) & (distance > 0)]
    threshold = float(finite_positive.max()) if finite_positive.size else 1.0
    # Ripser may reject infinities even when a finite threshold is provided.
    # Values above threshold are irrelevant, so replace missing edges safely.
    distance_finite = np.where(np.isfinite(distance), distance, threshold * 1.01)
    return {
        "distance": distance_finite,
        "threshold": threshold,
        "membership": membership,
        "graph": graph,
    }


# -------------------------------------------------------------------------
# Full preprocessing, UMAP visualisation and persistent cohomology
# -------------------------------------------------------------------------


def prepare_gardner_point_cloud(
    activations: np.ndarray,
    n_pca: int = 6,
    max_active_points: int = 15_000,
    n_landmarks: int = 1_200,
    density_neighbors: int = 1_500,
    density_omega: float = 1.0,
    activity_mode: str = "auto",
    metric: str = "cosine",
    seed: int = 42,
    n_jobs: int = -1,
) -> Dict[str, Any]:
    """Create the six-dimensional, density-denoised point cloud."""
    x = _require_2d_finite(activations)
    finite_rows = np.isfinite(x).all(axis=1)
    x_finite = x[finite_rows]
    finite_input_indices = np.flatnonzero(finite_rows)
    if x_finite.shape[0] < 3:
        raise ValueError("Fewer than three finite population vectors remain")

    active_local = select_active_population_vectors(
        x_finite, max_points=max_active_points, mode=activity_mode,
    )
    active_raw = x_finite[active_local]
    active_input_indices = finite_input_indices[active_local]

    pca_result = pca_population_activity(
        active_raw,
        n_components=n_pca,
        random_state=seed,
    )
    pcs = pca_result["projection"]
    landmark_result = density_landmark_sample(
        pcs,
        n_landmarks=n_landmarks,
        n_neighbors=density_neighbors,
        omega=density_omega,
        metric=metric,
        n_jobs=n_jobs,
    )
    landmark_local = landmark_result["indices"]
    return {
        "active_input_indices": active_input_indices,
        "landmark_active_indices": landmark_local,
        "landmark_input_indices": active_input_indices[landmark_local],
        "active_raw": active_raw,
        "active_standardized": pca_result["standardized"],
        "active_pcs": pcs,
        "landmark_pcs": pcs[landmark_local],
        "pca": pca_result["pca"],
        "pca_explained_variance_ratio": pca_result["explained_variance_ratio"],
        "retained_unit_mask": pca_result["unit_mask"],
        "unit_mean": pca_result["unit_mean"],
        "unit_std": pca_result["unit_std"],
        "density_support": landmark_result["support"],
    }


def prepare_umap_input(
    activations: np.ndarray,
    n_pca: int = 6,
    max_points: int = 30_000,
    seed: int = 42,
    standardization_reference: Optional[np.ndarray] = None,
    min_unit_std: float = 1e-8,
) -> Dict[str, Any]:
    """Prepare a uniformly sampled PCA representation for UMAP visualisation.

    The paper standardised each cell's full movement-filtered firing-rate trace
    *before* the 250-ms temporal downsampling used for the figure.  The optional
    ``standardization_reference`` reproduces that order when ``activations`` is
    already a temporally downsampled subset.  With no reference, moments are
    estimated from all finite input rows before the optional random cap.
    """
    x = _require_2d_finite(activations)
    reference = x if standardization_reference is None else _require_2d_finite(
        standardization_reference,
        name="standardization_reference",
    )
    if reference.shape[1] != x.shape[1]:
        raise ValueError(
            "activations and standardization_reference must have the same "
            f"number of units, got {x.shape[1]} and {reference.shape[1]}"
        )

    reference = reference[np.isfinite(reference).all(axis=1)]
    if reference.shape[0] < 3:
        raise ValueError("Fewer than three finite standardisation rows remain")
    unit_mean_all = reference.mean(axis=0)
    unit_std_all = reference.std(axis=0)
    unit_mask = (
        np.isfinite(unit_mean_all)
        & np.isfinite(unit_std_all)
        & (unit_std_all > float(min_unit_std))
    )
    if int(unit_mask.sum()) < 2:
        raise ValueError("Fewer than two non-constant units remain")
    unit_mean = unit_mean_all[unit_mask]
    unit_std = unit_std_all[unit_mask]

    finite_rows = np.isfinite(x).all(axis=1)
    finite_idx = np.flatnonzero(finite_rows)
    standardized = (x[finite_rows][:, unit_mask] - unit_mean) / unit_std
    if standardized.shape[0] > max_points:
        rng = np.random.default_rng(seed)
        local = np.sort(
            rng.choice(standardized.shape[0], size=max_points, replace=False)
        )
        standardized = standardized[local]
        input_indices = finite_idx[local]
    else:
        input_indices = finite_idx

    n_actual = min(int(n_pca), standardized.shape[1], standardized.shape[0] - 1)
    if n_actual < 2:
        raise ValueError("At least two principal components are required")
    pca = PCA(n_components=n_actual, random_state=seed)
    projection = pca.fit_transform(standardized)
    return {
        "input_indices": input_indices,
        "projection": projection,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "pca": pca,
        "retained_unit_mask": unit_mask,
        "unit_mean": unit_mean,
        "unit_std": unit_std,
        "standardized": standardized,
    }


def umap_from_pcs(
    pcs: np.ndarray,
    n_components: int = 3,
    n_neighbors: int = 5_000,
    min_dist: float = 0.8,
    metric: str = "cosine",
    init: str = "spectral",
    seed: int = 42,
) -> Dict[str, Any]:
    """Run UMAP on PCA coordinates using Gardner visualisation defaults."""
    try:
        import umap
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("umap-learn is required: pip install umap-learn") from exc

    x = _require_2d_finite(pcs, name="pcs")
    n_nb = min(max(2, int(n_neighbors)), x.shape[0] - 1)
    reducer = umap.UMAP(
        n_components=int(n_components),
        n_neighbors=n_nb,
        min_dist=float(min_dist),
        metric=metric,
        init=init,
        random_state=int(seed),
    )
    projection = reducer.fit_transform(x)
    return {
        "projection": projection,
        "reducer": reducer,
        "n_components": int(n_components),
        "n_neighbors": n_nb,
        "min_dist": float(min_dist),
        "metric": metric,
        "init": init,
    }


def persistence_lifetimes(diagram: np.ndarray) -> np.ndarray:
    """Return finite persistence lifetimes in descending order."""
    dgm = np.asarray(diagram, dtype=np.float64)
    if dgm.size == 0:
        return np.empty(0, dtype=np.float64)
    finite = dgm[np.isfinite(dgm[:, 1])]
    if finite.size == 0:
        return np.empty(0, dtype=np.float64)
    return np.sort(finite[:, 1] - finite[:, 0])[::-1]


def persistence_essential_count(diagram: np.ndarray) -> int:
    """Count essential (infinite-death) persistence classes.

    In a connected point cloud, H0 normally contains one essential class.
    Keeping this count separate avoids the common error of reporting H0=0
    after finite-only lifetime filtering.
    """
    dgm = np.asarray(diagram, dtype=np.float64)
    if dgm.size == 0:
        return 0
    return int(np.sum(~np.isfinite(dgm[:, 1])))


def circular_coordinate_from_cocycle(
    cocycle: np.ndarray,
    distance_matrix: np.ndarray,
    threshold: float,
    coeff: int,
) -> Dict[str, Any]:
    """Smooth one persistent 1-cocycle into a circular coordinate.

    This is a sparse implementation of the cohomological-coordinate step
    used in the Gardner repository.  Cocycle coefficients are lifted from
    ``Z/coeff`` to signed integers and a least-squares vertex potential is
    solved over all Vietoris--Rips edges present at ``threshold``.

    Returns a length-N coordinate in turns (``[0, 1)``), with ``NaN`` only
    for vertices absent from the threshold graph.
    """
    dists = np.asarray(distance_matrix, dtype=np.float64)
    if dists.ndim != 2 or dists.shape[0] != dists.shape[1]:
        raise ValueError("distance_matrix must be square")
    n = dists.shape[0]
    coc = np.asarray(cocycle, dtype=np.int64).copy()
    if coc.ndim != 2 or (coc.size and coc.shape[1] < 3):
        raise ValueError("cocycle must have columns [vertex_i, vertex_j, value]")

    # Ripser normally returns i < j.  Store values in the lower triangle to
    # match the reference implementation's edge orientation (j -> i).
    cocycle_values: Dict[Tuple[int, int], float] = {}
    half = int(coeff) / 2.0
    for i_raw, j_raw, value_raw, *_ in coc:
        i, j = int(i_raw), int(j_raw)
        value = int(value_raw)
        if value > half:
            value -= int(coeff)
        if j >= i:
            key = (j, i)
            signed = float(value)
        else:
            key = (i, j)
            signed = float(-value)
        cocycle_values[key] = signed

    lower_row, lower_col = np.tril_indices(n, k=-1)
    edge_mask = (
        np.isfinite(dists[lower_row, lower_col])
        & (dists[lower_row, lower_col] > 0.0)
        & (dists[lower_row, lower_col] <= float(threshold))
    )
    edge_row = lower_row[edge_mask]
    edge_col = lower_col[edge_mask]
    n_edges = int(edge_row.size)
    if n_edges == 0:
        return {
            "coordinate": np.full(n, np.nan, dtype=np.float64),
            "vertices": np.empty(0, dtype=int),
            "n_edges": 0,
            "residual_norm": float("nan"),
        }

    vertices = np.unique(np.concatenate([edge_row, edge_col]))
    vertex_lookup = np.full(n, -1, dtype=np.int64)
    vertex_lookup[vertices] = np.arange(vertices.size, dtype=np.int64)

    incidence_rows = np.repeat(np.arange(n_edges, dtype=np.int64), 2)
    incidence_cols = np.column_stack(
        [vertex_lookup[edge_row], vertex_lookup[edge_col]]
    ).ravel()
    incidence_vals = np.tile(np.array([-1.0, 1.0]), n_edges)
    incidence = sparse.coo_matrix(
        (incidence_vals, (incidence_rows, incidence_cols)),
        shape=(n_edges, vertices.size),
    ).tocsr()

    values = np.fromiter(
        (cocycle_values.get((int(r), int(c)), 0.0) for r, c in zip(edge_row, edge_col)),
        dtype=np.float64,
        count=n_edges,
    )
    solution = lsmr(incidence, values)
    vertex_coordinate = np.mod(solution[0], 1.0)
    coordinate = np.full(n, np.nan, dtype=np.float64)
    coordinate[vertices] = vertex_coordinate
    return {
        "coordinate": coordinate,
        "vertices": vertices,
        "n_edges": n_edges,
        "residual_norm": float(solution[3]),
    }


def decode_toroidal_landmark_coordinates(
    persistence_result: Mapping[str, Any],
    coeff: int = 47,
    lifetime_fraction: float = 0.99,
) -> Dict[str, Any]:
    """Decode two circular coordinates from the two longest H1 classes.

    The Vietoris--Rips scale is chosen at the birth plus 0.99 times the
    lifetime of the *second* longest H1 class, so both candidate circles are
    present on the largest practical common domain, matching the paper.
    """
    ripser_result = persistence_result.get("ripser", {})
    diagrams = persistence_result.get("diagrams", [])
    if len(diagrams) < 2 or "cocycles" not in ripser_result:
        raise ValueError("Persistence result does not contain H1 cocycles")

    h1 = np.asarray(diagrams[1], dtype=np.float64)
    if h1.ndim != 2 or h1.shape[0] < 2:
        raise ValueError("At least two H1 persistence classes are required")
    finite = np.isfinite(h1[:, 1])
    finite_indices = np.flatnonzero(finite)
    if finite_indices.size < 2:
        raise ValueError("At least two finite H1 persistence classes are required")
    finite_lifetimes = h1[finite_indices, 1] - h1[finite_indices, 0]
    order_local = np.argsort(finite_lifetimes)[::-1]
    class_indices = finite_indices[order_local[:2]]
    second = int(class_indices[1])
    threshold = float(
        h1[second, 0]
        + float(lifetime_fraction) * (h1[second, 1] - h1[second, 0])
    )

    h1_cocycles = ripser_result["cocycles"][1]
    coordinates = []
    diagnostics = []
    for class_index in class_indices:
        decoded = circular_coordinate_from_cocycle(
            h1_cocycles[int(class_index)],
            persistence_result["fuzzy_distance"],
            threshold=threshold,
            coeff=coeff,
        )
        coordinates.append(decoded["coordinate"])
        diagnostics.append({
            "class_index": int(class_index),
            "n_edges": int(decoded["n_edges"]),
            "residual_norm": float(decoded["residual_norm"]),
        })

    return {
        "coordinates_turns": np.column_stack(coordinates),
        "coordinates_radians": 2.0 * np.pi * np.column_stack(coordinates),
        "class_indices": class_indices.astype(int),
        "threshold": threshold,
        "lifetime_fraction": float(lifetime_fraction),
        "diagnostics": diagnostics,
    }


def _ripser_on_landmarks(
    landmark_pcs: np.ndarray,
    persistence_neighbors: int,
    metric: str,
    max_dim: int,
    coeff: int,
    n_jobs: int,
    do_cocycles: bool = False,
) -> Dict[str, Any]:
    try:
        from ripser import ripser
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("ripser is required: pip install ripser") from exc

    fuzzy = fuzzy_distance_matrix(
        landmark_pcs,
        n_neighbors=persistence_neighbors,
        metric=metric,
        n_jobs=n_jobs,
    )
    result = ripser(
        fuzzy["distance"],
        maxdim=int(max_dim),
        coeff=int(coeff),
        do_cocycles=bool(do_cocycles),
        distance_matrix=True,
        thresh=float(fuzzy["threshold"]),
    )
    return {
        "ripser": result,
        "diagrams": result["dgms"],
        "fuzzy_distance": fuzzy["distance"],
        "fuzzy_membership": fuzzy["membership"],
        "threshold": fuzzy["threshold"],
    }


def circularly_shift_units(
    activations: np.ndarray,
    rng: np.random.Generator,
    group_ids: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Independently circular-shift every unit while preserving trace structure.

    Gardner et al. circularly shifted each cell's continuous session-long trace.
    Model evaluations in this project are often concatenations of independent
    trajectories.  When ``group_ids`` is supplied, each unit is shifted
    independently *within every trajectory* so a null sample never rolls a
    post-reset state into the middle of a different trajectory.
    """
    x = _require_2d_finite(activations)
    shifted = np.empty_like(x)
    if group_ids is None:
        n = x.shape[0]
        for unit in range(x.shape[1]):
            offset = int(rng.integers(0, n))
            shifted[:, unit] = np.roll(x[:, unit], offset)
        return shifted

    groups = np.asarray(group_ids).reshape(-1)
    if groups.shape[0] != x.shape[0]:
        raise ValueError("group_ids and activations must have the same sample count")
    if np.issubdtype(groups.dtype, np.number) and not np.isfinite(groups).all():
        raise ValueError("group_ids must not contain NaN or infinite values")
    # np.unique sorts labels, which is harmless because each group's original
    # within-trajectory order is retained by the Boolean/index selection.
    group_indices = [np.flatnonzero(groups == group) for group in np.unique(groups)]
    for unit in range(x.shape[1]):
        for indices in group_indices:
            if indices.size < 2:
                shifted[indices, unit] = x[indices, unit]
                continue
            offset = int(rng.integers(0, indices.size))
            shifted[indices, unit] = np.roll(x[indices, unit], offset)
    return shifted


def _shuffle_rng_states(
    seed: int,
    n_shuffles: int,
    n_samples: int,
    n_units: int,
    group_ids: Optional[np.ndarray],
) -> List[Dict[str, Any]]:
    """Capture each shuffle's RNG state without constructing shifted arrays.

    Advancing the generator with the same scalar draws as
    :func:`circularly_shift_units` preserves the legacy serial null samples
    exactly, while allowing those samples to be evaluated out of order.
    """
    rng = np.random.default_rng(seed)
    states: List[Dict[str, Any]] = []

    if group_ids is None:
        group_sizes = None
    else:
        groups = np.asarray(group_ids).reshape(-1)
        group_sizes = [
            int(np.count_nonzero(groups == group)) for group in np.unique(groups)
        ]

    for _ in range(int(n_shuffles)):
        states.append(deepcopy(rng.bit_generator.state))
        if group_sizes is None:
            for _unit in range(int(n_units)):
                rng.integers(0, int(n_samples))
        else:
            for _unit in range(int(n_units)):
                for group_size in group_sizes:
                    if group_size >= 2:
                        rng.integers(0, group_size)
    return states


def _gardner_shuffle_max_lifetimes(
    shuffle_idx: int,
    rng_state: Mapping[str, Any],
    activations: np.ndarray,
    group_ids: Optional[np.ndarray],
    *,
    n_pca: int,
    max_active_points: int,
    n_landmarks: int,
    density_neighbors: int,
    persistence_neighbors: int,
    density_omega: float,
    activity_mode: str,
    metric: str,
    max_dim: int,
    coeff: int,
    seed: int,
) -> Tuple[int, Tuple[float, ...]]:
    """Run one null shuffle and return its maximum lifetime per dimension."""
    rng = np.random.default_rng()
    rng.bit_generator.state = deepcopy(dict(rng_state))
    shuffled = circularly_shift_units(activations, rng, group_ids=group_ids)

    # Outer process parallelism owns the available CPU cores.  Keeping each
    # worker's neighbour searches single-threaded prevents nested oversubscription.
    shuffled_prepared = prepare_gardner_point_cloud(
        shuffled,
        n_pca=n_pca,
        max_active_points=max_active_points,
        n_landmarks=n_landmarks,
        density_neighbors=density_neighbors,
        density_omega=density_omega,
        activity_mode=activity_mode,
        metric=metric,
        seed=seed + shuffle_idx + 1,
        n_jobs=1,
    )
    null_result = _ripser_on_landmarks(
        shuffled_prepared["landmark_pcs"],
        persistence_neighbors=persistence_neighbors,
        metric=metric,
        max_dim=max_dim,
        coeff=coeff,
        n_jobs=1,
        do_cocycles=False,
    )
    maxima = []
    for dim in range(max_dim + 1):
        lt = persistence_lifetimes(null_result["diagrams"][dim])
        maxima.append(float(lt[0]) if lt.size else 0.0)
    return int(shuffle_idx), tuple(maxima)


def persistent_cohomology_gardner(
    activations: np.ndarray,
    n_pca: int = 6,
    max_active_points: int = 15_000,
    n_landmarks: int = 1_200,
    density_neighbors: int = 1_500,
    persistence_neighbors: int = 800,
    density_omega: float = 1.0,
    activity_mode: str = "auto",
    metric: str = "cosine",
    max_dim: int = 2,
    coeff: int = 47,
    n_shuffles: int = 0,
    seed: int = 42,
    n_jobs: int = -1,
    do_cocycles: bool = False,
    trajectory_ids: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Quantify toroidal topology without applying the UMAP projection.

    Persistent cohomology is computed on the fuzzy representation of the six
    PCs, not on a 2-D/3-D UMAP embedding.  With shuffles enabled, a feature is
    called significant only when its lifetime exceeds the largest feature in
    the corresponding dimension across cell-wise circular-shift controls.
    ``is_torus_like`` is left undefined unless at least 1,000 shuffles are run;
    smaller controls are reported as ``is_torus_like_exploratory``.  ``n_jobs``
    controls process-level parallelism across null shuffles; worker-local
    neighbour searches are kept single-threaded to avoid oversubscription.
    """
    x = _require_2d_finite(activations)
    groups = None
    if trajectory_ids is not None:
        groups = np.asarray(trajectory_ids).reshape(-1)
        if groups.shape[0] != x.shape[0]:
            raise ValueError(
                "trajectory_ids and activations must have the same sample count"
            )
    prepared = prepare_gardner_point_cloud(
        x,
        n_pca=n_pca,
        max_active_points=max_active_points,
        n_landmarks=n_landmarks,
        density_neighbors=density_neighbors,
        density_omega=density_omega,
        activity_mode=activity_mode,
        metric=metric,
        seed=seed,
        n_jobs=n_jobs,
    )
    observed = _ripser_on_landmarks(
        prepared["landmark_pcs"],
        persistence_neighbors=persistence_neighbors,
        metric=metric,
        max_dim=max_dim,
        coeff=coeff,
        n_jobs=n_jobs,
        do_cocycles=do_cocycles,
    )
    lifetimes = {
        dim: persistence_lifetimes(observed["diagrams"][dim])
        for dim in range(min(max_dim + 1, len(observed["diagrams"])))
    }
    essential_counts = {
        dim: persistence_essential_count(observed["diagrams"][dim])
        for dim in range(min(max_dim + 1, len(observed["diagrams"])))
    }

    n_shuffles_int = int(n_shuffles)
    shuffle_max: Dict[int, List[float]] = {dim: [] for dim in range(max_dim + 1)}
    if n_shuffles_int > 0:
        rng_states = _shuffle_rng_states(
            seed,
            n_shuffles=n_shuffles_int,
            n_samples=x.shape[0],
            n_units=x.shape[1],
            group_ids=groups,
        )
        tasks = (
            delayed(_gardner_shuffle_max_lifetimes)(
                shuffle_idx,
                rng_states[shuffle_idx],
                x,
                groups,
                n_pca=n_pca,
                max_active_points=max_active_points,
                n_landmarks=n_landmarks,
                density_neighbors=density_neighbors,
                persistence_neighbors=persistence_neighbors,
                density_omega=density_omega,
                activity_mode=activity_mode,
                metric=metric,
                max_dim=max_dim,
                coeff=coeff,
                seed=seed,
            )
            for shuffle_idx in range(n_shuffles_int)
        )

        if n_jobs == 1 or n_shuffles_int == 1:
            shuffle_results = [
                _gardner_shuffle_max_lifetimes(
                    shuffle_idx,
                    rng_states[shuffle_idx],
                    x,
                    groups,
                    n_pca=n_pca,
                    max_active_points=max_active_points,
                    n_landmarks=n_landmarks,
                    density_neighbors=density_neighbors,
                    persistence_neighbors=persistence_neighbors,
                    density_omega=density_omega,
                    activity_mode=activity_mode,
                    metric=metric,
                    max_dim=max_dim,
                    coeff=coeff,
                    seed=seed,
                )
                for shuffle_idx in range(n_shuffles_int)
            ]
        else:
            with parallel_config(backend="loky", inner_max_num_threads=1):
                shuffle_results = Parallel(
                    n_jobs=n_jobs,
                    max_nbytes="1M",
                    mmap_mode="r",
                )(tasks)

        # Joblib preserves submission order, but sorting also protects this
        # invariant if the execution backend changes later.
        for _, maxima in sorted(shuffle_results, key=lambda item: item[0]):
            for dim, maximum in enumerate(maxima):
                shuffle_max[dim].append(float(maximum))

    null_max = {
        dim: (float(np.max(values)) if values else float("nan"))
        for dim, values in shuffle_max.items()
    }

    p_values: Dict[str, float] = {}
    if n_shuffles > 0:
        for dim, count in ((1, 2), (2, 1)):
            observed_lt = lifetimes.get(dim, np.empty(0))
            null_dist = np.asarray(shuffle_max[dim], dtype=np.float64)
            for rank in range(count):
                key = f"H{dim}_{rank + 1}"
                if rank >= observed_lt.size:
                    p_values[key] = 1.0
                else:
                    p_values[key] = float(
                        (1 + np.sum(null_dist >= observed_lt[rank]))
                        / (len(null_dist) + 1)
                    )

    significant_counts: Dict[int, Optional[int]] = {}
    for dim in range(max_dim + 1):
        if n_shuffles > 0:
            significant_counts[dim] = int(
                essential_counts.get(dim, 0)
                + np.sum(lifetimes.get(dim, np.empty(0)) > null_max[dim])
            )
        else:
            significant_counts[dim] = None

    h1 = lifetimes.get(1, np.empty(0))
    h2 = lifetimes.get(2, np.empty(0))
    if n_shuffles > 0:
        # Require the torus Betti pattern itself, not merely two long H1 bars
        # while ignoring a third significant loop or multiple H2 cavities.
        torus_like_exploratory: Optional[bool] = bool(
            significant_counts.get(0) == 1
            and significant_counts.get(1) == 2
            and significant_counts.get(2) == 1
        )
        # With N shuffles, the smallest rank-based p-value is 1/(N+1).
        # Reserve the unqualified verdict for the paper's 1000-shuffle scale.
        torus_like: Optional[bool] = (
            torus_like_exploratory if int(n_shuffles) >= 1000 else None
        )
    else:
        torus_like_exploratory = None
        torus_like = None

    h1_gap = float("nan")
    if h1.size >= 3 and h1[2] > 0:
        h1_gap = float(h1[1] / h1[2])
    elif h1.size >= 2:
        h1_gap = float("inf")

    decoded_coordinates = None
    if do_cocycles:
        try:
            decoded_coordinates = decode_toroidal_landmark_coordinates(
                observed,
                coeff=coeff,
                lifetime_fraction=0.99,
            )
        except ValueError:
            # A missing/weak pair of H1 classes is a scientific outcome, not a
            # runtime failure.  The caller can inspect the barcode and the
            # explicit ``None`` decoding result.
            decoded_coordinates = None

    return {
        "diagrams": observed["diagrams"],
        "lifetimes": lifetimes,
        "essential_counts": essential_counts,
        "significant_counts": significant_counts,
        "shuffle_max_lifetimes": shuffle_max,
        "null_max_lifetime": null_max,
        "p_values": p_values,
        "is_torus_like": torus_like,
        "is_torus_like_exploratory": torus_like_exploratory,
        "h1_second_to_third_gap": h1_gap,
        "h1_first_to_second_ratio": (
            float(h1[0] / h1[1]) if h1.size >= 2 and h1[1] > 0 else float("nan")
        ),
        "preprocessing": prepared,
        "decoded_landmark_coordinates": decoded_coordinates,
        "fuzzy_threshold": observed["threshold"],
        "coeff": int(coeff),
        "n_shuffles": int(n_shuffles),
        "metric": metric,
        "shuffle_within_trajectories": groups is not None,
    }


def pca_dimension_sweep(
    activations: np.ndarray,
    dimensions: Sequence[int] = (3, 4, 5, 6, 7, 8, 10, 12),
    max_active_points: int = 15_000,
    n_landmarks: int = 800,
    density_neighbors: int = 1_000,
    persistence_neighbors: int = 600,
    density_omega: float = 1.0,
    activity_mode: str = "auto",
    metric: str = "cosine",
    max_dim: int = 2,
    coeff: int = 47,
    seed: int = 42,
    n_jobs: int = -1,
) -> Dict[str, Any]:
    """Assess whether six PCs are empirically appropriate for this dataset.

    Gardner et al. did not choose six PCs merely by convention: they compared
    explained variance, spatial modulation and persistence separation across
    dimensionalities.  This helper performs the topology-facing part of that
    sensitivity analysis.  It is intentionally optional because repeated H2
    persistence calculations can be expensive.
    """
    dims = sorted({int(d) for d in dimensions if int(d) >= 2})
    if not dims:
        raise ValueError("dimensions must contain at least one value >= 2")

    x = _require_2d_finite(activations)
    finite_rows = np.isfinite(x).all(axis=1)
    x_finite = x[finite_rows]
    active_local = select_active_population_vectors(
        x_finite,
        max_points=max_active_points,
        mode=activity_mode,
    )
    active_raw = x_finite[active_local]
    standardized, _, unit_mask, unit_mean, unit_std = standardize_units(active_raw)
    max_components = min(max(dims), standardized.shape[1], standardized.shape[0] - 1)
    if max_components < 2:
        raise ValueError("Insufficient rank for PCA-dimension sweep")
    dims = [d for d in dims if d <= max_components]
    pca = PCA(n_components=max_components, random_state=seed)
    all_pcs = pca.fit_transform(standardized)
    evr = pca.explained_variance_ratio_

    rows = []
    diagrams_by_dimension: Dict[int, Sequence[np.ndarray]] = {}
    landmark_indices_by_dimension: Dict[int, np.ndarray] = {}
    for dimension in dims:
        pcs = all_pcs[:, :dimension]
        sampled = density_landmark_sample(
            pcs,
            n_landmarks=n_landmarks,
            n_neighbors=density_neighbors,
            omega=density_omega,
            metric=metric,
            n_jobs=n_jobs,
        )
        persistence = _ripser_on_landmarks(
            sampled["sampled"],
            persistence_neighbors=persistence_neighbors,
            metric=metric,
            max_dim=max_dim,
            coeff=coeff,
            n_jobs=n_jobs,
            do_cocycles=False,
        )
        h1 = (
            persistence_lifetimes(persistence["diagrams"][1])
            if len(persistence["diagrams"]) > 1
            else np.empty(0)
        )
        h2 = (
            persistence_lifetimes(persistence["diagrams"][2])
            if len(persistence["diagrams"]) > 2
            else np.empty(0)
        )
        if h1.size >= 3 and h1[2] > 0:
            h1_gap = float(h1[1] / h1[2])
        elif h1.size >= 2:
            h1_gap = float("inf")
        else:
            h1_gap = float("nan")
        rows.append({
            "n_pca": int(dimension),
            "cumulative_explained_variance": float(np.sum(evr[:dimension])),
            "h1_first": float(h1[0]) if h1.size >= 1 else float("nan"),
            "h1_second": float(h1[1]) if h1.size >= 2 else float("nan"),
            "h1_third": float(h1[2]) if h1.size >= 3 else float("nan"),
            "h1_second_to_third_gap": h1_gap,
            "h2_first": float(h2[0]) if h2.size >= 1 else float("nan"),
        })
        diagrams_by_dimension[int(dimension)] = persistence["diagrams"]
        landmark_indices_by_dimension[int(dimension)] = sampled["indices"]

    return {
        "rows": rows,
        "dimensions": dims,
        "explained_variance_ratio": evr,
        "diagrams_by_dimension": diagrams_by_dimension,
        "landmark_indices_by_dimension": landmark_indices_by_dimension,
        "active_input_indices": np.flatnonzero(finite_rows)[active_local],
        "retained_unit_mask": unit_mask,
        "unit_mean": unit_mean,
        "unit_std": unit_std,
        "coeff": int(coeff),
        "metric": metric,
    }


# -------------------------------------------------------------------------
# Selecting one coherent grid module and excluding directional units
# -------------------------------------------------------------------------


def head_direction_mvl(
    activations: np.ndarray,
    headings: np.ndarray,
    n_bins: int = 60,
) -> np.ndarray:
    """Mean-vector length of each unit's head-direction tuning curve."""
    x = _require_2d_finite(activations)
    hd = np.asarray(headings, dtype=np.float64).reshape(-1)
    if hd.shape[0] != x.shape[0]:
        raise ValueError("headings and activations must have the same sample count")
    valid = np.isfinite(hd) & np.isfinite(x).all(axis=1)
    x = x[valid]
    hd = np.mod(hd[valid], 2.0 * np.pi)
    if x.shape[0] < n_bins:
        return np.full(x.shape[1], np.nan)

    bin_idx = np.floor(hd / (2.0 * np.pi) * n_bins).astype(int)
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    tuning = np.full((n_bins, x.shape[1]), np.nan, dtype=np.float64)
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.any():
            tuning[b] = x[mask].mean(axis=0)

    # Fill unvisited direction bins by circular interpolation.  In normal
    # synthetic navigation all bins are visited, so this is rarely invoked.
    for unit in range(tuning.shape[1]):
        vals = tuning[:, unit]
        finite = np.isfinite(vals)
        if finite.sum() == 0:
            continue
        if finite.sum() < n_bins:
            xp = np.flatnonzero(finite)
            fp = vals[finite]
            xp_ext = np.concatenate([xp - n_bins, xp, xp + n_bins])
            fp_ext = np.concatenate([fp, fp, fp])
            vals[~finite] = np.interp(np.flatnonzero(~finite), xp_ext, fp_ext)
        if np.nanmin(vals) < 0:
            vals = vals - np.nanmin(vals)
        tuning[:, unit] = vals

    angles = (np.arange(n_bins) + 0.5) * 2.0 * np.pi / n_bins
    numerator = np.abs(np.nansum(tuning * np.exp(1j * angles[:, None]), axis=0))
    denominator = np.nansum(np.abs(tuning), axis=0)
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan, dtype=np.float64),
        where=denominator > _EPS,
    )


def _orientation_distance_deg(a: np.ndarray, b: float) -> np.ndarray:
    delta = np.abs(np.mod(a - b, 60.0))
    return np.minimum(delta, 60.0 - delta)


def _circular_orientation_mean_deg(orientations: np.ndarray) -> float:
    if orientations.size == 0:
        return float("nan")
    phase = np.exp(1j * np.deg2rad(orientations * 6.0))
    return float(np.mod(np.rad2deg(np.angle(np.mean(phase))) / 6.0, 60.0))


def select_grid_module(
    grid_results: Sequence[Mapping[str, Any]],
    activations: Optional[np.ndarray] = None,
    headings: Optional[np.ndarray] = None,
    score_key: str = "grid_score_conservative",
    score_threshold: float = 0.3,
    spacing_tolerance: float = 0.25,
    orientation_tolerance_deg: float = 15.0,
    exclude_head_direction: bool = True,
    head_direction_threshold: float = 0.3,
    min_module_units: int = 20,
) -> ModuleSelection:
    """Select the largest coherent grid module from per-unit grid statistics.

    Units must exceed the grid-score threshold and have similar grid spacing
    and orientation.  Orientation is treated as circular with period 60°.
    When headings are supplied, units with MVL above 0.3 are excluded to mimic
    the paper's initial restriction to non-conjunctive ("pure") grid cells.
    """
    records: List[Tuple[int, float, float, float]] = []
    for fallback_unit, item in enumerate(grid_results):
        unit = int(item.get("unit", fallback_unit))
        score = _safe_float(item.get(score_key))
        if not np.isfinite(score):
            # Conservative scores can be unavailable for marginal maps.
            for fallback_key in ("grid_score_google", "grid_score"):
                score = _safe_float(item.get(fallback_key))
                if np.isfinite(score):
                    break
        stats = item.get("grid_stats", {}) or {}
        spacing = _safe_float(stats.get("grid_spacing"))
        orientation = _safe_float(stats.get("grid_orientation"))
        if (
            np.isfinite(score)
            and score >= score_threshold
            and np.isfinite(spacing)
            and spacing > 0
            and np.isfinite(orientation)
        ):
            records.append((unit, score, spacing, orientation % 60.0))

    if not records:
        return ModuleSelection(
            unit_indices=np.empty(0, dtype=int),
            candidate_indices=np.empty(0, dtype=int),
            excluded_head_direction_indices=np.empty(0, dtype=int),
            score_key=score_key,
            score_threshold=float(score_threshold),
            median_score=float("nan"),
            median_spacing=float("nan"),
            mean_orientation_deg=float("nan"),
            n_units=0,
            n_candidates=0,
            sufficient_for_topology=False,
            note="No units passed grid-score/spacing/orientation filters",
        )

    units = np.asarray([r[0] for r in records], dtype=int)
    scores = np.asarray([r[1] for r in records], dtype=np.float64)
    spacings = np.asarray([r[2] for r in records], dtype=np.float64)
    orientations = np.asarray([r[3] for r in records], dtype=np.float64)
    candidate_indices = units.copy()

    excluded_hd = np.empty(0, dtype=int)
    if (
        exclude_head_direction
        and activations is not None
        and headings is not None
        and units.size > 0
    ):
        x = _require_2d_finite(activations)
        valid_units = units[(units >= 0) & (units < x.shape[1])]
        # Keep records aligned if malformed unit IDs were supplied.
        keep_valid = np.isin(units, valid_units)
        units, scores, spacings, orientations = (
            arr[keep_valid] for arr in (units, scores, spacings, orientations)
        )
        mvl = head_direction_mvl(x[:, units], headings)
        pure = ~np.isfinite(mvl) | (mvl <= head_direction_threshold)
        excluded_hd = units[~pure]
        units, scores, spacings, orientations = (
            arr[pure] for arr in (units, scores, spacings, orientations)
        )

    if units.size == 0:
        return ModuleSelection(
            unit_indices=np.empty(0, dtype=int),
            candidate_indices=candidate_indices,
            excluded_head_direction_indices=excluded_hd,
            score_key=score_key,
            score_threshold=float(score_threshold),
            median_score=float("nan"),
            median_spacing=float("nan"),
            mean_orientation_deg=float("nan"),
            n_units=0,
            n_candidates=int(candidate_indices.size),
            sufficient_for_topology=False,
            note="All grid candidates were excluded as head-direction modulated",
        )

    # Find the densest module around each candidate; this avoids transitive
    # chaining that can merge gradually changing grid scales.
    best_mask = np.zeros(units.size, dtype=bool)
    best_key = (-1, -np.inf)
    log_tol = np.log1p(max(0.0, float(spacing_tolerance)))
    log_spacing = np.log(spacings)
    for i in range(units.size):
        spacing_ok = np.abs(log_spacing - log_spacing[i]) <= log_tol
        orientation_ok = (
            _orientation_distance_deg(orientations, orientations[i])
            <= orientation_tolerance_deg
        )
        mask = spacing_ok & orientation_ok
        key = (int(mask.sum()), float(np.nanmedian(scores[mask])))
        if key > best_key:
            best_key = key
            best_mask = mask

    # One refinement around robust module centre.
    centre_spacing = float(np.exp(np.median(log_spacing[best_mask])))
    centre_orientation = _circular_orientation_mean_deg(orientations[best_mask])
    best_mask = (
        (np.abs(log_spacing - np.log(centre_spacing)) <= log_tol)
        & (
            _orientation_distance_deg(orientations, centre_orientation)
            <= orientation_tolerance_deg
        )
    )
    chosen = np.sort(units[best_mask])
    n_chosen = int(chosen.size)
    sufficient = n_chosen >= int(min_module_units)
    note = (
        "Selected largest spacing/orientation-consistent pure-grid module"
        if sufficient
        else (
            f"Only {n_chosen} units in the largest module; topology is underpowered "
            f"relative to the configured minimum of {int(min_module_units)}"
        )
    )
    return ModuleSelection(
        unit_indices=chosen,
        candidate_indices=candidate_indices,
        excluded_head_direction_indices=np.sort(excluded_hd),
        score_key=score_key,
        score_threshold=float(score_threshold),
        median_score=float(np.nanmedian(scores[best_mask])) if best_mask.any() else float("nan"),
        median_spacing=float(np.nanmedian(spacings[best_mask])) if best_mask.any() else float("nan"),
        mean_orientation_deg=(
            _circular_orientation_mean_deg(orientations[best_mask])
            if best_mask.any() else float("nan")
        ),
        n_units=n_chosen,
        n_candidates=int(candidate_indices.size),
        sufficient_for_topology=sufficient,
        note=note,
    )


def json_safe_persistence_summary(result: Mapping[str, Any]) -> Dict[str, Any]:
    """Extract JSON-serialisable diagnostics from a persistence result."""
    lifetimes = result.get("lifetimes", {})
    return {
        "done": True,
        "lifetimes": {
            str(dim): np.asarray(values, dtype=float)[:10].tolist()
            for dim, values in lifetimes.items()
        },
        "significant_counts": {
            str(dim): value
            for dim, value in result.get("significant_counts", {}).items()
        },
        "essential_counts": {
            str(dim): value
            for dim, value in result.get("essential_counts", {}).items()
        },
        "null_max_lifetime": {
            str(dim): value
            for dim, value in result.get("null_max_lifetime", {}).items()
        },
        "p_values": dict(result.get("p_values", {})),
        "is_torus_like": result.get("is_torus_like"),
        "is_torus_like_exploratory": result.get("is_torus_like_exploratory"),
        "h1_second_to_third_gap": result.get("h1_second_to_third_gap"),
        "h1_first_to_second_ratio": result.get("h1_first_to_second_ratio"),
        "coeff": result.get("coeff"),
        "n_shuffles": result.get("n_shuffles"),
        "metric": result.get("metric"),
        "shuffle_within_trajectories": result.get(
            "shuffle_within_trajectories", False
        ),
        "cohomological_decoding": (
            {
                "available": True,
                "class_indices": np.asarray(
                    result["decoded_landmark_coordinates"]["class_indices"],
                    dtype=int,
                ).tolist(),
                "threshold": float(
                    result["decoded_landmark_coordinates"]["threshold"]
                ),
                "lifetime_fraction": float(
                    result["decoded_landmark_coordinates"]["lifetime_fraction"]
                ),
                "diagnostics": result["decoded_landmark_coordinates"]["diagnostics"],
            }
            if result.get("decoded_landmark_coordinates") is not None
            else {"available": False}
        ),
    }
    