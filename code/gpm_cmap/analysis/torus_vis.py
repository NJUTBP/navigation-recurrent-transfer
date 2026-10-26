"""Torus topology analysis and visualisation.

The public helpers in this module are retained for backward compatibility.
The standard population-activity workflow now delegates to
:mod:`gpm_cmap.analysis.gardner_torus`, which keeps PCA/UMAP visualisation
separate from persistent-cohomology quantification.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.decomposition import PCA

from .gardner_torus import (
    json_safe_persistence_summary,
    pca_population_activity,
    persistent_cohomology_gardner,
    prepare_umap_input,
    umap_from_pcs,
)


# =========================================================================
#  Backward-compatible population-activity helpers
# =========================================================================

def torus_projection_summary(
    activations: np.ndarray,
    n_components: int = 3,
) -> Dict[str, Any]:
    """Unit-wise z-score followed by PCA.

    The previous implementation only mean-centred the matrix.  Grid-cell
    analyses require standardising each unit because firing-rate/activation
    scales differ substantially across units.
    """
    result = pca_population_activity(activations, n_components=n_components)
    return {
        "projection": result["projection"],
        "explained_variance_ratio": result["explained_variance_ratio"],
        "pca": result["pca"],
        "row_mask": result["row_mask"],
        "unit_mask": result["unit_mask"],
    }


def umap_projection(
    activations: np.ndarray,
    n_components: int = 3,
    n_neighbors: int = 5000,
    min_dist: float = 0.8,
    seed: int = 42,
    metric: str = "cosine",
    init: str = "spectral",
    pre_pca_components: Optional[int] = 6,
    max_points: int = 30_000,
) -> Dict[str, Any]:
    """PCA -> UMAP projection for torus visualisation.

    Parameters mirror Gardner et al.'s visualisation settings by default:
    six PCs, three UMAP dimensions, cosine metric, 5,000 neighbours,
    ``min_dist=0.8`` and spectral initialisation.  Set
    ``pre_pca_components=None`` only when *activations* are already PCA
    coordinates.
    """
    if pre_pca_components is None:
        pcs = np.asarray(activations, dtype=np.float64)
        input_indices = np.arange(len(pcs), dtype=int)
        evr = None
    else:
        prepared = prepare_umap_input(
            activations,
            n_pca=pre_pca_components,
            max_points=max_points,
            seed=seed,
        )
        pcs = prepared["projection"]
        input_indices = prepared["input_indices"]
        evr = prepared["explained_variance_ratio"]

    result = umap_from_pcs(
        pcs,
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        init=init,
        seed=seed,
    )
    result.update({
        "input_indices": input_indices,
        "pca_projection": pcs,
        "pca_explained_variance_ratio": evr,
    })
    return result


def persistent_homology_analysis(
    activations: np.ndarray,
    max_dim: int = 2,
    n_samples: Optional[int] = 1200,
    seed: int = 42,
    n_pca: int = 6,
    max_active_points: int = 15_000,
    density_neighbors: int = 1_500,
    persistence_neighbors: int = 800,
    coeff: int = 47,
    n_shuffles: int = 0,
    trajectory_ids: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Gardner-style persistent cohomology on the six-PC fuzzy graph.

    ``n_samples`` is interpreted as the number of density-selected landmarks.
    A binary torus verdict is only returned when shuffled controls are run;
    without controls ``is_torus_like`` is ``None`` rather than an arbitrary
    thresholded Betti-number count.
    """
    n_landmarks = 1200 if n_samples is None else int(n_samples)
    result = persistent_cohomology_gardner(
        activations,
        n_pca=n_pca,
        max_active_points=max_active_points,
        n_landmarks=n_landmarks,
        density_neighbors=density_neighbors,
        persistence_neighbors=persistence_neighbors,
        metric="cosine",
        max_dim=max_dim,
        coeff=coeff,
        n_shuffles=n_shuffles,
        seed=seed,
        trajectory_ids=trajectory_ids,
    )
    summary = json_safe_persistence_summary(result)
    betti = {
        int(dim): value
        for dim, value in result["significant_counts"].items()
    }
    return {
        "diagrams": result["diagrams"],
        "lifetimes": result["lifetimes"],
        "betti_numbers": betti,
        "is_torus_like": result["is_torus_like"],
        "null_max_lifetime": result["null_max_lifetime"],
        "p_values": result["p_values"],
        "preprocessing": result["preprocessing"],
        "summary": summary,
    }


# =========================================================================
#  Fourier phase analysis  (from run_exp1_torus_v8)
# =========================================================================

def fourier_phase_analysis(
    rate_map_mod: np.ndarray,
    res: int,
) -> Dict[str, Any]:
    """Fourier analysis of module rate maps to extract k-vectors and phases.

    Parameters
    ----------
    rate_map_mod : (n_units, res*res) — flattened rate maps for a module.
    res : spatial resolution.

    Returns
    -------
    dict with 'rm_fft', 'mean_power', 'k_vectors', 'phases', 'amplitudes'.
    """
    n_units = rate_map_mod.shape[0]
    rm_fft = np.zeros((n_units, res, res), dtype=np.complex128)
    for i in range(n_units):
        rm_fft[i] = np.fft.fft2(rate_map_mod[i].reshape(res, res))

    power = (np.abs(rm_fft) ** 2).mean(axis=0)
    power[0, 0] = 0
    k_vectors = _auto_detect_hex_kvectors(power, res)

    phases, amplitudes = [], []
    for kx, ky in k_vectors:
        mode = rm_fft[:, ky % res, kx % res]
        phases.append(np.angle(mode))
        amplitudes.append(np.abs(mode))

    return {
        "rm_fft": rm_fft,
        "mean_power": power,
        "k_vectors": k_vectors,
        "phases": phases,
        "amplitudes": amplitudes,
    }


def _auto_detect_hex_kvectors(power: np.ndarray, res: int) -> List[Tuple[int, int]]:
    """Auto-detect the three hex k-vectors from the mean power spectrum."""
    width = min(6, res // 2)
    idxs_range = np.arange(-width + 1, width)
    local_power = power[idxs_range][:, idxs_range].copy()
    local_power[width - 1, width - 1] = 0

    flat_order = np.argsort(local_power.ravel())[::-1]
    candidates = []
    for fi in flat_order:
        ky, kx = np.unravel_index(fi, local_power.shape)
        kx_real, ky_real = idxs_range[kx], idxs_range[ky]
        if kx_real == 0 and ky_real == 0:
            continue
        is_dup = any(
            (fk[0] == -kx_real and fk[1] == -ky_real) or
            (fk[0] == kx_real and fk[1] == ky_real)
            for fk in candidates
        )
        if not is_dup:
            candidates.append((int(kx_real), int(ky_real)))
        if len(candidates) >= 8:
            break

    best_triplet = candidates[:3] if len(candidates) >= 3 else [(3, 0), (2, 3), (-1, 3)]
    best_residual = np.inf

    if len(candidates) >= 3:
        from itertools import combinations
        for tri in combinations(candidates[:6], 3):
            k_arr = np.array(tri)
            for perm in [(0, 1, 2), (0, 2, 1), (1, 2, 0)]:
                for sign in [1, -1]:
                    residual = np.linalg.norm(
                        k_arr[perm[0]] + k_arr[perm[1]] + sign * k_arr[perm[2]]
                    )
                    if residual < best_residual:
                        best_residual = residual
                        best_triplet = list(tri)

    return best_triplet


# =========================================================================
#  SVD torus fit  (from run_exp1_torus_v8)
# =========================================================================

def svd_torus_fit(
    rate_map_mod: np.ndarray,
    res: int,
    k_vectors: List[Tuple[int, int]],
    n_svd: int = 10,
) -> Dict[str, Any]:
    """Fit the torus structure using SVD on rate maps.

    Finds a linear mapping from the top SVD components to ideal
    cosine/sine functions defined by the hex k-vectors.

    Parameters
    ----------
    rate_map_mod : (n_units, res*res) — flattened rate maps.
    res : spatial resolution.
    k_vectors : list of 3 (kx, ky) tuples.
    n_svd : number of SVD components to use.

    Returns
    -------
    dict with 'ring_coords', 'R', 'S', 'Ua', 'ideal'.
    """
    X = rate_map_mod - rate_map_mod.mean(axis=1, keepdims=True)
    Ua, S, _V = np.linalg.svd(X, full_matrices=False)
    # Project the centred rate maps.  Applying the singular vectors to the
    # uncentred maps leaks the per-unit DC component into the fitted rings.
    rm_embed = Ua.T @ X

    x_grid = np.mgrid[:res, :res] * 2 * np.pi / res
    x_flat = x_grid.reshape(2, -1)
    k = np.array(k_vectors[:3], dtype=np.float64)
    ideal = np.concatenate([np.cos(k @ x_flat), np.sin(k @ x_flat)], axis=0)

    RM = rm_embed[:n_svd]
    R = np.linalg.lstsq(RM.T, ideal.T, rcond=None)[0].T
    U2, s2, _Vt = np.linalg.svd(R, full_matrices=False)
    S2 = U2 @ np.diag(1.0 / (s2 + 1e-12)) @ U2.T
    R_orth = S2 @ R
    ring_coords = R_orth @ RM

    return {
        "ring_coords": ring_coords,
        "R": R_orth,
        "S": S[:20] if len(S) >= 20 else S,
        "Ua": Ua,
        "ideal": ideal,
    }


def extract_torus_angles(
    svd_res: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract θ₁ and θ₂ from SVD ring coordinates.

    For a perfect hex grid module:
        ring[0] ≈ cos(θ₁),  ring[3] ≈ sin(θ₁)
        ring[1] ≈ cos(θ₂),  ring[4] ≈ sin(θ₂)

    Returns
    -------
    theta1, theta2 : 1-D arrays of torus angles in [-π, π].
    """
    ring = svd_res["ring_coords"]
    theta1 = np.arctan2(ring[3], ring[0])
    theta2 = np.arctan2(ring[4], ring[1])
    return theta1, theta2


def torus_3d_embedding(
    theta1: np.ndarray,
    theta2: np.ndarray,
    R_big: float = 2.0,
    r_small: float = 0.9,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Embed torus angles into ℝ³ as a donut.

    x = (R + r·cos(θ₂)) · cos(θ₁)
    y = (R + r·cos(θ₂)) · sin(θ₁)
    z = r · sin(θ₂)

    Returns x, y, z arrays.
    """
    x = (R_big + r_small * np.cos(theta2)) * np.cos(theta1)
    y = (R_big + r_small * np.cos(theta2)) * np.sin(theta1)
    z = r_small * np.sin(theta2)
    return x, y, z


# =========================================================================
#  Rate-map-based topology  (Sorscher et al. 2023)
# =========================================================================

def build_spatial_point_cloud(
    rate_map_mod: np.ndarray,
    res: int,
    crop_frac: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Build the spatial point cloud from rate maps.

    Parameters
    ----------
    rate_map_mod : (n_units, res*res)
    res : spatial resolution.
    crop_frac : fraction to remove from each edge.

    Returns
    -------
    pop_spatial : (n_bins_cropped, n_units)
    pos_labels : (n_bins_cropped, 2)
    crop : number of bins cropped per edge.
    """
    crop = max(1, int(res * crop_frac))
    cx_arr = np.arange(crop, res - crop)
    if len(cx_arr) < 4:
        cx_arr = np.arange(res)
        crop = 0

    ci, cj = np.meshgrid(cx_arr, cx_arr, indexing="ij")
    spatial_idx = np.ravel_multi_index((ci.ravel(), cj.ravel()), (res, res))
    pop_spatial = rate_map_mod[:, spatial_idx].T  # (n_bins, n_units)

    xs = cx_arr / res - 0.5
    ys = cx_arr / res - 0.5
    xg, yg = np.meshgrid(xs, ys, indexing="ij")
    pos_labels = np.stack([xg.ravel(), yg.ravel()], axis=1)

    return pop_spatial, pos_labels, crop


def zscore_spatial(pop_spatial: np.ndarray) -> np.ndarray:
    """Z-score each unit (column) across spatial bins."""
    mu = pop_spatial.mean(axis=0, keepdims=True)
    sd = pop_spatial.std(axis=0, keepdims=True) + 1e-8
    return (pop_spatial - mu) / sd


def cosine_dist_matrix(x: np.ndarray) -> np.ndarray:
    """Full cosine distance matrix for (n_points, n_dims)."""
    x = np.asarray(x, dtype=np.float64)
    norms = np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    sim = np.clip((x / norms) @ (x / norms).T, -1.0, 1.0)
    d = 1.0 - sim
    np.fill_diagonal(d, 0.0)
    return d


def _top_lifetimes(dgm: np.ndarray, n: int = 10) -> List[float]:
    if len(dgm) == 0:
        return []
    f = dgm[np.isfinite(dgm[:, 1])]
    if len(f) == 0:
        return []
    return sorted((f[:, 1] - f[:, 0]).tolist(), reverse=True)[:n]


def topology_ratemap(
    rate_map_mod: np.ndarray,
    res: int,
    crop_frac: float = 0.1,
    n_pca: int = 6,
    n_shuffles: int = 5,
    coeff: int = 47,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Rate-map-based topology analysis following Sorscher et al. 2023.

    Parameters
    ----------
    rate_map_mod : (n_units, res*res)
    res : spatial resolution.
    crop_frac : boundary crop fraction.
    n_pca : PCA dimensions.
    n_shuffles : null-model shuffles.
    coeff : ripser field coefficient.
    verbose : print progress.

    Returns
    -------
    dict with 'available', 'diagrams', 'h0'/'h1'/'h2',
    'null_h1_max'/'null_h2_max', 'pca_evr', 'n_bins', 'n_units',
    'coords_pca', 'pos_labels'.
    """
    try:
        from ripser import ripser
    except ImportError:
        return {"available": False}

    pop_spatial, pos_labels, crop = build_spatial_point_cloud(
        rate_map_mod, res, crop_frac,
    )
    n_bins, n_units = pop_spatial.shape
    if verbose:
        print(f"  Spatial point cloud: {n_bins} bins × {n_units} units (crop={crop})")

    pop_z = zscore_spatial(pop_spatial)
    n_pca_actual = min(n_pca, n_units, n_bins - 1)
    pca = PCA(n_components=n_pca_actual)
    coords_pca = pca.fit_transform(pop_z)
    evr = pca.explained_variance_ratio_

    dist = cosine_dist_matrix(coords_pca)
    r = ripser(dist, maxdim=2, distance_matrix=True, coeff=coeff)
    dgms = r["dgms"]
    h0 = _top_lifetimes(dgms[0])
    h1 = _top_lifetimes(dgms[1])
    h2 = _top_lifetimes(dgms[2])

    # Null model
    null_h1_maxes, null_h2_maxes = [], []
    rng = np.random.default_rng(42)
    for _ in range(n_shuffles):
        shuffled = pop_z.copy()
        for col in range(shuffled.shape[1]):
            rng.shuffle(shuffled[:, col])
        pca_null = PCA(n_components=n_pca_actual)
        coords_null = pca_null.fit_transform(shuffled)
        dist_null = cosine_dist_matrix(coords_null)
        rn = ripser(dist_null, maxdim=2, distance_matrix=True, coeff=coeff)
        nh1 = _top_lifetimes(rn["dgms"][1], 1)
        nh2 = _top_lifetimes(rn["dgms"][2], 1)
        null_h1_maxes.append(nh1[0] if nh1 else 0.0)
        null_h2_maxes.append(nh2[0] if nh2 else 0.0)

    return {
        "available": True,
        "diagrams": dgms,
        "h0": h0,
        "h1": h1,
        "h2": h2,
        "null_h1_max": float(np.max(null_h1_maxes)) if null_h1_maxes else 0.0,
        "null_h2_max": float(np.max(null_h2_maxes)) if null_h2_maxes else 0.0,
        "pca_evr": evr.tolist(),
        "n_bins": n_bins,
        "n_units": n_units,
        "coords_pca": coords_pca,
        "pos_labels": pos_labels,
    }


# =========================================================================
#  Torus verdict  (from run_exp1_torus_v8)
# =========================================================================

def torus_verdict(
    topo: Dict[str, Any],
    null_factor: float = 2.0,
    h1_balance_limit: float = 5.0,
) -> Tuple[bool, str]:
    """Principled torus verdict following Sorscher/Gardner approach.

    Criteria:
      - Both H1[0] and H1[1] > null_factor × null_h1_max
      - H2[0] > null_factor × null_h2_max
      - H1 balance ratio (1st / 2nd lifetime) < h1_balance_limit

    Parameters
    ----------
    topo : output of ``topology_ratemap``.
    null_factor : multiplier above null maximum.
    h1_balance_limit : max allowed ratio between the two longest H1 bars.

    Returns
    -------
    (verdict, note) — bool and explanatory string.
    """
    if not topo.get("available", False):
        return False, "ripser unavailable"

    h1 = topo.get("h1", [])
    h2 = topo.get("h2", [])
    null_h1 = topo.get("null_h1_max", 0.0)
    null_h2 = topo.get("null_h2_max", 0.0)

    h1_thresh = max(null_factor * null_h1, 0.05)
    h2_thresh = max(null_factor * null_h2, 0.05)

    has_2_h1 = (
        len(h1) >= 2 and h1[0] > h1_thresh and h1[1] > h1_thresh
    )
    has_1_h2 = len(h2) >= 1 and h2[0] > h2_thresh
    h1_ratio = h1[0] / h1[1] if len(h1) >= 2 else float("inf")
    balanced = h1_ratio < h1_balance_limit

    verdict = has_2_h1 and has_1_h2 and balanced

    reasons = []
    if not has_2_h1:
        if len(h1) < 2:
            reasons.append(f"only {len(h1)} H1 bars")
        else:
            reasons.append(
                f"H1[0]={h1[0]:.3f} or H1[1]={h1[1]:.3f} "
                f"<= threshold={h1_thresh:.3f}"
            )
    if not has_1_h2:
        reasons.append(
            f"H2[0]={h2[0] if h2 else 0:.3f} <= threshold={h2_thresh:.3f}"
        )
    if not balanced:
        reasons.append(f"H1 ratio={h1_ratio:.2f} >= {h1_balance_limit}")
    if verdict:
        reasons = [
            f"H1=[{h1[0]:.3f},{h1[1]:.3f}] >> null={null_h1:.3f}; "
            f"H2={h2[0]:.3f} >> null={null_h2:.3f}; ratio={h1_ratio:.2f}"
        ]

    return verdict, "; ".join(reasons) if reasons else "all OK"


# =========================================================================
#  Trajectory-based torus analysis (v8)
# =========================================================================

def compute_grid_phase(
    positions: np.ndarray,
    k_vectors: List[Tuple[int, int]],
    res: int,
    arena_size: float,
) -> np.ndarray:
    """Compute two independent grid phases (θ₁, θ₂) for each position.

    Parameters
    ----------
    positions : (N, 2) — (x, y) in arena coordinates.
    k_vectors : at least 2 (kx, ky) tuples in Fourier-pixel units.
    res : spatial resolution.
    arena_size : arena width.

    Returns
    -------
    phases : (N, 2) — θ₁ and θ₂ in [0, 2π].
    """
    x_pix = (positions[:, 0] + arena_size / 2.0) / arena_size * res
    y_pix = (positions[:, 1] + arena_size / 2.0) / arena_size * res
    phases = []
    for kv in k_vectors[:2]:
        kx, ky = kv[0], kv[1]
        phi = (2.0 * np.pi / res) * (kx * x_pix + ky * y_pix)
        phi = phi % (2.0 * np.pi)
        phases.append(phi)
    return np.stack(phases, axis=1)


def build_trajectory_embedding(
    pop_traj: np.ndarray,
    positions: np.ndarray,
    k_vectors: List[Tuple[int, int]],
    res: int,
    arena_size: float,
    n_pca: int = 6,
    n_subsample: int = 10000,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PCA embedding of trajectory population activity for UMAP / torus.

    Parameters
    ----------
    pop_traj : (N, n_units) — per-timepoint population vectors.
    positions : (N, 2)
    k_vectors : hex k-vectors.
    res : spatial resolution.
    arena_size : arena width.
    n_pca, n_subsample, seed : PCA/subsample params.

    Returns
    -------
    coords_pca : (n_sub, n_pca)
    positions_sub : (n_sub, 2)
    phases_sub : (n_sub, 2) — θ₁ and θ₂ in [0, 2π].
    """
    n = len(pop_traj)
    rng = np.random.default_rng(seed)
    if n > n_subsample:
        idx = np.sort(rng.choice(n, n_subsample, replace=False))
        pop_sub = pop_traj[idx]
        pos_sub = positions[idx]
    else:
        pop_sub = pop_traj
        pos_sub = positions

    mu = pop_sub.mean(axis=0, keepdims=True)
    sd = pop_sub.std(axis=0, keepdims=True) + 1e-8
    pop_z = (pop_sub - mu) / sd

    n_pca_actual = min(n_pca, pop_z.shape[1], pop_z.shape[0] - 1)
    pca = PCA(n_components=n_pca_actual)
    coords_pca = pca.fit_transform(pop_z)

    phases_sub = compute_grid_phase(pos_sub, k_vectors, res, arena_size)
    return coords_pca, pos_sub, phases_sub


def run_umap_2d3d(
    coords: np.ndarray,
    n_neighbors: int = 5000,
    min_dist: float = 0.8,
    metric: str = "cosine",
    seed: int = 42,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Run UMAP in 2-D and 3-D.

    Returns (emb2, emb3) or (None, None) if umap-learn is missing.
    """
    try:
        import umap as umap_lib
    except ImportError:
        return None, None

    import warnings
    n_nb = min(n_neighbors, len(coords) - 1)
    kw = dict(metric=metric, n_neighbors=n_nb, min_dist=min_dist,
              init="spectral", random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        emb2 = umap_lib.UMAP(n_components=2, **kw).fit_transform(coords)
        emb3 = umap_lib.UMAP(n_components=3, **kw).fit_transform(coords)
    return emb2, emb3


# =========================================================================
#  Manifold distance maps
# =========================================================================

def manifold_distance_maps(
    rate_map_mod: np.ndarray,
    res: int,
    n_origins: int = 9,
) -> np.ndarray:
    """Compute normalised population-vector distance maps from several origins.

    Parameters
    ----------
    rate_map_mod : (n_units, res*res)
    res : spatial resolution.
    n_origins : number of origin points (arranged in a sqrt grid).

    Returns
    -------
    dist_maps : (n_origins, res, res) — normalised distance from each origin.
    """
    side = int(np.round(np.sqrt(n_origins)))
    origins_1d = np.linspace(res // 4, res - res // 4, side, dtype=int)
    origins = np.stack(
        np.meshgrid(origins_1d, origins_1d), axis=-1,
    ).reshape(-1, 2)
    dist_maps = np.zeros((len(origins), res, res))
    for idx, (ox, oy) in enumerate(origins):
        r0 = rate_map_mod[:, np.ravel_multi_index((ox, oy), (res, res)), None]
        dists = np.linalg.norm(r0 - rate_map_mod, axis=0)
        dist_maps[idx] = (dists / (dists.max() + 1e-12)).reshape(res, res)
    return dist_maps
