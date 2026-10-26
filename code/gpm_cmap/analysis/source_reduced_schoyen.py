"""Schøyen-selected torus coordinate construction for Figure A3.

This module deliberately separates two scientific operations:

1. *Formal topology qualification* is supplied by the already-completed
   Schøyen Euclidean persistent-homology analysis.
2. *Coordinate chart construction* is performed on an independent navigation
   calibration split.  The Schøyen-selected recurrent ensemble is converted
   into two periodic position-anchored Fourier coordinates and a frozen
   hidden-state -> circle-pair encoder is learned without using source test
   trajectories.

No legacy cohomological landmarks are used.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence, Tuple

import numpy as np
from sklearn.decomposition import PCA

from gpm_cmap.analysis.rate_map import compute_rate_map
from gpm_cmap.analysis.shared_manifold import CircularTorusChart, TrajectoryData
from gpm_cmap.analysis.source_reduced_v3 import FrozenTorusPhaseEncoder, wrap_angles
from gpm_cmap.analysis.torus_vis import compute_grid_phase

_EPS = 1e-12


def _canonical_kvector(kx: int, ky: int) -> Tuple[int, int]:
    """Collapse the conjugate FFT pair k and -k to one deterministic sign."""
    kx = int(kx); ky = int(ky)
    if kx < 0 or (kx == 0 and ky < 0):
        return -kx, -ky
    return kx, ky


def _fft_peak_candidates(
    rate_maps: np.ndarray,
    *,
    max_frequency: int,
    top_n: int = 12,
) -> list[Dict[str, Any]]:
    """Return dominant physical (kx, ky) peaks of a rate-map population.

    ``compute_rate_map`` stores x on array axis 0 and y on axis 1.  We keep
    that convention explicitly here instead of inheriting the image-coordinate
    convention of the older ``fourier_phase_analysis`` helper.
    """
    maps = np.asarray(rate_maps, dtype=np.float64)
    if maps.ndim != 3 or maps.shape[1] != maps.shape[2]:
        raise ValueError(f"Expected rate_maps [units,res,res], got {maps.shape}")
    res = int(maps.shape[1])
    centered = maps - maps.mean(axis=(1, 2), keepdims=True)
    fft = np.fft.fftshift(np.fft.fft2(centered, axes=(-2, -1)), axes=(-2, -1))
    power = np.mean(np.abs(fft) ** 2, axis=0)
    centre = res // 2
    power[centre, centre] = 0.0
    maxf = max(2, min(int(max_frequency), centre - 1))
    rows = []
    for ix in range(res):
        kx = ix - centre
        for iy in range(res):
            ky = iy - centre
            radius = float(np.hypot(kx, ky))
            if radius < 1.0 or radius > maxf:
                continue
            ckx, cky = _canonical_kvector(kx, ky)
            rows.append((float(power[ix, iy]), ckx, cky, radius))
    rows.sort(reverse=True, key=lambda z: z[0])
    selected: list[Dict[str, Any]] = []
    seen = set()
    total = float(np.sum(power))
    for peak, kx, ky, radius in rows:
        key = (kx, ky)
        if key in seen:
            continue
        seen.add(key)
        selected.append({
            "k": key,
            "power": peak,
            "power_fraction": peak / max(total, _EPS),
            "radius": radius,
        })
        if len(selected) >= int(top_n):
            break
    return selected


def _schoyen_wavevector_pair(
    rate_maps: np.ndarray,
    *,
    family_labels: np.ndarray | None,
    max_frequency: int,
) -> tuple[Tuple[Tuple[int, int], Tuple[int, int]], Dict[str, Any]]:
    """Select two independent periodic axes using the Schøyen 3-family refinement.

    One dominant Fourier direction is first extracted from each available
    KMeans band family.  Population-level peaks are appended as a fallback.
    Pair selection rewards spectral strength, linear independence, and similar
    spatial scale; it never uses source validation/test outcomes.
    """
    maps = np.asarray(rate_maps, dtype=np.float64)
    candidates: list[Dict[str, Any]] = []
    family_diag = []
    if family_labels is not None:
        labels = np.asarray(family_labels, dtype=np.int64).reshape(-1)
        if labels.size != maps.shape[0]:
            raise ValueError(
                f"kmeans3_family_labels has {labels.size} entries for {maps.shape[0]} selected units"
            )
        for label in sorted(np.unique(labels).tolist()):
            subset = maps[labels == int(label)]
            peaks = _fft_peak_candidates(
                subset, max_frequency=int(max_frequency), top_n=4
            )
            family_diag.append({
                "family": int(label),
                "n_units": int(subset.shape[0]),
                "peaks": [
                    {**z, "k": list(z["k"])} for z in peaks
                ],
            })
            if peaks:
                row = dict(peaks[0]); row["source"] = f"family_{int(label)}"
                candidates.append(row)

    population_peaks = _fft_peak_candidates(
        maps, max_frequency=int(max_frequency), top_n=12
    )
    for row in population_peaks:
        z = dict(row); z["source"] = "population"
        candidates.append(z)

    # Deduplicate conjugate-equivalent vectors, retaining the strongest entry.
    by_k: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for row in candidates:
        key = tuple(row["k"])
        old = by_k.get(key)
        if old is None or float(row["power_fraction"]) > float(old["power_fraction"]):
            by_k[key] = row
    candidates = list(by_k.values())
    if len(candidates) < 2:
        raise RuntimeError("Fewer than two Fourier wavevector candidates were detected")

    max_power = max(float(z["power_fraction"]) for z in candidates)
    best = None
    best_score = -np.inf
    pair_rows = None
    for i in range(len(candidates)):
        a = np.asarray(candidates[i]["k"], dtype=float)
        na = float(np.linalg.norm(a))
        if na <= _EPS:
            continue
        for j in range(i + 1, len(candidates)):
            b = np.asarray(candidates[j]["k"], dtype=float)
            nb = float(np.linalg.norm(b))
            if nb <= _EPS:
                continue
            sine = float(abs(a[0] * b[1] - a[1] * b[0]) / max(na * nb, _EPS))
            if sine < 0.15:
                continue
            strength = np.sqrt(
                max(float(candidates[i]["power_fraction"]) / max(max_power, _EPS), _EPS)
                * max(float(candidates[j]["power_fraction"]) / max(max_power, _EPS), _EPS)
            )
            # Same toroidal module should have comparable spatial wavelength.
            scale_similarity = float(np.exp(-abs(np.log(na / nb))))
            score = sine * strength * scale_similarity
            if score > best_score:
                best_score = score
                best = (tuple(map(int, a)), tuple(map(int, b)))
                pair_rows = (candidates[i], candidates[j], sine, scale_similarity)
    if best is None or pair_rows is None:
        raise RuntimeError("Could not find two sufficiently independent periodic wavevectors")

    diagnostics = {
        "family_candidates": family_diag,
        "population_peaks": [{**z, "k": list(z["k"])} for z in population_peaks],
        "selected_pair": [list(best[0]), list(best[1])],
        "selected_sources": [pair_rows[0].get("source"), pair_rows[1].get("source")],
        "selected_pair_sine": float(pair_rows[2]),
        "selected_pair_scale_similarity": float(pair_rows[3]),
        "selected_pair_score": float(best_score),
        "max_frequency": int(max_frequency),
        "axis_convention": "rate-map axis0=x, axis1=y; selected k vectors are physical (kx,ky)",
    }
    return best, diagnostics


def _population_rate_maps(
    data: TrajectoryData,
    *,
    resolution: int,
    arena_size: float,
    smooth_sigma: float,
) -> np.ndarray:
    maps = []
    for unit in range(data.n_units):
        rate_map, _, _ = compute_rate_map(
            data.coords,
            data.states[:, unit],
            bins=int(resolution),
            arena_size=float(arena_size),
            smooth_sigma=float(smooth_sigma),
            min_occupancy=0.0,
        )
        maps.append(np.nan_to_num(rate_map, nan=0.0, posinf=0.0, neginf=0.0))
    return np.asarray(maps, dtype=np.float64)


def target_phase_angles(
    coords: np.ndarray,
    *,
    k_vectors: Sequence[Sequence[int]],
    resolution: int,
    arena_size: float,
) -> np.ndarray:
    return np.asarray(
        compute_grid_phase(
            np.asarray(coords, dtype=np.float64),
            [tuple(map(int, v[:2])) for v in k_vectors],
            int(resolution),
            float(arena_size),
        ),
        dtype=np.float64,
    )


def circular_rmse(actual: np.ndarray, expected: np.ndarray) -> float:
    residual = wrap_angles(np.asarray(actual, dtype=np.float64) - np.asarray(expected, dtype=np.float64))
    return float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))


def phase_bin_coverage(theta: np.ndarray, bins_per_axis: int = 12) -> float:
    theta = np.asarray(theta, dtype=np.float64)
    scaled = (theta % (2.0 * np.pi)) / (2.0 * np.pi)
    bins = np.floor(scaled * int(bins_per_axis)).astype(np.int64) % int(bins_per_axis)
    occupied = np.unique(bins[:, 0] * int(bins_per_axis) + bins[:, 1]).size
    return float(occupied / float(int(bins_per_axis) ** 2))


def build_schoyen_phase_encoder(
    phase_data: TrajectoryData,
    *,
    selected_unit_indices: Sequence[int],
    arena_size: float,
    resolution: int = 64,
    smooth_sigma: float = 1.0,
    family_labels: np.ndarray | None = None,
    max_fft_frequency: int = 16,
    chart_dim: int = 8,
    max_landmarks: int = 8000,
    seed: int = 42,
) -> FrozenTorusPhaseEncoder:
    """Fit a frozen two-circle encoder on an independent calibration split.

    The target circle coordinates come from the dominant periodic wavevectors of
    the *Schøyen-selected* recurrent population.  They are position anchored,
    but the fitted encoder itself maps recurrent activity to the two circles and
    can therefore be evaluated on target-RNN hidden states without position.
    """
    selected = np.asarray(selected_unit_indices, dtype=np.int64).reshape(-1)
    if phase_data.n_units != selected.size:
        raise ValueError(
            f"phase_data has {phase_data.n_units} selected units; expected {selected.size}"
        )

    rate_maps = _population_rate_maps(
        phase_data,
        resolution=int(resolution),
        arena_size=float(arena_size),
        smooth_sigma=float(smooth_sigma),
    )
    pair, wavevector_diagnostics = _schoyen_wavevector_pair(
        rate_maps,
        family_labels=family_labels,
        max_frequency=int(max_fft_frequency),
    )
    theta_target = target_phase_angles(
        phase_data.coords,
        k_vectors=pair,
        resolution=int(resolution),
        arena_size=float(arena_size),
    )
    turns = (theta_target % (2.0 * np.pi)) / (2.0 * np.pi)

    full_mean = phase_data.states.mean(axis=0)
    full_std = phase_data.states.std(axis=0)
    retained = np.isfinite(full_mean) & np.isfinite(full_std) & (full_std > 1e-8)
    if int(retained.sum()) < 3:
        raise RuntimeError("Too few non-constant Schøyen-selected units for phase encoding")
    mean = full_mean[retained]
    std = full_std[retained]
    standardized = (phase_data.states[:, retained] - mean) / std
    n_components = min(int(chart_dim), standardized.shape[1], standardized.shape[0] - 1)
    if n_components < 2:
        raise RuntimeError("Insufficient selected-population rank for a two-circle chart")
    pca = PCA(n_components=n_components, random_state=int(seed))
    chart_states = pca.fit_transform(standardized)

    n = phase_data.n_states
    if n > int(max_landmarks):
        rng = np.random.default_rng(int(seed) + 701)
        # Sample within each trajectory before a final deterministic cap so the
        # chart is not dominated by a handful of long trajectories.
        groups = phase_data.trajectory_ids
        unique = np.unique(groups)
        per_group = max(2, int(np.ceil(int(max_landmarks) / max(len(unique), 1))))
        chosen = []
        for group in unique:
            rows = np.flatnonzero(groups == group)
            if rows.size > per_group:
                rows = np.sort(rng.choice(rows, per_group, replace=False))
            chosen.extend(rows.tolist())
        landmark_rows = np.asarray(sorted(set(chosen)), dtype=np.int64)
        if landmark_rows.size > int(max_landmarks):
            landmark_rows = np.sort(
                rng.choice(landmark_rows, int(max_landmarks), replace=False)
            )
    else:
        landmark_rows = np.arange(n, dtype=np.int64)

    chart = CircularTorusChart.fit(
        chart_states[landmark_rows],
        standardized[landmark_rows],
        turns[landmark_rows],
        phase_data.trajectory_ids[landmark_rows],
        seed=int(seed),
    )
    encoder = FrozenTorusPhaseEncoder(
        selected_unit_indices=selected,
        retained_unit_mask=retained,
        unit_mean=np.asarray(mean, dtype=np.float64),
        unit_std=np.asarray(std, dtype=np.float64),
        pca=pca,
        chart=chart,
        diagnostics={
            "fit_domain": "independent_schoyen_phase_calibration_trajectories",
            "coordinate_definition": "true_position_fourier_phase_on_schoyen_selected_ensemble",
            "n_phase_states": int(phase_data.n_states),
            "n_phase_trajectories": int(phase_data.unique_trajectories.size),
            "n_landmarks_used": int(landmark_rows.size),
            "chart_dim": int(n_components),
            "ratemap_resolution": int(resolution),
            "ratemap_smooth_sigma": float(smooth_sigma),
            "arena_size": float(arena_size),
            "selected_k_vectors": [list(pair[0]), list(pair[1])],
            "wavevector_selection": wavevector_diagnostics,
            "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "chart_fit": chart.fit_diagnostics,
        },
    )
    predicted = encoder.raw_angles(phase_data.states)
    encoder.diagnostics["calibration_phase_rmse_rad"] = circular_rmse(predicted, theta_target)
    encoder.diagnostics["calibration_target_phase_coverage_12x12"] = phase_bin_coverage(theta_target, 12)
    return encoder


def evaluate_schoyen_phase_encoder(
    encoder: FrozenTorusPhaseEncoder,
    data: TrajectoryData,
) -> Dict[str, Any]:
    diag = dict(encoder.diagnostics)
    pair = diag.get("selected_k_vectors")
    if pair is None:
        raise RuntimeError("Schøyen phase encoder diagnostics lack selected_k_vectors")
    theta_target = target_phase_angles(
        data.coords,
        k_vectors=pair,
        resolution=int(diag["ratemap_resolution"]),
        arena_size=float(diag["arena_size"]),
    )
    predicted = encoder.raw_angles(data.states)
    residual = wrap_angles(predicted - theta_target)
    return {
        "phase_rmse_rad": circular_rmse(predicted, theta_target),
        "phase_rmse_axis0_rad": float(np.sqrt(np.mean(residual[:, 0] ** 2))),
        "phase_rmse_axis1_rad": float(np.sqrt(np.mean(residual[:, 1] ** 2))),
        "target_phase_coverage_12x12": phase_bin_coverage(theta_target, 12),
        "predicted_phase_coverage_12x12": phase_bin_coverage(predicted, 12),
        "n_states": int(data.n_states),
        "n_trajectories": int(data.unique_trajectories.size),
    }
