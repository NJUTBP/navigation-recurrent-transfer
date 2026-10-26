"""Spatial rate-map utilities.

Upgraded following opexebo conventions:
  - proper occupancy-based masking (bins with < min_occupancy are masked)
  - optional adaptive smoothing (smooth occupancy & spike map separately)
  - rate-map coherence (Muller & Kubie 1989)
  - rate-map statistics: spatial information, sparsity, selectivity
    (Skaggs et al. 1993/1996)

All public functions preserve backward-compatible call signatures.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import numpy as np
from scipy.ndimage import gaussian_filter


# =========================================================================
# Core rate map
# =========================================================================

def compute_rate_map(
    positions: np.ndarray,
    activations: np.ndarray,
    bins: int = 32,
    arena_size: float = 2.2,
    smooth_sigma: float = 0.0,
    min_occupancy: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute a 2-D spatial rate map.

    Parameters
    ----------
    positions : (N, 2) array of (x, y) coordinates.
    activations : (N,) array of per-step scalar activations.
    bins : int — number of spatial bins per side.
    arena_size : float — full width of the square arena (same units as
        *positions*).
    smooth_sigma : float — Gaussian smoothing sigma (in bins).  When > 0
        both the occupancy and the weighted-activity maps are smoothed
        **before** division (adaptive smoothing), which is the standard
        approach in the place / grid cell literature.
    min_occupancy : float — bins whose (smoothed) occupancy count is
        below this value are masked with ``np.nan``.  Set to 0.0
        (default) to reproduce the old behaviour.

    Returns
    -------
    rate_map : (bins, bins) ndarray — firing-rate map.  Bins with
        insufficient occupancy contain ``np.nan`` when *min_occupancy* > 0.
    xedges : (bins+1,) bin edges along x.
    yedges : (bins+1,) bin edges along y.
    """
    half = arena_size / 2.0
    x = positions[:, 0]
    y = positions[:, 1]
    hist_range = [[-half, half], [-half, half]]

    occupancy, xedges, yedges = np.histogram2d(
        x, y, bins=bins, range=hist_range,
    )
    weighted, _, _ = np.histogram2d(
        x, y, bins=bins, range=hist_range, weights=activations,
    )

    # Adaptive smoothing: smooth numerator and denominator separately
    if smooth_sigma > 0:
        occupancy = gaussian_filter(occupancy.astype(np.float64), smooth_sigma)
        weighted = gaussian_filter(weighted.astype(np.float64), smooth_sigma)

    # Build rate map with masking
    low_occ = occupancy < max(min_occupancy, 1e-6)
    rate_map = np.where(low_occ, 0.0, weighted / np.maximum(occupancy, 1e-6))
    if min_occupancy > 0:
        rate_map = np.where(low_occ, np.nan, rate_map)

    return rate_map, xedges, yedges


# =========================================================================
# Occupancy map (standalone, useful when spike map is built separately)
# =========================================================================

def compute_occupancy_map(
    positions: np.ndarray,
    bins: int = 32,
    arena_size: float = 2.2,
    smooth_sigma: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a 2-D time/count occupancy map.

    Returns
    -------
    occupancy : (bins, bins) ndarray — visit count per bin.
    xedges, yedges : bin edges.
    """
    half = arena_size / 2.0
    hist_range = [[-half, half], [-half, half]]
    occ, xedges, yedges = np.histogram2d(
        positions[:, 0], positions[:, 1], bins=bins, range=hist_range,
    )
    if smooth_sigma > 0:
        occ = gaussian_filter(occ.astype(np.float64), smooth_sigma)
    return occ, xedges, yedges


# =========================================================================
# Rate map coherence  (Muller & Kubie 1989)
# =========================================================================

def rate_map_coherence(rate_map_raw: np.ndarray) -> float:
    """Coherence of a *non-smoothed* rate map.

    Computes the Pearson correlation between each bin's firing rate
    and the mean firing rate of its eight immediate neighbours.

    Parameters
    ----------
    rate_map_raw : 2-D ndarray (may contain ``np.nan`` for unvisited bins).

    Returns
    -------
    coherence : float in [-1, 1].

    Reference
    ---------
    Muller & Kubie, J. Neuroscience 1989 9(12):4101-4110.
    """
    kernel = np.array(
        [[0.125, 0.125, 0.125],
         [0.125, 0.0,   0.125],
         [0.125, 0.125, 0.125]],
    )
    rmap = np.nan_to_num(np.asarray(rate_map_raw, dtype=np.float64))

    # Neighbour-average map (zero-padded borders)
    from scipy.signal import convolve2d
    avg_map = convolve2d(rmap, kernel, mode="same", boundary="fill", fillvalue=0.0)

    avg_flat = avg_map.ravel()
    rmap_flat = rmap.ravel()
    cc = np.corrcoef(avg_flat, rmap_flat)
    return float(cc[0, 1])


# =========================================================================
# Rate map statistics  (Skaggs et al. 1993 / 1996)
# =========================================================================

def rate_map_stats(
    rate_map: np.ndarray,
    occupancy_map: np.ndarray,
) -> Dict[str, float]:
    """Spatial information, sparsity, selectivity of a rate map.

    Parameters
    ----------
    rate_map : (bins, bins) — smoothed firing-rate map.  May contain
        ``np.nan`` at unvisited bins.
    occupancy_map : (bins, bins) — time (or count) spent in each bin.
        Same shape as *rate_map*.

    Returns
    -------
    dict with keys:
        spatial_information_rate   — bits / sec
        spatial_information_content — bits / spike
        sparsity
        selectivity
        peak_rate
        mean_rate
    """
    rmap = np.ma.masked_invalid(np.asarray(rate_map, dtype=np.float64))
    tmap = np.ma.masked_invalid(np.asarray(occupancy_map, dtype=np.float64))

    duration = np.ma.sum(tmap)
    pos_pdf = tmap / (duration + np.spacing(1))

    if rmap.mask.all():
        return {
            "spatial_information_rate": np.nan,
            "spatial_information_content": np.nan,
            "sparsity": np.nan,
            "selectivity": np.nan,
            "peak_rate": np.nan,
            "mean_rate": np.nan,
        }

    mean_rate = float(np.ma.sum(rmap * pos_pdf))
    mean_rate_sq = float(np.ma.sum(np.ma.power(rmap, 2) * pos_pdf))
    max_rate = float(np.nanmax(rmap))
    rmap_mean = float(np.nanmean(rmap))
    rmap_peak = float(np.nanmax(rmap))

    sparsity = (mean_rate ** 2 / mean_rate_sq) if mean_rate_sq != 0 else np.nan
    selectivity = (max_rate / mean_rate) if mean_rate != 0 else np.nan

    inf_rate = np.nan
    inf_content = np.nan
    if mean_rate > 0:
        log_arg = np.ma.array(rmap / mean_rate)
        log_arg[log_arg < 1] = 1
        inf_rate = float(np.ma.sum(pos_pdf * rmap * np.ma.log2(log_arg)))
        inf_content = inf_rate / mean_rate

    return {
        "spatial_information_rate": float(inf_rate),
        "spatial_information_content": float(inf_content),
        "sparsity": float(sparsity),
        "selectivity": float(selectivity),
        "peak_rate": rmap_peak,
        "mean_rate": rmap_mean,
    }
