"""Google/Banino-style grid-score diagnostic.

This module intentionally mirrors the mask-sweep scorer used in the original
Google grid-cell codebase.  It is not meant to replace the stricter opexebo/BNT
score in :mod:`gpm_cmap.analysis.grid_score`; it is a compatibility diagnostic
for checking whether historical "40% cells > 0.3" claims depend on scorer
choice.
"""
from __future__ import annotations

import math
from typing import Dict, Sequence, Tuple

import numpy as np
from scipy.ndimage import rotate as ndrotate
from scipy.signal import convolve2d


def _circle_mask(shape: Tuple[int, int], radius: float,
                 in_val: float = 1.0, out_val: float = 0.0) -> np.ndarray:
    cy, cx = math.floor(shape[0] / 2), math.floor(shape[1] / 2)
    # Match the original scorer's coordinate convention: nbins*2-1 SAC with
    # coordinate range approximately [-center, center].
    x = np.linspace(-cy, cx, shape[1])[None, :].repeat(shape[0], axis=0)
    y = np.linspace(-cy, cx, shape[0])[:, None].repeat(shape[1], axis=1)
    z = np.sqrt(x ** 2 + y ** 2)
    return np.where(z <= radius, in_val, out_val)


def _calculate_sac(rate_map: np.ndarray) -> np.ndarray:
    seq1 = np.asarray(rate_map, dtype=np.float64).copy()
    seq2 = seq1.copy()

    ones_seq1 = np.ones(seq1.shape, dtype=np.float64)
    ones_seq1[np.isnan(seq1)] = 0
    ones_seq2 = np.ones(seq2.shape, dtype=np.float64)
    ones_seq2[np.isnan(seq2)] = 0

    seq1[np.isnan(seq1)] = 0
    seq2[np.isnan(seq2)] = 0

    def filt(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return convolve2d(a, np.rot90(b, 2), mode="full")

    seq1_sq = np.square(seq1)
    seq2_sq = np.square(seq2)
    seq1_x_seq2 = filt(seq1, seq2)
    sum_seq1 = filt(seq1, ones_seq2)
    sum_seq2 = filt(ones_seq1, seq2)
    sum_seq1_sq = filt(seq1_sq, ones_seq2)
    sum_seq2_sq = filt(ones_seq1, seq2_sq)
    n_bins = filt(ones_seq1, ones_seq2)
    n_bins_sq = np.square(n_bins)

    std_seq1 = np.sqrt(np.maximum(sum_seq1_sq / n_bins - (sum_seq1 ** 2) / n_bins_sq, 0.0))
    std_seq2 = np.sqrt(np.maximum(sum_seq2_sq / n_bins - (sum_seq2 ** 2) / n_bins_sq, 0.0))
    covar = seq1_x_seq2 / n_bins - (sum_seq1 * sum_seq2) / n_bins_sq
    denom = std_seq1 * std_seq2
    x_coef = np.divide(covar, denom, out=np.zeros_like(covar), where=denom > 1e-12)
    return np.real(np.nan_to_num(x_coef))


def _grid_score_60(corr: Dict[int, float], min_max: bool = False) -> float:
    if min_max:
        return float(min(corr[60], corr[120]) - max(corr[30], corr[90], corr[150]))
    return float((corr[60] + corr[120]) / 2.0 - (corr[30] + corr[90] + corr[150]) / 3.0)


def compute_google_grid_score(
    rate_map: np.ndarray,
    bins: int | None = None,
    mask_parameters: Sequence[Tuple[float, float]] | None = None,
    min_max: bool = False,
) -> Dict[str, object]:
    """Compute the original Google/Banino mask-sweep 60-degree grid score."""
    nbins = int(bins if bins is not None else rate_map.shape[0])
    if mask_parameters is None:
        mask_parameters = [(0.2, float(x)) for x in np.linspace(0.4, 1.0, num=10)]

    sac = _calculate_sac(rate_map)
    angles = [30, 45, 60, 90, 120, 135, 150]
    rotated = {angle: ndrotate(sac, angle, reshape=False, order=1) for angle in angles}

    best_score = -np.inf
    best_mask = None
    best_corr = None
    best_idx = -1

    for i, (mask_min, mask_max) in enumerate(mask_parameters):
        mask = (_circle_mask(sac.shape, mask_max * nbins) *
                (1.0 - _circle_mask(sac.shape, mask_min * nbins)))
        ring_area = float(mask.sum())
        if ring_area < 6:
            continue
        masked_sac = sac * mask
        masked_sac_mean = float(masked_sac.sum() / ring_area)
        masked_sac_centered = (masked_sac - masked_sac_mean) * mask
        variance = float((masked_sac_centered ** 2).sum() / ring_area + 1e-5)
        corr = {}
        for angle in angles:
            masked_rotated_sac = (rotated[angle] - masked_sac_mean) * mask
            cross_prod = float((masked_sac_centered * masked_rotated_sac).sum() / ring_area)
            corr[angle] = cross_prod / variance
        score = _grid_score_60(corr, min_max=min_max)
        if score > best_score:
            best_score = score
            best_mask = (float(mask_min), float(mask_max))
            best_corr = corr
            best_idx = i

    if best_mask is None:
        return {
            "grid_score_google": float("nan"),
            "best_mask_google": (0.0, 0.0),
            "corrs_google": {a: 0.0 for a in angles},
            "autocorr_google": sac,
            "best_mask_index_google": -1,
        }
    return {
        "grid_score_google": float(best_score),
        "best_mask_google": best_mask,
        "corrs_google": best_corr,
        "autocorr_google": sac,
        "best_mask_index_google": int(best_idx),
    }
