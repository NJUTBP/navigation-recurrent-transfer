"""Grid-score computation.

Upgraded following opexebo conventions (Sargolini et al. / Hafting et al.):

1. **SAC** — local-mean normalised spatial autocorrelogram (unchanged).
2. **Grid score** — Pearson-correlation of annular SAC ring with its
   rotated copies at 30/60/90/120/150°.  The ring is expanded from the
   centre-field radius to the half-width of the autocorrelogram, and the
   final score is the maximum of a sliding-mean window over those radii.
   This matches the opexebo / BNT reference implementation.
3. **Grid statistics** — spacing, orientation, peak positions, and
   ellipse fit of the six closest fields (ported from opexebo).

All public call signatures are backward-compatible.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import rotate as ndrotate
from scipy.signal import convolve2d
from scipy.stats import pearsonr
from scipy.ndimage import maximum_filter, label as ndlabel

from .rate_map import compute_rate_map


# =========================================================================
#  Spatial autocorrelogram  (reference-style, local-mean normalisation)
# =========================================================================

def spatial_autocorrelogram(rate_map: np.ndarray) -> np.ndarray:
    """Compute the normalised spatial autocorrelogram of a 2-D rate map.

    Uses the convolution-based local-mean normalisation that is standard
    in the grid-cell literature (Langston et al. 2010, etc.).
    """
    seq = np.nan_to_num(np.asarray(rate_map, dtype=np.float64))
    ones = np.ones_like(seq)

    def _filt(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return convolve2d(a, np.rot90(b, 2), mode="full")

    n = _filt(ones, ones)
    n_sq = n ** 2

    cross = _filt(seq, seq)
    s1 = _filt(seq, ones)
    s2 = _filt(ones, seq)
    sq1 = _filt(seq ** 2, ones)
    sq2 = _filt(ones, seq ** 2)

    std1 = np.sqrt(np.maximum(sq1 / n - (s1 / n) ** 2, 0.0))
    std2 = np.sqrt(np.maximum(sq2 / n - (s2 / n) ** 2, 0.0))
    covar = cross / n - s1 * s2 / n_sq
    denom = std1 * std2
    sac = np.divide(covar, denom, out=np.zeros_like(covar), where=denom > 1e-12)
    return np.real(np.nan_to_num(sac))


# =========================================================================
#  Low-level helpers
# =========================================================================

def _circle_mask(shape: Tuple[int, int], radius: float,
                 in_val: float = 1.0, out_val: float = 0.0) -> np.ndarray:
    cy, cx = shape[0] // 2, shape[1] // 2
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    return np.where(rr <= radius, in_val, out_val)


def _ring_mask(shape: Tuple[int, int], inner: float, outer: float) -> np.ndarray:
    return _circle_mask(shape, outer) * (1.0 - _circle_mask(shape, inner))


def _find_centre_radius(sac: np.ndarray, threshold: float = 0.3) -> int:
    """Estimate the radius of the central peak of the SAC.

    Walks outward from the centre until the SAC drops below *threshold*
    of its peak value.
    """
    h, w = sac.shape
    cy, cx = h // 2, w // 2
    peak_val = sac[cy, cx]
    if peak_val <= 0:
        return 2
    thresh = peak_val * threshold
    yy, xx = np.ogrid[:h, :w]
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    max_r = min(h, w) // 2
    for r in range(1, max_r):
        ring = (rr >= r - 0.5) & (rr < r + 0.5)
        if ring.sum() == 0:
            continue
        if sac[ring].mean() < thresh:
            return max(r, 2)
    return max(2, max_r // 4)


# =========================================================================
#  Radial-profile grid spacing  (Sargolini 2006 / Stensola 2012)
# =========================================================================

def sac_radial_profile(sac: np.ndarray) -> np.ndarray:
    """Angle-averaged SAC as a function of radius from centre.

    For a hexagonal grid the first ring of 6 peaks all sit at the same
    radius, so the radial profile has a clear peak at the grid spacing.
    This 1-D representation is far more robust than searching for
    individual 2-D peaks.
    """
    h, w = sac.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    max_r = int(min(h, w) // 2)
    profile = np.zeros(max_r, dtype=np.float64)
    for r in range(max_r):
        ring = (rr >= r - 0.5) & (rr < r + 0.5)
        if ring.sum() > 0:
            profile[r] = sac[ring].mean()
    return profile


def estimate_grid_spacing(
    sac: np.ndarray,
    min_centre_r: int = 2,
    centre_frac: float = 0.2,
) -> Tuple[float, float, bool]:
    """Estimate grid spacing and orientation from a SAC.

    Uses the **radial profile** method (Sargolini et al. 2006;
    Stensola et al. 2012):

    1. Compute angle-averaged SAC vs radius.
    2. Detect centre-field radius (where profile drops below
       *centre_frac* × peak).
    3. ``scipy.signal.find_peaks`` on the profile beyond the centre
       → first peak = grid spacing.
    4. Set a narrow annular shell at that spacing, find 2-D peaks,
       take median angle mod 60° → orientation.

    Parameters
    ----------
    sac : 2-D spatial autocorrelogram.
    min_centre_r : minimum centre radius.
    centre_frac : threshold fraction of peak for centre detection.

    Returns
    -------
    spacing : float — grid period in SAC-pixel units.
    orientation : float — degrees in [0, 60).
    valid : bool
    """
    from scipy.signal import find_peaks as _fp
    from scipy.ndimage import gaussian_filter1d

    sac = np.asarray(sac, dtype=np.float64)
    if sac.ndim != 2 or min(sac.shape) < 9:
        return 0.0, 0.0, False

    h, w = sac.shape
    cy, cx = h // 2, w // 2

    profile = sac_radial_profile(sac)
    centre_r = _find_centre_radius(sac, threshold=centre_frac)
    centre_r = max(centre_r, min_centre_r)

    sub = profile[centre_r:]
    if len(sub) < 4:
        return 0.0, 0.0, False

    sub_smooth = gaussian_filter1d(sub, sigma=0.8)
    peaks_1d, _ = _fp(sub_smooth, distance=max(2, centre_r // 2), prominence=0.01)
    if len(peaks_1d) == 0:
        peaks_1d = np.array([np.argmax(sub_smooth)])

    spacing = float(peaks_1d[0] + centre_r)
    if spacing < centre_r + 1:
        return 0.0, 0.0, False

    # 2-D peaks in annular shell
    yy, xx = np.ogrid[:h, :w]
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    sw = max(2.0, 0.25 * spacing)
    shell = (rr >= spacing - sw) & (rr <= spacing + sw)
    mf = maximum_filter(sac, size=5, mode="nearest")
    pk2d = (sac == mf) & shell & (sac > 0.0)
    coords = np.argwhere(pk2d)

    if len(coords) < 3:
        return spacing, 0.0, True

    dy = coords[:, 0] - cy
    dx = coords[:, 1] - cx
    pr = np.sqrt(dy ** 2 + dx ** 2)
    pa = np.arctan2(dy, dx)
    order = np.argsort(np.abs(pr - spacing))
    pa = pa[order]

    # De-duplicate within 15°
    kept = []
    for a in pa:
        if all(abs(np.angle(np.exp(1j * a) / np.exp(1j * k))) > np.radians(15) for k in kept):
            kept.append(a)
        if len(kept) >= 6:
            break

    if len(kept) < 3:
        return spacing, 0.0, True

    orient = float(np.median(np.degrees(np.array(kept)) % 180.0 % 60.0))
    return spacing, orient, True


def _select_best_gridness_window(values: np.ndarray, width: int) -> Dict[str, object]:
    """Select a best contiguous gridness window and its terminal radius index.

    The score, returned mask radius and rotation correlations must refer to the
    same radius selection.  ``len(values)-width+1`` windows are legal; omitting
    the final one biases the result whenever gridness improves at larger radii.
    """
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size == 0:
        return {"start": 0, "end": 0, "score": float("nan"), "all_scores": np.asarray([])}
    width = max(1, min(int(width), values.size))
    scores = np.asarray(
        [np.nanmean(values[start:start + width]) for start in range(values.size - width + 1)],
        dtype=float,
    )
    if np.all(np.isnan(scores)):
        start = 0
        score = float("nan")
    else:
        start = int(np.nanargmax(scores))
        score = float(scores[start])
    return {
        "start": start,
        "end": start + width - 1,
        "score": score,
        "all_scores": scores,
    }


# =========================================================================
#  Core grid score — opexebo / BNT reference style
# =========================================================================

def compute_grid_score(
    rate_map: np.ndarray,
    bins: int | None = None,
    mask_parameters: Sequence[Tuple[float, float]] | None = None,
    use_min_max: bool = False,
    num_gridness_radii: int = 3,
) -> Dict[str, object]:
    """Compute the grid score of a single rate map.

    Algorithm (matching opexebo / BNT):
      1. Build the spatial autocorrelogram (SAC).
      2. Find the radius of the central peak.
      3. For every integer radius from ``centre_radius + 1`` to the outer
         bound, mask an annular ring in the SAC, extract the values inside,
         and compute Pearson correlations with the same ring rotated by
         30 / 60 / 90 / 120 / 150°.
      4. ``GNS[r] = min(corr_60, corr_120) − max(corr_30, corr_90, corr_150)``
      5. Final score = max of a sliding mean of width *num_gridness_radii*
         over the radius series.

    If *mask_parameters* is supplied (list of ``(inner_frac, outer_frac)``
    pairs), the old multi-annular-mask sweep is used instead (backward
    compat).

    Parameters
    ----------
    rate_map : (N, N) ndarray
    bins : int or None — treated as the spatial resolution (defaults to
        ``rate_map.shape[0]``).
    mask_parameters : optional — legacy parameter.  When ``None`` the
        opexebo expanding-circle method is used (recommended).
    use_min_max : bool — if True use ``min - max``; if False (default) use
        ``mean(60,120) − mean(30,90,150)``.
    num_gridness_radii : int — sliding-window width for the final score.

    Returns
    -------
    dict with keys:
        grid_score   — float
        autocorr     — 2-D SAC
        corrs        — dict {angle: correlation} at the best radius
        best_mask    — (inner, outer) fractions (legacy) or (centre_r, best_r)
        grid_stats   — dict with spacing, orientation, positions info
    """
    sac = spatial_autocorrelogram(rate_map)
    nbins = int(bins if bins is not None else rate_map.shape[0])

    # ------------------------------------------------------------------
    # Legacy multi-annular-mask sweep (backward compat)
    # ------------------------------------------------------------------
    if mask_parameters is not None:
        return _grid_score_legacy(sac, nbins, list(mask_parameters), use_min_max)

    # ------------------------------------------------------------------
    # opexebo-style expanding-circle method
    # ------------------------------------------------------------------
    centre_r = _find_centre_radius(sac)
    outer_bound = min(sac.shape[0], sac.shape[1]) // 2
    if outer_bound <= centre_r:
        return _empty_result(sac)

    radii = np.arange(max(3, centre_r + 1), outer_bound + 1, dtype=int)
    if len(radii) < 1:
        return _empty_result(sac)

    rot_angles = [30, 60, 90, 120, 150]
    # Pre-rotate the full SAC
    rotated_sacs = {}
    for angle in rot_angles:
        rotated_sacs[angle] = ndrotate(sac, angle, reshape=False, order=1)

    h, w = sac.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    inner_circle = rr > centre_r

    GNS = np.full(len(radii), np.nan)
    corrs_per_radius = []
    GNS_conservative = np.full(len(radii), np.nan)  # T8: parallel min-max gridness
    for i, radius in enumerate(radii):
        mask = inner_circle & (rr < radius)
        if mask.sum() < 6:
            corrs_per_radius.append({a: 0.0 for a in rot_angles})
            continue
        sac_vals = sac[mask]
        corrs = {}
        for angle in rot_angles:
            rot_vals = rotated_sacs[angle][mask]
            if np.std(sac_vals) < 1e-12 or np.std(rot_vals) < 1e-12:
                corrs[angle] = 0.0
            else:
                corrs[angle], _ = pearsonr(sac_vals, rot_vals)
        corrs_per_radius.append(corrs)
        if use_min_max:
            GNS[i] = min(corrs[60], corrs[120]) - max(corrs[30], corrs[90], corrs[150])
        else:
            GNS[i] = (corrs[60] + corrs[120]) / 2.0 - (corrs[30] + corrs[90] + corrs[150]) / 3.0
        # T8: always also track the conservative min-max gridness from the SAME
        # correlations, so callers can report it alongside the (liberal) default
        # without a second SAC pass and without changing the default behaviour.
        GNS_conservative[i] = (
            min(corrs[60], corrs[120]) - max(corrs[30], corrs[90], corrs[150])
        )

    # Select the sliding window once and use its terminal radius for every
    # radius-dependent artifact (mask, correlations and grid statistics).
    primary_window = _select_best_gridness_window(GNS, num_gridness_radii)
    best_idx = int(primary_window["end"])
    gscore = float(primary_window["score"])
    best_corrs = corrs_per_radius[best_idx] if corrs_per_radius else {a: 0.0 for a in rot_angles}

    conservative_window = _select_best_gridness_window(
        GNS_conservative, num_gridness_radii
    )
    gscore_conservative = float(conservative_window["score"])

    # Grid statistics
    best_radius = int(radii[best_idx]) if len(radii) > 0 else outer_bound
    grid_stats = _compute_grid_stats(sac, centre_r, best_radius)

    return {
        "grid_score": float(gscore),
        "grid_score_conservative": gscore_conservative,
        "autocorr": sac,
        "corrs": best_corrs,
        "best_mask": (float(centre_r), float(best_radius)),
        "best_window": {
            "start": int(primary_window["start"]),
            "end": int(primary_window["end"]),
            "scores": primary_window["all_scores"],
        },
        "grid_stats": grid_stats,
    }


# =========================================================================
#  Grid statistics — spacing, orientation, positions  (from opexebo)
# =========================================================================

def _compute_grid_stats(
    sac: np.ndarray,
    centre_r: int,
    best_radius: int,
) -> Dict[str, object]:
    """Extract spatial grid statistics from the SAC.

    Finds the six closest local maxima to the SAC centre (excluding the
    centre field), computes their distances (spacing), orientations, and
    an ellipse fit.
    """
    h, w = sac.shape
    cy, cx = h // 2, w // 2

    # Mask centre and fringes
    yy, xx = np.ogrid[:h, :w]
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    mask_keep = (rr > centre_r * 1.5) & (rr < best_radius * 1.25)
    masked_sac = sac * mask_keep

    # Peak detection
    mf = maximum_filter(masked_sac, size=3, mode="nearest")
    peaks = (masked_sac == mf) & (masked_sac > 0.05) & mask_keep
    coords = np.argwhere(peaks)  # (n_peaks, 2)  — [row, col]

    nan3 = np.full(3, np.nan)
    nan6 = np.full(6, np.nan)
    default_stats: Dict[str, object] = {
        "grid_spacings": nan3.copy(),
        "grid_spacing": np.nan,
        "grid_orientations": nan3.copy(),
        "grid_orientations_std": np.nan,
        "grid_orientation": np.nan,
        "grid_positions": nan6.copy(),
        "grid_ellipse": np.nan,
        "grid_ellipse_aspect_ratio": np.nan,
        "grid_ellipse_theta": np.nan,
    }

    if len(coords) < 6:
        return default_stats

    # Distances and orientations from centre
    dy = coords[:, 0] - cy
    dx = coords[:, 1] - cx
    distances = np.sqrt(dy ** 2 + dx ** 2)
    orientations = np.arctan2(dy, dx)  # radians

    # Remove duplicate orientations (keep closer one)
    distances, orientations, coords = _remove_close_orientations(
        distances, orientations, coords, min_angle_rad=np.radians(10.0),
    )

    if len(distances) < 6:
        return default_stats

    # Take six closest
    order = np.argsort(distances)[:6]
    positions = coords[order]
    dists = distances[order]
    orients = orientations[order]

    # Re-sort by angle
    ang_order = np.argsort(orients)

    stats = dict(default_stats)
    stats["grid_positions"] = positions[ang_order]
    stats["grid_spacings"] = dists[ang_order[:3]]
    stats["grid_spacing"] = float(np.nanmean(dists[ang_order[:3]]))
    orient_deg = np.degrees(orients[ang_order[:3]]) % 180.0
    stats["grid_orientations"] = orient_deg
    mean_orient, orient_std = _extract_grid_orientation(orient_deg)
    stats["grid_orientation"] = float(mean_orient)
    stats["grid_orientations_std"] = float(orient_std)

    # Ellipse fit (best effort)
    if len(positions) >= 3:
        try:
            ell = _fit_ellipse_simple(positions[ang_order, 1].astype(float),
                                      positions[ang_order, 0].astype(float))
            stats["grid_ellipse"] = ell
            if ell is not None and len(ell) == 5:
                stats["grid_ellipse_theta"] = float(np.degrees(ell[4] + np.pi) % 360)
                stats["grid_ellipse_aspect_ratio"] = float(ell[2] / max(ell[3], 1e-8))
        except Exception:
            pass

    return stats


def _remove_close_orientations(distances, orientations, coords, min_angle_rad):
    """Remove peaks whose angular separation is too small, keeping the closer one.

    Returns filtered (distances, orientations, coords) arrays.
    """
    n = len(distances)
    to_del = set()
    for i in range(n):
        for j in range(i + 1, n):
            if abs(_circ_dist(orientations[i], orientations[j])) < min_angle_rad:
                if distances[i] > distances[j]:
                    to_del.add(i)
                else:
                    to_del.add(j)
    keep = sorted(set(range(n)) - to_del)
    return distances[keep], orientations[keep], coords[keep]


def _circ_dist(a: float, b: float) -> float:
    return np.angle(np.exp(1j * a) / np.exp(1j * b))


def _extract_grid_orientation(orientations_deg: np.ndarray):
    """Extract mean grid orientation from the three closest field angles."""
    o60 = orientations_deg % 60
    corr = []
    for v in o60:
        d60 = v - 60
        corr.append(d60 if abs(d60) < abs(v) else v)
    corr = np.array(corr)
    if np.mean(np.abs(np.abs(corr) - 30)) < np.mean(np.abs(corr)):
        if np.median(corr) < 0:
            corr = np.where(corr > 0, -corr, corr)
        else:
            corr = np.where(corr < 0, -corr, corr)
    mean_o = float(np.nanmean(corr))
    std_o = float(np.nanstd(corr))
    if abs(mean_o - 60) < abs(mean_o):
        mean_o -= 60
    return mean_o, std_o


def _fit_ellipse_simple(x, y):
    """Minimal ellipse fit returning (cx, cy, a, b, theta)."""
    try:
        from numpy.linalg import eig, inv
        D = np.column_stack([x ** 2, x * y, y ** 2, x, y, np.ones_like(x)])
        S = D.T @ D
        C = np.zeros((6, 6))
        C[0, 2] = C[2, 0] = 2
        C[1, 1] = -1
        eigvals, eigvecs = eig(inv(S) @ C)
        idx = np.argmax(np.real(eigvals[np.isfinite(eigvals)]))
        a_coeff = np.real(eigvecs[:, idx])
        # Recover parameters
        a, b, c, d, e, f = a_coeff
        cx = (b * e - 2 * c * d) / (4 * a * c - b ** 2 + 1e-12)
        cy = (b * d - 2 * a * e) / (4 * a * c - b ** 2 + 1e-12)
        theta = 0.5 * np.arctan2(b, a - c)
        ct, st = np.cos(theta), np.sin(theta)
        ap = a * ct ** 2 + b * ct * st + c * st ** 2
        cp = a * st ** 2 - b * ct * st + c * ct ** 2
        val = a_coeff @ np.array([cx ** 2, cx * cy, cy ** 2, cx, cy, 1.0])
        ra = np.sqrt(abs(-val / (ap + 1e-12)))
        rb = np.sqrt(abs(-val / (cp + 1e-12)))
        if ra < rb:
            ra, rb = rb, ra
            theta += np.pi / 2
        return (cx, cy, ra, rb, theta)
    except Exception:
        return None


# =========================================================================
#  Legacy multi-annular-mask sweep  (backward compat)
# =========================================================================

def _grid_score_legacy(sac, nbins, mask_params, use_min_max):
    rot_angles = [30, 60, 90, 120, 150]
    rotated_sacs = {a: ndrotate(sac, a, reshape=False, order=1) for a in rot_angles}

    best_score = -np.inf
    best_corrs = None
    best_mask_params = None

    for inner_rel, outer_rel in mask_params:
        ring = _ring_mask(sac.shape, inner=inner_rel * nbins, outer=outer_rel * nbins)
        ring_area = ring.sum()
        if ring_area < 6:
            continue
        sac_vals = sac[ring > 0.5]
        corrs = {}
        for angle in rot_angles:
            rot_vals = rotated_sacs[angle][ring > 0.5]
            if np.std(sac_vals) < 1e-12 or np.std(rot_vals) < 1e-12:
                corrs[angle] = 0.0
            else:
                corrs[angle], _ = pearsonr(sac_vals, rot_vals)
        if use_min_max:
            score = min(corrs[60], corrs[120]) - max(corrs[30], corrs[90], corrs[150])
        else:
            score = (corrs[60] + corrs[120]) / 2.0 - (corrs[30] + corrs[90] + corrs[150]) / 3.0
        if score > best_score:
            best_score = float(score)
            best_corrs = corrs
            best_mask_params = (float(inner_rel), float(outer_rel))

    if best_corrs is None:
        return _empty_result(sac)
    return {
        "grid_score": float(best_score),
        # Legacy SAC path does not track both gridness variants in parallel;
        # expose the conservative score only when this path was run in min-max
        # mode, else NaN so downstream code can detect its absence (rule #12).
        "grid_score_conservative": (float(best_score) if use_min_max else float("nan")),
        "autocorr": sac,
        "corrs": best_corrs,
        "best_mask": best_mask_params,
        "grid_stats": {},
    }


def _empty_result(sac):
    return {
        "grid_score": float("nan"),
        "grid_score_conservative": float("nan"),
        "autocorr": sac,
        "corrs": {a: 0.0 for a in [30, 60, 90, 120, 150]},
        "best_mask": (0.0, 0.0),
        "grid_stats": {},
    }


# =========================================================================
#  Batch helper
# =========================================================================

def batch_grid_scores(
    positions: np.ndarray,
    hidden_activations: np.ndarray,
    bins: int = 20,
    arena_size: float = 2.2,
    smooth_sigma: float = 0.0,
    mask_parameters: Sequence[Tuple[float, float]] | None = None,
    include_google_ref: bool = False,
) -> List[Dict[str, object]]:
    """Compute grid scores for every hidden unit.

    Parameters
    ----------
    positions : (N, 2) trajectory positions.
    hidden_activations : (N, H) per-step hidden-unit activations.
    bins, arena_size, smooth_sigma : forwarded to ``compute_rate_map``.
    mask_parameters : optional — forwarded to ``compute_grid_score``.
    include_google_ref : bool — if True, also compute the original Google/Banino
        mask-sweep score and attach ``grid_score_google`` to each unit result.

    Returns
    -------
    list of dicts, one per unit, each containing the fields returned by
    ``compute_grid_score`` plus ``rate_map`` and ``unit``.
    """
    results = []
    for unit in range(hidden_activations.shape[1]):
        rm, _, _ = compute_rate_map(
            positions=positions,
            activations=hidden_activations[:, unit],
            bins=bins,
            arena_size=arena_size,
            smooth_sigma=smooth_sigma,
        )
        res = compute_grid_score(rm, bins=bins, mask_parameters=mask_parameters)
        if include_google_ref:
            from gpm_cmap.analysis.grid_score_google import compute_google_grid_score
            res.update(compute_google_grid_score(rm, bins=bins))
        res["rate_map"] = rm
        res["unit"] = unit
        results.append(res)
    return results