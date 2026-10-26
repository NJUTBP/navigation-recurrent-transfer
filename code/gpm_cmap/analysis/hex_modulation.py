"""Six-fold (hexagonal) modulation analysis.

Core function
    sixfold_modulation_strength(angles, activity) -> float

Batch analysis
    batch_hex_modulation(directions, hidden, top_k) -> dict

Plotting
    plot_hex_modulation_polar(angles, activity, unit_id, path) -> None
    plot_hex_summary(modulation_strengths, path) -> None
"""
from __future__ import annotations
from typing import Dict, List, Optional
from pathlib import Path

import numpy as np


def sixfold_modulation_strength(angles: np.ndarray, activity: np.ndarray) -> float:
    """Compute |A_6| / mean(|A_k|) for a single unit."""
    angles = np.asarray(angles, dtype=np.float64)
    activity = np.asarray(activity, dtype=np.float64)
    if len(angles) != len(activity):
        raise ValueError("angles and activity must have the same length")
    if len(angles) < 6:
        return 0.0
    coeff6 = np.abs(np.mean(activity * np.exp(1j * 6 * angles)))
    norm = np.mean(np.abs(activity)) + 1e-8
    return float(coeff6 / norm)


def batch_hex_modulation(
    directions: np.ndarray,
    hidden: np.ndarray,
    top_k: int = 20,
) -> Dict[str, object]:
    """Compute six-fold modulation for all hidden units.

    Parameters
    ----------
    directions : (N,) array of movement direction angles (radians).
    hidden : (N, H) array of hidden unit activations at each step.
    top_k : int
        Number of top units to return detailed results for.

    Returns
    -------
    dict with keys:
        - ``strengths`` : (H,) array of modulation strength per unit
        - ``mean`` : scalar
        - ``max`` : scalar
        - ``top_units`` : list of ``(unit_idx, strength)``
        - ``significant_count`` : how many units exceed the shuffle p<0.05 threshold
    """
    n_steps, n_units = hidden.shape
    strengths = np.zeros(n_units)
    for u in range(n_units):
        strengths[u] = sixfold_modulation_strength(directions, hidden[:, u])

    # simple shuffle test for significance threshold
    n_shuffles = 200
    shuffle_max = np.zeros(n_shuffles)
    rng = np.random.default_rng(0)
    for s in range(n_shuffles):
        perm = rng.permutation(n_steps)
        shuffled_dirs = directions[perm]
        shuffle_max[s] = max(
            sixfold_modulation_strength(shuffled_dirs, hidden[:, u])
            for u in range(min(n_units, 50))  # subsample for speed
        )
    threshold_95 = float(np.percentile(shuffle_max, 95))

    sorted_idx = np.argsort(strengths)[::-1]
    top_units = [(int(sorted_idx[i]), float(strengths[sorted_idx[i]])) for i in range(min(top_k, n_units))]

    return {
        "strengths": strengths,
        "mean": float(np.mean(strengths)),
        "max": float(np.max(strengths)),
        "top_units": top_units,
        "threshold_95": threshold_95,
        "significant_count": int(np.sum(strengths > threshold_95)),
    }


# -----------------------------------------------------------------------
#  Plotting
# -----------------------------------------------------------------------

def plot_hex_modulation_polar(
    angles: np.ndarray,
    activity: np.ndarray,
    unit_id: int,
    output_path: Path,
    n_bins: int = 36,
) -> None:
    """Polar plot of a single unit's activity vs movement direction."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bin_edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    means = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (angles >= bin_edges[i]) & (angles < bin_edges[i + 1])
        if mask.sum() > 0:
            means[i] = activity[mask].mean()

    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(5, 5))
    # close the polar curve
    theta = np.concatenate([bin_centers, [bin_centers[0]]])
    r = np.concatenate([means, [means[0]]])
    ax.plot(theta, r, linewidth=2)
    ax.fill(theta, r, alpha=0.25)
    strength = sixfold_modulation_strength(angles, activity)
    ax.set_title(f"Unit {unit_id}  |  hex mod = {strength:.3f}", pad=15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_hex_summary(
    strengths: np.ndarray,
    output_path: Path,
    threshold: Optional[float] = None,
) -> None:
    """Histogram of six-fold modulation strengths across all units."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 4))
    plt.hist(strengths, bins=30, edgecolor="white", alpha=0.8)
    if threshold is not None:
        plt.axvline(threshold, color="red", linestyle="--", label=f"p<0.05 ({threshold:.3f})")
        plt.legend()
    plt.xlabel("Six-fold modulation strength")
    plt.ylabel("# hidden units")
    plt.title("Hexagonal modulation distribution")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
