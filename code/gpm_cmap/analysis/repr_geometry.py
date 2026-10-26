from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from sklearn.decomposition import PCA


def pairwise_distance_correlation(representations: np.ndarray, coordinates: np.ndarray) -> float:
    rep_d = pdist(representations, metric="euclidean")
    coord_d = pdist(coordinates, metric="euclidean")
    if len(rep_d) < 2 or len(coord_d) < 2:
        return float("nan")
    if not np.isfinite(rep_d).all() or not np.isfinite(coord_d).all():
        raise FloatingPointError("Non-finite pairwise distances in geometry metric")
    # Spearman correlation is undefined when either distance vector is
    # constant.  Check explicitly so SciPy does not emit a warning that may be
    # promoted to an exception, and report the metric as N/A rather than 0.
    if np.ptp(rep_d) == 0 or np.ptp(coord_d) == 0:
        return float("nan")
    correlation = float(spearmanr(rep_d, coord_d).correlation)
    return correlation if np.isfinite(correlation) else float("nan")


def pca_projection(representations: np.ndarray, n_components: int = 2) -> Tuple[np.ndarray, PCA]:
    pca = PCA(n_components=n_components)
    proj = pca.fit_transform(representations)
    return proj, pca


def population_sparsity(activations: np.ndarray, threshold: float = 0.01) -> float:
    """Fraction of activations below threshold."""
    return float((np.abs(activations) < threshold).mean())


def lifetime_sparsity(activations: np.ndarray) -> float:
    """Kurtosis-based lifetime sparsity: 1 - (mean(a)^2 / mean(a^2))."""
    mean_a = activations.mean(axis=0)
    mean_a2 = (activations ** 2).mean(axis=0)
    eps = 1e-8
    ls = 1.0 - (mean_a ** 2) / (mean_a2 + eps)
    return float(ls.mean())


def concept_selectivity(activations: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Concept selectivity index per unit.

    SI = (max_class_mean - other_mean) / (max_class_mean + other_mean + eps).
    """
    unique_labels = np.unique(labels)
    n_units = activations.shape[1]
    class_means = np.zeros((len(unique_labels), n_units))
    for i, c in enumerate(unique_labels):
        mask = labels == c
        class_means[i] = activations[mask].mean(axis=0)

    max_mean = class_means.max(axis=0)
    sum_all = class_means.sum(axis=0)
    n_classes = len(unique_labels)
    other_mean = (sum_all - max_mean) / max(1, n_classes - 1)
    eps = 1e-8
    return (max_mean - other_mean) / (max_mean + other_mean + eps)
