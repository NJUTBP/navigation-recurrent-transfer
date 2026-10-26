from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


@dataclass
class AverageMeter:
    total: float = 0.0
    count: int = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.total / max(1, self.count)


# ---------------------------------------------------------------------------
# Sample efficiency
# ---------------------------------------------------------------------------

def compute_sample_efficiency(
    history: List[Dict[str, float]],
    metric_key: str = "test_acc",
    threshold: float = 0.8,
) -> int:
    """Compute the number of epochs to reach a given performance threshold.

    Parameters
    ----------
    history : list of dicts
        Each dict has at least 'epoch' and metric_key.
    metric_key : str
        Which metric to track.
    threshold : float
        Performance threshold.

    Returns
    -------
    int
        Number of epochs to reach threshold, or -1 if never reached.
    """
    for entry in history:
        if entry.get(metric_key, 0.0) >= threshold:
            return entry.get("epoch", -1)
    return -1


# ---------------------------------------------------------------------------
# Concept selectivity
# ---------------------------------------------------------------------------

def concept_selectivity_index(
    activations: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """Compute concept selectivity index for each unit.

    For each unit, the selectivity index is:
        SI = (max_class_mean - other_class_mean) / (max_class_mean + other_class_mean + eps)

    Parameters
    ----------
    activations : (n_samples, n_units)
    labels : (n_samples,) — integer class labels

    Returns
    -------
    selectivity : (n_units,) — values in [0, 1]; 1 = perfectly selective
    """
    unique_labels = np.unique(labels)
    n_units = activations.shape[1]
    class_means = np.zeros((len(unique_labels), n_units))
    for i, c in enumerate(unique_labels):
        mask = labels == c
        class_means[i] = activations[mask].mean(axis=0)

    max_mean = class_means.max(axis=0)
    # Mean of all other classes
    sum_all = class_means.sum(axis=0)
    n_classes = len(unique_labels)
    other_mean = (sum_all - max_mean) / max(1, n_classes - 1)
    eps = 1e-8
    selectivity = (max_mean - other_mean) / (max_mean + other_mean + eps)
    return selectivity


# ---------------------------------------------------------------------------
# Retrieval accuracy (for Hebb memory testing)
# ---------------------------------------------------------------------------

def retrieval_accuracy(
    retrieved: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """Compute retrieval accuracy — cosine similarity between retrieved and target.

    Parameters
    ----------
    retrieved : (n_samples, dim)
    targets : (n_samples, dim)
    threshold : float
        Cosine similarity above this is counted as correct.

    Returns
    -------
    float — fraction of correct retrievals.
    """
    # Normalize
    r_norm = retrieved / (np.linalg.norm(retrieved, axis=-1, keepdims=True) + 1e-8)
    t_norm = targets / (np.linalg.norm(targets, axis=-1, keepdims=True) + 1e-8)
    cosine_sim = (r_norm * t_norm).sum(axis=-1)
    return float((cosine_sim > threshold).mean())


# ---------------------------------------------------------------------------
# Interference rate
# ---------------------------------------------------------------------------

def memory_interference_rate(
    old_retrieved: np.ndarray,
    new_retrieved: np.ndarray,
    old_targets: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """Measure how much new memories interfere with old ones.

    Compare retrieval quality of old items *after* new items are written.

    Parameters
    ----------
    old_retrieved : (n_old, dim) — retrieval of old keys after new writes
    new_retrieved : (n_new, dim) — for reference (not used in this metric)
    old_targets : (n_old, dim) — ground-truth values for old keys
    threshold : float

    Returns
    -------
    float — fraction of old items that are no longer correctly retrieved.
    """
    r_norm = old_retrieved / (np.linalg.norm(old_retrieved, axis=-1, keepdims=True) + 1e-8)
    t_norm = old_targets / (np.linalg.norm(old_targets, axis=-1, keepdims=True) + 1e-8)
    cosine_sim = (r_norm * t_norm).sum(axis=-1)
    interference = float((cosine_sim < threshold).mean())
    return interference


# ---------------------------------------------------------------------------
# Remapping quality
# ---------------------------------------------------------------------------

def remapping_quality(
    representations_context1: np.ndarray,
    representations_context2: np.ndarray,
) -> float:
    """Measure representation separation across contexts (remapping).

    Uses cosine distance between mean representations in two contexts.
    Higher = better remapping.

    Parameters
    ----------
    representations_context1 : (n1, dim)
    representations_context2 : (n2, dim)

    Returns
    -------
    float — cosine distance between context centroids, range [0, 2].
    """
    c1 = representations_context1.mean(axis=0)
    c2 = representations_context2.mean(axis=0)
    cos_sim = np.dot(c1, c2) / (np.linalg.norm(c1) * np.linalg.norm(c2) + 1e-8)
    return float(1.0 - cos_sim)


# ---------------------------------------------------------------------------
# Sparsity metrics
# ---------------------------------------------------------------------------

def population_sparsity(activations: np.ndarray, threshold: float = 0.01) -> float:
    """Fraction of activations below threshold (population sparsity)."""
    return float((np.abs(activations) < threshold).mean())


def lifetime_sparsity(activations: np.ndarray) -> float:
    """Mean kurtosis-based lifetime sparsity across units.

    Each unit's lifetime sparsity = 1 - (mean(a)^2 / mean(a^2)).
    """
    mean_a = activations.mean(axis=0)
    mean_a2 = (activations ** 2).mean(axis=0)
    eps = 1e-8
    ls = 1.0 - (mean_a ** 2) / (mean_a2 + eps)
    return float(ls.mean())
