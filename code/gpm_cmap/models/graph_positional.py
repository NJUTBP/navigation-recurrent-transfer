"""Structural coordinate extraction ξ(s) for abstract graph tasks.

This module computes a fixed structural coordinate for each graph node.
The main experiment code can then register those coordinates inside the
model and optionally apply an auxiliary periodic loss.

Default strategies
------------------
* social_hierarchy -> real_coords
* transitive       -> rank
* family_tree*     -> spectral

Negative-control strategies
---------------------------
* shuffled         -> compute the task's default structural coordinates,
                      then shuffle them across nodes with a fixed seed.
* random_gaussian  -> assign each node a fixed random Gaussian vector.
"""
from __future__ import annotations

from typing import Dict, Optional

import networkx as nx
import numpy as np


# -----------------------------------------------------------------------
#  Core embedding functions
# -----------------------------------------------------------------------

def _social_hierarchy_coords(
    graph: nx.DiGraph,
    node_coords: Dict[str, np.ndarray],
    embed_dim: int,
) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for name, coord in node_coords.items():
        xi = np.zeros(embed_dim, dtype=np.float32)
        xi[: len(coord)] = coord
        out[name] = xi
    return out


def _transitive_rank_coords(
    graph: nx.DiGraph,
    node_coords: Dict[str, np.ndarray],
    embed_dim: int,
) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for name, coord in node_coords.items():
        xi = np.zeros(embed_dim, dtype=np.float32)
        xi[0] = coord[0]
        out[name] = xi
    return out


def _spectral_embedding(
    graph: nx.DiGraph,
    embed_dim: int,
) -> Dict[str, np.ndarray]:
    undirected = graph.to_undirected()
    nodes = sorted(undirected.nodes())
    n = len(nodes)
    node_idx = {name: i for i, name in enumerate(nodes)}

    L = nx.normalized_laplacian_matrix(undirected, nodelist=nodes).toarray()
    eigenvalues, eigenvectors = np.linalg.eigh(L)

    k = min(embed_dim, n - 1)
    coords = eigenvectors[:, 1 : 1 + k].astype(np.float32)

    if k < embed_dim:
        pad = np.zeros((n, embed_dim - k), dtype=np.float32)
        coords = np.concatenate([coords, pad], axis=1)

    out: Dict[str, np.ndarray] = {}
    for name in nodes:
        out[name] = coords[node_idx[name]].copy()
    return out


def _successor_feature_coords(
    graph: nx.DiGraph,
    embed_dim: int,
    gamma: float = 0.9,
) -> Dict[str, np.ndarray]:
    nodes = sorted(graph.nodes())
    n = len(nodes)
    node_idx = {name: i for i, name in enumerate(nodes)}

    T = np.zeros((n, n), dtype=np.float64)
    for u in nodes:
        successors = list(graph.successors(u))
        if successors:
            for v in successors:
                T[node_idx[u], node_idx[v]] = 1.0 / len(successors)

    M = np.linalg.inv(np.eye(n) - gamma * T)

    M_centered = M - M.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(M_centered, full_matrices=False)
    k = min(embed_dim, n)
    coords = (U[:, :k] * S[:k]).astype(np.float32)

    if k < embed_dim:
        pad = np.zeros((n, embed_dim - k), dtype=np.float32)
        coords = np.concatenate([coords, pad], axis=1)

    out: Dict[str, np.ndarray] = {}
    for name in nodes:
        out[name] = coords[node_idx[name]].copy()
    return out


def _shuffled_coords(
    base_coords: Dict[str, np.ndarray],
    seed: int,
) -> Dict[str, np.ndarray]:
    names = sorted(base_coords.keys())
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(names))
    shuffled: Dict[str, np.ndarray] = {}
    for i, name in enumerate(names):
        shuffled[name] = base_coords[names[int(perm[i])]].copy()
    return shuffled


def _random_gaussian_coords(
    graph: nx.DiGraph,
    embed_dim: int,
    seed: int,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    out: Dict[str, np.ndarray] = {}
    for name in sorted(graph.nodes()):
        out[name] = rng.standard_normal(embed_dim).astype(np.float32)
    return out


# -----------------------------------------------------------------------
#  Public API
# -----------------------------------------------------------------------

_TASK_STRATEGY: Dict[str, str] = {
    "social_hierarchy": "real_coords",
    "transitive": "rank",
    "family_tree": "spectral",
    "family_tree_split": "spectral",
}


def compute_structural_coords(
    task_name: str,
    graph: nx.DiGraph,
    node_coords: Dict[str, np.ndarray],
    embed_dim: int = 2,
    strategy: Optional[str] = None,
    sr_gamma: float = 0.9,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Compute structural coordinates ξ(s) for every node in *graph*.

    Parameters
    ----------
    strategy : str, optional
        One of ``real_coords``, ``rank``, ``spectral``,
        ``successor_features``, ``shuffled``, ``random_gaussian``.
    seed : int
        Random seed for reproducible control strategies.
    """
    if strategy is None:
        strategy = _TASK_STRATEGY.get(task_name, "spectral")

    if strategy == "real_coords":
        return _social_hierarchy_coords(graph, node_coords, embed_dim)
    if strategy == "rank":
        return _transitive_rank_coords(graph, node_coords, embed_dim)
    if strategy == "spectral":
        return _spectral_embedding(graph, embed_dim)
    if strategy == "successor_features":
        return _successor_feature_coords(graph, embed_dim, gamma=sr_gamma)
    if strategy == "shuffled":
        base_strategy = _TASK_STRATEGY.get(task_name, "spectral")
        base_coords = compute_structural_coords(
            task_name=task_name,
            graph=graph,
            node_coords=node_coords,
            embed_dim=embed_dim,
            strategy=base_strategy,
            sr_gamma=sr_gamma,
            seed=seed,
        )
        return _shuffled_coords(base_coords, seed=seed)
    if strategy == "random_gaussian":
        return _random_gaussian_coords(graph, embed_dim=embed_dim, seed=seed)

    raise ValueError(
        f"Unknown structural coordinate strategy: {strategy!r}. "
        "Choose from: real_coords, rank, spectral, successor_features, "
        "shuffled, random_gaussian."
    )



def coords_dict_to_tensor(
    coords_dict: Dict[str, np.ndarray],
    node_to_id: Dict[str, int],
) -> "np.ndarray":
    """Convert a node-name → ξ mapping to a (num_nodes, embed_dim) array."""
    n = len(node_to_id)
    sample_key = next(iter(coords_dict))
    d = coords_dict[sample_key].shape[0]
    arr = np.zeros((n, d), dtype=np.float32)
    for name, idx in node_to_id.items():
        arr[idx] = coords_dict[name]
    return arr
