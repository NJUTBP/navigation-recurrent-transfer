"""Pairwise comparison data for a two-dimensional social hierarchy.

Training/validation examples vary exactly one coordinate, making the requested
comparison dimension identifiable from the pair itself.  Test examples vary both
coordinates and use the original cross-dimension aggregate rule.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from gpm_cmap.utils.seed import make_torch_generator, make_worker_init_fn


class SocialComparisonDataset(Dataset):
    def __init__(self, samples: List[Dict[str, torch.Tensor]]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.samples[idx]


def _build_node_index(n_levels: int):
    node_to_id: Dict[str, int] = {}
    id_to_coord: Dict[int, np.ndarray] = {}
    idx = 0
    for i in range(n_levels):
        for j in range(n_levels):
            name = f"p_{i}_{j}"
            node_to_id[name] = idx
            id_to_coord[idx] = np.array([float(i), float(j)], dtype=np.float32)
            idx += 1
    return node_to_id, id_to_coord


def _pair_id(sample: Dict[str, torch.Tensor]) -> Tuple[int, int]:
    a, b = sample["observations"].tolist()
    return int(a), int(b)


def _make_single_dim_pairs(id_to_coord, n_levels, rng, n_pairs):
    """Generate pairs that differ on exactly one coordinate.

    The earlier generator only required the selected coordinate to differ; the
    other coordinate often differed too.  The same ordered pair could therefore
    receive opposite labels when sampled with a different ``compare_dim``.
    """
    coord_to_id = {tuple(coord.astype(int)): node_id for node_id, coord in id_to_coord.items()}
    samples = []
    for _ in range(n_pairs):
        dim = int(rng.integers(0, 2))
        fixed_dim = 1 - dim
        fixed_value = int(rng.integers(0, n_levels))
        a_value, b_value = rng.choice(n_levels, size=2, replace=False)
        coord_a = [0, 0]
        coord_b = [0, 0]
        coord_a[fixed_dim] = fixed_value
        coord_b[fixed_dim] = fixed_value
        coord_a[dim] = int(a_value)
        coord_b[dim] = int(b_value)
        a = coord_to_id[tuple(coord_a)]
        b = coord_to_id[tuple(coord_b)]
        ca, cb = id_to_coord[a], id_to_coord[b]
        label = 1.0 if ca[dim] > cb[dim] else 0.0
        dist = float(np.linalg.norm(ca - cb))
        samples.append({
            "observations": torch.tensor([a, b], dtype=torch.long),
            "actions": torch.zeros(2, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.float),
            "distance": torch.tensor(dist, dtype=torch.float),
            "compare_dim": torch.tensor(dim, dtype=torch.long),
            "coord_a": torch.from_numpy(ca.copy()),
            "coord_b": torch.from_numpy(cb.copy()),
        })
    return samples


def _make_cross_dim_pairs(id_to_coord, n_levels, rng, n_pairs):
    ids = list(id_to_coord.keys())
    samples = []
    for _ in range(n_pairs):
        for _attempt in range(200):
            a, b = rng.choice(ids, size=2, replace=False)
            ca, cb = id_to_coord[a], id_to_coord[b]
            if ca[0] != cb[0] and ca[1] != cb[1] and ca.sum() != cb.sum():
                break
        else:
            raise RuntimeError("Could not sample an unambiguous cross-dimension pair")
        label = 1.0 if ca.sum() > cb.sum() else 0.0
        dist = float(np.linalg.norm(ca - cb))
        samples.append({
            "observations": torch.tensor([a, b], dtype=torch.long),
            "actions": torch.zeros(2, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.float),
            "distance": torch.tensor(dist, dtype=torch.float),
            "compare_dim": torch.tensor(-1, dtype=torch.long),
            "coord_a": torch.from_numpy(ca.copy()),
            "coord_b": torch.from_numpy(cb.copy()),
        })
    return samples


def _group_split(samples, validation_ratio: float, seed: int):
    """Split by ordered pair identity so repeated pairs cannot cross."""
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError(f"validation_ratio must lie in (0,1); got {validation_ratio}")
    groups: Dict[Tuple[int, int], List[Dict[str, torch.Tensor]]] = {}
    for sample in samples:
        groups.setdefault(_pair_id(sample), []).append(sample)
    keys = list(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(keys)
    n_val = max(1, min(len(keys) - 1, int(round(len(keys) * validation_ratio))))
    val_keys = set(keys[:n_val])
    train = [sample for key, group in groups.items() if key not in val_keys for sample in group]
    validation = [sample for key, group in groups.items() if key in val_keys for sample in group]
    return train, validation


def build_social_comparison_loaders(
    n_levels: int = 4,
    n_train: int = 400,
    n_test: int = 200,
    batch_size: int = 64,
    seed: int = 42,
    validation_ratio: float = 0.0,
    num_workers: int = 0,
):
    """Build train/(validation)/test loaders.

    With ``validation_ratio=0`` the legacy ``(train, test, meta)`` interface is
    retained.  A positive ratio returns ``(train, validation, test, meta)``.
    """
    rng = np.random.default_rng(seed)
    node_to_id, id_to_coord = _build_node_index(n_levels)
    training_pool = _make_single_dim_pairs(id_to_coord, n_levels, rng, n_train)
    test_samples = _make_cross_dim_pairs(id_to_coord, n_levels, rng, n_test)
    validation_samples: List[Dict[str, torch.Tensor]] = []
    if validation_ratio > 0:
        training_pool, validation_samples = _group_split(training_pool, validation_ratio, seed + 1)

    train_pairs = {_pair_id(sample) for sample in training_pool}
    val_pairs = {_pair_id(sample) for sample in validation_samples}
    test_pairs = {_pair_id(sample) for sample in test_samples}
    if train_pairs & val_pairs or train_pairs & test_pairs or val_pairs & test_pairs:
        raise RuntimeError("Ordered pair leakage detected across social comparison splits")

    meta = {
        "node_to_id": node_to_id,
        "id_to_coord": id_to_coord,
        "observation_vocab": len(node_to_id),
        "action_vocab": 1,
        "n_levels": n_levels,
        "split_pair_overlap": {
            "train_validation": len(train_pairs & val_pairs),
            "train_test": len(train_pairs & test_pairs),
            "validation_test": len(val_pairs & test_pairs),
        },
    }
    common = {
        "batch_size": batch_size,
        "num_workers": int(num_workers),
        "worker_init_fn": make_worker_init_fn(seed),
    }
    train_loader = DataLoader(
        SocialComparisonDataset(training_pool),
        shuffle=True,
        generator=make_torch_generator(seed),
        **common,
    )
    test_loader = DataLoader(SocialComparisonDataset(test_samples), shuffle=False, **common)
    if validation_samples:
        validation_loader = DataLoader(
            SocialComparisonDataset(validation_samples), shuffle=False, **common
        )
        return train_loader, validation_loader, test_loader, meta
    return train_loader, test_loader, meta
