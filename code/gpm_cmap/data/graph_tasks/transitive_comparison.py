"""Transitive inference comparison dataset.

Training/validation use adjacent relations.  The headline test contains only
symbolic distances >=2, so it cannot be inflated by relations explicitly seen
in training.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from gpm_cmap.utils.seed import make_torch_generator, make_worker_init_fn


class TransitiveComparisonDataset(Dataset):
    def __init__(self, samples: List[Dict[str, torch.Tensor]]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.samples[idx]


def build_transitive_comparison_loaders(
    n_items: int = 7,
    n_train_repeats: int = 80,
    n_test_repeats: int = 10,
    batch_size: int = 64,
    seed: int = 42,
    validation_ratio: float = 0.0,
    num_workers: int = 0,
):
    rng = np.random.default_rng(seed)
    node_names = [chr(ord("A") + i) for i in range(n_items)]
    node_to_id = {name: idx for idx, name in enumerate(node_names)}

    adjacent_samples: List[Dict[str, torch.Tensor]] = []
    for _ in range(n_train_repeats):
        for i in range(n_items - 1):
            higher, lower = i, i + 1
            if rng.random() < 0.5:
                a_id, b_id, label = higher, lower, 1.0
            else:
                a_id, b_id, label = lower, higher, 0.0
            adjacent_samples.append(_make_sample(a_id, b_id, label, sym_dist=1))
    rng.shuffle(adjacent_samples)

    validation_samples: List[Dict[str, torch.Tensor]] = []
    if validation_ratio > 0:
        if not 0.0 < validation_ratio < 1.0:
            raise ValueError(f"validation_ratio must lie in (0,1); got {validation_ratio}")
        groups: Dict[Tuple[int, int], List[Dict[str, torch.Tensor]]] = {}
        for sample in adjacent_samples:
            a_id, b_id = map(int, sample["observations"].tolist())
            groups.setdefault(tuple(sorted((a_id, b_id))), []).append(sample)
        edge_keys = list(groups)
        if len(edge_keys) < 2:
            raise ValueError("At least two adjacent relation groups are required for validation")
        rng.shuffle(edge_keys)
        n_val_groups = max(
            1,
            min(len(edge_keys) - 1, int(round(len(edge_keys) * validation_ratio))),
        )
        validation_edges = set(edge_keys[:n_val_groups])
        validation_samples = [
            sample
            for edge, group in groups.items()
            if edge in validation_edges
            for sample in group
        ]
        train_samples = [
            sample
            for edge, group in groups.items()
            if edge not in validation_edges
            for sample in group
        ]
    else:
        train_samples = adjacent_samples

    # Headline generalization set: non-adjacent pairs only.
    test_samples: List[Dict[str, torch.Tensor]] = []
    for _ in range(n_test_repeats):
        for i in range(n_items):
            for j in range(i + 2, n_items):
                sym_dist = j - i
                if rng.random() < 0.5:
                    a_id, b_id, label = i, j, 1.0
                else:
                    a_id, b_id, label = j, i, 0.0
                test_samples.append(_make_sample(a_id, b_id, label, sym_dist))
    rng.shuffle(test_samples)

    meta = {
        "node_names": node_names,
        "node_to_id": node_to_id,
        "observation_vocab": n_items,
        "action_vocab": 1,
        "n_items": n_items,
        "headline_test_min_symbolic_distance": 2,
        "split_relation_overlap": {
            "train_validation": len(
                {_relation_id(sample) for sample in train_samples}
                & {_relation_id(sample) for sample in validation_samples}
            ) if validation_samples else 0,
        },
    }
    common = {
        "batch_size": batch_size,
        "num_workers": int(num_workers),
        "worker_init_fn": make_worker_init_fn(seed),
    }
    train_loader = DataLoader(
        TransitiveComparisonDataset(train_samples),
        shuffle=True,
        generator=make_torch_generator(seed),
        **common,
    )
    test_loader = DataLoader(TransitiveComparisonDataset(test_samples), shuffle=False, **common)
    if validation_samples:
        validation_loader = DataLoader(
            TransitiveComparisonDataset(validation_samples), shuffle=False, **common
        )
        return train_loader, validation_loader, test_loader, meta
    return train_loader, test_loader, meta


def _make_sample(a_id: int, b_id: int, label: float, sym_dist: int) -> Dict[str, torch.Tensor]:
    return {
        "observations": torch.tensor([a_id, b_id], dtype=torch.long),
        "actions": torch.zeros(2, dtype=torch.long),
        "label": torch.tensor(label, dtype=torch.float),
        "distance": torch.tensor(float(sym_dist), dtype=torch.float),
        "compare_dim": torch.tensor(0, dtype=torch.long),
    }


def _relation_id(sample: Dict[str, torch.Tensor]) -> Tuple[int, int]:
    a_id, b_id = map(int, sample["observations"].tolist())
    return tuple(sorted((a_id, b_id)))
