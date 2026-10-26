"""Family tree task loaders -- original + path-length-split variant."""
from __future__ import annotations
from torch.utils.data import DataLoader
from gpm_cmap.data.graph_tasks.family_tree import FamilyTreeTask
from .abstract_common import split_dataset, split_dataset_three_way
from gpm_cmap.utils.seed import make_torch_generator, make_worker_init_fn


def build_family_tree_loaders(
    num_walks,
    seq_len,
    batch_size,
    split_ratio=0.8,
    seed=42,
    validation_ratio=0.0,
    num_workers=0,
):
    task = FamilyTreeTask(seed=seed)
    dataset = task.make_dataset(num_walks=num_walks, seq_len=seq_len)
    common = {
        "batch_size": batch_size,
        "num_workers": int(num_workers),
        "worker_init_fn": make_worker_init_fn(seed),
    }
    if validation_ratio > 0:
        train_ds, validation_ds, test_ds = split_dataset_three_way(
            dataset,
            train_ratio=split_ratio,
            validation_ratio=validation_ratio,
            seed=seed,
        )
        return (
            DataLoader(train_ds, shuffle=True, generator=make_torch_generator(seed), **common),
            DataLoader(validation_ds, shuffle=False, **common),
            DataLoader(test_ds, shuffle=False, **common),
            task,
        )
    train_ds, test_ds = split_dataset(dataset, split_ratio=split_ratio, seed=seed)
    return (
        DataLoader(train_ds, shuffle=True, generator=make_torch_generator(seed), **common),
        DataLoader(test_ds, shuffle=False, **common),
        task,
    )


def build_family_tree_split_loaders(
    num_walks_train: int = 500,
    num_walks_test: int = 200,
    max_train_len: int = 2,
    test_len_min: int = 3,
    test_len_max: int = 4,
    batch_size: int = 64,
    seed: int = 42,
    validation_ratio: float = 0.0,
    num_workers: int = 0,
    **kwargs,
):
    """Short-path train / long-path test for generalization experiments."""
    task = FamilyTreeTask(seed=seed)
    train_ds, test_ds = task.make_split_dataset(
        num_walks_train=num_walks_train,
        num_walks_test=num_walks_test,
        max_train_len=max_train_len,
        test_len_min=test_len_min,
        test_len_max=test_len_max,
    )
    common = {
        "batch_size": batch_size,
        "num_workers": int(num_workers),
        "worker_init_fn": make_worker_init_fn(seed),
    }
    if validation_ratio > 0:
        # Preserve the long-path test protocol; validation is carved only from
        # the short-path training pool.
        relative_train = 1.0 - float(validation_ratio)
        train_ds, validation_ds = split_dataset(
            train_ds, split_ratio=relative_train, seed=seed + 1
        )
        return (
            DataLoader(train_ds, shuffle=True, generator=make_torch_generator(seed), **common),
            DataLoader(validation_ds, shuffle=False, **common),
            DataLoader(test_ds, shuffle=False, **common),
            task,
        )
    return (
        DataLoader(train_ds, shuffle=True, generator=make_torch_generator(seed), **common),
        DataLoader(test_ds, shuffle=False, **common),
        task,
    )
