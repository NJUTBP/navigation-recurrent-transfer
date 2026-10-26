from __future__ import annotations

from torch.utils.data import DataLoader

from gpm_cmap.data.graph_tasks.social_hierarchy import SocialHierarchyTask
from .abstract_common import split_dataset, split_dataset_three_way
from gpm_cmap.utils.seed import make_torch_generator, make_worker_init_fn


def build_social_hierarchy_loaders(
    num_walks: int,
    seq_len: int,
    batch_size: int,
    split_ratio: float = 0.8,
    seed: int = 42,
    n_levels: int = 4,
    validation_ratio: float = 0.0,
    num_workers: int = 0,
    **kwargs,
):
    """Build leakage-audited train/(validation)/test loaders."""
    task = SocialHierarchyTask(n_levels=n_levels, seed=seed)
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
