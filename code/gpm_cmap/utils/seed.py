from __future__ import annotations

import os
import random
from typing import Callable

import numpy as np


def set_seed(seed: int = 42, *, deterministic_algorithms: bool = False) -> None:
    """Seed Python, NumPy and PyTorch.

    ``deterministic_algorithms`` is opt-in because some CUDA operations do not
    have deterministic implementations.  When requested, PyTorch fails loudly
    instead of silently selecting a nondeterministic kernel.
    """
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if deterministic_algorithms:
        torch.use_deterministic_algorithms(True)


def make_torch_generator(seed: int):
    """Return an explicitly seeded generator for DataLoader/sampling APIs."""
    import torch

    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


def make_worker_init_fn(seed: int) -> Callable[[int], None]:
    """Build a deterministic DataLoader worker initializer.

    Each worker receives a distinct seed derived from the run seed and worker
    index.  The function seeds Python and NumPy; PyTorch already seeds each
    worker from the loader generator.
    """
    base_seed = int(seed)

    def _seed_worker(worker_id: int) -> None:
        worker_seed = (base_seed + int(worker_id)) % (2**32)
        random.seed(worker_seed)
        np.random.seed(worker_seed)

    return _seed_worker
