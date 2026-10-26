#!/usr/bin/env python3
from __future__ import annotations
import os

import csv
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get('NRT_PROJECT_ROOT', str(Path(__file__).resolve().parents[4])))
MECH_ROOT = PROJECT_ROOT / 'rerun_best_validation' / 'followups' / 'mechanism_v1'
OUTPUT_ROOT = MECH_ROOT / 'E6_navigation_separability'
E6_ROOT = MECH_ROOT / 'E6_rho_scan'
SCHEMA_VERSION = 'e6_navigation_separability_v1'
RHOS = (0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.9)
TRAIN_TRAJECTORIES = 2048
VAL_TRAJECTORIES = 512
BATCH_SIZE = 128
EPOCHS = 100
EVAL_EVERY = 5
STATISTICS_SEED = 982451653
BOOTSTRAP_SAMPLES = 20000
KEY_RHOS = (0.05, 0.1)


def cell_dir(root: Path, source: int, target: int, rho: float) -> Path:
    tag = str(rho).replace('.', 'p')
    return root / f'source_{source}' / f'target_{target}' / f'rho_{tag}'


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))
