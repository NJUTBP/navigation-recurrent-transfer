#!/usr/bin/env python3
from __future__ import annotations
import os

import csv
import hashlib
import json
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get('NRT_PROJECT_ROOT', str(Path(__file__).resolve().parents[4])))
MECH_ROOT = PROJECT_ROOT / 'rerun_best_validation' / 'followups' / 'mechanism_v1'
OUTPUT_ROOT = MECH_ROOT / 'E6_retention_intervention'
C1_ROOT = MECH_ROOT / 'E6_autonomous_retention'
E6_ROOT = MECH_ROOT / 'E6_rho_scan'
SCHEMA_VERSION = 'e6_retention_intervention_v1'
STATISTICS_SEED = 982451653
BOOTSTRAP_SAMPLES = 20000
C1_RESULT_SHA256 = '29bfc8b0a7afecda45a5f5a38fa699bdbd59315385563756bc426ee2f95c171b'
EXPECTED_C = 0.8032013898784912

ARMS = {
    'nav_impair': {'base_rho': 0.0, 'scale_role': 'c'},
    'rho01_rescue': {'base_rho': 0.1, 'scale_role': 'inverse_c'},
}

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def load_c1() -> tuple[dict, float]:
    path = C1_ROOT / 'aggregate' / 'RETENTION_RESULT.json'
    if not path.is_file():
        raise RuntimeError(f'missing C1 result: {path}')
    digest = sha256(path)
    if digest != C1_RESULT_SHA256:
        raise RuntimeError(f'C1 result hash mismatch: {digest}')
    result = json.loads(path.read_text(encoding='utf-8'))
    if result.get('status') != 'complete' or int(result.get('n_rows', -1)) != 210:
        raise RuntimeError('C1 result is not complete')
    c = float(result['rho01_retention_ratio_to_rho0'])
    if abs(c - EXPECTED_C) > 1e-12:
        raise RuntimeError(f'C1 intervention ratio changed: {c}')
    if not 0.0 < c < 1.0:
        raise RuntimeError('C1 intervention ratio must be in (0,1)')
    return result, c

def arm_spec(arm: str) -> dict:
    if arm not in ARMS:
        raise ValueError(f'unknown arm: {arm}')
    _, c = load_c1()
    meta = dict(ARMS[arm])
    meta['arm'] = arm
    meta['scale'] = c if meta['scale_role'] == 'c' else 1.0 / c
    return meta

def cell_dir(root: Path, source: int, target: int, arm: str) -> Path:
    return root / f'source_{source}' / f'target_{target}' / f'arm_{arm}'

def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))

def c1_retention(source: int, target: int, rho: float) -> float:
    path = C1_ROOT / 'aggregate' / 'all_retention.csv'
    matches = [r for r in read_csv(path)
               if int(r['source_seed']) == source and int(r['target_seed']) == target
               and abs(float(r['rho']) - rho) < 1e-12]
    if len(matches) != 1:
        raise RuntimeError(f'C1 retention row count {len(matches)} for {source}/{target}/rho={rho}')
    return float(matches[0]['autonomous_retention_mean'])
