# Shared reconstruction and statistics helpers for re-evaluating the formal target models.
# Origin: review-stage analysis package (common/nd2_common.py). Kept because it rebuilds the frozen-core
# target checkpoints of Phase F, E0.9a, E6 and E6i (core omitted on disk, reconstructed from the navigation
# checkpoint and the recorded seeds) and was checked against the stored test accuracies (Gate R).
#!/usr/bin/env python3
"""Shared infrastructure for the ND2 pre-emptive review package (B1, B3, B4, B5, B6).

Design rules (inherited from E0.9a / E6 of tools/mechanism_v1):
  * nothing in the frozen Phase F, mechanism_v1 or code/ trees is modified; their
    modules are imported read-only and their identity is asserted at import;
  * every new output goes under OUTPUT_ROOT (a new sidecar directory);
  * the Phase F protocol is derived at run time from the canonical PROTOCOL.json;
    only explicitly listed keys change and ``training_examples`` (a seed
    component of run_seed / loader_seed) never changes;
  * frozen-core target checkpoints omit the recurrent core; cores are rebuilt
    from origin + seed and every rebuilt model must pass Gate R (re-evaluated
    test accuracy equals the stored metrics.csv value within GATE_R_TOLERANCE);
  * the inferential unit is the navigation source (B1 draw k is paired with
    source 6100+k); target splits are averaged within source before inference.
Only numpy is imported at module level so that planning, status and collectors
run without torch.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import importlib
import itertools
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

PACKAGE_NAME = 'reevaluation'
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
PROJECT_ROOT = Path(os.environ.get('NRT_PROJECT_ROOT', os.environ.get('ND2_PROJECT_ROOT', DEFAULT_PROJECT_ROOT)))
OUTPUT_ROOT = Path(os.environ.get(
    'ND2_OUTPUT_ROOT', str(PROJECT_ROOT / 'outputs' / PACKAGE_NAME)))
LOG_ROOT = Path(os.environ.get('ND2_LOG_ROOT', str(PROJECT_ROOT / 'outputs' / 'logs')))

PHASEF_PACKAGE = PROJECT_ROOT / 'tools' / 'hard8x8_phaseF_fullscale_v1'
PHASEF_SCRIPTS = PHASEF_PACKAGE / 'scripts'
PHASEF_PROTOCOL = PHASEF_PACKAGE / 'PROTOCOL.json'
PHASEF_OUTPUT = (PROJECT_ROOT / 'rerun_best_validation' / 'hard8x8_diagnostics'
                 / 'audit_v1' / 'phaseF_fullscale_v1')
MECH_TOOLS = PROJECT_ROOT / 'tools' / 'mechanism_v1'
MECH_OUTPUT = PROJECT_ROOT / 'rerun_best_validation' / 'followups' / 'mechanism_v1'
E09A_OUTPUT = MECH_OUTPUT / 'E0_9a_extended_budget'
E6_OUTPUT = MECH_OUTPUT / 'E6_rho_scan'
E6I_OUTPUT = MECH_OUTPUT / 'E6_retention_intervention'
C1_OUTPUT = MECH_OUTPUT / 'E6_autonomous_retention'
E01_OUTPUT = MECH_OUTPUT / 'E0_1_nonrecurrent_baselines'
RERUN_ROOT = PROJECT_ROOT / 'rerun_best_validation'

SOURCES = tuple(range(6101, 6111))
TARGETS = (84001, 84002, 84003, 84004, 84005)
TARGETS_32K = (84001, 84002, 84003)
TEST_SPLITS = ('id_seen_short', 'heldout_short', 'seen_long', 'heldout_long')
QUADRANT = {'id_seen_short': 'SS', 'heldout_short': 'HS', 'seen_long': 'SL', 'heldout_long': 'HL'}
RHO_SCAN = (0.0, 0.05, 0.1, 0.2, 0.35, 0.5)
RESCALE_C = 0.8032013898784912          # E6_retention_intervention EXPECTED_C (C1 ratio)
RESCALE_ARMS = {'nav_impair': (0.0, RESCALE_C), 'rho01_rescue': (0.1, 1.0 / RESCALE_C)}
N_NODES = 64
N_LEVELS = 8
HIDDEN_DIM = 4096
CHANCE = 1.0 / N_NODES
VELOCITY_SCALE = 0.02
GATE_R_TOLERANCE = 2e-3                  # same as E0_67 forward diagnostics
STATISTICS_SEED = 982451653              # config_frozen.yaml statistics_seed
BOOTSTRAP_SAMPLES = 20000                # extended-task convention (SI S6.2)
SCALAR_SPEC = {'origin': 'matched_random', 'kind': 'scalar', 'core': 'frozen'}
E09A_GROUP = {
    'navigation_trainable_scalar_gain': 'scalar_pair',
    'matched_random_trainable_scalar_gain': 'scalar_pair',
    'navigation_identity_frozen': 'identity_pair',
    'matched_random_identity_frozen': 'identity_pair',
}
MECHANISM_MODULES = {
    'extended_common': 'E0_9a_extended_budget_v1',
    'delta_common': 'E5b_delta_prediction_v1',
    'source_measures': 'E234_source_measures_v1',
    'rho_core': 'E6_rho_scan_v1',
    'baseline_common': 'E0_1_nonrecurrent_baselines_v1',
}


# ---------------------------------------------------------------- imports
def ensure_sys_path(path: Path | str) -> None:
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)


def load_phasef_modules():
    """Import the frozen Phase F driver read-only (identity-guarded as in E0.9a)."""
    if not PHASEF_SCRIPTS.is_dir():
        raise RuntimeError(f'Phase F scripts not found: {PHASEF_SCRIPTS}')
    ensure_sys_path(PHASEF_SCRIPTS)
    import fullscale_common as fullscale  # noqa: E402
    import run_group as phasef  # noqa: E402
    resolved = Path(phasef.__file__).resolve()
    if PHASEF_SCRIPTS.resolve() not in resolved.parents:
        raise RuntimeError(f'wrong run_group import: {resolved}')
    return fullscale, phasef


def load_mechanism_module(name: str):
    path = MECH_TOOLS / MECHANISM_MODULES[name] / 'scripts'
    if not path.is_dir():
        raise RuntimeError(f'mechanism_v1 scripts not found: {path}')
    ensure_sys_path(path)
    module = importlib.import_module(name)
    resolved = Path(module.__file__).resolve()
    if path.resolve() not in resolved.parents:
        raise RuntimeError(f'wrong {name} import: {resolved}')
    return module


def load_package_module(experiment_dir: str, name: str):
    ensure_sys_path(PACKAGE_ROOT / experiment_dir / 'scripts')
    return importlib.import_module(name)


# ---------------------------------------------------------------- seeds / protocol
def stable_seed32(*parts: Any) -> int:
    """Byte-identical copy of fullscale_common.stable_seed32 (asserted in preflights)."""
    payload = json.dumps(parts, separators=(',', ':'), default=str).encode('utf-8')
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], byteorder='little', signed=False)


def derive_protocol(*, updates: int, targets: Sequence[int], schema_version: str, status: str,
                    conditions_by_group: Mapping[str, Sequence[str]] | None = None) -> dict[str, Any]:
    """Phase F protocol with only the update budget, target list and labels changed."""
    base = json.loads(PHASEF_PROTOCOL.read_text(encoding='utf-8'))
    protocol = copy.deepcopy(base)
    protocol['schema_version'] = schema_version
    protocol['status'] = status
    protocol['derived_from'] = {
        'phasef_protocol': str(PHASEF_PROTOCOL),
        'phasef_schema_version': base['schema_version'],
        'phasef_max_gradient_updates': base['train']['max_gradient_updates'],
    }
    protocol['output_root'] = str(OUTPUT_ROOT)
    protocol['train'] = dict(base['train'])
    protocol['train']['max_gradient_updates'] = int(updates)
    protocol['train']['minimum_updates'] = int(updates)
    protocol['target_split_seeds'] = [int(t) for t in targets]
    protocol['expected_manifest_hashes'] = {str(t): base['expected_manifest_hashes'][str(t)] for t in targets}
    if conditions_by_group is not None:
        protocol['conditions_by_group'] = {k: list(v) for k, v in conditions_by_group.items()}
    if protocol['training_examples'] != base['training_examples']:
        raise RuntimeError('training_examples is a seed component and must not change')
    eval_every = int(protocol['train']['eval_every_updates'])
    n_evals = max(1, int(updates) // eval_every)
    if int(protocol['train']['early_stopping_patience_evals']) <= n_evals:
        raise RuntimeError('early stopping could fire before the update budget is exhausted')
    protocol['n_evaluations'] = n_evals
    return protocol


def phasef_seeds(fullscale, protocol: Mapping[str, Any], source: int, target: int) -> dict[str, int]:
    te = protocol['training_examples']
    return {
        'run_seed': int(fullscale.stable_seed32('phaseF_run', int(target), te)),
        'loader_seed': int(fullscale.stable_seed32('phaseF_loader', int(target), te)),
        'matched_seed': int(fullscale.stable_seed32('matched_random', int(target))),
        'random_core_seed': int(fullscale.stable_seed32('phaseF_fresh_random_core', int(source), int(target))),
        'rho_core_seed': int(fullscale.stable_seed32('e6_rho_core', int(target))),
    }


def phasef_condition_group() -> dict[str, str]:
    protocol = json.loads(PHASEF_PROTOCOL.read_text(encoding='utf-8'))
    return {c: g for g, conds in protocol['conditions_by_group'].items() for c in conds}


def build_split(fullscale, hist, protocol: Mapping[str, Any], target: int):
    """Rebuild one locked Phase F split and verify manifest hash + historical fingerprints."""
    pools_nodes, manifest = fullscale.build_fullscale_split(protocol, int(target))
    expected = protocol['expected_manifest_hashes'][str(target)]
    if manifest['manifest_hash'] != expected:
        raise RuntimeError(f'manifest hash mismatch for {target}: {manifest["manifest_hash"]} != {expected}')
    pools = {k: fullscale.to_historical_episodes(v, hist) for k, v in pools_nodes.items()}
    for name, episodes in pools.items():
        if [hist['episode_fingerprint'](e) for e in episodes] != manifest['fingerprints'][name]:
            raise RuntimeError(f'historical fingerprint mismatch: {name}')
    return pools, pools_nodes, manifest


# ---------------------------------------------------------------- paths of existing artefacts
def nav_checkpoint(source: int) -> Path:
    return PROJECT_ROOT / 'sources' / f'nav_seed_{int(source)}' / 'artifacts' / 'navigation' / 'best.pt'


def dr_checkpoint(source: int) -> Path:
    return PROJECT_ROOT / 'sources' / f'nav_seed_{int(source)}' / 'artifacts' / 'sequence_control' / 'best_validation.pt'


def phasef_cell_dir(source: int, target: int, condition: str) -> Path:
    group = phasef_condition_group()[condition]
    return PHASEF_OUTPUT / f'source_{int(source)}' / f'target_{int(target)}' / f'group_{group}'


def phasef_checkpoint(source: int, target: int, condition: str) -> Path:
    return phasef_cell_dir(source, target, condition) / 'checkpoints' / condition / 'best_validation.pt'


def e09a_cell_dir(source: int, target: int, condition: str) -> Path:
    return E09A_OUTPUT / f'source_{int(source)}' / f'target_{int(target)}' / f'group_{E09A_GROUP[condition]}'


def e09a_checkpoint(source: int, target: int, condition: str) -> Path:
    return e09a_cell_dir(source, target, condition) / 'checkpoints' / condition / 'best_validation.pt'


def e6_cell_dir(source: int, target: int, rho: float) -> Path:
    return E6_OUTPUT / f'source_{int(source)}' / f'target_{int(target)}' / f'rho_{int(round(float(rho) * 1000)):04d}'


def e6_checkpoint(source: int, target: int, rho: float) -> Path:
    return e6_cell_dir(source, target, rho) / 'checkpoints' / 'rho_core_trainable_scalar_gain' / 'best_validation.pt'


def e6i_cell_dir(source: int, target: int, arm: str) -> Path:
    return E6I_OUTPUT / f'source_{int(source)}' / f'target_{int(target)}' / f'arm_{arm}'


def e6i_checkpoint(source: int, target: int, arm: str) -> Path:
    return (e6i_cell_dir(source, target, arm) / 'checkpoints'
            / 'retention_intervention_trainable_scalar_gain' / 'best_validation.pt')


def e01_cell_dir(target: int, budget: int) -> Path:
    return E01_OUTPUT / f'target_{int(target)}' / f'budget_{int(budget):05d}'


def e01_checkpoint(target: int, budget: int, name: str = 'linear_accumulator_2d') -> Path:
    return e01_cell_dir(target, budget) / 'checkpoints' / name / 'best_validation.pt'


# ---------------------------------------------------------------- io
def atomic_json(path: Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + '\n', encoding='utf-8')
    tmp.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (json.dumps(v, sort_keys=True) if isinstance(v, (dict, list, tuple)) else v)
                             for k, v in row.items()})
    tmp.replace(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def to_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float('nan')
    return out


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(tensor) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def cell_status(cell_dir: Path) -> str:
    cell_dir = Path(cell_dir)
    if (cell_dir / 'COMPLETE.json').is_file():
        info = read_json(cell_dir / 'COMPLETE.json')
        return str(info.get('status', 'complete'))
    if cell_dir.exists() and any(cell_dir.iterdir()):
        return 'partial'
    return 'missing'


# ---------------------------------------------------------------- statistics (source level)
def _sign_matrix(n: int) -> np.ndarray:
    return np.asarray(list(itertools.product((-1.0, 1.0), repeat=n)), dtype=float)


def exact_signflip_p(values: Iterable[float], alternative: str = 'two-sided') -> float:
    """Exhaustive sign-flip test of mean(values) = 0 (SI S6.2); one-sided variants for TOST."""
    x = np.asarray(list(values), dtype=float)
    if x.size == 0 or not np.all(np.isfinite(x)):
        return float('nan')
    if x.size > 20:
        raise ValueError('exact sign-flip enumeration is capped at 20 units')
    means = _sign_matrix(x.size) @ x / x.size
    observed = float(x.mean())
    tol = 1e-12 * max(1.0, abs(observed))
    if alternative == 'two-sided':
        return float(np.mean(np.abs(means) >= abs(observed) - tol))
    if alternative == 'greater':
        return float(np.mean(means >= observed - tol))
    if alternative == 'less':
        return float(np.mean(means <= observed + tol))
    raise ValueError(alternative)


def exact_two_sample_p(a: Iterable[float], b: Iterable[float]) -> float:
    """Exact two-sided permutation test of a difference in means (unpaired robustness check)."""
    a = np.asarray(list(a), dtype=float)
    b = np.asarray(list(b), dtype=float)
    pooled = np.concatenate([a, b])
    n, na = pooled.size, a.size
    if na == 0 or b.size == 0 or n > 24:
        return float('nan')
    observed = abs(a.mean() - b.mean())
    total = pooled.sum()
    combos = np.asarray(list(itertools.combinations(range(n), na)), dtype=np.int64)
    sums = pooled[combos].sum(axis=1)
    diffs = np.abs(sums / na - (total - sums) / (n - na))
    return float(np.mean(diffs >= observed - 1e-12 * max(1.0, observed)))


def bootstrap_ci(values: Iterable[float], *, n: int = BOOTSTRAP_SAMPLES, seed: int = STATISTICS_SEED,
                 level: float = 0.95) -> tuple[float, float]:
    x = np.asarray(list(values), dtype=float)
    if x.size < 2:
        return float('nan'), float('nan')
    rng = np.random.default_rng(int(seed))
    boot = x[rng.integers(0, x.size, size=(int(n), x.size))].mean(axis=1)
    alpha = (1.0 - level) / 2.0
    return float(np.quantile(boot, alpha)), float(np.quantile(boot, 1.0 - alpha))


def t_ci(values: Iterable[float], level: float = 0.95) -> tuple[float, float]:
    from scipy import stats
    x = np.asarray(list(values), dtype=float)
    if x.size < 2:
        return float('nan'), float('nan')
    se = float(x.std(ddof=1) / math.sqrt(x.size))
    half = float(stats.t.ppf(0.5 + level / 2.0, x.size - 1)) * se
    return float(x.mean() - half), float(x.mean() + half)


def holm(pvalues: Sequence[float]) -> list[float]:
    p = np.asarray(pvalues, dtype=float)
    out = np.full(p.size, np.nan)
    finite = np.flatnonzero(np.isfinite(p))
    order = finite[np.argsort(p[finite])]
    m = len(finite)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[index]))
        out[index] = running
    return out.tolist()


def tost(values: Iterable[float], margin: float, *, seed: int = STATISTICS_SEED) -> dict[str, Any]:
    """Equivalence within +-margin: two one-sided exact sign-flip tests AND both 90% CIs inside."""
    x = np.asarray(list(values), dtype=float)
    p_lower = exact_signflip_p(x + margin, 'greater')     # H1: mean > -margin
    p_upper = exact_signflip_p(x - margin, 'less')        # H1: mean < +margin
    t_lo, t_hi = t_ci(x, 0.90)
    b_lo, b_hi = bootstrap_ci(x, seed=seed, level=0.90)
    p_tost = max(p_lower, p_upper)
    equivalent = bool(np.isfinite(p_tost) and p_tost < 0.05 and -margin < t_lo and t_hi < margin
                      and -margin < b_lo and b_hi < margin)
    return {'margin': float(margin), 'p_tost': float(p_tost), 'p_lower': float(p_lower), 'p_upper': float(p_upper),
            'ci90_t_low': t_lo, 'ci90_t_high': t_hi, 'ci90_boot_low': b_lo, 'ci90_boot_high': b_hi,
            'equivalent': equivalent}


def paired_contrast(left: Mapping[int, float], right: Mapping[int, float], *, label: str,
                    expected_units: Sequence[int] = SOURCES, seed: int = STATISTICS_SEED) -> dict[str, Any]:
    """Source-level paired contrast; confirmatory only when every expected unit is present."""
    units = [u for u in expected_units if u in left and u in right
             and np.isfinite(left[u]) and np.isfinite(right[u])]
    d = np.asarray([left[u] - right[u] for u in units], dtype=float)
    out: dict[str, Any] = {'contrast': label, 'n_units': int(d.size), 'n_expected': len(expected_units),
                           'complete': d.size == len(expected_units),
                           'mean_difference': float(d.mean()) if d.size else float('nan'),
                           'sd_difference': float(d.std(ddof=1)) if d.size > 1 else float('nan'),
                           'n_positive': int((d > 0).sum()),
                           'per_unit': {str(u): float(v) for u, v in zip(units, d)}}
    if d.size >= 2:
        lo, hi = bootstrap_ci(d, seed=seed)
        tlo, thi = t_ci(d)
        out.update(p_signflip=exact_signflip_p(d), ci95_boot_low=lo, ci95_boot_high=hi, ci95_t_low=tlo, ci95_t_high=thi,
                   p_two_sample_unpaired=exact_two_sample_p([left[u] for u in units], [right[u] for u in units]))
    else:
        out.update(p_signflip=float('nan'), ci95_boot_low=float('nan'), ci95_boot_high=float('nan'),
                   ci95_t_low=float('nan'), ci95_t_high=float('nan'), p_two_sample_unpaired=float('nan'))
    out['inference_status'] = 'confirmatory_complete' if out['complete'] else 'incomplete_descriptive_only'
    return out


def summarize(values: Iterable[float], *, seed: int = STATISTICS_SEED) -> dict[str, float]:
    x = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if x.size == 0:
        return {'n': 0, 'mean': float('nan'), 'sem': float('nan'), 'ci95_t_low': float('nan'),
                'ci95_t_high': float('nan'), 'ci95_boot_low': float('nan'), 'ci95_boot_high': float('nan')}
    tlo, thi = t_ci(x)
    blo, bhi = bootstrap_ci(x, seed=seed)
    return {'n': int(x.size), 'mean': float(x.mean()),
            'sem': float(x.std(ddof=1) / math.sqrt(x.size)) if x.size > 1 else float('nan'),
            'ci95_t_low': tlo, 'ci95_t_high': thi, 'ci95_boot_low': blo, 'ci95_boot_high': bhi}


def isotonic_fit(x: Sequence[float], y: Sequence[float], *, increasing: bool = True):
    """Pool-adjacent-violators fit; returns a predictor with linear interpolation and flat extrapolation."""
    xs = np.asarray(x, dtype=float)
    ys = np.asarray(y, dtype=float)
    keep = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[keep], ys[keep]
    if xs.size == 0:
        return lambda q: np.full(np.shape(q), np.nan)
    order = np.argsort(xs, kind='mergesort')
    xs, ys = xs[order], ys[order]
    sign = 1.0 if increasing else -1.0
    vals = list(sign * ys)
    weights = [1.0] * len(vals)
    blocks = [[i] for i in range(len(vals))]
    i = 0
    while i < len(vals) - 1:
        if vals[i] > vals[i + 1] + 1e-15:
            merged = (vals[i] * weights[i] + vals[i + 1] * weights[i + 1]) / (weights[i] + weights[i + 1])
            vals[i] = merged
            weights[i] += weights[i + 1]
            blocks[i] += blocks[i + 1]
            del vals[i + 1], weights[i + 1], blocks[i + 1]
            i = max(i - 1, 0)
        else:
            i += 1
    fitted = np.empty(xs.size)
    for value, block in zip(vals, blocks):
        fitted[block] = sign * value
    ux, inverse = np.unique(xs, return_inverse=True)
    uy = np.asarray([fitted[inverse == k].mean() for k in range(ux.size)])

    def predict(query):
        return np.interp(np.asarray(query, dtype=float), ux, uy)
    return predict


# ---------------------------------------------------------------- torch-dependent helpers
def fresh_random_core(hist, fullscale, model_cfg: Mapping[str, Any], source: int, target: int):
    """The Phase F fresh-random donor core (U(+-1/sqrt(N)) init at random_core_seed).

    Initialisation happens on CPU inside AbstractPathIntModel.__init__ before
    ``.to(device)``, so building the donor on CPU reproduces Phase F exactly.
    Changes the global RNG: callers must reseed afterwards.
    """
    import torch
    seed = int(fullscale.stable_seed32('phaseF_fresh_random_core', int(source), int(target)))
    hist['set_seed'](seed)
    donor = hist['build_hard_model'](model_cfg, n_nodes=N_NODES, device=torch.device('cpu'))
    hh = donor.grid_rnn.cell.h_proj.weight.detach().cpu().clone()
    ih = donor.grid_rnn.cell.in_proj.weight.detach().cpu().clone()
    del donor
    return hh, ih, {'random_core_seed': seed, 'construction': 'phaseF_fresh_random_donor_uniform_1_over_sqrtN'}


def rebuild_frozen_target(checkpoint_path: Path, *, hist, phasef, device, core_hh, core_ih,
                          model_cfg: Mapping[str, Any] | None = None):
    """Rebuild a frozen-core target model: install the given core, then load the compact state.

    The core always enters through the matched slot of configure_origin (as E6
    does), which installs the tensors with frozen trainability; the stored
    interface/readout is then loaded with strict=False and only grid_rnn keys
    may be missing.
    """
    import torch
    payload = torch.load(Path(checkpoint_path), map_location='cpu', weights_only=False)
    storage = payload.get('checkpoint_storage_mode')
    if storage != 'frozen_core_omitted_reconstruct_from_origin_and_seed':
        raise RuntimeError(f'unexpected storage mode {storage!r} in {checkpoint_path}')
    cfg = model_cfg if model_cfg is not None else payload.get('model_config')
    if cfg is None:
        raise RuntimeError(f'no model config for {checkpoint_path}')
    stored = dict(payload.get('condition_spec') or {})
    spec = {'origin': 'matched_random', 'kind': stored.get('kind', 'scalar'), 'core': 'frozen'}
    if 'gain' in stored:
        spec['gain'] = stored['gain']
    if spec['kind'] not in ('scalar', 'fixed_gain', 'adapter2x2'):
        raise RuntimeError(f'unsupported interface kind {spec["kind"]!r}')
    hist['set_seed'](int(payload.get('run_seed', 0)))
    base = hist['build_hard_model'](cfg, n_nodes=N_NODES, device=device)
    phasef.configure_origin(base, spec, hist, nav_hh=core_hh, nav_ih=core_ih, matched_hh=core_hh, matched_ih=core_ih,
                            cell_cls=hist['FactorialCell'], model_cfg=cfg, device=device, random_core_seed=0)
    model = phasef.wrap_model(base, spec, hist, device)
    result = model.load_state_dict(payload['model_state'], strict=False)
    if list(result.unexpected_keys):
        raise RuntimeError(f'unexpected keys in {checkpoint_path}: {list(result.unexpected_keys)[:6]}')
    not_core = [k for k in result.missing_keys if '.grid_rnn.' not in k and not k.startswith('grid_rnn.')]
    if not_core:
        raise RuntimeError(f'missing non-core keys in {checkpoint_path}: {not_core[:6]}')
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, payload, spec


def evaluate_episodes(model, episodes, *, hist, fullscale, device, loader_seed: int, split_name: str,
                      batch_size: int = 128):
    loader = fullscale.make_loader(list(episodes), hist, batch_size, False, int(loader_seed) + 2)
    return hist['evaluate_loader_detailed'](model, loader, device, split_name=split_name, calibration_bins=15)


def gate_r(summaries: Mapping[str, Mapping[str, Any]], reference_rows: Sequence[Mapping[str, Any]],
           *, tolerance: float = GATE_R_TOLERANCE, splits: Sequence[str] = TEST_SPLITS):
    reference = {r['split']: to_float(r['accuracy']) for r in reference_rows if r.get('split') in splits}
    rows = []
    for split in splits:
        observed = float(summaries[split]['accuracy'])
        expected = reference.get(split, float('nan'))
        deviation = abs(observed - expected) if np.isfinite(expected) else float('nan')
        rows.append({'split': split, 'observed': observed, 'reference': expected, 'abs_deviation': deviation,
                     'pass': bool(np.isfinite(deviation) and deviation <= tolerance)})
    return rows, all(r['pass'] for r in rows)


def one_batch(episodes, *, hist, fullscale, seed: int = 0):
    """Collate a list of equal- or mixed-length episodes into one padded batch (Phase F collate)."""
    loader = fullscale.make_loader(list(episodes), hist, max(1, len(episodes)), False, int(seed))
    return next(iter(loader))


class ModelResolver:
    """Map a model specification to (core, checkpoint, Gate R reference, state alignment).

    spec kinds: phasef (8k, five splits), e09a (32k), e6 (rho core, 32k), e6i
    (rescaled core, 32k).  ``permutation`` is the P of the
    signed permutation Q = D P for nav-derived cores (None otherwise): P-aligned
    navigation states are h_nav[..., permutation], the C1 convention.
    """

    def __init__(self, *, hist, fullscale, phasef, model_cfg: Mapping[str, Any]):
        self.hist, self.fullscale, self.phasef, self.model_cfg = hist, fullscale, phasef, model_cfg
        self._nav: dict[int, Any] = {}
        self._matched: dict[tuple[int, int], Any] = {}

    def nav(self, source: int):
        source = int(source)
        if source not in self._nav:
            path = nav_checkpoint(source)
            if not path.is_file():
                raise FileNotFoundError(path)
            self._nav[source] = self.hist['load_navigation_grid_weights'](path)
        return self._nav[source]

    def matched(self, source: int, target: int):
        key = (int(source), int(target))
        if key not in self._matched:
            nav_hh, nav_ih, _ = self.nav(source)
            seed = int(self.fullscale.stable_seed32('matched_random', int(target)))
            hh, ih, diag = self.hist['make_matched_random_weights'](
                nav_hh, nav_ih, seed=seed, iterations=8, singular_rank=16, probe_count=64, strict=True)
            sm = load_mechanism_module('source_measures')
            perm, flipped, fraction = sm.matched_transform(nav_hh, nav_ih, hh, ih, diag)
            self._matched[key] = (hh, ih, {'matched_seed': seed, 'sign_flip_fraction': float(fraction)},
                                  np.asarray(perm), np.asarray(flipped))
        return self._matched[key]

    def rho(self, source: int, target: int, rho: float):
        rho_core = load_mechanism_module('rho_core')
        nav_hh, nav_ih, _ = self.nav(source)
        seed = int(self.fullscale.stable_seed32('e6_rho_core', int(target)))
        hh, ih, meta = rho_core.signed_similarity_core(nav_hh, nav_ih, seed=seed, rho=float(rho))
        perm, flipped = rho_core.flip_units(int(nav_hh.shape[0]), float(rho), seed)
        return hh, ih, {**meta, 'core_seed': seed}, np.asarray(perm), np.asarray(flipped)

    def resolve(self, spec: Mapping[str, Any]) -> dict[str, Any]:
        kind = spec['kind']
        source, target = int(spec['source']), int(spec['target'])
        out: dict[str, Any] = {'kind': kind, 'source': source, 'target': target, 'permutation': None}
        if kind in ('phasef', 'e09a'):
            condition = spec['condition']
            if condition.startswith('navigation'):
                hh, ih, meta = self.nav(source)
                out.update(family='navigation', permutation=np.arange(int(hh.shape[0])), meta={'nav_sha256': meta['checkpoint_sha256']})
            elif condition.startswith('matched_random'):
                hh, ih, meta, perm, _ = self.matched(source, target)
                out.update(family='matched', permutation=perm, meta=meta)
            elif condition.startswith('fresh_random'):
                hh, ih, meta = fresh_random_core(self.hist, self.fullscale, self.model_cfg, source, target)
                out.update(family='fresh_random', meta=meta)
            else:
                raise ValueError(condition)
            if kind == 'phasef':
                out.update(checkpoint=phasef_checkpoint(source, target, condition), budget=8000,
                           reference_csv=phasef_cell_dir(source, target, condition) / 'metrics.csv')
            else:
                out.update(checkpoint=e09a_checkpoint(source, target, condition), budget=32000,
                           reference_csv=e09a_cell_dir(source, target, condition) / 'metrics.csv')
            out.update(reference_filter={'condition': condition}, label=f'{kind}:{condition}')
        elif kind == 'e6':
            rho = float(spec['rho'])
            hh, ih, meta, perm, _ = self.rho(source, target, rho)
            out.update(family='rho', rho=rho, permutation=perm, meta=meta, budget=32000,
                       checkpoint=e6_checkpoint(source, target, rho),
                       reference_csv=e6_cell_dir(source, target, rho) / 'metrics.csv', reference_filter={},
                       label=f'e6:rho{rho:g}')
        elif kind == 'e6i':
            arm = str(spec['arm'])
            base_rho, scale = RESCALE_ARMS[arm]
            hh0, ih, meta, perm, _ = self.rho(source, target, base_rho)
            hh = (hh0 * float(scale)).contiguous()
            out.update(family='rescaled', arm=arm, rho=base_rho, scale=float(scale), permutation=perm,
                       meta={**meta, 'recurrent_scale': float(scale)}, budget=32000,
                       checkpoint=e6i_checkpoint(source, target, arm),
                       reference_csv=e6i_cell_dir(source, target, arm) / 'metrics.csv', reference_filter={},
                       label=f'e6i:{arm}')
        else:
            raise ValueError(kind)
        out['hh'], out['ih'] = hh, ih
        return out


def reference_rows(core: Mapping[str, Any]) -> list[dict[str, str]]:
    rows = read_csv(core['reference_csv'])
    for key, value in core.get('reference_filter', {}).items():
        rows = [r for r in rows if r.get(key) == value]
    return rows
