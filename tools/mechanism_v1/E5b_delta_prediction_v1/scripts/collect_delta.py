#!/usr/bin/env python3
"""E5b' collector: does the rectified displacement delta predict the gain ratio?

Pre-registered rule (fixed on 2026-09-19 before any real delta was measured;
do not edit after the first cluster run)
------------------------------------------------------------------------------
Prediction.  For each source, pred = mean over targets of
    log2( delta_nav(g=1) / delta_matched(g=1) )
on H_nav, restricted to the E5c targets (84001, 84002).

Observation (primary).  The E5c post-hoc threshold-crossing shift on
id_seen_short at accuracy 0.90: obs = log2 g_0.90(matched) - log2 g_0.90(nav)
from ``E5c_gain_sweep/aggregate_posthoc/threshold_gain_by_source.csv``.
Left-censored navigation values enter as 0, an upper bound, which makes obs
conservative.  Right-censored sources are excluded and counted.

Statistic.  e = obs - pred per source.  Report the mean, the 95 % bootstrap
interval (seed 982451653, 20,000 resamples) and the two-sided exact sign-flip p.

Decision.
    supported     the 95 % interval of mean e lies inside [-0.5, +0.5]
    falsified     exact p < 0.05 and the interval excludes 0
    inconclusive  otherwise
The margin 0.5 is half the E5c grid step in log2 units; the observed side
cannot resolve smaller differences.

Consistency check.  The same rule with H_init in place of H_nav.
Secondary, descriptive.  The same errors against the pre-registered E5c argmax
ratio and against the learned-gain ratios (Phase F at 8,000 updates on five
targets, E0.9a at 32,000 updates on three), each with its own target set;
Spearman between pred and obs across sources; delta at every grid gain, which
shows whether the ratio depends on the reference gain.

Source-side stationarity (descriptive).  Autonomous retention ||ReLU(W h)||/||h||
on each core's own H_nav states and the norm log-slope along navigation
trajectories and along H_init trajectories, navigation minus matched.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from delta_common import E09A_ROOT, E5C_ROOT, GAIN_GRID, OUTPUT_ROOT, PHASEF_OUTPUT, SCHEMA_VERSION

STATISTICS_SEED = 982451653
BOOTSTRAP_SAMPLES = 20000
MARGIN = 0.5
E5C_TARGETS = (84001, 84002)
PRIMARY_SPLIT = 'id_seen_short'
PRIMARY_THRESHOLD = 0.90
SCALAR = {'navigation_trainable_scalar_gain': 'navigation',
          'matched_random_trainable_scalar_gain': 'matched_random'}

def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open('r', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))

def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
            writer.writerow(row)
    tmp.replace(path)

def exact_sign_flip_p(values) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float('nan')
    observed = abs(float(values.mean()))
    hits = sum(1 for signs in itertools.product((1.0, -1.0), repeat=values.size)
               if abs(float(np.mean(np.asarray(signs) * values))) >= observed - 1e-12)
    return hits / float(2 ** values.size)

def bootstrap_ci(values) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float('nan'), float('nan')
    rng = np.random.default_rng(STATISTICS_SEED)
    index = rng.integers(0, values.size, size=(BOOTSTRAP_SAMPLES, values.size))
    means = values[index].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

def spearman(x, y) -> float:
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if x.size < 3:
        return float('nan')
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    return float(np.corrcoef(rx, ry)[0, 1])

def verdict(mean: float, low: float, high: float, p: float) -> str:
    if not all(math.isfinite(v) for v in (mean, low, high)):
        return 'not_evaluable'
    if -MARGIN <= low and high <= MARGIN:
        return 'supported'
    if p < 0.05 and (low > 0 or high < 0):
        return 'falsified'
    return 'inconclusive'

def evaluate(name: str, pred: dict[int, float], obs: dict[int, float], **extra) -> dict[str, Any]:
    common = sorted(s for s in set(pred) & set(obs) if math.isfinite(pred[s]) and math.isfinite(obs[s]))
    errors = np.array([obs[s] - pred[s] for s in common], dtype=np.float64)
    low, high = bootstrap_ci(errors)
    p = exact_sign_flip_p(errors)
    mean = float(errors.mean()) if errors.size else float('nan')
    return {'test': name, **extra, 'n_sources': len(common),
            'mean_prediction_log2': float(np.mean([pred[s] for s in common])) if common else float('nan'),
            'mean_observation_log2': float(np.mean([obs[s] for s in common])) if common else float('nan'),
            'mean_error_log2': mean, 'ci_low': low, 'ci_high': high, 'exact_p': p,
            'spearman_pred_obs': spearman([pred[s] for s in common], [obs[s] for s in common]),
            'verdict': verdict(mean, low, high, p)}

def load_cells(root: Path):
    delta, retention, profile = [], [], []
    complete = 0
    for cell in sorted(root.glob('source_*/target_*')):
        if not (cell / 'COMPLETE.json').is_file():
            continue
        complete += 1
        delta += read_csv(cell / 'delta.csv')
        retention += read_csv(cell / 'retention.csv')
        profile += read_csv(cell / 'trajectory_profile.csv')
    return complete, delta, retention, profile

def predictions(delta_rows, state_set: str, targets, gain: float = 1.0) -> dict[int, float]:
    table = defaultdict(dict)
    for row in delta_rows:
        if row['state_set'] != state_set or float(row['gain']) != gain:
            continue
        table[(int(row['source_seed']), int(row['target_seed']))][row['origin']] = float(row['delta_mean'])
    per_source = defaultdict(list)
    for (source, target), values in table.items():
        if target in targets and 'navigation' in values and 'matched_random' in values:
            per_source[source].append(math.log2(values['navigation'] / values['matched_random']))
    return {s: float(np.mean(v)) for s, v in per_source.items() if v}

def observed_threshold(root: Path) -> tuple[dict[int, float], dict[str, int]]:
    rows = read_csv(root / 'aggregate_posthoc' / 'threshold_gain_by_source.csv')
    table = defaultdict(dict)
    for row in rows:
        if row['split'] == PRIMARY_SPLIT and abs(float(row['threshold']) - PRIMARY_THRESHOLD) < 1e-9:
            table[int(row['source_seed'])][row['origin']] = (float(row['log2_gain']), row['status'])
    obs, counts = {}, defaultdict(int)
    for source, values in table.items():
        if len(values) < 2:
            continue
        (nav, nav_status), (mat, mat_status) = values['navigation'], values['matched_random']
        counts[f'navigation_{nav_status}'] += 1
        counts[f'matched_{mat_status}'] += 1
        if math.isfinite(nav) and math.isfinite(mat):
            obs[source] = mat - nav
    return obs, dict(counts)

def observed_argmax(root: Path) -> dict[int, float]:
    by_source = defaultdict(dict)
    for row in read_csv(root / 'aggregate' / 'optima_by_source.csv'):
        by_source[int(row['source_seed'])][row['origin']] = float(row['gstar_id_seen_short'])
    return {s: math.log2(v['matched_random'] / v['navigation']) for s, v in by_source.items()
            if 'navigation' in v and 'matched_random' in v and v['navigation'] > 0 and v['matched_random'] > 0}

def observed_learned(root: Path, group: str) -> tuple[dict[int, float], list[int]]:
    per_cell = defaultdict(dict)
    for cell in sorted(root.glob(f'source_*/target_*/{group}')):
        if not (cell / 'COMPLETE.json').is_file():
            continue
        for row in read_csv(cell / 'condition_diagnostics.csv'):
            origin = SCALAR.get(row.get('condition'))
            if origin and row.get('gain') not in (None, ''):
                per_cell[(int(row['source_seed']), int(row['target_seed']))][origin] = float(row['gain'])
    per_source = defaultdict(list)
    targets = set()
    for (source, target), values in per_cell.items():
        if len(values) == 2:
            per_source[source].append(math.log2(values['matched_random'] / values['navigation']))
            targets.add(target)
    return {s: float(np.mean(v)) for s, v in per_source.items()}, sorted(targets)

def paired(name: str, left: dict[int, float], right: dict[int, float], **extra) -> dict[str, Any]:
    common = sorted(set(left) & set(right))
    diffs = np.array([left[s] - right[s] for s in common], dtype=np.float64)
    low, high = bootstrap_ci(diffs)
    return {'contrast': name, **extra, 'n_sources': len(common),
            'navigation_mean': float(np.mean([left[s] for s in common])) if common else float('nan'),
            'matched_mean': float(np.mean([right[s] for s in common])) if common else float('nan'),
            'mean_difference': float(diffs.mean()) if diffs.size else float('nan'),
            'ci_low': low, 'ci_high': high, 'exact_p': exact_sign_flip_p(diffs),
            'n_positive': int(np.sum(diffs > 0))}

def retention_by_source(rows, state_set: str, field: str, gain=None) -> dict[str, dict[int, float]]:
    bucket = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row['state_set'] != state_set or row.get(field) in (None, ''):
            continue
        if gain is not None and (row.get('gain') in (None, '') or float(row['gain']) != float(gain)):
            continue
        bucket[row['origin']][int(row['source_seed'])].append(float(row[field]))
    return {origin: {s: float(np.mean(v)) for s, v in values.items()} for origin, values in bucket.items()}

def main() -> int:
    parser = argparse.ArgumentParser(description="E5b' collector")
    parser.add_argument('--root', type=Path, default=OUTPUT_ROOT)
    parser.add_argument('--e5c-root', type=Path, default=E5C_ROOT)
    parser.add_argument('--phasef-root', type=Path, default=PHASEF_OUTPUT)
    parser.add_argument('--e09a-root', type=Path, default=E09A_ROOT)
    args = parser.parse_args()
    out = args.root / 'aggregate'
    out.mkdir(parents=True, exist_ok=True)

    complete, delta, retention, profile = load_cells(args.root)
    if not delta:
        raise SystemExit("no completed E5b' cells found")
    write_csv(out / 'all_delta.csv', delta)
    write_csv(out / 'all_retention.csv', retention)
    write_csv(out / 'all_trajectory_profile.csv', profile)

    obs_threshold, censoring = observed_threshold(args.e5c_root)
    obs_argmax = observed_argmax(args.e5c_root)
    obs_phasef, phasef_targets = observed_learned(args.phasef_root, 'group_learned_interface')
    obs_e09a, e09a_targets = observed_learned(args.e09a_root, 'group_scalar_pair')

    tests = []
    for state_set in ('H_nav', 'H_init'):
        role = 'primary' if state_set == 'H_nav' else 'consistency_check'
        pred = predictions(delta, state_set, E5C_TARGETS)
        tests.append(evaluate('threshold_shift_0.90', pred, obs_threshold, state_set=state_set, role=role,
                              targets='84001 84002', **{f'censoring_{k}': v for k, v in censoring.items()}))
        tests.append(evaluate('argmax_ratio', pred, obs_argmax, state_set=state_set, role='secondary',
                              targets='84001 84002'))
        if phasef_targets:
            tests.append(evaluate('learned_gain_ratio_8000', predictions(delta, state_set, phasef_targets),
                                  obs_phasef, state_set=state_set, role='secondary',
                                  targets=' '.join(map(str, phasef_targets))))
        if e09a_targets:
            tests.append(evaluate('learned_gain_ratio_32000', predictions(delta, state_set, e09a_targets),
                                  obs_e09a, state_set=state_set, role='secondary',
                                  targets=' '.join(map(str, e09a_targets))))
    write_csv(out / 'prediction_tests.csv', tests)

    by_source = []
    pred_nav = predictions(delta, 'H_nav', E5C_TARGETS)
    pred_init = predictions(delta, 'H_init', E5C_TARGETS)
    for source in sorted(set(pred_nav) | set(pred_init) | set(obs_threshold)):
        by_source.append({'source_seed': source, 'pred_H_nav_log2': pred_nav.get(source, float('nan')),
                          'pred_H_init_log2': pred_init.get(source, float('nan')),
                          'obs_threshold_0.90_log2': obs_threshold.get(source, float('nan')),
                          'obs_argmax_log2': obs_argmax.get(source, float('nan')),
                          'obs_learned_8000_log2': obs_phasef.get(source, float('nan')),
                          'obs_learned_32000_log2': obs_e09a.get(source, float('nan'))})
    write_csv(out / 'prediction_by_source.csv', by_source)

    reference = []
    for state_set in ('H_nav', 'H_init'):
        for gain in GAIN_GRID:
            pred = predictions(delta, state_set, E5C_TARGETS, float(gain))
            if pred:
                reference.append({'state_set': state_set, 'reference_gain': gain,
                                  'mean_pred_log2': float(np.mean(list(pred.values()))),
                                  'min_pred_log2': float(min(pred.values())),
                                  'max_pred_log2': float(max(pred.values()))})
    write_csv(out / 'prediction_vs_reference_gain.csv', reference)

    stationarity = []
    for state_set, field, gain in (('H_nav', 'autonomous_retention_mean', None),
                                   ('H_nav_trajectory', 'norm_log_slope_per_step', None),
                                   ('H_init_start', 'autonomous_retention_mean', None),
                                   ('H_init_seen_long', 'norm_log_slope_per_step', 1.0),
                                   ('H_init_seen_long', 'norm_log_slope_per_step', 64.0)):
        values = retention_by_source(retention, state_set, field, gain)
        if 'navigation' in values and 'matched_random' in values:
            stationarity.append(paired('navigation_minus_matched', values['navigation'],
                                       values['matched_random'], state_set=state_set, metric=field,
                                       gain='' if gain is None else gain))
    write_csv(out / 'stationarity_contrasts.csv', stationarity)

    result = {'schema_version': SCHEMA_VERSION, 'n_cells_complete': complete, 'n_cells_expected': 50,
              'margin_log2': MARGIN, 'tests': tests, 'stationarity': stationarity,
              'prediction_vs_reference_gain': reference}
    (out / 'DELTA_RESULT.json').write_text(json.dumps(result, indent=2), encoding='utf-8')

    lines = ["# E5b' rectified-displacement prediction of the operating gain\n",
             f'- completed cells: {complete} of 50\n',
             '## Pre-registered tests (log2 units; error = observation - prediction)\n',
             '| role | state set | observation | targets | n | prediction | observation | error | 95% CI | exact p | Spearman | verdict |',
             '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for t in tests:
        lines.append(f"| {t['role']} | {t['state_set']} | {t['test']} | {t['targets']} | {t['n_sources']} | "
                     f"{t['mean_prediction_log2']:+.3f} | {t['mean_observation_log2']:+.3f} | "
                     f"{t['mean_error_log2']:+.3f} | [{t['ci_low']:+.3f}, {t['ci_high']:+.3f}] | "
                     f"{t['exact_p']:.6f} | {t['spearman_pred_obs']:+.3f} | {t['verdict']} |")
    lines += ['', f'Decision rule: supported if the interval lies inside [-{MARGIN}, +{MARGIN}]; '
                  'falsified if exact p < 0.05 and the interval excludes 0; otherwise inconclusive.', '',
              '## Prediction against the reference gain (should be flat if delta is linear in g)\n',
              '| state set | reference gain | mean pred | min | max |', '|---|---|---|---|---|']
    for r in reference:
        lines.append(f"| {r['state_set']} | {r['reference_gain']} | {r['mean_pred_log2']:+.3f} | "
                     f"{r['min_pred_log2']:+.3f} | {r['max_pred_log2']:+.3f} |")
    lines += ['', '## Source-side stationarity, navigation minus matched\n',
              '| state set | metric | gain | navigation | matched | difference | 95% CI | exact p |',
              '|---|---|---|---|---|---|---|---|']
    for r in stationarity:
        lines.append(f"| {r['state_set']} | {r['metric']} | {r['gain']} | {r['navigation_mean']:.4f} | "
                     f"{r['matched_mean']:.4f} | {r['mean_difference']:+.4f} | "
                     f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] | {r['exact_p']:.6f} |")
    (out / 'DELTA_RESULT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
