#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

import intervention_common as common

SPLITS = ('id_seen_short', 'heldout_short', 'seen_long', 'heldout_long')

def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

def exact_sign_flip_p(values) -> float:
    x = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    n = len(x)
    if n == 0:
        return float('nan')
    observed = abs(float(x.mean()))
    hits = 0
    for mask in range(1 << n):
        signs = np.array([1.0 if (mask >> i) & 1 else -1.0 for i in range(n)])
        hits += abs(float((x * signs).mean())) >= observed - 1e-15
    return hits / float(1 << n)

def bootstrap_ci(values) -> tuple[float, float]:
    x = np.asarray(values, dtype=float)
    rng = np.random.default_rng(common.STATISTICS_SEED)
    means = np.empty(common.BOOTSTRAP_SAMPLES, dtype=float)
    for i in range(common.BOOTSTRAP_SAMPLES):
        means[i] = rng.choice(x, size=len(x), replace=True).mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

def load_intervention(root: Path):
    rows = []
    for cell in sorted(root.glob('source_*/target_*/arm_*')):
        if not (cell / 'COMPLETE.json').is_file():
            continue
        rows += common.read_csv(cell / 'metrics.csv')
    return rows

def load_e6_baselines(root: Path):
    rows = []
    for source in range(6101, 6111):
        for target in (84001, 84002, 84003):
            for rho in (0.0, 0.1):
                cell = root / f'source_{source}' / f'target_{target}' / f'rho_{int(round(rho*1000)):04d}'
                if not (cell / 'COMPLETE.json').is_file():
                    raise RuntimeError(f'missing E6 baseline cell: {cell}')
                rows += common.read_csv(cell / 'metrics.csv')
    return rows

def main() -> int:
    out = common.OUTPUT_ROOT / 'aggregate'
    out.mkdir(parents=True, exist_ok=True)
    intervention = load_intervention(common.OUTPUT_ROOT)
    expected_cells = 10 * 3 * len(common.ARMS)
    complete_cells = len(list(common.OUTPUT_ROOT.glob('source_*/target_*/arm_*/COMPLETE.json')))
    if complete_cells != expected_cells:
        raise RuntimeError(f'intervention incomplete: {complete_cells}/{expected_cells}')
    baseline = load_e6_baselines(common.E6_ROOT)

    int_map = {(int(r['source_seed']), int(r['target_seed']), r['arm'], r['split']): float(r['accuracy'])
               for r in intervention}
    base_map = {(int(r['source_seed']), int(r['target_seed']), round(float(r['rho']), 4), r['split']): float(r['accuracy'])
                for r in baseline}
    cell_rows = []
    for source in range(6101, 6111):
        for target in (84001, 84002, 84003):
            for arm, spec in common.ARMS.items():
                rho = round(float(spec['base_rho']), 4)
                for split in SPLITS:
                    key = (source, target, arm, split)
                    bkey = (source, target, rho, split)
                    if key not in int_map or bkey not in base_map:
                        raise RuntimeError(f'missing paired metric: {key}')
                    cell_rows.append({'source_seed': source, 'target_seed': target, 'arm': arm,
                                      'base_rho': rho, 'split': split,
                                      'intervention_accuracy': int_map[key], 'baseline_accuracy': base_map[bkey],
                                      'difference': int_map[key] - base_map[bkey]})
    write_csv(out / 'cell_contrasts.csv', cell_rows)

    source_bucket = defaultdict(list)
    for r in cell_rows:
        source_bucket[(r['source_seed'], r['arm'], r['split'])].append(r['difference'])
    source_rows = [{'source_seed': s, 'arm': arm, 'split': split, 'difference': float(np.mean(v))}
                   for (s, arm, split), v in sorted(source_bucket.items())]
    write_csv(out / 'source_contrasts.csv', source_rows)

    by = {(r['source_seed'], r['arm'], r['split']): r['difference'] for r in source_rows}
    tests = []
    directions = {'nav_impair': -1, 'rho01_rescue': 1}
    for arm in common.ARMS:
        for split in ('seen_long', 'heldout_long'):
            vals = [by[(s, arm, split)] for s in range(6101, 6111)]
            lo, hi = bootstrap_ci(vals)
            mean = float(np.mean(vals))
            direction = directions[arm]
            supported = bool((mean * direction > 0) and (exact_sign_flip_p(vals) < 0.05)
                             and ((hi < 0) if direction < 0 else (lo > 0)))
            tests.append({'role': 'primary' if split == 'seen_long' else 'secondary', 'test': f'{arm}_{split}',
                          'arm': arm, 'split': split, 'n_sources': len(vals), 'difference': mean,
                          'ci_low': lo, 'ci_high': hi, 'exact_p': exact_sign_flip_p(vals),
                          'n_positive': int(np.sum(np.asarray(vals) > 0)), 'supported': supported})
        long_vals = [by[(s, arm, 'seen_long')] for s in range(6101, 6111)]
        short_vals = [by[(s, arm, 'id_seen_short')] for s in range(6101, 6111)]
        vals = [a - b for a, b in zip(long_vals, short_vals)]
        lo, hi = bootstrap_ci(vals)
        mean = float(np.mean(vals))
        direction = directions[arm]
        supported = bool((mean * direction > 0) and (exact_sign_flip_p(vals) < 0.05)
                         and ((hi < 0) if direction < 0 else (lo > 0)))
        tests.append({'role': 'secondary', 'test': f'{arm}_seen_long_minus_id_seen_short_effect',
                      'arm': arm, 'split': 'difference_of_effects', 'n_sources': len(vals), 'difference': mean,
                      'ci_low': lo, 'ci_high': hi, 'exact_p': exact_sign_flip_p(vals),
                      'n_positive': int(np.sum(np.asarray(vals) > 0)), 'supported': supported})
    write_csv(out / 'tests.csv', tests)
    primary = [t for t in tests if t['role'] == 'primary']
    bidirectional = len(primary) == 2 and all(t['supported'] for t in primary)

    diagnostics = []
    for cell in sorted(common.OUTPUT_ROOT.glob('source_*/target_*/arm_*')):
        diagnostics += common.read_csv(cell / 'condition_diagnostics.csv')
    write_csv(out / 'all_condition_diagnostics.csv', diagnostics)
    diag_summary = []
    for arm in common.ARMS:
        rows = [r for r in diagnostics if r['arm'] == arm]
        diag_summary.append({'arm': arm, 'n_cells': len(rows),
                             'learned_gain_mean': float(np.mean([float(r['learned_gain']) for r in rows])),
                             'source_retention_before_mean': float(np.mean([float(r['source_retention_before']) for r in rows])),
                             'source_retention_after_exact_mean': float(np.mean([float(r['source_retention_after_exact']) for r in rows]))})
    write_csv(out / 'diagnostic_summary.csv', diag_summary)

    result = {'schema_version': common.SCHEMA_VERSION, 'status': 'complete',
              'n_cells': complete_cells, 'c': common.EXPECTED_C, 'inverse_c': 1.0/common.EXPECTED_C,
              'primary_tests': primary, 'bidirectional_causal_support': bidirectional,
              'all_tests': tests, 'diagnostic_summary': diag_summary}
    (out / 'INTERVENTION_RESULT.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    lines = ['# Autonomous-retention intervention', '',
             f'- completed cells: {complete_cells}/{expected_cells}',
             f'- c: {common.EXPECTED_C:.9f}; inverse c: {1.0/common.EXPECTED_C:.9f}',
             f'- bidirectional causal support: {"YES" if bidirectional else "NO"}', '',
             '## Registered tests', '',
             '| role | test | difference | 95% CI | exact p | positive sources | supported |',
             '|---|---|---:|---:|---:|---:|---|']
    for t in tests:
        lines.append(f"| {t['role']} | {t['test']} | {t['difference']:+.4f} | "
                     f"[{t['ci_low']:+.4f}, {t['ci_high']:+.4f}] | {t['exact_p']:.6f} | "
                     f"{t['n_positive']}/{t['n_sources']} | {'YES' if t['supported'] else 'NO'} |")
    lines += ['', '## Source-side retention targets', '',
              '| arm | before | after exact | learned gain |', '|---|---:|---:|---:|']
    for r in diag_summary:
        lines.append(f"| {r['arm']} | {r['source_retention_before_mean']:.6f} | "
                     f"{r['source_retention_after_exact_mean']:.6f} | {r['learned_gain_mean']:.2f} |")
    lines += ['', 'Interpretation is governed by ANALYSIS_PLAN_retention_intervention.md. No c retuning is permitted.']
    (out / 'INTERVENTION_RESULT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
