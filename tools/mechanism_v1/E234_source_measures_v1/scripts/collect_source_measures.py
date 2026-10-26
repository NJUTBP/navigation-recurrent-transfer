#!/usr/bin/env python3
"""Step 6 collector: E2/E3/E4 aggregated over (source, target) cells.

Readouts follow ANALYSIS_PLAN.md (mechanism_v1).  The primary inference of the
mechanism family is the E6 mixed model and lives in the E6 collector; this
collector reports, at every working point:
  - navigation minus matched, paired over sources (targets averaged within
    source first), exact sign-flip p and source-bootstrap interval;
  - the matched-core null margin (navigation - mean_matched) / sd_matched, the
    five matched cores of a source forming its null;
  - the random-direction null z already computed per cell;
  - secondary check 5: Spearman(R_ret, seen_long) over the 10 unperturbed
    navigation sources, labelled underpowered;
  - secondary check 6: Spearman(S_cons, seen_long), expected null;
  - R^2(L), R^2_diff(L) and c(t, 12) profiles for Figure 7B.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from run_source_measures import OUTPUT_ROOT, SCHEMA_VERSION
from source_measures import delta_common

PRIMARY_METRIC = 'R_ret_diff'
PRIMARY_WORKING_POINT = 'Bstar'
WORKING_POINTS = ('A', 'g3', 'g10', 'B8', 'B32', 'Bstar')
CORES = ('navigation', 'matched_random')
METRICS = ('R_ret', 'R_ret_diff', 'R2_5', 'R2_12', 'S_opp', 'S_orth', 'S_cons', 'r_eff', 'C_comm', 'C_inv')
STATISTICS_SEED = 982451653
BOOTSTRAP_SAMPLES = 20000
SCALAR = {'navigation': 'navigation_trainable_scalar_gain', 'matched_random': 'matched_random_trainable_scalar_gain'}

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
    means = values[rng.integers(0, values.size, size=(BOOTSTRAP_SAMPLES, values.size))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

def spearman(x, y) -> float:
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 3:
        return float('nan')
    return float(np.corrcoef(np.argsort(np.argsort(x)), np.argsort(np.argsort(y)))[0, 1])

def permutation_p(x, y, n=20000) -> float:
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    rho = abs(spearman(x, y))
    if not np.isfinite(rho):
        return float('nan')
    rng = np.random.default_rng(STATISTICS_SEED)
    return float(np.mean([abs(spearman(x, rng.permutation(y))) >= rho - 1e-12 for _ in range(n)]))

def seen_long_by_source(root: Path, group: str, origin: str) -> dict[int, float]:
    bucket = defaultdict(list)
    for cell in sorted(root.glob(f'source_*/target_*/{group}')):
        if not (cell / 'COMPLETE.json').is_file():
            continue
        for row in read_csv(cell / 'metrics.csv'):
            if row.get('condition') == SCALAR[origin] and row.get('split') == 'seen_long':
                bucket[int(row['source_seed'])].append(float(row['accuracy']))
    return {s: float(np.mean(v)) for s, v in bucket.items()}

def main() -> int:
    parser = argparse.ArgumentParser(description='Step 6 collector')
    parser.add_argument('--root', type=Path, default=OUTPUT_ROOT)
    parser.add_argument('--phasef-root', type=Path, default=delta_common.PHASEF_OUTPUT)
    parser.add_argument('--e09a-root', type=Path, default=delta_common.E09A_ROOT)
    args = parser.parse_args()
    out = args.root / 'aggregate'
    out.mkdir(parents=True, exist_ok=True)

    tables = defaultdict(list)
    complete = 0
    for cell in sorted(args.root.glob('source_*/target_*')):
        if not (cell / 'COMPLETE.json').is_file():
            continue
        complete += 1
        for name in ('working_points', 'geometry', 'composition', 'horizon_summary',
                     'horizon_by_length', 'contribution'):
            tables[name] += read_csv(cell / f'{name}.csv')
    if not complete:
        raise SystemExit('no completed Step 6 cells found')
    for name, rows in tables.items():
        write_csv(out / f'all_{name}.csv', rows)

    cell_value = defaultdict(dict)
    for name in ('geometry', 'composition', 'horizon_summary'):
        for row in tables[name]:
            for metric in METRICS:
                if metric in row and row[metric] not in ('', None):
                    key = (metric, row['working_point'], row['core'])
                    cell_value[key][(int(row['source_seed']), int(row['target_seed']))] = float(row[metric])
    def by_source(table):
        bucket = defaultdict(list)
        for (s, _), v in table.items():
            if np.isfinite(v):
                bucket[s].append(v)
        return {s: float(np.mean(v)) for s, v in bucket.items()}

    contrasts, margins = [], []
    for metric in METRICS:
        for wp in WORKING_POINTS:
            nav = by_source(cell_value.get((metric, wp, 'navigation'), {}))
            mat_cells = cell_value.get((metric, wp, 'matched_random'), {})
            mat = by_source(mat_cells)
            common = sorted(set(nav) & set(mat))
            if not common:
                continue
            diffs = np.array([nav[s] - mat[s] for s in common])
            low, high = bootstrap_ci(diffs)
            contrasts.append({'metric': metric, 'working_point': wp,
                              'primary': (metric, wp) == (PRIMARY_METRIC, PRIMARY_WORKING_POINT),
                              'n_sources': len(common), 'navigation_mean': float(np.mean([nav[s] for s in common])),
                              'matched_mean': float(np.mean([mat[s] for s in common])),
                              'difference': float(diffs.mean()), 'ci_low': low, 'ci_high': high,
                              'exact_p': exact_sign_flip_p(diffs), 'n_positive': int((diffs > 0).sum())})
            z = []
            for s in common:
                null = np.array([v for (ss, _), v in mat_cells.items() if ss == s and np.isfinite(v)])
                if null.size >= 2 and null.std(ddof=1) > 0:
                    z.append((nav[s] - null.mean()) / null.std(ddof=1))
            if z:
                margins.append({'metric': metric, 'working_point': wp, 'null': 'matched_core',
                                'n_sources': len(z), 'mean_margin_z': float(np.mean(z)),
                                'min_margin_z': float(np.min(z)), 'max_margin_z': float(np.max(z))})
    for row_metric in ('S_opp', 'S_orth', 'S_cons', 'r_eff'):
        for wp in WORKING_POINTS:
            for core in CORES:
                values = [float(r[f'{row_metric}_z']) for r in tables['geometry']
                          if r['working_point'] == wp and r['core'] == core and r.get(f'{row_metric}_z') not in ('', None)]
                if values:
                    margins.append({'metric': row_metric, 'working_point': wp, 'null': f'random_direction_{core}',
                                    'n_sources': len(values), 'mean_margin_z': float(np.mean(values)),
                                    'min_margin_z': float(np.min(values)), 'max_margin_z': float(np.max(values))})
    write_csv(out / 'core_contrasts.csv', contrasts)
    write_csv(out / 'null_margins.csv', margins)

    checks = []
    accuracy = {('navigation', 8000): seen_long_by_source(args.phasef_root, 'group_learned_interface', 'navigation'),
                ('navigation', 32000): seen_long_by_source(args.e09a_root, 'group_scalar_pair', 'navigation'),
                ('matched_random', 8000): seen_long_by_source(args.phasef_root, 'group_learned_interface', 'matched_random'),
                ('matched_random', 32000): seen_long_by_source(args.e09a_root, 'group_scalar_pair', 'matched_random')}
    for (core, budget), acc in accuracy.items():
        for metric, label in (('R_ret_diff', 'check5_R_ret_diff'), ('R_ret', 'check5_plan_literal_R_ret'),
                              ('S_cons', 'check6_S_cons_expected_null')):
            for wp in WORKING_POINTS:
                values = by_source(cell_value.get((metric, wp, core), {}))
                common = sorted(set(values) & set(acc))
                if len(common) < 3:
                    continue
                x, y = [values[s] for s in common], [acc[s] for s in common]
                checks.append({'check': label, 'core': core, 'budget': budget, 'working_point': wp,
                               'primary': (metric, wp) == (PRIMARY_METRIC, PRIMARY_WORKING_POINT),
                               'n_sources': len(common),
                               'spearman': spearman(x, y), 'permutation_p': permutation_p(x, y),
                               'status': 'underpowered consistency check' if core == 'navigation' else 'exploratory'})
    write_csv(out / 'consistency_checks.csv', checks)

    profile = defaultdict(list)
    for row in tables['horizon_by_length']:
        for field in ('R2', 'R2_diff', 'O_L', 'signal_vs_zero_input', 'hidden_norm', 'active_fraction'):
            if row.get(field) not in ('', None):
                profile[(row['core'], row['working_point'], int(row['L']), field)].append(float(row[field]))
    profile_rows = [{'core': c, 'working_point': wp, 'L': L, 'field': f, 'mean': float(np.mean(v)), 'n_cells': len(v)}
                    for (c, wp, L, f), v in sorted(profile.items())]
    write_csv(out / 'horizon_profile.csv', profile_rows)
    spectrum = defaultdict(list)
    for row in tables['contribution']:
        spectrum[(row['core'], row['working_point'], int(row['t']))].append(float(row['c_t']))
    write_csv(out / 'contribution_spectrum.csv',
              [{'core': c, 'working_point': wp, 't': t, 'mean_c_t': float(np.mean(v)), 'n_cells': len(v)}
               for (c, wp, t), v in sorted(spectrum.items())])

    result = {'schema_version': SCHEMA_VERSION, 'n_cells_complete': complete, 'n_cells_expected': 50,
              'primary_metric': PRIMARY_METRIC, 'primary_working_point': PRIMARY_WORKING_POINT,
              'contrasts': contrasts, 'checks': checks}
    (out / 'SOURCE_MEASURES_RESULT.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    lines = ['# Step 6: E2/E3/E4 source-side measures\n', f'- completed cells: {complete} of 50',
             f'- primary (ANALYSIS_PLAN.md): {PRIMARY_METRIC} at {PRIMARY_WORKING_POINT}\n',
             '## Navigation minus matched, paired over sources\n',
             '| metric | working point | navigation | matched | difference | 95% CI | exact p | nav higher |',
             '|---|---|---|---|---|---|---|---|']
    for r in contrasts:
        mark = ' (primary)' if r['primary'] else ''
        lines.append(f"| {r['metric']} | {r['working_point']}{mark} | {r['navigation_mean']:.4f} | {r['matched_mean']:.4f} | "
                     f"{r['difference']:+.4f} | [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] | {r['exact_p']:.6f} | "
                     f"{r['n_positive']}/{r['n_sources']} |")
    lines += ['', '## Consistency checks over sources (Spearman, permutation p)\n',
              '| check | core | budget | working point | n | rho | p | status |', '|---|---|---|---|---|---|---|---|']
    for r in checks:
        lines.append(f"| {r['check']} | {r['core']} | {r['budget']} | {r['working_point']} | {r['n_sources']} | "
                     f"{r['spearman']:+.3f} | {r['permutation_p']:.4f} | {r['status']} |")
    (out / 'SOURCE_MEASURES_RESULT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines[:40]))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
