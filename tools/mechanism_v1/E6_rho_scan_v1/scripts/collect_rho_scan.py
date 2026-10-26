#!/usr/bin/env python3
"""E6 collector: the sign-mixing scan and the primary mediation inference.

Pre-registered in ANALYSIS_PLAN.md (mechanism_v1); targets are averaged within
(source, rho) before any test.

Main      seen_long falls with rho on [0, 0.5]: per-source OLS slope of
          accuracy on rho, exact sign-flip over the 10 sources.
Anchors   rho = 0 against the navigation core trained identically (E0.9a,
          same sources and targets), and rho = 0.9 against rho = 0.1; both are
          provably equivalent networks, so the differences must sit inside
          optimisation noise.
Primary   long_acc ~ R + (1 | source), n = 10 x 6 = 60, with R = PRIMARY_METRIC
          at PRIMARY_WORKING_POINT; the plan-literal pair (R_ret at B) is
          reported beside it under its own label.  The fixed slope is estimated within source (source
          means removed), which is the random-intercept slope when intercepts
          absorb the between-source variation.  95 % interval: bootstrap over
          sources.  p: permutation of R_ret across the six levels inside each
          source, source as the exchangeability unit.
Secondary the same model at the other working points and for R_ret_diff;
          learned gain against rho; S_cons against long_acc (expected null).
Not run   the per-level navigation-readout control (separability); see
          ANALYSIS_PLAN.md.
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

from run_rho_scan import OUTPUT_ROOT, SCHEMA_VERSION
from rho_core import RHO_GRID, SCAN_RHOS
from source_measures import delta_common

PRIMARY_METRIC = 'R_ret_diff'
PRIMARY_WORKING_POINT = 'Bstar'
PLAN_LITERAL = ('R_ret', 'B')
WORKING_POINTS = ('A', 'g3', 'g10', 'B', 'Bstar')
STATISTICS_SEED = 982451653
BOOTSTRAP_SAMPLES = 20000
PERMUTATIONS = 20000

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

def paired(name, left: dict, right: dict, **extra) -> dict[str, Any]:
    common = sorted(set(left) & set(right))
    diffs = np.array([left[s] - right[s] for s in common], dtype=np.float64)
    low, high = bootstrap_ci(diffs)
    return {'test': name, **extra, 'n_sources': len(common), 'difference': float(diffs.mean()) if diffs.size else float('nan'),
            'ci_low': low, 'ci_high': high, 'exact_p': exact_sign_flip_p(diffs), 'n_positive': int((diffs > 0).sum())}

def within_slope(groups: dict[int, tuple[np.ndarray, np.ndarray]]) -> float:
    num = den = scale = 0.0
    for x, y in groups.values():
        xc, yc = x - x.mean(), y - y.mean()
        num += float((xc * yc).sum())
        den += float((xc * xc).sum())
        scale += float((x * x).sum())
    return num / den if den > 1e-12 * max(scale, 1e-300) else float('nan')

def mediation(groups: dict[int, tuple[np.ndarray, np.ndarray]], **extra) -> dict[str, Any]:
    sources = sorted(groups)
    slope = within_slope(groups)
    if not np.isfinite(slope):
        return {**extra, 'status': 'degenerate_no_within_source_variation', 'n_sources': len(sources),
                'n_observations': int(sum(len(x) for x, _ in groups.values())), 'within_slope': float('nan'),
                'ci_low': float('nan'), 'ci_high': float('nan'), 'permutation_p': float('nan'),
                'within_r': float('nan'), 'per_source_slope_sign_flip_p': float('nan'),
                'per_source_negative': 0, 'per_source_positive': 0}
    rng = np.random.default_rng(STATISTICS_SEED)
    boot = []
    for _ in range(BOOTSTRAP_SAMPLES):
        pick = rng.choice(sources, size=len(sources), replace=True)
        boot.append(within_slope({i: groups[s] for i, s in enumerate(pick)}))
    boot = np.asarray([b for b in boot if np.isfinite(b)])
    hits = 0
    for _ in range(PERMUTATIONS):
        shuffled = {s: (rng.permutation(x), y) for s, (x, y) in groups.items()}
        hits += abs(within_slope(shuffled)) >= abs(slope) - 1e-15
    per_source = []
    for x, y in groups.values():
        xc = x - x.mean()
        per_source.append(float((xc * (y - y.mean())).sum() / (xc * xc).sum()) if (xc * xc).sum() > 0 else float('nan'))
    xs = np.concatenate([x - x.mean() for x, _ in groups.values()])
    ys = np.concatenate([y - y.mean() for _, y in groups.values()])
    return {**extra, 'status': 'estimated', 'n_sources': len(sources),
            'n_observations': int(sum(len(x) for x, _ in groups.values())), 'within_slope': slope, 'ci_low': float(np.percentile(boot, 2.5)) if boot.size else float('nan'),
            'ci_high': float(np.percentile(boot, 97.5)) if boot.size else float('nan'),
            'permutation_p': (hits + 1) / (PERMUTATIONS + 1),
            'within_r': float(np.corrcoef(xs, ys)[0, 1]) if xs.std() > 0 and ys.std() > 0 else float('nan'),
            'per_source_slope_sign_flip_p': exact_sign_flip_p(per_source),
            'per_source_negative': int(np.sum(np.asarray(per_source) < 0)),
            'per_source_positive': int(np.sum(np.asarray(per_source) > 0))}

def main() -> int:
    parser = argparse.ArgumentParser(description='E6 collector')
    parser.add_argument('--root', type=Path, default=OUTPUT_ROOT)
    parser.add_argument('--e09a-root', type=Path, default=delta_common.E09A_ROOT)
    args = parser.parse_args()
    out = args.root / 'aggregate'
    out.mkdir(parents=True, exist_ok=True)

    tables = defaultdict(list)
    complete = 0
    for cell in sorted(args.root.glob('source_*/target_*/rho_*')):
        if not (cell / 'COMPLETE.json').is_file():
            continue
        complete += 1
        for name in ('metrics', 'condition_diagnostics', 'horizon_summary', 'geometry', 'working_points',
                     'downstream_profile', 'per_length'):
            tables[name] += read_csv(cell / f'{name}.csv')
    if not complete:
        raise SystemExit('no completed E6 cells found')
    for name, rows in tables.items():
        write_csv(out / f'all_{name}.csv', rows)

    def level_mean(rows, field, filt=lambda r: True):
        bucket = defaultdict(list)
        for r in rows:
            if filt(r) and r.get(field) not in ('', None):
                bucket[(int(r['source_seed']), round(float(r['rho']), 4))].append(float(r[field]))
        return {k: float(np.mean(v)) for k, v in bucket.items()}
    accuracy = {split: level_mean(tables['metrics'], 'accuracy', lambda r, s=split: r['split'] == s)
                for split in ('id_seen_short', 'heldout_short', 'seen_long', 'heldout_long')}
    gain = level_mean(tables['condition_diagnostics'], 'learned_gain')
    sources = sorted({s for s, _ in accuracy['seen_long']})

    curve = []
    for rho in RHO_GRID:
        row = {'rho': rho}
        for split, table in accuracy.items():
            values = [table[(s, round(rho, 4))] for s in sources if (s, round(rho, 4)) in table]
            row[f'{split}_mean'] = float(np.mean(values)) if values else float('nan')
            row[f'{split}_n_sources'] = len(values)
        values = [gain[(s, round(rho, 4))] for s in sources if (s, round(rho, 4)) in gain]
        row['learned_gain_mean'] = float(np.mean(values)) if values else float('nan')
        curve.append(row)
    write_csv(out / 'rho_curve.csv', curve)

    tests = []
    slopes = {}
    for s in sources:
        pts = [(rho, accuracy['seen_long'][(s, round(rho, 4))]) for rho in SCAN_RHOS if (s, round(rho, 4)) in accuracy['seen_long']]
        if len(pts) >= 3:
            x, y = np.array(pts).T
            slopes[s] = float(np.polyfit(x, y, 1)[0])
    low, high = bootstrap_ci(list(slopes.values()))
    tests.append({'test': 'main_seen_long_slope_on_rho', 'n_sources': len(slopes),
                  'difference': float(np.mean(list(slopes.values()))) if slopes else float('nan'),
                  'ci_low': low, 'ci_high': high, 'exact_p': exact_sign_flip_p(list(slopes.values())),
                  'n_positive': int(sum(v > 0 for v in slopes.values()))})
    nav = defaultdict(list)
    for cell in sorted(args.e09a_root.glob('source_*/target_*/group_scalar_pair')):
        for row in read_csv(cell / 'metrics.csv'):
            if row.get('condition') == 'navigation_trainable_scalar_gain':
                nav[(int(row['source_seed']), row['split'])].append(float(row['accuracy']))
    for split in ('id_seen_short', 'seen_long', 'heldout_long'):
        rho0 = {s: accuracy[split][(s, 0.0)] for s in sources if (s, 0.0) in accuracy[split]}
        navs = {s: float(np.mean(nav[(s, split)])) for s in sources if (s, split) in nav}
        tests.append(paired('anchor_rho0_minus_navigation_E09a', rho0, navs, split=split))
        low_side = {s: accuracy[split][(s, 0.1)] for s in sources if (s, 0.1) in accuracy[split]}
        high_side = {s: accuracy[split][(s, 0.9)] for s in sources if (s, 0.9) in accuracy[split]}
        tests.append(paired('symmetry_rho0p9_minus_rho0p1', high_side, low_side, split=split))
    write_csv(out / 'main_and_anchor_tests.csv', tests)

    measures = {}
    for metric in ('R_ret', 'R_ret_diff'):
        for wp in WORKING_POINTS:
            measures[(metric, wp)] = level_mean(tables['horizon_summary'], metric, lambda r, w=wp: r['working_point'] == w)
    for wp in WORKING_POINTS:
        measures[('S_cons', wp)] = level_mean(tables['geometry'], 'S_cons', lambda r, w=wp: r['working_point'] == w)
    mediations = []
    for (metric, wp), table in measures.items():
        groups = {}
        for s in sources:
            pts = [(table[(s, round(r, 4))], accuracy['seen_long'][(s, round(r, 4))]) for r in SCAN_RHOS
                   if (s, round(r, 4)) in table and (s, round(r, 4)) in accuracy['seen_long']
                   and np.isfinite(table[(s, round(r, 4))])]
            if len(pts) >= 3:
                x, y = np.array(pts).T
                groups[s] = (x, y)
        if len(groups) >= 3:
            role = ('primary' if (metric, wp) == (PRIMARY_METRIC, PRIMARY_WORKING_POINT)
                    else 'plan_literal' if (metric, wp) == PLAN_LITERAL
                    else 'expected_null_control' if metric == 'S_cons' else 'secondary')
            mediations.append(mediation(groups, metric=metric, working_point=wp, role=role))
    available = sum(1 for r in tables['working_points']
                    if r['working_point'] == PRIMARY_WORKING_POINT and r.get('gain') not in ('', None)
                    and np.isfinite(float(r['gain'])))
    if not any(m['role'] == 'primary' for m in mediations):
        mediations.insert(0, {'metric': PRIMARY_METRIC, 'working_point': PRIMARY_WORKING_POINT, 'role': 'primary',
                              'status': 'not_available', 'n_sources': 0, 'n_observations': 0})
    write_csv(out / 'mediation.csv', mediations)

    result = {'schema_version': SCHEMA_VERSION, 'n_cells_complete': complete, 'n_cells_expected': 210,
              'primary_metric': PRIMARY_METRIC, 'primary_working_point': PRIMARY_WORKING_POINT,
              'cells_with_primary_working_point': available,
              'curve': curve, 'tests': tests, 'mediation': mediations,
              'not_run': 'per-level navigation-readout control (separability)'}
    (out / 'RHO_SCAN_RESULT.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    lines = ['# E6 sign-mixing scan\n', f'- completed cells: {complete} of 210',
             f'- primary (ANALYSIS_PLAN.md): {PRIMARY_METRIC} at {PRIMARY_WORKING_POINT}; '
             f'cells where it could be set: {available} of {complete}\n',
             '## Accuracy and learned gain by rho (source means)\n',
             '| rho | id_seen_short | seen_long | heldout_long | learned gain | sources |', '|---|---|---|---|---|---|']
    for r in curve:
        lines.append(f"| {r['rho']:g} | {r['id_seen_short_mean']:.4f} | {r['seen_long_mean']:.4f} | "
                     f"{r['heldout_long_mean']:.4f} | {r['learned_gain_mean']:.2f} | {r['seen_long_n_sources']} |")
    lines += ['', '## Main test and anchors\n', '| test | split | n | difference or slope | 95% CI | exact p | positive |',
              '|---|---|---|---|---|---|---|']
    for t in tests:
        lines.append(f"| {t['test']} | {t.get('split', 'seen_long')} | {t['n_sources']} | {t['difference']:+.4f} | "
                     f"[{t['ci_low']:+.4f}, {t['ci_high']:+.4f}] | {t['exact_p']:.6f} | {t['n_positive']} |")
    lines += ['', '## Mediation: seen_long ~ measure + (1|source), within-source slope\n',
              '| role | measure | working point | n | slope | 95% CI | permutation p | within r | sources with negative slope |',
              '|---|---|---|---|---|---|---|---|---|']
    for m in mediations:
        if m['status'] != 'estimated':
            lines.append(f"| {m['role']} | {m['metric']} | {m['working_point']} | {m['n_observations']} | "
                         f"{m['status']} | | | | |")
            continue
        lines.append(f"| {m['role']} | {m['metric']} | {m['working_point']} | {m['n_observations']} | {m['within_slope']:+.4f} | "
                     f"[{m['ci_low']:+.4f}, {m['ci_high']:+.4f}] | {m['permutation_p']:.5f} | {m['within_r']:+.3f} | "
                     f"{m['per_source_negative']}/{m['n_sources']} |")
    lines += ['', 'Not run in this package: the per-level navigation-readout control (separability test).']
    (out / 'RHO_SCAN_RESULT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
