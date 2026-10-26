#!/usr/bin/env python3
"""E5c collector: build the operating curve and answer the three questions.

The g = 1 anchor is joined from E0.9a rather than rerun, because the E0.9a
identity conditions are the same model at the same budget on the same targets.

Pre-registered readouts
-----------------------
Q1  Shape.  accuracy(g) per split per origin, source-level means, with g = 1
    from E0.9a.  Descriptive, but the peak location is what Q2 and Q3 use.

Q2  Optimum split.  Per source, ``g*_short = argmax_g accuracy(g, id_seen_short)``
    and ``g*_long = argmax_g accuracy(g, seen_long)``.  Test
    ``log2(g*_long) - log2(g*_short)`` against zero with an exact sign-flip.
    A non-zero difference means the learned interface, selected on
    ``validation_seen_short`` alone, is not optimal for the long horizon.

Q3  Required magnitude.  Test ``log2(g*_matched) - log2(g*_nav)`` against zero.
    This is the version of the Phase F learned-gain contrast that does not
    depend on gain-learning dynamics.  Compare it to the learned-gain ratios
    already measured: 71.04/48.59 = 1.46 at 8,000 updates and about 93.7/74.4 =
    1.26 at 32,000 updates.

Q4  D2 completion.  Does small g sit on a genuinely lower plateau?  Reports
    ``accuracy(g=1)`` against ``max_g accuracy(g)`` per origin and split, and the
    end-of-budget slope at each g so that "low" can be distinguished from "slow".

Q5  Rate surface.  Updates needed to first reach validation accuracy 0.5 / 0.8 /
    0.95 as a function of g.  This is the direct test of the H5'' claim that the
    gain governs optimisation conditioning.

Usage
-----
    "$PYTHON_BIN" collect_gain_sweep.py --e09a-root .../E0_9a_extended_budget
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

from gain_sweep_common import (
    E09A_IDENTITY, E09A_ROOT, GAIN_GRID, ORIGINS, OUTPUT_ROOT, SCHEMA_VERSION,
    TARGET_SEEDS,
)

STATISTICS_SEED = 982451653
BOOTSTRAP_SAMPLES = 20000
SPLITS = ('validation_seen_short', 'id_seen_short', 'heldout_short',
          'seen_long', 'heldout_long')
RATE_THRESHOLDS = (0.5, 0.8, 0.95)


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open('r', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, '') for key in fields})
    tmp.replace(path)


def as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float('nan')


def exact_sign_flip_p(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    n = values.size
    if n == 0:
        return float('nan')
    observed = abs(float(values.mean()))
    return sum(
        1 for signs in itertools.product((1.0, -1.0), repeat=n)
        if abs(float(np.mean(np.asarray(signs) * values))) >= observed - 1e-12
    ) / float(2 ** n)


def bootstrap_ci(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float('nan'), float('nan')
    rng = np.random.default_rng(STATISTICS_SEED)
    index = rng.integers(0, values.size, size=(BOOTSTRAP_SAMPLES, values.size))
    means = values[index].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired(name: str, values: dict[int, float], expectation: str) -> dict[str, Any]:
    array = np.array([values[s] for s in sorted(values)], dtype=np.float64)
    low, high = bootstrap_ci(array)
    return {'contrast': name, 'n_sources': int(np.isfinite(array).sum()),
            'mean_difference': float(np.nanmean(array)) if array.size else float('nan'),
            'ci_low': low, 'ci_high': high,
            'exact_p': exact_sign_flip_p(array),
            'n_positive': int(np.sum(array > 0)), 'expectation': expectation}


def main() -> int:
    parser = argparse.ArgumentParser(description='E5c collector')
    parser.add_argument('--root', type=Path, default=OUTPUT_ROOT)
    parser.add_argument('--e09a-root', type=Path, default=E09A_ROOT)
    args = parser.parse_args()

    root: Path = args.root
    out = root / 'aggregate'
    out.mkdir(parents=True, exist_ok=True)

    cells, missing = [], []
    for source_dir in sorted(root.glob('source_*')):
        for target_dir in sorted(source_dir.glob('target_*')):
            for gain in GAIN_GRID:
                cell = target_dir / f'gain_{gain:03d}'
                (cells if (cell / 'COMPLETE.json').is_file() else missing).append(cell)
    if missing:
        print(f'WARNING: {len(missing)} incomplete cells, first few: '
              f'{[str(c) for c in missing[:5]]}')

    metrics, per_length, curves = [], [], []
    for cell in cells:
        metrics += read_csv(cell / 'metrics.csv')
        per_length += read_csv(cell / 'per_length.csv')
        curves += read_csv(cell / 'update_curves.csv')
    if not metrics:
        raise SystemExit('no completed E5c cells found')

    # ---- join the g = 1 anchor from E0.9a -------------------------------
    anchor_rows, anchor_curves = [], []
    for source_dir in sorted(args.e09a_root.glob('source_*')):
        for target_dir in sorted(source_dir.glob('target_*')):
            target = int(target_dir.name.split('_')[1])
            if target not in TARGET_SEEDS:
                continue
            cell = target_dir / 'group_identity_pair'
            if not (cell / 'COMPLETE.json').is_file():
                continue
            lookup = {v: k for k, v in E09A_IDENTITY.items()}
            for row in read_csv(cell / 'metrics.csv'):
                origin = lookup.get(row.get('condition'))
                if origin is None:
                    continue
                anchor_rows.append({**row, 'gain': 1, 'origin': origin,
                                    'source': 'E0_9a_identity'})
            for row in read_csv(cell / 'update_curves.csv'):
                origin = lookup.get(row.get('condition'))
                if origin is None:
                    continue
                anchor_curves.append({**row, 'gain': 1, 'origin': origin})
    metrics_all = metrics + anchor_rows
    curves_all = curves + anchor_curves
    if not anchor_rows:
        print('WARNING: no g=1 anchor joined from E0.9a; the curve starts at '
              f'g={min(GAIN_GRID)}')

    write_csv(out / 'all_metrics.csv', metrics_all, list(metrics_all[0].keys()))
    if per_length:
        write_csv(out / 'all_per_length.csv', per_length, list(per_length[0].keys()))

    # ---- Q1: source-level surface ---------------------------------------
    bucket: dict[tuple[int, str, int, str], list[float]] = defaultdict(list)
    for row in metrics_all:
        key = (int(row['source_seed']), str(row['origin']),
               int(as_float(row['gain'])), str(row['split']))
        bucket[key].append(as_float(row['accuracy']))
    surface = {k: float(np.mean(v)) for k, v in bucket.items() if v}

    gains_all = sorted({k[2] for k in surface})
    sources = sorted({k[0] for k in surface})
    surface_rows = [
        {'source_seed': s, 'origin': o, 'gain': g, 'split': sp, 'accuracy': surface[(s, o, g, sp)]}
        for (s, o, g, sp) in sorted(surface)
    ]
    write_csv(out / 'operating_curve.csv', surface_rows,
              ['source_seed', 'origin', 'gain', 'split', 'accuracy'])

    def mean_curve(origin: str, split: str) -> dict[int, float]:
        return {g: float(np.mean([surface[(s, origin, g, split)] for s in sources
                                  if (s, origin, g, split) in surface]))
                for g in gains_all
                if any((s, origin, g, split) in surface for s in sources)}

    # ---- Q2 and Q3: optima ----------------------------------------------
    def argmax_gain(source: int, origin: str, split: str) -> float:
        candidates = [(surface[(source, origin, g, split)], g) for g in gains_all
                      if (source, origin, g, split) in surface]
        return float(max(candidates)[1]) if candidates else float('nan')

    optima_rows = []
    for source in sources:
        for origin in ORIGINS:
            record = {'source_seed': source, 'origin': origin}
            for split in ('id_seen_short', 'seen_long', 'heldout_long',
                          'validation_seen_short'):
                record[f'gstar_{split}'] = argmax_gain(source, origin, split)
            optima_rows.append(record)
    write_csv(out / 'optima_by_source.csv', optima_rows,
              ['source_seed', 'origin', 'gstar_validation_seen_short',
               'gstar_id_seen_short', 'gstar_seen_long', 'gstar_heldout_long'])

    contrasts = []
    for origin in ORIGINS:
        diffs = {}
        for record in optima_rows:
            if record['origin'] != origin:
                continue
            short, long = record['gstar_id_seen_short'], record['gstar_seen_long']
            if math.isfinite(short) and math.isfinite(long) and short > 0 and long > 0:
                diffs[record['source_seed']] = math.log2(long) - math.log2(short)
        if diffs:
            contrasts.append(paired(
                f'Q2_log2_gstar_long_minus_short_{origin}', diffs,
                'non-zero means the short-horizon optimum is not the long-horizon optimum'))

    by_origin = {o: {r['source_seed']: r for r in optima_rows if r['origin'] == o}
                 for o in ORIGINS}
    for split in ('id_seen_short', 'seen_long'):
        diffs = {}
        for source in sources:
            left = by_origin['matched_random'].get(source, {}).get(f'gstar_{split}', float('nan'))
            right = by_origin['navigation'].get(source, {}).get(f'gstar_{split}', float('nan'))
            if math.isfinite(left) and math.isfinite(right) and left > 0 and right > 0:
                diffs[source] = math.log2(left) - math.log2(right)
        if diffs:
            contrasts.append(paired(
                f'Q3_log2_gstar_matched_minus_nav_{split}', diffs,
                'positive means the matched core needs a larger operating gain'))

    # ---- Q4: is small g low or merely slow? ------------------------------
    slope_rows = []
    per_run: dict[tuple[int, int, str, int], list[tuple[int, float]]] = defaultdict(list)
    for row in curves_all:
        key = (int(row['source_seed']), int(row['target_seed']),
               str(row['origin']), int(as_float(row['gain'])))
        per_run[key].append((int(as_float(row['update'])),
                             as_float(row['validation_accuracy'])))
    end_slope: dict[tuple[str, int], list[float]] = defaultdict(list)
    rate: dict[tuple[str, int, float], list[float]] = defaultdict(list)
    for (source, target, origin, gain), points in per_run.items():
        points.sort()
        if len(points) < 10:
            continue
        last = points[-1][0]
        window = [a for u, a in points if u > last - 1000]
        head = [a for u, a in points if last - 2000 < u <= last - 1000]
        if window and head:
            end_slope[(origin, gain)].append(float(np.mean(window)) - float(np.mean(head)))
        for threshold in RATE_THRESHOLDS:
            reached = [u for u, a in points if a >= threshold]
            rate[(origin, gain, threshold)].append(float(min(reached)) if reached else float('nan'))
    for (origin, gain), values in sorted(end_slope.items()):
        slope_rows.append({'origin': origin, 'gain': gain, 'n_runs': len(values),
                           'val_acc_gain_last_1000': float(np.mean(values))})
    if slope_rows:
        write_csv(out / 'end_of_budget_slope.csv', slope_rows,
                  ['origin', 'gain', 'n_runs', 'val_acc_gain_last_1000'])

    rate_rows = []
    for (origin, gain, threshold), values in sorted(rate.items()):
        finite = [v for v in values if math.isfinite(v)]
        rate_rows.append({
            'origin': origin, 'gain': gain, 'threshold': threshold,
            'n_runs': len(values), 'n_reached': len(finite),
            'median_updates_to_reach': float(np.median(finite)) if finite else float('nan')})
    if rate_rows:
        write_csv(out / 'rate_surface.csv', rate_rows,
                  ['origin', 'gain', 'threshold', 'n_runs', 'n_reached',
                   'median_updates_to_reach'])

    if contrasts:
        write_csv(out / 'gain_sweep_contrasts.csv', contrasts,
                  ['contrast', 'n_sources', 'mean_difference', 'ci_low', 'ci_high',
                   'exact_p', 'n_positive', 'expectation'])

    result = {'schema_version': SCHEMA_VERSION,
              'n_cells_complete': len(cells), 'n_cells_missing': len(missing),
              'n_anchor_rows_from_e09a': len(anchor_rows),
              'gains': gains_all, 'contrasts': contrasts,
              'end_of_budget_slope': slope_rows, 'rate_surface': rate_rows}
    (out / 'GAIN_SWEEP_RESULT.json').write_text(json.dumps(result, indent=2), encoding='utf-8')

    lines = ['# E5c fixed-gain operating curve\n',
             f'- completed cells: {len(cells)} (missing {len(missing)})',
             f'- g = 1 anchor rows joined from E0.9a: {len(anchor_rows)}',
             f'- gain grid: {gains_all}\n']
    for split in ('id_seen_short', 'seen_long'):
        lines += [f'## accuracy(g) on {split}\n',
                  '| origin | ' + ' | '.join(str(g) for g in gains_all) + ' |',
                  '|---|' + '---|' * len(gains_all)]
        for origin in ORIGINS:
            curve = mean_curve(origin, split)
            lines.append(f'| {origin} | ' + ' | '.join(
                f'{curve[g]:.3f}' if g in curve else '-' for g in gains_all) + ' |')
        lines.append('')
    if contrasts:
        lines += ['## Declared contrasts (log2 gain units)\n',
                  '| contrast | n | difference | 95% CI | exact p |',
                  '|---|---|---|---|---|']
        for record in contrasts:
            lines.append(
                f'| {record["contrast"]} | {record["n_sources"]} | '
                f'{record["mean_difference"]:+.3f} | '
                f'[{record["ci_low"]:+.3f}, {record["ci_high"]:+.3f}] | '
                f'{record["exact_p"]:.6f} |')
        lines += ['', 'A difference of +1.0 means a factor of two in gain.\n']
    if slope_rows:
        lines += ['## End-of-budget slope, low versus slow\n',
                  '| origin | ' + ' | '.join(str(g) for g in gains_all) + ' |',
                  '|---|' + '---|' * len(gains_all)]
        table = {(r['origin'], r['gain']): r['val_acc_gain_last_1000'] for r in slope_rows}
        for origin in ORIGINS:
            lines.append(f'| {origin} | ' + ' | '.join(
                f'{table[(origin, g)]:+.4f}' if (origin, g) in table else '-'
                for g in gains_all) + ' |')
        lines += ['', 'A gain whose accuracy is low AND whose slope is flat sits on a lower '
                      'plateau. A gain whose accuracy is low but whose slope is still steep is '
                      'merely slow, and E0.9a D2 cannot be strengthened from it.\n']
    (out / 'GAIN_SWEEP_RESULT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
