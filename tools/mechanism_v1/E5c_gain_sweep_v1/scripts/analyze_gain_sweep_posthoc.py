#!/usr/bin/env python3
"""E5c post-hoc robustness analysis.  Written AFTER the E5c results were seen.

Nothing here replaces the pre-registered readouts of ``collect_gain_sweep.py``;
that script and its ``aggregate/`` directory are left untouched.  This script
writes to ``aggregate_posthoc/`` and every table in it is labelled post hoc.

Why it exists
-------------
1. The id_seen_short curve of the navigation core is flat from g = 8 to 256
   (0.980-0.986 on average), inside two binomial standard errors of the test
   set, so the pre-registered argmax g* is dominated by sampling noise there.
2. The matched core never reaches the navigation level on seen_long at any
   gain, so "the gain the matched core needs on long paths" is not defined.
3. The rate surface in ``aggregate/`` takes the median over runs that reached
   the threshold only, which biases low gains towards fast times.

Readouts (all post hoc)
-----------------------
P1  Envelope.  Per source, max over g of the source-mean accuracy, navigation
    minus matched, per split; plus navigation minus matched at every gain and
    the count of (source, gain) cells in which navigation is higher.
P2  Plateau identifiability.  Gains whose accuracy is within two standard
    errors of the maximum, and the width of that set in log2 units.  The
    standard error is the larger of the binomial test-set error and the
    replication error, the pooled between-target spread at g >= 16 divided by
    the square root of the number of targets.  Test-set sampling alone
    understates how much a single training run moves.
P3  Threshold-crossing gain.  The log2 gain at which a source curve first
    reaches a fixed accuracy, linearly interpolated in log2 g.  Left-censored
    when g = 1 already reaches it (value 0, an upper bound), right-censored
    when no gain reaches it.  Matched minus navigation is reported where both
    are finite, with the censoring counts.
P4  Censoring-aware rate surface.  Updates to first reach a validation
    threshold per run, with non-reaching runs censored at the budget: the
    median over all runs (reported only when at least half reached) and the
    restricted mean min(t, budget).
P5  Learned versus fixed on the same targets.  E0.9a learned scalar gain at
    32,000 updates against the fixed-gain envelope and the single best fixed
    gain, restricted to the E5c targets.
P6  Over-gain sensitivity.  Drop from the source peak to g = 256, long split
    minus short split, per origin.
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

from gain_sweep_common import E09A_IDENTITY, E09A_ROOT, GAIN_GRID, ORIGINS, OUTPUT_ROOT, TARGET_SEEDS

STATISTICS_SEED = 982451653
BOOTSTRAP_SAMPLES = 20000
BUDGET = 32000
SPLIT_SIZE = {'validation_seen_short': 1000, 'id_seen_short': 2000, 'heldout_short': 2000,
              'seen_long': 2000, 'heldout_long': 2000}
SPLITS = tuple(SPLIT_SIZE)
THRESHOLDS = {'validation_seen_short': (0.85, 0.90, 0.95), 'id_seen_short': (0.85, 0.90, 0.95),
              'seen_long': (0.5, 0.8), 'heldout_long': (0.5, 0.8)}
RATE_THRESHOLDS = (0.5, 0.8, 0.95)
E09A_SCALAR = {'navigation': 'navigation_trainable_scalar_gain',
               'matched_random': 'matched_random_trainable_scalar_gain'}

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

def paired_row(name: str, values: dict[int, float], **extra) -> dict[str, Any]:
    array = np.array([values[s] for s in sorted(values)], dtype=np.float64)
    finite = array[np.isfinite(array)]
    low, high = bootstrap_ci(finite)
    return {'contrast': name, **extra, 'n_sources': int(finite.size),
            'mean_difference': float(finite.mean()) if finite.size else float('nan'),
            'ci_low': low, 'ci_high': high, 'exact_p': exact_sign_flip_p(finite),
            'n_positive': int(np.sum(finite > 0)), 'n_negative': int(np.sum(finite < 0))}

def load_runs(root: Path, e09a_root: Path):
    """Cell-level accuracy and validation curves, g = 1 joined from E0.9a."""
    accuracy: dict[tuple, float] = {}
    curves: dict[tuple, list[tuple[int, float]]] = defaultdict(list)
    missing = []
    for source_dir in sorted(root.glob('source_*')):
        for target_dir in sorted(source_dir.glob('target_*')):
            for gain in GAIN_GRID:
                cell = target_dir / f'gain_{gain:03d}'
                if not (cell / 'COMPLETE.json').is_file():
                    missing.append((source_dir.name, target_dir.name, gain))
                    continue
                for row in read_csv(cell / 'metrics.csv'):
                    key = (int(row['source_seed']), int(row['target_seed']), row['origin'],
                           int(float(row['gain'])), row['split'])
                    accuracy[key] = float(row['accuracy'])
                for row in read_csv(cell / 'update_curves.csv'):
                    key = (int(row['source_seed']), int(row['target_seed']), row['origin'],
                           int(float(row['gain'])))
                    curves[key].append((int(float(row['update'])), float(row['validation_accuracy'])))
    lookup = {v: k for k, v in E09A_IDENTITY.items()}
    learned: dict[tuple, float] = {}
    for cell in sorted(e09a_root.glob('source_*/target_*/group_*')):
        if not (cell / 'COMPLETE.json').is_file():
            continue
        target = int(cell.parent.name.split('_')[1])
        if target not in TARGET_SEEDS:
            continue
        for row in read_csv(cell / 'metrics.csv'):
            condition = row.get('condition')
            if condition in lookup:
                accuracy[(int(row['source_seed']), target, lookup[condition], 1, row['split'])] = float(row['accuracy'])
            for origin, name in E09A_SCALAR.items():
                if condition == name:
                    learned[(int(row['source_seed']), target, origin, row['split'])] = float(row['accuracy'])
        for row in read_csv(cell / 'update_curves.csv'):
            condition = row.get('condition')
            if condition in lookup:
                curves[(int(row['source_seed']), target, lookup[condition], 1)].append(
                    (int(float(row['update'])), float(row['validation_accuracy'])))
    return accuracy, curves, learned, missing

def source_surface(accuracy):
    bucket = defaultdict(list)
    for (source, target, origin, gain, split), value in accuracy.items():
        bucket[(source, origin, gain, split)].append(value)
    return ({k: float(np.mean(v)) for k, v in bucket.items()},
            {k: len(v) for k, v in bucket.items()})

def crossing_log2(gains: list[int], values: list[float], threshold: float) -> tuple[float, str]:
    if values[0] >= threshold:
        return 0.0, 'left_censored'
    for i in range(1, len(gains)):
        if values[i] >= threshold:
            x0, x1 = math.log2(gains[i - 1]), math.log2(gains[i])
            y0, y1 = values[i - 1], values[i]
            return x0 + (threshold - y0) * (x1 - x0) / (y1 - y0), 'observed'
    return float('inf'), 'right_censored'

def main() -> int:
    parser = argparse.ArgumentParser(description='E5c post-hoc robustness analysis')
    parser.add_argument('--root', type=Path, default=OUTPUT_ROOT)
    parser.add_argument('--e09a-root', type=Path, default=E09A_ROOT)
    args = parser.parse_args()
    out = args.root / 'aggregate_posthoc'
    out.mkdir(parents=True, exist_ok=True)

    accuracy, curves, learned, missing = load_runs(args.root, args.e09a_root)
    surface, n_targets = source_surface(accuracy)
    sources = sorted({k[0] for k in surface})
    gains = sorted({k[2] for k in surface})

    envelope, per_gain, dominance = [], [], []
    for split in SPLITS:
        diffs = {}
        for s in sources:
            nav = [surface[(s, 'navigation', g, split)] for g in gains if (s, 'navigation', g, split) in surface]
            mat = [surface[(s, 'matched_random', g, split)] for g in gains if (s, 'matched_random', g, split) in surface]
            if nav and mat:
                diffs[s] = max(nav) - max(mat)
        if diffs:
            envelope.append(paired_row('P1_envelope_nav_minus_matched', diffs, split=split))
        wins = total = 0
        for g in gains:
            values = {s: surface[(s, 'navigation', g, split)] - surface[(s, 'matched_random', g, split)]
                      for s in sources if (s, 'navigation', g, split) in surface
                      and (s, 'matched_random', g, split) in surface}
            if values:
                per_gain.append(paired_row('P1_nav_minus_matched_at_gain', values, split=split, gain=g))
                wins += sum(1 for v in values.values() if v > 0)
                total += len(values)
        dominance.append({'split': split, 'cells_nav_higher': wins, 'cells_total': total})
    write_csv(out / 'envelope_contrast.csv', envelope)
    write_csv(out / 'per_gain_contrast.csv', per_gain)
    write_csv(out / 'dominance_count.csv', dominance)

    spread = defaultdict(list)
    for (s, t, origin, g, split), value in accuracy.items():
        if g >= 16:
            spread[(origin, split, s, g)].append(value)
    pooled = defaultdict(list)
    for (origin, split, s, g), values in spread.items():
        if len(values) >= 2:
            pooled[(origin, split)].append(float(np.var(values, ddof=1)))
    replication_sd = {k: math.sqrt(float(np.mean(v))) for k, v in pooled.items()}
    plateau = []
    for s in sources:
        for origin in ORIGINS:
            for split in SPLITS:
                points = [(g, surface[(s, origin, g, split)], n_targets[(s, origin, g, split)])
                          for g in gains if (s, origin, g, split) in surface]
                if not points:
                    continue
                best_g, best, n = max(points, key=lambda x: (x[1], x[0]))
                binomial = math.sqrt(max(best * (1.0 - best), 1e-12) / (SPLIT_SIZE[split] * n))
                replication = replication_sd.get((origin, split), 0.0) / math.sqrt(n)
                se = max(binomial, replication)
                near = [g for g, a, _ in points if a >= best - 2.0 * se]
                plateau.append({'source_seed': s, 'origin': origin, 'split': split,
                                'argmax_gain': best_g, 'max_accuracy': best,
                                'binomial_se': binomial, 'replication_se': replication,
                                'two_se': 2.0 * se,
                                'gains_within_two_se': ' '.join(str(g) for g in near),
                                'plateau_width_log2': math.log2(max(near)) - math.log2(min(near))})
    write_csv(out / 'plateau_identifiability.csv', plateau)

    crossing, crossing_contrast = [], []
    for split, thresholds in THRESHOLDS.items():
        for threshold in thresholds:
            record = defaultdict(dict)
            for s in sources:
                for origin in ORIGINS:
                    pts = [(g, surface[(s, origin, g, split)]) for g in gains if (s, origin, g, split) in surface]
                    if not pts or pts[0][0] != 1:
                        continue
                    value, status = crossing_log2([p[0] for p in pts], [p[1] for p in pts], threshold)
                    record[s][origin] = (value, status)
                    crossing.append({'source_seed': s, 'origin': origin, 'split': split,
                                     'threshold': threshold, 'log2_gain': value, 'status': status})
            diffs, counts = {}, defaultdict(int)
            for s, pair in record.items():
                if len(pair) < 2:
                    continue
                (nav, nav_status), (mat, mat_status) = pair['navigation'], pair['matched_random']
                counts[f'nav_{nav_status}'] += 1
                counts[f'matched_{mat_status}'] += 1
                if math.isfinite(nav) and math.isfinite(mat):
                    diffs[s] = mat - nav
            if diffs:
                crossing_contrast.append(paired_row(
                    'P3_log2_threshold_gain_matched_minus_nav', diffs, split=split,
                    threshold=threshold, **dict(counts)))
    write_csv(out / 'threshold_gain_by_source.csv', crossing)
    write_csv(out / 'threshold_gain_contrast.csv', crossing_contrast)

    rate_rows, rate_contrast = [], []
    times = defaultdict(list)
    per_source_time = defaultdict(lambda: defaultdict(list))
    for (s, t, origin, g), points in curves.items():
        points = sorted(points)
        for threshold in RATE_THRESHOLDS:
            reached = [u for u, a in points if a >= threshold]
            value = float(min(reached)) if reached else float('inf')
            times[(origin, g, threshold)].append(value)
            per_source_time[(origin, g, threshold)][s].append(min(value, float(BUDGET)))
    for (origin, g, threshold), values in sorted(times.items()):
        array = np.sort(np.asarray(values))
        reached = int(np.isfinite(array).sum())
        median = float(np.median(array)) if reached * 2 >= array.size and np.isfinite(np.median(array)) else float('nan')
        rate_rows.append({'origin': origin, 'gain': g, 'threshold': threshold,
                          'n_runs': int(array.size), 'n_reached': reached,
                          'censored_median_updates': median,
                          'restricted_mean_updates': float(np.mean(np.minimum(array, BUDGET)))})
    for g in gains:
        for threshold in RATE_THRESHOLDS:
            nav = per_source_time.get(('navigation', g, threshold), {})
            mat = per_source_time.get(('matched_random', g, threshold), {})
            diffs = {s: float(np.mean(mat[s])) - float(np.mean(nav[s])) for s in nav if s in mat}
            if diffs:
                rate_contrast.append(paired_row('P4_restricted_mean_updates_matched_minus_nav',
                                                diffs, gain=g, threshold=threshold))
    write_csv(out / 'rate_surface_censored.csv', rate_rows)
    write_csv(out / 'rate_contrast.csv', rate_contrast)

    learned_rows = []
    for origin in ORIGINS:
        for split in SPLITS:
            per_source_learned = defaultdict(list)
            for (s, t, o, sp), value in learned.items():
                if o == origin and sp == split:
                    per_source_learned[s].append(value)
            vs_envelope, vs_best = {}, {}
            mean_curve = {g: np.mean([surface[(s, origin, g, split)] for s in sources
                                      if (s, origin, g, split) in surface]) for g in gains}
            best_gain = max(mean_curve, key=lambda g: mean_curve[g])
            for s, values in per_source_learned.items():
                fixed = [surface[(s, origin, g, split)] for g in gains if (s, origin, g, split) in surface]
                if fixed:
                    vs_envelope[s] = float(np.mean(values)) - max(fixed)
                if (s, origin, best_gain, split) in surface:
                    vs_best[s] = float(np.mean(values)) - surface[(s, origin, best_gain, split)]
            if vs_envelope:
                learned_rows.append(paired_row('P5_learned_minus_fixed_envelope', vs_envelope,
                                               origin=origin, split=split))
            if vs_best:
                learned_rows.append(paired_row('P5_learned_minus_best_single_fixed_gain', vs_best,
                                               origin=origin, split=split, best_fixed_gain=best_gain))
    write_csv(out / 'learned_vs_fixed_same_targets.csv', learned_rows)

    overgain = []
    top = max(gains)
    for origin in ORIGINS:
        diffs = {}
        for s in sources:
            drops = {}
            for split in ('id_seen_short', 'seen_long'):
                values = [surface[(s, origin, g, split)] for g in gains if (s, origin, g, split) in surface]
                if values and (s, origin, top, split) in surface:
                    drops[split] = max(values) - surface[(s, origin, top, split)]
            if len(drops) == 2:
                diffs[s] = drops['seen_long'] - drops['id_seen_short']
        if diffs:
            overgain.append(paired_row('P6_overgain_drop_long_minus_short', diffs, origin=origin,
                                       top_gain=top))
    write_csv(out / 'overgain_sensitivity.csv', overgain)

    result = {'posthoc': True, 'n_missing_cells': len(missing),
              'missing_cells': [f'{a}/{b}/gain_{g:03d}' for a, b, g in missing],
              'n_sources': len(sources), 'gains': gains,
              'envelope': envelope, 'dominance': dominance,
              'threshold_contrasts': crossing_contrast, 'learned_vs_fixed': learned_rows,
              'overgain': overgain}
    (out / 'POSTHOC_RESULT.json').write_text(json.dumps(result, indent=2), encoding='utf-8')

    lines = ['# E5c post-hoc robustness analysis (written after seeing E5c; not pre-registered)\n',
             f'- missing cells: {len(missing)}; sources: {len(sources)}; gains: {gains}\n',
             '## P1 envelope: per-source best gain, navigation minus matched\n',
             '| split | difference | 95% CI | exact p | nav higher | cells nav higher |',
             '|---|---|---|---|---|---|']
    dom = {d['split']: d for d in dominance}
    for r in envelope:
        d = dom[r['split']]
        lines.append(f"| {r['split']} | {r['mean_difference']:+.4f} | [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] | "
                     f"{r['exact_p']:.6f} | {r['n_positive']}/{r['n_sources']} | {d['cells_nav_higher']}/{d['cells_total']} |")
    widths = defaultdict(list)
    for r in plateau:
        widths[(r['origin'], r['split'])].append(r['plateau_width_log2'])
    lines += ['', '## P2 plateau width in log2 units (gains within two standard errors of the maximum; '
              'standard error = max(binomial, replication))\n',
              '| origin | split | median width | sources with width >= 2 |', '|---|---|---|---|']
    for (origin, split), values in sorted(widths.items()):
        lines.append(f'| {origin} | {split} | {np.median(values):.1f} | {sum(v >= 2 for v in values)}/{len(values)} |')
    lines += ['', '## P3 threshold-crossing gain, matched minus navigation (log2 units)\n',
              '| split | threshold | difference | 95% CI | exact p | n | censoring |', '|---|---|---|---|---|---|---|']
    for r in crossing_contrast:
        cens = ', '.join(f'{k}={v}' for k, v in r.items() if k.startswith(('nav_', 'matched_')) and 'censored' in k)
        lines.append(f"| {r['split']} | {r['threshold']} | {r['mean_difference']:+.3f} | "
                     f"[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}] | {r['exact_p']:.6f} | {r['n_sources']} | {cens or 'none'} |")
    lines += ['', '## P4 censoring-aware time to validation threshold (median over all runs; nan if fewer than half reached)\n',
              '| origin | threshold | ' + ' | '.join(str(g) for g in gains) + ' |', '|---|---|' + '---|' * len(gains)]
    table = {(r['origin'], r['gain'], r['threshold']): r for r in rate_rows}
    for origin in ORIGINS:
        for threshold in RATE_THRESHOLDS:
            cells = []
            for g in gains:
                r = table.get((origin, g, threshold))
                cells.append('-' if r is None else (f"{r['censored_median_updates']:.0f} ({r['n_reached']}/{r['n_runs']})"
                                                     if math.isfinite(r['censored_median_updates'])
                                                     else f"nan ({r['n_reached']}/{r['n_runs']})"))
            lines.append(f'| {origin} | {threshold} | ' + ' | '.join(cells) + ' |')
    lines += ['', '## P5 learned scalar gain (E0.9a, 32,000 updates) minus fixed gain, same targets\n',
              '| origin | split | contrast | difference | 95% CI | exact p |', '|---|---|---|---|---|---|']
    for r in learned_rows:
        lines.append(f"| {r['origin']} | {r['split']} | {r['contrast']} | {r['mean_difference']:+.4f} | "
                     f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] | {r['exact_p']:.6f} |")
    lines += ['', '## P6 over-gain sensitivity: (peak - g_max) on seen_long minus the same on id_seen_short\n',
              '| origin | difference | 95% CI | exact p | positive |', '|---|---|---|---|---|']
    for r in overgain:
        lines.append(f"| {r['origin']} | {r['mean_difference']:+.4f} | [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] | "
                     f"{r['exact_p']:.6f} | {r['n_positive']}/{r['n_sources']} |")
    (out / 'POSTHOC_RESULT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
