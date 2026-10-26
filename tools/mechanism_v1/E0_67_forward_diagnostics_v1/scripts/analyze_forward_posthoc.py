#!/usr/bin/env python3
"""E0.6 / E0.7 / E0.4 post-hoc analysis.  Written AFTER the results were seen.

``collect_forward_diagnostics.py`` and its ``aggregate/`` stay untouched; this
script reads ``aggregate/`` and writes ``aggregate_posthoc/``.

Why it exists
-------------
1. The chance-normalised drop divides by (accuracy - chance).  For last_k the
   absolute ablated accuracy of both cores is about 0.07, so the normalised
   difference mostly restates the difference in intact accuracy.
2. last_k needs an algorithmic reference.  An exact integrator given only the
   last k relations is right exactly when the earlier relations sum to zero.
   That probability is computed here from the regenerated splits.
3. The state diagnostics show a stationarity difference that the collector does
   not summarise: the navigation state keeps its norm along a trajectory while
   the matched state shrinks, so the relative drive grows with path length.

Readouts (all post hoc; associations are exploratory)
-----------------------------------------------------
A1  Absolute accuracy under every ablation, source means, navigation minus
    matched per mode.
A2  Exact ideal-integrator accuracy under last_k per split, and the excess of
    each condition over it.  Shuffle context: the share of permuted sequences
    that leave the 8x8 grid or create a held-out bigram (Monte Carlo, numpy).
A3  Stationarity per (source, target, condition): per-step norm retention
    exp(slope of log||h_t|| over t = 3..12), delta_rectified(12)/(3), and
    delta_rectified at step 1.  Navigation minus matched, paired over sources.
    Within-origin Spearman of retention against intact seen_long accuracy.
A4  Error structure: growth exponent of the error size with path length,
    error size conditional on being wrong, the share of wrong answers that
    land on an adjacent node, and the row/column variance ratio.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from forward_common import CHANCE, OUTPUT_ROOT, PHASEF_PROTOCOL, load_phasef_modules

STATISTICS_SEED = 982451653
BOOTSTRAP_SAMPLES = 20000
PAIRS = (('navigation_trainable_scalar_gain', 'matched_random_trainable_scalar_gain'),
         ('navigation_fixed_gain10_frozen', 'matched_random_fixed_gain10_frozen'))
SPLITS = ('id_seen_short', 'heldout_short', 'seen_long', 'heldout_long')
MODES = ('intact', 'no_recurrence', 'last_1', 'last_2', 'last_3', 'shuffle_order')
SHUFFLE_DRAWS = 20
N_LEVELS = 8

def read_rows(path: Path) -> list[dict[str, Any]]:
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

def paired(name: str, left: dict[int, float], right: dict[int, float], **extra) -> dict[str, Any]:
    common = sorted(set(left) & set(right))
    diffs = np.array([left[s] - right[s] for s in common], dtype=np.float64)
    low, high = bootstrap_ci(diffs)
    return {'contrast': name, **extra, 'n_sources': len(common),
            'left_mean': float(np.mean([left[s] for s in common])) if common else float('nan'),
            'right_mean': float(np.mean([right[s] for s in common])) if common else float('nan'),
            'mean_difference': float(diffs.mean()) if diffs.size else float('nan'),
            'ci_low': low, 'ci_high': high, 'exact_p': exact_sign_flip_p(diffs),
            'n_positive': int(np.sum(diffs > 0))}

def spearman(x, y) -> float:
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if x.size < 3:
        return float('nan')
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    return float(np.corrcoef(rx, ry)[0, 1])

def source_mean(table: dict[tuple, float]) -> dict[int, float]:
    bucket = defaultdict(list)
    for (source, _target), value in table.items():
        if np.isfinite(value):
            bucket[source].append(value)
    return {s: float(np.mean(v)) for s, v in bucket.items() if v}

def ideal_references(targets: list[int]) -> dict[tuple, dict[str, float]]:
    fullscale, _ = load_phasef_modules()
    protocol = json.loads(PHASEF_PROTOCOL.read_text(encoding='utf-8'))
    held = {tuple(int(v) for v in pair) for pair in protocol['held_out_action_bigrams']}
    rng = np.random.default_rng(STATISTICS_SEED)
    result = {}
    for target in targets:
        pools, manifest = fullscale.build_fullscale_split(protocol, target)
        if manifest['manifest_hash'] != protocol['expected_manifest_hashes'][str(target)]:
            raise RuntimeError(f'manifest hash mismatch for target {target}')
        for split in SPLITS:
            episodes = pools[split]
            stats = defaultdict(float)
            for nodes in episodes:
                rows, cols = np.divmod(np.asarray(nodes, dtype=np.int64), N_LEVELS)
                steps = np.stack([np.diff(rows), np.diff(cols)], axis=1)
                length = steps.shape[0]
                for k in (1, 2, 3):
                    prefix = steps[:max(length - k, 0)]
                    stats[f'ideal_last_{k}'] += float(prefix.shape[0] == 0 or not prefix.sum(axis=0).any())
                actions = [fullscale.action_id(a, b) for a, b in zip(nodes[:-1], nodes[1:])]
                for _ in range(SHUFFLE_DRAWS):
                    order = rng.permutation(length)
                    moved = steps[order]
                    path_r = rows[0] + np.cumsum(moved[:, 0])
                    path_c = cols[0] + np.cumsum(moved[:, 1])
                    stats['shuffle_leaves_grid'] += float(
                        (path_r.min() < 0) or (path_r.max() >= N_LEVELS)
                        or (path_c.min() < 0) or (path_c.max() >= N_LEVELS)) / SHUFFLE_DRAWS
                    permuted = [actions[i] for i in order]
                    stats['shuffle_has_heldout_bigram'] += float(any(
                        (permuted[i], permuted[i + 1]) in held for i in range(length - 1))) / SHUFFLE_DRAWS
            result[(target, split)] = {k: v / len(episodes) for k, v in stats.items()}
    return result

def log_slope(steps, values) -> float:
    x = np.asarray(steps, dtype=np.float64)
    y = np.log(np.maximum(np.asarray(values, dtype=np.float64), 1e-12))
    if x.size < 2:
        return float('nan')
    return float(np.polyfit(x, y, 1)[0])

def main() -> int:
    parser = argparse.ArgumentParser(description='E0.6/E0.7/E0.4 post-hoc analysis')
    parser.add_argument('--root', type=Path, default=OUTPUT_ROOT)
    parser.add_argument('--skip-ideal', action='store_true')
    args = parser.parse_args()
    aggregate = args.root / 'aggregate'
    out = args.root / 'aggregate_posthoc'
    out.mkdir(parents=True, exist_ok=True)

    ablation = read_rows(aggregate / 'all_ablation.csv')
    states = read_rows(aggregate / 'all_state_diagnostics.csv')
    displacement = read_rows(aggregate / 'all_error_displacement.csv')
    histogram = read_rows(aggregate / 'all_error_histogram.csv')

    acc = defaultdict(dict)
    for row in ablation:
        acc[(row['condition'], row['split'], row['mode'])][
            (int(row['source_seed']), int(row['target_seed']))] = float(row['accuracy'])
    targets = sorted({t for table in acc.values() for (_, t) in table})

    absolute = []
    for (condition, split, mode), table in sorted(acc.items()):
        values = list(table.values())
        absolute.append({'condition': condition, 'split': split, 'mode': mode,
                         'mean_accuracy': float(np.mean(values)),
                         'mean_normalised_drop': float(np.mean([
                             (acc[(condition, split, 'intact')][k] - v)
                             / max(acc[(condition, split, 'intact')][k] - CHANCE, 1e-9)
                             for k, v in table.items()]))})
    write_csv(out / 'absolute_ablation.csv', absolute)
    contrasts = []
    for left, right in PAIRS:
        for split in SPLITS:
            for mode in MODES:
                contrasts.append(paired('A1_absolute_accuracy_nav_minus_matched',
                                        source_mean(acc[(left, split, mode)]),
                                        source_mean(acc[(right, split, mode)]),
                                        pair=f'{left} vs {right}', split=split, mode=mode))

    ideal_rows = []
    if not args.skip_ideal:
        ideal = ideal_references(targets)
        for (target, split), values in sorted(ideal.items()):
            ideal_rows.append({'target_seed': target, 'split': split, **values})
        write_csv(out / 'ideal_integrator_reference.csv', ideal_rows)
        for condition in sorted({c for c, _, _ in acc}):
            for split in SPLITS:
                for k in (1, 2, 3):
                    table = acc.get((condition, split, f'last_{k}'), {})
                    excess = {key: value - ideal[(key[1], split)][f'ideal_last_{k}']
                              for key, value in table.items()}
                    per_source = source_mean(excess)
                    contrasts.append({'contrast': 'A2_excess_over_ideal_last_k', 'condition': condition,
                                      'split': split, 'mode': f'last_{k}',
                                      'ideal_mean': float(np.mean([ideal[(t, split)][f'ideal_last_{k}']
                                                                   for t in targets])),
                                      'mean_difference': float(np.mean(list(per_source.values()))),
                                      'n_sources': len(per_source)})

    series = defaultdict(lambda: defaultdict(dict))
    for row in states:
        key = (row['condition'], int(row['source_seed']), int(row['target_seed']))
        step = int(row['path_step'])
        for name in ('hidden_norm', 'delta_rectified', 'drive_ratio_linear', 'active_fraction'):
            series[key][name][step] = float(row[name])
    stationarity = []
    index = defaultdict(dict)
    for (condition, source, target), values in sorted(series.items()):
        steps = [t for t in range(3, 13) if t in values['hidden_norm']]
        retention = math.exp(log_slope(steps, [values['hidden_norm'][t] for t in steps]))
        growth = values['delta_rectified'].get(12, float('nan')) / max(values['delta_rectified'].get(3, float('nan')), 1e-12)
        record = {'condition': condition, 'source_seed': source, 'target_seed': target,
                  'norm_retention_per_step': retention, 'delta_growth_12_over_3': growth,
                  'delta_step1': values['delta_rectified'].get(1, float('nan')),
                  'delta_step12': values['delta_rectified'].get(12, float('nan')),
                  'drive_ratio_step12': values['drive_ratio_linear'].get(12, float('nan'))}
        stationarity.append(record)
        for metric in ('norm_retention_per_step', 'delta_growth_12_over_3', 'delta_step1', 'delta_step12'):
            index[(condition, metric)][(source, target)] = record[metric]
    write_csv(out / 'stationarity_by_cell.csv', stationarity)
    for left, right in PAIRS:
        for metric in ('norm_retention_per_step', 'delta_growth_12_over_3', 'delta_step1', 'delta_step12'):
            contrasts.append(paired('A3_stationarity_nav_minus_matched',
                                    source_mean(index[(left, metric)]), source_mean(index[(right, metric)]),
                                    pair=f'{left} vs {right}', metric=metric))
    association = []
    for condition in sorted({c for c, _ in index}):
        retention = source_mean(index[(condition, 'norm_retention_per_step')])
        long_acc = source_mean(acc.get((condition, 'seen_long', 'intact'), {}))
        common = sorted(set(retention) & set(long_acc))
        association.append({'condition': condition, 'n_sources': len(common),
                            'spearman_retention_vs_seen_long': spearman(
                                [retention[s] for s in common], [long_acc[s] for s in common]),
                            'exploratory': True})
    write_csv(out / 'stationarity_association.csv', association)

    growth_rows = []
    by_cell = defaultdict(list)
    for row in displacement:
        if row['split'] != 'seen_long':
            continue
        by_cell[(row['condition'], int(row['source_seed']), int(row['target_seed']))].append(row)
    exponents = defaultdict(dict)
    conditional = defaultdict(dict)
    for (condition, source, target), rows in by_cell.items():
        rows = sorted(rows, key=lambda r: int(r['path_length']))
        lengths = np.array([int(r['path_length']) for r in rows], dtype=np.float64)
        size = np.array([float(r['mean_abs_displacement']) for r in rows])
        wrong = np.array([1.0 - float(r['accuracy']) for r in rows])
        cond_size = np.where(wrong > 0, size / np.maximum(wrong, 1e-12), np.nan)
        ok = size > 0
        exponent = float(np.polyfit(np.log(lengths[ok]), np.log(size[ok]), 1)[0]) if ok.sum() >= 3 else float('nan')
        finite = np.isfinite(cond_size) & (cond_size > 0)
        cond_exp = (float(np.polyfit(np.log(lengths[finite]), np.log(cond_size[finite]), 1)[0])
                    if finite.sum() >= 3 else float('nan'))
        exponents[condition][(source, target)] = exponent
        conditional[condition][(source, target)] = cond_exp
        growth_rows.append({'condition': condition, 'source_seed': source, 'target_seed': target,
                            'size_exponent': exponent, 'conditional_size_exponent': cond_exp,
                            'conditional_size_L12': float(cond_size[-1]) if finite[-1] else float('nan')})
    write_csv(out / 'error_growth_by_cell.csv', growth_rows)
    for left, right in PAIRS:
        contrasts.append(paired('A4_size_exponent_nav_minus_matched', source_mean(exponents[left]),
                                source_mean(exponents[right]), pair=f'{left} vs {right}'))
        contrasts.append(paired('A4_conditional_size_exponent_nav_minus_matched',
                                source_mean(conditional[left]), source_mean(conditional[right]),
                                pair=f'{left} vs {right}'))

    structure = defaultdict(lambda: defaultdict(float))
    for row in histogram:
        if row['split'] != 'seen_long':
            continue
        dr, dc, count = int(row['row_shift']), int(row['col_shift']), float(row['count'])
        if dr == 0 and dc == 0:
            continue
        key = (row['condition'], int(row['source_seed']))
        record = structure[key]
        record['n'] += count
        record['adjacent'] += count * float(abs(dr) + abs(dc) == 1)
        record['l1'] += count * (abs(dr) + abs(dc))
        record['row2'] += count * dr * dr
        record['col2'] += count * dc * dc
        record['row'] += count * dr
        record['col'] += count * dc
    error_rows = []
    per_condition = defaultdict(list)
    for (condition, source), r in sorted(structure.items()):
        n = max(r['n'], 1.0)
        row = {'condition': condition, 'source_seed': source, 'n_errors': r['n'],
               'share_adjacent': r['adjacent'] / n, 'mean_l1_error': r['l1'] / n,
               'mean_row_error': r['row'] / n, 'mean_col_error': r['col'] / n,
               'row_col_variance_ratio': (r['row2'] / n) / max(r['col2'] / n, 1e-12)}
        error_rows.append(row)
        per_condition[condition].append(row)
    write_csv(out / 'error_structure_by_source.csv', error_rows)
    write_csv(out / 'posthoc_contrasts.csv', contrasts)

    summary = []
    for condition, rows in sorted(per_condition.items()):
        summary.append({'condition': condition,
                        **{k: float(np.mean([r[k] for r in rows]))
                           for k in ('share_adjacent', 'mean_l1_error', 'mean_row_error',
                                     'mean_col_error', 'row_col_variance_ratio')}})
    (out / 'FORWARD_POSTHOC.json').write_text(json.dumps(
        {'posthoc': True, 'contrasts': contrasts, 'association': association,
         'error_structure': summary, 'ideal_reference': ideal_rows}, indent=2), encoding='utf-8')

    lines = ['# E0.6 / E0.7 / E0.4 post-hoc analysis (written after seeing the results)\n',
             '## A1 absolute accuracy, navigation minus matched (scalar-gain pair)\n',
             '| split | mode | nav | matched | difference | exact p |', '|---|---|---|---|---|---|']
    for r in contrasts:
        if r['contrast'] == 'A1_absolute_accuracy_nav_minus_matched' and r['pair'].startswith('navigation_trainable'):
            lines.append(f"| {r['split']} | {r['mode']} | {r['left_mean']:.4f} | {r['right_mean']:.4f} | "
                         f"{r['mean_difference']:+.4f} | {r['exact_p']:.6f} |")
    if ideal_rows:
        lines += ['', '## A2 ideal-integrator accuracy under last_k and excess over it (mean over targets)\n',
                  '| condition | split | mode | ideal | excess |', '|---|---|---|---|---|']
        for r in contrasts:
            if r['contrast'] == 'A2_excess_over_ideal_last_k' and 'scalar' in r['condition'] and r['split'] in ('id_seen_short', 'seen_long'):
                lines.append(f"| {r['condition']} | {r['split']} | {r['mode']} | {r['ideal_mean']:.4f} | {r['mean_difference']:+.4f} |")
        lines += ['', '| split | shuffled order leaves grid | shuffled order creates held-out bigram |', '|---|---|---|']
        for split in SPLITS:
            rows = [r for r in ideal_rows if r['split'] == split]
            lines.append(f"| {split} | {np.mean([r['shuffle_leaves_grid'] for r in rows]):.3f} | "
                         f"{np.mean([r['shuffle_has_heldout_bigram'] for r in rows]):.3f} |")
    lines += ['', '## A3 stationarity, navigation minus matched\n',
              '| pair | metric | nav | matched | difference | 95% CI | exact p |', '|---|---|---|---|---|---|---|']
    for r in contrasts:
        if r['contrast'] == 'A3_stationarity_nav_minus_matched':
            lines.append(f"| {r['pair'].split(' vs ')[0]} | {r['metric']} | {r['left_mean']:.4f} | {r['right_mean']:.4f} | "
                         f"{r['mean_difference']:+.4f} | [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] | {r['exact_p']:.6f} |")
    lines += ['', 'Exploratory within-origin Spearman, norm retention against intact seen_long accuracy:\n',
              '| condition | n | rho |', '|---|---|---|']
    for r in association:
        lines.append(f"| {r['condition']} | {r['n_sources']} | {r['spearman_retention_vs_seen_long']:+.3f} |")
    lines += ['', '## A4 error structure on seen_long\n',
              '| pair | readout | nav | matched | difference | exact p |', '|---|---|---|---|---|---|']
    for r in contrasts:
        if r['contrast'].startswith('A4_'):
            lines.append(f"| {r['pair'].split(' vs ')[0]} | {r['contrast'][3:]} | {r['left_mean']:.3f} | "
                         f"{r['right_mean']:.3f} | {r['mean_difference']:+.3f} | {r['exact_p']:.6f} |")
    lines += ['', '| condition | share of errors on an adjacent node | mean L1 error | mean row error | mean col error | row/col variance |',
              '|---|---|---|---|---|---|']
    for r in summary:
        lines.append(f"| {r['condition']} | {r['share_adjacent']:.3f} | {r['mean_l1_error']:.3f} | "
                     f"{r['mean_row_error']:+.3f} | {r['mean_col_error']:+.3f} | {r['row_col_variance_ratio']:.3f} |")
    (out / 'FORWARD_POSTHOC.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))
    return 0

if __name__ == '__main__':
    sys.exit(main())
