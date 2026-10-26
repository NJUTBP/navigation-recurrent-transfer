#!/usr/bin/env python3
"""E0.6 / E0.7 / E0.4 collector.

Gate R comes first.  The intact re-evaluation must reproduce the accuracy the
original run stored; if it does not, the frozen core was rebuilt wrongly and no
number below means anything.

E0.6  Recurrence and history use
    * chance-normalised drop (acc - acc_ablated) / (acc - 1/64) for every mode;
      the raw drop is bounded by accuracy minus chance and therefore mostly
      tracks accuracy, which is the same trap the 4x4 figure falls into.
    * ``shuffle_order`` measures order invariance.  The endpoint is the sum of
      the relations, so a correct additive solution is invariant to order; a
      large drop means the model is not solving it additively.

E0.7  Long-horizon error displacement
    * ``E[e]`` (a signed vector) against path length distinguishes systematic
      drift from isotropic diffusion.  Declared test: the slope of
      ``E||e||`` against L, navigation versus matched, exact sign-flip across
      sources.

E0.4  State diagnostics and delta
    * per-step hidden norm, active fraction, signal against the zero-input run,
      linear drive ratio, rectified relative displacement.
    * ``delta_by_gain`` is the E5b' input: delta(g) measured on real start
      states.  The prediction to test once E5c returns is that the ratio of
      required operating gains equals the inverse ratio of delta.

Usage
-----
    "$PYTHON_BIN" collect_forward_diagnostics.py
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

from forward_common import (
    ABLATION_MODES, CHANCE, DELTA_GAIN_GRID, GATE_R_TOLERANCE, OUTPUT_ROOT,
    SCHEMA_VERSION, TEST_SPLITS,
)

STATISTICS_SEED = 982451653
BOOTSTRAP_SAMPLES = 20000
PRIMARY_LEFT = 'navigation_trainable_scalar_gain'
PRIMARY_RIGHT = 'matched_random_trainable_scalar_gain'


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


def paired(name: str, left: dict[int, float], right: dict[int, float],
           expectation: str) -> dict[str, Any]:
    sources = sorted(set(left) & set(right))
    values = np.array([left[s] - right[s] for s in sources], dtype=np.float64)
    low, high = bootstrap_ci(values)
    return {'contrast': name, 'n_sources': len(sources),
            'left_mean': float(np.mean([left[s] for s in sources])) if sources else float('nan'),
            'right_mean': float(np.mean([right[s] for s in sources])) if sources else float('nan'),
            'mean_difference': float(values.mean()) if values.size else float('nan'),
            'ci_low': low, 'ci_high': high,
            'exact_p': exact_sign_flip_p(values),
            'n_positive': int(np.sum(values > 0)), 'expectation': expectation}


def source_mean(rows, key_fields: tuple[str, ...], key_values: tuple,
                value_field: str) -> dict[int, float]:
    bucket: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        if all(str(row.get(f)) == str(v) for f, v in zip(key_fields, key_values)):
            bucket[int(row['source_seed'])].append(as_float(row[value_field]))
    return {s: float(np.mean(v)) for s, v in bucket.items() if v}


def main() -> int:
    parser = argparse.ArgumentParser(description='E0.6/E0.7/E0.4 collector')
    parser.add_argument('--root', type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()

    root: Path = args.root
    out = root / 'aggregate'
    out.mkdir(parents=True, exist_ok=True)

    cells, missing = [], []
    for source_dir in sorted(root.glob('source_*')):
        for target_dir in sorted(source_dir.glob('target_*')):
            (cells if (target_dir / 'COMPLETE.json').is_file() else missing).append(target_dir)
    if missing:
        print(f'WARNING: {len(missing)} incomplete cells, first few: '
              f'{[str(c) for c in missing[:5]]}')
    if not cells:
        raise SystemExit('no completed cells found')

    ablation, gate, displacement, histogram, state, delta, diagnostics = ([] for _ in range(7))
    for cell in cells:
        ablation += read_csv(cell / 'ablation.csv')
        gate += read_csv(cell / 'gate_r.csv')
        displacement += read_csv(cell / 'error_displacement.csv')
        histogram += read_csv(cell / 'error_histogram.csv')
        state += read_csv(cell / 'state_diagnostics.csv')
        delta += read_csv(cell / 'delta_by_gain.csv')
        diagnostics += read_csv(cell / 'condition_diagnostics.csv')

    for name, rows in (('all_ablation.csv', ablation), ('all_gate_r.csv', gate),
                       ('all_error_displacement.csv', displacement),
                       ('all_error_histogram.csv', histogram),
                       ('all_state_diagnostics.csv', state),
                       ('all_delta_by_gain.csv', delta),
                       ('all_condition_diagnostics.csv', diagnostics)):
        if rows:
            write_csv(out / name, rows, list(rows[0].keys()))

    # ---- Gate R ---------------------------------------------------------
    differences = np.array([as_float(r['abs_difference']) for r in gate], dtype=np.float64)
    differences = differences[np.isfinite(differences)]
    failures = [r for r in gate if str(r.get('within_tolerance')).lower() != 'true']
    gate_r = {'n_comparisons': len(gate), 'n_failures': len(failures),
              'tolerance': GATE_R_TOLERANCE,
              'mean_abs_difference': float(differences.mean()) if differences.size else float('nan'),
              'max_abs_difference': float(differences.max()) if differences.size else float('nan'),
              'note': 'the intact re-evaluation must reproduce the stored accuracy; '
                      'a failure means the frozen core was rebuilt wrongly'}

    # ---- E0.6 -----------------------------------------------------------
    conditions = sorted({r['condition'] for r in ablation})
    normalized_rows = []
    for condition in conditions:
        for split in TEST_SPLITS:
            intact = source_mean(ablation, ('condition', 'split', 'mode'),
                                 (condition, split, 'intact'), 'accuracy')
            for mode in ABLATION_MODES:
                if mode == 'intact':
                    continue
                ablated = source_mean(ablation, ('condition', 'split', 'mode'),
                                      (condition, split, mode), 'accuracy')
                shared = sorted(set(intact) & set(ablated))
                if not shared:
                    continue
                raw = np.array([intact[s] - ablated[s] for s in shared])
                headroom = np.array([max(intact[s] - CHANCE, 1e-9) for s in shared])
                normalized_rows.append({
                    'condition': condition, 'split': split, 'mode': mode,
                    'n_sources': len(shared),
                    'intact_accuracy': float(np.mean([intact[s] for s in shared])),
                    'ablated_accuracy': float(np.mean([ablated[s] for s in shared])),
                    'raw_drop': float(raw.mean()),
                    'normalized_drop': float((raw / headroom).mean())})
    if normalized_rows:
        write_csv(out / 'ablation_normalized.csv', normalized_rows,
                  ['condition', 'split', 'mode', 'n_sources', 'intact_accuracy',
                   'ablated_accuracy', 'raw_drop', 'normalized_drop'])

    contrasts: list[dict[str, Any]] = []
    for split in ('id_seen_short', 'seen_long'):
        for mode in ('no_recurrence', 'shuffle_order'):
            left_i = source_mean(ablation, ('condition', 'split', 'mode'),
                                 (PRIMARY_LEFT, split, 'intact'), 'accuracy')
            left_a = source_mean(ablation, ('condition', 'split', 'mode'),
                                 (PRIMARY_LEFT, split, mode), 'accuracy')
            right_i = source_mean(ablation, ('condition', 'split', 'mode'),
                                  (PRIMARY_RIGHT, split, 'intact'), 'accuracy')
            right_a = source_mean(ablation, ('condition', 'split', 'mode'),
                                  (PRIMARY_RIGHT, split, mode), 'accuracy')
            left = {s: (left_i[s] - left_a[s]) / max(left_i[s] - CHANCE, 1e-9)
                    for s in set(left_i) & set(left_a)}
            right = {s: (right_i[s] - right_a[s]) / max(right_i[s] - CHANCE, 1e-9)
                     for s in set(right_i) & set(right_a)}
            if left and right:
                contrasts.append(paired(
                    f'E06_normalized_{mode}_drop_{split}_nav_minus_matched', left, right,
                    'near zero would mean both cores depend on this equally'))

    # ---- E0.7 -----------------------------------------------------------
    def displacement_slope(condition: str, field: str) -> dict[int, float]:
        bucket: dict[int, list[tuple[float, float]]] = defaultdict(list)
        for row in displacement:
            if row['condition'] != condition or row['split'] != 'seen_long':
                continue
            bucket[int(row['source_seed'])].append(
                (as_float(row['path_length']), as_float(row[field])))
        out_map = {}
        for source, points in bucket.items():
            points = [(x, y) for x, y in points if math.isfinite(x) and math.isfinite(y)]
            if len(points) >= 3:
                x = np.array([p[0] for p in points]); y = np.array([p[1] for p in points])
                out_map[source] = float(np.polyfit(x, y, 1)[0])
        return out_map

    for field, label in (('mean_abs_displacement', 'magnitude'),
                         ('mean_displacement_row', 'signed_row'),
                         ('mean_displacement_col', 'signed_col')):
        left = displacement_slope(PRIMARY_LEFT, field)
        right = displacement_slope(PRIMARY_RIGHT, field)
        if left and right:
            contrasts.append(paired(
                f'E07_slope_{label}_vs_length_nav_minus_matched', left, right,
                'negative for magnitude means navigation drifts less per extra step'))

    if contrasts:
        write_csv(out / 'forward_contrasts.csv', contrasts,
                  ['contrast', 'n_sources', 'left_mean', 'right_mean', 'mean_difference',
                   'ci_low', 'ci_high', 'exact_p', 'n_positive', 'expectation'])

    # ---- E0.4 -----------------------------------------------------------
    state_summary = []
    fields = ('hidden_norm', 'active_fraction', 'signal_vs_zero_input',
              'delta_rectified', 'drive_ratio_linear')
    for condition in conditions:
        by_step: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for row in state:
            if row['condition'] != condition:
                continue
            step = int(as_float(row['path_step']))
            for field in fields:
                by_step[step][field].append(as_float(row.get(field)))
        for step in sorted(by_step):
            record = {'condition': condition, 'path_step': step}
            for field in fields:
                values = [v for v in by_step[step][field] if math.isfinite(v)]
                record[field] = float(np.mean(values)) if values else float('nan')
            state_summary.append(record)
    if state_summary:
        write_csv(out / 'state_summary.csv', state_summary,
                  ['condition', 'path_step', *fields])

    delta_summary = []
    for condition in conditions:
        for gain in DELTA_GAIN_GRID:
            values = [as_float(r['delta_mean']) for r in delta
                      if r['condition'] == condition and abs(as_float(r['gain']) - gain) < 1e-9]
            values = [v for v in values if math.isfinite(v)]
            if values:
                delta_summary.append({'condition': condition, 'gain': gain,
                                      'n_cells': len(values),
                                      'delta_mean': float(np.mean(values)),
                                      'delta_sd': float(np.std(values, ddof=1))
                                      if len(values) > 1 else float('nan')})
    if delta_summary:
        write_csv(out / 'delta_summary.csv', delta_summary,
                  ['condition', 'gain', 'n_cells', 'delta_mean', 'delta_sd'])

    delta_ratio = None
    for gain in DELTA_GAIN_GRID:
        left = source_mean([r for r in delta if abs(as_float(r['gain']) - gain) < 1e-9],
                           ('condition',), (PRIMARY_LEFT,), 'delta_mean')
        right = source_mean([r for r in delta if abs(as_float(r['gain']) - gain) < 1e-9],
                            ('condition',), (PRIMARY_RIGHT,), 'delta_mean')
        shared = sorted(set(left) & set(right))
        if shared and delta_ratio is None:
            delta_ratio = {
                'gain': gain, 'n_sources': len(shared),
                'delta_nav': float(np.mean([left[s] for s in shared])),
                'delta_matched': float(np.mean([right[s] for s in shared])),
                'predicted_gain_ratio_matched_over_nav': float(
                    np.mean([left[s] / max(right[s], 1e-12) for s in shared])),
                'note': 'E5b prediction: this should match the observed ratio of required '
                        'operating gains. Observed learned-gain ratios are 1.46 at 8000 '
                        'updates and about 1.26 at 32000; E5c will give the fixed-gain version.'}

    result = {'schema_version': SCHEMA_VERSION,
              'n_cells_complete': len(cells), 'n_cells_missing': len(missing),
              'gate_r': gate_r, 'contrasts': contrasts,
              'delta_ratio_prediction': delta_ratio}
    (out / 'FORWARD_RESULT.json').write_text(json.dumps(result, indent=2), encoding='utf-8')

    lines = ['# E0.6 / E0.7 / E0.4 forward diagnostics\n',
             f'- completed cells: {len(cells)} (missing {len(missing)})\n',
             '## Gate R: does the rebuilt frozen core reproduce the stored accuracy?\n',
             f'- comparisons: {gate_r["n_comparisons"]}, failures: {gate_r["n_failures"]} '
             f'(tolerance {GATE_R_TOLERANCE})',
             f'- mean absolute difference {gate_r["mean_abs_difference"]:.6f}, '
             f'max {gate_r["max_abs_difference"]:.6f}',
             '- A failure means the core was rebuilt wrongly. Read this before anything else.\n',
             '## E0.6 chance-normalised ablation drop\n',
             '| condition | split | ' + ' | '.join(
                 m for m in ABLATION_MODES if m != 'intact') + ' |',
             '|---|---|' + '---|' * (len(ABLATION_MODES) - 1)]
    table = {(r['condition'], r['split'], r['mode']): r['normalized_drop']
             for r in normalized_rows}
    for condition in conditions:
        for split in ('id_seen_short', 'seen_long'):
            cells_text = [f'{table[(condition, split, m)]:.3f}'
                          if (condition, split, m) in table else '-'
                          for m in ABLATION_MODES if m != 'intact']
            lines.append(f'| {condition} | {split} | ' + ' | '.join(cells_text) + ' |')
    lines += ['', 'Normalised by accuracy minus chance, so the column is comparable across '
                  'conditions with very different accuracy. shuffle_order measures order '
                  'invariance: the endpoint is the sum of the relations, so an additive '
                  'solution should be unaffected.\n']
    if contrasts:
        lines += ['## Declared contrasts\n',
                  '| contrast | n | nav | matched | difference | 95% CI | exact p |',
                  '|---|---|---|---|---|---|---|']
        for record in contrasts:
            lines.append(
                f'| {record["contrast"]} | {record["n_sources"]} | '
                f'{record["left_mean"]:.4g} | {record["right_mean"]:.4g} | '
                f'{record["mean_difference"]:+.4g} | '
                f'[{record["ci_low"]:+.4g}, {record["ci_high"]:+.4g}] | '
                f'{record["exact_p"]:.6f} |')
        lines.append('')
    if delta_summary:
        gains = sorted({r['gain'] for r in delta_summary})
        lines += ['## E0.4 rectified relative displacement delta(g)\n',
                  '| condition | ' + ' | '.join(f'g={g:g}' for g in gains) + ' |',
                  '|---|' + '---|' * len(gains)]
        lookup = {(r['condition'], r['gain']): r['delta_mean'] for r in delta_summary}
        for condition in conditions:
            lines.append(f'| {condition} | ' + ' | '.join(
                f'{lookup[(condition, g)]:.4f}' if (condition, g) in lookup else '-'
                for g in gains) + ' |')
        lines.append('')
    if delta_ratio:
        lines += [f'- At g = {delta_ratio["gain"]:g}: delta(nav) = '
                  f'{delta_ratio["delta_nav"]:.4f}, delta(matched) = '
                  f'{delta_ratio["delta_matched"]:.4f}, so the predicted ratio of required '
                  f'operating gains is {delta_ratio["predicted_gain_ratio_matched_over_nav"]:.3f}.',
                  '- Observed learned-gain ratios: 1.46 at 8,000 updates, about 1.26 at 32,000. '
                  'E5c will give the version that does not depend on gain-learning dynamics.\n']
    (out / 'FORWARD_RESULT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
