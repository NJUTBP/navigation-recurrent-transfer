#!/usr/bin/env python3
"""E0.9a collector: reproduction check plus the three pre-registered decisions.

Runs after every group cell reports ``COMPLETE.json``.

Gate 0 (integrity, not a scientific result)
    The first 8,000 updates of this run must re-execute the Phase F trajectory,
    because the dataset, the model initialisation and the minibatch order are
    seed-identical.  The collector compares the two validation curves on the
    shared evaluation grid and reports the maximum and mean absolute deviation.
    Large deviation means a configuration fault, not a finding; read it before
    reading anything else.

D1  Primary.  Does the navigation-minus-matched advantage on ``id_seen_short``
    survive at 32,000 updates?  Survives -> the published contrast is an
    asymptotic capability claim.  Vanishes -> it is a sample/compute efficiency
    claim and the manuscript must say so.

D2  Watershed.  Does ``navigation_identity_frozen`` at gain 1, trained to 32,000
    updates, reach the Phase F 8,000-update accuracy of
    ``navigation_fixed_gain10_frozen``?  Reaches it -> the interface gain governs
    learning RATE only.  Does not -> the gain also governs the attainable
    solution, and H5'' is a representational-resolvability hypothesis rather than
    an optimisation-conditioning one.  These lead to different Discussions.

D3  Gain trajectory.  Have the learned gains plateaued by 32,000 updates, and is
    the navigation gain still below the matched gain?  If both have plateaued and
    the ordering holds, the budget caveat attached to the Phase F gain comparison
    can be dropped.

Usage
-----
    "$PYTHON_BIN" collect_extended.py \
        --phasef-followup .../aggregate_full/hard8x8_phaseF_followup
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

from extended_common import (
    GROUPS, OUTPUT_ROOT, PHASEF_OUTPUT, SCHEMA_VERSION, TARGET_SEEDS,
)

STATISTICS_SEED = 982451653
BOOTSTRAP_SAMPLES = 20000
TEST_SPLITS = ('id_seen_short', 'heldout_short', 'seen_long', 'heldout_long')
PHASEF_BUDGET = 8000
WATERSHED_REFERENCE = 'navigation_fixed_gain10_frozen'


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
    count = sum(
        1 for signs in itertools.product((1.0, -1.0), repeat=n)
        if abs(float(np.mean(np.asarray(signs) * values))) >= observed - 1e-12
    )
    return count / float(2 ** n)


def bootstrap_ci(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float('nan'), float('nan')
    rng = np.random.default_rng(STATISTICS_SEED)
    index = rng.integers(0, values.size, size=(BOOTSTRAP_SAMPLES, values.size))
    return (float(np.percentile(values[index].mean(axis=1), 2.5)),
            float(np.percentile(values[index].mean(axis=1), 97.5)))


def holm(p_values: list[float]) -> list[float]:
    finite = [(i, p) for i, p in enumerate(p_values) if math.isfinite(p)]
    adjusted = [float('nan')] * len(p_values)
    running = 0.0
    for rank, (index, p) in enumerate(sorted(finite, key=lambda item: item[1])):
        running = max(running, min(1.0, (len(finite) - rank) * p))
        adjusted[index] = running
    return adjusted


def source_mean(rows: list[dict[str, Any]], condition: str, split: str) -> dict[int, float]:
    bucket: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        if row.get('condition') == condition and row.get('split') == split:
            bucket[int(row['source_seed'])].append(as_float(row['accuracy']))
    return {source: float(np.mean(values)) for source, values in bucket.items() if values}


def paired(name: str, family: str, left: dict[int, float], right: dict[int, float],
           expectation: str) -> dict[str, Any]:
    sources = sorted(set(left) & set(right))
    values = np.array([left[s] - right[s] for s in sources], dtype=np.float64)
    low, high = bootstrap_ci(values)
    return {
        'contrast': name, 'family': family, 'n_sources': len(sources),
        'left_mean': float(np.mean([left[s] for s in sources])) if sources else float('nan'),
        'right_mean': float(np.mean([right[s] for s in sources])) if sources else float('nan'),
        'mean_difference': float(values.mean()) if values.size else float('nan'),
        'ci_low': low, 'ci_high': high,
        'exact_p': exact_sign_flip_p(values),
        'n_positive': int(np.sum(values > 0)),
        'expectation': expectation,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description='E0.9a collector')
    parser.add_argument('--root', type=Path, default=OUTPUT_ROOT)
    parser.add_argument('--phasef-root', type=Path, default=PHASEF_OUTPUT)
    parser.add_argument('--phasef-followup', type=Path, required=True,
                        help='aggregate_full/hard8x8_phaseF_followup')
    args = parser.parse_args()

    root: Path = args.root
    out = root / 'aggregate'
    out.mkdir(parents=True, exist_ok=True)

    cells, missing = [], []
    for source_dir in sorted(root.glob('source_*')):
        for target_dir in sorted(source_dir.glob('target_*')):
            for group in sorted(GROUPS):
                cell = target_dir / f'group_{group}'
                (cells if (cell / 'COMPLETE.json').is_file() else missing).append(cell)
    if missing:
        print(f'WARNING: {len(missing)} incomplete cells, first few: '
              f'{[str(c) for c in missing[:5]]}')

    metrics, per_length, curves, diagnostics, gains = [], [], [], [], []
    for cell in cells:
        metrics += read_csv(cell / 'metrics.csv')
        per_length += read_csv(cell / 'per_length.csv')
        curves += read_csv(cell / 'update_curves.csv')
        diagnostics += read_csv(cell / 'condition_diagnostics.csv')
        gains += read_csv(cell / 'gain_trace.csv')

    if not metrics:
        raise SystemExit('no completed cells found')

    for name, rows in (('all_metrics.csv', metrics), ('all_per_length.csv', per_length),
                       ('all_update_curves.csv', curves),
                       ('all_condition_diagnostics.csv', diagnostics),
                       ('all_gain_trace.csv', gains)):
        if rows:
            write_csv(out / name, rows, list(rows[0].keys()))

    # ---------------- Gate 0: reproduction of the first 8000 updates ------
    phasef_curves = read_csv(args.phasef_followup / 'all_update_curves.csv')
    phasef_lookup: dict[tuple[int, int, str, int], float] = {}
    for row in phasef_curves:
        try:
            key = (int(row['source_seed']), int(row['target_seed']),
                   row['condition'], int(row['update']))
        except (KeyError, ValueError):
            continue
        phasef_lookup[key] = as_float(row['validation_accuracy'])

    deviations: list[dict[str, Any]] = []
    grouped: dict[tuple[int, int, str], list[float]] = defaultdict(list)
    for row in curves:
        update = int(as_float(row.get('update')))
        if update <= 0 or update > PHASEF_BUDGET:
            continue
        key = (int(row['source_seed']), int(row['target_seed']), row['condition'])
        reference = phasef_lookup.get((*key, update))
        if reference is None or not math.isfinite(reference):
            continue
        grouped[key].append(abs(as_float(row['validation_accuracy']) - reference))
    for key, values in sorted(grouped.items()):
        array = np.asarray(values, dtype=np.float64)
        deviations.append({
            'source_seed': key[0], 'target_seed': key[1], 'condition': key[2],
            'n_matched_evals': int(array.size),
            'mean_abs_deviation': float(array.mean()),
            'max_abs_deviation': float(array.max()),
        })
    if deviations:
        write_csv(out / 'reproduction_check.csv', deviations,
                  ['source_seed', 'target_seed', 'condition', 'n_matched_evals',
                   'mean_abs_deviation', 'max_abs_deviation'])
    gate0 = {
        'n_cells_compared': len(deviations),
        'mean_abs_deviation': float(np.mean([d['mean_abs_deviation'] for d in deviations]))
        if deviations else float('nan'),
        'worst_max_abs_deviation': float(np.max([d['max_abs_deviation'] for d in deviations]))
        if deviations else float('nan'),
        'note': 'the first 8000 updates should re-execute the Phase F trajectory; '
                'GPU nondeterminism allows small drift, a large deviation is a '
                'configuration fault rather than a result',
    }

    # ---------------- D1: primary contrast at the extended budget ---------
    contrasts: list[dict[str, Any]] = []
    for split in TEST_SPLITS:
        contrasts.append(paired(
            f'nav_minus_matched_scalar_{split}', 'D1_primary'
            if split == 'id_seen_short' else 'D1_generalization',
            source_mean(metrics, 'navigation_trainable_scalar_gain', split),
            source_mean(metrics, 'matched_random_trainable_scalar_gain', split),
            'positive if the Phase F advantage is asymptotic rather than a rate effect',
        ))
    family = [c for c in contrasts if c['family'] == 'D1_generalization']
    for record, adjusted in zip(family, holm([c['exact_p'] for c in family])):
        record['holm_p'] = adjusted
    contrasts[0]['holm_p'] = contrasts[0]['exact_p']

    phasef_metrics = read_csv(args.phasef_followup / 'source_level_metrics.csv')
    phasef_source: dict[tuple[str, str], dict[int, float]] = defaultdict(dict)
    for row in phasef_metrics:
        phasef_source[(row['condition'], row['split'])][int(row['source_seed'])] = \
            as_float(row['accuracy'])

    comparison: list[dict[str, Any]] = []
    for split in TEST_SPLITS:
        extended = paired('x', 'x',
                          source_mean(metrics, 'navigation_trainable_scalar_gain', split),
                          source_mean(metrics, 'matched_random_trainable_scalar_gain', split), '')
        left = phasef_source[('navigation_trainable_scalar_gain', split)]
        right = phasef_source[('matched_random_trainable_scalar_gain', split)]
        shared = sorted(set(left) & set(right))
        phasef_delta = float(np.mean([left[s] - right[s] for s in shared])) if shared else float('nan')
        comparison.append({
            'split': split,
            'phasef_8000_difference': phasef_delta,
            'extended_difference': extended['mean_difference'],
            'change': extended['mean_difference'] - phasef_delta,
        })
    write_csv(out / 'phasef_vs_extended.csv', comparison,
              ['split', 'phasef_8000_difference', 'extended_difference', 'change'])

    # ---------------- D2: watershed --------------------------------------
    identity_extended = source_mean(metrics, 'navigation_identity_frozen', 'id_seen_short')
    reference = phasef_source[(WATERSHED_REFERENCE, 'id_seen_short')]
    watershed_rows = []
    for source in sorted(set(identity_extended) & set(reference)):
        watershed_rows.append({
            'source_seed': source,
            'identity_gain1_at_extended_budget': identity_extended[source],
            'phasef_gain10_at_8000': reference[source],
            'difference': identity_extended[source] - reference[source],
            'reached': bool(identity_extended[source] >= reference[source]),
        })
    if watershed_rows:
        write_csv(out / 'watershed_identity_vs_gain10.csv', watershed_rows,
                  ['source_seed', 'identity_gain1_at_extended_budget',
                   'phasef_gain10_at_8000', 'difference', 'reached'])
        differences = np.array([r['difference'] for r in watershed_rows], dtype=np.float64)
        low, high = bootstrap_ci(differences)
        watershed = {
            'n_sources': len(watershed_rows),
            'n_reached': int(sum(r['reached'] for r in watershed_rows)),
            'mean_difference': float(differences.mean()),
            'ci_low': low, 'ci_high': high,
            'exact_p': exact_sign_flip_p(differences),
            'verdict': ('gain governs RATE only' if float(differences.mean()) >= 0
                        else 'gain also governs the attainable solution'),
        }
    else:
        watershed = {'n_sources': 0, 'verdict': 'not evaluable'}

    # ---------------- D3: gain trajectory --------------------------------
    gain_summary: list[dict[str, Any]] = []
    traces: dict[tuple[int, int, str], list[tuple[int, float]]] = defaultdict(list)
    for row in gains:
        traces[(int(row['source_seed']), int(row['target_seed']), row['condition'])].append(
            (int(as_float(row['update'])), as_float(row['gain'])))
    plateau: dict[str, list[float]] = defaultdict(list)
    final: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for (source, _, condition), points in traces.items():
        points.sort()
        if len(points) < 4:
            continue
        last_update = points[-1][0]
        window = [g for u, g in points if u >= 0.9 * last_update]
        head = [g for u, g in points if 0.8 * last_update <= u < 0.9 * last_update]
        if window and head:
            relative = (float(np.mean(window)) - float(np.mean(head))) / max(float(np.mean(head)), 1e-9)
            plateau[condition].append(relative)
        final[condition][source].append(points[-1][1])
    for condition in sorted(set(list(plateau) + list(final))):
        per_source = {s: float(np.mean(v)) for s, v in final[condition].items()}
        gain_summary.append({
            'condition': condition,
            'n_sources': len(per_source),
            'final_gain_mean': float(np.mean(list(per_source.values()))) if per_source else float('nan'),
            'final_gain_sd': float(np.std(list(per_source.values()), ddof=1)) if len(per_source) > 1 else float('nan'),
            'relative_growth_last_10pct': float(np.mean(plateau[condition])) if plateau[condition] else float('nan'),
        })
    if gain_summary:
        write_csv(out / 'gain_summary.csv', gain_summary,
                  ['condition', 'n_sources', 'final_gain_mean', 'final_gain_sd',
                   'relative_growth_last_10pct'])

    gain_contrast = None
    nav_final = {s: float(np.mean(v)) for s, v in final['navigation_trainable_scalar_gain'].items()}
    matched_final = {s: float(np.mean(v)) for s, v in final['matched_random_trainable_scalar_gain'].items()}
    if nav_final and matched_final:
        gain_contrast = paired('final_gain_nav_minus_matched', 'D3_gain',
                               nav_final, matched_final,
                               'negative if navigation settles at a smaller operating gain')

    write_csv(out / 'extended_contrasts.csv',
              contrasts + ([gain_contrast] if gain_contrast else []),
              ['contrast', 'family', 'n_sources', 'left_mean', 'right_mean',
               'mean_difference', 'ci_low', 'ci_high', 'exact_p', 'holm_p',
               'n_positive', 'expectation'])

    result = {
        'schema_version': SCHEMA_VERSION,
        'n_cells_complete': len(cells), 'n_cells_missing': len(missing),
        'gate0_reproduction': gate0,
        'D1_contrasts': contrasts,
        'phasef_vs_extended': comparison,
        'D2_watershed': watershed,
        'D3_gain_summary': gain_summary,
        'D3_gain_contrast': gain_contrast,
    }
    (out / 'EXTENDED_RESULT.json').write_text(json.dumps(result, indent=2), encoding='utf-8')

    lines = ['# E0.9a extended-budget result\n',
             f'- completed cells: {len(cells)} (missing {len(missing)})',
             f'- extended budget: {int(as_float(metrics[0].get("max_gradient_updates")))} updates\n',
             '## Gate 0: reproduction of the first 8,000 updates\n',
             f'- cells compared: {gate0["n_cells_compared"]}',
             f'- mean absolute deviation: {gate0["mean_abs_deviation"]:.5f}',
             f'- worst maximum deviation: {gate0["worst_max_abs_deviation"]:.5f}',
             '- Large deviation is a configuration fault, not a finding. Read this first.\n',
             '## D1: primary and generalization contrasts at the extended budget\n',
             '| split | n | nav | matched | difference | 95% CI | exact p | Holm |',
             '|---|---|---|---|---|---|---|---|']
    for record in contrasts:
        split = record['contrast'].rsplit('scalar_', 1)[-1]
        lines.append(
            f'| {split} | {record["n_sources"]} | {record["left_mean"]:.4f} | '
            f'{record["right_mean"]:.4f} | {record["mean_difference"]:+.4f} | '
            f'[{record["ci_low"]:+.4f}, {record["ci_high"]:+.4f}] | '
            f'{record["exact_p"]:.6f} | {record.get("holm_p", float("nan")):.6f} |')
    lines += ['', '## Phase F (8,000) versus extended\n',
              '| split | Phase F difference | extended difference | change |',
              '|---|---|---|---|']
    for record in comparison:
        lines.append(f'| {record["split"]} | {record["phasef_8000_difference"]:+.4f} | '
                     f'{record["extended_difference"]:+.4f} | {record["change"]:+.4f} |')
    lines += ['', '## D2: watershed, gain 1 at the extended budget versus gain 10 at 8,000\n']
    if watershed.get('n_sources'):
        lines += [f'- sources reaching the gain-10 reference: '
                  f'{watershed["n_reached"]}/{watershed["n_sources"]}',
                  f'- mean difference {watershed["mean_difference"]:+.4f}, '
                  f'95% CI [{watershed["ci_low"]:+.4f}, {watershed["ci_high"]:+.4f}], '
                  f'exact p {watershed["exact_p"]:.6f}',
                  f'- verdict: **{watershed["verdict"]}**\n']
    else:
        lines.append('- not evaluable\n')
    lines += ['## D3: learned gain trajectory\n',
              '| condition | n | final gain | sd | relative growth over the last 10% |',
              '|---|---|---|---|---|']
    for record in gain_summary:
        lines.append(f'| {record["condition"]} | {record["n_sources"]} | '
                     f'{record["final_gain_mean"]:.2f} | {record["final_gain_sd"]:.2f} | '
                     f'{record["relative_growth_last_10pct"]:+.4f} |')
    if gain_contrast:
        lines += ['', f'- navigation minus matched final gain: '
                      f'{gain_contrast["mean_difference"]:+.2f}, exact p '
                      f'{gain_contrast["exact_p"]:.6f}, '
                      f'{gain_contrast["n_sources"]} sources',
                  '- If both conditions have plateaued and the ordering holds, the budget '
                  'caveat on the Phase F gain comparison can be dropped.\n']
    (out / 'EXTENDED_RESULT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
