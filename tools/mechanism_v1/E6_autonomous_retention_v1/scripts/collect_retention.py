#!/usr/bin/env python3
"""Collect C1 autonomous-retention rows and compare their rho dose curve with E6."""
from __future__ import annotations

import argparse
import csv
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from run_retention import OUTPUT_ROOT, E6_ROOT, E5B_ROOT, SCHEMA_VERSION
from rho_core import RHO_GRID, SCAN_RHOS

STATISTICS_SEED = 982451653
BOOTSTRAP_SAMPLES = 20000


def read_csv(path: Path):
    if not path.is_file():
        return []
    with path.open('r', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def bootstrap_ci(values):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(STATISTICS_SEED)
    means = values[rng.integers(0, values.size, size=(BOOTSTRAP_SAMPLES, values.size))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def exact_sign_flip_p(values):
    values = np.asarray(values, dtype=np.float64)
    observed = abs(float(values.mean()))
    hits = sum(1 for signs in itertools.product((1.0, -1.0), repeat=values.size)
               if abs(float(np.mean(np.asarray(signs) * values))) >= observed - 1e-12)
    return hits / float(2 ** values.size)


def source_means(rows, value_field):
    bucket = defaultdict(list)
    for row in rows:
        bucket[(int(row['source_seed']), float(row['rho']))].append(float(row[value_field]))
    return {(s, rho): float(np.mean(v)) for (s, rho), v in bucket.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description='Collect C1 retention')
    parser.add_argument('--root', type=Path, default=OUTPUT_ROOT)
    parser.add_argument('--e6-root', type=Path, default=E6_ROOT)
    parser.add_argument('--e5b-root', type=Path, default=E5B_ROOT)
    args = parser.parse_args()
    out = args.root / 'aggregate'
    out.mkdir(parents=True, exist_ok=True)

    rows, complete, anchor0, anchorcomp = [], 0, [], []
    for cell in sorted(args.root.glob('source_*/target_*')):
        marker = cell / 'COMPLETE.json'
        if not marker.is_file():
            continue
        complete += 1
        meta = json.loads(marker.read_text(encoding='utf-8'))
        anchor0.append(float(meta['rho0_navigation_relative_deviation']))
        anchorcomp.append(float(meta['rho01_rho09_relative_deviation']))
        rows += read_csv(cell / 'retention.csv')
    if complete != 30 or len(rows) != 210:
        raise SystemExit(f'incomplete C1 output: pairs={complete}/30 rows={len(rows)}/210')
    write_csv(out / 'all_retention.csv', rows)

    primary = source_means(rows, 'autonomous_retention_mean')
    source_rows = [{'source_seed': s, 'rho': rho, 'autonomous_retention_mean': value}
                   for (s, rho), value in sorted(primary.items())]
    write_csv(out / 'source_level_retention.csv', source_rows)

    e6_metrics = [r for r in read_csv(args.e6_root / 'aggregate' / 'all_metrics.csv') if r['split'] == 'seen_long']
    accuracy = source_means(e6_metrics, 'accuracy')
    sources = sorted({s for s, _ in primary})
    curve = []
    for rho in RHO_GRID:
        rv = np.asarray([primary[(s, float(rho))] for s in sources], dtype=np.float64)
        av = np.asarray([accuracy[(s, float(rho))] for s in sources], dtype=np.float64)
        rlo, rhi = bootstrap_ci(rv)
        alo, ahi = bootstrap_ci(av)
        curve.append({'rho': rho, 'n_sources': len(sources), 'retention_mean': float(rv.mean()),
                      'retention_ci_low': rlo, 'retention_ci_high': rhi,
                      'seen_long_mean': float(av.mean()), 'seen_long_ci_low': alo, 'seen_long_ci_high': ahi})
    base_r = next(r['retention_mean'] for r in curve if r['rho'] == 0.0)
    base_a = next(r['seen_long_mean'] for r in curve if r['rho'] == 0.0)
    end_r = next(r['retention_mean'] for r in curve if r['rho'] == 0.5)
    end_a = next(r['seen_long_mean'] for r in curve if r['rho'] == 0.5)
    for row in curve:
        row['retention_ratio_to_rho0'] = row['retention_mean'] / base_r
        row['retention_drop_fraction_of_0_to_05'] = ((base_r - row['retention_mean']) / (base_r - end_r)
                                                       if abs(base_r - end_r) > 1e-12 else float('nan'))
        row['accuracy_drop_fraction_of_0_to_05'] = ((base_a - row['seen_long_mean']) / (base_a - end_a)
                                                    if abs(base_a - end_a) > 1e-12 else float('nan'))
    write_csv(out / 'rho_curve.csv', curve)

    contrasts = []
    for rho in SCAN_RHOS[1:]:
        diffs = np.asarray([primary[(s, float(rho))] - primary[(s, 0.0)] for s in sources])
        lo, hi = bootstrap_ci(diffs)
        ratios = np.asarray([primary[(s, float(rho))] / primary[(s, 0.0)] for s in sources])
        rlo, rhi = bootstrap_ci(ratios)
        contrasts.append({'rho': rho, 'n_sources': len(sources), 'difference_from_rho0': float(diffs.mean()),
                          'difference_ci_low': lo, 'difference_ci_high': hi, 'exact_sign_flip_p': exact_sign_flip_p(diffs),
                          'n_negative': int((diffs < 0).sum()), 'retention_ratio_to_rho0': float(ratios.mean()),
                          'ratio_ci_low': rlo, 'ratio_ci_high': rhi})
    write_csv(out / 'rho0_contrasts.csv', contrasts)

    xs, ys = [], []
    for source in sources:
        rv = np.asarray([primary[(source, float(r))] for r in SCAN_RHOS])
        av = np.asarray([accuracy[(source, float(r))] for r in SCAN_RHOS])
        xs.append(rv - rv.mean())
        ys.append(av - av.mean())
    x, y = np.concatenate(xs), np.concatenate(ys)
    within_r = float(np.corrcoef(x, y)[0, 1])

    e5b_rows = [r for r in read_csv(args.e5b_root / 'aggregate' / 'all_retention.csv')
                 if r['state_set'] == 'H_nav' and r['origin'] == 'navigation']
    e5b_by_source = defaultdict(list)
    for row in e5b_rows:
        e5b_by_source[int(row['source_seed'])].append(float(row['autonomous_retention_mean']))
    e5b_source = {s: float(np.mean(v)) for s, v in e5b_by_source.items()}
    anchor_diffs = np.asarray([primary[(s, 0.0)] - e5b_source[s] for s in sources])
    c01 = next(r for r in contrasts if abs(float(r['rho']) - 0.1) < 1e-12)
    result = {
        'schema_version': SCHEMA_VERSION, 'status': 'complete', 'exploratory_followup': True,
        'n_pairs': complete, 'n_rows': len(rows), 'n_sources': len(sources),
        'primary_state_set': 'H_nav_reference_permuted',
        'primary_alignment': 'permutation_only_input_sign_invariant',
        'anchors': {'max_pair_rho0_navigation_relative_deviation': max(anchor0),
                    'max_pair_rho01_rho09_relative_deviation': max(anchorcomp),
                    'max_source_difference_vs_e5b_navigation_rho0': float(np.max(np.abs(anchor_diffs)))},
        'descriptive_within_source_retention_accuracy_r': within_r,
        'rho01_retention_ratio_to_rho0': c01['retention_ratio_to_rho0'],
        'rho01_ratio_ci': [c01['ratio_ci_low'], c01['ratio_ci_high']],
        'note': 'Post-E6 exploratory measurement. Correlation is descriptive; rho is the manipulated common cause.'}
    (out / 'RETENTION_RESULT.json').write_text(json.dumps(result, indent=2), encoding='utf-8')

    lines = ['# C1: E6 source-side autonomous retention', '',
             '- exploratory follow-up defined after E6; no model retraining',
             '- primary state set: unperturbed E5b H_nav states, permutation-aligned only',
             '- this makes the measure invariant to the E6 input-sign complement (rho 0.1 vs 0.9)',
             f'- completed pairs: {complete}/30; rho cores measured: {len(rows)}/210',
             f"- max rho=0 anchor deviation: {max(anchor0):.3e}",
             f"- max rho=0.1 vs 0.9 complement deviation: {max(anchorcomp):.3e}",
             f"- max source difference vs E5b navigation retention: {np.max(np.abs(anchor_diffs)):.3e}", '',
             '## Dose curve', '',
             '| rho | retention | 95% CI | ratio to rho0 | seen_long | retention drop fraction | accuracy drop fraction |',
             '|---:|---:|---:|---:|---:|---:|---:|']
    for row in curve:
        lines.append(f"| {row['rho']:.2f} | {row['retention_mean']:.6f} | [{row['retention_ci_low']:.6f}, {row['retention_ci_high']:.6f}] | "
                     f"{row['retention_ratio_to_rho0']:.4f} | {row['seen_long_mean']:.4f} | "
                     f"{row['retention_drop_fraction_of_0_to_05']:.3f} | {row['accuracy_drop_fraction_of_0_to_05']:.3f} |")
    lines += ['', '## rho=0 contrasts (exploratory)', '',
              '| rho | retention difference | 95% CI | exact p | negative sources | ratio to rho0 | ratio 95% CI |',
              '|---:|---:|---:|---:|---:|---:|---:|']
    for row in contrasts:
        lines.append(f"| {row['rho']:.2f} | {row['difference_from_rho0']:+.6f} | [{row['difference_ci_low']:+.6f}, {row['difference_ci_high']:+.6f}] | "
                     f"{row['exact_sign_flip_p']:.6f} | {row['n_negative']}/{row['n_sources']} | {row['retention_ratio_to_rho0']:.4f} | "
                     f"[{row['ratio_ci_low']:.4f}, {row['ratio_ci_high']:.4f}] |")
    lines += ['', f'- descriptive within-source correlation(retention, seen_long) over rho in [0, 0.5]: {within_r:+.4f}',
              '- Do not interpret that correlation as mediation: rho is the manipulated common cause.',
              f"- Candidate c for the next W->cW preregistration, if used, is the source-side rho=0.1/rho=0 retention ratio: {c01['retention_ratio_to_rho0']:.4f} "
              f"[{c01['ratio_ci_low']:.4f}, {c01['ratio_ci_high']:.4f}].", '']
    (out / 'RETENTION_RESULT.md').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines[:28]))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
