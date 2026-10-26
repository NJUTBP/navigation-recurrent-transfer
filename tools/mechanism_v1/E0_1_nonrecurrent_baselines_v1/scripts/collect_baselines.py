#!/usr/bin/env python3
"""E0.1 collector: baseline accuracy, horizon profile, and the recurrent references.

Pre-registered readouts (written before any E0.1 cell ran; do not edit afterwards)
----------------------------------------------------------------------------------
R0  Gate.  The preflight installs the exact nearest-lattice-point solution in both
    baselines and requires accuracy 1.0 on every split.  If that failed no cell
    ran, so every number below comes from a function class that contains the
    exact answer.

R1  Four-split accuracy of both baselines at 8,000 and 32,000 updates, mean over
    targets with the per-target range.  No screening threshold is applied.

R2  Horizon profile.  Accuracy by path length 1-12 for both baselines, placed
    next to the navigation and matched scalar-gain conditions on the same
    targets and the same budget (Phase F for 8,000, E0.9a for 32,000).  An
    exactly additive state cannot lose accuracy with length; if the trained
    accumulator does, its learned relation vectors are not exactly antipodal and
    the error grows with the number of steps.

R3  Descriptive contrasts.  The baselines have no source dimension, so for each
    source the recurrent value minus the target-matched baseline value is
    reported with the usual exact sign-flip and bootstrap interval.  With a
    constant right-hand side this is a one-sample location test on the
    recurrent values and is read descriptively.

Framing fixed in advance: the accumulator is an algorithmic reference, not a
recurrent-memory control.  Long-horizon extrapolation in the recurrent
comparison tests whether a learned recurrent transition can maintain an additive
path code; an explicit accumulator implements that operation directly.
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

from baseline_common import (
    BASELINES, BUDGETS, E09A_ROOT, OUTPUT_ROOT, PHASEF_OUTPUT, SCHEMA_VERSION, TEST_SPLITS,
)

STATISTICS_SEED = 982451653
BOOTSTRAP_SAMPLES = 20000
REFERENCE_CONDITIONS = ('navigation_trainable_scalar_gain', 'matched_random_trainable_scalar_gain')
REFERENCE_LABEL = {'navigation_trainable_scalar_gain': 'nav_scalar',
                   'matched_random_trainable_scalar_gain': 'matched_scalar'}

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

def load_cells(root: Path):
    metrics, per_length, diagnostics = [], [], []
    complete = 0
    for cell in sorted(root.glob('target_*/budget_*')):
        if not (cell / 'COMPLETE.json').is_file():
            continue
        complete += 1
        metrics += read_csv(cell / 'metrics.csv')
        per_length += read_csv(cell / 'per_length.csv')
        diagnostics += read_csv(cell / 'condition_diagnostics.csv')
    return complete, metrics, per_length, diagnostics

def load_reference(root: Path, group_glob: str, budget: int):
    metrics, per_length = [], []
    for cell in sorted(root.glob(f'source_*/target_*/{group_glob}')):
        if not (cell / 'COMPLETE.json').is_file():
            continue
        for row in read_csv(cell / 'metrics.csv'):
            if row.get('condition') in REFERENCE_CONDITIONS:
                metrics.append({**row, 'budget': budget})
        for row in read_csv(cell / 'per_length.csv'):
            if row.get('condition') in REFERENCE_CONDITIONS:
                per_length.append({**row, 'budget': budget})
    return metrics, per_length

def main() -> int:
    parser = argparse.ArgumentParser(description='E0.1 collector')
    parser.add_argument('--root', type=Path, default=OUTPUT_ROOT)
    parser.add_argument('--phasef-root', type=Path, default=PHASEF_OUTPUT)
    parser.add_argument('--e09a-root', type=Path, default=E09A_ROOT)
    args = parser.parse_args()
    out = args.root / 'aggregate'
    out.mkdir(parents=True, exist_ok=True)

    complete, metrics, per_length, diagnostics = load_cells(args.root)
    if not metrics:
        raise SystemExit('no completed E0.1 cells found')
    write_csv(out / 'all_metrics.csv', metrics)
    write_csv(out / 'all_per_length.csv', per_length)
    write_csv(out / 'all_condition_diagnostics.csv', diagnostics)

    base_acc = defaultdict(dict)
    for row in metrics:
        base_acc[(row['baseline'], int(row['budget']), row['split'])][int(row['target_seed'])] = float(row['accuracy'])
    summary = []
    for (name, budget, split), values in sorted(base_acc.items()):
        array = np.array(list(values.values()))
        summary.append({'baseline': name, 'budget': budget, 'split': split,
                        'n_targets': array.size, 'mean_accuracy': float(array.mean()),
                        'min_accuracy': float(array.min()), 'max_accuracy': float(array.max())})
    write_csv(out / 'baseline_summary.csv', summary)

    references = {8000: load_reference(args.phasef_root, 'group_learned_interface', 8000),
                  32000: load_reference(args.e09a_root, 'group_scalar_pair', 32000)}
    ref_acc = defaultdict(lambda: defaultdict(dict))
    ref_len = defaultdict(lambda: defaultdict(dict))
    for budget, (rows, lengths) in references.items():
        for row in rows:
            key = (REFERENCE_LABEL[row['condition']], budget, row['split'])
            ref_acc[key][int(row['source_seed'])][int(row['target_seed'])] = float(row['accuracy'])
        for row in lengths:
            if row['split'] not in ('id_seen_short', 'seen_long'):
                continue
            key = (REFERENCE_LABEL[row['condition']], budget, int(row['path_length']))
            ref_len[key][int(row['source_seed'])][int(row['target_seed'])] = float(row['accuracy'])

    base_len = defaultdict(dict)
    for row in per_length:
        if row['split'] in ('id_seen_short', 'seen_long'):
            base_len[(row['baseline'], int(row['budget']), int(row['path_length']))][int(row['target_seed'])] = float(row['accuracy'])

    contrasts, profile = [], []
    for budget in BUDGETS:
        for split in TEST_SPLITS:
            for label in REFERENCE_LABEL.values():
                by_source = ref_acc.get((label, budget, split), {})
                for name in BASELINES:
                    baseline = base_acc.get((name, budget, split), {})
                    common = sorted(set(baseline) & set.intersection(
                        *[set(v) for v in by_source.values()])) if by_source else []
                    if not common:
                        continue
                    b = float(np.mean([baseline[t] for t in common]))
                    diffs = [float(np.mean([v[t] for t in common])) - b
                             for _, v in sorted(by_source.items())]
                    low, high = bootstrap_ci(diffs)
                    contrasts.append({'budget': budget, 'split': split,
                                      'contrast': f'{label}_minus_{name}',
                                      'targets': ' '.join(str(t) for t in common),
                                      'n_sources': len(diffs), 'baseline_accuracy': b,
                                      'recurrent_mean': b + float(np.mean(diffs)),
                                      'mean_difference': float(np.mean(diffs)),
                                      'ci_low': low, 'ci_high': high,
                                      'exact_p': exact_sign_flip_p(diffs),
                                      'n_positive': int(np.sum(np.asarray(diffs) > 0))})
        baseline_targets = {t for (n, u, _), v in base_len.items() if u == budget for t in v}
        reference_targets = {t for (lab, u, _), by_source in ref_len.items() if u == budget
                             for v in by_source.values() for t in v}
        common = baseline_targets & reference_targets if reference_targets else baseline_targets
        for length in range(1, 13):
            row = {'budget': budget, 'path_length': length,
                   'targets': ' '.join(str(t) for t in sorted(common))}
            for name in BASELINES:
                values = base_len.get((name, budget, length), {})
                picked = [values[t] for t in sorted(common) if t in values]
                row[name] = float(np.mean(picked)) if picked else float('nan')
            for label in REFERENCE_LABEL.values():
                by_source = ref_len.get((label, budget, length), {})
                means = [np.mean([v[t] for t in sorted(common) if t in v])
                         for v in by_source.values() if any(t in v for t in common)]
                row[label] = float(np.mean(means)) if means else float('nan')
            profile.append(row)
    write_csv(out / 'baseline_contrasts.csv', contrasts)
    write_csv(out / 'horizon_profile.csv', profile)

    result = {'schema_version': SCHEMA_VERSION, 'n_cells_complete': complete,
              'n_cells_expected': len(BUDGETS) * 5, 'summary': summary,
              'contrasts': contrasts, 'horizon_profile': profile}
    (out / 'BASELINE_RESULT.json').write_text(json.dumps(result, indent=2), encoding='utf-8')

    lines = ['# E0.1 non-recurrent baselines on the 8x8 task\n',
             f'- completed cells: {complete} of {len(BUDGETS) * 5}\n',
             '## R1 accuracy (mean over targets, [min, max])\n',
             '| baseline | budget | ' + ' | '.join(TEST_SPLITS) + ' |',
             '|---|---|' + '---|' * len(TEST_SPLITS)]
    table = {(r['baseline'], r['budget'], r['split']): r for r in summary}
    for name in BASELINES:
        for budget in BUDGETS:
            cells = []
            for split in TEST_SPLITS:
                r = table.get((name, budget, split))
                cells.append(f"{r['mean_accuracy']:.3f} [{r['min_accuracy']:.3f}, {r['max_accuracy']:.3f}]"
                             if r else '-')
            lines.append(f'| {name} | {budget} | ' + ' | '.join(cells) + ' |')
    lines += ['', '## R2 accuracy by path length (id_seen_short for 1-5, seen_long for 6-12)\n',
              '| budget | L | ' + ' | '.join(BASELINES) + ' | nav_scalar | matched_scalar |',
              '|---|---|' + '---|' * (len(BASELINES) + 2)]
    for row in profile:
        values = [row[k] for k in BASELINES] + [row['nav_scalar'], row['matched_scalar']]
        lines.append(f"| {row['budget']} | {row['path_length']} | "
                     + ' | '.join('-' if not np.isfinite(v) else f'{v:.3f}' for v in values) + ' |')
    if contrasts:
        lines += ['', '## R3 descriptive contrasts (recurrent minus baseline)\n',
                  '| budget | split | contrast | targets | difference | 95% CI | exact p |',
                  '|---|---|---|---|---|---|---|']
        for r in contrasts:
            lines.append(f"| {r['budget']} | {r['split']} | {r['contrast']} | {r['targets']} | "
                         f"{r['mean_difference']:+.4f} | [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] | "
                         f"{r['exact_p']:.6f} |")
    (out / 'BASELINE_RESULT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
