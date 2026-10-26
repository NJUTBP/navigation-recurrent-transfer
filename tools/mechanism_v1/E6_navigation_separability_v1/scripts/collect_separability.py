#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

import separability_common as common


def exact_sign_flip(values):
    x = np.asarray(values, dtype=float)
    obs = abs(float(x.mean()))
    total = 0
    extreme = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(x)):
        stat = abs(float((x * np.asarray(signs)).mean()))
        total += 1
        extreme += stat >= obs - 1e-15
    return extreme / total


def bootstrap_ci(values, rng, n=common.BOOTSTRAP_SAMPLES):
    x = np.asarray(values, dtype=float)
    means = np.empty(n, dtype=float)
    for i in range(n):
        means[i] = x[rng.integers(0, len(x), size=len(x))].mean()
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def holm(pvals):
    items = sorted(enumerate(pvals), key=lambda z: z[1])
    out = [1.0] * len(pvals)
    running = 0.0
    m = len(pvals)
    for rank, (idx, p) in enumerate(items):
        adj = min(1.0, (m - rank) * p)
        running = max(running, adj)
        out[idx] = running
    return out


def e6_seen_long(source, target, rho):
    path = common.E6_ROOT / f'source_{source}' / f'target_{target}' / f'rho_{int(round(rho * 1000)):04d}' / 'metrics.csv'
    rows = common.read_csv(path)
    hit = [r for r in rows if r['split'] == 'seen_long']
    if len(hit) != 1:
        raise RuntimeError(f'E6 seen_long row count {len(hit)} for {source}/{target}/{rho}')
    return float(hit[0]['accuracy'])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-root', type=Path, default=common.OUTPUT_ROOT)
    args = parser.parse_args()
    root = args.output_root
    rows = []
    missing = []
    for source in range(6101, 6111):
        for target in (84001, 84002, 84003):
            for rho in common.RHOS:
                cdir = common.cell_dir(root, source, target, rho)
                if not (cdir / 'COMPLETE.json').is_file():
                    missing.append((source, target, rho))
                    continue
                metrics = common.read_csv(cdir / 'navigation_metrics.csv')
                for r in metrics:
                    rows.append({**r, 'transfer_seen_long': e6_seen_long(source, target, rho)})
    if missing:
        raise RuntimeError(f'missing cells: {missing[:10]} total={len(missing)}')
    if len(rows) != 420:
        raise RuntimeError(f'expected 420 navigation rows, got {len(rows)}')
    agg = root / 'aggregate'
    agg.mkdir(parents=True, exist_ok=True)
    # write all rows with stable union of keys
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    import csv
    with (agg / 'all_navigation_metrics.csv').open('w', newline='', encoding='utf-8') as h:
        w = csv.DictWriter(h, fieldnames=keys); w.writeheader(); w.writerows(rows)

    refit = [r for r in rows if r['phase'] == 'refit_best']
    source_rows = []
    for source in range(6101, 6111):
        for rho in common.RHOS:
            group = [r for r in refit if int(r['source_seed']) == source and abs(float(r['rho']) - rho) < 1e-12]
            if len(group) != 3:
                raise RuntimeError('target aggregation failed')
            source_rows.append({
                'source_seed': source, 'rho': rho,
                'nav_frac_loss_captured': float(np.mean([float(r['frac_loss_captured']) for r in group])),
                'nav_pi_error_relative': float(np.mean([float(r['pi_error_relative']) for r in group])),
                'transfer_seen_long': float(np.mean([float(r['transfer_seen_long']) for r in group])),
                'best_epoch_mean': float(np.mean([float(r['best_epoch']) for r in group])),
                'n_best_in_last_eval': int(sum(float(r['best_epoch']) >= common.EPOCHS - common.EVAL_EVERY for r in group)),
            })
    with (agg / 'source_level.csv').open('w', newline='', encoding='utf-8') as h:
        w = csv.DictWriter(h, fieldnames=list(source_rows[0])); w.writeheader(); w.writerows(source_rows)

    curve = []
    for rho in common.RHOS:
        g = [r for r in source_rows if abs(float(r['rho']) - rho) < 1e-12]
        curve.append({
            'rho': rho,
            'nav_frac_loss_captured_mean': float(np.mean([r['nav_frac_loss_captured'] for r in g])),
            'nav_pi_error_relative_mean': float(np.mean([r['nav_pi_error_relative'] for r in g])),
            'transfer_seen_long_mean': float(np.mean([r['transfer_seen_long'] for r in g])),
            'cells_best_in_last_eval': int(sum(r['n_best_in_last_eval'] for r in g)),
        })
    with (agg / 'rho_curve.csv').open('w', newline='', encoding='utf-8') as h:
        w = csv.DictWriter(h, fieldnames=list(curve[0])); w.writeheader(); w.writerows(curve)

    rng = np.random.default_rng(common.STATISTICS_SEED)
    tests = []
    base = {r['source_seed']: r for r in source_rows if float(r['rho']) == 0.0}
    raw_p = []
    for rho in common.KEY_RHOS:
        excess = []
        transfer_drop = []
        nav_drop = []
        for source in range(6101, 6111):
            now = next(r for r in source_rows if r['source_seed'] == source and abs(float(r['rho']) - rho) < 1e-12)
            td = base[source]['transfer_seen_long'] - now['transfer_seen_long']
            nd = base[source]['nav_frac_loss_captured'] - now['nav_frac_loss_captured']
            transfer_drop.append(td); nav_drop.append(nd); excess.append(td - nd)
        p = exact_sign_flip(excess)
        raw_p.append(p)
        lo, hi = bootstrap_ci(excess, rng)
        tests.append({'rho': rho, 'transfer_drop': float(np.mean(transfer_drop)),
                      'navigation_drop': float(np.mean(nav_drop)), 'excess_transfer_drop': float(np.mean(excess)),
                      'ci_low': lo, 'ci_high': hi, 'exact_p': p,
                      'positive_sources': int(sum(v > 0 for v in excess))})
    adj = holm(raw_p)
    for row, p_adj in zip(tests, adj):
        row['holm_p'] = p_adj
        row['separable'] = bool(row['excess_transfer_drop'] > 0 and row['ci_low'] > 0 and p_adj < 0.05)
    with (agg / 'separability_tests.csv').open('w', newline='', encoding='utf-8') as h:
        w = csv.DictWriter(h, fieldnames=list(tests[0])); w.writeheader(); w.writerows(tests)

    last_count = sum(int(r['best_epoch']) >= common.EPOCHS - common.EVAL_EVERY
                     for r in refit if abs(float(r['rho']) - 0.05) < 1e-12 or abs(float(r['rho']) - 0.1) < 1e-12)
    key_cells = 60
    convergence_ok = last_count <= int(0.25 * key_cells)
    support = bool(convergence_ok and all(r['separable'] for r in tests))
    result = {
        'schema_version': common.SCHEMA_VERSION, 'status': 'complete', 'n_cells': 210,
        'key_rhos': list(common.KEY_RHOS), 'convergence_ok': convergence_ok,
        'key_cells_best_in_last_eval': last_count, 'key_cells_total': key_cells,
        'separability_support': support, 'tests': tests, 'rho_curve': curve,
        'interpretation_rule': 'Support requires both rho=0.05 and rho=0.1 excess-transfer-drop tests to pass, and <=25% of key-rho cells selecting the final evaluation window.'
    }
    (agg / 'SEPARABILITY_RESULT.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    lines = [
        '# E6 navigation separability control', '',
        '- completed cells: 210/210',
        f'- convergence gate: {"PASS" if convergence_ok else "FAIL"} ({last_count}/{key_cells} key-rho cells best in final evaluation window)',
        f'- separability support: {"YES" if support else "NO"}', '',
        '## Key tests', '',
        '| rho | transfer drop | navigation drop | excess transfer drop | 95% CI | Holm p | separable |',
        '|---:|---:|---:|---:|---:|---:|---|',
    ]
    for r in tests:
        lines.append(f"| {r['rho']:.2f} | {r['transfer_drop']:.4f} | {r['navigation_drop']:.4f} | {r['excess_transfer_drop']:.4f} | [{r['ci_low']:.4f}, {r['ci_high']:.4f}] | {r['holm_p']:.6f} | {'YES' if r['separable'] else 'NO'} |")
    lines += ['', '## Rho curve', '',
              '| rho | navigation frac-loss-captured | navigation PI error / arena | transfer seen_long |',
              '|---:|---:|---:|---:|']
    for r in curve:
        lines.append(f"| {r['rho']:.2f} | {r['nav_frac_loss_captured_mean']:.4f} | {r['nav_pi_error_relative_mean']:.4f} | {r['transfer_seen_long_mean']:.4f} |")
    (agg / 'SEPARABILITY_RESULT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
