#!/usr/bin/env python3
"""Verdicts of the mechanism_v1 family (E2, E3, E4, E6).

Rules: ANALYSIS_PLAN.md with revision 1 of 2026-09-20, applied mechanically to
the rows the two collectors already wrote.

  Anchors   two anchors x (id_seen_short, seen_long, heldout_long) = 6 rows;
            pass only if every 95 % interval contains 0.
  Main      mean per-source slope of seen_long on rho < 0 and exact p < 0.05.
  Primary   R_ret_diff at Bstar, within-source slope: slope > 0, permutation
            p < 0.05 and interval lower bound > 0 -> supported.  Read only when
            the anchors pass.
  Holm      S1a  Spearman(R_ret_diff at Bstar, seen_long), navigation, 8000
            S1b  the same at 32000
            S2   Spearman(S_cons at Bstar, seen_long), navigation, 8000
            S3   E6 within-source slope of seen_long on S_cons at Bstar
            S4   R_ret_diff at Bstar, navigation minus matched, exact p
            S5   E6 within-source slope on R_ret at B (plan_literal)
            Adjusted p < 0.05 with the expected sign -> supported; for the
            expected-null S2 and S3, adjusted p >= 0.05 -> consistent with the
            expectation, otherwise contrary to it.

Validation recomputes, from the per-cell tables and with separate code, the
main slope, the six anchor differences, the primary, S3 and S5 slopes (least
squares with source dummies), S4 and S1b.  Any mismatch above 1e-9 stops the
run.

Exploratory tables (not part of any verdict, labelled as such):
  rho profile of accuracy, learned gain, downstream norm retention and relative
  drive drift; how well candidate measures follow the steep rho = 0 to 0.1
  segment; accuracy of the Phase F matched cores grouped by their realized flip
  count; rho = 0.9 against rho = 0.1 for H_nav-based and downstream measures.
"""
from __future__ import annotations
import os

import argparse
import csv
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

MECH = Path(os.environ.get('NRT_PROJECT_ROOT', str(Path(__file__).resolve().parents[4])) + '/rerun_best_validation/followups/mechanism_v1')
SOURCES = tuple(range(6101, 6111))
SCAN = (0.0, 0.05, 0.1, 0.2, 0.35, 0.5)
ANCHOR_SPLITS = ('id_seen_short', 'seen_long', 'heldout_long')
EXPECT = {'S1a': 'positive', 'S1b': 'positive', 'S2': 'null', 'S3': 'null', 'S4': 'positive', 'S5': 'positive'}
ALPHA = 0.05
TOL = 1e-9
STATISTICS_SEED = 982451653
BOOTSTRAP_SAMPLES = 20000
NAV = 'navigation_trainable_scalar_gain'
MATCHED = 'matched_random_trainable_scalar_gain'

def read_csv(path: Path) -> list[dict]:
    with path.open('r', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))

def write_csv(path: Path, rows: list[dict]) -> None:
    fields: list[str] = []
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

def num(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float('nan')

def one(rows: list[dict], **match) -> dict:
    hits = [r for r in rows if all(str(r.get(k)) == str(v) for k, v in match.items())]
    if len(hits) != 1:
        raise SystemExit(f'expected exactly one row for {match}, found {len(hits)}')
    return hits[0]

def mean_by(rows, keys, field, keep=lambda r: True) -> dict:
    bucket = defaultdict(list)
    for r in rows:
        if keep(r) and np.isfinite(num(r.get(field))):
            bucket[tuple(round(num(r[k]), 4) if k == 'rho' else int(r[k]) for k in keys)].append(num(r[field]))
    return {k: float(np.mean(v)) for k, v in bucket.items()}

def source_level(cell_means: dict) -> dict:
    """(source, target, rho) means -> (source, rho) means: targets first, as pre-registered."""
    bucket = defaultdict(list)
    for (s, _, r), v in cell_means.items():
        bucket[(s, r)].append(v)
    return {k: float(np.mean(v)) for k, v in bucket.items()}

def holm(pvalues: dict) -> dict:
    ordered = sorted(pvalues.items(), key=lambda kv: kv[1])
    adjusted, running = {}, 0.0
    for i, (name, p) in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - i) * p))
        adjusted[name] = running
    return adjusted

def exact_sign_flip_p(values) -> float:
    values = np.asarray(values, dtype=np.float64)
    observed = abs(float(values.mean()))
    hits = sum(abs(float(np.mean(np.asarray(signs) * values))) >= observed - 1e-12
               for signs in itertools.product((1.0, -1.0), repeat=values.size))
    return hits / 2.0 ** values.size

def bootstrap_ci(values) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(STATISTICS_SEED)
    means = values[rng.integers(0, values.size, size=(BOOTSTRAP_SAMPLES, values.size))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

def average_ranks(x) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind='mergesort')
    ranks = np.empty(x.size)
    i = 0
    while i < x.size:
        j = i
        while j + 1 < x.size and x[order[j + 1]] == x[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j)
        i = j + 1
    return ranks

def spearman(x, y) -> float:
    return float(np.corrcoef(average_ranks(x), average_ranks(y))[0, 1])

def dummy_slope(x_by_source: dict, y_by_source: dict) -> float:
    sources = sorted(x_by_source)
    rows, target = [], []
    for i, s in enumerate(sources):
        for xv, yv in zip(x_by_source[s], y_by_source[s]):
            rows.append([xv] + [1.0 if j == i else 0.0 for j in range(len(sources))])
            target.append(yv)
    return float(np.linalg.lstsq(np.asarray(rows), np.asarray(target), rcond=None)[0][0])

def levels(table: dict, rhos=SCAN) -> dict:
    return {s: np.array([table[(s, round(r, 4))] for r in rhos]) for s in SOURCES}

def secondary_verdict(name: str, estimate: float, adjusted: float) -> str:
    if EXPECT[name] == 'null':
        return 'consistent_with_expected_null' if adjusted >= ALPHA else 'contrary_to_expected_null'
    return 'supported' if adjusted < ALPHA and estimate > 0 else 'not_supported'

def within_r(x: dict, y: dict) -> float:
    xs = np.concatenate([x[s] - x[s].mean() for s in SOURCES])
    ys = np.concatenate([y[s] - y[s].mean() for s in SOURCES])
    return float(np.corrcoef(xs, ys)[0, 1])

def partial_r_given_linear_rho(x: dict, y: dict) -> float:
    design = np.c_[np.ones(len(SCAN)), np.asarray(SCAN)]
    def residual(v):
        return v - design @ np.linalg.lstsq(design, v, rcond=None)[0]
    xs = np.concatenate([residual(x[s]) for s in SOURCES])
    ys = np.concatenate([residual(y[s]) for s in SOURCES])
    return float(np.corrcoef(xs, ys)[0, 1])

def share_by_rho01(x: dict) -> float:
    full = np.mean([x[s][5] - x[s][0] for s in SOURCES])
    return float(np.mean([x[s][2] - x[s][0] for s in SOURCES]) / full) if abs(full) > 1e-12 else float('nan')

def main() -> int:
    parser = argparse.ArgumentParser(description='mechanism_v1 verdicts')
    parser.add_argument('--step6', type=Path, default=MECH / 'E234_source_measures' / 'aggregate')
    parser.add_argument('--e6', type=Path, default=MECH / 'E6_rho_scan' / 'aggregate')
    parser.add_argument('--e09a-metrics', type=Path, default=MECH / 'E0_9a_extended_budget' / 'aggregate' / 'all_metrics.csv')
    parser.add_argument('--out', type=Path, default=MECH / 'verdicts_v1')
    args = parser.parse_args()
    if (args.out / 'COMPLETE.json').exists():
        print('already complete')
        return 0
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit('partial output exists; inspect before rerun')
    args.out.mkdir(parents=True, exist_ok=True)

    s6_result = json.loads((args.step6 / 'SOURCE_MEASURES_RESULT.json').read_text(encoding='utf-8'))
    e6_result = json.loads((args.e6 / 'RHO_SCAN_RESULT.json').read_text(encoding='utf-8'))
    for result, cells in ((s6_result, 50), (e6_result, 210)):
        if (result['primary_metric'], result['primary_working_point']) != ('R_ret_diff', 'Bstar'):
            raise SystemExit('collector ran with a primary other than R_ret_diff at Bstar')
        if result['n_cells_complete'] != cells:
            raise SystemExit('collector aggregate is incomplete')
    if e6_result['cells_with_primary_working_point'] != 210:
        raise SystemExit('primary working point missing in some E6 cells')

    tests = read_csv(args.e6 / 'main_and_anchor_tests.csv')
    mediation = read_csv(args.e6 / 'mediation.csv')
    contrasts = read_csv(args.step6 / 'core_contrasts.csv')
    checks = read_csv(args.step6 / 'consistency_checks.csv')

    anchors = []
    for name in ('anchor_rho0_minus_navigation_E09a', 'symmetry_rho0p9_minus_rho0p1'):
        for split in ANCHOR_SPLITS:
            r = one(tests, test=name, split=split)
            low, high = num(r['ci_low']), num(r['ci_high'])
            anchors.append({'anchor': name, 'split': split, 'difference': num(r['difference']), 'ci_low': low,
                            'ci_high': high, 'exact_p': num(r['exact_p']), 'interval_contains_zero': low <= 0.0 <= high})
    anchors_pass = all(a['interval_contains_zero'] for a in anchors)
    main_row = one(tests, test='main_seen_long_slope_on_rho')
    main_effect = num(main_row['difference']) < 0 and num(main_row['exact_p']) < ALPHA
    prim = one(mediation, role='primary')
    primary_supported = (num(prim['within_slope']) > 0 and num(prim['permutation_p']) < ALPHA
                         and num(prim['ci_low']) > 0)
    primary_verdict = ('not_interpreted_anchor_failure' if not anchors_pass
                       else 'supported' if primary_supported else 'not_supported')

    s3 = one(mediation, metric='S_cons', working_point='Bstar')
    s5 = one(mediation, role='plan_literal')
    s4 = one(contrasts, metric='R_ret_diff', working_point='Bstar')
    rows = {'S1a': one(checks, check='check5_R_ret_diff', core='navigation', budget='8000', working_point='Bstar'),
            'S1b': one(checks, check='check5_R_ret_diff', core='navigation', budget='32000', working_point='Bstar'),
            'S2': one(checks, check='check6_S_cons_expected_null', core='navigation', budget='8000', working_point='Bstar')}
    secondary = {k: {'estimate': num(v['spearman']), 'p': num(v['permutation_p']), 'kind': 'spearman'} for k, v in rows.items()}
    secondary['S3'] = {'estimate': num(s3['within_slope']), 'p': num(s3['permutation_p']), 'kind': 'within_slope'}
    secondary['S4'] = {'estimate': num(s4['difference']), 'p': num(s4['exact_p']), 'kind': 'navigation_minus_matched'}
    secondary['S5'] = {'estimate': num(s5['within_slope']), 'p': num(s5['permutation_p']), 'kind': 'within_slope'}
    adjusted = holm({k: v['p'] for k, v in secondary.items()})
    verdict_rows = [{'test': 'anchors', 'rule': 'all 6 intervals contain 0', 'verdict': 'pass' if anchors_pass else 'fail'},
                    {'test': 'main_effect', 'estimate': num(main_row['difference']), 'ci_low': num(main_row['ci_low']),
                     'ci_high': num(main_row['ci_high']), 'p': num(main_row['exact_p']),
                     'verdict': 'holds' if main_effect else 'does_not_hold'},
                    {'test': 'primary_R_ret_diff_Bstar', 'estimate': num(prim['within_slope']), 'ci_low': num(prim['ci_low']),
                     'ci_high': num(prim['ci_high']), 'p': num(prim['permutation_p']), 'within_r': num(prim['within_r']),
                     'verdict': primary_verdict}]
    for name, v in secondary.items():
        verdict_rows.append({'test': name, 'kind': v['kind'], 'expected': EXPECT[name], 'estimate': v['estimate'],
                             'p': v['p'], 'holm_p': adjusted[name],
                             'verdict': secondary_verdict(name, v['estimate'], adjusted[name])})

    e6_metrics = read_csv(args.e6 / 'all_metrics.csv')
    acc = {split: source_level(mean_by(e6_metrics, ('source_seed', 'target_seed', 'rho'), 'accuracy',
                                       lambda r, s=split: r['split'] == s))
           for split in ('id_seen_short', 'seen_long', 'heldout_long')}
    e09a = read_csv(args.e09a_metrics)
    nav09 = {split: {s: v for (s,), v in mean_by(e09a, ('source_seed',), 'accuracy',
                                                  lambda r, sp=split: r['condition'] == NAV and r['split'] == sp).items()}
             for split in ANCHOR_SPLITS}
    horizon = read_csv(args.e6 / 'all_horizon_summary.csv')
    geometry = read_csv(args.e6 / 'all_geometry.csv')
    def e6_measure(table, field, wp):
        return levels(source_level(mean_by(table, ('source_seed', 'target_seed', 'rho'), field,
                                           lambda r: r['working_point'] == wp)))
    seen = levels(acc['seen_long'])
    validation = [('main_slope', num(main_row['difference']),
                   float(np.mean([np.polyfit(SCAN, seen[s], 1)[0] for s in SOURCES])))]
    for split in ANCHOR_SPLITS:
        validation.append((f'anchor_rho0_minus_nav_{split}', num(one(tests, test='anchor_rho0_minus_navigation_E09a', split=split)['difference']),
                           float(np.mean([acc[split][(s, 0.0)] - nav09[split][s] for s in SOURCES]))))
        validation.append((f'symmetry_{split}', num(one(tests, test='symmetry_rho0p9_minus_rho0p1', split=split)['difference']),
                           float(np.mean([acc[split][(s, 0.9)] - acc[split][(s, 0.1)] for s in SOURCES]))))
    validation.append(('primary_slope', num(prim['within_slope']), dummy_slope(e6_measure(horizon, 'R_ret_diff', 'Bstar'), seen)))
    validation.append(('S3_slope', num(s3['within_slope']), dummy_slope(e6_measure(geometry, 'S_cons', 'Bstar'), seen)))
    validation.append(('S5_slope', num(s5['within_slope']), dummy_slope(e6_measure(horizon, 'R_ret', 'B'), seen)))
    s6_horizon = read_csv(args.step6 / 'all_horizon_summary.csv')
    s6_cells = {core: {s: v for (s,), v in mean_by(s6_horizon, ('source_seed',), 'R_ret_diff',
                                                     lambda r, c=core: r['core'] == c and r['working_point'] == 'Bstar').items()}
                for core in ('navigation', 'matched_random')}
    validation.append(('S4_difference', num(s4['difference']),
                       float(np.mean([s6_cells['navigation'][s] - s6_cells['matched_random'][s] for s in SOURCES]))))
    validation.append(('S1b_spearman', secondary['S1b']['estimate'],
                       spearman([s6_cells['navigation'][s] for s in SOURCES], [nav09['seen_long'][s] for s in SOURCES])))
    validation_rows = [{'quantity': q, 'collector': a, 'recomputed': b, 'abs_difference': abs(a - b), 'ok': abs(a - b) <= TOL}
                       for q, a, b in validation]
    write_csv(args.out / 'validation.csv', validation_rows)
    if not all(v['ok'] for v in validation_rows):
        raise SystemExit('recomputation disagrees with the collectors; see validation.csv')

    profile = read_csv(args.e6 / 'all_downstream_profile.csv')
    norm = defaultdict(dict)
    drive = defaultdict(dict)
    for r in profile:
        key = (int(r['source_seed']), int(r['target_seed']), round(num(r['rho']), 4))
        norm[key][int(r['step'])] = num(r['hidden_norm'])
        drive[key][int(r['step'])] = num(r['delta_rectified'])
    retention = source_level({k: (v[12] / v[1]) ** (1.0 / 11.0) for k, v in norm.items()})
    drift = source_level({k: math.log(v[12] / v[1]) for k, v in drive.items()})
    gains = source_level(mean_by(read_csv(args.e6 / 'all_condition_diagnostics.csv'), ('source_seed', 'target_seed', 'rho'), 'learned_gain'))
    points = read_csv(args.e6 / 'all_working_points.csv')
    bstar = source_level(mean_by(points, ('source_seed', 'target_seed', 'rho'), 'gain', lambda r: r['working_point'] == 'Bstar'))
    rdiff = {wp: source_level(mean_by(horizon, ('source_seed', 'target_seed', 'rho'), 'R_ret_diff', lambda r, w=wp: r['working_point'] == w))
             for wp in ('A', 'g3', 'g10', 'B', 'Bstar')}
    scons = {wp: source_level(mean_by(geometry, ('source_seed', 'target_seed', 'rho'), 'S_cons', lambda r, w=wp: r['working_point'] == w))
             for wp in ('A', 'g3', 'g10', 'B', 'Bstar')}
    rho_profile = []
    for rho in SCAN + (0.9,):
        k = round(rho, 4)
        def avg(table):
            return float(np.mean([table[(s, k)] for s in SOURCES]))
        rho_profile.append({'rho': rho, 'id_seen_short': avg(acc['id_seen_short']), 'seen_long': avg(acc['seen_long']),
                            'learned_gain': avg(gains), 'downstream_norm_retention_per_step': avg(retention),
                            'downstream_log_drive_drift_12_over_1': avg(drift), 'bstar_gain': avg(bstar),
                            'R_ret_diff_A': avg(rdiff['A']), 'R_ret_diff_Bstar': avg(rdiff['Bstar']),
                            'S_cons_A': avg(scons['A']), 'S_cons_Bstar': avg(scons['Bstar'])})
    write_csv(args.out / 'exploratory_rho_profile.csv', rho_profile)
    candidates = {**{f'R_ret_diff_{w}': levels(t) for w, t in rdiff.items()}, **{f'S_cons_{w}': levels(t) for w, t in scons.items()},
                  'downstream_norm_retention': levels(retention), 'downstream_log_drive_drift': levels(drift),
                  'rho': {s: np.asarray(SCAN) for s in SOURCES}}
    follow = [{'measure': name, 'within_source_r': within_r(x, seen),
               'partial_r_given_linear_rho': partial_r_given_linear_rho(x, seen) if name != 'rho' else float('nan'),
               'share_of_rho0_to_0p5_change_by_rho0p1': share_by_rho01(x)} for name, x in candidates.items()]
    follow.append({'measure': 'seen_long itself', 'within_source_r': 1.0, 'partial_r_given_linear_rho': 1.0,
                   'share_of_rho0_to_0p5_change_by_rho0p1': share_by_rho01(seen)})
    write_csv(args.out / 'exploratory_measure_follow.csv', follow)

    flips = {}
    for r in read_csv(args.step6 / 'all_working_points.csv'):
        if r['core'] == 'matched_random' and r['working_point'] == 'A':
            flips[(int(r['source_seed']), int(r['target_seed']))] = int(round(num(r['matched_realized_flip_fraction']) * 4096))
    matched09 = mean_by(e09a, ('source_seed', 'target_seed'), 'accuracy', lambda r: r['condition'] == MATCHED and r['split'] == 'seen_long')
    by_flip = defaultdict(list)
    for key, value in matched09.items():
        by_flip[flips[key]].append(value)
    flip_rows = [{'phasef_matched_flip_count': f, 'realized_rho': f / 4096, 'n_cells_e09a': len(v),
                  'e09a_matched_seen_long_mean': float(np.mean(v))} for f, v in sorted(by_flip.items())]
    flip_rows.append({'phasef_matched_flip_count': 'all', 'n_cells_phasef': len(flips),
                      'cells_by_flip_count': json.dumps({str(f): sum(1 for x in flips.values() if x == f) for f in sorted(set(flips.values()))})})
    write_csv(args.out / 'exploratory_matched_flip_count.csv', flip_rows)

    sym = []
    for name, table in (('bstar_gain_hnav', bstar), ('R_ret_diff_A_hnav', rdiff['A']), ('S_cons_A_hnav', scons['A']),
                        ('R_ret_diff_Bstar_hnav', rdiff['Bstar']), ('downstream_norm_retention', retention),
                        ('downstream_log_drive_drift', drift), ('seen_long', acc['seen_long'])):
        d = [table[(s, 0.9)] - table[(s, 0.1)] for s in SOURCES]
        low, high = bootstrap_ci(d)
        sym.append({'quantity': name, 'rho0p1_mean': float(np.mean([table[(s, 0.1)] for s in SOURCES])),
                    'rho0p9_mean': float(np.mean([table[(s, 0.9)] for s in SOURCES])), 'difference': float(np.mean(d)),
                    'ci_low': low, 'ci_high': high, 'exact_p': exact_sign_flip_p(d)})
    write_csv(args.out / 'exploratory_rho_symmetry.csv', sym)

    write_csv(args.out / 'anchors.csv', anchors)
    write_csv(args.out / 'verdicts.csv', verdict_rows)
    result = {'schema_version': 'mechanism_v1_verdicts_v1', 'rules': 'ANALYSIS_PLAN.md with revision 1 (2026-09-20)',
              'anchors_pass': anchors_pass, 'main_effect': main_effect, 'primary': primary_verdict,
              'verdicts': verdict_rows, 'anchors': anchors, 'validation_all_ok': True}
    (args.out / 'VERDICTS.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    lines = ['# mechanism_v1 verdicts (ANALYSIS_PLAN.md with revision 1)\n',
             f"- anchors: {'pass' if anchors_pass else 'FAIL'} ({sum(a['interval_contains_zero'] for a in anchors)} of 6 intervals contain 0)",
             f"- main effect: {'holds' if main_effect else 'does not hold'} (slope {num(main_row['difference']):+.4f}, "
             f"[{num(main_row['ci_low']):+.4f}, {num(main_row['ci_high']):+.4f}], exact p {num(main_row['exact_p']):.6f})",
             f"- primary R_ret_diff at Bstar: {primary_verdict} (slope {num(prim['within_slope']):+.4f}, "
             f"[{num(prim['ci_low']):+.4f}, {num(prim['ci_high']):+.4f}], permutation p {num(prim['permutation_p']):.5f})",
             f"- validation against the collectors: {len(validation_rows)} quantities, all within {TOL:g}\n",
             '| test | expected | estimate | p | Holm p | verdict |', '|---|---|---|---|---|---|']
    for r in verdict_rows[3:]:
        lines.append(f"| {r['test']} | {r['expected']} | {r['estimate']:+.4f} | {r['p']:.5f} | {r['holm_p']:.5f} | {r['verdict']} |")
    lines += ['', 'Exploratory tables (not part of any verdict): exploratory_*.csv']
    (args.out / 'VERDICTS.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    (args.out / 'COMPLETE.json').write_text(json.dumps({'status': 'complete', 'schema_version': result['schema_version']}), encoding='utf-8')
    print('\n'.join(lines))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
