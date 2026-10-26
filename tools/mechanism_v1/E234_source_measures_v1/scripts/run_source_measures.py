#!/usr/bin/env python3
"""Step 6 driver: E2, E3 and E4 on H_nav for one (source, target) cell.

    "$PYTHON_BIN" run_source_measures.py --preflight
    "$PYTHON_BIN" run_source_measures.py --source-seed 6101 --target-seed 84001

Cores: the navigation core of the source and the matched core of the target,
built exactly as in Phase F.  Both run on the same H_nav positions, the same
relation sequences and the same fit/test split, so every readout is paired.

Working points, each for both cores:
    A        g = 1, |u| = 0.02 m
    g3, g10  the two extra gains named in plan v3/v4
    B8       the learned gain of this cell and core in Phase F (8,000 updates)
    B32      the same in E0.9a (32,000 updates), targets 84001-84003 only
    Bstar    the gain at which delta on this core's H_nav states equals the
             delta this cell and core showed downstream at step 1 (E0.4)
Transformed cores start H_nav from Q h0 (see source_measures.navigation_states);
the matched core's Q is rebuilt from its recorded seed and fraction and must
reproduce its weights bit for bit.  The realized flip fraction of the matched
control is recorded: for target 84001 it is 423/4096 = 0.103, not 0.5.

B8 and B32 follow the plan literally.  Bstar applies the homogeneity result:
the dynamics depend on g/|h|, and H_nav states are far smaller than the learned
downstream start states, so the literal B sits 15 times (navigation) to 370
times (matched) above the downstream relative drive.  All points are always
computed; which one is primary is fixed in ANALYSIS_PLAN.md, not here.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

import source_measures as sm
from source_measures import delta_common

OUTPUT_ROOT = delta_common.MECHANISM_ROOT / 'E234_source_measures'
E067_STATES = delta_common.MECHANISM_ROOT / 'E0_67_forward_diagnostics' / 'aggregate' / 'all_state_diagnostics.csv'
SCHEMA_VERSION = 'e234_source_measures_v1'
E09A_TARGETS = (84001, 84002, 84003)
CONDITION = {'navigation': 'navigation_trainable_scalar_gain',
             'matched_random': 'matched_random_trainable_scalar_gain'}

def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open('r', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))

def learned_gain(root: Path, group: str, source: int, target: int, origin: str) -> float:
    for row in read_csv(root / f'source_{source}' / f'target_{target}' / group / 'condition_diagnostics.csv'):
        if row.get('condition') == CONDITION[origin] and row.get('gain') not in (None, ''):
            return float(row['gain'])
    return float('nan')

def downstream_delta(source: int, target: int, origin: str) -> float:
    for row in read_csv(E067_STATES):
        if (int(row['source_seed']) == source and int(row['target_seed']) == target
                and row['condition'] == CONDITION[origin] and int(row['path_step']) == 1):
            return float(row['delta_rectified'])
    return float('nan')

def nav_checkpoint(source: int) -> Path:
    return delta_common.PROJECT_ROOT / 'sources' / f'nav_seed_{source}' / 'artifacts' / 'navigation' / 'best.pt'

def build_cores(fullscale, hist, source, target):
    nav_hh, nav_ih, nav_meta = hist['load_navigation_grid_weights'](nav_checkpoint(source))
    matched_hh, matched_ih, diagnostics = hist['make_matched_random_weights'](
        nav_hh, nav_ih, seed=fullscale.stable_seed32('matched_random', target),
        iterations=8, singular_rank=16, probe_count=64, strict=True)
    permutation, flipped, fraction = sm.matched_transform(nav_hh, nav_ih, matched_hh, matched_ih, diagnostics)
    cores = {'navigation': (nav_hh, nav_ih), 'matched_random': (matched_hh, matched_ih)}
    return cores, {'matched_random': (permutation, flipped)}, nav_meta, fraction

def preflight() -> int:
    import torch
    fullscale, _ = delta_common.load_phasef_modules()
    fullscale.import_historical()
    protocol = delta_common.load_protocol()
    report: dict[str, Any] = {'schema_version': SCHEMA_VERSION}
    missing = [s for s in protocol['all_sources'] if not nav_checkpoint(s).is_file()]
    if missing:
        raise RuntimeError(f'missing navigation checkpoints: {missing}')
    report['e067_state_diagnostics_present'] = E067_STATES.is_file()
    if not E067_STATES.is_file():
        raise RuntimeError('E0.4 state diagnostics are needed for Bstar')
    gains = {origin: learned_gain(delta_common.PHASEF_OUTPUT, 'group_learned_interface', 6101, 84001, origin)
             for origin in CONDITION}
    report['phasef_gain_example_6101_84001'] = gains
    if not all(np.isfinite(v) for v in gains.values()):
        raise RuntimeError('Phase F learned gains not readable')
    report['e09a_gain_example_6101_84001'] = {
        origin: learned_gain(delta_common.E09A_ROOT, 'group_scalar_pair', 6101, 84001, origin)
        for origin in CONDITION}
    report['downstream_delta_example_6101_84001'] = {
        origin: downstream_delta(6101, 84001, origin) for origin in CONDITION}
    report['torch'] = torch.__version__
    report['status'] = 'preflight_ok'
    print(json.dumps(report, indent=2))
    return 0

def run_cell(args) -> int:
    import torch
    fullscale, _ = delta_common.load_phasef_modules()
    protocol = delta_common.load_protocol()
    source, target = int(args.source_seed), int(args.target_seed)
    if source not in protocol['all_sources'] or target not in protocol['target_split_seeds']:
        raise ValueError('source or target outside the Phase F protocol')
    out = Path(args.output_root) / f'source_{source}' / f'target_{target}'
    if (out / 'COMPLETE.json').exists():
        print('already complete', flush=True)
        return 0
    if out.exists() and any(out.iterdir()):
        raise RuntimeError('partial output exists for this cell; inspect before rerun')
    out.mkdir(parents=True, exist_ok=True)

    hist = fullscale.import_historical()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cores, transforms, nav_meta, matched_fraction = build_cores(fullscale, hist, source, target)
    cores = {k: (hh.to(device), ih.to(device)) for k, (hh, ih) in cores.items()}
    payload = torch.load(nav_checkpoint(source), map_location='cpu', weights_only=False)
    nav_model = delta_common.build_navigation_model(payload, device)
    velocities, init_place = delta_common.navigation_trajectories(payload['config'], int(args.nav_trajectories))
    states = sm.navigation_states(nav_model, cores, velocities.to(device), init_place.to(device),
                                  int(args.n_states), fullscale.stable_seed32('e234_nav_states', source), transforms)
    del nav_model

    pools_nodes, manifest = fullscale.build_fullscale_split(protocol, target)
    if manifest['manifest_hash'] != protocol['expected_manifest_hashes'][str(target)]:
        raise RuntimeError('manifest hash mismatch')
    pools = {k: fullscale.to_historical_episodes(v, hist) for k, v in pools_nodes.items()}
    half = int(args.n_states) // 2
    rng = np.random.default_rng(fullscale.stable_seed32('e234_sequences', source, target))
    fit_seqs = sm.relation_sequences(pools, ('train_8000',), range(1, sm.FIT_MAX_L + 1), half, rng)
    test_seqs = {**sm.relation_sequences(pools, ('id_seen_short',), range(1, sm.FIT_MAX_L + 1), half, rng),
                 **sm.relation_sequences(pools, ('seen_long',), range(sm.FIT_MAX_L + 1, sm.MAX_L + 1), half, rng)}
    relations = delta_common.relation_vectors(protocol, device)
    scale = float(protocol['velocity_scale'])

    tag = {'source_seed': source, 'target_seed': target}
    points, geometry, composition, by_length, summary, contribution, curves = [], [], [], [], [], [], []
    for origin, (w_hh, w_in) in cores.items():
        h = states[origin]
        curve = sm.delta_curve(w_hh, w_in, h, relations)
        curves += [{**tag, 'core': origin, 'gain': g, 'delta': d} for g, d in curve]
        target_delta = downstream_delta(source, target, origin)
        g_star, how = sm.matching_gain(curve, target_delta) if np.isfinite(target_delta) else (float('nan'), 'no_downstream_delta')
        wanted = [('A', 1.0, 'plan'), ('g3', 3.0, 'plan'), ('g10', 10.0, 'plan'),
                  ('B8', learned_gain(delta_common.PHASEF_OUTPUT, 'group_learned_interface', source, target, origin), 'phasef_learned_gain'),
                  ('B32', learned_gain(delta_common.E09A_ROOT, 'group_scalar_pair', source, target, origin)
                   if target in E09A_TARGETS else float('nan'), 'e09a_learned_gain'),
                  ('Bstar', g_star, how)]
        for name, gain, origin_of_value in wanted:
            points.append({**tag, 'core': origin, 'working_point': name, 'gain': gain,
                           'value_from': origin_of_value, 'downstream_delta_step1': target_delta,
                           'matched_realized_flip_fraction': matched_fraction})
            if not np.isfinite(gain) or gain <= 0:
                continue
            key = {**tag, 'core': origin, 'working_point': name, 'gain': gain}
            geometry.append({**key, **sm.one_step_geometry(
                w_hh, w_in, h, relations, gain, fullscale.stable_seed32('e234_null', source, target, name))})
            composition.append({**key, **sm.composition(w_hh, w_in, h, relations, gain)})
            s_row, l_rows, c_rows = sm.horizon_retention(
                w_hh, w_in, h[:half], h[half:], fit_seqs, test_seqs, gain, scale,
                fullscale.stable_seed32('e234_ridge', source, target, name))
            summary.append({**key, **s_row})
            by_length += [{**key, **r} for r in l_rows]
            contribution += [{**key, **r} for r in c_rows]
        primary = [r for r in summary if r['core'] == origin and r['working_point'] == 'A']
        print(source, target, origin, 'R_ret(A)', round(primary[0]['R_ret'], 4) if primary else 'nan', flush=True)

    for name, rows in (('working_points.csv', points), ('geometry.csv', geometry), ('composition.csv', composition),
                       ('horizon_by_length.csv', by_length), ('horizon_summary.csv', summary),
                       ('contribution.csv', contribution), ('delta_curve.csv', curves)):
        fullscale.write_csv(out / name, rows)
    fullscale.atomic_json(out / 'COMPLETE.json', {
        'schema_version': SCHEMA_VERSION, 'status': 'complete', **tag,
        'navigation_checkpoint_sha256': nav_meta['checkpoint_sha256'],
        'matched_random_seed': int(fullscale.stable_seed32('matched_random', target)),
        'matched_realized_flip_fraction': matched_fraction,
        'matched_start': 'Q h0 with Q rebuilt exactly from the recorded construction',
        'matched_diagnostics_is_audit_selected_candidate': True,
        'n_states': int(args.n_states), 'nav_trajectories': int(args.nav_trajectories),
        'dataset_manifest_hash': manifest['manifest_hash']})
    return 0

def main() -> int:
    parser = argparse.ArgumentParser(description='Step 6: E2/E3/E4 source-side measures')
    parser.add_argument('--source-seed', type=int)
    parser.add_argument('--target-seed', type=int)
    parser.add_argument('--output-root', type=Path, default=OUTPUT_ROOT)
    parser.add_argument('--n-states', type=int, default=sm.N_STATES)
    parser.add_argument('--nav-trajectories', type=int, default=sm.NAV_TRAJECTORIES)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    if args.preflight:
        return preflight()
    if args.source_seed is None or args.target_seed is None:
        parser.error('--source-seed and --target-seed are required')
    return run_cell(args)

if __name__ == '__main__':
    sys.exit(main())
