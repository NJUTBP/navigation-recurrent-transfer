#!/usr/bin/env python3
"""C1: source-side autonomous retention for all E6 rho cores.

This is an exploratory follow-up defined after E6 completed.  It does not
modify ANALYSIS_PLAN.md and it does not retrain any model.

Primary state set
-----------------
Use the *same unperturbed navigation H_nav states* as E5b' (512 held-out
navigation trajectories, burn-in 2, 4096 sampled states with seed
stable_seed32('e5b_nav_states', source)).  For each E6 target, align that state
set only by the E6 permutation P, not by the sign matrix D:

    h_ref = P h_nav
    retention_rho = ||ReLU(W_rho h_ref)|| / ||h_ref||

where W_rho = D_rho P W P^T D_rho.

Why permutation-only alignment is primary: E6 established that rho=0.1 and
rho=0.9 have the same recurrent matrix and differ only in the input sign, yet
trajectory-derived H_nav measures can differ because the two input signs drive
complementary ReLU branches.  A source-side *core* measure used for C1 must not
change merely because the input sign changes.  With the fixed aligned reference
states above, rho=0.1 and rho=0.9 are exactly identical by construction.

Anchors enforced per (source, target):
  1. rho=0 retention equals the original navigation retention on H_nav.
  2. rho=0.1 and rho=0.9 retention are equal (same W_rho, same h_ref).
  3. The reconstructed rho-core metadata matches the completed E6 cell.
"""
from __future__ import annotations
import os

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get('NRT_PROJECT_ROOT', str(Path(__file__).resolve().parents[4])))
MECH_TOOLS = PROJECT_ROOT / 'tools' / 'mechanism_v1'
for sibling in ('E5b_delta_prediction_v1', 'E234_source_measures_v1', 'E6_rho_scan_v1'):
    path = str(MECH_TOOLS / sibling / 'scripts')
    if path not in sys.path:
        sys.path.insert(0, path)
import delta_common  # noqa: E402
import source_measures as sm  # noqa: E402
import rho_core  # noqa: E402

OUTPUT_ROOT = delta_common.MECHANISM_ROOT / 'E6_autonomous_retention'
E6_ROOT = delta_common.MECHANISM_ROOT / 'E6_rho_scan'
E5B_ROOT = delta_common.MECHANISM_ROOT / 'E5b_delta_prediction'
SCHEMA_VERSION = 'e6_autonomous_retention_v1'
ANCHOR_TOL = 2e-6


def read_csv(path: Path):
    with path.open('r', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def nav_checkpoint(source: int) -> Path:
    return PROJECT_ROOT / 'sources' / f'nav_seed_{source}' / 'artifacts' / 'navigation' / 'best.pt'


def e6_cell(root: Path, source: int, target: int, rho: float) -> Path:
    return root / f'source_{source}' / f'target_{target}' / f'rho_{int(round(rho * 1000)):04d}'


def aligned_reference_states(states, permutation):
    import torch
    index = torch.as_tensor(permutation, dtype=torch.long, device=states.device)
    return states.index_select(-1, index).contiguous()


def relative_deviation(a: float, b: float) -> float:
    return abs(float(a) - float(b)) / max(abs(float(a)), abs(float(b)), 1e-12)


def discover_e6_cells(e6_root: Path):
    rows = read_csv(e6_root / 'aggregate' / 'all_condition_diagnostics.csv')
    cells = sorted({(int(r['source_seed']), int(r['target_seed']), float(r['rho'])) for r in rows})
    return rows, cells


def preflight(args) -> int:
    import torch
    fullscale, _ = delta_common.load_phasef_modules()
    hist = fullscale.import_historical()
    diagnostics, cells = discover_e6_cells(Path(args.e6_root))
    expected_rhos = tuple(float(r) for r in rho_core.RHO_GRID)
    sources = sorted({s for s, _, _ in cells})
    targets = sorted({t for _, t, _ in cells})
    pairs = sorted({(s, t) for s, t, _ in cells})
    if len(cells) != 210 or len(sources) != 10 or len(targets) != 3 or len(pairs) != 30:
        raise RuntimeError(f'unexpected E6 coverage: cells={len(cells)} sources={len(sources)} targets={len(targets)} pairs={len(pairs)}')
    for pair in pairs:
        observed = tuple(sorted(r for s, t, r in cells if (s, t) == pair))
        if observed != expected_rhos:
            raise RuntimeError(f'rho grid mismatch for {pair}: {observed}')
    missing = [str(e6_cell(Path(args.e6_root), s, t, r) / 'COMPLETE.json')
               for s, t, r in cells if not (e6_cell(Path(args.e6_root), s, t, r) / 'COMPLETE.json').is_file()]
    if missing:
        raise RuntimeError(f'missing E6 COMPLETE cells: {len(missing)}; first={missing[0]}')
    e5b_ret = read_csv(Path(args.e5b_root) / 'aggregate' / 'all_retention.csv')
    e5b_nav = {(int(r['source_seed']), int(r['target_seed'])): float(r['autonomous_retention_mean'])
               for r in e5b_ret if r['state_set'] == 'H_nav' and r['origin'] == 'navigation'}
    if len(e5b_nav) != 50:
        raise RuntimeError(f'expected 50 E5b H_nav navigation rows, found {len(e5b_nav)}')
    for source in sources:
        values = [v for (s, _), v in e5b_nav.items() if s == source]
        if max(values) - min(values) > 1e-10:
            raise RuntimeError(f'E5b navigation retention unexpectedly target-dependent for source {source}')
    nav_hh, nav_ih, _ = hist['load_navigation_grid_weights'](nav_checkpoint(sources[0]))
    seed = fullscale.stable_seed32('e6_rho_core', targets[0])
    anchors = rho_core.anchor_report(nav_hh, nav_ih, seed=seed)
    if not anchors['all_pass']:
        raise RuntimeError('rho_core anchor_report failed')
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    plan = out_root / 'CELL_PLAN.csv'
    with plan.with_suffix('.csv.tmp').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['source_seed', 'target_seed'])
        writer.writeheader()
        for source, target in pairs:
            writer.writerow({'source_seed': source, 'target_seed': target})
    plan.with_suffix('.csv.tmp').replace(plan)
    report = {'schema_version': SCHEMA_VERSION, 'status': 'preflight_ok', 'torch': torch.__version__,
              'n_e6_cells': len(cells), 'n_pairs': len(pairs), 'sources': sources, 'targets_from_e6_artifact': targets,
              'rho_grid_from_code': list(expected_rhos), 'e5b_h_nav_navigation_rows': len(e5b_nav),
              'rho_core_anchor_report': anchors, 'cell_plan': str(plan)}
    print(json.dumps(report, indent=2))
    return 0


def run_pair(args) -> int:
    import torch
    fullscale, _ = delta_common.load_phasef_modules()
    hist = fullscale.import_historical()
    source, target = int(args.source_seed), int(args.target_seed)
    out = Path(args.output_root) / f'source_{source}' / f'target_{target}'
    if (out / 'COMPLETE.json').is_file():
        print('already complete', flush=True)
        return 0
    if out.exists() and any(out.iterdir()):
        raise RuntimeError('partial output exists for this pair; inspect before rerun')
    out.mkdir(parents=True, exist_ok=True)

    diagnostics, cells = discover_e6_cells(Path(args.e6_root))
    rhos = tuple(sorted(r for s, t, r in cells if s == source and t == target))
    expected_rhos = tuple(float(r) for r in rho_core.RHO_GRID)
    if rhos != expected_rhos:
        raise RuntimeError(f'E6 artifact does not define the expected rho grid for ({source}, {target}): {rhos}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = nav_checkpoint(source)
    nav_hh_cpu, nav_ih_cpu, nav_meta = hist['load_navigation_grid_weights'](checkpoint)
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    nav_model = delta_common.build_navigation_model(payload, device)
    velocities, init_place = delta_common.navigation_trajectories(payload['config'], delta_common.NAV_TRAJECTORIES)
    nav_hh = nav_hh_cpu.to(device)
    nav_ih = nav_ih_cpu.to(device)
    states = sm.navigation_states(
        nav_model, {'navigation': (nav_hh, nav_ih)}, velocities.to(device), init_place.to(device),
        delta_common.MAX_NAV_STATES, fullscale.stable_seed32('e5b_nav_states', source))['navigation']
    nav_ret = delta_common.autonomous_retention(nav_hh, states)
    core_seed = fullscale.stable_seed32('e6_rho_core', target)
    rows = []
    for rho in rhos:
        cell = e6_cell(Path(args.e6_root), source, target, rho)
        meta_e6 = json.loads((cell / 'COMPLETE.json').read_text(encoding='utf-8'))
        rho_hh_cpu, rho_ih_cpu, core_meta = rho_core.signed_similarity_core(
            nav_hh_cpu, nav_ih_cpu, seed=core_seed, rho=rho)
        if int(meta_e6['core_seed']) != int(core_meta['core_seed']):
            raise RuntimeError(f'core_seed mismatch at rho={rho}')
        if int(meta_e6['flip_count']) != int(core_meta['flip_count']):
            raise RuntimeError(f'flip_count mismatch at rho={rho}')
        if abs(float(meta_e6['realized_rho']) - float(core_meta['realized_rho'])) > 1e-12:
            raise RuntimeError(f'realized_rho mismatch at rho={rho}')
        if meta_e6.get('navigation_checkpoint_sha256') != nav_meta['checkpoint_sha256']:
            raise RuntimeError(f'navigation checkpoint hash mismatch at rho={rho}')
        permutation, _ = rho_core.flip_units(int(nav_hh_cpu.shape[0]), rho, core_seed)
        aligned = aligned_reference_states(states, permutation)
        ret = delta_common.autonomous_retention(rho_hh_cpu.to(device), aligned)
        rows.append({'schema_version': SCHEMA_VERSION, 'source_seed': source, 'target_seed': target,
                     'rho': rho, 'flip_count': core_meta['flip_count'], 'realized_rho': core_meta['realized_rho'],
                     'core_seed': int(core_seed), 'state_set': 'H_nav_reference_permuted',
                     'state_alignment': 'permutation_only_input_sign_invariant',
                     **ret, 'navigation_reference_retention_mean': nav_ret['autonomous_retention_mean'],
                     'navigation_reference_retention_median': nav_ret['autonomous_retention_median']})
    row0 = next(r for r in rows if abs(float(r['rho'])) < 1e-12)
    row01 = next(r for r in rows if abs(float(r['rho']) - 0.1) < 1e-12)
    row09 = next(r for r in rows if abs(float(r['rho']) - 0.9) < 1e-12)
    anchor_rho0 = relative_deviation(row0['autonomous_retention_mean'], nav_ret['autonomous_retention_mean'])
    anchor_complement = relative_deviation(row01['autonomous_retention_mean'], row09['autonomous_retention_mean'])
    if anchor_rho0 > ANCHOR_TOL:
        raise RuntimeError(f'rho=0 retention anchor failed: relative deviation {anchor_rho0:.3e}')
    if anchor_complement > ANCHOR_TOL:
        raise RuntimeError(f'rho=0.1/0.9 input-sign invariance anchor failed: relative deviation {anchor_complement:.3e}')
    fullscale.write_csv(out / 'retention.csv', rows)
    fullscale.atomic_json(out / 'COMPLETE.json', {
        'schema_version': SCHEMA_VERSION, 'status': 'complete', 'source_seed': source, 'target_seed': target,
        'n_rhos': len(rows), 'rho_grid': list(rhos), 'n_states': int(states.shape[0]),
        'navigation_checkpoint_sha256': nav_meta['checkpoint_sha256'], 'core_seed': int(core_seed),
        'rho0_navigation_relative_deviation': anchor_rho0,
        'rho01_rho09_relative_deviation': anchor_complement,
        'state_set': 'H_nav_reference_permuted', 'state_alignment': 'permutation_only_input_sign_invariant'})
    print(source, target, 'retention rho0', round(row0['autonomous_retention_mean'], 6),
          'rho0.1', round(row01['autonomous_retention_mean'], 6),
          'rho0.5', round(next(r for r in rows if abs(float(r['rho']) - 0.5) < 1e-12)['autonomous_retention_mean'], 6),
          flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description='C1 E6 source-side autonomous retention')
    parser.add_argument('--source-seed', type=int)
    parser.add_argument('--target-seed', type=int)
    parser.add_argument('--output-root', type=Path, default=OUTPUT_ROOT)
    parser.add_argument('--e6-root', type=Path, default=E6_ROOT)
    parser.add_argument('--e5b-root', type=Path, default=E5B_ROOT)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    if args.preflight:
        return preflight(args)
    if args.source_seed is None or args.target_seed is None:
        parser.error('--source-seed and --target-seed are required unless --preflight is used')
    return run_pair(args)


if __name__ == '__main__':
    raise SystemExit(main())
