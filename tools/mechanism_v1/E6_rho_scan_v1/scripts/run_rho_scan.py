#!/usr/bin/env python3
"""E6 driver: one (source, target, rho) cell of the sign-mixing scan.

    "$PYTHON_BIN" run_rho_scan.py --preflight
    "$PYTHON_BIN" run_rho_scan.py --source-seed 6101 --target-seed 84001 --rho 0

Training follows E0.9a exactly: its derived protocol (32,000 updates, the
Phase F seeds and minibatch order), its GainTrace, and the Phase F wiring in
which a learned scalar gain feeds a frozen core.  The rho-core enters through
the matched-core slot of configure_origin, so nothing in the trainer changes.

After training the cell also records, for the same core:
  - the downstream state profile (norm, delta and active fraction by step)
    at the learned operating point, as in E0.4;
  - E2, E3 and E4 on H_nav at A, g3, g10, B (this cell's learned gain) and
    Bstar (delta-matched to this cell's step-1 downstream delta), with the
    H_nav positions and relation sequences of Step 6, so rho = 0 must
    reproduce the Step 6 navigation values up to floating-point order.
"""
from __future__ import annotations
import os

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(os.environ.get('NRT_PROJECT_ROOT', str(Path(__file__).resolve().parents[4])))
MECH_TOOLS = PROJECT_ROOT / 'tools' / 'mechanism_v1'
for sibling in ('E234_source_measures_v1', 'E0_9a_extended_budget_v1'):
    path = str(MECH_TOOLS / sibling / 'scripts')
    if path not in sys.path:
        sys.path.insert(0, path)
import extended_common  # noqa: E402
import source_measures as sm  # noqa: E402
from source_measures import delta_common  # noqa: E402
import rho_core  # noqa: E402

OUTPUT_ROOT = delta_common.MECHANISM_ROOT / 'E6_rho_scan'
SCHEMA_VERSION = 'e6_rho_scan_v1'
CONDITION = 'rho_core_trainable_scalar_gain'
SPEC = {'origin': 'matched_random', 'kind': 'scalar', 'core': 'frozen'}
PROFILE_EPISODES = 512

def cell_dir(root: Path, source: int, target: int, rho: float) -> Path:
    return root / f'source_{source}' / f'target_{target}' / f'rho_{int(round(rho * 1000)):04d}'

def derive_protocol(updates: int) -> dict[str, Any]:
    protocol = extended_common.derive_protocol(updates=updates)
    protocol['schema_version'] = SCHEMA_VERSION
    protocol['status'] = 'sign_mixing_scan'
    protocol['output_root'] = str(OUTPUT_ROOT)
    protocol['conditions_by_group'] = {'rho_scan': [CONDITION]}
    protocol['rho_grid'] = list(rho_core.RHO_GRID)
    return protocol

def nav_checkpoint(source: int) -> Path:
    return PROJECT_ROOT / 'sources' / f'nav_seed_{source}' / 'artifacts' / 'navigation' / 'best.pt'

def preflight() -> int:
    import torch
    fullscale, phasef = extended_common.load_phasef_modules()
    hist = fullscale.import_historical()
    protocol = derive_protocol(extended_common.EXTENDED_UPDATES)
    report: dict[str, Any] = {'schema_version': SCHEMA_VERSION, 'updates': protocol['train']['max_gradient_updates'],
                              'targets': protocol['target_split_seeds'], 'rho_grid': list(rho_core.RHO_GRID)}
    missing = [s for s in protocol['all_sources'] if not nav_checkpoint(s).is_file()]
    if missing:
        raise RuntimeError(f'missing navigation checkpoints: {missing}')
    nav_hh, nav_ih, _ = hist['load_navigation_grid_weights'](nav_checkpoint(protocol['all_sources'][0]))
    anchors = rho_core.anchor_report(nav_hh, nav_ih, seed=fullscale.stable_seed32('e6_rho_core', 84001))
    report['anchors_source_6101_target_84001'] = anchors
    if not anchors['all_pass']:
        raise RuntimeError('rho-core anchor checks failed')
    report['n_cells'] = len(protocol['all_sources']) * len(protocol['target_split_seeds']) * len(rho_core.RHO_GRID)
    report['torch'] = torch.__version__
    report['status'] = 'preflight_ok'
    print(json.dumps(report, indent=2))
    return 0

def downstream_profile(model, pools, hist, fullscale, loader_seed, w_hh, device):
    import torch
    episodes = pools['seen_long'][:PROFILE_EPISODES]
    loader = fullscale.make_loader(episodes, hist, 128, False, loader_seed + 2)
    hidden, first, lengths = [], [], []
    with torch.no_grad():
        for batch in loader:
            out = model(start_node=batch['start_node'].to(device), velocities=batch['velocities'].to(device),
                        lengths=batch['lengths'].to(device), start_coords=batch['start_coords'].to(device),
                        validate_inputs=False)
            pad = sm.MAX_L - out['hidden_seq'].shape[1]
            hidden.append(torch.nn.functional.pad(out['hidden_seq'], (0, 0, 0, pad)))
            first.append(out['h0'])
            lengths.append(batch['lengths'].to(device))
    return delta_common.trajectory_profile(torch.cat(hidden), torch.cat(first), w_hh, torch.cat(lengths))

def run_cell(args) -> int:
    import torch
    fullscale, phasef = extended_common.load_phasef_modules()
    protocol = derive_protocol(int(args.updates))
    source, target, rho = int(args.source_seed), int(args.target_seed), float(args.rho)
    if source not in protocol['all_sources'] or target not in protocol['target_split_seeds']:
        raise ValueError('source or target outside the E6 protocol')
    if not any(abs(rho - r) < 1e-9 for r in rho_core.RHO_GRID):
        raise ValueError('rho outside the pre-registered grid')
    out = cell_dir(Path(args.output_root), source, target, rho)
    if (out / 'COMPLETE.json').exists():
        print('already complete', flush=True)
        return 0
    if out.exists() and any(out.iterdir()):
        raise RuntimeError('partial output exists for this cell; inspect before rerun')
    out.mkdir(parents=True, exist_ok=True)
    fullscale.atomic_json(out / 'protocol_used.json', protocol)

    hist = fullscale.import_historical()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pools_nodes, manifest = fullscale.build_fullscale_split(protocol, target)
    if manifest['manifest_hash'] != protocol['expected_manifest_hashes'][str(target)]:
        raise RuntimeError('manifest hash mismatch')
    pools = {k: fullscale.to_historical_episodes(v, hist) for k, v in pools_nodes.items()}
    for name, episodes in pools.items():
        if [hist['episode_fingerprint'](e) for e in episodes] != manifest['fingerprints'][name]:
            raise RuntimeError(f'historical fingerprint mismatch: {name}')

    nav_hh, nav_ih, nav_meta = hist['load_navigation_grid_weights'](nav_checkpoint(source))
    core_seed = fullscale.stable_seed32('e6_rho_core', target)
    rho_hh, rho_ih, core_meta = rho_core.signed_similarity_core(nav_hh, nav_ih, seed=core_seed, rho=rho)
    if abs(float(rho_hh.double().norm()) - float(nav_hh.double().norm())) > 1e-6 * float(nav_hh.double().norm()):
        raise RuntimeError('rho-core changed a linear invariant')

    model_cfg, train_cfg = protocol['model'], protocol['train']
    batch_size = int(protocol['batch_size'])
    run_seed = fullscale.stable_seed32('phaseF_run', target, protocol['training_examples'])
    loader_seed = fullscale.stable_seed32('phaseF_loader', target, protocol['training_examples'])
    random_core_seed = fullscale.stable_seed32('phaseF_fresh_random_core', source, target)
    hist['set_seed'](run_seed)
    base = hist['build_hard_model'](model_cfg, n_nodes=64, device=device)
    phasef.configure_origin(base, SPEC, hist, nav_hh=nav_hh, nav_ih=nav_ih, matched_hh=rho_hh, matched_ih=rho_ih,
                            cell_cls=hist['FactorialCell'], model_cfg=model_cfg, device=device,
                            random_core_seed=random_core_seed)
    if not torch.equal(base.grid_rnn.cell.h_proj.weight.detach().cpu(), rho_hh.to(torch.float32)):
        raise RuntimeError('rho-core was not installed exactly')
    model = phasef.wrap_model(base, SPEC, hist, device)
    hist['set_seed'](run_seed)
    if any(p.requires_grad for p in model.grid_rnn.parameters()):
        raise RuntimeError('grid unexpectedly trainable')
    if abs(float(model.gain().detach().cpu()) - 1.0) > 1e-8:
        raise RuntimeError('scalar did not start at unit gain')

    tracer = extended_common.GainTrace(model)
    tr_loader = fullscale.make_loader(pools['train_8000'], hist, batch_size, True, loader_seed)
    va_loader = fullscale.make_loader(pools['validation_seen_short'], hist, batch_size, False, loader_seed + 1)
    training = hist['train_fixed_updates'](model, tr_loader, va_loader, device, train_cfg, seed=run_seed)
    if training.stopped_update != int(train_cfg['max_gradient_updates']):
        raise RuntimeError(f'fixed update contract violated: {training.stopped_update}')
    model.load_state_dict(training.state_dict, strict=True)
    tag = {'source_seed': source, 'target_seed': target, 'rho': rho, 'condition': CONDITION}
    gain_trace = [{**tag, **record} for record in tracer.close(final_model=model)]

    metrics, per_length = [], []
    for split in ('train_8000', 'validation_seen_short', 'id_seen_short', 'heldout_short', 'seen_long', 'heldout_long'):
        loader = fullscale.make_loader(pools[split], hist, batch_size, False, loader_seed + 2)
        summary, _ = hist['evaluate_loader_detailed'](model, loader, device, split_name=split, calibration_bins=15)
        metrics.append({**tag, 'schema_version': SCHEMA_VERSION, 'split': split, 'accuracy': summary['accuracy'],
                        'nll': summary['nll'], 'ece': summary['ece'], 'mean_confidence': summary['mean_confidence'],
                        'mean_readout_margin': summary['mean_readout_margin'], 'best_update': training.best_update,
                        'stopped_update': training.stopped_update, 'run_seed': run_seed, 'loader_seed': loader_seed,
                        'dataset_manifest_hash': manifest['manifest_hash']})
        per_length += [{**tag, 'split': split, 'path_length': int(k), 'accuracy': float(v)}
                       for k, v in summary['accuracy_by_path_length'].items()]
    learned = float(model.gain().detach().cpu())
    diagnostics = [{**tag, **core_meta, 'learned_gain': learned, 'log_gain': float(model.log_gain.detach().cpu()),
                    'best_update': training.best_update}]

    w_hh = base.grid_rnn.cell.h_proj.weight.detach()
    w_in = base.grid_rnn.cell.in_proj.weight.detach()
    model.eval()
    profile = [{**tag, **row} for row in downstream_profile(model, pools, hist, fullscale, loader_seed, w_hh, device)]
    step1 = next(r['delta_rectified'] for r in profile if r['step'] == 1)

    payload = torch.load(nav_checkpoint(source), map_location='cpu', weights_only=False)
    nav_model = delta_common.build_navigation_model(payload, device)
    velocities, init_place = delta_common.navigation_trajectories(payload['config'], int(args.nav_trajectories))
    transform = rho_core.flip_units(int(nav_hh.shape[0]), rho, core_seed)
    states = sm.navigation_states(nav_model, {'rho_core': (w_hh, w_in)}, velocities.to(device), init_place.to(device),
                                  int(args.n_states), fullscale.stable_seed32('e234_nav_states', source),
                                  {'rho_core': transform})['rho_core']
    del nav_model
    half = int(args.n_states) // 2
    rng = np.random.default_rng(fullscale.stable_seed32('e234_sequences', source, target))
    fit_seqs = sm.relation_sequences(pools, ('train_8000',), range(1, sm.FIT_MAX_L + 1), half, rng)
    test_seqs = {**sm.relation_sequences(pools, ('id_seen_short',), range(1, sm.FIT_MAX_L + 1), half, rng),
                 **sm.relation_sequences(pools, ('seen_long',), range(sm.FIT_MAX_L + 1, sm.MAX_L + 1), half, rng)}
    relations = delta_common.relation_vectors(protocol, device)
    curve = sm.delta_curve(w_hh, w_in, states, relations)
    g_star, how = sm.matching_gain(curve, step1)
    points, geometry, composition, summary_rows, by_length, contribution = [], [], [], [], [], []
    for name, gain, source_of_value in (('A', 1.0, 'plan'), ('g3', 3.0, 'plan'), ('g10', 10.0, 'plan'),
                                        ('B', learned, 'this_cell_learned_gain'), ('Bstar', g_star, how)):
        points.append({**tag, 'working_point': name, 'gain': gain, 'value_from': source_of_value,
                       'downstream_delta_step1': step1})
        if not np.isfinite(gain) or gain <= 0:
            continue
        key = {**tag, 'working_point': name, 'gain': gain}
        geometry.append({**key, **sm.one_step_geometry(w_hh, w_in, states, relations, gain,
                                                       fullscale.stable_seed32('e234_null', source, target, name))})
        composition.append({**key, **sm.composition(w_hh, w_in, states, relations, gain)})
        s_row, l_rows, c_rows = sm.horizon_retention(w_hh, w_in, states[:half], states[half:], fit_seqs, test_seqs,
                                                     gain, float(protocol['velocity_scale']),
                                                     fullscale.stable_seed32('e234_ridge', source, target, name))
        summary_rows.append({**key, **s_row})
        by_length += [{**key, **r} for r in l_rows]
        contribution += [{**key, **r} for r in c_rows]

    for name, rows in (('metrics.csv', metrics), ('per_length.csv', per_length),
                       ('update_curves.csv', [{**tag, **h} for h in training.history]), ('gain_trace.csv', gain_trace),
                       ('condition_diagnostics.csv', diagnostics), ('downstream_profile.csv', profile),
                       ('working_points.csv', points), ('geometry.csv', geometry), ('composition.csv', composition),
                       ('horizon_summary.csv', summary_rows), ('horizon_by_length.csv', by_length),
                       ('contribution.csv', contribution)):
        fullscale.write_csv(out / name, rows)
    state, storage_mode = phasef.compact_state_dict(model, False)
    checkpoint_dir = out / 'checkpoints' / CONDITION
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save({'schema_version': SCHEMA_VERSION, 'model_state': state, 'checkpoint_storage_mode': storage_mode,
                'core_reconstruction': {'rho': rho, 'core_seed': int(core_seed), 'constructor': 'rho_core.signed_similarity_core'},
                'source_seed': source, 'target_seed': target, 'run_seed': run_seed, 'loader_seed': loader_seed,
                'best_update': training.best_update}, checkpoint_dir / 'best_validation.pt')
    acc = {m['split']: m['accuracy'] for m in metrics}
    print(source, target, rho, 'gain', round(learned, 2), 'ID', round(acc['id_seen_short'], 4),
          'long', round(acc['seen_long'], 4), flush=True)
    fullscale.atomic_json(out / 'COMPLETE.json', {
        'schema_version': SCHEMA_VERSION, 'status': 'complete', **tag, **core_meta,
        'navigation_checkpoint_sha256': nav_meta['checkpoint_sha256'], 'learned_gain': learned,
        'downstream_delta_step1': step1, 'bstar_gain': g_star, 'dataset_manifest_hash': manifest['manifest_hash']})
    return 0

def main() -> int:
    parser = argparse.ArgumentParser(description='E6 sign-mixing scan')
    parser.add_argument('--source-seed', type=int)
    parser.add_argument('--target-seed', type=int)
    parser.add_argument('--rho', type=float)
    parser.add_argument('--updates', type=int, default=extended_common.EXTENDED_UPDATES)
    parser.add_argument('--output-root', type=Path, default=OUTPUT_ROOT)
    parser.add_argument('--n-states', type=int, default=sm.N_STATES)
    parser.add_argument('--nav-trajectories', type=int, default=sm.NAV_TRAJECTORIES)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    if args.preflight:
        return preflight()
    if args.source_seed is None or args.target_seed is None or args.rho is None:
        parser.error('--source-seed, --target-seed and --rho are required')
    return run_cell(args)

if __name__ == '__main__':
    sys.exit(main())
