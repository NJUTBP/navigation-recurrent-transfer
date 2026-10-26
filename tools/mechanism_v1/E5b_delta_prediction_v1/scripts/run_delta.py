#!/usr/bin/env python3
"""E5b' driver: measure delta and state retention for one (source, target) cell.

    "$PYTHON_BIN" run_delta.py --preflight
    "$PYTHON_BIN" run_delta.py --source-seed 6101 --target-seed 84001

No training.  The matched core depends on the target (its seed is
stable_seed32('matched_random', target)), the navigation core on the source,
and the H_init start states on the target (run_seed).  Hence one cell per
(source, target).  Installation of each core is checked element by element
against the reconstructed weights, the same construction Gate R verified.

The H_init start states are taken from the model's own forward pass.  Each
trajectory batch recomputes h0 with a different number of rows, and on a GPU
that can select a different matrix kernel and summation order, so the batch h0
is compared with a tolerance relative to the largest start-state entry.
Summation-order noise is of order 1e-7 of that scale; a real mismatch, such
as a missing ReLU, is of order 1.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from delta_common import (
    GAIN_GRID, INIT_EPISODES, INIT_SPLITS, MAX_NAV_STATES, NAV_BURN_IN, NAV_TRAJECTORIES,
    ORIGINS, OUTPUT_ROOT, PROJECT_ROOT, RETENTION_STEPS, SCHEMA_VERSION, E5C_ROOT,
    autonomous_retention, build_navigation_model, load_phasef_modules, load_protocol,
    log_slope, navigation_trajectories, relation_vectors, relative_displacement,
    trajectory_profile, with_core,
)

H0_TOLERANCE = 1e-4

def nav_checkpoint(source: int) -> Path:
    return PROJECT_ROOT / 'sources' / f'nav_seed_{source}' / 'artifacts' / 'navigation' / 'best.pt'

def preflight(args) -> int:
    import torch
    report: dict[str, Any] = {'schema_version': SCHEMA_VERSION}
    fullscale, phasef = load_phasef_modules()
    fullscale.import_historical()
    protocol = load_protocol()
    report['sources'] = protocol['all_sources']
    report['targets'] = protocol['target_split_seeds']
    report['gain_grid'] = list(GAIN_GRID)
    report['relation_scale'] = protocol['velocity_scale']
    missing = [s for s in protocol['all_sources'] if not nav_checkpoint(s).is_file()]
    if missing:
        raise RuntimeError(f'missing navigation checkpoints for sources {missing}')
    payload = torch.load(nav_checkpoint(protocol['all_sources'][0]), map_location='cpu',
                         weights_only=False)
    cfg = payload.get('config')
    if not isinstance(cfg, dict) or 'data' not in cfg or 'model' not in cfg:
        raise RuntimeError('navigation checkpoint lacks the training config needed for H_nav')
    report['navigation_config'] = {
        'seed': cfg['seed'], 'seq_len': cfg['data']['seq_len'], 'dt': cfg['data'].get('dt'),
        'place_cells': cfg['data']['place_cells'], 'hidden_dim': cfg['model']['hidden_dim'],
        'init_relu': cfg['model'].get('init_relu', False)}
    if int(cfg['model']['hidden_dim']) != int(protocol['model']['hidden_dim']):
        raise RuntimeError('navigation hidden size differs from the downstream model')
    report['e5c_posthoc_present'] = (E5C_ROOT / 'aggregate_posthoc' / 'threshold_gain_by_source.csv').is_file()
    report['status'] = 'preflight_ok'
    print(json.dumps(report, indent=2))
    return 0

def init_block(source, target, protocol, fullscale, phasef, hist, device, cores, n_episodes):
    import torch
    run_seed = fullscale.stable_seed32('phaseF_run', target, protocol['training_examples'])
    random_core_seed = fullscale.stable_seed32('phaseF_fresh_random_core', source, target)
    hist['set_seed'](run_seed)
    base = hist['build_hard_model'](protocol['model'], n_nodes=64, device=device).eval()
    with torch.no_grad():
        nodes = torch.arange(64, device=device)
        h0 = base(start_node=nodes, velocities=torch.zeros(64, 1, 2, device=device),
                  lengths=torch.ones(64, dtype=torch.long, device=device),
                  validate_inputs=False)['h0']
    scale = max(float(h0.abs().max()), 1e-12)
    worst = 0.0
    pools_nodes, manifest = fullscale.build_fullscale_split(protocol, target)
    if manifest['manifest_hash'] != protocol['expected_manifest_hashes'][str(target)]:
        raise RuntimeError('manifest hash mismatch')
    loader_seed = fullscale.stable_seed32('phaseF_loader', target, protocol['training_examples'])
    relations = relation_vectors(protocol, device)
    delta_rows, profile_rows, retention_rows = [], [], []
    for origin in ORIGINS:
        spec = {'origin': origin, 'kind': 'fixed_gain', 'gain': 1.0, 'core': 'frozen'}
        phasef.configure_origin(base, spec, hist, nav_hh=cores['navigation'][0],
                                nav_ih=cores['navigation'][1],
                                matched_hh=cores['matched_random'][0],
                                matched_ih=cores['matched_random'][1],
                                cell_cls=hist['FactorialCell'], model_cfg=protocol['model'],
                                device=device, random_core_seed=random_core_seed)
        w_hh = base.grid_rnn.cell.h_proj.weight.detach()
        w_in = base.grid_rnn.cell.in_proj.weight.detach()
        if not (torch.equal(w_hh.cpu(), cores[origin][0].to(w_hh.dtype))
                and torch.equal(w_in.cpu(), cores[origin][1].to(w_in.dtype))):
            raise RuntimeError(f'{origin} core was not installed exactly')
        for row in relative_displacement(w_hh, w_in, h0, relations, GAIN_GRID):
            delta_rows.append({'state_set': 'H_init', 'origin': origin, **row})
        retention_rows.append({'state_set': 'H_init_start', 'origin': origin,
                               **autonomous_retention(w_hh, h0)})
        for gain in GAIN_GRID:
            model = phasef.FixedGainPathIntModel(base, float(gain)).to(device).eval()
            for split in INIT_SPLITS:
                episodes = pools_nodes[split][:n_episodes]
                loader = fullscale.make_loader(fullscale.to_historical_episodes(episodes, hist),
                                               hist, 128, False, loader_seed + 2)
                hidden, first, lengths = [], [], []
                with torch.no_grad():
                    for batch in loader:
                        out = model(start_node=batch['start_node'].to(device),
                                    velocities=batch['velocities'].to(device),
                                    lengths=batch['lengths'].to(device),
                                    start_coords=batch['start_coords'].to(device),
                                    validate_inputs=False)
                        deviation = float((out['h0'] - h0[batch['start_node'].to(device)]).abs().max()) / scale
                        worst = max(worst, deviation)
                        if deviation > H0_TOLERANCE:
                            raise RuntimeError('model h0 differs from the H_init start states: '
                                               f'max deviation {deviation:.2e} of the start-state scale')
                        width = out['hidden_seq'].shape[1]
                        pad = 12 - width
                        hidden.append(torch.nn.functional.pad(out['hidden_seq'], (0, 0, 0, pad)))
                        first.append(out['h0'])
                        lengths.append(batch['lengths'].to(device))
                profile = trajectory_profile(torch.cat(hidden), torch.cat(first), w_hh,
                                             torch.cat(lengths))
                for row in profile:
                    profile_rows.append({'state_set': 'H_init', 'origin': origin, 'gain': float(gain),
                                         'split': split, **row})
                steps = [r['step'] for r in profile if RETENTION_STEPS[0] <= r['step'] <= RETENTION_STEPS[1]]
                norms = [r['hidden_norm'] for r in profile if RETENTION_STEPS[0] <= r['step'] <= RETENTION_STEPS[1]]
                if len(steps) >= 2:
                    retention_rows.append({'state_set': f'H_init_{split}', 'origin': origin,
                                           'gain': float(gain), 'norm_log_slope_per_step': log_slope(steps, norms)})
    return delta_rows, profile_rows, retention_rows, {'run_seed': int(run_seed),
                                                      'dataset_manifest_hash': manifest['manifest_hash'],
                                                      'h0_max_relative_deviation': worst}

def nav_block(source, protocol, fullscale, device, cores, n_trajectories):
    import torch
    payload = torch.load(nav_checkpoint(source), map_location='cpu', weights_only=False)
    cfg = payload['config']
    nav_model = build_navigation_model(payload, device)
    if not (torch.equal(nav_model.grid_rnn.cell.h_proj.weight.detach().cpu(), cores['navigation'][0])
            and torch.equal(nav_model.grid_rnn.cell.in_proj.weight.detach().cpu(), cores['navigation'][1])):
        raise RuntimeError('navigation model weights differ from load_navigation_grid_weights')
    velocities, init_place = navigation_trajectories(cfg, n_trajectories)
    velocities, init_place = velocities.to(device), init_place.to(device)
    steps = velocities.shape[1]
    generator = torch.Generator().manual_seed(int(fullscale.stable_seed32('e5b_nav_states', source)))
    positions = torch.randperm(velocities.shape[0] * (steps - NAV_BURN_IN), generator=generator)[:MAX_NAV_STATES]
    relations = relation_vectors(protocol, device)
    delta_rows, profile_rows, retention_rows = [], [], []
    for origin in ORIGINS:
        w_hh = cores[origin][0].to(device)
        w_in = cores[origin][1].to(device)
        model = nav_model if origin == 'navigation' else with_core(nav_model, w_hh, w_in)
        with torch.no_grad():
            out = model(velocities, init_place)
            h0 = model.init_encoder(init_place)
            if model.nonnegative and model.init_relu:
                h0 = torch.relu(h0)
        hidden = out['hidden_seq']
        states = hidden[:, NAV_BURN_IN:, :].reshape(-1, hidden.shape[-1])[positions.to(device)]
        for row in relative_displacement(w_hh, w_in, states, relations, GAIN_GRID):
            delta_rows.append({'state_set': 'H_nav', 'origin': origin, **row})
        retention_rows.append({'state_set': 'H_nav', 'origin': origin, **autonomous_retention(w_hh, states)})
        profile = trajectory_profile(hidden, h0, w_hh)
        for row in profile:
            profile_rows.append({'state_set': 'H_nav', 'origin': origin, **row})
        chosen = [r for r in profile if r['step'] > NAV_BURN_IN]
        retention_rows.append({'state_set': 'H_nav_trajectory', 'origin': origin,
                               'norm_log_slope_per_step': log_slope([r['step'] for r in chosen],
                                                                    [r['hidden_norm'] for r in chosen])})
        del model
    provenance = {'navigation_seed': int(cfg['seed']), 'navigation_seq_len': int(steps),
                  'navigation_trajectories': int(velocities.shape[0]),
                  'navigation_burn_in_steps': NAV_BURN_IN, 'navigation_states_sampled': int(positions.numel()),
                  'navigation_trajectory_seed': int(cfg['seed']) + 9999}
    return delta_rows, profile_rows, retention_rows, provenance

def run_cell(args) -> int:
    import torch
    fullscale, phasef = load_phasef_modules()
    protocol = load_protocol()
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
    nav_hh, nav_ih, nav_meta = hist['load_navigation_grid_weights'](nav_checkpoint(source))
    matched_seed = fullscale.stable_seed32('matched_random', target)
    matched_hh, matched_ih, _ = hist['make_matched_random_weights'](
        nav_hh, nav_ih, seed=matched_seed, iterations=8, singular_rank=16, probe_count=64, strict=True)
    cores = {'navigation': (nav_hh, nav_ih), 'matched_random': (matched_hh, matched_ih)}

    init_delta, init_profile, init_retention, init_meta = init_block(
        source, target, protocol, fullscale, phasef, hist, device, cores, int(args.init_episodes))
    nav_delta, nav_profile, nav_retention, nav_meta_rows = nav_block(
        source, protocol, fullscale, device, cores, int(args.nav_trajectories))

    tag = {'source_seed': source, 'target_seed': target}
    fullscale.write_csv(out / 'delta.csv', [{**tag, **r} for r in init_delta + nav_delta])
    fullscale.write_csv(out / 'trajectory_profile.csv', [{**tag, **r} for r in init_profile + nav_profile])
    fullscale.write_csv(out / 'retention.csv', [{**tag, **r} for r in init_retention + nav_retention])
    summary = {r['origin']: r['delta_mean'] for r in nav_delta if r['gain'] == 1.0}
    print(source, target, 'H_nav delta(g=1) nav', round(summary['navigation'], 5),
          'matched', round(summary['matched_random'], 5), flush=True)
    fullscale.atomic_json(out / 'COMPLETE.json', {
        'schema_version': SCHEMA_VERSION, 'status': 'complete', **tag,
        'navigation_checkpoint_sha256': nav_meta['checkpoint_sha256'],
        'matched_random_seed': int(matched_seed), **init_meta, **nav_meta_rows,
        'init_episodes_per_split': int(args.init_episodes), 'gain_grid': list(GAIN_GRID)})
    return 0

def main() -> int:
    parser = argparse.ArgumentParser(description="E5b' rectified displacement")
    parser.add_argument('--source-seed', type=int)
    parser.add_argument('--target-seed', type=int)
    parser.add_argument('--output-root', type=Path, default=OUTPUT_ROOT)
    parser.add_argument('--init-episodes', type=int, default=INIT_EPISODES)
    parser.add_argument('--nav-trajectories', type=int, default=NAV_TRAJECTORIES)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    if args.preflight:
        return preflight(args)
    if args.source_seed is None or args.target_seed is None:
        parser.error('--source-seed and --target-seed are required')
    return run_cell(args)

if __name__ == '__main__':
    sys.exit(main())
