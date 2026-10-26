#!/usr/bin/env python3
"""Bidirectional W_hh scaling intervention registered after C1."""
from __future__ import annotations
import os

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(os.environ.get('NRT_PROJECT_ROOT', str(Path(__file__).resolve().parents[4])))
MECH_TOOLS = PROJECT_ROOT / 'tools' / 'mechanism_v1'
for sibling in ('E6_rho_scan_v1', 'E0_9a_extended_budget_v1'):
    path = str(MECH_TOOLS / sibling / 'scripts')
    if path not in sys.path:
        sys.path.insert(0, path)

import extended_common  # noqa: E402
import rho_core  # noqa: E402
import run_rho_scan as e6  # noqa: E402
import intervention_common as common  # noqa: E402

CONDITION = 'retention_intervention_trainable_scalar_gain'
SPEC = {'origin': 'matched_random', 'kind': 'scalar', 'core': 'frozen'}
PROFILE_EPISODES = 512

def protocol_for(updates: int) -> dict:
    protocol = e6.derive_protocol(updates)
    protocol['schema_version'] = common.SCHEMA_VERSION
    protocol['status'] = 'autonomous_retention_intervention'
    protocol['output_root'] = str(common.OUTPUT_ROOT)
    protocol['conditions_by_group'] = {'retention_intervention': [CONDITION]}
    protocol['intervention_arms'] = list(common.ARMS)
    return protocol

def construct_core(nav_hh, nav_ih, *, target: int, arm: str, fullscale):
    meta = common.arm_spec(arm)
    seed = fullscale.stable_seed32('e6_rho_core', target)
    hh, ih, rho_meta = rho_core.signed_similarity_core(nav_hh, nav_ih, seed=seed, rho=meta['base_rho'])
    scaled_hh = (hh * float(meta['scale'])).contiguous()
    return scaled_hh, ih.contiguous(), {**rho_meta, **meta}

def preflight() -> int:
    import torch
    fullscale, _ = extended_common.load_phasef_modules()
    hist = fullscale.import_historical()
    result, c = common.load_c1()
    protocol = protocol_for(extended_common.EXTENDED_UPDATES)
    missing = []
    for source in protocol['all_sources']:
        if not e6.nav_checkpoint(source).is_file():
            missing.append(str(e6.nav_checkpoint(source)))
    if missing:
        raise RuntimeError(f'missing navigation checkpoints: {missing[:3]}')
    baseline_missing = []
    for source in protocol['all_sources']:
        for target in protocol['target_split_seeds']:
            for rho in (0.0, 0.1):
                if not (e6.cell_dir(common.E6_ROOT, source, target, rho) / 'COMPLETE.json').is_file():
                    baseline_missing.append((source, target, rho))
    if baseline_missing:
        raise RuntimeError(f'missing E6 baseline cells: {baseline_missing[:5]}')
    nav_hh, nav_ih, _ = hist['load_navigation_grid_weights'](e6.nav_checkpoint(protocol['all_sources'][0]))
    checks = {}
    g = torch.Generator().manual_seed(6101)
    h = torch.rand(32, nav_hh.shape[0], generator=g, dtype=nav_hh.dtype)
    for arm in common.ARMS:
        hh, _, meta = construct_core(nav_hh, nav_ih, target=84001, arm=arm, fullscale=fullscale)
        base_hh, _, _ = rho_core.signed_similarity_core(nav_hh, nav_ih,
                                                        seed=fullscale.stable_seed32('e6_rho_core', 84001),
                                                        rho=meta['base_rho'])
        expected = float(meta['scale']) * torch.relu(h @ base_hh.t())
        observed = torch.relu(h @ hh.t())
        denom = expected.abs().max().clamp_min(1e-12)
        rel = float((expected - observed).abs().max() / denom)
        frob_ratio = float(hh.double().norm() / base_hh.double().norm())
        checks[arm] = {'base_rho': meta['base_rho'], 'scale': meta['scale'],
                       'homogeneity_relative_deviation': rel, 'frob_ratio': frob_ratio}
        if rel > 2e-6 or abs(frob_ratio - float(meta['scale'])) > 2e-6:
            raise RuntimeError(f'intervention algebra failed for {arm}')
    report = {'schema_version': common.SCHEMA_VERSION, 'status': 'preflight_ok',
              'torch': torch.__version__, 'c1_result_sha256': common.C1_RESULT_SHA256,
              'c': c, 'inverse_c': 1.0 / c, 'c1_anchors': result['anchors'],
              'n_cells': len(protocol['all_sources']) * len(protocol['target_split_seeds']) * len(common.ARMS),
              'checks': checks}
    print(json.dumps(report, indent=2))
    return 0

def run_cell(args) -> int:
    import torch
    fullscale, phasef = extended_common.load_phasef_modules()
    protocol = protocol_for(int(args.updates))
    source, target, arm = int(args.source_seed), int(args.target_seed), str(args.arm)
    if source not in protocol['all_sources'] or target not in protocol['target_split_seeds']:
        raise ValueError('source or target outside the intervention protocol')
    if arm not in common.ARMS:
        raise ValueError('arm outside the intervention protocol')
    out = common.cell_dir(Path(args.output_root), source, target, arm)
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

    nav_hh, nav_ih, nav_meta = hist['load_navigation_grid_weights'](e6.nav_checkpoint(source))
    scaled_hh, scaled_ih, intervention = construct_core(nav_hh, nav_ih, target=target, arm=arm, fullscale=fullscale)
    base_hh, base_ih, _ = rho_core.signed_similarity_core(
        nav_hh, nav_ih, seed=fullscale.stable_seed32('e6_rho_core', target), rho=intervention['base_rho'])
    if not torch.equal(scaled_ih, base_ih):
        raise RuntimeError('intervention changed W_in')
    ratio = float(scaled_hh.double().norm() / base_hh.double().norm())
    if abs(ratio - float(intervention['scale'])) > 2e-6:
        raise RuntimeError('W_hh scale mismatch')

    model_cfg, train_cfg = protocol['model'], protocol['train']
    batch_size = int(protocol['batch_size'])
    run_seed = fullscale.stable_seed32('phaseF_run', target, protocol['training_examples'])
    loader_seed = fullscale.stable_seed32('phaseF_loader', target, protocol['training_examples'])
    random_core_seed = fullscale.stable_seed32('phaseF_fresh_random_core', source, target)
    hist['set_seed'](run_seed)
    base = hist['build_hard_model'](model_cfg, n_nodes=64, device=device)
    phasef.configure_origin(base, SPEC, hist, nav_hh=nav_hh, nav_ih=nav_ih,
                            matched_hh=scaled_hh, matched_ih=scaled_ih,
                            cell_cls=hist['FactorialCell'], model_cfg=model_cfg, device=device,
                            random_core_seed=random_core_seed)
    if not torch.equal(base.grid_rnn.cell.h_proj.weight.detach().cpu(), scaled_hh.to(torch.float32)):
        raise RuntimeError('scaled recurrent core was not installed exactly')
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
    if not np.isfinite(float(model.gain().detach().cpu())):
        raise RuntimeError('non-finite learned gain')

    tag = {'schema_version': common.SCHEMA_VERSION, 'source_seed': source, 'target_seed': target,
           'arm': arm, 'base_rho': intervention['base_rho'], 'recurrent_scale': intervention['scale'],
           'condition': CONDITION}
    gain_trace = [{**tag, **record} for record in tracer.close(final_model=model)]
    metrics, per_length = [], []
    for split in ('train_8000', 'validation_seen_short', 'id_seen_short', 'heldout_short', 'seen_long', 'heldout_long'):
        loader = fullscale.make_loader(pools[split], hist, batch_size, False, loader_seed + 2)
        summary, _ = hist['evaluate_loader_detailed'](model, loader, device, split_name=split, calibration_bins=15)
        values = [summary['accuracy'], summary['nll'], summary['ece'], summary['mean_confidence'], summary['mean_readout_margin']]
        if not all(np.isfinite(float(v)) for v in values):
            raise RuntimeError(f'non-finite evaluation metric in {split}')
        metrics.append({**tag, 'split': split, 'accuracy': summary['accuracy'], 'nll': summary['nll'],
                        'ece': summary['ece'], 'mean_confidence': summary['mean_confidence'],
                        'mean_readout_margin': summary['mean_readout_margin'], 'best_update': training.best_update,
                        'stopped_update': training.stopped_update, 'run_seed': run_seed, 'loader_seed': loader_seed,
                        'dataset_manifest_hash': manifest['manifest_hash']})
        per_length += [{**tag, 'split': split, 'path_length': int(k), 'accuracy': float(v)}
                       for k, v in summary['accuracy_by_path_length'].items()]

    learned = float(model.gain().detach().cpu())
    base_retention = common.c1_retention(source, target, float(intervention['base_rho']))
    expected_retention = base_retention * float(intervention['scale'])
    diagnostics = [{**tag, 'learned_gain': learned, 'log_gain': float(model.log_gain.detach().cpu()),
                    'best_update': training.best_update, 'core_seed': intervention['core_seed'],
                    'flip_count': intervention['flip_count'], 'realized_rho': intervention['realized_rho'],
                    'source_retention_before': base_retention,
                    'source_retention_after_exact': expected_retention}]
    w_hh = base.grid_rnn.cell.h_proj.weight.detach()
    model.eval()
    profile = [{**tag, **row} for row in e6.downstream_profile(model, pools, hist, fullscale, loader_seed, w_hh, device)]

    for name, rows in (('metrics.csv', metrics), ('per_length.csv', per_length),
                       ('update_curves.csv', [{**tag, **h} for h in training.history]),
                       ('gain_trace.csv', gain_trace), ('condition_diagnostics.csv', diagnostics),
                       ('downstream_profile.csv', profile)):
        fullscale.write_csv(out / name, rows)
    state, storage_mode = phasef.compact_state_dict(model, False)
    checkpoint_dir = out / 'checkpoints' / CONDITION
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save({'schema_version': common.SCHEMA_VERSION, 'model_state': state,
                'checkpoint_storage_mode': storage_mode,
                'core_reconstruction': {'base_rho': intervention['base_rho'],
                                        'recurrent_scale': intervention['scale'],
                                        'core_seed': int(intervention['core_seed'])},
                'source_seed': source, 'target_seed': target, 'run_seed': run_seed,
                'loader_seed': loader_seed, 'best_update': training.best_update},
               checkpoint_dir / 'best_validation.pt')
    acc = {m['split']: m['accuracy'] for m in metrics}
    print(source, target, arm, 'scale', round(float(intervention['scale']), 6),
          'gain', round(learned, 2), 'ID', round(acc['id_seen_short'], 4),
          'long', round(acc['seen_long'], 4), flush=True)
    fullscale.atomic_json(out / 'COMPLETE.json', {
        'schema_version': common.SCHEMA_VERSION, 'status': 'complete', **tag,
        'navigation_checkpoint_sha256': nav_meta['checkpoint_sha256'],
        'learned_gain': learned, 'source_retention_before': base_retention,
        'source_retention_after_exact': expected_retention,
        'dataset_manifest_hash': manifest['manifest_hash']})
    return 0

def main() -> int:
    parser = argparse.ArgumentParser(description='Autonomous-retention impairment/rescue experiment')
    parser.add_argument('--source-seed', type=int)
    parser.add_argument('--target-seed', type=int)
    parser.add_argument('--arm', choices=tuple(common.ARMS))
    parser.add_argument('--updates', type=int, default=extended_common.EXTENDED_UPDATES)
    parser.add_argument('--output-root', type=Path, default=common.OUTPUT_ROOT)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    if args.preflight:
        return preflight()
    if args.source_seed is None or args.target_seed is None or args.arm is None:
        parser.error('--source-seed, --target-seed and --arm are required')
    return run_cell(args)

if __name__ == '__main__':
    raise SystemExit(main())
