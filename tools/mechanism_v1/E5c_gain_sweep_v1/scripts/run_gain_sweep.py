#!/usr/bin/env python3
"""E5c driver: train one (source, target, gain) cell, both origins.

    "$PYTHON_BIN" run_gain_sweep.py --source-seed 6101 --target-seed 84001 --gain 32

Preflight (no GPU, seconds):

    "$PYTHON_BIN" run_gain_sweep.py --preflight

The gain is a fixed buffer inside ``FixedGainPathIntModel``, so the only
trainable parameters are the start embedding and the endpoint readout.  That
makes every cell structurally identical to the E0.9a identity conditions apart
from the value of the gain, which is what lets the E0.9a g = 1 cells serve as the
left-hand anchor of the operating curve without being rerun.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from gain_sweep_common import (
    EXTENDED_UPDATES, GAIN_GRID, ORIGINS, OUTPUT_ROOT, PROJECT_ROOT,
    SCHEMA_VERSION, TARGET_SEEDS, build_specs, condition_name,
    derive_protocol, load_phasef_modules,
)


def preflight() -> int:
    report: dict[str, Any] = {'schema_version': SCHEMA_VERSION}
    fullscale, phasef = load_phasef_modules()
    report['fullscale_common'] = str(Path(fullscale.__file__).resolve())
    report['run_group'] = str(Path(phasef.__file__).resolve())

    protocol = derive_protocol()
    report['max_gradient_updates'] = protocol['train']['max_gradient_updates']
    report['eval_every_updates'] = protocol['train']['eval_every_updates']
    report['early_stopping_patience_evals'] = protocol['train']['early_stopping_patience_evals']
    report['n_evaluations'] = protocol['n_evaluations']
    report['training_examples_seed_component'] = protocol['training_examples']
    report['gain_grid'] = protocol['gain_grid']
    report['n_cells'] = protocol['n_cells']

    # The spec shape must match Phase F exactly, otherwise wrap_model will reject it.
    reference = phasef.condition_specs()['navigation_fixed_gain3_frozen']
    example = build_specs(32)[condition_name('navigation', 32)]
    if set(example) != set(reference):
        raise RuntimeError(f'spec key mismatch: {sorted(example)} vs {sorted(reference)}')
    report['spec_keys'] = sorted(example)
    report['example_spec'] = example

    # Our condition names must not collide with any Phase F condition.
    collisions = sorted(set(phasef.condition_specs()) & {
        condition_name(o, g) for o in ORIGINS for g in GAIN_GRID})
    if collisions:
        raise RuntimeError(f'condition name collision with Phase F: {collisions}')
    report['no_name_collision'] = True

    seeds = {}
    for target in TARGET_SEEDS:
        seeds[str(target)] = {
            'run_seed': int(fullscale.stable_seed32('phaseF_run', target, protocol['training_examples'])),
            'loader_seed': int(fullscale.stable_seed32('phaseF_loader', target, protocol['training_examples'])),
            'matched_random_seed': int(fullscale.stable_seed32('matched_random', target)),
        }
    report['seeds'] = seeds

    splits = {}
    for target in TARGET_SEEDS:
        _, manifest = fullscale.build_fullscale_split(protocol, target)
        expected = protocol['expected_manifest_hashes'][str(target)]
        if manifest['manifest_hash'] != expected:
            raise RuntimeError(
                f'manifest hash mismatch for target {target}: '
                f'{manifest["manifest_hash"]} != {expected}')
        splits[str(target)] = manifest['manifest_hash']
    report['verified_manifest_hashes'] = splits

    navigation = [{'source_seed': s,
                   'exists': (PROJECT_ROOT / 'sources' / f'nav_seed_{s}' / 'artifacts'
                              / 'navigation' / 'best.pt').is_file()}
                  for s in protocol['all_sources']]
    report['navigation_checkpoints'] = navigation
    if not all(item['exists'] for item in navigation):
        raise RuntimeError('missing navigation checkpoints')

    report['status'] = 'preflight_ok'
    print(json.dumps(report, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description='E5c fixed-gain sweep driver')
    parser.add_argument('--source-seed', type=int)
    parser.add_argument('--target-seed', type=int)
    parser.add_argument('--gain', type=int)
    parser.add_argument('--updates', type=int, default=EXTENDED_UPDATES)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()

    if args.preflight:
        return preflight()
    if args.source_seed is None or args.target_seed is None or args.gain is None:
        parser.error('--source-seed, --target-seed and --gain are required')

    import torch

    fullscale, phasef = load_phasef_modules()
    protocol = derive_protocol(updates=int(args.updates))

    source, target, gain = int(args.source_seed), int(args.target_seed), int(args.gain)
    if source not in protocol['all_sources']:
        raise ValueError(f'source {source} outside the frozen protocol')
    if target not in protocol['target_split_seeds']:
        raise ValueError(f'target {target} outside the E5c split set')
    if gain not in protocol['gain_grid']:
        raise ValueError(f'gain {gain} outside the E5c grid {protocol["gain_grid"]}')

    out = OUTPUT_ROOT / f'source_{source}' / f'target_{target}' / f'gain_{gain:03d}'
    if (out / 'COMPLETE.json').exists():
        print('already complete', out, flush=True)
        return 0
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f'partial output exists: {out}; inspect before rerun')
    out.mkdir(parents=True, exist_ok=True)
    fullscale.atomic_json(out / 'protocol_used.json', protocol)

    hist = fullscale.import_historical()
    device = torch.device('cuda')

    pools_nodes, manifest = fullscale.build_fullscale_split(protocol, target)
    expected = protocol['expected_manifest_hashes'][str(target)]
    if manifest['manifest_hash'] != expected:
        raise RuntimeError(f'manifest hash mismatch {manifest["manifest_hash"]} != {expected}')
    pools = {k: fullscale.to_historical_episodes(v, hist) for k, v in pools_nodes.items()}
    for name, episodes in pools.items():
        if [hist['episode_fingerprint'](e) for e in episodes] != manifest['fingerprints'][name]:
            raise RuntimeError(f'historical fingerprint mismatch: {name}')
    fullscale.atomic_json(out / 'dataset_manifest.json', manifest)

    train_eps = pools['train_8000']
    val_eps = pools['validation_seen_short']
    test_names = ['id_seen_short', 'heldout_short', 'seen_long', 'heldout_long']

    nav_path = PROJECT_ROOT / 'sources' / f'nav_seed_{source}' / 'artifacts' / 'navigation' / 'best.pt'
    if not nav_path.is_file():
        raise FileNotFoundError(nav_path)
    nav_hh, nav_ih, nav_meta = hist['load_navigation_grid_weights'](nav_path)

    matched_seed = fullscale.stable_seed32('matched_random', target)
    matched_hh, matched_ih, matched_diag = hist['make_matched_random_weights'](
        nav_hh, nav_ih, seed=matched_seed,
        iterations=8, singular_rank=16, probe_count=64, strict=True)
    fullscale.atomic_json(out / 'matched_random_diagnostics.json', matched_diag)

    model_cfg = protocol['model']
    train_cfg = protocol['train']
    batch_size = int(protocol['batch_size'])
    run_seed = fullscale.stable_seed32('phaseF_run', target, protocol['training_examples'])
    loader_seed = fullscale.stable_seed32('phaseF_loader', target, protocol['training_examples'])
    random_core_seed = fullscale.stable_seed32('phaseF_fresh_random_core', source, target)

    specs = build_specs(gain)
    cell_cls = hist['FactorialCell']
    metrics: list[dict[str, Any]] = []
    per_length: list[dict[str, Any]] = []
    updates_log: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []

    for condition in sorted(specs):
        spec = specs[condition]
        hist['set_seed'](run_seed)
        base = hist['build_hard_model'](model_cfg, n_nodes=64, device=device)
        phasef.configure_origin(
            base, spec, hist, nav_hh=nav_hh, nav_ih=nav_ih,
            matched_hh=matched_hh, matched_ih=matched_ih,
            cell_cls=cell_cls, model_cfg=model_cfg, device=device,
            random_core_seed=random_core_seed)
        model = phasef.wrap_model(base, spec, hist, device)
        hist['set_seed'](run_seed)

        if any(p.requires_grad for p in model.grid_rnn.parameters()):
            raise RuntimeError(f'grid unexpectedly trainable: {condition}')
        realised = float(model.gain().detach().cpu()) if hasattr(model, 'gain') else float(gain)
        if abs(realised - float(gain)) > 1e-8:
            raise RuntimeError(f'fixed gain not installed: {realised} != {gain}')

        tr_loader = fullscale.make_loader(train_eps, hist, batch_size, True, loader_seed)
        va_loader = fullscale.make_loader(val_eps, hist, batch_size, False, loader_seed + 1)
        training = hist['train_fixed_updates'](model, tr_loader, va_loader, device,
                                               train_cfg, seed=run_seed)
        if training.stopped_update != int(train_cfg['max_gradient_updates']):
            raise RuntimeError(
                f'fixed update contract violated in {condition}: {training.stopped_update}')
        model.load_state_dict(training.state_dict, strict=True)

        eval_splits = [('validation_seen_short', val_eps)]
        eval_splits += [(name, pools[name]) for name in test_names]
        evaluations: dict[str, Any] = {}
        for split, episodes in eval_splits:
            loader = fullscale.make_loader(episodes, hist, batch_size, False, loader_seed + 2)
            summary, _ = hist['evaluate_loader_detailed'](
                model, loader, device, split_name=split, calibration_bins=15)
            evaluations[split] = summary
            metrics.append({
                'schema_version': SCHEMA_VERSION, 'source_seed': source,
                'target_seed': target, 'gain': gain, 'condition': condition,
                'origin': spec['origin'], 'split': split,
                'accuracy': summary['accuracy'], 'nll': summary['nll'],
                'ece': summary['ece'], 'mean_confidence': summary['mean_confidence'],
                'mean_readout_margin': summary['mean_readout_margin'],
                'dataset_manifest_hash': manifest['manifest_hash'],
                'max_gradient_updates': int(train_cfg['max_gradient_updates']),
                'best_update': training.best_update,
                'stopped_update': training.stopped_update,
                'run_seed': run_seed, 'loader_seed': loader_seed,
            })
            for length, accuracy in summary['accuracy_by_path_length'].items():
                per_length.append({
                    'source_seed': source, 'target_seed': target, 'gain': gain,
                    'condition': condition, 'origin': spec['origin'], 'split': split,
                    'path_length': int(length), 'accuracy': float(accuracy)})

        for entry in training.history:
            updates_log.append({'source_seed': source, 'target_seed': target,
                                'gain': gain, 'condition': condition,
                                'origin': spec['origin'], **entry})
        diagnostics.append({
            'source_seed': source, 'target_seed': target, 'gain': gain,
            'max_gradient_updates': int(train_cfg['max_gradient_updates']),
            'best_update': training.best_update,
            **phasef.diagnostic_row(model, spec, condition)})

        state, storage_mode = phasef.compact_state_dict(model, False)
        checkpoint_dir = out / 'checkpoints' / condition
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / 'best_validation.pt'
        torch.save({
            'schema_version': SCHEMA_VERSION, 'model_state': state,
            'checkpoint_storage_mode': storage_mode, 'model_config': model_cfg,
            'source_seed': source, 'target_seed': target, 'gain': gain,
            'condition': condition, 'condition_spec': spec,
            'run_seed': run_seed, 'loader_seed': loader_seed,
            'matched_random_seed': matched_seed,
            'max_gradient_updates': int(train_cfg['max_gradient_updates']),
            'best_update': training.best_update,
            'validation_metrics': {k: v for k, v
                                   in evaluations['validation_seen_short'].items()
                                   if not isinstance(v, dict)},
            'dataset_manifest_hash': manifest['manifest_hash'],
            'navigation_checkpoint': str(nav_path),
            'navigation_checkpoint_sha256': nav_meta['checkpoint_sha256'],
        }, checkpoint_path)
        checkpoints.append({
            'source_seed': source, 'target_seed': target, 'gain': gain,
            'condition': condition, 'path': str(checkpoint_path),
            'storage_mode': storage_mode, 'best_update': training.best_update})

        print(source, target, 'gain', gain, condition,
              'val', evaluations['validation_seen_short']['accuracy'],
              'ID', evaluations['id_seen_short']['accuracy'],
              'long', evaluations['seen_long']['accuracy'],
              'best', training.best_update, flush=True)
        del model, base
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    fullscale.write_csv(out / 'metrics.csv', metrics)
    fullscale.write_csv(out / 'per_length.csv', per_length)
    fullscale.write_csv(out / 'update_curves.csv', updates_log)
    fullscale.write_csv(out / 'condition_diagnostics.csv', diagnostics)
    fullscale.write_csv(out / 'checkpoint_manifest.csv', checkpoints)
    fullscale.atomic_json(out / 'COMPLETE.json', {
        'schema_version': SCHEMA_VERSION, 'status': 'complete',
        'source_seed': source, 'target_seed': target, 'gain': gain,
        'max_gradient_updates': int(train_cfg['max_gradient_updates']),
        'dataset_manifest_hash': manifest['manifest_hash'],
        'conditions': sorted(specs)})
    return 0


if __name__ == '__main__':
    sys.exit(main())
