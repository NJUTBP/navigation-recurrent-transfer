#!/usr/bin/env python3
"""E0.9a driver: rerun the four decisive Phase F arms at 32,000 updates.

One invocation trains every condition of one group for one (source, target)
cell.  Model construction, weight installation, wrapping, seeding and evaluation
are taken from the frozen Phase F ``run_group.py`` so that nothing except the
update budget differs.

    "$PYTHON_BIN" run_extended.py --source-seed 6101 --target-seed 84001 \
        --group scalar_pair

Preflight (no GPU, no training, seconds) validates imports, protocol derivation,
split reconstruction and the manifest hashes:

    "$PYTHON_BIN" run_extended.py --preflight
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from extended_common import (
    EXTENDED_UPDATES, GROUPS, OUTPUT_ROOT, PROJECT_ROOT, SCHEMA_VERSION,
    TARGET_SEEDS, GainTrace, derive_protocol, load_phasef_modules,
)


def preflight() -> int:
    """Cheap end-to-end check of everything except the training loop."""
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

    specs = phasef.condition_specs()
    missing = [c for conditions in GROUPS.values() for c in conditions if c not in specs]
    if missing:
        raise RuntimeError(f'conditions absent from the Phase F spec table: {missing}')
    report['conditions_by_group'] = {k: list(v) for k, v in GROUPS.items()}

    # Seeds must be identical to Phase F, because training_examples is unchanged.
    seeds = {}
    for target in TARGET_SEEDS:
        seeds[str(target)] = {
            'run_seed': int(fullscale.stable_seed32('phaseF_run', target, protocol['training_examples'])),
            'loader_seed': int(fullscale.stable_seed32('phaseF_loader', target, protocol['training_examples'])),
            'matched_random_seed': int(fullscale.stable_seed32('matched_random', target)),
        }
    report['seeds'] = seeds

    # Dataset reconstruction and hash verification.
    splits = {}
    for target in TARGET_SEEDS:
        _, manifest = fullscale.build_fullscale_split(protocol, target)
        expected = protocol['expected_manifest_hashes'][str(target)]
        if manifest['manifest_hash'] != expected:
            raise RuntimeError(
                f'manifest hash mismatch for target {target}: '
                f'{manifest["manifest_hash"]} != {expected}'
            )
        splits[str(target)] = manifest['manifest_hash']
    report['verified_manifest_hashes'] = splits

    navigation = []
    for source in protocol['all_sources']:
        path = PROJECT_ROOT / 'sources' / f'nav_seed_{source}' / 'artifacts' / 'navigation' / 'best.pt'
        navigation.append({'source_seed': source, 'exists': path.is_file()})
    report['navigation_checkpoints'] = navigation
    if not all(item['exists'] for item in navigation):
        raise RuntimeError('missing navigation checkpoints')

    report['n_cells'] = len(protocol['all_sources']) * len(TARGET_SEEDS) * len(GROUPS)
    report['status'] = 'preflight_ok'
    print(json.dumps(report, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description='E0.9a extended-budget driver')
    parser.add_argument('--source-seed', type=int)
    parser.add_argument('--target-seed', type=int)
    parser.add_argument('--group', choices=sorted(GROUPS))
    parser.add_argument('--updates', type=int, default=EXTENDED_UPDATES)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()

    if args.preflight:
        return preflight()
    if args.source_seed is None or args.target_seed is None or args.group is None:
        parser.error('--source-seed, --target-seed and --group are required')

    import torch

    fullscale, phasef = load_phasef_modules()
    protocol = derive_protocol(updates=int(args.updates))

    source, target, group = int(args.source_seed), int(args.target_seed), str(args.group)
    if source not in protocol['all_sources']:
        raise ValueError(f'source {source} outside the frozen protocol')
    if target not in protocol['target_split_seeds']:
        raise ValueError(f'target {target} outside the E0.9a split set')

    out = OUTPUT_ROOT / f'source_{source}' / f'target_{target}' / f'group_{group}'
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
        fingerprints = [hist['episode_fingerprint'](e) for e in episodes]
        if fingerprints != manifest['fingerprints'][name]:
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
        iterations=8, singular_rank=16, probe_count=64, strict=True,
    )
    fullscale.atomic_json(out / 'matched_random_diagnostics.json', matched_diag)

    model_cfg = protocol['model']
    train_cfg = protocol['train']
    batch_size = int(protocol['batch_size'])
    # Seed components are identical to Phase F by construction.
    run_seed = fullscale.stable_seed32('phaseF_run', target, protocol['training_examples'])
    loader_seed = fullscale.stable_seed32('phaseF_loader', target, protocol['training_examples'])
    random_core_seed = fullscale.stable_seed32('phaseF_fresh_random_core', source, target)

    specs = phasef.condition_specs()
    cell_cls = hist['FactorialCell']
    metrics: list[dict[str, Any]] = []
    per_length: list[dict[str, Any]] = []
    updates_log: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    gain_traces: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []

    for condition in GROUPS[group]:
        spec = specs[condition]
        hist['set_seed'](run_seed)
        base = hist['build_hard_model'](model_cfg, n_nodes=64, device=device)
        phasef.configure_origin(
            base, spec, hist, nav_hh=nav_hh, nav_ih=nav_ih,
            matched_hh=matched_hh, matched_ih=matched_ih,
            cell_cls=cell_cls, model_cfg=model_cfg, device=device,
            random_core_seed=random_core_seed,
        )
        model = phasef.wrap_model(base, spec, hist, device)
        hist['set_seed'](run_seed)

        grid_any = any(p.requires_grad for p in model.grid_rnn.parameters())
        if spec['core'] == 'frozen' and grid_any:
            raise RuntimeError(f'grid unexpectedly trainable: {condition}')
        if spec['kind'] == 'scalar' and abs(float(model.gain().detach().cpu()) - 1.0) > 1e-8:
            raise RuntimeError('scalar did not start at unit gain')

        tracer = GainTrace(model)
        tr_loader = fullscale.make_loader(train_eps, hist, batch_size, True, loader_seed)
        va_loader = fullscale.make_loader(val_eps, hist, batch_size, False, loader_seed + 1)
        training = hist['train_fixed_updates'](model, tr_loader, va_loader, device,
                                               train_cfg, seed=run_seed)
        if training.stopped_update != int(train_cfg['max_gradient_updates']):
            raise RuntimeError(
                f'fixed update contract violated in {condition}: {training.stopped_update}'
            )
        model.load_state_dict(training.state_dict, strict=True)
        for record in tracer.close(final_model=model):
            gain_traces.append({'source_seed': source, 'target_seed': target,
                                'group': group, 'condition': condition, **record})

        eval_splits = [('train_8000', train_eps), ('validation_seen_short', val_eps)]
        eval_splits += [(name, pools[name]) for name in test_names]
        evaluations: dict[str, Any] = {}
        for split, episodes in eval_splits:
            loader = fullscale.make_loader(episodes, hist, batch_size, False, loader_seed + 2)
            summary, _ = hist['evaluate_loader_detailed'](
                model, loader, device, split_name=split, calibration_bins=15)
            evaluations[split] = summary
            metrics.append({
                'schema_version': SCHEMA_VERSION, 'source_seed': source,
                'target_seed': target, 'group': group, 'condition': condition,
                'origin': spec['origin'], 'kind': spec['kind'],
                'core_trainability': spec['core'], 'split': split,
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
                    'source_seed': source, 'target_seed': target, 'group': group,
                    'condition': condition, 'split': split,
                    'path_length': int(length), 'accuracy': float(accuracy),
                })

        for entry in training.history:
            updates_log.append({'source_seed': source, 'target_seed': target,
                                'group': group, 'condition': condition, **entry})
        diagnostics.append({
            'source_seed': source, 'target_seed': target, 'group': group,
            'max_gradient_updates': int(train_cfg['max_gradient_updates']),
            'best_update': training.best_update,
            **phasef.diagnostic_row(model, spec, condition),
        })

        state, storage_mode = phasef.compact_state_dict(model, spec['core'] == 'trainable')
        checkpoint_dir = out / 'checkpoints' / condition
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / 'best_validation.pt'
        torch.save({
            'schema_version': SCHEMA_VERSION, 'model_state': state,
            'checkpoint_storage_mode': storage_mode, 'model_config': model_cfg,
            'source_seed': source, 'target_seed': target, 'group': group,
            'condition': condition, 'condition_spec': spec,
            'run_seed': run_seed, 'loader_seed': loader_seed,
            'matched_random_seed': matched_seed,
            'max_gradient_updates': int(train_cfg['max_gradient_updates']),
            'best_update': training.best_update,
            'validation_metrics': {k: v for k, v in evaluations['validation_seen_short'].items()
                                   if not isinstance(v, dict)},
            'dataset_manifest_hash': manifest['manifest_hash'],
            'navigation_checkpoint': str(nav_path),
            'navigation_checkpoint_sha256': nav_meta['checkpoint_sha256'],
        }, checkpoint_path)
        checkpoints.append({
            'source_seed': source, 'target_seed': target, 'group': group,
            'condition': condition, 'path': str(checkpoint_path),
            'storage_mode': storage_mode, 'best_update': training.best_update,
        })

        print(source, target, group, condition,
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
    if gain_traces:
        fullscale.write_csv(out / 'gain_trace.csv', gain_traces)
    fullscale.atomic_json(out / 'COMPLETE.json', {
        'schema_version': SCHEMA_VERSION, 'status': 'complete',
        'source_seed': source, 'target_seed': target, 'group': group,
        'max_gradient_updates': int(train_cfg['max_gradient_updates']),
        'dataset_manifest_hash': manifest['manifest_hash'],
        'conditions': list(GROUPS[group]),
    })
    return 0


if __name__ == '__main__':
    sys.exit(main())
