#!/usr/bin/env python3
"""E0.1 driver: train both non-recurrent baselines for one (target, budget) cell.

    "$PYTHON_BIN" run_baselines.py --preflight
    "$PYTHON_BIN" run_baselines.py --target-seed 84001 --updates 32000

The baselines have no navigation core, so the source seed plays no role and a
cell is (target, budget).  Everything else follows the Phase F contract: the
same splits (manifest hash checked), the same run_seed and loader_seed, hence
the same initialisation stream and minibatch order as every recurrent
condition on that target, the same optimiser and validation-only selection.

``--output-root`` exists so that the smoke cell never shares a directory with
the full array.  E5c lost its smoke cell to exactly that collision.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from baseline_common import (
    BASELINES, BUDGETS, EVAL_SPLITS, LONG_SPLITS, OUTPUT_ROOT, SCHEMA_VERSION,
    accumulator_geometry, build_baseline, derive_protocol, install_oracle,
    load_phasef_modules, make_wrapper_class, velocity_to_onehot,
)

def mapping_check(pools, hist) -> dict[str, int]:
    import numpy as np
    import torch
    from gpm_cmap.data.graph_tasks.hard_pathint import velocity_action_labels
    checked = 0
    for name, episodes in pools.items():
        for episode in episodes:
            reference = np.asarray(velocity_action_labels(episode.velocities))
            onehot = velocity_to_onehot(torch.as_tensor(episode.velocities))
            if not bool(torch.all(onehot.sum(dim=-1) == 1)):
                raise RuntimeError(f'padding-free episode lost a relation in {name}')
            if not np.array_equal(onehot.argmax(dim=-1).numpy(), reference):
                raise RuntimeError(f'one-hot labels disagree with velocity_action_labels in {name}')
            checked += 1
    return {'episodes_checked': checked}

def oracle_accuracy(pools, hist, fullscale, batch_size: int) -> dict[str, dict[str, float]]:
    import torch
    wrapper = make_wrapper_class()
    device = torch.device('cpu')
    result: dict[str, dict[str, float]] = {}
    for name in BASELINES:
        baseline = build_baseline(name)
        install_oracle(name, baseline)
        model = wrapper(baseline).to(device)
        result[name] = {}
        for split in EVAL_SPLITS:
            loader = fullscale.make_loader(pools[split], hist, batch_size, False, 1)
            summary, _ = hist['evaluate_loader_detailed'](
                model, loader, device, split_name=split, calibration_bins=15)
            result[name][split] = float(summary['accuracy'])
    return result

def preflight(args) -> int:
    report: dict[str, Any] = {'schema_version': SCHEMA_VERSION}
    fullscale, _ = load_phasef_modules()
    report['fullscale_common'] = str(Path(fullscale.__file__).resolve())
    hist = fullscale.import_historical()
    protocols = {u: derive_protocol(u) for u in BUDGETS}
    base = protocols[BUDGETS[0]]
    report['budgets'] = {str(u): {'n_evaluations': p['n_evaluations'],
                                  'eval_every_updates': p['train']['eval_every_updates'],
                                  'training_examples_seed_component': p['training_examples']}
                         for u, p in protocols.items()}
    report['targets'] = base['target_split_seeds']
    report['baselines'] = list(BASELINES)
    targets = [int(t) for t in (args.targets or base['target_split_seeds'])]
    checks = {}
    for target in targets:
        pools_nodes, manifest = fullscale.build_fullscale_split(base, target)
        expected = base['expected_manifest_hashes'][str(target)]
        if manifest['manifest_hash'] != expected:
            raise RuntimeError(f'manifest hash mismatch for target {target}')
        pools = {k: fullscale.to_historical_episodes(v, hist) for k, v in pools_nodes.items()}
        entry = {'manifest_hash': manifest['manifest_hash'], **mapping_check(pools, hist)}
        oracle = oracle_accuracy(pools, hist, fullscale, int(base['batch_size']))
        worst = min(v for acc in oracle.values() for v in acc.values())
        if worst < 1.0:
            raise RuntimeError(f'oracle below 1.0 on target {target}: {oracle}')
        entry['oracle_accuracy_min'] = worst
        entry['run_seed'] = int(fullscale.stable_seed32('phaseF_run', target, base['training_examples']))
        entry['loader_seed'] = int(fullscale.stable_seed32('phaseF_loader', target, base['training_examples']))
        checks[str(target)] = entry
    report['per_target'] = checks
    report['status'] = 'preflight_ok'
    print(json.dumps(report, indent=2))
    return 0

def train_cell(args) -> int:
    import torch
    fullscale, _ = load_phasef_modules()
    target, updates = int(args.target_seed), int(args.updates)
    if updates not in BUDGETS:
        raise ValueError(f'budget {updates} outside {BUDGETS}')
    protocol = derive_protocol(updates)
    if target not in protocol['target_split_seeds']:
        raise ValueError(f'target {target} outside the Phase F protocol')

    out = Path(args.output_root) / f'target_{target}' / f'budget_{updates:05d}'
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
    fullscale.atomic_json(out / 'dataset_manifest.json', manifest)

    train_cfg = protocol['train']
    batch_size = int(protocol['batch_size'])
    run_seed = fullscale.stable_seed32('phaseF_run', target, protocol['training_examples'])
    loader_seed = fullscale.stable_seed32('phaseF_loader', target, protocol['training_examples'])
    wrapper = make_wrapper_class()

    metrics, per_length, curves, diagnostics, checkpoints, long_rows = [], [], [], [], [], []
    for name in BASELINES:
        hist['set_seed'](run_seed)
        model = wrapper(build_baseline(name)).to(device)
        hist['set_seed'](run_seed)
        n_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
        tr_loader = fullscale.make_loader(pools['train_8000'], hist, batch_size, True, loader_seed)
        va_loader = fullscale.make_loader(pools['validation_seen_short'], hist, batch_size,
                                          False, loader_seed + 1)
        training = hist['train_fixed_updates'](model, tr_loader, va_loader, device,
                                               train_cfg, seed=run_seed)
        if training.stopped_update != updates:
            raise RuntimeError(f'fixed update contract violated in {name}: {training.stopped_update}')
        model.load_state_dict(training.state_dict, strict=True)

        for split in EVAL_SPLITS:
            loader = fullscale.make_loader(pools[split], hist, batch_size, False, loader_seed + 2)
            summary, rows = hist['evaluate_loader_detailed'](
                model, loader, device, split_name=split, calibration_bins=15)
            metrics.append({
                'schema_version': SCHEMA_VERSION, 'target_seed': target, 'budget': updates,
                'baseline': name, 'split': split, 'accuracy': summary['accuracy'],
                'nll': summary['nll'], 'ece': summary['ece'],
                'mean_confidence': summary['mean_confidence'],
                'mean_readout_margin': summary['mean_readout_margin'],
                'dataset_manifest_hash': manifest['manifest_hash'],
                'best_update': training.best_update, 'stopped_update': training.stopped_update,
                'run_seed': run_seed, 'loader_seed': loader_seed})
            for length, accuracy in summary['accuracy_by_path_length'].items():
                per_length.append({'target_seed': target, 'budget': updates, 'baseline': name,
                                   'split': split, 'path_length': int(length),
                                   'accuracy': float(accuracy)})
            if split in LONG_SPLITS:
                for row in rows:
                    long_rows.append({'target_seed': target, 'budget': updates, 'baseline': name,
                                      **{k: row[k] for k in ('split', 'trajectory_index',
                                                             'start_node', 'target_node',
                                                             'predicted_node', 'path_length',
                                                             'confidence')}})
        for entry in training.history:
            curves.append({'target_seed': target, 'budget': updates, 'baseline': name, **entry})
        diag = {'target_seed': target, 'budget': updates, 'baseline': name,
                'trainable_parameters': n_params, 'best_update': training.best_update}
        if name == 'linear_accumulator_2d':
            diag.update(accumulator_geometry(model.baseline))
        diagnostics.append(diag)

        checkpoint_dir = out / 'checkpoints' / name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / 'best_validation.pt'
        torch.save({'schema_version': SCHEMA_VERSION,
                    'model_state': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    'baseline': name, 'target_seed': target, 'budget': updates,
                    'run_seed': run_seed, 'loader_seed': loader_seed,
                    'best_update': training.best_update,
                    'dataset_manifest_hash': manifest['manifest_hash']}, checkpoint_path)
        checkpoints.append({'target_seed': target, 'budget': updates, 'baseline': name,
                            'best_update': training.best_update})
        acc = {m['split']: m['accuracy'] for m in metrics if m['baseline'] == name}
        print(target, updates, name, 'best', training.best_update,
              'ID', round(acc['id_seen_short'], 4), 'long', round(acc['seen_long'], 4), flush=True)

    fullscale.write_csv(out / 'metrics.csv', metrics)
    fullscale.write_csv(out / 'per_length.csv', per_length)
    fullscale.write_csv(out / 'update_curves.csv', curves)
    fullscale.write_csv(out / 'condition_diagnostics.csv', diagnostics)
    fullscale.write_csv(out / 'checkpoint_manifest.csv', checkpoints)
    fullscale.write_csv(out / 'long_trajectory_rows.csv', long_rows)
    fullscale.atomic_json(out / 'COMPLETE.json', {
        'schema_version': SCHEMA_VERSION, 'status': 'complete', 'target_seed': target,
        'budget': updates, 'baselines': list(BASELINES),
        'dataset_manifest_hash': manifest['manifest_hash']})
    return 0

def main() -> int:
    parser = argparse.ArgumentParser(description='E0.1 non-recurrent baselines')
    parser.add_argument('--target-seed', type=int)
    parser.add_argument('--updates', type=int)
    parser.add_argument('--output-root', type=Path, default=OUTPUT_ROOT)
    parser.add_argument('--targets', type=int, nargs='+')
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    if args.preflight:
        return preflight(args)
    if args.target_seed is None or args.updates is None:
        parser.error('--target-seed and --updates are required')
    return train_cell(args)

if __name__ == '__main__':
    sys.exit(main())
