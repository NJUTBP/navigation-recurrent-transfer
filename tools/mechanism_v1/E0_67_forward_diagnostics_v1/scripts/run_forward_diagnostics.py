#!/usr/bin/env python3
"""E0.6 / E0.7 / E0.4 driver: forward-only diagnostics on stored checkpoints.

One invocation handles one (source, target) cell and every selected condition.
Nothing is trained.  The three analyses share the same forward passes:

  E0.6  recurrence ablation.  Accuracy on the four test splits under
        intact / no_recurrence / last_1 / last_2 / last_3 / shuffle_order,
        with the chance-normalised drop (acc - acc_ablated) / (acc - 1/64).
        The 4x4 task has this control (1.000 -> 0.101); the 8x8 does not, and at
        a learned gain near 48-94 a reviewer will ask whether the recurrence is
        used at all.

  E0.7  long-horizon error displacement.  For seen_long and heldout_long, the
        predicted node is converted to integer lattice coordinates and compared
        with the true node.  Reports E||e||, E[e] (a vector) and the joint
        displacement histogram per path length.  Phase F showed the matched
        control is confidently wrong on long paths (accuracy 0.458, confidence
        0.824, ECE 0.366), which points at systematic drift rather than decay to
        noise.  A drift shows up as E[e] growing with L in a consistent
        direction; isotropic diffusion shows up as E||e|| growing while E[e]
        stays near zero.

  E0.4  state diagnostics.  From one forward with hidden_seq: ||h_t||, the ReLU
        active fraction, the signal relative to a zero-input run, the linear
        drive ratio, and the rectified relative displacement delta.  delta is
        also swept over a gain grid, which is the measurement E5b' needs.

Usage
-----
    "$PYTHON_BIN" run_forward_diagnostics.py --source-seed 6101 --target-seed 84001
    "$PYTHON_BIN" run_forward_diagnostics.py --preflight
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from forward_common import (
    ABLATION_MODES, CHANCE, DEFAULT_CONDITIONS, DELTA_GAIN_GRID, GATE_R_TOLERANCE,
    N_LEVELS, N_NODES, OUTPUT_ROOT, PHASEF_OUTPUT, PHASEF_PROTOCOL, PROJECT_ROOT,
    SCHEMA_VERSION, STATE_DIAGNOSTIC_EPISODES, TEST_SPLITS, ZeroedRecurrence,
    apply_ablation, core_weights, find_checkpoints, load_phasef_modules,
    node_to_lattice, rebuild_model,
)


def read_stored_accuracy(checkpoint_root: Path, source: int, target: int) -> dict:
    """Accuracy the original run recorded, used as the Gate R reference."""
    import csv
    stored: dict[tuple[str, str], float] = {}
    base = checkpoint_root / f'source_{source}' / f'target_{target}'
    for path in sorted(base.glob('group_*/metrics.csv')):
        with path.open('r', encoding='utf-8') as handle:
            for row in csv.DictReader(handle):
                try:
                    stored[(row['condition'], row['split'])] = float(row['accuracy'])
                except (KeyError, TypeError, ValueError):
                    continue
    return stored


def preflight(args) -> int:
    report: dict[str, Any] = {'schema_version': SCHEMA_VERSION}
    fullscale, phasef = load_phasef_modules()
    report['fullscale_common'] = str(Path(fullscale.__file__).resolve())
    report['run_group'] = str(Path(phasef.__file__).resolve())

    protocol = json.loads(PHASEF_PROTOCOL.read_text(encoding='utf-8'))
    report['sources'] = protocol['all_sources']
    report['targets'] = protocol['target_split_seeds']
    report['conditions'] = list(args.conditions)
    report['ablation_modes'] = list(ABLATION_MODES)
    report['delta_gain_grid'] = list(DELTA_GAIN_GRID)

    specs = phasef.condition_specs()
    unknown = [c for c in args.conditions if c not in specs]
    if unknown:
        raise RuntimeError(f'conditions not in the Phase F spec table: {unknown}')

    missing = []
    for source in protocol['all_sources']:
        for target in protocol['target_split_seeds']:
            found = find_checkpoints(args.checkpoint_root, source, target, args.conditions)
            absent = [c for c in args.conditions if c not in found]
            if absent:
                missing.append({'source': source, 'target': target, 'absent': absent})
    report['cells_with_missing_checkpoints'] = missing
    if missing:
        raise RuntimeError(f'{len(missing)} cells are missing checkpoints; first: {missing[0]}')

    sample = find_checkpoints(args.checkpoint_root,
                              protocol['all_sources'][0],
                              protocol['target_split_seeds'][0], args.conditions)
    report['example_checkpoints'] = {k: str(v) for k, v in sample.items()}

    # lattice convention, asserted rather than assumed
    ids = np.arange(N_NODES)
    lattice = node_to_lattice(ids)
    if not (lattice[:, 0].max() == N_LEVELS - 1 and lattice[:, 1].max() == N_LEVELS - 1):
        raise RuntimeError('lattice conversion does not span the 8x8 grid')
    report['lattice_check'] = 'divmod(node, 8) spans the full 8x8 grid'

    report['status'] = 'preflight_ok'
    print(json.dumps(report, indent=2))
    return 0


def evaluate(model, loader, device, *, mode: str, generator=None,
             want_predictions: bool = False):
    """Forward the whole loader under one ablation mode."""
    import torch

    correct = 0
    total = 0
    predictions: list[np.ndarray] = []
    truths: list[np.ndarray] = []
    lengths_all: list[np.ndarray] = []
    context = ZeroedRecurrence(model) if mode == 'no_recurrence' else None
    if context is not None:
        context.__enter__()
    try:
        with torch.no_grad():
            for batch in loader:
                start_node = batch['start_node'].to(device)
                velocities = batch['velocities'].to(device)
                lengths = batch['lengths'].to(device)
                start_coords = batch['start_coords'].to(device)
                target_node = batch['target_node'].to(device)
                velocities = apply_ablation(mode, velocities, lengths, generator=generator)
                outputs = model(start_node=start_node, velocities=velocities,
                                lengths=lengths, start_coords=start_coords,
                                validate_inputs=False)
                predicted = outputs['node_logits'].argmax(dim=-1)
                correct += int((predicted == target_node).sum().item())
                total += int(target_node.numel())
                if want_predictions:
                    predictions.append(predicted.cpu().numpy())
                    truths.append(target_node.cpu().numpy())
                    lengths_all.append(lengths.cpu().numpy())
    finally:
        if context is not None:
            context.__exit__(None, None, None)

    accuracy = correct / max(total, 1)
    if not want_predictions:
        return accuracy, None
    return accuracy, (np.concatenate(predictions), np.concatenate(truths),
                      np.concatenate(lengths_all))


def state_diagnostics(model, loader, device, *, max_episodes: int):
    """Per-step state statistics from hidden_seq, plus the zero-input reference."""
    import torch

    w_hh, w_in = core_weights(model)
    gain = float(model.gain().detach().cpu()) if hasattr(model, 'gain') else 1.0
    accumulate: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    seen = 0
    with torch.no_grad():
        for batch in loader:
            if seen >= max_episodes:
                break
            start_node = batch['start_node'].to(device)
            velocities = batch['velocities'].to(device)
            lengths = batch['lengths'].to(device)
            start_coords = batch['start_coords'].to(device)

            real = model(start_node=start_node, velocities=velocities, lengths=lengths,
                         start_coords=start_coords, validate_inputs=False)
            zero = model(start_node=start_node, velocities=torch.zeros_like(velocities),
                         lengths=lengths, start_coords=start_coords, validate_inputs=False)
            hidden = real['hidden_seq']          # [B, T, H], post ReLU
            hidden_zero = zero['hidden_seq']
            h_prev = torch.cat([real['h0'].unsqueeze(1), hidden[:, :-1]], dim=1)

            # zero-input one-step state from the same h_{t-1}: ReLU(W_hh h_{t-1})
            recurrent = torch.relu(torch.matmul(h_prev, w_hh.t()))
            drive = torch.matmul(velocities * gain, w_in.t())
            linear_recurrent = torch.matmul(h_prev, w_hh.t())

            steps = hidden.shape[1]
            positions = torch.arange(steps, device=device).view(1, -1)
            mask = positions < lengths.view(-1, 1)
            for t in range(steps):
                keep = mask[:, t]
                if not bool(keep.any()):
                    continue
                h_t = hidden[keep, t]
                h_zero = hidden_zero[keep, t]
                rec = recurrent[keep, t]
                record = accumulate[t + 1]
                record['hidden_norm'].append(float(h_t.norm(dim=-1).mean()))
                record['active_fraction'].append(float((h_t > 0).float().mean()))
                record['signal_vs_zero_input'].append(float((h_t - h_zero).norm(dim=-1).mean()))
                denom_rec = rec.norm(dim=-1).clamp_min(1e-9)
                record['delta_rectified'].append(
                    float(((h_t - rec).norm(dim=-1) / denom_rec).mean()))
                record['drive_ratio_linear'].append(float(
                    (drive[keep, t].norm(dim=-1)
                     / linear_recurrent[keep, t].norm(dim=-1).clamp_min(1e-9)).mean()))
                record['n'].append(float(keep.sum()))
            seen += int(start_node.shape[0])

    rows = []
    for step in sorted(accumulate):
        record = accumulate[step]
        weights = np.asarray(record['n'], dtype=np.float64)
        row = {'path_step': step, 'n_episodes': float(weights.sum()), 'gain': gain}
        for key, values in record.items():
            if key == 'n':
                continue
            row[key] = float(np.average(np.asarray(values, dtype=np.float64), weights=weights))
        rows.append(row)
    return rows


def delta_by_gain(model, loader, device, *, gains, max_episodes: int):
    """Rectified relative displacement at step 1 as a function of input gain.

    delta(g) = E_h ||ReLU(W_hh h + g W_in u) - ReLU(W_hh h)|| / ||ReLU(W_hh h)||

    computed on the model's own start states.  This is the quantity E5b' predicts
    the required operating gain from, and it is free here because the start
    states come from the same forward.
    """
    import torch

    w_hh, w_in = core_weights(model)
    rows = []
    with torch.no_grad():
        collected: list[tuple] = []
        seen = 0
        for batch in loader:
            if seen >= max_episodes:
                break
            start_node = batch['start_node'].to(device)
            velocities = batch['velocities'].to(device)
            start_coords = batch['start_coords'].to(device)
            outputs = model(start_node=start_node,
                            velocities=torch.zeros_like(velocities),
                            lengths=batch['lengths'].to(device),
                            start_coords=start_coords, validate_inputs=False)
            collected.append((outputs['h0'], velocities[:, 0]))
            seen += int(start_node.shape[0])
        if not collected:
            return rows
        h0 = torch.cat([c[0] for c in collected], dim=0)
        u = torch.cat([c[1] for c in collected], dim=0)
        recurrent_linear = torch.matmul(h0, w_hh.t())
        recurrent = torch.relu(recurrent_linear)
        denom = recurrent.norm(dim=-1).clamp_min(1e-9)
        for gain in gains:
            driven = torch.relu(recurrent_linear + torch.matmul(u * float(gain), w_in.t()))
            delta = ((driven - recurrent).norm(dim=-1) / denom)
            rows.append({'gain': float(gain), 'n_states': int(h0.shape[0]),
                         'delta_mean': float(delta.mean()),
                         'delta_sd': float(delta.std(unbiased=False)),
                         'active_fraction_driven': float((driven > 0).float().mean()),
                         'active_fraction_zero_input': float((recurrent > 0).float().mean())})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description='E0.6 / E0.7 / E0.4 forward diagnostics')
    parser.add_argument('--source-seed', type=int)
    parser.add_argument('--target-seed', type=int)
    parser.add_argument('--checkpoint-root', type=Path, default=PHASEF_OUTPUT)
    parser.add_argument('--conditions', nargs='+', default=list(DEFAULT_CONDITIONS))
    parser.add_argument('--state-episodes', type=int, default=STATE_DIAGNOSTIC_EPISODES)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()

    if args.preflight:
        return preflight(args)
    if args.source_seed is None or args.target_seed is None:
        parser.error('--source-seed and --target-seed are required')

    import torch

    fullscale, phasef = load_phasef_modules()
    protocol = json.loads(PHASEF_PROTOCOL.read_text(encoding='utf-8'))
    source, target = int(args.source_seed), int(args.target_seed)

    out = OUTPUT_ROOT / f'source_{source}' / f'target_{target}'
    if (out / 'COMPLETE.json').exists():
        print('already complete', out, flush=True)
        return 0
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f'partial output exists: {out}; inspect before rerun')
    out.mkdir(parents=True, exist_ok=True)

    hist = fullscale.import_historical()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    pools_nodes, manifest = fullscale.build_fullscale_split(protocol, target)
    expected = protocol['expected_manifest_hashes'][str(target)]
    if manifest['manifest_hash'] != expected:
        raise RuntimeError(f'manifest hash mismatch {manifest["manifest_hash"]} != {expected}')
    pools = {k: fullscale.to_historical_episodes(v, hist) for k, v in pools_nodes.items()}
    fullscale.atomic_json(out / 'dataset_manifest.json', manifest)

    nav_path = PROJECT_ROOT / 'sources' / f'nav_seed_{source}' / 'artifacts' / 'navigation' / 'best.pt'
    nav_hh, nav_ih, nav_meta = hist['load_navigation_grid_weights'](nav_path)
    matched_seed = fullscale.stable_seed32('matched_random', target)
    matched_hh, matched_ih, _ = hist['make_matched_random_weights'](
        nav_hh, nav_ih, seed=matched_seed,
        iterations=8, singular_rank=16, probe_count=64, strict=True)

    model_cfg = protocol['model']
    batch_size = int(protocol['batch_size'])
    loader_seed = fullscale.stable_seed32('phaseF_loader', target, protocol['training_examples'])
    stored = read_stored_accuracy(args.checkpoint_root, source, target)
    checkpoints = find_checkpoints(args.checkpoint_root, source, target, args.conditions)
    absent = [c for c in args.conditions if c not in checkpoints]
    if absent:
        raise RuntimeError(f'missing checkpoints for {absent} at source {source} target {target}')

    ablation_rows: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    displacement_rows: list[dict[str, Any]] = []
    histogram_rows: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []
    delta_rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []

    for condition in args.conditions:
        model, spec, payload, missing_keys = rebuild_model(
            checkpoints[condition], hist=hist, phasef=phasef, fullscale=fullscale,
            device=device, source=source, target=target, model_cfg=model_cfg,
            nav_hh=nav_hh, nav_ih=nav_ih, nav_meta=nav_meta,
            matched_hh=matched_hh, matched_ih=matched_ih, matched_seed=matched_seed)
        gain = float(model.gain().detach().cpu()) if hasattr(model, 'gain') else 1.0
        diagnostics.append({
            'source_seed': source, 'target_seed': target, 'condition': condition,
            'origin': spec['origin'], 'kind': spec['kind'], 'gain': gain,
            'checkpoint': str(checkpoints[condition]),
            'best_update': payload.get('best_update'),
            'n_core_keys_reconstructed': len(missing_keys)})

        generator = torch.Generator(device=device)

        for split in TEST_SPLITS:
            loader = fullscale.make_loader(pools[split], hist, batch_size, False, loader_seed + 2)
            for mode in ABLATION_MODES:
                generator.manual_seed(fullscale.stable_seed32('e067_shuffle', target, split))
                want = (mode == 'intact' and split in ('seen_long', 'heldout_long'))
                accuracy, extra = evaluate(model, loader, device, mode=mode,
                                           generator=generator, want_predictions=want)
                ablation_rows.append({
                    'source_seed': source, 'target_seed': target, 'condition': condition,
                    'origin': spec['origin'], 'gain': gain, 'split': split,
                    'mode': mode, 'accuracy': accuracy})
                if mode == 'intact':
                    reference = stored.get((condition, split))
                    gate_rows.append({
                        'source_seed': source, 'target_seed': target,
                        'condition': condition, 'split': split,
                        'recomputed_accuracy': accuracy,
                        'stored_accuracy': reference if reference is not None else float('nan'),
                        'abs_difference': (abs(accuracy - reference)
                                           if reference is not None else float('nan')),
                        'within_tolerance': (bool(abs(accuracy - reference) <= GATE_R_TOLERANCE)
                                             if reference is not None else False)})
                if want and extra is not None:
                    predicted, truth, lengths = extra
                    error = node_to_lattice(predicted) - node_to_lattice(truth)
                    for length in sorted(set(int(x) for x in lengths)):
                        keep = lengths == length
                        subset = error[keep]
                        displacement_rows.append({
                            'source_seed': source, 'target_seed': target,
                            'condition': condition, 'origin': spec['origin'],
                            'split': split, 'path_length': int(length),
                            'n_trials': int(keep.sum()),
                            'accuracy': float((subset == 0).all(axis=1).mean()),
                            'mean_abs_displacement': float(
                                np.linalg.norm(subset, axis=1).mean()),
                            'mean_displacement_row': float(subset[:, 0].mean()),
                            'mean_displacement_col': float(subset[:, 1].mean()),
                            'sd_displacement_row': float(subset[:, 0].std()),
                            'sd_displacement_col': float(subset[:, 1].std())})
                    uniques, counts = np.unique(error, axis=0, return_counts=True)
                    for (row_shift, col_shift), count in zip(uniques, counts):
                        histogram_rows.append({
                            'source_seed': source, 'target_seed': target,
                            'condition': condition, 'split': split,
                            'row_shift': int(row_shift), 'col_shift': int(col_shift),
                            'count': int(count)})

        diagnostic_loader = fullscale.make_loader(
            pools['seen_long'], hist, batch_size, False, loader_seed + 2)
        for row in state_diagnostics(model, diagnostic_loader, device,
                                     max_episodes=int(args.state_episodes)):
            state_rows.append({'source_seed': source, 'target_seed': target,
                               'condition': condition, 'origin': spec['origin'],
                               'split': 'seen_long', **row})
        delta_loader = fullscale.make_loader(
            pools['seen_long'], hist, batch_size, False, loader_seed + 2)
        for row in delta_by_gain(model, delta_loader, device, gains=DELTA_GAIN_GRID,
                                 max_episodes=int(args.state_episodes)):
            delta_rows.append({'source_seed': source, 'target_seed': target,
                               'condition': condition, 'origin': spec['origin'],
                               'learned_gain': gain, **row})

        intact_long = [r['accuracy'] for r in ablation_rows
                       if r['condition'] == condition and r['split'] == 'seen_long'
                       and r['mode'] == 'intact']
        no_rec_long = [r['accuracy'] for r in ablation_rows
                       if r['condition'] == condition and r['split'] == 'seen_long'
                       and r['mode'] == 'no_recurrence']
        print(source, target, condition, 'gain', round(gain, 3),
              'seen_long intact', round(intact_long[0], 4) if intact_long else 'NA',
              'no_recurrence', round(no_rec_long[0], 4) if no_rec_long else 'NA',
              flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    fullscale.write_csv(out / 'ablation.csv', ablation_rows)
    fullscale.write_csv(out / 'gate_r.csv', gate_rows)
    fullscale.write_csv(out / 'error_displacement.csv', displacement_rows)
    fullscale.write_csv(out / 'error_histogram.csv', histogram_rows)
    fullscale.write_csv(out / 'state_diagnostics.csv', state_rows)
    fullscale.write_csv(out / 'delta_by_gain.csv', delta_rows)
    fullscale.write_csv(out / 'condition_diagnostics.csv', diagnostics)

    failed = [r for r in gate_rows if not r['within_tolerance']]
    fullscale.atomic_json(out / 'COMPLETE.json', {
        'schema_version': SCHEMA_VERSION, 'status': 'complete',
        'source_seed': source, 'target_seed': target,
        'conditions': list(args.conditions),
        'ablation_modes': list(ABLATION_MODES),
        'dataset_manifest_hash': manifest['manifest_hash'],
        'gate_r_tolerance': GATE_R_TOLERANCE,
        'gate_r_failures': len(failed),
        'gate_r_worst_abs_difference': (max(r['abs_difference'] for r in gate_rows)
                                        if gate_rows else None)})
    if failed:
        print(f'WARNING: Gate R failed for {len(failed)} (condition, split) pairs; '
              'the core reconstruction is suspect', file=sys.stderr)
    return 0


if __name__ == '__main__':
    sys.exit(main())
