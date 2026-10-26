#!/usr/bin/env python3
"""E6 planned separability control: refit navigation interface on each frozen rho core."""
from __future__ import annotations
import os

import argparse
import copy
import json
import math
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(os.environ.get('NRT_PROJECT_ROOT', str(Path(__file__).resolve().parents[4])))
MECH_TOOLS = PROJECT_ROOT / 'tools' / 'mechanism_v1'
for sibling in ('E6_rho_scan_v1', 'E0_9a_extended_budget_v1', 'E5b_delta_prediction_v1'):
    path = str(MECH_TOOLS / sibling / 'scripts')
    if path not in sys.path:
        sys.path.insert(0, path)

import extended_common  # noqa: E402
import rho_core  # noqa: E402
import run_rho_scan as e6  # noqa: E402
import delta_common  # noqa: E402
import separability_common as common  # noqa: E402


def _dataset(cfg: dict, n_trajectories: int, trajectory_seed: int):
    from gpm_cmap.data.nav_env.dataset import NavDatasetConfig, NavTrajectoryDataset
    from gpm_cmap.data.nav_env.open_field import OpenFieldConfig
    data = cfg['data']
    open_field = OpenFieldConfig(
        arena_size=data['arena_size'], dt=data.get('dt', 0.05),
        speed_scale=data.get('speed_scale', 0.13), sigma_theta=data.get('sigma_theta', 5.0),
        max_speed=data.get('max_speed', 0.5), border_region=data.get('border_region', 0.03),
        boundary_mode=data.get('boundary_mode', 'avoid'))
    return NavTrajectoryDataset(NavDatasetConfig(
        num_trajectories=int(n_trajectories), seq_len=int(data['seq_len']), open_field=open_field,
        place_cells=int(data['place_cells']), place_sigma=float(data['place_sigma']),
        arena_size=float(data['arena_size']), seed=int(cfg['seed']), trajectory_seed=int(trajectory_seed),
        place_seed=int(cfg['seed']), use_dog=data.get('use_dog', True),
        surround_scale=data.get('surround_scale', 2.0),
        surround_scale_mode=data.get('surround_scale_mode', 'variance')))


def _loaders(cfg: dict, source: int, target: int, fullscale):
    import torch
    from torch.utils.data import DataLoader
    train_seed = fullscale.stable_seed32('e6_nav_probe_train_data', source)
    loader_seed = fullscale.stable_seed32('e6_nav_probe_loader', source, target)
    train_ds = _dataset(cfg, common.TRAIN_TRAJECTORIES, train_seed)
    val_ds = _dataset(cfg, common.VAL_TRAJECTORIES, int(cfg['seed']) + 9999)
    train_loader = DataLoader(
        train_ds, batch_size=common.BATCH_SIZE, shuffle=True, num_workers=0,
        generator=torch.Generator().manual_seed(int(loader_seed)))
    val_loader = DataLoader(val_ds, batch_size=common.BATCH_SIZE, shuffle=False, num_workers=0)
    return train_loader, val_loader, val_ds.place_encoder, int(train_seed), int(loader_seed)


def _aligned_model(payload: dict, rho: float, target: int, device, fullscale):
    import torch
    model = delta_common.build_navigation_model(payload, device)
    hidden = int(model.grid_rnn.cell.h_proj.weight.shape[0])
    core_seed = fullscale.stable_seed32('e6_rho_core', int(target))
    nav_hh = model.grid_rnn.cell.h_proj.weight.detach().cpu().clone()
    nav_ih = model.grid_rnn.cell.in_proj.weight.detach().cpu().clone()
    hh, ih, meta = rho_core.signed_similarity_core(nav_hh, nav_ih, seed=core_seed, rho=float(rho))
    permutation, flipped = rho_core.flip_units(hidden, float(rho), int(core_seed))
    index = torch.as_tensor(permutation, dtype=torch.long)
    signs = torch.ones(hidden, dtype=nav_hh.dtype)
    signs[torch.as_tensor(flipped, dtype=torch.long)] = -1.0
    with torch.no_grad():
        init_w = model.init_encoder.weight.detach().cpu().clone()
        head_w = model.place_head.weight.detach().cpu().clone()
        if model.init_encoder.bias is not None or model.place_head.bias is not None:
            raise RuntimeError('navigation interface unexpectedly has bias')
        model.grid_rnn.cell.h_proj.weight.copy_(hh.to(device))
        model.grid_rnn.cell.in_proj.weight.copy_(ih.to(device))
        model.init_encoder.weight.copy_((signs[:, None] * init_w.index_select(0, index)).to(device))
        model.place_head.weight.copy_((head_w.index_select(1, index) * signs[None, :]).to(device))
    for p in model.grid_rnn.parameters():
        p.requires_grad_(False)
    for p in model.init_encoder.parameters():
        p.requires_grad_(True)
    for p in model.place_head.parameters():
        p.requires_grad_(True)
    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    if trainable != ['place_head.weight', 'init_encoder.weight'] and set(trainable) != {'place_head.weight', 'init_encoder.weight'}:
        raise RuntimeError(f'unexpected trainable parameters: {trainable}')
    return model, {**meta, 'permutation_checksum': int(np.asarray(permutation, dtype=np.int64).sum())}


def _evaluate(model, loader, place_encoder, device, arena_size: float) -> dict[str, float]:
    import torch
    import torch.nn.functional as F
    model.eval()
    total_loss = 0.0
    total_entropy = 0.0
    total_n = 0
    errors = []
    with torch.no_grad():
        for batch in loader:
            velocities = batch['velocities'].to(device)
            targets = batch['place_targets'].to(device)
            init_place = batch['init_place'].to(device)
            outputs = model(velocities, init_place=init_place)
            pred = outputs['place_pred'].reshape(-1, outputs['place_pred'].shape[-1])
            target = targets.reshape(-1, targets.shape[-1])
            loss_rows = -(target * F.log_softmax(pred, dim=-1)).sum(-1)
            ent_rows = -(target * torch.log(target.clamp_min(1e-30))).sum(-1)
            total_loss += float(loss_rows.sum())
            total_entropy += float(ent_rows.sum())
            total_n += int(loss_rows.numel())
            prob = F.softmax(outputs['place_pred'], dim=-1).reshape(-1, outputs['place_pred'].shape[-1])
            decoded = place_encoder.decode(prob).detach().cpu()
            true_pos = batch['positions'].reshape(-1, 2).cpu()
            errors.append(torch.linalg.vector_norm(decoded - true_pos, dim=-1))
    if total_n == 0:
        raise RuntimeError('empty navigation validation loader')
    loss = total_loss / total_n
    entropy = total_entropy / total_n
    n_place = int(place_encoder.n_cells)
    uniform = float(math.log(n_place))
    denom = uniform - entropy
    frac = (uniform - loss) / denom if denom > 1e-12 else float('nan')
    err = torch.cat(errors)
    return {
        'ce_loss': float(loss), 'target_entropy': float(entropy), 'uniform_loss': float(uniform),
        'frac_loss_captured': float(frac), 'pi_error_mean': float(err.mean()),
        'pi_error_median': float(err.median()), 'pi_error_relative': float(err.mean() / float(arena_size)),
        'n_state_samples': int(total_n),
    }


def _train_interface(model, train_loader, val_loader, place_encoder, cfg, device) -> tuple[dict, list[dict]]:
    import torch
    import torch.nn.functional as F
    params = list(model.init_encoder.parameters()) + list(model.place_head.parameters())
    lr = float(cfg['train']['lr'])
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=0.0)
    grad_clip = float(cfg['train'].get('grad_clip', 0.0))
    arena_size = float(cfg['data']['arena_size'])
    best = _evaluate(model, val_loader, place_encoder, device, arena_size)
    best_epoch = 0
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                  if k.startswith('init_encoder.') or k.startswith('place_head.')}
    curve = [{'epoch': 0, **best}]
    for epoch in range(1, common.EPOCHS + 1):
        model.train()
        for batch in train_loader:
            velocities = batch['velocities'].to(device)
            targets = batch['place_targets'].to(device)
            init_place = batch['init_place'].to(device)
            outputs = model(velocities, init_place=init_place)
            pred = outputs['place_pred'].reshape(-1, outputs['place_pred'].shape[-1])
            target = targets.reshape(-1, targets.shape[-1])
            loss = -(target * F.log_softmax(pred, dim=-1)).sum(-1).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('non-finite navigation interface loss')
            optimizer.zero_grad()
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(params, grad_clip)
            optimizer.step()
        if epoch % common.EVAL_EVERY == 0 or epoch == common.EPOCHS:
            metrics = _evaluate(model, val_loader, place_encoder, device, arena_size)
            curve.append({'epoch': epoch, **metrics})
            if metrics['ce_loss'] < best['ce_loss']:
                best = metrics
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                              if k.startswith('init_encoder.') or k.startswith('place_head.')}
    state = model.state_dict()
    state.update(best_state)
    model.load_state_dict(state, strict=True)
    best = _evaluate(model, val_loader, place_encoder, device, arena_size)
    return {'best_epoch': int(best_epoch), **best}, curve


def preflight() -> int:
    import torch
    fullscale, _ = extended_common.load_phasef_modules()
    hist = fullscale.import_historical()
    protocol = e6.derive_protocol(extended_common.EXTENDED_UPDATES)
    missing = [str(e6.nav_checkpoint(s)) for s in protocol['all_sources'] if not e6.nav_checkpoint(s).is_file()]
    if missing:
        raise RuntimeError(f'missing navigation checkpoints: {missing[:3]}')
    for source in protocol['all_sources']:
        payload = torch.load(e6.nav_checkpoint(source), map_location='cpu', weights_only=False)
        cfg = payload['config']
        if int(cfg['model']['hidden_dim']) != 4096 or int(cfg['data']['seq_len']) != 20:
            raise RuntimeError(f'unexpected source config for {source}')
    nav_hh, nav_ih, _ = hist['load_navigation_grid_weights'](e6.nav_checkpoint(protocol['all_sources'][0]))
    report = rho_core.anchor_report(nav_hh, nav_ih, seed=fullscale.stable_seed32('e6_rho_core', 84001))
    if not report['all_pass']:
        raise RuntimeError('rho core algebra anchor failed')
    print(json.dumps({
        'schema_version': common.SCHEMA_VERSION, 'status': 'preflight_ok', 'torch': torch.__version__,
        'n_cells': len(protocol['all_sources']) * len(protocol['target_split_seeds']) * len(common.RHOS),
        'train_trajectories': common.TRAIN_TRAJECTORIES, 'val_trajectories': common.VAL_TRAJECTORIES,
        'batch_size': common.BATCH_SIZE, 'epochs': common.EPOCHS, 'eval_every': common.EVAL_EVERY,
        'rho_anchor': report,
    }, indent=2))
    return 0


def run_cell(args) -> int:
    import torch
    fullscale, _ = extended_common.load_phasef_modules()
    source, target, rho = int(args.source_seed), int(args.target_seed), float(args.rho)
    protocol = e6.derive_protocol(extended_common.EXTENDED_UPDATES)
    if source not in protocol['all_sources'] or target not in protocol['target_split_seeds'] or rho not in common.RHOS:
        raise ValueError('cell outside navigation separability protocol')
    out = common.cell_dir(Path(args.output_root), source, target, rho)
    if (out / 'COMPLETE.json').exists():
        print('already complete', flush=True)
        return 0
    if out.exists() and any(out.iterdir()):
        raise RuntimeError('partial output exists for this cell; inspect before rerun')
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    payload = torch.load(e6.nav_checkpoint(source), map_location='cpu', weights_only=False)
    cfg = payload['config']
    train_loader, val_loader, place_encoder, train_data_seed, loader_seed = _loaders(cfg, source, target, fullscale)
    original = delta_common.build_navigation_model(payload, device)
    model, core_meta = _aligned_model(payload, rho, target, device, fullscale)
    zero = _evaluate(model, val_loader, place_encoder, device, float(cfg['data']['arena_size']))
    if rho == 0.0:
        reference = _evaluate(original, val_loader, place_encoder, device, float(cfg['data']['arena_size']))
        for key in ('ce_loss', 'frac_loss_captured', 'pi_error_mean'):
            denom = max(abs(float(reference[key])), 1e-12)
            rel = abs(float(zero[key]) - float(reference[key])) / denom
            tol = 1e-4 if key == 'pi_error_mean' else 2e-5
            if rel > tol:
                raise RuntimeError(f'rho0 aligned navigation anchor failed for {key}: {rel}')
    del original
    refit, curve = _train_interface(model, train_loader, val_loader, place_encoder, cfg, device)
    tag = {'schema_version': common.SCHEMA_VERSION, 'source_seed': source, 'target_seed': target,
           'rho': rho, **core_meta, 'train_data_seed': train_data_seed, 'loader_seed': loader_seed,
           'probe_epochs': common.EPOCHS, 'probe_lr': float(cfg['train']['lr'])}
    rows = []
    for phase, metrics in (('zero_shot', zero), ('refit_best', refit)):
        rows.append({**tag, 'phase': phase, **metrics})
    fullscale.write_csv(out / 'navigation_metrics.csv', rows)
    fullscale.write_csv(out / 'refit_curve.csv', [{**tag, **row} for row in curve])
    fullscale.atomic_json(out / 'COMPLETE.json', {
        **tag, 'status': 'complete', 'zero_shot_frac_loss_captured': zero['frac_loss_captured'],
        'refit_frac_loss_captured': refit['frac_loss_captured'], 'zero_shot_pi_error_relative': zero['pi_error_relative'],
        'refit_pi_error_relative': refit['pi_error_relative'], 'best_epoch': refit['best_epoch'],
    })
    print(source, target, rho, 'zero_frac', round(zero['frac_loss_captured'], 4),
          'refit_frac', round(refit['frac_loss_captured'], 4), 'pi_rel', round(refit['pi_error_relative'], 4),
          'best_epoch', refit['best_epoch'], flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description='E6 navigation separability control')
    parser.add_argument('--source-seed', type=int)
    parser.add_argument('--target-seed', type=int)
    parser.add_argument('--rho', type=float)
    parser.add_argument('--output-root', type=Path, default=common.OUTPUT_ROOT)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    if args.preflight:
        return preflight()
    if args.source_seed is None or args.target_seed is None or args.rho is None:
        parser.error('--source-seed, --target-seed and --rho are required')
    return run_cell(args)


if __name__ == '__main__':
    raise SystemExit(main())
