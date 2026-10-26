#!/usr/bin/env python3
"""Shared pieces for E5b', the rectified-displacement prediction of operating gain.

Quantity (plan v4, E5b')
------------------------
    delta_core(g) = E_h || ReLU(W h + g W_in u) - ReLU(W h) || / || ReLU(W h) ||

over a FIXED state set h and the four 8x8 relation vectors u.  The E0.4
``delta_by_gain`` evaluated this on the learned start state of each trained
model, which is not fixed: the network has no bias, so (h, g) -> (c h, c g)
leaves every trajectory unchanged up to scale and delta depends on g/||h||
only.  Two state sets are therefore used here, neither of which has been
shaped by downstream learning.

H_nav   (formal)  States of held-out navigation trajectories.  Both cores run
                  the same velocity sequences from the same place-cell start
                  state h0 = init_encoder(place), each on its own dynamics.
                  States from step NAV_BURN_IN + 1 on; the same (trajectory,
                  step) positions are sampled for both cores.
H_init  (check)   The downstream model at initialisation.  With run_seed fixed
                  the start embedding and projection are identical for the
                  navigation and matched conditions, so both cores start from
                  the same 64 states before any learning.

Both blocks also record how the state norm evolves along a trajectory.  The
autonomous map F(h) = ReLU(W h) is positively homogeneous, so the per-step
retention ||F(h)|| / ||h|| depends only on the direction of h.  No input gain
and no rescaling of h0 can change it; it is a property of the core.
"""
from __future__ import annotations
import os

import copy
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(os.environ.get('NRT_PROJECT_ROOT', str(Path(__file__).resolve().parents[4])))
PHASEF_PACKAGE = PROJECT_ROOT / 'tools' / 'hard8x8_phaseF_fullscale_v1'
PHASEF_SCRIPTS = PHASEF_PACKAGE / 'scripts'
PHASEF_PROTOCOL = PHASEF_PACKAGE / 'PROTOCOL.json'
PHASEF_OUTPUT = (PROJECT_ROOT / 'rerun_best_validation' / 'hard8x8_diagnostics'
                 / 'audit_v1' / 'phaseF_fullscale_v1')
MECHANISM_ROOT = PROJECT_ROOT / 'rerun_best_validation' / 'followups' / 'mechanism_v1'
OUTPUT_ROOT = MECHANISM_ROOT / 'E5b_delta_prediction'
E5C_ROOT = MECHANISM_ROOT / 'E5c_gain_sweep'
E09A_ROOT = MECHANISM_ROOT / 'E0_9a_extended_budget'

SCHEMA_VERSION = 'e5b_delta_prediction_v1'
ORIGINS = ('navigation', 'matched_random')
GAIN_GRID = (1, 2, 4, 8, 16, 32, 64, 128, 256)
INIT_SPLITS = ('train_8000', 'seen_long')
INIT_EPISODES = 512
NAV_TRAJECTORIES = 512
NAV_BURN_IN = 2
MAX_NAV_STATES = 4096
STATE_CHUNK = 1024
RETENTION_STEPS = (3, 12)

def load_phasef_modules():
    if not PHASEF_SCRIPTS.is_dir():
        raise RuntimeError(f'Phase F scripts not found: {PHASEF_SCRIPTS}')
    path = str(PHASEF_SCRIPTS)
    if path not in sys.path:
        sys.path.insert(0, path)
    import fullscale_common as fullscale
    import run_group as phasef
    resolved = Path(phasef.__file__).resolve()
    if PHASEF_SCRIPTS.resolve() not in resolved.parents:
        raise RuntimeError(f'wrong run_group import: {resolved}')
    return fullscale, phasef

def load_protocol() -> dict[str, Any]:
    return json.loads(PHASEF_PROTOCOL.read_text(encoding='utf-8'))

def relation_vectors(protocol, device):
    import torch
    scale = float(protocol['velocity_scale'])
    return torch.tensor([[scale, 0.0], [-scale, 0.0], [0.0, scale], [0.0, -scale]],
                        dtype=torch.float32, device=device)

def relative_displacement(w_hh, w_in, states, relations, gains) -> list[dict[str, float]]:
    """delta(g) averaged over states and relations, computed in state chunks."""
    import torch
    totals = {float(g): [] for g in gains}
    active_zero, active_driven = [], {float(g): [] for g in gains}
    with torch.no_grad():
        for start in range(0, states.shape[0], STATE_CHUNK):
            h = states[start:start + STATE_CHUNK]
            linear = h @ w_hh.t()
            base = torch.relu(linear)
            denom = base.norm(dim=-1).clamp_min(1e-9)
            active_zero.append((base > 0).float().mean(dim=-1))
            for g in gains:
                drives = (relations * float(g)) @ w_in.t()
                for drive in drives:
                    driven = torch.relu(linear + drive)
                    totals[float(g)].append((driven - base).norm(dim=-1) / denom)
                    active_driven[float(g)].append((driven > 0).float().mean(dim=-1))
    rows = []
    zero = torch.cat(active_zero)
    for g in gains:
        values = torch.cat(totals[float(g)])
        rows.append({'gain': float(g), 'n_state_relation_pairs': int(values.numel()),
                     'delta_mean': float(values.mean()), 'delta_sd': float(values.std(unbiased=False)),
                     'delta_median': float(values.median()),
                     'active_fraction_driven': float(torch.cat(active_driven[float(g)]).mean()),
                     'active_fraction_zero_input': float(zero.mean())})
    return rows

def autonomous_retention(w_hh, states) -> dict[str, float]:
    import torch
    ratios = []
    with torch.no_grad():
        for start in range(0, states.shape[0], STATE_CHUNK):
            h = states[start:start + STATE_CHUNK]
            ratios.append(torch.relu(h @ w_hh.t()).norm(dim=-1) / h.norm(dim=-1).clamp_min(1e-9))
    values = torch.cat(ratios)
    return {'autonomous_retention_mean': float(values.mean()),
            'autonomous_retention_median': float(values.median()),
            'n_states': int(values.numel())}

def log_slope(steps, values) -> float:
    import numpy as np
    x = np.asarray(steps, dtype=np.float64)
    y = np.log(np.maximum(np.asarray(values, dtype=np.float64), 1e-12))
    return float(np.polyfit(x, y, 1)[0]) if x.size >= 2 else float('nan')

def trajectory_profile(hidden, h0, w_hh, lengths=None) -> list[dict[str, float]]:
    """Per-step norm, rectified relative displacement and active fraction."""
    import torch
    rows = []
    with torch.no_grad():
        prev = torch.cat([h0.unsqueeze(1), hidden[:, :-1]], dim=1)
        recurrent = torch.relu(prev @ w_hh.t())
        steps = hidden.shape[1]
        if lengths is None:
            mask = torch.ones(hidden.shape[:2], dtype=torch.bool, device=hidden.device)
        else:
            mask = torch.arange(steps, device=hidden.device).view(1, -1) < lengths.view(-1, 1)
        for t in range(steps):
            keep = mask[:, t]
            if not bool(keep.any()):
                continue
            h_t, rec = hidden[keep, t], recurrent[keep, t]
            rows.append({'step': t + 1, 'n': int(keep.sum()),
                         'hidden_norm': float(h_t.norm(dim=-1).mean()),
                         'delta_rectified': float(((h_t - rec).norm(dim=-1)
                                                   / rec.norm(dim=-1).clamp_min(1e-9)).mean()),
                         'active_fraction': float((h_t > 0).float().mean())})
    return rows

def build_navigation_model(payload, device):
    from gpm_cmap.models.grid_rnn import GridNavigationModel
    cfg = payload['config']
    model = GridNavigationModel(
        velocity_dim=cfg['model']['velocity_dim'], hidden_dim=cfg['model']['hidden_dim'],
        place_dim=cfg['data']['place_cells'], activation=cfg['model']['activation'],
        nonnegative=cfg['model']['nonnegative'], cell_type=cfg['model'].get('cell_type', 'rnn'),
        dropout=cfg['model'].get('dropout', 0.0), use_bias=cfg['model'].get('use_bias', False),
        init_relu=cfg['model'].get('init_relu', False))
    model.load_state_dict(payload['model_state'], strict=True)
    return model.to(device).eval()

def navigation_trajectories(cfg, n_trajectories: int):
    """Held-out navigation trajectories, built exactly as exp1's validation set."""
    import torch
    from gpm_cmap.data.nav_env.dataset import NavDatasetConfig, NavTrajectoryDataset
    from gpm_cmap.data.nav_env.open_field import OpenFieldConfig
    data = cfg['data']
    open_field = OpenFieldConfig(
        arena_size=data['arena_size'], dt=data.get('dt', 0.05),
        speed_scale=data.get('speed_scale', 0.13), sigma_theta=data.get('sigma_theta', 5.0),
        max_speed=data.get('max_speed', 0.5), border_region=data.get('border_region', 0.03),
        boundary_mode=data.get('boundary_mode', 'avoid'))
    dataset = NavTrajectoryDataset(NavDatasetConfig(
        num_trajectories=int(n_trajectories), seq_len=data['seq_len'], open_field=open_field,
        place_cells=data['place_cells'], place_sigma=data['place_sigma'],
        arena_size=data['arena_size'], seed=cfg['seed'], trajectory_seed=cfg['seed'] + 9999,
        place_seed=cfg['seed'], use_dog=data.get('use_dog', True),
        surround_scale=data.get('surround_scale', 2.0),
        surround_scale_mode=data.get('surround_scale_mode', 'variance')))
    velocities = torch.stack([dataset[i]['velocities'] for i in range(len(dataset))])
    init_place = torch.stack([dataset[i]['init_place'] for i in range(len(dataset))])
    return velocities, init_place

def with_core(model, w_hh, w_in):
    import torch
    clone = copy.deepcopy(model)
    with torch.no_grad():
        clone.grid_rnn.cell.h_proj.weight.copy_(w_hh)
        clone.grid_rnn.cell.in_proj.weight.copy_(w_in)
    return clone.eval()
