#!/usr/bin/env python3
"""Shared pieces for E0.1, the non-recurrent baselines on the 8x8 task.

Why this experiment
-------------------
The paper compares recurrent cores with one another.  It never states how far an
explicit additive algorithm gets on the same splits.  The task is an abelian
group action: the endpoint equals the start plus the sum of the relations, so
the start identity and the relation counts determine the answer exactly.

Both baselines contain the exact solution in their function class.  With lattice
coordinates p_j, the logit

    logit_j = 2 p_j . (p_s + D c) - |p_j|^2 = |p_s + D c|^2 - |p_s + D c - p_j|^2

peaks at the node whose coordinate equals p_s + D c, which is the target.  For
ActionCountLinearBaseline it is a linear map of [onehot(s), c]; for
LinearAccumulatorBaseline it is the readout of the 2-D state p_s + D c.  The
preflight installs these oracle weights and requires accuracy 1.0 on every
split, which also verifies the relation mapping below.  Whatever the trained
baselines miss is therefore a learning or extrapolation shortfall, not capacity.

Relation mapping (confirmed in code)
------------------------------------
fullscale_common.action_id and gpm_cmap.data.graph_tasks.hard_pathint.
velocity_action_labels agree: row+1 -> 0, row-1 -> 1, col+1 -> 2, col-1 -> 3,
zero velocity -> padding.  ``velocity_to_onehot`` is a vectorised copy of
velocity_action_labels; the preflight checks it episode by episode.
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
OUTPUT_ROOT = MECHANISM_ROOT / 'E0_1_nonrecurrent_baselines'
E09A_ROOT = MECHANISM_ROOT / 'E0_9a_extended_budget'

SCHEMA_VERSION = 'e0_1_nonrecurrent_baselines_v1'
N_LEVELS = 8
N_NODES = 64
N_ACTIONS = 4
BUDGETS = (8000, 32000)
BASELINES = ('action_count_linear', 'linear_accumulator_2d')
TEST_SPLITS = ('id_seen_short', 'heldout_short', 'seen_long', 'heldout_long')
EVAL_SPLITS = ('train_8000', 'validation_seen_short') + TEST_SPLITS
LONG_SPLITS = ('seen_long', 'heldout_long')
ACTION_DISPLACEMENT = ((1, 0), (-1, 0), (0, 1), (0, -1))

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

def derive_protocol(updates: int) -> dict[str, Any]:
    base = json.loads(PHASEF_PROTOCOL.read_text(encoding='utf-8'))
    protocol = copy.deepcopy(base)
    protocol['schema_version'] = SCHEMA_VERSION
    protocol['status'] = 'nonrecurrent_algorithmic_reference'
    protocol['derived_from'] = {
        'phasef_protocol': str(PHASEF_PROTOCOL),
        'phasef_schema_version': base['schema_version'],
        'phasef_max_gradient_updates': base['train']['max_gradient_updates'],
    }
    protocol['output_root'] = str(OUTPUT_ROOT)
    protocol['train'] = dict(base['train'])
    protocol['train']['max_gradient_updates'] = int(updates)
    protocol['train']['minimum_updates'] = int(updates)
    protocol['budgets'] = list(BUDGETS)
    protocol['baselines'] = list(BASELINES)
    assert protocol['training_examples'] == base['training_examples']
    n_evals = int(updates) // int(protocol['train']['eval_every_updates'])
    if int(protocol['train']['early_stopping_patience_evals']) <= n_evals:
        raise RuntimeError('early stopping could fire before the budget is exhausted')
    protocol['n_evaluations'] = n_evals
    return protocol

def velocity_to_onehot(velocities, atol: float = 1e-10):
    import torch
    import torch.nn.functional as F
    axis = velocities.abs().argmax(dim=-1)
    value = velocities.gather(-1, axis.unsqueeze(-1)).squeeze(-1)
    label = axis * 2 + (value <= 0).long()
    valid = velocities.norm(dim=-1) > atol
    onehot = F.one_hot(label, num_classes=N_ACTIONS)
    return (onehot * valid.unsqueeze(-1)).to(velocities.dtype)

def build_baseline(name: str):
    from gpm_cmap.models.symbolic_controls import (
        ActionCountLinearBaseline, LinearAccumulatorBaseline,
    )
    if name == 'action_count_linear':
        return ActionCountLinearBaseline(N_NODES, N_ACTIONS)
    if name == 'linear_accumulator_2d':
        return LinearAccumulatorBaseline(N_NODES, N_ACTIONS, latent_dim=2)
    raise ValueError(name)

def make_wrapper_class():
    import torch
    from torch import nn

    class RelationOneHotModel(nn.Module):
        """Feeds Phase F displacement batches to a one-hot relation baseline."""

        supports_prevalidated_inputs = True

        def __init__(self, baseline) -> None:
            super().__init__()
            self.baseline = baseline

        @property
        def start_init(self):
            return self.baseline.start_init

        @property
        def n_nodes(self):
            return self.baseline.n_nodes

        def recurrent_l2(self):
            return torch.zeros((), device=next(self.parameters()).device)

        def forward(self, start_node, velocities, lengths=None, start_coords=None, *,
                    validate_inputs=True):
            return self.baseline(start_node, velocity_to_onehot(velocities),
                                 lengths=lengths, validate_inputs=validate_inputs)

    return RelationOneHotModel

def lattice_coordinates():
    import torch
    ids = torch.arange(N_NODES)
    return torch.stack([ids // N_LEVELS, ids % N_LEVELS], dim=-1).to(torch.float32)

def install_oracle(name: str, baseline) -> None:
    import torch
    p = lattice_coordinates()
    d = torch.tensor(ACTION_DISPLACEMENT, dtype=torch.float32).t()
    bias = -(p ** 2).sum(dim=1)
    with torch.no_grad():
        if name == 'action_count_linear':
            weight = torch.cat([2.0 * p @ p.t(), 2.0 * p @ d], dim=1)
            baseline.classifier.weight.copy_(weight)
            baseline.classifier.bias.copy_(bias)
        elif name == 'linear_accumulator_2d':
            baseline.start_embed.weight.copy_(p)
            baseline.action_adapter.weight.copy_(d)
            baseline.node_head.weight.copy_(2.0 * p)
            baseline.node_head.bias.copy_(bias)
        else:
            raise ValueError(name)

def accumulator_geometry(baseline) -> dict[str, float]:
    """Affine fit of the learned 2-D codes to lattice coordinates."""
    import numpy as np
    p = lattice_coordinates().numpy().astype(np.float64)
    e = baseline.start_embed.weight.detach().cpu().numpy().astype(np.float64)
    design = np.concatenate([p, np.ones((N_NODES, 1))], axis=1)
    coef, *_ = np.linalg.lstsq(design, e, rcond=None)
    residual = e - design @ coef
    total = e - e.mean(axis=0, keepdims=True)
    r2 = 1.0 - float((residual ** 2).sum()) / max(float((total ** 2).sum()), 1e-12)
    linear = coef[:2]
    a = baseline.action_adapter.weight.detach().cpu().numpy().astype(np.float64)
    predicted = np.asarray(ACTION_DISPLACEMENT, dtype=np.float64) @ linear
    action_fit = 1.0 - float(((a.T - predicted) ** 2).sum()) / max(
        float(((a.T - a.T.mean(axis=0)) ** 2).sum()), 1e-12)
    singular = np.linalg.svd(a, compute_uv=False)
    return {
        'start_embedding_affine_r2': r2,
        'action_matrix_consistency_r2': action_fit,
        'action_matrix': a.tolist(),
        'action_singular_1': float(singular[0]),
        'action_singular_2': float(singular[1]),
        'antipodal_row_pair': float(np.linalg.norm(a[:, 0] + a[:, 1])
                                    / max(np.linalg.norm(a[:, 0]) + np.linalg.norm(a[:, 1]), 1e-12)),
        'antipodal_col_pair': float(np.linalg.norm(a[:, 2] + a[:, 3])
                                    / max(np.linalg.norm(a[:, 2]) + np.linalg.norm(a[:, 3]), 1e-12)),
    }
