#!/usr/bin/env python3
"""Shared pieces for E0.6 / E0.7 / E0.4, the frozen-core forward diagnostics.

No training happens here.  Every number comes from re-evaluating checkpoints that
Phase F (or E0.9a) already produced.

The one hard problem this module solves
---------------------------------------
Frozen-core checkpoints do NOT contain the recurrent core.  ``compact_state_dict``
in the Phase F driver strips every ``grid_rnn.*`` key and records
``checkpoint_storage_mode = 'frozen_core_omitted_reconstruct_from_origin_and_seed'``.
So the core has to be rebuilt from the origin and the seeds before the stored
interface and readout can be loaded on top of it:

    nav_hh, nav_ih, nav_meta = load_navigation_grid_weights(nav_best_pt)
    matched_hh, matched_ih, _ = make_matched_random_weights(
        nav_hh, nav_ih, seed=stable_seed32('matched_random', target),
        iterations=8, singular_rank=16, probe_count=64, strict=True)
    random_core_seed = stable_seed32('phaseF_fresh_random_core', source, target)

``rebuild_model`` does that, then loads the compact state with ``strict=False``
and asserts that the ONLY missing keys are the core keys and that there are no
unexpected keys.  Provenance is cross-checked against the fields the checkpoint
itself stores (``navigation_checkpoint_sha256``, ``matched_random_seed``,
``run_seed``).

Gate R
------
Reconstruction correctness is verified behaviourally, not by inspection: the
intact model is re-evaluated on every test split and the accuracy is compared to
the value the original run wrote into ``metrics.csv``.  A mismatch above
``GATE_R_TOLERANCE`` means the core was rebuilt wrongly and every downstream
number in this package is meaningless.  Read Gate R before anything else.

Lattice geometry (confirmed from fullscale_common.BridgeEpisode)
----------------------------------------------------------------
    node n  ->  (row, col) = divmod(n, 8),  coords = (row, col) * 0.02
    velocity for n_a -> n_b is (row_b - row_a, col_b - col_a) * 0.02
So a predicted node id converts to integer lattice coordinates with no lookup
table, and the error vector of E0.7 is exact.
"""
from __future__ import annotations
import os

import sys
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(os.environ.get('NRT_PROJECT_ROOT', str(Path(__file__).resolve().parents[4])))
PHASEF_PACKAGE = PROJECT_ROOT / 'tools' / 'hard8x8_phaseF_fullscale_v1'
PHASEF_SCRIPTS = PHASEF_PACKAGE / 'scripts'
PHASEF_PROTOCOL = PHASEF_PACKAGE / 'PROTOCOL.json'
PHASEF_OUTPUT = (PROJECT_ROOT / 'rerun_best_validation' / 'hard8x8_diagnostics'
                 / 'audit_v1' / 'phaseF_fullscale_v1')

MECHANISM_ROOT = PROJECT_ROOT / 'rerun_best_validation' / 'followups' / 'mechanism_v1'
OUTPUT_ROOT = MECHANISM_ROOT / 'E0_67_forward_diagnostics'

SCHEMA_VERSION = 'e0_67_forward_diagnostics_v1'
N_LEVELS = 8
N_NODES = 64
CHANCE = 1.0 / N_NODES
GATE_R_TOLERANCE = 2e-3

TEST_SPLITS = ('id_seen_short', 'heldout_short', 'seen_long', 'heldout_long')

DEFAULT_CONDITIONS = (
    'navigation_trainable_scalar_gain',
    'matched_random_trainable_scalar_gain',
    'fresh_random_trainable_scalar_gain',
    'navigation_fixed_gain10_frozen',
    'matched_random_fixed_gain10_frozen',
)

# E0.6 ablation modes.  ``shuffle_order`` is an addition to the plan: the task is
# an abelian group action, so an additive solution must be invariant to the order
# of the relations within a trial.  It costs one extra forward and measures that
# invariance directly on the trained model.
ABLATION_MODES = ('intact', 'no_recurrence', 'last_1', 'last_2', 'last_3',
                  'shuffle_order')

# E0.4 gain sweep for the rectified relative displacement delta.  This is the
# measurement E5b' needs; it is produced here because it is the same forward.
DELTA_GAIN_GRID = (1.0, 3.0, 10.0, 32.0, 100.0)
STATE_DIAGNOSTIC_EPISODES = 512


def load_phasef_modules():
    if not PHASEF_SCRIPTS.is_dir():
        raise RuntimeError(f'Phase F scripts not found: {PHASEF_SCRIPTS}')
    path = str(PHASEF_SCRIPTS)
    if path not in sys.path:
        sys.path.insert(0, path)
    import fullscale_common as fullscale  # noqa: E402
    import run_group as phasef  # noqa: E402
    resolved = Path(phasef.__file__).resolve()
    if PHASEF_SCRIPTS.resolve() not in resolved.parents:
        raise RuntimeError(f'wrong run_group import: {resolved}')
    return fullscale, phasef


def find_checkpoints(checkpoint_root: Path, source: int, target: int,
                     conditions: Iterable[str]) -> dict[str, Path]:
    """Locate best_validation.pt for each condition under any group directory.

    Phase F and E0.9a use the same ``group_<name>/checkpoints/<condition>/`` layout
    but different group names, so the group is globbed rather than assumed.
    """
    base = checkpoint_root / f'source_{source}' / f'target_{target}'
    found: dict[str, Path] = {}
    for condition in conditions:
        hits = sorted(base.glob(f'group_*/checkpoints/{condition}/best_validation.pt'))
        if len(hits) > 1:
            raise RuntimeError(f'ambiguous checkpoint for {condition}: {hits}')
        if hits:
            found[condition] = hits[0]
    return found


def node_to_lattice(node_ids):
    """node id -> integer (row, col).  Confirmed: divmod(n, N_LEVELS)."""
    import numpy as np
    ids = np.asarray(node_ids, dtype=np.int64)
    return np.stack([ids // N_LEVELS, ids % N_LEVELS], axis=-1)


def rebuild_model(checkpoint_path: Path, *, hist, phasef, fullscale, device,
                  source: int, target: int, model_cfg: dict[str, Any],
                  nav_hh, nav_ih, nav_meta, matched_hh, matched_ih, matched_seed: int):
    """Rebuild the frozen core, then load the stored interface and readout."""
    import torch

    payload = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    spec = payload['condition_spec']
    storage_mode = payload.get('checkpoint_storage_mode')
    if storage_mode != 'frozen_core_omitted_reconstruct_from_origin_and_seed':
        raise RuntimeError(
            f'unexpected storage mode {storage_mode!r} in {checkpoint_path}; '
            'this package only handles frozen-core checkpoints')

    stored_sha = payload.get('navigation_checkpoint_sha256')
    if stored_sha and stored_sha != nav_meta['checkpoint_sha256']:
        raise RuntimeError(
            f'navigation checkpoint sha mismatch for {checkpoint_path}: '
            f'{stored_sha} != {nav_meta["checkpoint_sha256"]}')
    stored_matched_seed = payload.get('matched_random_seed')
    if stored_matched_seed is not None and int(stored_matched_seed) != int(matched_seed):
        raise RuntimeError(
            f'matched_random_seed mismatch for {checkpoint_path}: '
            f'{stored_matched_seed} != {matched_seed}')

    random_core_seed = fullscale.stable_seed32('phaseF_fresh_random_core', source, target)
    hist['set_seed'](int(payload.get('run_seed', 0)))
    base = hist['build_hard_model'](model_cfg, n_nodes=N_NODES, device=device)
    phasef.configure_origin(
        base, spec, hist, nav_hh=nav_hh, nav_ih=nav_ih,
        matched_hh=matched_hh, matched_ih=matched_ih,
        cell_cls=hist['FactorialCell'], model_cfg=model_cfg, device=device,
        random_core_seed=random_core_seed)
    model = phasef.wrap_model(base, spec, hist, device)

    result = model.load_state_dict(payload['model_state'], strict=False)
    unexpected = list(result.unexpected_keys)
    missing = list(result.missing_keys)
    if unexpected:
        raise RuntimeError(f'unexpected keys when loading {checkpoint_path}: {unexpected[:6]}')
    not_core = [k for k in missing
                if '.grid_rnn.' not in k and not k.startswith('grid_rnn.')]
    if not_core:
        raise RuntimeError(
            f'missing non-core keys when loading {checkpoint_path}: {not_core[:6]}')
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, spec, payload, missing


def core_weights(model):
    """Return (W_hh, W_in) of the frozen core as they are actually used."""
    cell = model.grid_rnn.cell
    return cell.h_proj.weight, cell.in_proj.weight


class ZeroedRecurrence:
    """Context manager that temporarily sets W_hh to zero."""

    def __init__(self, model) -> None:
        self.model = model

    def __enter__(self):
        import torch
        weight = self.model.grid_rnn.cell.h_proj.weight
        self.saved = weight.detach().clone()
        with torch.no_grad():
            weight.zero_()
        return self.model

    def __exit__(self, *exc):
        import torch
        with torch.no_grad():
            self.model.grid_rnn.cell.h_proj.weight.copy_(self.saved)
        self.saved = None
        return False


def apply_ablation(mode: str, velocities, lengths, generator=None):
    """Return a modified copy of ``velocities`` for the given ablation mode.

    ``last_k`` zeroes every relation input before the last k real steps of each
    row.  The recurrent map still runs on those steps, so this isolates the
    contribution of the early inputs rather than shortening the trajectory.

    ``shuffle_order`` permutes the real steps of each row independently.  The
    endpoint is the sum of the relations and therefore unchanged, so accuracy
    under this mode measures order invariance directly.
    """
    import torch

    if mode in ('intact', 'no_recurrence'):
        return velocities
    out = velocities.clone()
    batch, steps, _ = out.shape
    positions = torch.arange(steps, device=out.device).view(1, -1)
    real = positions < lengths.view(-1, 1)

    if mode.startswith('last_'):
        k = int(mode.split('_')[1])
        keep = positions >= (lengths.view(-1, 1) - k)
        out = out * (keep & real).unsqueeze(-1).to(out.dtype)
        return out

    if mode == 'shuffle_order':
        if generator is None:
            raise ValueError('shuffle_order needs a generator for reproducibility')
        noise = torch.rand(batch, steps, generator=generator, device=out.device)
        noise = noise.masked_fill(~real, float('inf'))
        order = noise.argsort(dim=1)
        gather = order.unsqueeze(-1).expand(-1, -1, out.shape[-1])
        return torch.gather(out, 1, gather)

    raise ValueError(f'unknown ablation mode: {mode}')
