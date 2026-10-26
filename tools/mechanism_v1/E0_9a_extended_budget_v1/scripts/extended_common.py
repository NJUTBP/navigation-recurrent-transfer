#!/usr/bin/env python3
"""Shared pieces for E0.9a, the extended-budget convergence control.

Design rule
-----------
This package does NOT reimplement Phase F.  It imports the frozen Phase F
driver and reuses its model construction, weight installation, wrapping,
seeding and evaluation verbatim, then changes exactly two things:

  * ``train.max_gradient_updates`` and ``train.minimum_updates`` 8000 -> 32000;
  * the condition list, cut to the four arms that decide the manuscript framing.

Everything else, including ``training_examples: 8000``, is copied from the
canonical ``PROTOCOL.json``.  That matters: the Phase F run and loader seeds are

    stable_seed32('phaseF_run',    target, protocol['training_examples'])
    stable_seed32('phaseF_loader', target, protocol['training_examples'])

so keeping ``training_examples`` fixed leaves both seeds unchanged.  The model
initialisation, the minibatch order and the dataset are therefore identical to
Phase F, and the first 8000 updates of this run are a re-execution of the Phase F
trajectory rather than an independent sample.  ``collect_extended.py`` checks
that explicitly and treats a mismatch as a configuration fault.

``eval_every_updates`` is also left at 200.  Evaluation runs under ``no_grad``
with a non-shuffling loader, so it cannot perturb training, but keeping it equal
to Phase F removes one free parameter and lets the reproduction check run on the
full evaluation grid instead of a subsampled one.
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

OUTPUT_ROOT = (PROJECT_ROOT / 'rerun_best_validation' / 'followups' / 'mechanism_v1'
               / 'E0_9a_extended_budget')
SCHEMA_VERSION = 'e0_9a_extended_budget_v1'

EXTENDED_UPDATES = 32000
TARGET_SEEDS = (84001, 84002, 84003)
GROUPS: dict[str, tuple[str, ...]] = {
    'scalar_pair': (
        'navigation_trainable_scalar_gain',
        'matched_random_trainable_scalar_gain',
    ),
    'identity_pair': (
        'navigation_identity_frozen',
        'matched_random_identity_frozen',
    ),
}
GAIN_TRACE_EVERY = 50


def load_phasef_modules():
    """Import the frozen Phase F driver and its helpers."""
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


def derive_protocol(updates: int = EXTENDED_UPDATES,
                    targets: tuple[int, ...] = TARGET_SEEDS) -> dict[str, Any]:
    """Phase F protocol with only the update budget and condition list changed."""
    base = json.loads(PHASEF_PROTOCOL.read_text(encoding='utf-8'))
    protocol = copy.deepcopy(base)

    protocol['schema_version'] = SCHEMA_VERSION
    protocol['status'] = 'extended_budget_convergence_control'
    protocol['derived_from'] = {
        'phasef_protocol': str(PHASEF_PROTOCOL),
        'phasef_schema_version': base['schema_version'],
        'phasef_max_gradient_updates': base['train']['max_gradient_updates'],
    }
    protocol['output_root'] = str(OUTPUT_ROOT)

    protocol['train'] = dict(base['train'])
    protocol['train']['max_gradient_updates'] = int(updates)
    protocol['train']['minimum_updates'] = int(updates)
    # eval cadence and the disabled early-stopping rule are inherited unchanged
    protocol['train']['eval_every_updates'] = int(base['train']['eval_every_updates'])
    protocol['train']['early_stopping_patience_evals'] = int(
        base['train']['early_stopping_patience_evals'])

    protocol['conditions_by_group'] = {name: list(conditions)
                                       for name, conditions in GROUPS.items()}
    protocol['target_split_seeds'] = list(targets)
    protocol['expected_manifest_hashes'] = {
        str(t): base['expected_manifest_hashes'][str(t)] for t in targets
    }
    # training_examples is a SEED COMPONENT and must not change
    assert protocol['training_examples'] == base['training_examples']

    n_evals = updates // protocol['train']['eval_every_updates']
    if protocol['train']['early_stopping_patience_evals'] <= n_evals:
        raise RuntimeError(
            'early stopping could fire before the budget is exhausted: '
            f'patience={protocol["train"]["early_stopping_patience_evals"]} '
            f'<= n_evals={n_evals}'
        )
    protocol['n_evaluations'] = n_evals
    return protocol


class GainTrace:
    """Record the scalar gain during training without touching the trainer.

    ``log_gain`` receives a gradient exactly once per optimiser update, so a
    backward hook on that leaf gives an update-indexed trace.  Validation runs
    under ``no_grad`` and therefore never fires the hook.  The recorded value is
    the parameter BEFORE the step at that index, which is the convention used in
    the emitted CSV.
    """

    def __init__(self, model, every: int = GAIN_TRACE_EVERY) -> None:
        import torch
        self.records: list[dict[str, float]] = []
        self.count = 0
        self.every = max(1, int(every))
        self.handle = None
        self._torch = torch
        parameter = getattr(model, 'log_gain', None)
        if parameter is None or not getattr(parameter, 'requires_grad', False):
            self.parameter = None
            return
        self.parameter = parameter
        self.handle = parameter.register_hook(self._hook)

    def _hook(self, grad):
        self.count += 1
        if self.count == 1 or self.count % self.every == 0:
            value = float(self.parameter.detach().cpu())
            self.records.append({
                'update': self.count,
                'log_gain': value,
                'gain': float(self._torch.exp(self._torch.tensor(value))),
            })
        return grad

    def close(self, final_model=None) -> list[dict[str, float]]:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
        if final_model is not None and getattr(final_model, 'log_gain', None) is not None:
            value = float(final_model.log_gain.detach().cpu())
            self.records.append({
                'update': self.count,
                'log_gain': value,
                'gain': float(self._torch.exp(self._torch.tensor(value))),
                })
        return self.records
