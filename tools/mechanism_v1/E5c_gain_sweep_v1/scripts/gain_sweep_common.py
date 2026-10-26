#!/usr/bin/env python3
"""Shared pieces for E5c, the fixed-gain operating-curve sweep.

Why this experiment
-------------------
E0.9a D3 showed the learned scalar gain is budget dependent: it climbs from 1 to
about 48 by update 8,000 and on to about 74 (navigation) / 94 (matched) by
update 32,000, and navigation and matched actually CROSS at roughly update
3,000-4,000, which is why Phase D at 2,000 updates measured navigation ABOVE
matched while Phase F at 8,000 measured it below.  A learned gain is therefore
not a clean estimate of the operating scale a frozen core requires.

E5c measures the operating curve directly.  The gain is a fixed buffer, only the
start embedding and the endpoint readout are learned, and the budget is the same
32,000 used by E0.9a so that the g = 1 cells already produced by E0.9a
(``navigation_identity_frozen`` and ``matched_random_identity_frozen``) join the
grid without being rerun.

What it decides
---------------
* the shape of accuracy(g) per split, and where it peaks;
* whether the short-horizon and long-horizon optima coincide.  If
  ``g*_long != g*_short`` then the learned interface, which is selected on
  ``validation_seen_short`` (path lengths 1-5 only), is not optimal for the long
  horizon and the long-horizon advantage is an out-of-sample by-product;
* the completion of E0.9a D2.  D2 found that g = 1 at 32,000 updates reaches
  0.8157 and is still climbing at about +0.009 per 1,000 updates, so it did not
  establish an asymptote.  The full accuracy(g) curve shows whether small g sits
  on a genuinely lower plateau or is merely slow.

Design constraints inherited from E0.9a (do not change)
-------------------------------------------------------
* the protocol is derived at run time from the canonical Phase F PROTOCOL.json;
* ``training_examples`` stays 8000 because it is a component of ``run_seed`` and
  ``loader_seed``;
* ``eval_every_updates`` stays 200;
* condition specs use the exact Phase F shape
  ``{'origin', 'kind': 'fixed_gain', 'gain', 'core': 'frozen'}`` so that
  ``configure_origin`` / ``wrap_model`` / ``diagnostic_row`` /
  ``compact_state_dict`` are reused verbatim.
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

MECHANISM_ROOT = PROJECT_ROOT / 'rerun_best_validation' / 'followups' / 'mechanism_v1'
OUTPUT_ROOT = MECHANISM_ROOT / 'E5c_gain_sweep'
E09A_ROOT = MECHANISM_ROOT / 'E0_9a_extended_budget'

SCHEMA_VERSION = 'e5c_gain_sweep_v1'
EXTENDED_UPDATES = 32000
TARGET_SEEDS = (84001, 84002)
ORIGINS = ('navigation', 'matched_random')

# Log2 grid spanning two decades.  g = 1 is NOT here: E0.9a already produced it
# as the two identity conditions at the same budget and the same targets, and the
# collector joins those cells in.
GAIN_GRID = (2, 4, 8, 16, 32, 64, 128, 256)

# The E0.9a conditions that supply the g = 1 anchor.
E09A_IDENTITY = {
    'navigation': 'navigation_identity_frozen',
    'matched_random': 'matched_random_identity_frozen',
}


def condition_name(origin: str, gain: int) -> str:
    """Zero padded so the name never collides with the Phase F gain3 / gain10."""
    return f'{origin}_fixed_gain{int(gain):03d}_frozen'


def build_specs(gain: int) -> dict[str, dict[str, Any]]:
    """Phase F spec shape, one condition per origin at this gain."""
    return {
        condition_name(origin, gain): {
            'origin': origin, 'kind': 'fixed_gain',
            'gain': float(gain), 'core': 'frozen',
        }
        for origin in ORIGINS
    }


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


def derive_protocol(updates: int = EXTENDED_UPDATES,
                    targets: tuple[int, ...] = TARGET_SEEDS) -> dict[str, Any]:
    base = json.loads(PHASEF_PROTOCOL.read_text(encoding='utf-8'))
    protocol = copy.deepcopy(base)

    protocol['schema_version'] = SCHEMA_VERSION
    protocol['status'] = 'fixed_gain_operating_curve'
    protocol['derived_from'] = {
        'phasef_protocol': str(PHASEF_PROTOCOL),
        'phasef_schema_version': base['schema_version'],
        'phasef_max_gradient_updates': base['train']['max_gradient_updates'],
        'g1_anchor_from': str(E09A_ROOT),
    }
    protocol['output_root'] = str(OUTPUT_ROOT)

    protocol['train'] = dict(base['train'])
    protocol['train']['max_gradient_updates'] = int(updates)
    protocol['train']['minimum_updates'] = int(updates)

    protocol['gain_grid'] = list(GAIN_GRID)
    protocol['origins'] = list(ORIGINS)
    protocol['target_split_seeds'] = list(targets)
    protocol['expected_manifest_hashes'] = {
        str(t): base['expected_manifest_hashes'][str(t)] for t in targets
    }
    assert protocol['training_examples'] == base['training_examples']

    n_evals = updates // protocol['train']['eval_every_updates']
    if protocol['train']['early_stopping_patience_evals'] <= n_evals:
        raise RuntimeError(
            'early stopping could fire before the budget is exhausted: '
            f'patience={protocol["train"]["early_stopping_patience_evals"]} '
            f'<= n_evals={n_evals}'
        )
    protocol['n_evaluations'] = n_evals
    protocol['n_cells'] = (len(GAIN_GRID) * len(ORIGINS)
                           * len(protocol['all_sources']) * len(targets))
    return protocol
