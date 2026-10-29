# Extended 8 × 8 task ("Phase F")

Frozen-core transfer on an 8 × 8 lattice with an orthogonal composition × horizon test design: the experiment behind Figure 6 and Supplementary Figures S5 and S6. The protocol is fixed in `PROTOCOL.json` and the analysis in `PREREGISTERED_ANALYSIS.md`.

## Design

- 10 navigation sources (6101–6110) × 5 fresh target splits (84001–84005) × 15 training conditions = 750 training cells.
- 8,000 unique training trajectories per split and a fixed budget of 8,000 gradient updates per cell; 1,000 validation trajectories (`validation_seen_short`); four independent test sets of 2,000 trajectories: seen composition × short horizon (`id_seen_short`), held-out composition × short (`heldout_short`), seen × long (`seen_long`) and held-out × long (`heldout_long`).
- Condition groups (`--group`): `fixed_scale` (frozen core with fixed relation-input gain 1, 3 or 10), `learned_interface` (learned scalar gain or full 2 × 2 interface, including fresh-random cores), `full_finetune` (recurrent core trained).
- Seeds: run_seed = `stable_seed32('phaseF_run', target, 8000)`, loader_seed = `stable_seed32('phaseF_loader', target, 8000)`, matched core `stable_seed32('matched_random', target)`, fresh-random core `stable_seed32('phaseF_fresh_random_core', source, target)`. Splits are rebuilt from the protocol and checked against `expected_manifest_hashes`.

## Analysis rules

- The inferential unit is the navigation source; the five target splits are averaged within source.
- Primary contrast: `navigation_trainable_scalar_gain` − `matched_random_trainable_scalar_gain` on `id_seen_short`, two-sided exhaustive exact sign-flip test over the 10 sources; the four test splits form the generalization family (Holm), and six secondary contrasts on `id_seen_short` form a second family (`PREREGISTERED_ANALYSIS.md`).
- Test sets are never used for checkpoint selection.
- Fresh-random controls use source–target-specific random recurrent cores with paired start and readout initialization.
- Frozen-core checkpoints omit the reconstructible recurrent matrices; full-core fine-tuning checkpoints keep the full recurrent state.

## Files

| File | Content |
|---|---|
| `PROTOCOL.json`, `PREREGISTERED_ANALYSIS.md` | Protocol and prespecified analysis |
| `scripts/fullscale_common.py` | Split construction, seeds, data loaders and import of the model code from `code/` |
| `scripts/run_group.py` | Training and evaluation of one source × split × condition group |
| `scripts/collect_fullscale.py` | Aggregate tables and the primary and secondary contrast families |
| `tests/` | Contracts for protocol, seeds, splits, model wrappers and condition construction (`python -m pytest tools/hard8x8_phaseF_fullscale_v1/tests`; all but the protocol test need PyTorch) |

## Usage

```bash
python tools/hard8x8_phaseF_fullscale_v1/scripts/run_group.py --source-seed 6101 --target-seed 84001 --group learned_interface
python tools/hard8x8_phaseF_fullscale_v1/scripts/collect_fullscale.py
```

Output: `rerun_best_validation/hard8x8_diagnostics/audit_v1/phaseF_fullscale_v1/` (one folder per source, target and group, and `aggregate/`); `tools/FigureA5_PhaseF_workflow_v1/` copies the aggregate into figure source data. Cluster job scripts are not included; each job was one source × split × group.
