# E0.9a — extended training budget

Question: do the Phase F differences reflect faster learning or a different attainable solution? Four Phase F conditions are retrained with 32,000 instead of 8,000 gradient updates.

## Design

- Conditions: `navigation_trainable_scalar_gain` and `matched_random_trainable_scalar_gain` (group `scalar_pair`), `navigation_identity_frozen` and `matched_random_identity_frozen` (group `identity_pair`); 10 sources × targets 84001–84003 × 2 groups = 60 jobs, 120 training cells.
- `run_extended.py` imports the Phase F driver (`run_group.py`, `fullscale_common.py`) and reuses its condition specifications (`condition_specs`, `configure_origin`, `wrap_model`, `diagnostic_row`, `compact_state_dict`) and its model construction, weight loading and evaluation. Only the update budget and the condition list change; the protocol is derived from Phase F `PROTOCOL.json` at run time.
- `training_examples` (8,000) is kept (asserted), so model initialization, minibatch order and data are those of Phase F and the first 8,000 updates re-execute the Phase F trajectory. Training restarts from update 0 because Phase F stores no optimizer state.
- Evaluation every 200 updates (160 evaluations); early stopping cannot trigger, and `stopped_update == 32000` is asserted. A backward hook on `log_gain` records the learned gain every 50 updates without changing training.

## Checks

- `--preflight` (no GPU): Phase F modules are imported from `tools/hard8x8_phaseF_fullscale_v1/scripts`, the derived protocol (budget, number of evaluations, early stopping), the four conditions, the seeds of the three targets, the split manifest hashes and the ten navigation checkpoints; writes `PREFLIGHT.json`.
- Gate 0 (collector): the validation curves are compared with the Phase F `all_update_curves.csv` on the common evaluation grid (maximum and mean absolute deviation). GPU non-determinism allows small drift, not bitwise identity.

## Prespecified readouts

Stated in the module docstring of `collect_extended.py`:

- D1: navigation minus matched scalar gain on `id_seen_short` at 32,000 updates (the other three test splits Holm-adjusted); `phasef_vs_extended.csv` compares 8,000 and 32,000 updates.
- D2: whether `navigation_identity_frozen` (gain 1) at 32,000 updates reaches the Phase F accuracy of `navigation_fixed_gain10_frozen` at 8,000 updates.
- D3: whether both learned gains have levelled off at 32,000 updates (relative growth over the last 10 %) and whether navigation stays below matched.

## Usage

```bash
python tools/mechanism_v1/E0_9a_extended_budget_v1/scripts/run_extended.py --preflight
python tools/mechanism_v1/E0_9a_extended_budget_v1/scripts/run_extended.py --source-seed 6101 --target-seed 84001 --group scalar_pair
python tools/mechanism_v1/E0_9a_extended_budget_v1/scripts/collect_extended.py --phasef-followup data/aggregate_full/hard8x8_phaseF_followup
```

Output: `rerun_best_validation/followups/mechanism_v1/E0_9a_extended_budget/` (cells, `aggregate/EXTENDED_RESULT.md`, `phasef_vs_extended.csv`, `gain_summary.csv`). P2 contains the 60 scalar-gain models (`reevaluate_models.py --family e09a`). Figure: S7.
