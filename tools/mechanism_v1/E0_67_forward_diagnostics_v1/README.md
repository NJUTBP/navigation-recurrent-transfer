# E0.6 / E0.7 / E0.4 — forward diagnostics of the Phase F models

Forward passes only, no training. Three diagnostics share the same forward passes over the Phase F target models.

## Diagnostics

**E0.6, use of recurrence.** Ablation modes: `intact` (reference for Gate R); `no_recurrence` (W_rec set to zero inside a context manager and restored on exit); `last_1`, `last_2`, `last_3` (relation inputs before the last k real steps set to zero; the recurrent map still runs over those steps, so this isolates the contribution of early inputs, not shorter paths); `shuffle_order` (the real steps of each trial permuted independently; the endpoint is invariant under permutation because the task is a commutative group action). Drops are normalized to chance, (acc − acc_ablated)/(acc − 1/64). Shuffles are fixed by `stable_seed32('e067_shuffle', target, split)` and identical across conditions and sources.

**E0.7, error displacement.** For `seen_long` and `heldout_long`, predicted and true endpoints are mapped to lattice coordinates (node → (row, col) = divmod(node, 8), from `fullscale_common.BridgeEpisode`); E‖e‖, the signed E[e] and its SD per path length, and 2-D displacement histograms per split. The collector tests navigation against matched on the slope of E‖e‖ over path length.

**E0.4, state diagnostics.** Per step (`hidden_seq` is taken after the ReLU): `hidden_norm`, `active_fraction`, `signal_vs_zero_input`, `drive_ratio_linear` = ‖g W_in u_t‖ / ‖W_rec h_{t−1}‖ and `delta_rectified` = ‖h_t − ReLU(W_rec h_{t−1})‖ / ‖ReLU(W_rec h_{t−1})‖; and `delta_by_gain`, δ(g) at the model's own start state for g ∈ {1, 3, 10, 32, 100}.

## Model reconstruction and Gate R

Phase F frozen-core checkpoints omit `grid_rnn.*`. `rebuild_model` rebuilds the navigation core, the matched core (`stable_seed32('matched_random', target)`) or the fresh-random core (`stable_seed32('phaseF_fresh_random_core', source, target)`), loads the checkpoint with `strict=False`, requires that unexpected keys are empty and all missing keys are `grid_rnn.*`, and cross-checks the navigation checksum and seeds stored in the checkpoint. Gate R re-evaluates the intact model on the four test splits and compares with `metrics.csv` (`GATE_R_TOLERANCE = 2e-3`; `gate_r.csv`, `gate_r_failures` and `gate_r_worst_abs_difference` in `COMPLETE.json`). The results of a cell are interpretable only if Gate R passes.

Default conditions: `navigation_trainable_scalar_gain`, `matched_random_trainable_scalar_gain`, `fresh_random_trainable_scalar_gain`, `navigation_fixed_gain10_frozen`, `matched_random_fixed_gain10_frozen` (`--conditions` to change). Checkpoints are read from Phase F; `--checkpoint-root` can point to the E0.9a tree, which has the same layout. 10 sources × 5 targets = 50 cells.

## Usage

```bash
python tools/mechanism_v1/E0_67_forward_diagnostics_v1/scripts/run_forward_diagnostics.py --preflight
python tools/mechanism_v1/E0_67_forward_diagnostics_v1/scripts/run_forward_diagnostics.py --source-seed 6101 --target-seed 84001
python tools/mechanism_v1/E0_67_forward_diagnostics_v1/scripts/collect_forward_diagnostics.py
python tools/mechanism_v1/E0_67_forward_diagnostics_v1/scripts/analyze_forward_posthoc.py
```

`analyze_forward_posthoc.py` (post hoc; reads `aggregate/`, writes `aggregate_posthoc/FORWARD_POSTHOC.md`): A1 absolute accuracy under each ablation, navigation minus matched; A2 exact accuracy of an ideal integrator under `last_k` (from the rebuilt splits; `--skip-ideal` omits it) and the excess of each condition, and the fraction of shuffled sequences that leave the lattice or contain held-out bigrams; A3 stationarity (per-step norm retention, δ at step 12 relative to step 3, first-step δ); A4 error structure.

Limitations: state diagnostics use the first 512 `seen_long` trials (`--state-episodes`); `delta_by_gain` uses the first relation vector at h0 (for the state-averaged version use `delta_rectified` of `state_diagnostics.csv`); histograms are aggregated per split.

Output: `rerun_best_validation/followups/mechanism_v1/E0_67_forward_diagnostics/`. Figures: 7a, S8.
