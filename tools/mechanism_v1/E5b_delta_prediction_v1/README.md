# E5b′ — rectified displacement and the required gain ratio

Measures, on two fixed state sets,

    delta_core(g) = E_h ‖ReLU(W h + g W_in u) − ReLU(W h)‖ / ‖ReLU(W h)‖

with u the four 8 × 8 relation vectors (±0.02), and tests the prediction log2(g_matched / g_nav) = log2(delta_nav / delta_matched).

## State sets

Apart from the readout the models have no bias, so (h, g) → (c h, c g) scales the whole trajectory by c and δ depends only on g/‖h‖. The δ of E0.4 is measured at the learned start states, whose scale is itself learned, so E5b′ uses state sets fixed before learning:

| Set | Definition | Role |
|---|---|---|
| H_nav | states of navigation-validation trajectories: both cores run the same velocity sequences from the same place-cell start, states taken at the same positions after step 3 | primary |
| H_init | the 64 start states of the downstream models at initialization (same run_seed, identical for navigation and matched) | consistency check |

## Decision rule

Stated in the module docstring of `collect_delta.py`: the observation is the shift of the threshold crossing at `id_seen_short` accuracy 0.90 from the E5c post hoc analysis; the error e = observed − predicted is computed per source (n = 10). Support if the 95 % bootstrap interval lies within ±0.5 log2 units (half the E5c grid step); falsification if exact p < 0.05 and the interval excludes 0; otherwise inconclusive.

Also reported: the autonomous retention ‖ReLU(W h)‖/‖h‖ of both cores on H_nav and the log-slope of the norm along trajectories. F(h) = ReLU(W h) is positively homogeneous, so retention depends only on the state direction and is a property of the core.

## Self-checks per cell

- The cores loaded into the model equal the rebuilt weights element by element.
- The H_init start states are taken from the model's own forward pass; per-batch recomputation must agree within 1e-4 of the largest start-state value (recorded as `h0_max_relative_deviation` in `COMPLETE.json`).
- The recurrent weights in the navigation checkpoint equal those returned by `load_navigation_grid_weights`.

## Usage

The collector reads the E5c post hoc output, so run `E5c_gain_sweep_v1/scripts/analyze_gain_sweep_posthoc.py` first.

```bash
python tools/mechanism_v1/E5b_delta_prediction_v1/scripts/run_delta.py --preflight
python tools/mechanism_v1/E5b_delta_prediction_v1/scripts/run_delta.py --source-seed 6101 --target-seed 84001
python tools/mechanism_v1/E5b_delta_prediction_v1/scripts/collect_delta.py
```

Output: `rerun_best_validation/followups/mechanism_v1/E5b_delta_prediction/`. `scripts/delta_common.py` (state sets, core loading) is also used by E234, E6 and C1. Figure: S8.
