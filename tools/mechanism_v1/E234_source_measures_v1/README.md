# E2–E4 — source-side measures of the cores

For every (source, target) cell, the navigation core and the matched core of that target are measured on the navigation state set H_nav (see `E5b_delta_prediction_v1`).

| Measure | Definition |
|---|---|
| E2: S_opp, S_orth, S_cons, r_eff | one-step displacement geometry on centred displacements, with z scores against a random-direction null |
| E3: C_comm, C_inv | commutativity and inverse consistency of two-step compositions, relative to F²(h, 0) |
| E4: R²(L), R_ret, R_ret_diff | ridge regression of the net displacement fitted on L ≤ 5 only (L = 6–12 is extrapolation); R_ret = R²(12)/R²(5) from h_L, R_ret_diff the same from h_L − F^L(h0, 0) |
| auxiliary | O(L), c(t, 12), signal and norm, active fraction |

Working points per core: A (g = 1), g3, g10, B8 (learned gain of Phase F), B32 (learned gain of E0.9a, targets 84001–84003) and Bstar (the gain at which the one-step δ on H_nav equals the first-step δ of the downstream model from E0.4). All are computed; the primary measure is set by the constants `PRIMARY_METRIC = 'R_ret_diff'` and `PRIMARY_WORKING_POINT = 'Bstar'` of the collectors, and the tests are evaluated by `verdicts_v1`.

Dependencies (checked in the preflight): `E5b_delta_prediction_v1/scripts/delta_common.py`, the Phase F package, the Phase F and E0.9a output trees, and `E0_67_forward_diagnostics/aggregate/all_state_diagnostics.csv`.

Run-time checks: the permutation and signs of the matched core are rebuilt from its recorded seed and flip fraction and must reproduce the matched weights exactly; the hand-unrolled navigation trajectory must agree with the navigation model's own forward pass (relative difference ≤ 1e−4).

```bash
python tools/mechanism_v1/E234_source_measures_v1/scripts/run_source_measures.py --preflight
python tools/mechanism_v1/E234_source_measures_v1/scripts/run_source_measures.py --source-seed 6101 --target-seed 84001
python tools/mechanism_v1/E234_source_measures_v1/scripts/collect_source_measures.py
```

50 cells (forward passes and ridge regressions, a few minutes each on one GPU). Output: `rerun_best_validation/followups/mechanism_v1/E234_source_measures/` (`aggregate/SOURCE_MEASURES_RESULT.md`). `scripts/source_measures.py` is also used by E6. Figure: S10.
