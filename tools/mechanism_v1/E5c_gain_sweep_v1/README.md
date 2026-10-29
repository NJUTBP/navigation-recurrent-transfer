# E5c — fixed-gain operating curve

Question: how does transfer depend on the relation-input gain g when g is fixed rather than learned? In E0.9a the learned gain kept growing after accuracy had converged, so it is not a clean estimate of the input scale a core requires.

## Design

- The gain is a buffer of `FixedGainPathIntModel`; only the start embedding and the endpoint readout are learned. Conditions use the Phase F form `{'origin', 'kind': 'fixed_gain', 'gain', 'core': 'frozen'}`, so the Phase F functions `configure_origin`, `wrap_model`, `diagnostic_row` and `compact_state_dict` are reused unchanged (the preflight compares the specification keys with `navigation_fixed_gain3_frozen`).
- Grid g ∈ {2, 4, 8, 16, 32, 64, 128, 256}. The g = 1 point is the pair of E0.9a identity conditions (same budget and targets), joined by the collector.
- 32,000 updates, evaluation every 200, patience 999, `training_examples` 8,000 (asserted): identical to E0.9a.
- 8 gains × 2 origins (navigation, matched) × 10 sources × targets 84001–84002 = 320 cells. Condition names are zero-padded (`navigation_fixed_gain032_frozen`) so that they cannot collide with the Phase F names; the preflight checks this.

## Prespecified readouts

Stated in the module docstring of `collect_gain_sweep.py`:

- Q1: accuracy(g) per test split and origin, including g = 1.
- Q2: whether the optimal gain for long paths differs from that for short paths, log2 g*_long − log2 g*_short per source.
- Q3: log2 g*_matched − log2 g*_nav, the gain ratio independent of the dynamics of gain learning.
- Q4: end-of-budget slope for every gain (`end_of_budget_slope.csv`), separating slow learning from a lower plateau.
- Q5: updates needed to reach validation accuracy 0.5, 0.8 and 0.95 as a function of g.

The collector warns if no g = 1 rows are joined from E0.9a (check `--e09a-root`).

## Post hoc analysis

`analyze_gain_sweep_posthoc.py` writes only `aggregate_posthoc/` (`POSTHOC_RESULT.md`): P1 best accuracy over gains per source, navigation minus matched; P2 plateau width (standard error from the larger of binomial and between-target error); P3 threshold-crossing gain interpolated in log2 g with left/right censoring; P4 censoring-aware rate surface (needs the per-cell update curves); P5 learned versus fixed gain on the same targets; P6 over-gain sensitivity, long versus short paths. The threshold shift at 0.90 accuracy is the observation used by E5b′.

## Usage

```bash
python tools/mechanism_v1/E5c_gain_sweep_v1/scripts/run_gain_sweep.py --preflight
python tools/mechanism_v1/E5c_gain_sweep_v1/scripts/run_gain_sweep.py --source-seed 6101 --target-seed 84001 --gain 32
python tools/mechanism_v1/E5c_gain_sweep_v1/scripts/collect_gain_sweep.py
python tools/mechanism_v1/E5c_gain_sweep_v1/scripts/analyze_gain_sweep_posthoc.py
```

Output: `rerun_best_validation/followups/mechanism_v1/E5c_gain_sweep/`. Figure: S7. Limitations: two targets; g* is resolved to a factor of 2 on the grid.
