# Formal topology, reduced dynamics and mechanism tests (Supplementary Fig. S2)

This folder holds the analysis chain behind Supplementary Fig. S2 (builder `FigureA3`) on the best-validation cohort (`navigation/best.pt`) and the unified aggregate of all 4 × 4 results. It reuses the completed torus topology (`topology/schoyen_topology_best_validation_cluster_v3`), the best-validation factorial checkpoints and the downstream results without retraining. The protocol is `PROTOCOL_FIGUREA3_SCHOYEN_FORMAL.md`.

## Analysis chain

```text
completed torus_topology
        |  selected Schøyen candidate, Euclidean persistent homology
        v
formal_topology_schoyen        H0_inf == 1, H1 #2/#3 >= 1.5, H2 #1/#2 >= 1.5
        |
        v
source_splits_schoyen          four fresh, disjoint navigation trajectory sets
        |
        v
source_reduced_schoyen         independent true-position Fourier phase chart on the selected ensemble
        |
        v
action_schoyen_t*, path_schoyen_t*
        |
        v
unified aggregate and Supplementary Fig. S2
```

The formal topology decision is the Euclidean persistent-homology result of the already selected Schøyen candidate. The Fourier phase chart is a separate coordinate construction for the action and path dynamics and is not used to decide whether topology passes. Candidate selection is the locked Schøyen dual-metric rule and is not revisited using downstream outcomes.

## Panels

- a, Euclidean formal topology: signed log2 margins of `H1_second_to_third` and `H2_first_to_second`; H0 connectedness is a discrete gate reported in the CSV and the denominator table.
- b, reduced model: held-out reduced-dynamics and fibre quality-gate margins, anchored on the independent Fourier phase chart.
- c, action causality: minimum confirmatory null/correct-action error ratio on the best-validation factorial checkpoints.
- d, path invariance: path diagnostics divided by their prespecified upper limits.

## Per-source computations (sources 6101–6110)

1. `formal_topology_schoyen`: converts the Euclidean persistent homology of the selected candidate into a hashed formal contract (no new TDA).
2. `source_splits_schoyen`: four fresh trajectory sets, phase calibration 64 × 500, fit 128 × 500, validation 64 × 500 and test 64 × 500, with seed offsets +50001, +51001, +52001 and +53001 (distinct from the Schøyen clustering, +30001, and TDA, +40001, seeds).
3. `source_reduced_schoyen`: fits the phase chart and reuses `PhaseDynamicsV3`, the Fourier torus decoder and the fibre model.
4. `action_schoyen_t*` and `path_schoyen_t*`: mechanisms on the best-validation factorial target checkpoints; `multinav.stage` first verifies that the target checkpoint and the reduced model refer to the same navigation `best.pt` (SHA-256).

The modules of this route are part of `code/gpm_cmap`: `analysis/source_reduced_schoyen.py`, `experiments/derive_schoyen_formal_topology.py`, `export_schoyen_source_splits.py`, `analyze_source_reduced_schoyen.py` and `configs/source_reduced_schoyen.yaml`; `analysis/source_reduced_v3.py` accepts an injected phase encoder and `analysis/upgrade_contracts.py` accepts the schema `source_reduced_schoyen_v1` besides `source_reduced_v3`.

## Usage

```bash
python topology/schoyen_formal_figureA3_patch_v1/run_figureA3_source.py preflight \
    --run-root <run root> --result-root <result root> --code-root code --seed 6101
# then the modes prepare, reduced and mechanism for every source
python topology/schoyen_formal_figureA3_patch_v1/unified_aggregate/aggregate_best_validation_full_v2.py \
    --run-root <run root> --result-root <result root>
```

The run root contains `plan.json`, `code/` and `sources/` (for example the repository root with P1); the result root holds the best-validation results (`rerun_best_validation` in the released runs). Every stage is backed by an attempt directory and a state JSON, and a completed stage is reused. Per-source outputs:

```text
<result root>/sources/nav_seed_<s>/artifacts/
  formal_topology_schoyen/      formal_topology_schoyen.json, formal_topology_components.csv
  source_splits_schoyen/        source_{phase,fit,validation,test}_arrays.npz, split manifest
  source_reduced_schoyen/       source_reduced_schoyen_v1.pkl, summary, test and per-trajectory metrics
  action_schoyen_t<target>/
  path_schoyen_t<target>/
```

The aggregate adds `source_gate_components.csv`, `source_gate_pass_counts.csv`, `schoyen_phase_chart_metrics.csv`, `mechanism_replication_by_source.csv` and `mechanism_replication_counts.csv` to the aggregate tables. `plot_best_validation_full_v2.py` draws figures directly from such an aggregate; the article figures are drawn by `figures/redraw_all_figures.py` from `data/aggregate_full`.

## Missingness

A source that fails the Euclidean formal-topology gate is not converted into a mechanism failure: it is marked `not_evaluable`, and its reduced-model, action and path stages are skipped. The denominator files keep planned and evaluable source counts. A source that passes formal topology but not the reduced-model exact or fibre gate still gets the action and path evaluation; the reduced-model gates are reported outcomes, not inclusion filters.

## Phase chart

On the independent phase-calibration split the chart

1. builds 64 × 64 true-position rate maps of the Schøyen-selected ensemble;
2. uses the saved three-family KMeans labels of the torus analysis when available;
3. extracts the dominant Fourier wavevectors per family, with population peaks as fallback;
4. selects two spectrally strong, linearly independent physical wavevectors of comparable scale, in an explicit physical (kx, ky) convention;
5. converts true positions into two periodic target phases;
6. fits a frozen recurrent-activity → two-circle chart;
7. reports calibration, source-validation and held-out source-test circular RMSE and phase coverage.

These RMSE values are diagnostics; the held-out reduced-model gates remain the confirmatory criteria.
