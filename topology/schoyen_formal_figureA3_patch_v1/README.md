# FigureA3 rebuild: Schøyen Euclidean formal topology

This package rebuilds FigureA3 on the completed **best-validation (`navigation/best.pt`)** cohort without rerunning navigation, the main cognitive experiments, factorial training, or `torus_topology`.

## Scientific replacement

The legacy route

```text
topology_export -> legacy formal_topology -> cohomological phase coordinates -> Source-v3
```

is not used.

The rebuilt route is

```text
completed torus_topology
        |
        | selected candidate, Euclidean persistent homology
        v
formal_topology_schoyen
        |
        | H0_inf == 1
        | H1 #2/#3 >= 1.5
        | H2 #1/#2 >= 1.5
        v
source_splits_schoyen
        |
        | four fresh, disjoint navigation trajectory sets
        v
source_reduced_schoyen
        |
        | independent TRUE-position Fourier phase chart
        | on the Schøyen-selected recurrent ensemble
        v
  action_schoyen_t* + path_schoyen_t*
        |
        v
aggregate_full + rebuilt FigureA3
```

Important distinction: the **formal topology decision** is the Euclidean PH result of the already-selected Schøyen candidate. The Fourier phase chart is a separate coordinate construction needed for action/path dynamics; it is not used to decide whether topology passes.

Candidate selection itself remains the already-completed, locked Schøyen dual-metric selection rule. This patch does not retrospectively reselect an ensemble from downstream mechanism outcomes.

## FigureA3 panels

- **A — Schøyen Euclidean formal topology**: signed log2 margins for `H1_second_to_third` and `H2_first_to_second`; H0 connectedness is retained as a discrete gate in the CSV/denominator report.
- **B — Schøyen-anchored reduced model**: the existing held-out reduced-dynamics/fibre quality-gate margins, with the obsolete cohomological phase anchor replaced by an independent Fourier phase chart on the Schøyen-selected ensemble.
- **C — Action causality**: minimum confirmatory null/correct-action error ratio, using the completed best-validation factorial checkpoints.
- **D — Path invariance**: path diagnostics divided by their prespecified upper limits.

## What is reused

The following completed results are read only and are not retrained:

- `sources/nav_seed_<seed>/artifacts/navigation/best.pt`
- `rerun_best_validation/sources/nav_seed_<seed>/artifacts/torus_topology/`
- `rerun_best_validation/sources/nav_seed_<seed>/artifacts/factorial_t*/`
- all completed best-validation main/hard/adapter/symbolic results used by the final unified aggregate.

## What is newly computed

For each source 6101–6110:

1. `formal_topology_schoyen`: converts the selected Schøyen candidate's Euclidean PH into an explicit hashed formal contract. No new TDA is run.
2. `source_splits_schoyen`: generates four fresh source trajectory sets:
   - phase calibration: 64 x 500
   - fit: 128 x 500
   - validation: 64 x 500
   - test: 64 x 500
3. `source_reduced_schoyen`: fits the phase chart and reuses the existing PhaseDynamicsV3 / Fourier decoder / fibre model machinery.
4. `action_schoyen_t*` and `path_schoyen_t*`: evaluate mechanisms on the already-trained best-validation factorial target checkpoints.
5. a new `aggregate_full` attempt is produced and the public `aggregate_full` symlink is atomically updated.

The fresh source split offsets are **+50001/+51001/+52001/+53001**, deliberately distinct from the Schøyen clustering (+30001) and TDA (+40001) trajectory seeds.

## Installation

From the existing frozen run root:

```bash
cd <PROJECT_ROOT>

# unzip/copy this directory here first
PYTHON_BIN=<PYTHON> \
  bash schoyen_formal_figureA3_patch_v1/install_patch.sh "$PWD"
```

The installer backs up the two modified shared modules before replacing them:

```text
code_backup_pre_figureA3_schoyen_<timestamp>/
```

It then installs:

```text
code/gpm_cmap/analysis/source_reduced_v3.py              # backward-compatible phase_encoder injection
code/gpm_cmap/analysis/upgrade_contracts.py              # accepts source_reduced_schoyen_v1
code/gpm_cmap/analysis/source_reduced_schoyen.py          # new Schøyen phase chart
code/gpm_cmap/experiments/derive_schoyen_formal_topology.py
code/gpm_cmap/experiments/export_schoyen_source_splits.py
code/gpm_cmap/experiments/analyze_source_reduced_schoyen.py
code/gpm_cmap/configs/source_reduced_schoyen.yaml
```

It does **not** modify the frozen `plan.json`.

## Submit the complete FigureA3 chain

```bash
cd <PROJECT_ROOT>
bash schoyen_formal_figureA3_patch_v1/submit_figureA3_chain.sh
```

The submission order is:

```text
fa3_preflight
   -> fa3_prepare[1-10]%2
   -> fa3_reduced[1-10]%2
   -> fa3_mechanism[1-10]%2
   -> fa3_aggregate
```

As with the corrected topology V3 launcher, jobs do not depend on LSF inheriting custom shell variables. Absolute paths are written to:

```text
<RUN_ROOT>/.schoyen_figureA3_runtime.env
```

and each batch job explicitly sources that file.

## Optional path overrides

```bash
export RUN_ROOT=/path/to/multinav_paper_v1
export RESULT_ROOT="$RUN_ROOT/rerun_best_validation"
export CODE_ROOT="$RUN_ROOT/code"
export PYTHON_BIN=/path/to/python
bash schoyen_formal_figureA3_patch_v1/submit_figureA3_chain.sh
```

## Per-source outputs

For source 6101, for example:

```text
rerun_best_validation/sources/nav_seed_6101/artifacts/
├── formal_topology_schoyen/
│   ├── formal_topology_schoyen.json
│   └── formal_topology_components.csv
├── source_splits_schoyen/
│   ├── source_phase_arrays.npz
│   ├── source_fit_arrays.npz
│   ├── source_validation_arrays.npz
│   ├── source_test_arrays.npz
│   └── source_reduced_schoyen_split_manifest.json
├── source_reduced_schoyen/
│   ├── source_reduced_schoyen_v1.pkl
│   ├── source_reduced_schoyen_summary.json
│   ├── source_reduced_schoyen_test_metrics.csv
│   └── source_reduced_schoyen_per_trajectory_metrics.csv
├── action_schoyen_t1060/
└── path_schoyen_t1060/
```

Each artifact is backed by an attempt directory and a state JSON; reruns reuse a completed committed stage.

## Final aggregate

After `fa3_aggregate` succeeds:

```text
rerun_best_validation/aggregate_full/
```

points to the newest successful `aggregate_full_attempts/try_###`.

Important new outputs include:

```text
source_gate_components.csv
source_gate_pass_counts.csv
schoyen_phase_chart_metrics.csv
mechanism_replication_by_source.csv
mechanism_replication_counts.csv
figures/FigureA3.pdf
figures/FigureA3.png
figures/FigureA3/a_formal_topology.*
figures/FigureA3/b_reduced_model.*
figures/FigureA3/b2_gate_denominators.*
figures/FigureA3/c_action_null.*
figures/FigureA3/d_path_diagnostics.*
```

The other manuscript figures are regenerated from the same hybrid best-validation data sources, so the public `aggregate_full` remains the canonical manuscript result directory.

## Scientific missingness behavior

A source that fails the Euclidean formal-topology gate is **not** converted into a mechanism failure. It is marked `not_evaluable`, and downstream source-reduced/action/path stages are skipped for that source. The denominator files preserve planned versus evaluable source counts.

A source that passes formal topology but does not pass the reduced-model exact/fibre gate still gets action/path evaluation. Those reduced-model gates are reported outcomes, not post-hoc inclusion filters.

## Coordinate-chart diagnostics

The new coordinate chart does not use the old cohomological landmarks. On the independent phase-calibration split it:

1. builds 64x64 TRUE-position rate maps for the Schøyen-selected ensemble;
2. uses the saved Schøyen 3-family KMeans labels when available;
3. extracts dominant Fourier wavevectors per family, with population peaks as fallback;
4. selects two spectrally strong, linearly independent, comparable-scale physical wavevectors;
5. converts TRUE positions into two periodic target phases;
6. fits a frozen recurrent-activity -> two-circle chart;
7. reports calibration, source-validation and held-out source-test circular RMSE and phase coverage.

These RMSE values are diagnostics. This patch does not invent a new RMSE gate; the existing held-out reduced-model gates remain the confirmatory reduced-model criteria.

## Validation performed before packaging

- Python `py_compile` and import checks for all new/modified Python modules.
- `bash -n` checks for installer, submit script and all LSF files.
- Synthetic Euclidean-formal-contract test.
- Synthetic periodic-population test for the Schøyen/Fourier phase chart, including KMeans-family wavevector selection.
- Synthetic end-to-end injected-phase-encoder test through `SourceReducedModelV3.fit` and held-out quality-gate evaluation.
- Synthetic CLI run of `analyze_source_reduced_schoyen` producing the new pickle/summary/CSV artifacts.
- Installation smoke test against the current merged code tree.

These tests verify interfaces and control flow; they do not substitute for the 10-source scientific results on the cluster.
