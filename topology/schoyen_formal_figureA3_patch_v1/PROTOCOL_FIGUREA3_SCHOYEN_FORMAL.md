# Protocol: FigureA3 with Schøyen Euclidean formal topology

## 1. Formal topology definition

The candidate ensemble is the candidate already selected by the completed Schøyen automatic topology analysis. No candidate is reselected using FigureA3 outcomes.

For that selected candidate, the **Euclidean** persistent-homology result is the formal topology result for FigureA3.

A source passes the formal `(1,2,1)` gate iff:

- `H0_infinite_components == 1`
- `H1_second_to_third >= 1.5`
- `H2_first_to_second >= 1.5`

The H1 and H2 signed margins plotted in FigureA3a are:

```text
log2(H1_second_to_third / 1.5)
log2(H2_first_to_second / 1.5)
```

H0 is a discrete equality gate and is reported in the formal contract and gate-component table rather than converted into an artificial numerical margin.

## 2. Data independence

The completed Schøyen analysis already used fresh source trajectories with seed offsets +30001 (clustering) and +40001 (TDA).

The FigureA3 extension uses four new source datasets:

- phase calibration: +50001
- reduced-model fit: +51001
- reduced-model validation: +52001
- reduced-model held-out test: +53001

Trajectory fingerprints are audited pairwise. Source test is never used to select the phase chart, phase dynamics, decoder order/regularization, fibre dimension, or fibre dynamics.

## 3. Schøyen-anchored phase chart

Formal topology proves the operational Euclidean `(1,2,1)` signature but does not itself supply the old cohomological coordinates. Therefore FigureA3 uses a separate coordinate construction.

On the independent phase-calibration trajectories:

1. restrict recurrent activity to the Schøyen-selected unit indices;
2. form 64x64 TRUE-position rate maps;
3. use the saved 3-family KMeans refinement from `torus_topology` when available;
4. estimate dominant physical Fourier wavevectors for each family; population-level spectral peaks are fallback candidates;
5. select two non-collinear, spectrally strong, similar-scale wavevectors;
6. define two periodic TRUE-position phases from those vectors;
7. fit the existing circular chart machinery from recurrent population activity to those two phases.

The coordinate chart is then frozen.

Phase-chart circular RMSE is reported on source validation and held-out source test but is not an additional confirmatory inclusion gate.

## 4. Reduced model

The patch reuses the existing mature SourceReducedModelV3 mechanics after replacing only the phase anchor:

- `PhaseDynamicsV3`
- GL(2,Z) action-basis alignment
- Fourier torus decoder
- fibre selection and fibre dynamics
- matched linear baseline
- existing held-out source quality gates

The old prespecified quality thresholds are kept unchanged.

This yields the new artifact schema:

```text
source_reduced_schoyen_v1
```

The loader accepts both the legacy schema and the new schema so existing action/path experiment code remains usable.

## 5. Mechanism tests

Action causality and path invariance reuse the already-completed **best-validation factorial checkpoints**. Before running, `multinav.stage` verifies that both target checkpoint and new source-reduced artifact point to the same navigation `best.pt` SHA256.

No factorial model is retrained.

## 6. FigureA3 panels

### A. Euclidean formal topology

Per-source H1 and H2 signed margins around zero. H0 and evaluability counts are reported alongside the plot.

### B. Schøyen-anchored reduced model

Per-source signed log2 margins for the existing exact-2D and toroidal-base-plus-fibre held-out criteria. Phase-chart validation/test RMSE is a separate diagnostic table and denominator note.

### C. Action causality

`minimum_confirmatory_null_ratio`, reference line 1.10.

### D. Path invariance

Each path diagnostic is divided by its prespecified upper limit. Reference line 1.0; values <= 1 pass.

## 7. Statistical unit and missingness

The independent statistical unit remains the navigation source.

- planned source denominator: 10;
- formal-topology nonpass: explicit scientific `not_evaluable` for FigureA3b-d;
- technical failure: never treated as scientific nonpass;
- reduced-model gate failure: reported as a result and does not suppress action/path computation for a formally evaluable source.

The aggregate writes planned, evaluable and confirmed-pass counts separately.
