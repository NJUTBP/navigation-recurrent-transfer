# Figure, table and statistics map

Scope: the final Main Text and Supplementary Information. Builder names are internal; the final number is what the paper shows.
Paths of plotting data are relative to `data/aggregate_full/`; experiment outputs are relative to the project root.
"P1/P2/P3" are the data packages described in `docs/DATA_PACKAGES.md`.

## Main figures

| Fig. | Question | Plotting (in `figures/`) | Experiment code | Data read | P1 | P2 | From seeds | Minimal files | Missing / notes |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Experimental logic | none (author-drawn) | – | – | – | – | – | raster image in `figures/figure1/` | editable authoring source not available |
| 2 | Navigation pretraining: decoding error, trajectories, rate maps, grid scores | `paper_figures/make_figure2_navigation.py` | `code/gpm_cmap/experiments/exp1_nav_pretrain.py` (config `nav_pretrain_full.yaml`, seeds 6101–6110, via `multinav.stage navigation`); data export `figures/source_data_builders/figure2/*.py` | `figure_source_data/Figure2/` (navigation_history, trajectory candidates, rate maps rank 2–5, grid scores, metadata) | L2 | – | trajectories yes | Figure2 folder; P1 navigation checkpoints | – |
| 3 | Frozen-core transfer, 4 × 4 aligned task | `paper_figures/make_figure3_frozen_transfer.py` | `exp5_abstract_pathint.py` (stages exp5_nav, exp5_ctl) via `lsf_best_validation/run_best_validation_source.py`; aggregation `topology/schoyen_formal_figureA3_patch_v1/unified_aggregate/aggregate_best_validation_full_v2.py`, `code/multinav/stats.py` | `figure_source_data/Figure3/` (frozen_transfer, training_curves, contrasts, paired_differences) | L3 | – | trials yes (target seeds in `plan.json`) | Figure3 folder; P3 rows | 4 × 4 target weights not included; retraining code provided |
| 4 | Relation-input transformations: test-time transforms, learned adapter | `paper_figures/make_figure4_interface.py` | exp5 stage `test_transforms` (`code/multinav/stage.py`), `exp7_adapter_rescue.py` | `figure_source_data/Figure4/` (test_transforms, adapter, adapter_contrasts, adapter_paired_differences) | L3 | – | yes | Figure4 folder; P3 rows | test transforms need the aligned target models: Level 3 |
| 5 | Arbitrary relation symbols, low budgets, recurrence ablation, non-recurrent baselines | `paper_figures/make_figure5_symbols.py` | `exp8_symbolic_adapter.py` (config `symbolic_adapter_ctl01.yaml`); `extensions/symbolic_baseline7000_extension_v1/` (7,000-trial baseline points) | `figure_source_data/Figure5/` (symbolic, contrasts, paired_differences, recurrence) | L3 | – | yes | Figure5 folder; P3 rows | – |
| 6 | Extended 8 × 8 task: seen/held-out × short/long, per length, core types | `hard8_phaseF/make_figure6.py` | `tools/hard8x8_phaseF_fullscale_v1/scripts/run_group.py`, `collect_fullscale.py`; freeze `tools/FigureA5_PhaseF_workflow_v1/freeze_phaseF_figureA5_into_aggregate.py` | `figure_source_data/FigureA5/`, `hard8x8_phaseF_followup/` (source_level_metrics, all_per_length, primary/secondary contrasts, baselines, diagnostics) | L2 | 150 scalar-gain models | splits 84001–84005 | FigureA5 folder, 10 Phase-F tables; P2 phasef | full-finetune conditions store their core: Level 1/3 only |
| 7 | Perturbation and rescaling: δ_t (a), dose response (b), path-length profile (c), retention κ(ρ) (d), rescaling effects (e) | `hard8_phaseF/make_figure7.py` | `tools/mechanism_v1/E0_67_forward_diagnostics_v1` (a), `E6_rho_scan_v1` (b, c), `E6_autonomous_retention_v1` (d), `E6_retention_intervention_v1` (e), `verdicts_v1` (tests) | `figure_source_data/Figure7/` (panelA–E tables, figure7_statistics.json; builder letters C/D are final D/C, see `export_source_data_final.py`) | L2 | 210 ρ-series + 60 rescaling models | yes | Figure7 folder; P2 e6, e6i | – |

## Supplementary figures

| Fig. | Builder | Question | Plotting | Experiment / analysis code | Data read | P1 | P2 | Notes |
|---|---|---|---|---|---|---|---|---|
| S1 | FigureA1 | Delayed-reconstruction pretraining | `paper_figures/make_figureA1_control.py` | `pretrain_sequence_control.py` (config `sequence_control_pretrain.yaml`, seeds 7101–7110) | `FigureA1/` (control_source_means, source_qualification) | L2 (decoders kept) | – | – |
| S2 | FigureA3 | Topology criteria, reduced dynamics, path invariance | `paper_figures/make_figureA3_mechanism.py` | `topology/schoyen_topology_best_validation_cluster_v3/torus_chain.py` (→ `analyze_schoyen_torus`, `aggregate_schoyen_torus`); `topology/schoyen_formal_figureA3_patch_v1/run_figureA3_source.py` (→ `export_schoyen_source_splits`, `analyze_source_reduced_schoyen`, `derive_schoyen_formal_topology`); path invariance on factorial models (`exp5_recurrent_factorial.py`, `analyze_path_invariance.py`) | `FigureA3/` (formal_topology, source_reduced_gates, source_gate_pass_counts, mechanism_replication_counts, path, source_qualification) | L2 (CPU-heavy) | – | factorial target models not released: Level 3 |
| S3 | FigureA4 | Torus visualization of a representative source | `paper_figures/make_figureA4_topology.py`, view tool `paper_figures/figureA4_view_picker.py` | `topology/figureA4_visualization/compute_cohomology_torus.py`, `run_figureA4_umap.py` | `FigureA4/` (torus_analysis_arrays.npz, representative_*.csv/json, torus_auto_summary.csv, figureA4C_view.json) | – | – | visualization only; stochastic UMAP embeddings are released as data (Level 1) |
| S4 | FigureA2 | Relation and endpoint mapping controls | `paper_figures/make_figureA2_mapping_controls.py` | exp5 stage `mapping` (`code/multinav/stage.py`) | `FigureA2/` (mapping, contrasts, paired_differences) | L3 | – | – |
| S5 | S5 | 8 × 8 transfer controls; held-out per length (e) | `hard8_phaseF/make_figureS5.py` | Phase F | `FigureA5/`, `hard8x8_phaseF_followup/all_per_length.csv` | L2 | phasef | – |
| S6 | S6 | Optimization, shortcut baseline, split diagnostics | `hard8_phaseF/make_figureS6.py` | Phase F | `FigureA5/`, `hard8x8_phaseF_followup/all_dataset_audit.csv`, `all_update_curves.csv` | – | – | Level 1/3 |
| S7 | S8 folder | Extended budget, gain operating curve, additive horizon | `hard8_phaseF/make_figureS7.py` | `E0_9a_extended_budget_v1`, `E5c_gain_sweep_v1`, `E0_1_nonrecurrent_baselines_v1` | `SupplementaryFigureS8/` | L2 | e09a (60) | – |
| S8 | S9 folder | Multi-step state maintenance, ablation and order shuffling, error growth, δ prediction | `hard8_phaseF/make_figureS8.py` | `E0_67_forward_diagnostics_v1` (on Phase-F scalar models), `E5b_delta_prediction_v1` | `SupplementaryFigureS9/` | L2 | phasef | – |
| S9 | S11 folder (`make_figureS10.py`) | Refitting of navigation-specific mappings | `hard8_phaseF/make_figureS10.py` | `E6_navigation_separability_v1` | `SupplementaryFigureS11/` | L3 | – | 210 refits |
| S10 | FigureA6 | Grid-like unit fraction versus transfer | `paper_figures/make_figureA6_associations.py` | statistics in `figures/source_data_builders/prepare_source_data.py` from source-level tables | `FigureA6/` (grid_transfer_associations, association_statistics) | – | – | Level 1; recomputable from P3 tables |
| S11 | S10 folder (`make_figureS9.py`) | Prespecified source-measure analyses | `hard8_phaseF/make_figureS9.py` | `E234_source_measures_v1`, `verdicts_v1` | `SupplementaryFigureS10/` | L2 | – | source side only |
| S12 | S12 | Rescaling interventions by path length and source | `hard8_phaseF/make_figureS12.py` | data: `tools/mechanism_v1/E6_retention_intervention_v1/scripts/export_figureS12_source_data.py`; task-state retention `tools/reevaluation/task_state_retention.py` | `SupplementaryFigureS12/` (panelA–C, s12_statistics.json; 22 KB) | L2 | e6, e6i | panel c uses the released task-state retention table |

## Supplementary tables

| Table | Content | Source |
|---|---|---|
| S1 | Navigation checkpoints: selected epoch, position error, grid-like fraction | P1 `history.json`/`summary.json`; `Figure2/`, `FigureA6/` |
| S2 | Delayed-reconstruction checkpoints: selected epoch, delay-10 NMSE | P1 `training_history.json`, `training_gate.json`; `FigureA1/` |
| S3 | Topology outcomes per source | `FigureA3/formal_topology.csv` |
| S4 | 8 × 8 primary generalization contrasts | `hard8x8_phaseF_followup/primary_generalization_contrasts.csv` |
| S5 | 8 × 8 secondary seen-short contrasts | `hard8x8_phaseF_followup/secondary_id_contrasts.csv` |
| S6 | Comparison families | definitions; implemented in `code/multinav/stats.py`, `collect_fullscale.py`, `verdicts_v1/scripts/compute_verdicts.py` |

## Statistics in the text

All tests use the ten source networks as the independent unit: exact sign-flip tests over 2^10 sign patterns, source-bootstrap confidence intervals, Holm correction within prespecified families.

| Results | Computed by | Stored in |
|---|---|---|
| 4 × 4 contrasts (Figs. 3–5, S4) | `aggregate_best_validation_full_v2.py`, `code/multinav/stats.py` | `Figure3/4/5/`, `FigureA2/` `contrasts.csv`, `paired_differences.csv` |
| 8 × 8 families (Fig. 6, Tables S4–S5) | `collect_fullscale.py` | `hard8x8_phaseF_followup/*contrasts.csv` |
| Mechanism tests (Fig. 7, S7–S9, S11) | collectors in each `tools/mechanism_v1/*/scripts/collect_*.py`, `verdicts_v1/scripts/compute_verdicts.py` | `*_statistics.json` in the figure folders; P3 `aggregate/` |
| Rescaling ranges and Δα (S12) | `export_figureS12_source_data.py` | `SupplementaryFigureS12/s12_statistics.json` |
| Associations (S10) | `prepare_source_data.py` | `FigureA6/association_statistics.csv` |
