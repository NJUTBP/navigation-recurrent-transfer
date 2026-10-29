# Figures

`redraw_all_figures.py` draws Figures 2–7 and Supplementary Figures S1–S12 from `data/aggregate_full`; `export_source_data_final.py` copies the source data into folders named by article figure number. Figure 1 and the graphical abstract are static images in `figure1/`. Commands are in the top-level README.

| File or folder | Role |
|---|---|
| `figure_numbering.py` | Table relating article figure numbers, builder modules, builder output names and source-data folders; read by both entry points |
| `paper_figures/` | Builders of Figs. 2–5, S1–S4 and S10 (`build(ctx)`), shared helpers (`common.py`), camera tool for Fig. S3 (`figureA4_view_picker.py`) |
| `hard8_phaseF/` | Builders of Figs. 6–7, S5–S9, S11 and S12; input and output locations (`figure_paths.py`); shared mechanism-figure helpers (`mechanism_plot_common.py`) |
| `paper_figures/nature_style.py`, `hard8_phaseF/nature_style.py` | Publication style; the two files are identical so that each package can be imported on its own (checked by `tests/test_figure_numbering.py`) |
| `source_data_builders/` | Scripts that produced the figure source data from the experiment outputs (not needed for redrawing) |
| `figure1/` | Author-drawn Figure 1 and graphical abstract |

## Builder names and article numbers

Builder modules, output names and source-data folders keep the identifiers under which the figures were developed. Where they differ from the article: the builders `paper_figures/make_figureA1…A4,A6` produce Supplementary Figs. S1, S4, S2, S3 and S10; `hard8_phaseF/make_figureS9.py` draws Supplementary Fig. S11 and `make_figureS10.py` draws Supplementary Fig. S9; `make_figureS7.py` and `make_figureS8.py` read the folders `SupplementaryFigureS8` and `SupplementaryFigureS9`. In Figure 7 the tables `panelC_retention_dose.csv` and `panelD_horizon_profile.csv` are drawn as panels d and c (the export renames them to the final letters). The complete table is in `figure_numbering.py` and `docs/FIGURE_DATA_MAP.md`.

## Outputs

By default the redraw writes to `outputs/figures/`: `<Figure>.{png,svg,pdf,tiff}` (and `.eps` when `pdftops` is available), the builder outputs and per-figure manifests in `_builder_outputs/`, and `figure_manifest.json`. Output folders inside the source data are refused. The builder of Supplementary Fig. S10 also writes the association statistics it recomputes to `_builder_outputs/FigureA6_association_statistics.csv`; the stored table `FigureA6/association_statistics.csv` is not modified.

`NRT_FIGURE_AGGREGATE_DIR` and `NRT_FIGURE_OUTPUT_DIR` pass the input and output folders to the 8 × 8 and mechanism builders; the redraw sets them, and a builder run on its own uses `data/aggregate_full` and `outputs/figures/_builder_outputs`. For Fig. S3, `FIGA4_C_VIEW_FILE` selects a camera file saved by `figureA4_view_picker.py` (default `outputs/figures/figureA4C_view.json`); `FIGA4_UMAP_RESTARTS`, `FIGA4_UMAP_BASE_SEED` and `FIGA4_FORCE_RECOMPUTE` control an optional recomputation of the UMAP embedding, which needs `umap-learn` and the torus-analysis arrays of the topology analysis.

## Source-data builders

- `source_data_builders/prepare_source_data.py --project <run root> --plan <plan.json> --aggregate <aggregate>` writes the 4 × 4, pretraining and topology folders into `<aggregate>/figure_source_data` from the aggregate tables and per-source artifacts.
- `source_data_builders/figure2/`: Figure 2 tables of the representative source 6101. `regen_fig2c_100steps.py` regenerates the 100-step trajectory from the P1 checkpoint; `export_fig2c_20step_candidates.py` and `regen_fig2d_top10_50bins.py` read the navigation evaluation arrays (`checkpoint_analysis/best_validation/nav_eval_arrays.npz` of the navigation run), which are not in the data packages. These three scripts write to `outputs/source_data_builders/Figure2/`. `regen_fig2d_rate_maps_50bins.py` takes `--project-dir` and `--aggregate-dir` and writes `*_50x50.csv` next to the existing tables unless `--overwrite` is given.
- The Figure 6, S5 and S6 tables are copied by `tools/FigureA5_PhaseF_workflow_v1/`. The tables of Figure 7 and Supplementary Figs. S7–S9 and S11 were frozen from the outputs of the mechanistic analyses; `figure_source_data/MECHANISM_FIGURE_SOURCE_FREEZE.json` and the `SOURCE_MANIFEST.json` of each folder record the input tables and their checksums. The S12 tables are written by `tools/mechanism_v1/E6_retention_intervention_v1/scripts/export_figureS12_source_data.py`.
