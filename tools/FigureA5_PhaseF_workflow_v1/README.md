# Copying the Phase F results into the figure source data

`freeze_phaseF_figureA5_into_aggregate.py` copies the Phase F aggregate tables into an aggregate directory used for plotting:

- `<aggregate>/hard8x8_phaseF_followup/`: the complete Phase F aggregate, read by Figure 6 and Supplementary Figures S5 and S6;
- `<aggregate>/figure_source_data/FigureA5/`: the 13 tables drawn in these figures (`FigureA5` is the builder identifier of the Figure 6 data) and `PHASEF_FIGURE_SOURCE_MANIFEST.json` with their checksums.

Only files are copied; nothing is retrained, aggregated or tested. The Phase F aggregate must report `ready` in `READINESS.json`.

```bash
python tools/FigureA5_PhaseF_workflow_v1/freeze_phaseF_figureA5_into_aggregate.py --aggregate outputs/aggregate_full
```

The input defaults to the Phase F aggregate below the project root (`rerun_best_validation/hard8x8_diagnostics/audit_v1/phaseF_fullscale_v1/aggregate`; `--phasef-aggregate` to change). Targets inside `data/` are refused: the released figure data already contain the tables used for the article. To redraw from rerun results, copy `data/aggregate_full` to a new folder, use it as `--aggregate`, and pass it to `figures/redraw_all_figures.py --aggregate`.
