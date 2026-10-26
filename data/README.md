# Data in this repository

- `aggregate_full/figure_source_data/` — source data of every figure, in the builder folders read by `figures/redraw_all_figures.py` (`figures/export_source_data_final.py` writes a copy named by final figure number).
- `aggregate_full/hard8x8_phaseF_followup/` — the ten 8 × 8 tables read by Figs. 6, S5 and S6.
- `source_level_results/mechanism_v1/E6_task_state_retention.csv` — one-step retention on task states (input of Supplementary Fig. S12c).

Large result tables, trained weights and the complete source-level results are in the data packages (`docs/DATA_PACKAGES.md`).

- `source_level_results/checkpoint_provenance/` — indices for the 20 released pretrained checkpoints and 480 target checkpoints. Checkpoint-selection conventions are explained in `docs/DATA_PACKAGES.md`.
- `PACKAGE_INVENTORY.json`, `PACKAGE_SHA256SUMS.txt` — measured data-package contents and checksums.
- `RELEASE_FILES.json` — checksums of the portable code/figure-data release; verify with `release_tools/local/verify_release.py`.

Models, data and supplied figure assets use CC BY 4.0 (`DATA_LICENSE.md`); software uses MIT (`LICENSE`).
