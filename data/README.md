# Data in this repository

- `aggregate_full/figure_source_data/` — source data of every computational figure, in the folders read by the builders of `figures/redraw_all_figures.py`. Folder names are builder identifiers; `figures/figure_numbering.py` and `docs/FIGURE_DATA_MAP.md` give the article figure of each folder, and `figures/export_source_data_final.py` writes a copy named by article figure number.
- `aggregate_full/hard8x8_phaseF_followup/` — the ten 8 × 8 (Phase F) tables read by Figs. 6, S5 and S6.
- `aggregate_full/source_qualification.csv` — identity and qualification records of the ten navigation sources (read to define the source cohort).
- `source_level_results/mechanism_v1/E6_task_state_retention.csv` — one-step retention on task states (input of Supplementary Fig. S12c).
- `source_level_results/checkpoint_provenance/` — indices of the 20 released pretrained checkpoints and the 480 target checkpoints; the checkpoint-selection convention is explained in `docs/DATA_PACKAGES.md`.
- `PACKAGE_INVENTORY.json`, `PACKAGE_SHA256SUMS.txt` — contents and checksums of the data packages P1–P3.

Large result tables, trained weights and the complete source-level results are in the data packages (`docs/DATA_PACKAGES.md`). The redraw and export tools do not write into this folder.

Models, data and supplied figure assets use CC BY 4.0 (`DATA_LICENSE.md`); software uses MIT (`LICENSE`).
