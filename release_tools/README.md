# Release tools

- `cluster/build_release.py` with `cluster/release_manifest.json`: selects the files of the data packages P1–P3 from the project tree (dry run → `--execute` → `--package`). It only reads the project and writes below `--out`; pretrained checkpoints are saved weights-only (optimizer and scheduler states dropped) and checked to keep identical cores.
- `local/assemble_local.py`: copies the figure source data read by `figures/redraw_all_figures.py` from a working figure folder into `data/aggregate_full/` and reports files that are too large for git.
- `local/compare_with_docx.py`: places redrawn figures next to the images embedded in the manuscript files.
