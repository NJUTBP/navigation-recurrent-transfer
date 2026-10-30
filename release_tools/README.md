# Release tools

Tools used to assemble and package this release. Use `extract_packages.py` to unpack the downloaded data packages.

| Script | Purpose |
|---|---|
| `local/extract_packages.py` | Checks the archives and extracts them into a short repository path (Windows path lengths). |
| `local/package_code.py` | Builds the code ZIP from the working files tracked by Git. |
| `local/assemble_local.py` | Copies the figure source data read by the builders from a working figure folder into `data/aggregate_full/` and reports files too large for git. |
| `local/compare_with_docx.py` | Places redrawn figures next to the images embedded in the manuscript files for visual comparison. |
| `cluster/build_release.py`, `cluster/release_manifest.json` | Select the files of the data packages P1–P3 from the project tree (dry run → `--execute` → `--package`). The project is only read; pretrained checkpoints are saved weights-only (optimizer and scheduler states dropped) and checked to keep identical cores. |
