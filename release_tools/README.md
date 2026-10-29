# Release tools

Tools used to assemble and check this release. Only `verify_release.py` and `extract_packages.py` are needed by users.

| Script | Purpose |
|---|---|
| `local/verify_release.py` | Checks every public file of the repository against `data/RELEASE_FILES.json` (size, SHA-256, Python syntax); with `--packages <folder>` also checks the archives P1–P3 against `data/PACKAGE_SHA256SUMS.txt`. `--write-manifest` rewrites the manifest from the current files after intentional edits. |
| `local/extract_packages.py` | Checks the archives and extracts them into a short repository path (Windows path lengths). |
| `local/package_code.py` | Builds the code ZIP from the verified manifest. |
| `local/assemble_local.py` | Copies the figure source data read by the builders from a working figure folder into `data/aggregate_full/` and reports files too large for git. |
| `local/compare_with_docx.py` | Places redrawn figures next to the images embedded in the manuscript files for visual comparison. |
| `cluster/build_release.py`, `cluster/release_manifest.json` | Select the files of the data packages P1–P3 from the project tree (dry run → `--execute` → `--package`). The project is only read; pretrained checkpoints are saved weights-only (optimizer and scheduler states dropped) and checked to keep identical cores. |
