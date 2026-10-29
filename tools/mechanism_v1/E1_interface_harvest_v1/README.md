# E1 — magnitude and shape of the learned relation interfaces

Question: does the relation interface learned on the 4 × 4 symbolic-adapter task differ between core types in magnitude or in shape? E1 reads the completed symbolic-adapter results only (JSON files; no GPU, no training).

## Harvest

`harvest_interface_rows.py` collects one row per (source, target, arm, budget) into `interface_rows.csv` (10 × 3 × 38 = 1,140 rows), with `harvest_manifest.json` (selected input files and their SHA-256, target list per source, completeness check) and `candidate_scan.json` (all candidate paths and reasons for rejection).

- Target seeds are discovered by glob, not derived from a formula; `symbolic_baseline7000_t*` is excluded by the regular expression `^symbolic_t(\d+)$`.
- Both result layouts are scanned, `artifacts/symbolic_t<T>/` and `attempts/symbolic_t<T>/try_<NNN>/`; duplicates are removed by real path, and `artifacts` is preferred, then the highest attempt number.
- `column_norms` exist only in the JSON rows, so the JSON files are read. The codebook magnitude is read from `protocol_manifest.json` (`seeds[].action_codebook`): M_rel_codebook = mean_k ‖W_rel[:,k]‖ / ‖codebook column k‖, in the same units as the 8 × 8 gain, which multiplies the same codebook.
- Consistency check: for an interface close to a rotation plus scaling of the codebook, M_rel_codebook ≈ 1/procrustes_scale; the relative difference is stored as `scale_consistency`, and its median for budgets ≥ 1,000 is reported in the manifest (values of a few percent are expected; larger values indicate misread fields).
- `--require-complete` returns a nonzero status unless 10 sources × 3 targets are present.

## Prespecified analysis

`analyze_interface_magnitude.py`; its module docstring states the tests:

- primary: M(navigation) < M(delayed reconstruction) at the largest common budget, where both are at the accuracy ceiling (budget- and performance-matched);
- secondary family (Holm, m = 4): navigation versus random; the performance-matched magnitude M* (at the first budget reaching accuracy 0.9); navigation magnitude levelled off; delayed-reconstruction magnitude not levelled off;
- prespecified null family (Holm, m = 3): antisymmetry index, Procrustes residual and deviation of the axis angle from 90°, navigation versus delayed reconstruction, expected not to favour navigation;
- descriptive: shape–accuracy dissociation, chance-normalized recurrence drops, shape reference of the 2-D accumulator without a core.

Outputs: `source_level_interface.csv`, `interface_contrasts.csv`, `recurrence_drop_normalized.csv`, `shape_accuracy_dissociation.csv`, `INTERFACE_RESULT.md/json`.

`analyze_interface_matched.py` (`--phasef` for the Phase F tables) compares the interface measures with the 8 × 8 results and writes `INTERFACE_ADDENDUM.md/json`; cross-experiment correlations are exploratory.

## Usage

```bash
OUT=rerun_best_validation/followups/mechanism_v1/E1_interface
python tools/mechanism_v1/E1_interface_harvest_v1/scripts/harvest_interface_rows.py --project-root . --out "$OUT" --require-complete
python tools/mechanism_v1/E1_interface_harvest_v1/scripts/analyze_interface_magnitude.py --in "$OUT" --out "$OUT"
```

No figure reads these outputs.
