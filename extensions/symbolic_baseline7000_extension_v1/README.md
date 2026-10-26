# Symbolic linear-baseline 7,000-trial extension

This package adds the missing **7,000 training-trial** point for exactly two Exp8 non-recurrent baselines:

- `baseline_action_count_linear`
- `baseline_linear_accumulator`

It does **not** rerun navigation, CTL, random recurrent, topology, FigureA3, or the existing baseline budgets 10--1000.

## Scientific design

For each of the 30 existing symbolic target replicates (10 navigation sources x 3 target seeds), the extension loads that target's committed `symbolic_t<target>/config_used.yaml`, keeps the same target seed, data generator, train/validation/test split, baseline training schedule and evaluation set, and runs only budget 7000.  The extension split audit must match the original symbolic split audit exactly.

Total new fits: `30 targets x 2 baselines x 1 budget = 60`.

## Run

From the current `multinav_paper_v1` root:

```bash
unzip symbolic_baseline7000_extension_v1.zip
bash symbolic_baseline7000_extension_v1/submit_baseline7000.sh
```

The LSF array uses GPU by default to match the device used by the original symbolic stage.  It runs one navigation source per array index and all three target replicates sequentially.

## New raw result artifacts

For source 6101 / target 1000, for example:

```text
rerun_best_validation/sources/nav_seed_6101/
  artifacts/symbolic_baseline7000_t1000/
    symbolic_adapter_rows.json
    symbolic_adapter_rows.csv
    summary_symbolic_adapter.json
    protocol_manifest.json
    split_audits/seed_1000.json
    baseline7000_extension_manifest.json
```

Each `symbolic_adapter_rows.json` contains exactly two rows, both at budget 7000.

The original `artifacts/symbolic_t<target>/` directories are **not modified or replaced**.

## Final aggregate update

After all 30 target extensions pass validation, `finalize_aggregate_extension.py`:

1. copies the current `rerun_best_validation/aggregate_full` into a new `aggregate_full_attempts/try_XXX`;
2. appends 60 target-level accuracy rows to `target_level_metrics.csv`;
3. appends 20 source-level rows (10 sources x 2 baselines) to `source_means.csv`;
4. appends the corresponding two rows to `source_level_summary.csv`;
5. writes `symbolic_baseline7000_extension_contrasts.csv` and `symbolic_baseline7000_extension_by_source.csv` as **secondary/descriptive extension** statistics;
6. regenerates paper figures from the new aggregate attempt;
7. atomically repoints `rerun_best_validation/aggregate_full` to the new attempt.

The original confirmatory 1,000-trial and 10--200-trial AUC contrast files are intentionally unchanged.

## What is replaced

No original raw experiment artifact is replaced.

The *current aggregate view* is replaced by a new aggregate attempt.  Therefore the following current-view files gain the added 7000-baseline data:

- `aggregate_full/target_level_metrics.csv`
- `aggregate_full/source_means.csv`
- `aggregate_full/source_level_summary.csv`
- `aggregate_full/audit.json`
- `aggregate_full/input_artifact_checksums.json`
- `aggregate_full/figure_source_data/Figure5/symbolic.csv`
- regenerated `aggregate_full/figures/Figure5.*`

The previous aggregate attempt remains intact under `aggregate_full_attempts/`.

## Future from-scratch runs

For a future full reproduction, patch the DEVELOPMENT source tree (not this already-frozen run):

```bash
python symbolic_baseline7000_extension_v1/patch_future_protocol.py /path/to/development/source
```

This changes both symbolic configs to `controls.baseline_budgets: all`, so the two simple baselines automatically follow the full episode budget list including 7000.
