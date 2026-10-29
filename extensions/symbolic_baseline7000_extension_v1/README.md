# 7,000-trial points of the non-recurrent symbolic baselines

Adds the 7,000-training-trial point of the two non-recurrent baselines of the arbitrary-symbol experiment (Fig. 5), `baseline_action_count_linear` and `baseline_linear_accumulator`. Navigation, delayed reconstruction, random recurrent cores, topology and the baseline budgets 10–1,000 are not rerun.

## Design

For each of the 30 symbolic target replicates (10 navigation sources × 3 target seeds), the extension loads that target's `symbolic_t<target>/config_used.yaml` and keeps its target seed, data generator, train/validation/test split, baseline training schedule and evaluation set; only budget 7,000 is run. The split audit of the extension must match the original symbolic split audit exactly. New fits: 30 targets × 2 baselines × 1 budget = 60.

## Usage

```bash
python extensions/symbolic_baseline7000_extension_v1/run_baseline7000_extension.py --project . preflight
python extensions/symbolic_baseline7000_extension_v1/run_baseline7000_extension.py --project . run-source --source-seed 6101
python extensions/symbolic_baseline7000_extension_v1/run_baseline7000_extension.py --project . verify
python extensions/symbolic_baseline7000_extension_v1/finalize_aggregate_extension.py --project .
```

`--project` is the run root that contains `rerun_best_validation/`. `run-source` handles one navigation source and its three targets; the GPU is used by default, as in the symbolic stage.

## Outputs

For source 6101 and target 1000, for example:

```text
rerun_best_validation/sources/nav_seed_6101/artifacts/symbolic_baseline7000_t1000/
  symbolic_adapter_rows.json      (two rows, both at budget 7000)
  symbolic_adapter_rows.csv
  summary_symbolic_adapter.json
  protocol_manifest.json
  split_audits/seed_1000.json
  baseline7000_extension_manifest.json
```

The original `artifacts/symbolic_t<target>/` folders are not modified.

After all 30 targets pass validation, `finalize_aggregate_extension.py`

1. copies the current `rerun_best_validation/aggregate_full` into a new `aggregate_full_attempts/try_<nnn>`;
2. appends 60 target-level accuracy rows to `target_level_metrics.csv`;
3. appends 20 source-level rows (10 sources × 2 baselines) to `source_means.csv`;
4. appends the corresponding two rows to `source_level_summary.csv`;
5. writes `symbolic_baseline7000_extension_contrasts.csv` and `symbolic_baseline7000_extension_by_source.csv` as secondary, descriptive statistics;
6. regenerates the figures from the new aggregate attempt;
7. points `rerun_best_validation/aggregate_full` to the new attempt.

The confirmatory 1,000-trial and 10–200-trial AUC contrast files are unchanged. The previous aggregate attempt remains under `aggregate_full_attempts/`; the updated aggregate view adds the 7,000-trial baseline rows to `target_level_metrics.csv`, `source_means.csv`, `source_level_summary.csv`, `audit.json`, `input_artifact_checksums.json` and `figure_source_data/Figure5/symbolic.csv`.

## Runs from scratch

`patch_future_protocol.py <code tree>` sets `controls.baseline_budgets` to `all` in the symbolic configurations of a code tree, so that the two baselines are trained at all budgets including 7,000 and this extension is not needed. `code/gpm_cmap/configs/symbolic_adapter_ctl01.yaml` keeps the budgets of the article run (10–1,000), to which this extension adds 7,000.
