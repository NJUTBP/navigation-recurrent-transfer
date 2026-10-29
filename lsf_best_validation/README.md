# Downstream experiments of the 4 × 4 task

`run_best_validation_source.py` runs, for one navigation source, the downstream experiments of the 4 × 4 task on the selected navigation checkpoint without retraining navigation. The released 4 × 4 results (Figures 3–5 and S4) and the factorial models used for the path-invariance tests of Supplementary Fig. S2 were produced with this driver.

## Checkpoints, seeds and budgets

Each source uses `sources/nav_seed_<s>/artifacts/navigation/best.pt`, the epoch with the minimum navigation validation task loss as defined in `exp1_nav_pretrain.py`; the preflight reads `history.json` and stops if the epoch recorded in `best.pt` is not the minimum of `validation_task_loss`. The delayed-reconstruction core of each source is `sequence_control/best_validation.pt` (not retrained). Target seeds are read from `plan.json`, and the downstream training budgets are those of the configurations in `code/gpm_cmap/configs`.

## Families

| `--family` | Stages |
|---|---|
| `preflight` | checkpoint and input checks, no GPU |
| `main` | Exp5 navigation conditions (`transfer_freeze`, `transfer_finetune`, `random_init`, `random_frozen`, `transfer_input_only`, `transfer_recurrent_only`), Exp5 delayed-reconstruction core, mapping controls, test-time transforms |
| `adapter_symbolic` | Exp7 adapter rescue, Exp8 symbolic adapter |
| `hard` | harder 8 × 8 benchmark of the plan |
| `factorial` | recurrent factorial design |
| `all` | all of the above |

The topology analyses of the same checkpoints are in `topology/`.

## Usage

The run root contains `plan.json`, `code/` and `sources/`, for example the repository root with P1 extracted:

```bash
python lsf_best_validation/run_best_validation_source.py --run-root . --output-root outputs/rerun_best_validation \
    --source-seed 6101 --family preflight --python "$(which python)"
python lsf_best_validation/run_best_validation_source.py --run-root . --output-root outputs/rerun_best_validation \
    --source-seed 6101 --family main --python "$(which python)" --device cuda
```

Never use the run root itself as output root. The released results use the layout `--output-root rerun_best_validation` (as in P3). Each source gets `<output-root>/sources/nav_seed_<s>/` with `preflight.json`, `source_identity.json`, `configs/`, `logs/`, `state/`, `attempts/try_<nnn>/` and `artifacts/<stage>` pointing to the committed attempt; completed stages with matching fingerprints are reused. The 4 × 4 aggregate is built by `topology/schoyen_formal_figureA3_patch_v1/unified_aggregate/aggregate_best_validation_full_v2.py`.
