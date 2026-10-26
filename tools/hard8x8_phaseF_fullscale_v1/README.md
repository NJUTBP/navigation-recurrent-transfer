# hard8x8_phaseF_fullscale_v1

This is the single large-scale harder-8×8 training and test package requested after Phases A–D.

It does **not** modify historical `hard_*`, `factorial_*`, `aggregate_full`, or prior Phase B/C/D outputs. Everything is written to a new sidecar.

## Scale

- 10 navigation sources: 6101–6110
- 5 fresh target splits: 84001–84005
- 15 training conditions
- 750 training cells total
- 8000 unique training trajectories per target split
- fixed 8000 gradient updates per training cell
- 1000 validation trajectories
- four independent 2000-trajectory test sets per target:
  - seen composition × short horizon
  - held-out composition × short horizon
  - seen composition × long horizon
  - held-out composition × long horizon

The three job arrays run in parallel:

- `hard8_fA[1-50]`: fixed-scale frozen-core controls (gain 1 / 3 / 10)
- `hard8_fB[1-50]`: learned scalar / full-2×2 interface conditions, including fresh-random controls
- `hard8_fC[1-50]`: full recurrent fine-tuning conditions

A dependent collector runs after all three arrays finish.

## Fixed absolute paths

Package:

`<PROJECT_ROOT>/tools/hard8x8_phaseF_fullscale_v1`

Output:

`<PROJECT_ROOT>/rerun_best_validation/hard8x8_diagnostics/audit_v1/phaseF_fullscale_v1`

Logs:

`<PROJECT_ROOT>/HPC_logs_hard8x8_diag/phaseF_fullscale_v1`

Python:

`<PYTHON>`

Queue:

`<LSF_QUEUE>`

## Submit

```bash
unzip -o hard8x8_phaseF_fullscale_v1.zip \
  -d <PROJECT_ROOT>/tools

bash <PROJECT_ROOT>/tools/hard8x8_phaseF_fullscale_v1/lsf/submit_absolute.sh
```

No environment variables need to be set first.

## Check after jobs finish

```bash
bash <PROJECT_ROOT>/tools/hard8x8_phaseF_fullscale_v1/scripts/check_status_absolute.sh
```

Final compact result bundle:

`<PROJECT_ROOT>/rerun_best_validation/hard8x8_diagnostics/audit_v1/phaseF_fullscale_v1/hard8x8_phaseF_fullscale_v1_results.zip`

Full checkpoints remain on the cluster and are not included in the result ZIP.

## Important analysis rules

- inferential unit = navigation source seed, not target repeat and not trajectory;
- average the five target repeats within source before source-level inference;
- primary contrast = Nav trainable scalar gain minus matched trainable scalar gain on `id_seen_short`;
- primary test = two-sided exhaustive exact sign-flip across 10 source seeds;
- test sets are never used for checkpoint selection;
- generalization tests form an explicit composition × horizon crossing;
- fresh-random controls use source-target-specific random recurrent cores while preserving paired start/readout initialization;
- frozen-core checkpoint files omit the large reconstructible recurrent matrices; full-core fine-tuning checkpoints keep the full recurrent state.

See `SUMMARY_8X8_EXPERIMENTS.md` for the complete history and `PREREGISTERED_ANALYSIS.md` for the frozen analysis plan.
