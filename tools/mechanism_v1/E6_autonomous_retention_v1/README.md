# C1: E6 source-side autonomous retention

## Purpose

Measure autonomous recurrent retention for all 210 E6 rho cores without retraining anything.  This is a post-E6 exploratory follow-up and does not alter the frozen `ANALYSIS_PLAN.md`.

Primary quantity:

`||ReLU(W_rho h)|| / ||h||`

on the same held-out navigation H_nav state set used by E5b' (4096 sampled states after burn-in).  To obey the N7 invariance requirement discovered after E6, states are aligned by the E6 permutation P only.  The sign matrix D is the perturbation and is not applied to the state set.  Therefore rho=0.1 and rho=0.9, which share the same recurrent matrix and differ only in input sign, must have identical primary retention.

## Why this differs from Step 6 H_nav

Step 6 let each transformed core run navigation input from Q h0.  That was correct for its own source-task probe, but E6 later showed such trajectory-derived source measures can differ between rho=0.1 and rho=0.9 even though their downstream networks are equivalent.  C1 specifically asks for a core-retention dose curve that can be used to set the later W->cW intervention, so the primary measure must be input-sign invariant.

## Compute organization

The scientific coverage is still 210 cores (10 sources x 3 E6 targets x 7 rho values), but the job array has only 30 jobs.  One job handles all seven rho values for one (source,target) pair so the navigation H_nav state set is generated only once.

The LSF resource request intentionally follows the already validated Step 6 pure-forward job: queue `<LSF_QUEUE>`, one host, one GPU.  There is no training in C1; the GPU is only for the 4096-dimensional recurrent forward products.

## Run

From the project root after deployment:

```bash
ROOT=<PROJECT_ROOT>
C1=$ROOT/tools/mechanism_v1/E6_autonomous_retention_v1/lsf

bash $C1/submit_absolute.sh --smoke
bash $C1/check_status_absolute.sh
# Only continue if smoke prints PASS and both anchor deviations are tiny.

bash $C1/submit_absolute.sh --all
bash $C1/check_status_absolute.sh
```

## Outputs

Sidecar only:

`rerun_best_validation/followups/mechanism_v1/E6_autonomous_retention/`

Key aggregate files:

- `all_retention.csv`: 210 core measurements
- `source_level_retention.csv`: targets averaged within source
- `rho_curve.csv`: retention and E6 seen_long dose curves
- `rho0_contrasts.csv`: exploratory paired source contrasts and retention ratios
- `RETENTION_RESULT.md/json`: anchors, curve, and the rho=0.1 / rho=0 source-side retention ratio for designing the later W->cW preregistration

No file under E6, E5b', `aggregate_full`, or historical LSF directories is modified.
