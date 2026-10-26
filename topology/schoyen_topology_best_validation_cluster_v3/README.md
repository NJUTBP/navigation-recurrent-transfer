# V3 cluster scheduling fix

This package fixes a cluster-specific LSF issue observed in jobs 14149443/14149445: arbitrary variables exported by the submission shell were not inherited by the batch job. `submit_topology_chain.sh` now writes resolved absolute paths to `<RUN_ROOT>/.schoyen_topology_runtime.env`, and every LSF job sources that file using `LS_SUBCWD`/`PWD`. The LSF scripts therefore no longer depend on environment propagation. Aggregate additionally requires successful preflight.

# Schøyen topology cohort on navigation `best.pt`

This package applies the new automatic Schøyen-style torus analysis to the
**current best-validation rerun protocol** without retraining navigation.

## Scientific contract

- Navigation source seeds: `6101..6110`.
- Topology checkpoint: each source's existing `artifacts/navigation/best.pt`.
- The wrapper verifies that `best.pt` is the minimum `validation_task_loss`
  checkpoint recorded in that source's navigation `history.json`.
- The automatic topology analysis uses the supplied patch defaults:
  fresh TRUE-position clustering rollouts, morphology-ranked DBSCAN clusters,
  strict-to-broad candidate unions, PCA6 + fuzzy downsampling, and independent
  Euclidean/cosine Ripser tests.
- The main transfer policy in the installed code is also changed to
  `checkpoint_policy: best_validation` for future new run IDs.

## Important: the legacy Source-v3 tail is retired

The new `torus_topology` replaces the old `topology_export + formal_topology`
route. It deliberately does **not** create the cohomological phase-coordinate
artifact expected by the old:

```text
source_splits -> source_reduced -> action_causality / path_invariance
```

Therefore those stages are not submitted here. Connecting them to the new torus
output would require a new reduced-model/phase-reference scientific definition,
not merely an LSF change.

For the current protocol, the executable topology chain is:

```text
preflight
   -> 10-source torus_topology array
   -> cohort aggregate
   -> postflight coverage check
```

## Where this writes

Inputs are read from the original full run root:

```text
<RUN_ROOT>/sources/nav_seed_<seed>/artifacts/navigation/best.pt
<RUN_ROOT>/sources/nav_seed_<seed>/artifacts/navigation/config_used.yaml
```

By default new outputs join the best-validation rerun:

```text
<RUN_ROOT>/rerun_best_validation/
  sources/nav_seed_<seed>/
    attempts/torus_topology/try_001/
    artifacts/torus_topology -> ../../attempts/torus_topology/try_001
    state/torus_topology.json
    topology_qualification.json
  aggregate/torus_auto/
    torus_auto_summary.csv
    torus_auto_candidates.csv
    torus_auto_robust_scores.png
    coverage.json
```

Existing behavioural rerun outputs are not modified.

## 1. Install the code patch into the frozen code copy

Unpack this directory under the existing run root, for example:

```text
multinav_paper_v1/
  code/
  sources/
  rerun_best_validation/
  schoyen_topology_best_validation_cluster/
```

Then:

```bash
cd /path/to/multinav_paper_v1
bash schoyen_topology_best_validation_cluster/install_into_frozen_run.sh "$PWD"
```

The installer:

1. backs up every replaced file under `code_backup_pre_schoyen_<timestamp>/`;
2. installs the two new topology experiment modules;
3. updates `multinav/{stage,run_source,plan,collect,plot,config}`;
4. sets the future formal `checkpoint_policy` to `best_validation`;
5. runs `py_compile`;
6. **does not modify `plan.json`**.

Do not resubmit the frozen `multinav_paper_v1/plan.json` as a new full run after
changing code/config. The standalone topology cohort is the intended route for
this existing dataset.

## 2. Submit the topology chain

From the run root:

```bash
bash schoyen_topology_best_validation_cluster/submit_topology_chain.sh
```

Defaults:

```text
RUN_ROOT    = parent directory of this package
RESULT_ROOT = <RUN_ROOT>/rerun_best_validation
CODE_ROOT   = <RUN_ROOT>/code
PYTHON_BIN  = <PYTHON>
```

Override if needed:

```bash
export RUN_ROOT=/path/to/multinav_paper_v1
export RESULT_ROOT=/path/to/new_best_validation_results
export CODE_ROOT="$RUN_ROOT/code"
export PYTHON_BIN=<PYTHON>
bash schoyen_topology_best_validation_cluster/submit_topology_chain.sh
```

The topology array is `[1-10]%2`: at most two source analyses run concurrently.
Increase `%2` only if your GPU quota permits it.

## LSF path handling

The `.lsf` jobs do **not** use `BASH_SOURCE[0]` to locate the package. LSF copies
submitted scripts into `~/.lsbatch`, so that pattern is unsafe. Instead the direct
submission script exports an absolute `TOPOLOGY_PACKAGE_DIR` before calling `bsub`.

## Logs

```text
<RUN_ROOT>/HPC_logs_torus/
```

Useful checks:

```bash
bjobs -w
find "$RESULT_ROOT/sources" -path '*/state/torus_topology.json' -print
cat "$RESULT_ROOT/aggregate/torus_auto/coverage.json"
```

## Rerunning failed seeds

The wrapper uses attempt directories and a fingerprint over source seed,
checkpoint SHA256, navigation config SHA256, and analyzer SHA256. A completed
matching source prints `[reuse]`; a failed or changed source creates the next
`try_XXX` attempt. You can therefore safely resubmit the array.
