# Torus topology of the navigation cores

`torus_chain.py` applies the automatic Schøyen-style torus analysis to every navigation source on its `navigation/best.pt`, without retraining navigation. The analysis (`code/gpm_cmap/experiments/analyze_schoyen_torus.py`, cohort summary `aggregate_schoyen_torus.py`) uses fresh true-position clustering rollouts, morphology-ranked DBSCAN clusters, strict-to-broad candidate unions, PCA6 with fuzzy downsampling and independent Euclidean and cosine Ripser tests; its settings are the `schoyen_topology` block of `code/multinav/config.yaml`. The wrapper verifies that `best.pt` is the minimum-`validation_task_loss` checkpoint recorded in the source's `history.json`; topology and all transfer experiments use this checkpoint.

The formal topology, reduced-dynamics and mechanism tests built on this analysis (Supplementary Fig. S2) are in `topology/schoyen_formal_figureA3_patch_v1`.

## Usage

Subcommands: `preflight`, `run` (one source: `--seed`, `--device`, `--force`), `aggregate` and `postflight`; `python topology/schoyen_topology_best_validation_cluster_v3/torus_chain.py <subcommand> --help` lists their arguments (`--run-root`, `--result-root`, `--code-root`).

Inputs: `<run-root>/sources/nav_seed_<s>/artifacts/navigation/best.pt` and `config_used.yaml`. Outputs:

```text
<result-root>/
  sources/nav_seed_<s>/
    attempts/torus_topology/try_<nnn>/
    artifacts/torus_topology -> ../../attempts/torus_topology/try_<nnn>
    state/torus_topology.json
    topology_qualification.json
  aggregate/torus_auto/
    torus_auto_summary.csv
    torus_auto_candidates.csv
    torus_auto_robust_scores.png
    coverage.json
```

A source is reused when its fingerprint (source seed, checkpoint SHA-256, navigation-config SHA-256 and analyzer SHA-256) matches a completed attempt; otherwise a new attempt is created. Figures: S2a (via the formal topology of `schoyen_formal_figureA3_patch_v1`) and S3.
