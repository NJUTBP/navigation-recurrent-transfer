# Navigation-trained recurrent organization facilitates relational learning

Code, configurations, seeds and figure source data for the article *Navigation-trained recurrent organization facilitates relational learning* (Li, Qin, Wu, Wang, Liu; submitted to *Nonlinear Dynamics*).
Trained network weights and the complete source-level results are distributed as separate data packages (see [Data](#data)).

Code repository: [NJUTBP/navigation-recurrent-transfer](https://github.com/NJUTBP/navigation-recurrent-transfer).

**What the study does.** Ten 4,096-unit ReLU RNNs were trained on 2D path integration. Their state-transition core (input and recurrent matrices) was then frozen while new task-specific mappings were learned for a relational endpoint task on 4 × 4 and 8 × 8 lattices. Navigation-trained cores were compared with delayed-reconstruction-trained, random, matched (signed-permutation) and fresh-random cores, and the long-path advantage was probed with graded signed-coordinate perturbations and recurrent rescaling.

## Repository layout

| Path | Content |
|---|---|
| `code/gpm_cmap/` | Models, tasks, navigation and delayed-reconstruction pretraining, 4 × 4 transfer experiments, topology analyses (formal code, only modules reached by the paper's experiments) |
| `code/multinav/` | Per-source orchestration, statistics and aggregation of the 4 × 4 experiments |
| `lsf_best_validation/` | Driver that reran all downstream experiments on the selected (best-validation) checkpoints |
| `tools/hard8x8_phaseF_fullscale_v1/` | Extended 8 × 8 task (protocol, training, collection) |
| `tools/FigureA5_PhaseF_workflow_v1/` | Freezes the 8 × 8 results into the figure source data |
| `tools/mechanism_v1/` | Mechanistic follow-ups: extended budget, gain sweep, additive baselines, forward diagnostics, source measures, signed-coordinate series, retention, rescaling, navigation-interface refitting, verdicts |
| `tools/reevaluation/` | Rebuilds the released frozen-core target models and recomputes their accuracies; task-state retention (Fig. S12c) |
| `topology/` | Schøyen-style torus analysis, formal topology / reduced dynamics / path invariance (Fig. S2), torus visualization (Fig. S3) |
| `extensions/symbolic_baseline7000_extension_v1/` | 7,000-trial points of the non-recurrent baselines (Fig. 5) |
| `figures/` | The single plotting entry `redraw_all_figures.py` for Figs. 2–7 and S1–S12, and the builders it calls |
| `data/aggregate_full/` | Figure source data (Level 1) |
| `data/source_level_results/` | Small source-level tables that are not part of the figure folders |
| `configs/frozen_plan/`, `plan.json` | Frozen experimental plan: sources, seeds, budgets, comparison contrasts |
| `release_tools/` | Scripts that assembled the data packages (cluster) and the figure data (local) |
| `docs/` | Figure–code–data map, reproduction guide and data-package contents |

## Installation

```bash
conda create -n nrt python=3.11
conda activate nrt
pip install -r requirements-figures.txt
python release_tools/local/verify_release.py --repo .
```
Use Python 3.11 for the pinned figure environment. To reconstruct target models, install `requirements-model-checks.txt` and choose the Torch CPU/CUDA build appropriate for your hardware. The original Linux/Python-3.12 cluster requirements for training and topology analyses are in `requirements.txt`. See [reproduction instructions](docs/REPRODUCIBILITY.md) for the environment and experiment coverage.

## Data

| Package | Content | Needed for |
|---|---|---|
| in this repository: `data/aggregate_full/` | Source data of every figure (CSV/JSON, one NPZ for Fig. S3) | Level 1 |
| `P1_pretrained_checkpoints.tar` | Selected navigation (seeds 6101–6110) and delayed-reconstruction (seeds 7101–7110) networks, weights only, with training histories | Levels 2–3 |
| `P2_target_checkpoints.tar` | Frozen-core target models of Figs. 6–7, S5–S8 and S12 (the frozen core is not stored; it is rebuilt from P1 and recorded seeds) | Level 2 |
| `P3_source_level_results.tar.gz` | Source-, cell- and target-level results, split manifests and protocols of all formal experiments | Levels 1–2 |

Model weights and source-level results are available from [Zenodo, DOI: 10.5281/zenodo.23660866](https://doi.org/10.5281/zenodo.23660866). Extract the packages **into the repository root**; they create `sources/` and `rerun_best_validation/` with the original relative paths, which the code reads without further configuration:

```bash
tar -xf P1_pretrained_checkpoints.tar
tar -xf P2_target_checkpoints.tar
tar -xzf P3_source_level_results.tar.gz
```

On Windows, use a short repository path and the checksum/path-checking helper instead of extracting under a deeply nested directory:

```bash
python release_tools/local/extract_packages.py --packages /path/to/packages --repo .
```

Archive checksums are supplied in `data/PACKAGE_SHA256SUMS.txt`. Model indices and the checkpoint-selection convention are described in [data-package contents](docs/DATA_PACKAGES.md).
The released weights cover 20 pretrained networks and 480 target checkpoints. Figure-specific commands are listed in `docs/REPRODUCIBILITY.md` and `docs/FIGURE_DATA_MAP.md`.
If the data live elsewhere, set `NRT_PROJECT_ROOT` to a folder that contains `code/`, `tools/`, `sources/` and `rerun_best_validation/`.

## Reproduction

### Level 1 — redraw the figures (CPU, minutes)

```bash
python figures/redraw_all_figures.py --aggregate data/aggregate_full --out outputs/figures
python figures/redraw_all_figures.py --aggregate data/aggregate_full --out outputs/figures --figures Figure7,SupplementaryFigureS12
python figures/export_source_data_final.py --aggregate data/aggregate_full --out outputs/source_data_final   # folders named by final figure number
```
Figure 1 is an author-drawn schematic (`figures/figure1/`). The statistics shown in each figure are stored next to its data (`*_statistics.json`, `contrasts.csv`); the scripts that computed them are listed in `docs/FIGURE_DATA_MAP.md`.

### Level 2 — re-evaluate trained networks (GPU recommended)

```bash
# 8 x 8 and mechanism target models: rebuild, recompute the four test accuracies, compare with the stored values
python tools/reevaluation/reevaluate_models.py --family phasef --out outputs/reeval_phasef.csv
python tools/reevaluation/reevaluate_models.py --family e09a   --out outputs/reeval_e09a.csv
python tools/reevaluation/reevaluate_models.py --family e6     --out outputs/reeval_e6.csv
python tools/reevaluation/reevaluate_models.py --family e6i    --out outputs/reeval_e6i.csv
# Supplementary Fig. S12: task-state retention and the figure's source data
python tools/reevaluation/task_state_retention.py --out outputs/task_state_retention.csv
python tools/mechanism_v1/E6_retention_intervention_v1/scripts/export_figureS12_source_data.py \
    --mech-root rerun_best_validation/followups/mechanism_v1 \
    --task-state-csv outputs/task_state_retention.csv --out outputs/S12_source_data
```
Source-side analyses of the pretrained cores (retention κ, Fig. 7d; source measures, Fig. S11) and the navigation and topology analyses (Figs. 2, S2) run from P1 with the scripts listed in `docs/FIGURE_DATA_MAP.md`.

### Level 3 — retrain (GPU, hours to days)

Seeds, budgets and conditions are fixed in `plan.json`, `tools/hard8x8_phaseF_fullscale_v1/PROTOCOL.json` and the protocol files of each `tools/mechanism_v1/*` experiment. Typical entry points:

```bash
export PYTHONPATH=$PWD/code
python -m multinav.stage navigation       --config <per-source navigation config>   # pretraining (P3 holds the configs used)
python -m multinav.stage sequence_control --config <per-source control config>
python lsf_best_validation/run_best_validation_source.py --run-root . --output-root outputs/rerun \
       --source-seed 6101 --family main --python "$(which python)" --device cuda      # 4 x 4 experiments
python tools/hard8x8_phaseF_fullscale_v1/scripts/run_group.py --source-seed 6101 --target-seed 84001 --group learned_interface
python tools/mechanism_v1/E6_rho_scan_v1/scripts/run_rho_scan.py --source-seed 6101 --target-seed 84001 --rho 0.1
python tools/mechanism_v1/E6_retention_intervention_v1/scripts/run_intervention.py --source-seed 6101 --target-seed 84001 --arm nav_impair
```
See `docs/REPRODUCIBILITY.md` for the full sequence, collectors and statistics.

## Citation and license

Please cite the article (see `CITATION.cff`). Code: MIT License (`LICENSE`). Models, data and author-provided figure images: CC BY 4.0 (`DATA_LICENSE.md`). The current release excludes the B1–B6 follow-up experiments.
