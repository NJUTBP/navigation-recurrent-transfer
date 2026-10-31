# Navigation-trained recurrent organization facilitates relational learning

Code, configurations, random seeds and figure source data for the article *Navigation-trained recurrent organization facilitates relational learning* (Li, Qin, Wu, Wang, Liu; submitted to *Nonlinear Dynamics*). Trained network weights and the complete source-level results are distributed as separate data packages (see [Data and pretrained models](#data-and-pretrained-models)).

Code repository: [NJUTBP/navigation-recurrent-transfer](https://github.com/NJUTBP/navigation-recurrent-transfer).

## Overview

The study asks whether the recurrent organization that a network acquires through spatial navigation can be reused for relational learning. Ten 4,096-unit ReLU RNNs were trained on two-dimensional path integration (navigation sources, seeds 6101–6110), each paired with a network trained on delayed reconstruction (seeds 7101–7110). The state-transition core of each network (input matrix W_in and recurrent matrix W_rec) was then frozen, and only new task-specific mappings were learned for a relational endpoint task on 4 × 4 and 8 × 8 lattices. Navigation-trained cores were compared with delayed-reconstruction-trained, random, matched (signed-permutation) and fresh-random cores, and the long-path advantage of the navigation cores was probed with graded signed-coordinate perturbations of the core and with recurrent rescaling.

The ten navigation sources are the independent statistical unit throughout: target repeats are averaged within source, tests are exact sign-flip tests over the 2^10 sign patterns, confidence intervals are source-bootstrap intervals, and Holm correction is applied within prespecified comparison families.

## Repository structure

| Component | Location |
|---|---|
| Model, task and training code | `code/gpm_cmap/`: models, navigation environment, relational tasks, experiment entry points and configurations ([code/README.md](code/README.md)) |
| Navigation and delayed-reconstruction pretraining | `code/gpm_cmap/experiments/exp1_nav_pretrain.py`, `pretrain_sequence_control.py`, run through `code/multinav/stage.py` |
| Frozen-core transfer and controls, 4 × 4 task (Figs. 3–5, S4) | `code/gpm_cmap/experiments/exp5_abstract_pathint.py`, `exp7_adapter_rescue.py`, `exp8_symbolic_adapter.py`; per-source driver `lsf_best_validation/`; 7,000-trial baseline points `extensions/symbolic_baseline7000_extension_v1/` |
| Orchestration and statistics of the 4 × 4 cohort | `code/multinav/`; aggregation `topology/schoyen_formal_figureA3_patch_v1/unified_aggregate/` |
| Extended 8 × 8 task, "Phase F" (Figs. 6, S5, S6) | `tools/hard8x8_phaseF_fullscale_v1/` (protocol, training, collection, tests); `tools/FigureA5_PhaseF_workflow_v1/` (copies the results into the figure data) |
| Mechanistic analyses (Figs. 7, S7–S11) | `tools/mechanism_v1/` ([index](tools/mechanism_v1/README.md)) |
| Topology analyses (Figs. S2, S3) | `topology/` |
| Re-evaluation of the released models | `tools/reevaluation/` |
| Figure reproduction | `figures/` ([figures/README.md](figures/README.md)) |
| Figure source data | `data/aggregate_full/` ([data/README.md](data/README.md)) |
| Frozen experimental plan: sources, seeds, budgets, comparison contrasts | `plan.json`, `configs/frozen_plan/` |
| Documentation | `docs/` |
| Tests | `tests/`, `tools/hard8x8_phaseF_fullscale_v1/tests/` |
| Release tooling | `release_tools/` |

Several folder names keep the identifiers under which the experiments were run, for example Phase F for the extended 8 × 8 task, E0.9a or E6 for mechanistic analyses, and FigureA3 for the builder of Supplementary Fig. S2. They are kept because the data packages, checkpoints and provenance records refer to them; [docs/FIGURE_DATA_MAP.md](docs/FIGURE_DATA_MAP.md) and [tools/mechanism_v1/README.md](tools/mechanism_v1/README.md) relate them to the article.

## Installation

```bash
conda create -n nrt python=3.11
conda activate nrt
pip install -r requirements-figures.txt
```

Use Python 3.11 for the pinned figure environment. To rebuild and re-evaluate target models, first install the CPU or CUDA build of PyTorch 2.7.0 appropriate for your hardware, then `requirements-model-checks.txt`. Training and topology analyses were run on Linux with Python 3.12; that environment is recorded in `requirements.txt` (it includes `ripser` and `umap-learn`). The tests run with `pip install pytest` and `python -m pytest` from the repository root; tests that need PyTorch are skipped when it is not installed. See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for environments and coverage.

## Data and pretrained models

| Package | Content | Needed for |
|---|---|---|
| in this repository: `data/aggregate_full/` | Source data of every computational figure (CSV/JSON) | Level 1 |
| `P1_pretrained_checkpoints.tar` | Selected navigation (seeds 6101–6110) and delayed-reconstruction (seeds 7101–7110) networks, weights only, with training histories | Levels 2–3 |
| `P2_target_checkpoints.tar` | Frozen-core target models of Figs. 6–7, S5–S8 and S11 (the frozen core is not stored; it is rebuilt from P1 and the recorded seeds) | Level 2 |
| `P3_source_level_results.tar.gz` | Source-, cell- and target-level results, split manifests and protocols of all formal experiments | Levels 1–2 |

Model weights and source-level results are available from [Zenodo, DOI: 10.5281/zenodo.23660866](https://doi.org/10.5281/zenodo.23660866). Extract the packages **into the repository root**; they create `sources/` and `rerun_best_validation/` with the original relative paths, which the code reads without further configuration:

```bash
tar -xf P1_pretrained_checkpoints.tar
tar -xf P2_target_checkpoints.tar
tar -xzf P3_source_level_results.tar.gz
```

On Windows, use a short repository path and the helper that checks checksums and path lengths before extracting:

```bash
python release_tools/local/extract_packages.py --packages /path/to/packages --repo .
```

Archive checksums are in `data/PACKAGE_SHA256SUMS.txt`. The released weights cover 20 pretrained networks and 480 target checkpoints; model indices and the checkpoint-selection convention are described in [docs/DATA_PACKAGES.md](docs/DATA_PACKAGES.md). If the packages are extracted elsewhere, set `NRT_PROJECT_ROOT` to a folder that contains `code/`, `tools/`, `sources/` and `rerun_best_validation/`.

## Reproducing the figures

Level 1 needs only this repository and runs on a CPU in about a minute:

```bash
python figures/redraw_all_figures.py --aggregate data/aggregate_full --out outputs/figures
python figures/redraw_all_figures.py --aggregate data/aggregate_full --out outputs/figures --figures Figure7,SupplementaryFigureS11
python figures/export_source_data_final.py --aggregate data/aggregate_full --out outputs/source_data_final
```

The first command writes `Figure2` … `Figure7` and `SupplementaryFigureS1` … `SupplementaryFigureS12` as PNG, SVG, PDF and TIFF to `outputs/figures/`; the second redraws selected figures; the third copies the source data into folders named by article figure number. The figures are drawn from the stored tables only, and the statistics shown in them are read from the tables stored next to the data (`*_statistics.json`, `contrasts.csv`). Neither tool writes into `data/`. Figure 1 is an author-drawn schematic (`figures/figure1/`).

| Figure | Content | Source data (`data/aggregate_full/figure_source_data/`) |
|---|---|---|
| 1 | Schematic of the experimental logic (author-drawn) | – |
| 2 | Navigation pretraining | `Figure2` |
| 3 | Frozen-core transfer, 4 × 4 task | `Figure3` |
| 4 | Relation-input transformations | `Figure4` |
| 5 | Arbitrary relation symbols | `Figure5` |
| 6 | Extended 8 × 8 task | `FigureA5`, `../hard8x8_phaseF_followup` |
| 7 | Recurrent perturbation and rescaling | `Figure7` |
| S1 | Delayed-reconstruction pretraining | `FigureA1` |
| S2 | Topology, reduced dynamics, path invariance | `FigureA3` |
| S3 | Torus visualization | `FigureA4` |
| S4 | Relation and endpoint mapping controls | `FigureA2` |
| S5 | 8 × 8 transfer controls | `FigureA5` |
| S6 | Optimization and split diagnostics | `FigureA5`, `../hard8x8_phaseF_followup` |
| S7 | Extended budget, gain and additive controls | `SupplementaryFigureS8` |
| S8 | Multi-step state maintenance | `SupplementaryFigureS9` |
| S9 | Refitting of the navigation mappings | `SupplementaryFigureS11` |
| S10 | Prespecified source-measure analyses | `SupplementaryFigureS10` |
| S11 | Rescaling by path length and source | `SupplementaryFigureS12` |
| S12 | Grid-like unit fraction and transfer | `FigureA6` |

The folder names are builder identifiers; `figures/figure_numbering.py` is the single table that relates them to the article. [docs/FIGURE_DATA_MAP.md](docs/FIGURE_DATA_MAP.md) lists, for every figure, the plotting and experiment code, the data read and where each statistic of the text is computed.

## Re-evaluating trained models

Level 2 needs P1, P2 and P3 extracted into the repository root, PyTorch, and preferably a GPU:

```bash
# 8 x 8 and mechanism target models: rebuild, recompute the four test accuracies, compare with the stored values
python tools/reevaluation/reevaluate_models.py --family phasef --out outputs/reeval_phasef.csv
python tools/reevaluation/reevaluate_models.py --family e09a   --out outputs/reeval_e09a.csv
python tools/reevaluation/reevaluate_models.py --family e6     --out outputs/reeval_e6.csv
python tools/reevaluation/reevaluate_models.py --family e6i    --out outputs/reeval_e6i.csv
# Supplementary Fig. S11: task-state retention and the figure's source data
python tools/reevaluation/task_state_retention.py --out outputs/task_state_retention.csv
python tools/mechanism_v1/E6_retention_intervention_v1/scripts/export_figureS12_source_data.py \
    --mech-root rerun_best_validation/followups/mechanism_v1 \
    --task-state-csv outputs/task_state_retention.csv --out outputs/S11_source_data
```

`reevaluate_models.py` rebuilds every model (navigation core from P1, transformed core from the recorded seed, interface and readout from P2), evaluates the four test sets and compares the accuracies with the stored `metrics.csv` (tolerance 0.002); a deviation gives a nonzero exit status, and `--limit N` evaluates only the first N models. The families are `phasef` (Figs. 6, S5, S8), `e09a` (S7), `e6` (Figs. 7b–d, S11) and `e6i` (Figs. 7e, S11). Missing packages or a missing PyTorch installation are reported before any computation. Source-side analyses of the pretrained cores (retention κ, Fig. 7d; source measures, Fig. S10) and the navigation and topology analyses (Figs. 2, S2) run from P1 with the scripts listed in [docs/FIGURE_DATA_MAP.md](docs/FIGURE_DATA_MAP.md). The 4 × 4 target models are not released; Figs. 3–5 and S4 are reproduced at Levels 1 and 3.

## Reproducing the experiments

Level 3 retrains networks from the recorded configurations and seeds and needs a GPU (hours to days). Seeds, budgets and conditions are fixed in `plan.json`, `tools/hard8x8_phaseF_fullscale_v1/PROTOCOL.json` and the scripts of each `tools/mechanism_v1/*` experiment. Typical entry points:

```bash
export PYTHONPATH=$PWD/code
python -m multinav.stage navigation       --config <per-source navigation config>   # pretraining; P3 holds the configs used
python -m multinav.stage sequence_control --config <per-source control config>
python lsf_best_validation/run_best_validation_source.py --run-root . --output-root outputs/rerun_best_validation \
       --source-seed 6101 --family main --python "$(which python)" --device cuda      # 4 x 4 experiments of one source
python tools/hard8x8_phaseF_fullscale_v1/scripts/run_group.py --source-seed 6101 --target-seed 84001 --group learned_interface
python tools/mechanism_v1/E6_rho_scan_v1/scripts/run_rho_scan.py --source-seed 6101 --target-seed 84001 --rho 0.1
python tools/mechanism_v1/E6_retention_intervention_v1/scripts/run_intervention.py --source-seed 6101 --target-seed 84001 --arm nav_impair
```

The 8 × 8 and mechanism scripts write below `rerun_best_validation/`, the same tree that P3 populates. Run new experiments in a separate copy of the repository that contains only P1 (or point `NRT_PROJECT_ROOT` to such a copy), so that new results do not mix with the released ones. [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) gives the full sequence, the collectors and the statistics.

### Checking a single result

| Goal | Needs | Command |
|---|---|---|
| Redraw one figure | this repository | `python figures/redraw_all_figures.py --figures Figure6` |
| Re-evaluate a few 8 × 8 models | P1–P3, PyTorch | `python tools/reevaluation/reevaluate_models.py --family phasef --limit 3 --out outputs/reeval_smoke.csv` |
| Re-evaluate the rescaling models (Figs. 7e, S11) | P1–P3, PyTorch | `python tools/reevaluation/reevaluate_models.py --family e6i --out outputs/reeval_e6i.csv` |
| Retrain one 8 × 8 cell | P1, GPU | `run_group.py --source-seed 6101 --target-seed 84001 --group learned_interface` |
| Rerun the 4 × 4 experiments of one source | P1, GPU | `run_best_validation_source.py ... --source-seed 6101 --family main` |

## Citation and license

Please cite the article (see `CITATION.cff`). Code: MIT License (`LICENSE`). Models, data and author-provided figure images: CC BY 4.0 (`DATA_LICENSE.md`).
