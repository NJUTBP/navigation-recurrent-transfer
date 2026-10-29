# Model, task and experiment code

Both packages are imported with `code/` on the Python path (`export PYTHONPATH=$PWD/code`).

## gpm_cmap

The scientific implementation used by all experiments.

| Subpackage | Content |
|---|---|
| `models/` | The recurrent navigation network (`grid_rnn.py`), the transfer models with frozen or trainable cores and relation interfaces (`abstract_pathint.py`, `adapter_wrap.py`), the delayed-reconstruction model (`sequence_control_model.py`), non-recurrent baselines and symbolic controls (`baselines.py`, `symbolic_controls.py`) and further model components |
| `data/` | Open-field navigation environment and place cells (`nav_env/`), relational lattice tasks and trial generation (`graph_tasks/`, trials in `pathint_episodes.py`) |
| `tasks/` | Training and evaluation of navigation (`navigation.py`) and of the relational endpoint task (`abstract_pathint.py`) |
| `experiments/` | Entry points: navigation pretraining (`exp1_nav_pretrain.py`), delayed-reconstruction pretraining (`pretrain_sequence_control.py`), 4 × 4 transfer (`exp5_abstract_pathint.py`), adapter (`exp7_adapter_rescue.py`), arbitrary symbols (`exp8_symbolic_adapter.py`), harder 8 × 8 benchmark and recurrent factorial design (`exp5_hard_benchmark.py`, `exp5_recurrent_factorial.py`), topology and reduced-dynamics analyses (`analyze_schoyen_torus.py`, `aggregate_schoyen_torus.py`, `derive_schoyen_formal_topology.py`, `export_schoyen_source_splits.py`, `analyze_source_reduced_schoyen.py`, `analyze_path_invariance.py`) |
| `analysis/` | Grid scores, rate maps, torus and reduced-model analyses |
| `configs/` | Experiment configurations |
| `utils/` | Configuration, checkpointing, seeding, metrics, runtime settings |

## multinav

Orchestration of the 4 × 4 cohort: `plan.py` freezes a run (copies the code, fixes seeds and writes `plan.json`), `stage.py` runs one experiment stage (`python -m multinav.stage <kind> --config <yaml>`), `run_source.py` and `submit.py` schedule the stages of a source, and `collect.py`, `aggregate.py` and `stats.py` collect the results and compute the source-level statistics. `config.yaml` is the cohort configuration.

