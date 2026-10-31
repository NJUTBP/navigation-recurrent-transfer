# Mechanistic analyses of the 8 × 8 transfer advantage

Follow-up analyses of the extended 8 × 8 task (`tools/hard8x8_phaseF_fullscale_v1`, "Phase F"). Folder names keep the experiment identifiers under which the analyses were run; the same identifiers name their output folders below `rerun_best_validation/followups/mechanism_v1/` (package P3).

| Folder | Analysis | Trains models | Figures |
|---|---|---|---|
| `E0_9a_extended_budget_v1` | Four Phase F conditions with 32,000 instead of 8,000 updates; learned-gain trajectories | yes (60 jobs) | S7 |
| `E5c_gain_sweep_v1` | Accuracy as a function of a fixed relation-input gain g = 2–256 | yes (320 cells) | S7 |
| `E0_1_nonrecurrent_baselines_v1` | Non-recurrent algorithmic baselines on the Phase F splits | yes (10 cells) | S7 |
| `E0_67_forward_diagnostics_v1` | Recurrence ablation and order shuffling (E0.6), error displacement (E0.7), state diagnostics and δ (E0.4) | no (forward passes) | 7a, S8 |
| `E5b_delta_prediction_v1` | Gain ratio predicted by the one-step rectified displacement δ (E5b′) | no | S8 |
| `E234_source_measures_v1` | Source-side measures E2–E4 of navigation and matched cores | no | S10 |
| `E6_rho_scan_v1` | Signed-mixing ρ series of cores: transfer at seven ρ levels and source measures | yes (210 cells) | 7b, 7c, S10 |
| `E6_autonomous_retention_v1` | Autonomous recurrent retention κ of all ρ cores (C1) | no | 7d |
| `E6_retention_intervention_v1` | Recurrent rescaling W_rec → c W_rec (impairment and rescue) | yes (60 cells) | 7e, S11 |
| `E6_navigation_separability_v1` | Refit of the navigation interface of every ρ core | interface refits (210 cells) | S9 |
| `E1_interface_harvest_v1` | Magnitude and shape of the learned relation interfaces of the 4 × 4 symbolic adapters | no | – |
| `verdicts_v1` | Evaluation of the prespecified E2–E4 and E6 tests with Holm correction | no | 7, S10 |

Dependencies: E0.9a provides the g = 1 anchor of E5c, the ρ = 0 anchor of E6 and the learned gains of E234; E0_67 provides δ for E234 (working point Bstar); E5c's post hoc analysis provides the observations of E5b′; E6 provides the cores of C1, the separability control and the intervention, whose rescaling factor c is the retention ratio measured by C1.

## Conventions

- Project root: `NRT_PROJECT_ROOT`, otherwise the repository root. Outputs: `rerun_best_validation/followups/mechanism_v1/<experiment>/`, the layout of P3; run new experiments in a separate working copy (see `docs/REPRODUCIBILITY.md`).
- Phase F is reused, not copied: its scripts are imported from `tools/hard8x8_phaseF_fullscale_v1/scripts` (the import location is checked), the protocol is derived at run time from its `PROTOCOL.json`, and `training_examples` (a component of run_seed and loader_seed) is kept, so initialization, data and minibatch order equal those of Phase F.
- Frozen-core checkpoints omit the recurrent core; it is rebuilt from the navigation checkpoint and the recorded seeds, and a rebuilt model must reproduce the stored test accuracies within 2e-3 (Gate R).
- Each `run_*.py` handles one cell (`--source-seed`, `--target-seed` and experiment-specific arguments) and offers `--preflight`; `collect_*.py` writes the cohort tables to `aggregate/`. Cluster job scripts are not included; one job array index corresponded to one cell.
- Statistics: the navigation source is the unit, targets are averaged within source, tests are two-sided exact sign-flip tests, intervals use 20,000 source-bootstrap resamples (seed 982451653), and Holm correction is applied within families.
- The prespecified decision rules are stated in the module docstrings of the collectors (`collect_extended.py`, `collect_gain_sweep.py`, `collect_delta.py`, `analyze_interface_magnitude.py`) and in `verdicts_v1/scripts/compute_verdicts.py`.
- `analyze_*_posthoc.py` are post hoc analyses; they write only to `aggregate_posthoc/` and do not replace the prespecified collectors.
