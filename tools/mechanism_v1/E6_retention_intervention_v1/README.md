# E6 rescaling intervention

Bidirectional manipulation of autonomous retention by rescaling the recurrent matrix of the E6 cores:

- `nav_impair`: the E6 ρ = 0 core with W_rec multiplied by c = 0.8032013898784912;
- `rho01_rescue`: the E6 ρ = 0.1 core with W_rec multiplied by 1/c = 1.245017765907104;
- W_in is unchanged.

c is the C1 ratio of the retention at ρ = 0.1 to that at ρ = 0 (`E6_autonomous_retention_v1`) and the only magnitude tested. For positive c and bias-free ReLU dynamics, the autonomous retention on any fixed state set scales exactly by c, so the intervention moves the ρ = 0 core to the retention of ρ = 0.1 and back.

Training and evaluation are those of E6 and E0.9a: frozen core, learned scalar gain, 32,000 updates, identical dataset manifests, run and loader seeds and evaluation splits; 10 sources × 3 targets × 2 arms = 60 training cells. The completed E6 ρ = 0 and ρ = 0.1 cells are the paired baselines and are not rerun.

`collect_intervention.py` evaluates the two co-primary paired tests on `seen_long` (impairment: rescaled ρ = 0 minus ρ = 0, predicted negative; rescue: rescaled ρ = 0.1 minus ρ = 0.1, predicted positive) with the same contrasts on `heldout_long`, and reports bidirectional support only if both co-primary tests are supported.

```bash
python tools/mechanism_v1/E6_retention_intervention_v1/scripts/run_intervention.py --preflight
python tools/mechanism_v1/E6_retention_intervention_v1/scripts/run_intervention.py --source-seed 6101 --target-seed 84001 --arm nav_impair
python tools/mechanism_v1/E6_retention_intervention_v1/scripts/collect_intervention.py
```

A GPU is needed for practical run times. Output: `rerun_best_validation/followups/mechanism_v1/E6_retention_intervention/`. P2 contains the 60 trained models (`reevaluate_models.py --family e6i`). `scripts/export_figureS12_source_data.py` writes the Supplementary Fig. S11 tables from these outputs and the task-state retention table of `tools/reevaluation/task_state_retention.py`. Figures: 7e, S11.
