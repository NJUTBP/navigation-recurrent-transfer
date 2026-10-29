# C1 — autonomous retention of the ρ cores

Measures the autonomous recurrent retention of all 210 E6 ρ cores without training:

    κ = ‖ReLU(W_ρ h)‖ / ‖h‖

on the held-out navigation state set H_nav of E5b′ (4,096 states after burn-in). The states are aligned by the E6 permutation P only; the sign matrix D is the perturbation and is not applied to the state set. Consequently ρ = 0.1 and ρ = 0.9, which share the same recurrent matrix and differ only in input sign, have identical retention. This differs from the trajectory-based measures of E234, where each transformed core is driven from Q h0 and the two levels can differ although their downstream networks are equivalent.

The analysis is exploratory. Its ρ = 0.1 / ρ = 0 retention ratio defines the rescaling factor c of `E6_retention_intervention_v1`.

The scientific coverage is 210 cores (10 sources × 3 targets × 7 ρ levels); one job handles the seven levels of a (source, target) pair so that H_nav is generated once (30 jobs). The GPU is used only for the 4096-dimensional forward products.

```bash
python tools/mechanism_v1/E6_autonomous_retention_v1/scripts/run_retention.py --preflight
python tools/mechanism_v1/E6_autonomous_retention_v1/scripts/run_retention.py --source-seed 6101 --target-seed 84001
python tools/mechanism_v1/E6_autonomous_retention_v1/scripts/collect_retention.py
```

Output: `rerun_best_validation/followups/mechanism_v1/E6_autonomous_retention/` with `all_retention.csv` (210 cores), `source_level_retention.csv` (targets averaged within source), `rho_curve.csv` (retention and E6 `seen_long` dose curves), `rho0_contrasts.csv` (exploratory paired source contrasts and retention ratios) and `RETENTION_RESULT.md/json`. Figure: 7d.
