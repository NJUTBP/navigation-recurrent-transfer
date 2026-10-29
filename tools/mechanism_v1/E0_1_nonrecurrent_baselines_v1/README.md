# E0.1 — non-recurrent baselines

Two models without a recurrent core, trained on the five Phase F splits. They are algorithmic references, not recurrent-memory controls.

| Baseline | Structure | Parameters |
|---|---|---|
| `action_count_linear` | start one-hot and counts of the four relations, linear classifier | 4,416 |
| `linear_accumulator_2d` | a 2-D start vector per node, relation vectors added step by step, linear readout | 328 |

The task is a commutative group action: the endpoint is the start plus the sum of the relations, so the start and the relation counts determine the answer. Both function classes contain the exact solution: with lattice coordinates p_j, the logit 2 p_j·(p_s + D c) − |p_j|² is maximal at the target. The preflight loads these exact weights and requires accuracy 1.0 on all splits, which also verifies the relation encoding; errors after training are therefore learning or extrapolation failures, not capacity limits.

Pairing with Phase F: splits are rebuilt with `build_fullscale_split` and checked against the manifest hashes; `run_seed` and `loader_seed` are those of the recurrent conditions (same initialization stream and minibatch order); optimizer, learning rate, gradient clipping and validation-only model selection follow Phase F. The conversion of displacements to one-hot relations is a vectorized `velocity_action_labels`, compared with it trajectory by trajectory in the preflight. Without a navigation core there is no source dimension: one cell is (target, budget), 5 × 2 = 10 cells (8,000 and 32,000 updates).

```bash
python tools/mechanism_v1/E0_1_nonrecurrent_baselines_v1/scripts/run_baselines.py --preflight
python tools/mechanism_v1/E0_1_nonrecurrent_baselines_v1/scripts/run_baselines.py --target-seed 84001 --updates 8000
python tools/mechanism_v1/E0_1_nonrecurrent_baselines_v1/scripts/collect_baselines.py
```

Output: `rerun_best_validation/followups/mechanism_v1/E0_1_nonrecurrent_baselines/` (`aggregate/BASELINE_RESULT.md`). Figure: S7 (additive horizon).
