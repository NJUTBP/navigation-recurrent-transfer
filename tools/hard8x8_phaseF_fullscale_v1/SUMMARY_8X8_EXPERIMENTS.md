# Harder 8×8 project: complete experimental summary before Phase F

This summary separates historical results, read-only diagnosis, development experiments, and confirmatory/mechanistic follow-ups. It is derived from the completed result bundles in this project; Phase E0 was proposed but not run and is superseded by Phase F.

## 1. Historical harder 8×8 benchmark (original 1000-example protocol)

Ten navigation source seeds, with target repeats averaged within source.

| condition | ID | composition | long |
|---|---:|---:|---:|
| navigation frozen | 0.0947 | 0.0796 | 0.0679 |
| matched-random frozen | 0.0952 | 0.0758 | 0.0690 |
| navigation trainable | 0.1272 | 0.1115 | 0.1032 |
| matched-random trainable | 0.1268 | 0.1086 | 0.1019 |

Initial observation: trainability improved absolute performance, but no stable navigation-specific advantage appeared on ID/long; the small composition difference was not accompanied by a competent frozen task assay.

## 2. Audit / Phase A0: provenance and implementation diagnosis

Confirmed issues:

- Historical sequential split construction (`train -> validation -> ID`) caused strong path-length distribution drift and finite-support depletion of short trajectories.
- Historical `composition` and `long` tests were not an orthogonal composition × horizon crossing.
- `1/64` was not a sufficient chance benchmark: start identity + path length alone could reach roughly 0.11–0.14 on 8×8 ID-like splits.
- The historical condition loop reused a DataLoader generator, so nominally paired conditions did not see exactly identical minibatch orders.
- The matched-random control is a Nav-derived signed-similarity control, not an independent random RNN.

Ruled out / not found:

- endpoint addition label error;
- padded final state accidentally used instead of true endpoint state;
- Nav/random plotting label swap.

Operational Phase A0 inventory initially over-collected target seeds from unrelated downstream stages; after artifact-path correction, the real hard benchmark was confirmed as 10 sources × 3 hard target repeats = 30 complete hard runs.

## 3. Phase A1: exact-historical read-only reconstruction and checkpoint replay

All 10/10 sources passed exact short-path fingerprint recovery and checkpoint replay gates.

### Historical shortcut baselines

- validation start+length: 0.1170
- ID start+length: 0.1121
- composition start+length: 0.0963

### Full train/validation competence at budget 1000

| condition | train acc | validation acc |
|---|---:|---:|
| navigation frozen | 0.4598 | 0.1055 |
| matched frozen | 0.4069 | 0.1037 |
| navigation trainable | 0.4954 | 0.1407 |
| matched trainable | 0.4724 | 0.1406 |

Key result: frozen models memorized substantial training structure but generalized below the exact start+length shortcut. Therefore the original frozen Nav≈matched null was not a clean assay of a learned relation computation.

### Relation dependence

On historical ID, zeroing/swapping relations changed log probabilities for all four conditions. Example zero-relation true-target logp drops:

- Nav frozen: 0.151
- matched frozen: 0.128
- Nav trainable: 0.140
- matched trainable: 0.091

Thus the models were not completely ignoring relation inputs; the failure was poor generalizable endpoint mapping.

### Matched-control real-state audit

After correct signed-basis alignment, Nav and matched one-step dynamics still differed materially:

- mean cosine ≈ 0.87
- one-step RMS differences ≈ 0.12–0.14

The historical matched control therefore was not simply “almost identical to Nav.”

## 4. Phase B1: locked-split bridge, 1000 vs 4000 examples

Development sources: 6101, 6105, 6110. Two locked target splits. Validation/test fixed across budgets; 1000 training examples nested in 4000.

### Budget 1000

| condition | validation | ID |
|---|---:|---:|
| Nav frozen | 0.1053 | 0.0887 |
| matched frozen | 0.1068 | 0.0859 |
| start+length baseline | 0.1140 | 0.1060 |

Clean splitting alone did not rescue frozen transfer.

### Budget 4000

| condition | validation | ID |
|---|---:|---:|
| Nav frozen | 0.1413 | 0.1237 |
| matched frozen | 0.1373 | 0.1178 |
| Nav fully trainable | 0.2813 | 0.2709 |
| matched fully trainable | 0.2877 | 0.2777 |
| start+length baseline | 0.1465 | 0.1362 |

Interpretation:

- more unique trajectory support helped materially;
- the target task was clearly learnable when the recurrent core could adapt;
- frozen cores still remained below the shortcut, so the small frozen Nav-minus-matched difference (+0.0059 ID) was not yet interpretable as a competent navigation advantage.

## 5. Phase B2: frozen core + trainable 2×2 relation-input adapter

Same development sources/splits, 4000 examples, 2000 updates. Only four adapter parameters were added; W_in and W_hh remained frozen.

| condition | validation | ID |
|---|---:|---:|
| Nav identity-interface control | 0.1413 | 0.1238 |
| matched identity-interface control | 0.1373 | 0.1178 |
| Nav + trainable 2×2 adapter | 0.2478 | 0.2356 |
| matched + trainable 2×2 adapter | 0.1825 | 0.1722 |
| start+length baseline | 0.1465 | 0.1362 |

Development source-level adapted Nav-minus-matched ID differences:

- 6101: +0.0225
- 6105: +0.0553
- 6110: +0.1125
- mean: +0.0634

Both adapted arms became competent and relation-dependent; the Nav rescue was systematically larger.

Adapter singular values were roughly 3× input gain, motivating the hypothesis that the main interface failure involved operating scale.

## 6. Phase C: predeclared confirmatory 2×2-adapter experiment

Fresh targets: 83001–83003. Primary inference excluded the three development sources and used seven untouched source seeds.

Competence gate passed:

- start+length ID baseline: 0.1340
- held-out-source Nav adapted ID: 0.2274
- held-out-source matched adapted ID: 0.1604
- relation zeroing and counterfactual-swap gates both positive.

Primary held-out-source contrast:

- mean Nav - matched ID: **+0.0670**
- 95% source-bootstrap CI: **[+0.0515, +0.0839]**
- two-sided exact sign-flip p: **0.015625**
- positive direction: **7/7 held-out sources**

Per-length held-out mean Nav-minus-matched differences were positive for every length 1–5:

- L1 +0.0260
- L2 +0.0547
- L3 +0.0746
- L4 +0.0548
- L5 +0.0811

This is the first independent confirmatory result showing a stable harder-8×8 navigation-specific advantage after minimal interface adaptation.

## 7. Phase D: one-parameter positive scalar-gain mechanism experiment

Same 10 sources and Phase-C target splits, frozen recurrent core, one learned parameter `gain = exp(log_gain)`, 2000 updates.

- Nav scalar ID: **0.5198**
- matched scalar ID: **0.2824**
- start+length baseline: 0.1340
- mean source-level Nav-minus-matched: **+0.2374**
- 95% CI: **[+0.1859, +0.2776]**
- exact sign-flip p: **0.001953**
- positive source means: **10/10**
- learned gain Nav: ~11.28
- learned gain matched: ~9.10

However, all 60 condition checkpoints (30 source-target blocks × 2 conditions) selected the maximum update 2000. Mean validation accuracy was still climbing sharply from update 1500 to 2000:

- Nav: 0.3195 -> 0.5147
- matched: 0.1924 -> 0.2858

Therefore Phase D strongly implicates relation-input operating scale, but it does **not** identify an optimum gain and cannot by itself prove that the scalar parameterization is intrinsically superior to the full 2×2 adapter.

## 8. Current synthesis

The accumulated evidence supports the following ordering of causes for the original unstable harder-8×8 result:

1. **Primary bottleneck: frozen-core input/interface operating-scale mismatch.** A tiny interface adaptation turns an incompetent frozen assay into a competent one and reveals a stable Nav advantage.
2. **Important secondary factor: sample support / 64-way target sample complexity.** 1000 -> 4000 unique trajectories materially improves generalization.
3. **Real but insufficient historical artifact: sequential split drift and coupled composition/long definitions.** Fixing them alone does not rescue frozen transfer.
4. **Not primary explanations:** padding/label bugs, complete relation neglect, mixed 1–5 horizon alone, or a matched control that is nearly identical to Nav.

## 9. Why Phase F is a single large-scale experiment

Phase F stops the sequence of small diagnostics and runs one integrated protocol that simultaneously resolves:

- convergence (8000 gradient updates);
- sample support (8000 unique training trajectories);
- fixed input scale (gain 1 / 3 / 10);
- trainable scalar gain;
- trainable full 2×2 adapter;
- generic fresh-random frozen-core controls;
- full recurrent fine-tuning controls;
- five fresh locked target splits;
- all 10 navigation sources;
- a true orthogonal composition × horizon test grid: seen-short, heldout-short, seen-long, heldout-long.

Total: **750 training cells** (10 sources × 5 target splits × 15 conditions), each trained for a fixed **8000 gradient updates**, plus large locked test batteries.
