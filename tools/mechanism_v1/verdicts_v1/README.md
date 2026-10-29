# Verdicts of the prespecified E2–E4 and E6 tests

`scripts/compute_verdicts.py` applies the prespecified tests of the source-measure (E234) and ρ-series (E6) analyses to the tables written by their collectors and Holm-corrects the secondary family. It reads the aggregates of both collectors and the E0.9a `aggregate/all_metrics.csv`; no training and no forward passes. The rules are listed in its module docstring:

- anchors: two anchors × (`id_seen_short`, `seen_long`, `heldout_long`), six paired differences whose 95 % intervals must all contain 0;
- main effect: mean per-source slope of `seen_long` on ρ < 0 with exact p < 0.05;
- primary inference: within-source slope of `seen_long` on R_ret_diff at working point Bstar; supported if the slope is positive, the permutation p < 0.05 and the interval's lower bound > 0 (read only when the anchors pass);
- secondary family S1a–S5 with Holm correction (expected-null tests S2 and S3 are judged as consistent or contrary to the expectation).

| Output | Content |
|---|---|
| `VERDICTS.md`, `VERDICTS.json` | anchors, main effect, primary inference, S1a–S5 with Holm-adjusted verdicts |
| `validation.csv` | quantities recomputed from the per-cell tables with separate code; a difference above 1e−9 from the collectors stops the run |
| `exploratory_*.csv` | exploratory tables, not used for any verdict |

```bash
python tools/mechanism_v1/verdicts_v1/scripts/compute_verdicts.py
```

Defaults read the P3 layout below `rerun_best_validation/followups/mechanism_v1/` (`--step6`, `--e6`, `--e09a-metrics`, `--out`). A completed output folder is skipped; a non-empty incomplete one stops the run. Figures: 7 and S11 (statistics).
