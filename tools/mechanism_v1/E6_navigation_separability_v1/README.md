# E6 navigation-separability control

Prespecified control of E6: for every frozen ρ core, only the navigation interface (`init_encoder` and `place_head`) is refitted and navigation decoding competence is measured. The question is whether transfer deteriorates faster than source navigation competence as ρ increases.

- The recurrent and velocity-input matrices are never trained. The interface starts from the source checkpoint transformed into the coordinates of the ρ core, so a ρ = 0 cell must reproduce the original navigation model before refitting (a failure invalidates the cell).
- 10 sources × 3 E6 target/core seeds × 7 ρ levels = 210 cells. The refit uses 2,048 fixed navigation trajectories per source (identical across ρ and targets), 512 held-out validation trajectories with the source validation seed, batch 128, 100 epochs, the navigation learning rate of the source checkpoint, validation every 5 epochs, and keeps the best validation cross-entropy.
- Primary navigation measure: `frac_loss_captured`, the fraction of the achievable reduction from the uniform place-code cross-entropy to the target entropy floor. Secondary: decoded position error divided by arena size. Zero-shot and refit values are both kept; inference uses the refit values.

`collect_separability.py` compares, at ρ = 0.05 and ρ = 0.1, the drop in `seen_long` transfer from ρ = 0 with the drop in navigation competence (excess transfer drop = transfer drop − navigation drop, averaged over targets within source), with exact sign-flip tests, source-bootstrap intervals and Holm correction over the two levels, and reports the convergence of the refits.

```bash
python tools/mechanism_v1/E6_navigation_separability_v1/scripts/run_separability.py --preflight
python tools/mechanism_v1/E6_navigation_separability_v1/scripts/run_separability.py --source-seed 6101 --target-seed 84001 --rho 0.1
python tools/mechanism_v1/E6_navigation_separability_v1/scripts/collect_separability.py
```

Output: `rerun_best_validation/followups/mechanism_v1/E6_navigation_separability/`. Figure: S9.
