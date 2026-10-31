# E6 — signed-mixing ρ series

Cores are transformed as W′ = Q W Qᵀ and W_in′ = Q W_in with Q = D P (permutation P, sign matrix D). All linear invariants are preserved; only the rectification geometry changes. For every (source, target, ρ) a transfer model with learned scalar gain is trained exactly as in E0.9a (32,000 updates, same seeds and minibatch order); afterwards the downstream state profile and the source measures E2–E4 (working points A, g3, g10, B and Bstar) are measured on the same core.

ρ ∈ {0, 0.05, 0.1, 0.2, 0.35, 0.5}, plus ρ = 0.9 as a symmetry check: 10 sources × 3 targets (84001–84003) × 7 levels = 210 cells.

## Constructor

`scripts/rho_core.py`, `signed_similarity_core`. All ρ of one seed share one permutation and one unit order. For ρ ≤ 0.5 the first round(ρ H) units of that order are sign-flipped, so the flip sets are nested; for ρ > 0.5 the exact complement of the 1 − ρ set is flipped, so D_{1−ρ} = −D_ρ holds exactly. (The Phase F constructor `_signed_similarity_candidate` always flips at least one unit and cannot express ρ = 0.) Before any submission the preflight checks the algebraic anchors on the real navigation weights: no flips at ρ = 0, W unchanged and W_in negated at ρ = 1, every level equivalent to its complement, nested flip sets, unchanged Frobenius norms.

## Anchors

Pipeline checks, not hypothesis tests: ρ = 0 against navigation in E0.9a (same source, target and budget; a pure permutation only relabels units) and ρ = 0.9 against ρ = 0.1 (equivalent networks). Their paired differences on `id_seen_short`, `seen_long` and `heldout_long` must have 95 % intervals containing 0; the evaluation is in `verdicts_v1`.

## Usage

```bash
python tools/mechanism_v1/E6_rho_scan_v1/scripts/run_rho_scan.py --preflight
python tools/mechanism_v1/E6_rho_scan_v1/scripts/run_rho_scan.py --source-seed 6101 --target-seed 84001 --rho 0.1
python tools/mechanism_v1/E6_rho_scan_v1/scripts/collect_rho_scan.py
```

Recommended order: one ρ = 0 cell, then the anchor levels (ρ = 0, 0.1, 0.9), then the remaining levels.

Output: `rerun_best_validation/followups/mechanism_v1/E6_rho_scan/source_<s>/target_<t>/rho_<thousandths>/` (for example `rho_0100`) and `aggregate/RHO_SCAN_RESULT.md`. Dependencies: `extended_common.py` (E0.9a), `source_measures.py` (E234), `delta_common.py` (E5b′). P2 contains the 210 trained models (`reevaluate_models.py --family e6`). Figures: 7b, 7c; tests in `verdicts_v1` (S10).
