# Checkpoints

Weights are not stored in git. Download `P1_pretrained_checkpoints.tar` and `P2_target_checkpoints.tar` (see `docs/DATA_PACKAGES.md`) and extract them in the repository root:

- `sources/nav_seed_61xx/artifacts/navigation/best.pt` — navigation network (all parameters: W_in, W_rec, W₀, place readout, initial-state encoder), selected at the lowest navigation validation loss.
- `sources/nav_seed_61xx/artifacts/sequence_control/best_validation.pt` — delayed-reconstruction control with its decoders.
- `rerun_best_validation/.../checkpoints/<condition>/best_validation.pt` — target models with storage mode `frozen_core_omitted_reconstruct_from_origin_and_seed`; load them with `tools/reevaluation/reevaluation_common.py` (`ModelResolver`, `rebuild_frozen_target`).
