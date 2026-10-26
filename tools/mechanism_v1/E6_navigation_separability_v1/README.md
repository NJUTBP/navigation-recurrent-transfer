# E6 navigation separability control

This implements the prespecified E6 add-on: for every frozen rho core, refit only the navigation interface (`init_encoder` and `place_head`) and measure navigation decoding competence.

The recurrent and velocity-input matrices are never trained. The interface starts from the source checkpoint transformed into the rho-core coordinates. A rho=0 cell must therefore reproduce the original source model before refitting.

Protocol: 10 sources x 3 E6 target/core seeds x 7 rho levels = 210 cells. Refit uses 2048 fixed navigation trajectories, 512 held-out validation trajectories, batch 128, 100 epochs, source navigation learning rate, validation every 5 epochs, best validation CE retained.

Primary navigation metric: fraction of achievable place-code CE reduction captured (`frac_loss_captured`). Secondary metric: decoded position error / arena size. The collector compares rho=0.05 and rho=0.1 navigation loss to the already-computed E6 seen_long transfer loss.
