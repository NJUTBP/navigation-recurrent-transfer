# Phase F full-scale harder 8×8 result

- complete group jobs: 150/150
- source seeds: 10
- fresh target splits per source: 5
- trained conditions: 15
- competence gate passed: **True**

## Primary scalar-gain navigation contrast

- start+length ID baseline: 0.1481
- Nav scalar ID: 0.9870
- matched scalar ID: 0.9506
- Nav - matched ID: +0.0364
- 95% source bootstrap CI: [+0.0179, +0.0581]
- exact sign-flip p: 0.001953

## Orthogonal composition × horizon generalization

- id_seen_short: Δ=+0.0364, CI [+0.0179, +0.0581], exact p=0.001953, Holm=0.007812
- heldout_short: Δ=+0.0440, CI [+0.0204, +0.0717], exact p=0.001953, Holm=0.007812
- seen_long: Δ=+0.4649, CI [+0.3995, +0.5305], exact p=0.001953, Holm=0.007812
- heldout_long: Δ=+0.4901, CI [+0.4200, +0.5619], exact p=0.001953, Holm=0.007812

## Secondary ID contrasts

- fixed_gain10_nav_minus_matched: Δ=+0.0793, exact p=0.001953, Holm=0.011719
- full2x2_nav_minus_matched: Δ=+0.1369, exact p=0.001953, Holm=0.011719
- scalar_nav_minus_fresh_random: Δ=+0.7068, exact p=0.001953, Holm=0.011719
- scalar_matched_minus_fresh_random: Δ=+0.6704, exact p=0.001953, Holm=0.011719
- fullcore_nav_minus_matched: Δ=-0.0015, exact p=0.439453, Holm=0.439453
- fullcore_nav_minus_fresh_random: Δ=+0.0088, exact p=0.001953, Holm=0.011719

## Interpretation guard

- The scalar Nav-vs-matched contrast is interpreted as navigation-specific only if the competence gate passes.
- Fixed-gain arms separate input scale from trainable-gain parameterization.
- Fresh-random arms test whether the effect is specific to navigation pretraining rather than generic recurrent dynamics.
- Full-core finetuning tests whether preserving the pretrained recurrent computation is necessary for the source advantage.
- heldout_short / seen_long / heldout_long form an orthogonal composition × horizon generalization grid and are never used for checkpoint selection.
