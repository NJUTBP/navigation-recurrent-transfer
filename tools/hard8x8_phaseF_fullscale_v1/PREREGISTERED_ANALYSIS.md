# Phase F predeclared analysis

## Primary endpoint

`id_seen_short` accuracy at the validation-selected checkpoint.

Primary paired contrast:

`navigation_trainable_scalar_gain - matched_random_trainable_scalar_gain`

For each navigation source, average the five fresh target-split contrasts first. The inferential n is therefore 10 navigation source seeds. Use a two-sided exhaustive exact sign-flip test and a 20,000-resample source bootstrap 95% CI.

## Competence gate

Interpret the primary contrast as a navigation-specific transfer advantage only if:

1. Nav scalar ID > same-split start+length baseline;
2. matched scalar ID > same-split start+length baseline;
3. both scalar arms show positive zero-relation true-target logp drop;
4. both scalar arms show positive counterfactual-target logp gain.

## Orthogonal generalization family

The same primary contrast is reported on four locked test quadrants:

- seen composition × short horizon (`id_seen_short`);
- held-out composition × short horizon (`heldout_short`);
- seen composition × long horizon (`seen_long`);
- held-out composition × long horizon (`heldout_long`).

Report exact sign-flip tests and Holm-adjust across these four endpoint tests. ID remains the primary endpoint.

## Predeclared secondary contrasts on ID

- fixed gain 10: Nav - matched;
- full 2×2 adapter: Nav - matched;
- scalar: Nav - fresh random;
- scalar: matched - fresh random;
- full recurrent fine-tuning: Nav - matched;
- full recurrent fine-tuning: Nav - fresh random.

Holm-adjust within this six-contrast secondary family.

## Mechanistic interpretation

- gain=1 / gain=3 / gain=10 frozen arms test input scale without trainable gain dynamics;
- trainable scalar tests one-dimensional adaptive scale;
- trainable 2×2 tests whether additional rotation/anisotropy remains useful after 8000 updates;
- fresh-random scalar/2×2 controls test whether low-dimensional interface learning is sufficient for a generic random reservoir;
- full-core fine-tuning tests whether preserving the pretrained recurrent computation is necessary for the source advantage.

No composition/long test is used for checkpoint selection.
