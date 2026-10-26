# Protocol update: best-validation transfer + Schøyen-style topology

This merged codebase combines the automatic Schøyen-style torus patch with the
current primary transfer rule.

## Primary source checkpoint

All behavioural transfer in future full runs uses `navigation/best.pt`, i.e. the
minimum navigation-validation-loss checkpoint (`checkpoint_policy: best_validation`).
`best_grid_balanced.pt` remains available only for sensitivity/mechanistic comparisons.

## Topology

Topology also uses `navigation/best.pt`. The new stage is `torus_topology`, based on
automatic hidden-layer ensemble discovery and dual-metric persistent homology.
It replaces legacy `topology_export + formal_topology`.

## Retired legacy tail

The new topology output intentionally does not produce the old cohomological phase
coordinate contract. Therefore legacy `source_splits -> source_reduced -> action/path`
are not scientifically connected to the new topology and remain retired/not-evaluable
until a new reduced-model contract is defined.

## Existing multinav_paper_v1 cohort

Do not regenerate or resubmit the frozen plan with this modified config. Use the
standalone `lsf_torus_best_validation/submit_topology_chain.sh` package. It reads
existing navigation `best.pt` files and writes only new topology outputs.
