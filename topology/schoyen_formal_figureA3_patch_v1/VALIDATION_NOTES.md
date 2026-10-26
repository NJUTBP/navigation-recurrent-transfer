# Patch validation notes

The package was checked locally against the current merged code snapshot.

## Static checks

- all modified/new Python files compile;
- all new modules import from the merged project tree;
- installer/submit/LSF shell syntax passes `bash -n`;
- installer was run against a copy of the merged `code/` tree and the installed modules imported successfully.

## Interface tests

- `schoyen_euclidean_formal_topology_v1` contract generated correctly from a synthetic selected-candidate PH summary;
- `upgrade_contracts.load_source_reduced_artifact` accepts `source_reduced_schoyen_v1` while retaining `source_reduced_v3` support;
- explicit `phase_encoder=` injection through `SourceReducedModelV3.fit` completes without the old cohomological landmark arguments.

## Numerical smoke tests

A synthetic population with three periodic wavevector families was used to verify:

- family-aware Fourier wavevector selection recovers two independent physical periodic axes;
- phase-chart circular RMSE is low on the synthetic known-phase population;
- reduced-model fitting and held-out source quality-gate evaluation complete;
- the CLI `analyze_source_reduced_schoyen` emits its pickle, summary, per-seed and per-trajectory CSV artifacts.

A coordinate-axis bug discovered during testing was fixed before packaging: `np.histogram2d(x,y)` stores x on array axis 0, so the new implementation now uses an explicit physical `(kx,ky)` convention rather than inheriting the older image-index convention.

The long-rollout exporter is also wrapped in `torch.no_grad()` to avoid retaining unnecessary 500-step autograd graphs for the 4096-unit navigation RNN.
