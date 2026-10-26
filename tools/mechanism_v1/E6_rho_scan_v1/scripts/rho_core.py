#!/usr/bin/env python3
"""Signed-similarity cores for the E6 rho scan (plan v2/v3).

Q = D P with P a permutation and D = diag(+-1):  W' = Q W Q^T, W_in' = Q W_in.
All linear invariants are preserved; only the ReLU active-set geometry changes.

The existing analysis/recurrent_factorial._signed_similarity_candidate flips
max(1, round(rho H)) units, so it cannot express rho = 0, and it draws each
flip set independently.  As the plan requires, it is left untouched and this
wrapper is used instead:
  - one permutation P and one unit order per seed, shared by every rho;
  - rho <= 0.5 flips the first round(rho H) units of that order, so the flip
    sets are nested along the scan;
  - rho > 0.5 flips the exact complement of the set used for 1 - rho, so
    D_{1-rho} = -D_rho holds exactly and the curve must be symmetric.

Provable anchors, checked by anchor_report():
  rho = 0   P ReLU(W h + W_in u) = ReLU(W' P h + W_in' u): relabelled units.
  rho = 1   W' equals the rho = 0 matrix and W_in' flips sign.
  rho, 1-rho   identical W', opposite W_in'.
"""
from __future__ import annotations

import numpy as np

RHO_GRID = (0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.9)
SCAN_RHOS = (0.0, 0.05, 0.1, 0.2, 0.35, 0.5)

def flip_units(hidden_dim: int, rho: float, seed: int):
    rng = np.random.default_rng(int(seed))
    permutation = rng.permutation(hidden_dim)
    order = rng.permutation(hidden_dim)
    if rho <= 0.5:
        flipped = order[:int(round(rho * hidden_dim))]
    else:
        flipped = order[int(round((1.0 - rho) * hidden_dim)):]
    return permutation, np.sort(flipped)

def _core_from_flips(nav_hh, nav_ih, permutation, flipped):
    import torch
    hidden_dim = int(nav_hh.shape[0])
    signs = torch.ones(hidden_dim, dtype=nav_hh.dtype)
    signs[torch.as_tensor(flipped, dtype=torch.long)] = -1.0
    index = torch.as_tensor(permutation, dtype=torch.long)
    hh = nav_hh.index_select(0, index).index_select(1, index)
    hh = signs[:, None] * hh * signs[None, :]
    ih = signs[:, None] * nav_ih.index_select(0, index)
    return hh.contiguous(), ih.contiguous()

def signed_similarity_core(nav_hh, nav_ih, *, seed: int, rho: float):
    hidden_dim = int(nav_hh.shape[0])
    permutation, flipped = flip_units(hidden_dim, rho, seed)
    hh, ih = _core_from_flips(nav_hh, nav_ih, permutation, flipped)
    return hh, ih, {
        'rho': float(rho), 'flip_count': int(len(flipped)), 'realized_rho': float(len(flipped) / hidden_dim),
        'core_seed': int(seed)}

def anchor_report(nav_hh, nav_ih, *, seed: int, probes: int = 64) -> dict:
    import torch
    report = {}
    hh0, ih0, meta0 = signed_similarity_core(nav_hh, nav_ih, seed=seed, rho=0.0)
    hh1, ih1, _ = signed_similarity_core(nav_hh, nav_ih, seed=seed, rho=1.0)
    report['rho0_flip_count'] = meta0['flip_count']
    report['rho1_same_recurrent'] = bool(torch.equal(hh1, hh0))
    report['rho1_negated_input'] = bool(torch.equal(ih1, -ih0))
    permutation, _ = flip_units(int(nav_hh.shape[0]), 0.0, seed)
    index = torch.as_tensor(permutation, dtype=torch.long)
    generator = torch.Generator().manual_seed(int(seed) % (2 ** 31))
    h = torch.rand(probes, nav_hh.shape[0], generator=generator, dtype=nav_hh.dtype)
    u = torch.randn(probes, nav_ih.shape[1], generator=generator, dtype=nav_hh.dtype) * 0.02
    reference = torch.relu(h @ nav_hh.t() + u @ nav_ih.t()).index_select(1, index)
    relabelled = torch.relu(h.index_select(1, index) @ hh0.t() + u @ ih0.t())
    report['rho0_dynamics_relative_deviation'] = float((reference - relabelled).abs().max()
                                                       / reference.abs().max().clamp_min(1e-12))
    pairs, branch = {}, {}
    hidden_dim = int(nav_hh.shape[0])
    for rho in SCAN_RHOS[1:]:
        low_hh, low_ih, _ = signed_similarity_core(nav_hh, nav_ih, seed=seed, rho=rho)
        perm, flipped = flip_units(hidden_dim, rho, seed)
        comp_hh, comp_ih = _core_from_flips(nav_hh, nav_ih, perm, np.setdiff1d(np.arange(hidden_dim), flipped))
        pairs[f'{rho:g}'] = bool(torch.equal(low_hh, comp_hh) and torch.equal(low_ih, -comp_ih))
        if rho < 0.5:
            high_hh, high_ih, _ = signed_similarity_core(nav_hh, nav_ih, seed=seed, rho=1.0 - rho)
            branch[f'{1.0 - rho:g}'] = bool(torch.equal(high_hh, comp_hh) and torch.equal(high_ih, comp_ih))
    report['complement_symmetry'] = pairs
    report['upper_branch_is_complement'] = branch
    sets = [set(flip_units(int(nav_hh.shape[0]), rho, seed)[1].tolist()) for rho in SCAN_RHOS]
    report['flip_sets_nested'] = all(a <= b for a, b in zip(sets, sets[1:]))
    frob = float(nav_hh.double().norm())
    report['frobenius_preserved'] = all(
        abs(float(signed_similarity_core(nav_hh, nav_ih, seed=seed, rho=r)[0].double().norm()) - frob) <= 1e-6 * frob
        for r in RHO_GRID)
    report['all_pass'] = bool(report['rho0_flip_count'] == 0 and report['rho1_same_recurrent']
                              and report['rho1_negated_input'] and report['rho0_dynamics_relative_deviation'] <= 1e-5
                              and all(pairs.values()) and all(branch.values())
                              and report['flip_sets_nested'] and report['frobenius_preserved'])
    return report
