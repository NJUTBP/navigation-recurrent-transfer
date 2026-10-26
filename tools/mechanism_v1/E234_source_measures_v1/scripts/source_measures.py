#!/usr/bin/env python3
"""Source-side measures E2, E3 and E4 (plan v1/v2 definitions, v3/v4 working points).

Core step, no bias:  F(h, u) = ReLU(W_hh h + g W_in u).

E2  one-step geometry on centred displacements
        D_a  = F(h, u_a) - F(h, 0),   Dc_a = D_a - mean_a D_a
        S_opp  = 1/2 [cos(Dc_+x, -Dc_-x) + cos(Dc_+y, -Dc_-y)]       mean over h
        S_orth = 1 - |dx.dy| / (|dx||dy|),  dx = Dc_+x - Dc_-x, dy = Dc_+y - Dc_-y
        S_cons = E_{h,a} cos(Dc_a(h), mean_h Dc_a(h))
        r_eff  = participation ratio of the singular values of the stacked Dc
    Null: the same four numbers for four random directions of the same
    magnitude; reported as (observed - null mean) / null sd.

E3  composition, reference F^2(h, 0) as required
        C_comm = E_{h, a<b} |F(F(h,u_a),u_b) - F(F(h,u_b),u_a)| / s_ab,
                 s_ab = 1/2(|F(F(h,u_a),u_b) - F^2(h,0)| + |F(F(h,u_b),u_a) - F^2(h,0)|)
        C_inv  = E_{h,a} |F(F(h,u_a),u_-a) - F^2(h,0)| / |F(F(h,u_a),0) - F^2(h,0)|
    The plan writes C_inv for (+x,-x) with an unspecified scale; here it is
    averaged over the four ordered inverse pairs and scaled by how far u_a
    alone moves the state over the same two steps.

E4  horizon-resolved additive retention
    Start states from H_nav, legal 8x8 relation sequences from the Phase F
    pools, y = sum of steps in lattice units.  Ridge y ~ A h_L + c is fitted on
    L <= 5 only (penalty chosen on a held-out fifth of the L <= 5 rows), then
    R^2(L) is measured per L on separate start states and sequences.
        R_ret = R^2(12) / R^2(5)
    Auxiliary: O(L) order dependence against F^L(h0, 0); signal and norm by L;
    active fraction by L; c(t, 12), the effect on h_12 of zeroing step t.

    Companion readout R_ret_diff, added after testing the literal definition:
    y is also decoded from h_L - F^L(h0, 0).  Decoding from h_L alone mixes the
    varying start state into the regression; for an exact integrator start
    position and displacement share one subspace, R^2 then rises with L as the
    displacement variance grows, and an exact integrator scores R_ret > 1.
    Subtracting the start state's own zero-input evolution removes that term;
    an exact integrator then scores R^2 = 1 at every L.
"""
from __future__ import annotations
import os

import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get('NRT_PROJECT_ROOT', str(Path(__file__).resolve().parents[4])))
E5B_SCRIPTS = PROJECT_ROOT / 'tools' / 'mechanism_v1' / 'E5b_delta_prediction_v1' / 'scripts'
if str(E5B_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(E5B_SCRIPTS))
import delta_common  # noqa: E402

N_STATES = 2000
NAV_TRAJECTORIES = 512
NAV_BURN_IN = 2
NULL_DRAWS = 20
MAX_L = 12
FIT_MAX_L = 5
LAMBDA_EXPONENTS = tuple(range(-6, 2))
DELTA_CURVE = tuple(2.0 ** (k / 4.0) for k in range(-28, 37))
CHUNK = 500

def step(w_hh, h, drive):
    import torch
    return torch.relu(h @ w_hh.t() + drive)

def transform_start(h0, transform):
    """Q h0 for Q = D P, with W' = Q W Q^T built as W[perm][:, perm] times signs."""
    if transform is None:
        return h0
    permutation, flipped = transform
    import torch
    signs = torch.ones(h0.shape[-1], dtype=h0.dtype, device=h0.device)
    signs[torch.as_tensor(flipped, dtype=torch.long, device=h0.device)] = -1.0
    return signs * h0.index_select(-1, torch.as_tensor(permutation, dtype=torch.long, device=h0.device))

def matched_transform(nav_hh, nav_ih, matched_hh, matched_ih, diagnostics):
    """Rebuild P and D of the Phase F matched core from its recorded seed and
    fraction, exactly as _signed_similarity_candidate draws them, and require
    that they reproduce the matched weights bit for bit."""
    import numpy as np
    import torch
    construction = diagnostics['construction']
    hidden_dim = int(nav_hh.shape[0])
    rng = np.random.default_rng(int(construction['seed']))
    permutation = rng.permutation(hidden_dim)
    count = max(1, min(hidden_dim, int(round(float(construction['requested_sign_flip_fraction']) * hidden_dim))))
    flipped = rng.choice(hidden_dim, size=count, replace=False)
    index = torch.as_tensor(permutation, dtype=torch.long)
    signs = torch.ones(hidden_dim, dtype=nav_hh.dtype)
    signs[torch.as_tensor(flipped, dtype=torch.long)] = -1.0
    hh = signs[:, None] * nav_hh.index_select(0, index).index_select(1, index) * signs[None, :]
    ih = signs[:, None] * nav_ih.index_select(0, index)
    if not (torch.equal(hh, matched_hh) and torch.equal(ih, matched_ih)):
        raise RuntimeError('matched core transform could not be reconstructed exactly')
    return permutation, flipped, count / hidden_dim

def navigation_states(nav_model, cores, velocities, init_place, n_states, seed, transforms=None):
    """Every core runs the same navigation inputs.  A transformed core Q W Q^T
    starts from Q h0, the linearly equivalent start, so the first
    pre-activation equals the navigation one up to Q and only rectification
    differs; rho = 0 then reproduces the navigation trajectory relabelled.
    The same (trajectory, step) positions after the burn-in are kept."""
    import torch
    transforms = transforms or {}
    steps = velocities.shape[1]
    generator = torch.Generator().manual_seed(int(seed))
    positions = torch.randperm(velocities.shape[0] * (steps - NAV_BURN_IN), generator=generator)[:n_states]
    with torch.no_grad():
        h0 = nav_model.init_encoder(init_place)
        if nav_model.nonnegative and nav_model.init_relu:
            h0 = torch.relu(h0)
        reference = nav_model(velocities, init_place)['hidden_seq']
    states = {}
    for name, (w_hh, w_in) in cores.items():
        h = transform_start(h0, transforms.get(name))
        rows = []
        with torch.no_grad():
            for t in range(steps):
                h = step(w_hh, h, velocities[:, t] @ w_in.t())
                rows.append(h)
        hidden = torch.stack(rows, dim=1)
        if name == 'navigation':
            deviation = float((hidden - reference).abs().max() / reference.abs().max().clamp_min(1e-12))
            if deviation > 1e-4:
                raise RuntimeError(f'manual rollout differs from the navigation model: {deviation:.2e}')
        flat = hidden[:, NAV_BURN_IN:, :].reshape(-1, hidden.shape[-1])
        states[name] = flat[positions.to(flat.device)].contiguous()
    return states

def _geometry_numbers(w_hh, w_in, states, directions, gain):
    import torch
    with torch.no_grad():
        z = states @ w_hh.t()
        base = torch.relu(z)
        drives = (directions * gain) @ w_in.t()
        d = torch.stack([torch.relu(z + drive) - base for drive in drives], dim=1)
        dc = d - d.mean(dim=1, keepdim=True)
        cos = torch.nn.functional.cosine_similarity
        s_opp = 0.5 * (cos(dc[:, 0], -dc[:, 1], dim=-1) + cos(dc[:, 2], -dc[:, 3], dim=-1))
        dx, dy = dc[:, 0] - dc[:, 1], dc[:, 2] - dc[:, 3]
        s_orth = 1.0 - (dx * dy).sum(-1).abs() / (dx.norm(dim=-1) * dy.norm(dim=-1)).clamp_min(1e-12)
        mean_a = dc.mean(dim=0, keepdim=True)
        s_cons = cos(dc, mean_a.expand_as(dc), dim=-1)
        stacked = dc.reshape(-1, dc.shape[-1])
        gram = stacked.t() @ stacked
        r_eff = float(stacked.double().pow(2).sum() ** 2 / gram.double().pow(2).sum().clamp_min(1e-300))
    return {'S_opp': float(s_opp.mean()), 'S_orth': float(s_orth.mean()),
            'S_cons': float(s_cons.mean()), 'r_eff': r_eff}

def one_step_geometry(w_hh, w_in, states, relations, gain, seed):
    import torch
    observed = _geometry_numbers(w_hh, w_in, states, relations, gain)
    magnitude = float(relations[0].norm())
    rng = torch.Generator().manual_seed(int(seed))
    draws = []
    for _ in range(NULL_DRAWS):
        theta = torch.rand(4, generator=rng, dtype=torch.float64) * 2.0 * math.pi
        directions = torch.stack([theta.cos(), theta.sin()], dim=-1).to(relations) * magnitude
        draws.append(_geometry_numbers(w_hh, w_in, states, directions, gain))
    row = dict(observed)
    for key in observed:
        values = torch.tensor([d[key] for d in draws], dtype=torch.float64)
        mean, sd = float(values.mean()), float(values.std(unbiased=True))
        row[f'{key}_null_mean'] = mean
        row[f'{key}_null_sd'] = sd
        row[f'{key}_z'] = (observed[key] - mean) / sd if sd > 0 else float('nan')
    return row

def composition(w_hh, w_in, states, relations, gain):
    import torch
    comm, inv = [], []
    with torch.no_grad():
        for start in range(0, states.shape[0], CHUNK):
            h = states[start:start + CHUNK]
            drives = (relations * gain) @ w_in.t()
            f0 = step(w_hh, h, 0.0)
            f00 = step(w_hh, f0, 0.0)
            once = [step(w_hh, h, drive) for drive in drives]
            for a in range(4):
                for b in range(a + 1, 4):
                    ab = step(w_hh, once[a], drives[b])
                    ba = step(w_hh, once[b], drives[a])
                    scale = 0.5 * ((ab - f00).norm(dim=-1) + (ba - f00).norm(dim=-1))
                    comm.append((ab - ba).norm(dim=-1) / scale.clamp_min(1e-12))
            for a, b in ((0, 1), (1, 0), (2, 3), (3, 2)):
                back = step(w_hh, once[a], drives[b])
                alone = step(w_hh, once[a], 0.0)
                inv.append((back - f00).norm(dim=-1) / (alone - f00).norm(dim=-1).clamp_min(1e-12))
    return {'C_comm': float(torch.cat(comm).mean()), 'C_inv': float(torch.cat(inv).mean())}

def relation_sequences(pools, split_by_length, lengths, n_rows, rng):
    """Sample one legal relation sequence per row for every length."""
    import numpy as np
    by_length = {}
    for split in split_by_length:
        for episode in pools[split]:
            length = int(len(episode.velocities))
            by_length.setdefault(length, []).append(np.asarray(episode.velocities, dtype=np.float32))
    return {length: np.stack([by_length[length][i] for i in rng.integers(0, len(by_length[length]), n_rows)])
            for length in lengths}

def roll(w_hh, w_in, h0, steps, gain, zero_mask=None):
    """h_L for a batch of equal-length sequences; zero_mask[t] removes step t."""
    import torch
    h = h0
    with torch.no_grad():
        for t in range(steps.shape[1]):
            u = steps[:, t]
            if zero_mask is not None and zero_mask[t]:
                u = torch.zeros_like(u)
            h = step(w_hh, h, (u * gain) @ w_in.t())
    return h

def _ridge_path(x, y):
    """Closed-form ridge for every penalty from one eigendecomposition; the
    n x n dual form is used when there are fewer rows than features."""
    import torch
    xm, ym = x.mean(0, keepdim=True), y.mean(0, keepdim=True)
    xc, yc = x - xm, y - ym
    if xc.shape[0] < xc.shape[1]:
        evals, evecs = torch.linalg.eigh(xc @ xc.t())
        proj = evecs.t() @ yc
        def solve(lam):
            coef = xc.t() @ (evecs @ (proj / (evals + lam).unsqueeze(-1)))
            return coef, ym - xm @ coef
    else:
        evals, evecs = torch.linalg.eigh(xc.t() @ xc)
        proj = evecs.t() @ (xc.t() @ yc)
        def solve(lam):
            coef = evecs @ (proj / (evals + lam).unsqueeze(-1))
            return coef, ym - xm @ coef
    return solve, float(evals.clamp_min(0.0).sum()) / x.shape[1]

def ridge_fit(x, y, rng_seed):
    import torch
    x, y = x.double(), y.double()
    n = x.shape[0]
    generator = torch.Generator().manual_seed(int(rng_seed))
    order = torch.randperm(n, generator=generator).to(x.device)
    hold, keep = order[: n // 5], order[n // 5:]
    solve, scale = _ridge_path(x[keep], y[keep])
    best = None
    for exponent in LAMBDA_EXPONENTS:
        lam = scale * 10.0 ** exponent
        coef, icpt = solve(lam)
        r2 = r_squared(y[hold], x[hold] @ coef + icpt)
        if best is None or r2 > best[0]:
            best = (r2, lam)
    solve_all, _ = _ridge_path(x, y)
    coef, icpt = solve_all(best[1])
    return coef, icpt, best[1], best[0]

def r_squared(y, pred):
    return 1.0 - float((y - pred).pow(2).sum()) / max(float((y - y.mean(0)).pow(2).sum()), 1e-300)

def horizon_retention(w_hh, w_in, fit_states, test_states, fit_seqs, test_seqs, gain, scale, seed):
    import numpy as np
    import torch
    device = fit_states.device
    xs, xd, ys = [], [], []
    for length in range(1, FIT_MAX_L + 1):
        steps = torch.as_tensor(fit_seqs[length], device=device)
        h_l = roll(w_hh, w_in, fit_states, steps, gain)
        xs.append(h_l)
        xd.append(h_l - roll(w_hh, w_in, fit_states, torch.zeros_like(steps), gain))
        ys.append(steps.sum(dim=1) / scale)
    coef, icpt, lam, val_r2 = ridge_fit(torch.cat(xs), torch.cat(ys), seed)
    coef_d, icpt_d, lam_d, val_r2_d = ridge_fit(torch.cat(xd), torch.cat(ys), seed)
    rows, contribution = [], []
    rng = np.random.default_rng(int(seed))
    for length in range(1, MAX_L + 1):
        steps = torch.as_tensor(test_seqs[length], device=device)
        h_l = roll(w_hh, w_in, test_states, steps, gain)
        zero_path = roll(w_hh, w_in, test_states, torch.zeros_like(steps), gain)
        y = (steps.sum(dim=1) / scale).double()
        pred = h_l.double() @ coef + icpt
        signal = (h_l - zero_path).norm(dim=-1)
        pred_d = (h_l - zero_path).double() @ coef_d + icpt_d
        row = {'L': length, 'n_test': int(steps.shape[0]), 'R2': r_squared(y, pred),
               'R2_diff': r_squared(y, pred_d),
               'signal_vs_zero_input': float(signal.mean()), 'hidden_norm': float(h_l.norm(dim=-1).mean()),
               'active_fraction': float((h_l > 0).float().mean())}
        if length >= 2:
            perm = torch.as_tensor(np.stack([rng.permutation(length) for _ in range(steps.shape[0])]), device=device)
            shuffled = torch.gather(steps, 1, perm.unsqueeze(-1).expand(-1, -1, 2))
            h_pi = roll(w_hh, w_in, test_states, shuffled, gain)
            row['O_L'] = float((h_l - h_pi).norm(dim=-1).mean() / signal.mean().clamp_min(1e-12))
        rows.append(row)
        if length == MAX_L:
            for t in range(MAX_L):
                mask = [i == t for i in range(MAX_L)]
                h_minus = roll(w_hh, w_in, test_states, steps, gain, zero_mask=mask)
                contribution.append({'t': t + 1, 'c_t': float((h_l - h_minus).norm(dim=-1).mean()
                                                              / signal.mean().clamp_min(1e-12))})
    by_l = {r['L']: r['R2'] for r in rows}
    by_d = {r['L']: r['R2_diff'] for r in rows}
    summary = {'R2_5': by_l[FIT_MAX_L], 'R2_12': by_l[MAX_L],
               'R_ret': by_l[MAX_L] / by_l[FIT_MAX_L] if by_l[FIT_MAX_L] > 0 else float('nan'),
               'R2_diff_5': by_d[FIT_MAX_L], 'R2_diff_12': by_d[MAX_L],
               'R_ret_diff': by_d[MAX_L] / by_d[FIT_MAX_L] if by_d[FIT_MAX_L] > 0 else float('nan'),
               'ridge_lambda': lam, 'ridge_validation_r2': val_r2,
               'ridge_lambda_diff': lam_d, 'ridge_validation_r2_diff': val_r2_d}
    return summary, rows, contribution

def delta_curve(w_hh, w_in, states, relations, gains=DELTA_CURVE):
    rows = delta_common.relative_displacement(w_hh, w_in, states, relations, gains)
    return [(r['gain'], r['delta_mean']) for r in rows]

def matching_gain(curve, target):
    """Gain at which delta(g) on the state set equals the downstream delta."""
    gains = [math.log(g) for g, _ in curve]
    deltas = [math.log(max(d, 1e-300)) for _, d in curve]
    t = math.log(target)
    if not (min(deltas) <= t <= max(deltas)):
        return float('nan'), 'outside_curve'
    for i in range(1, len(deltas)):
        lo, hi = deltas[i - 1], deltas[i]
        if (lo - t) * (hi - t) <= 0 and hi != lo:
            return math.exp(gains[i - 1] + (t - lo) * (gains[i] - gains[i - 1]) / (hi - lo)), 'interpolated'
    return float('nan'), 'not_monotone'
