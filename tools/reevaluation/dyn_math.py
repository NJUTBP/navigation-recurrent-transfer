#!/usr/bin/env python3
"""Exact tangent / attribution machinery for bias-free ReLU recurrent cores (B6).

Row-vector convention (as in the model code):  h_t = ReLU(h_{t-1} W^T + b_t),  b_t = (g v_t) W_in^T.
  realised mask   d_t = 1[W h_{t-1} + b_t > 0]      (ReLU'(0) := 0, the PyTorch convention)
  local Jacobian  J_t = D_t W,   D_t = diag(d_t)
  propagator      P_{s->t} = J_t ... J_{s+1},   P_{t->t} = I

1. Exact attribution (piecewise linearity, no approximation given the realised masks):
       h_L = P_{0->L} h_0 + sum_{t=1..L} P_{t->L} D_t b_t .
   The input perturbation enters through D_t at the injection step (the MD's P_{t->L} b omits D_t).
   ``exact_decomposition`` returns the L+1 contributions; ``decomposition_residual`` checks the sum.
2. Autonomous radial-angular decomposition (exact for b_t = 0 only).  F(h) = ReLU(W h) is
   positively homogeneous, so with h = r u (|u| = 1):  r' = r kappa(u), kappa(u) = |F(u)|, and
   u' = F(u)/|F(u)| independent of r.  The tangent map of the projective dynamics on the sphere is
       DP(u) v = (I - u'u'^T) J(u) v / kappa(u),   v perpendicular to u.
   Euler's identity J(u) u = F(u) makes the radial line invariant under the full tangent cocycle;
   the angular exponents are the exponents of the quotient cocycle minus lambda_rad.  They are
   computed here directly on the tangent space of the sphere (radial direction projected out at
   every step), so no spurious zero exponent enters the spectrum and d0 is not inflated by one,
   unlike 'lambda_i - lambda_rad' applied to the full top-k spectrum.
   Rescaling W -> cW (c > 0) leaves every mask, u_t and every angular exponent unchanged and shifts
   lambda_rad by log c exactly: this is a mathematical identity used as an implementation check,
   not as evidence about rescue/impairment.
3. Driven dynamics: J_t h_{t-1} != h_t, the radial line is not invariant, and no radial/angular split
   of a driven spectrum is claimed.  Driven quantities are finite-time singular values of P (called
   finite-time growth rates, (1/L) log sigma_i, not Lyapunov exponents) and the propagation of
   specified vectors.  Finite-difference / impulse checks validate every Jacobian product.
"""
from __future__ import annotations

import torch


def masked_step(h, w_hh, drive=None):
    pre = h @ w_hh.t()
    if drive is not None:
        pre = pre + drive
    mask = pre > 0
    return torch.where(mask, pre, torch.zeros_like(pre)), mask


def rollout(h0, w_hh, drives):
    """drives (B, L, N).  Returns states (B, L, N) = h_1..h_L and masks (B, L, N) bool."""
    h, states, masks = h0, [], []
    for t in range(drives.shape[1]):
        h, m = masked_step(h, w_hh, drives[:, t])
        states.append(h)
        masks.append(m)
    return torch.stack(states, 1), torch.stack(masks, 1)


def exact_decomposition(h0, drives, masks, w_hh):
    """Contributions (B, L+1, N): [:, 0] = P_{0->L} h0, [:, t] = P_{t->L} D_t b_t; plus D_t b_t (B, L, N)."""
    dtype = h0.dtype
    contrib = h0.unsqueeze(1)
    injected = []
    for t in range(drives.shape[1]):
        d = masks[:, t].to(dtype)
        contrib = (contrib @ w_hh.t()) * d.unsqueeze(1)
        inj = drives[:, t] * d
        injected.append(inj)
        contrib = torch.cat([contrib, inj.unsqueeze(1)], dim=1)
    return contrib, torch.stack(injected, 1)


def decomposition_residual(contrib, h_final):
    return (contrib.sum(dim=1) - h_final).norm(dim=-1) / h_final.norm(dim=-1).clamp_min(1e-30)


def propagate(vectors, w_hh, masks, start: int = 0):
    """Apply J_{start+1}..J_L (0-based mask indices start..L-1) to row vectors (B, k, N)."""
    v = vectors
    for t in range(start, masks.shape[1]):
        v = (v @ w_hh.t()) * masks[:, t].to(v.dtype).unsqueeze(1)
    return v


def propagate_adjoint(vectors, w_hh, masks, start: int = 0):
    """Apply P_{start->L}^T: y <- W^T (d_t * y) for t = L .. start+1."""
    v = vectors
    for t in range(masks.shape[1] - 1, start - 1, -1):
        v = (v * masks[:, t].to(v.dtype).unsqueeze(1)) @ w_hh
    return v


def propagator_singular_values(w_hh, masks, *, k: int = 16, iterations: int = 6, seed: int = 0, start: int = 0):
    """Top-k singular values of P_{start->L} per trial by subspace iteration on explicit products."""
    b, _, n = masks.shape
    generator = torch.Generator().manual_seed(int(seed))
    q = torch.randn(b, n, k, generator=generator, dtype=torch.float64).to(device=w_hh.device, dtype=w_hh.dtype)
    q, _ = torch.linalg.qr(q)
    for _ in range(int(iterations)):
        y = propagate(q.transpose(1, 2), w_hh, masks, start)
        z = propagate_adjoint(y, w_hh, masks, start)
        q, _ = torch.linalg.qr(z.transpose(1, 2))
    y = propagate(q.transpose(1, 2), w_hh, masks, start)
    return torch.linalg.svdvals(y)


def project_out(v, u):
    """Remove the component of each row of v (B, m, N) along the unit vector u (B, N)."""
    return v - (v * u.unsqueeze(1)).sum(-1, keepdim=True) * u.unsqueeze(1)


def autonomous_projective(h0, w_hh, *, burn_in: int = 50, steps: int = 200, k: int = 16, seed: int = 0,
                          eps: float = 1e-20):
    """Radial exponent and top-k angular exponents of the zero-input dynamics started at h0 (B, N).

    lambda_rad = time-average of log kappa(u_t) along the projective trajectory (after burn-in);
    lambda_ang = exponents of DP restricted to the tangent space of the sphere (QR re-orthonormalised).
    A trajectory whose state is mapped to 0 (kappa = 0) is absorbed and flagged as not survived.
    """
    b, n = h0.shape
    dtype, device = h0.dtype, h0.device
    norm0 = h0.norm(dim=-1)
    alive = norm0 > eps
    u = h0 / norm0.clamp_min(eps).unsqueeze(-1)
    generator = torch.Generator().manual_seed(int(seed))
    v = torch.randn(b, k, n, generator=generator, dtype=torch.float64).to(device=device, dtype=dtype)
    q, _ = torch.linalg.qr(project_out(v, u).transpose(1, 2))
    v = q.transpose(1, 2)
    sum_log_kappa = torch.zeros(b, dtype=torch.float64, device=device)
    sum_log_r = torch.zeros(b, k, dtype=torch.float64, device=device)
    count = torch.zeros(b, dtype=torch.float64, device=device)
    floor = torch.finfo(dtype).tiny
    kappa_first, active = None, []
    for step in range(int(burn_in) + int(steps)):
        pre = u @ w_hh.t()
        mask = pre > 0
        f = torch.where(mask, pre, torch.zeros_like(pre))
        kappa = f.norm(dim=-1)
        if kappa_first is None:
            kappa_first = kappa.clone()
        alive = alive & (kappa > eps)
        safe = torch.where(alive, kappa, torch.ones_like(kappa))
        u_next = torch.where(alive.unsqueeze(-1), f / safe.unsqueeze(-1), u)
        v = (v @ w_hh.t()) * mask.to(dtype).unsqueeze(1) / safe.view(b, 1, 1)
        q, r = torch.linalg.qr(project_out(v, u_next).transpose(1, 2))
        diag = r.diagonal(dim1=-2, dim2=-1).abs().clamp_min(floor)
        v = q.transpose(1, 2)
        if step >= burn_in:
            live = alive.to(torch.float64)
            sum_log_kappa += live * safe.log().double()
            sum_log_r += live.unsqueeze(-1) * diag.log().double()
            count += live
            active.append(mask.to(torch.float32).mean(dim=-1))
        u = u_next
    denom = count.clamp_min(1.0)
    lam_ang = (sum_log_r / denom.unsqueeze(-1)).sort(dim=-1, descending=True).values
    return {'kappa_first': kappa_first, 'lambda_rad': sum_log_kappa / denom, 'lambda_ang': lam_ang,
            'survived': alive & (count >= int(steps)),
            'active_fraction': torch.stack(active, 1).mean(1) if active else torch.full((b,), float('nan'))}


def input_angular_retention(h0, w_hh, directions, *, steps: int = 10, eps: float = 1e-20):
    """Finite-time angular growth rate of infinitesimal inputs injected at h0 along `directions` (m, N).

    Step 1 injects D_1 w (rectified at the injection step); steps 2..K propagate under the
    projective tangent map.  Returns (B, m) angular rates (1/(K-1)) log(|v_K|/|v_1|) and the
    state-relative linear gain |P w| / |P_state| (1 for an ideal integrator / gamma I).
    """
    b = h0.shape[0]
    dtype = h0.dtype
    u = h0 / h0.norm(dim=-1, keepdim=True).clamp_min(eps)
    pre = u @ w_hh.t()
    mask = pre > 0
    f = torch.where(mask, pre, torch.zeros_like(pre))
    kappa = f.norm(dim=-1).clamp_min(eps)
    u = f / kappa.unsqueeze(-1)
    full = directions.to(dtype).unsqueeze(0) * mask.to(dtype).unsqueeze(1)
    ang = project_out(full / kappa.view(b, 1, 1), u)
    full0 = full.norm(dim=-1).clamp_min(eps)
    ang0 = ang.norm(dim=-1).clamp_min(eps)
    log_state = torch.zeros(b, dtype=torch.float64, device=h0.device)
    for _ in range(int(steps) - 1):
        pre = u @ w_hh.t()
        mask = pre > 0
        f = torch.where(mask, pre, torch.zeros_like(pre))
        kappa = f.norm(dim=-1).clamp_min(eps)
        u_next = f / kappa.unsqueeze(-1)
        full = (full @ w_hh.t()) * mask.to(dtype).unsqueeze(1)
        ang = project_out(((ang @ w_hh.t()) * mask.to(dtype).unsqueeze(1)) / kappa.view(b, 1, 1), u_next)
        log_state += kappa.log().double()
        u = u_next
    denom = max(1, int(steps) - 1)
    rate = (ang.norm(dim=-1).clamp_min(eps) / ang0).log().double() / denom
    rel_gain = ((full.norm(dim=-1).clamp_min(eps) / full0).log().double() - log_state.unsqueeze(-1)).exp()
    return rate, rel_gain


def margin_attribution(contrib, h_final, w_out, b_out, targets):
    """Exact split of the correct-minus-runner-up logit margin over start and per-step contributions."""
    logits = h_final @ w_out.t() + b_out
    masked = logits.clone()
    rows = torch.arange(targets.shape[0], device=targets.device)
    masked[rows, targets] = -float('inf')
    runner = masked.argmax(dim=-1)
    direction = w_out[targets] - w_out[runner]
    centred = w_out[targets] - w_out.mean(dim=0, keepdim=True)
    return {'mu': (contrib * direction.unsqueeze(1)).sum(-1),
            'mu_centred': (contrib * centred.unsqueeze(1)).sum(-1),
            'bias': b_out[targets] - b_out[runner],
            'margin': logits[rows, targets] - logits[rows, runner],
            'correct': logits.argmax(dim=-1) == targets}


def fd_validation(h0, drives, masks, w_hh, contrib, steps_to_check, *, rel_eps: float = 1e-3, seed: int = 0):
    """Finite-difference checks of the exact attribution and of the propagator (impulse experiment)."""
    states, _ = rollout(h0, w_hh, drives)
    h_last = states[:, -1]
    generator = torch.Generator().manual_seed(int(seed))
    rows = []
    for t in steps_to_check:
        scaled = drives.clone()
        scaled[:, t] = scaled[:, t] * (1.0 + rel_eps)
        st, ms = rollout(h0, w_hh, scaled)
        predicted = rel_eps * contrib[:, t + 1]
        err = (st[:, -1] - h_last - predicted).norm(dim=-1) / predicted.norm(dim=-1).clamp_min(1e-30)
        flips = (ms != masks).flatten(1).sum(1)
        z = torch.randn(h0.shape, generator=generator, dtype=torch.float64).to(device=h0.device, dtype=h0.dtype)
        z = z / z.norm(dim=-1, keepdim=True)
        size = rel_eps * drives[:, t].norm(dim=-1, keepdim=True)
        impulsed = drives.clone()
        impulsed[:, t] = impulsed[:, t] + size * z
        st2, ms2 = rollout(h0, w_hh, impulsed)
        pred_z = size * propagate((z * masks[:, t].to(z.dtype)).unsqueeze(1), w_hh, masks, start=t + 1).squeeze(1)
        err_z = (st2[:, -1] - h_last - pred_z).norm(dim=-1) / pred_z.norm(dim=-1).clamp_min(1e-30)
        flips_z = (ms2 != masks).flatten(1).sum(1)
        clean, clean_z = flips == 0, flips_z == 0
        rows.append({'step_index_0based': int(t), 'n': int(h0.shape[0]),
                     'scale_relerr_median': float(err.median()), 'scale_relerr_max': float(err.max()),
                     'scale_zero_flip_fraction': float(clean.float().mean()),
                     'scale_relerr_max_zero_flip': float(err[clean].max()) if bool(clean.any()) else float('nan'),
                     'impulse_relerr_median': float(err_z.median()),
                     'impulse_zero_flip_fraction': float(clean_z.float().mean()),
                     'impulse_relerr_max_zero_flip': float(err_z[clean_z].max()) if bool(clean_z.any()) else float('nan'),
                     'mean_mask_flips_scale': float(flips.float().mean()), 'mean_mask_flips_impulse': float(flips_z.float().mean())})
    return rows
