"""Navigation training & evaluation.

Upgrades vs original (learning from reference code)
----------------------------------------------------
1. **Cross-entropy loss as default** — reference uses CE (place cell
   activations are soft probability distributions).  CE drives the
   network to produce a proper probability distribution at each step,
   which produces much sharper grid patterns than MSE.
2. **Separate W_hh regularisation** — reference penalises ONLY the
   recurrent weight: ``loss += weight_decay * (W_hh**2).sum()``.
   The original code used ``weight_decay`` in Adam which affects ALL
   parameters — this dilutes the regularisation signal that drives
   grid formation.  Now uses ``model.recurrent_l2()``.
3. **Learning-rate scheduler** — optional cosine annealing for more
   stable convergence in long training runs.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from gpm_cmap.analysis.grid_score import batch_grid_scores
from gpm_cmap.data.nav_env.place_cells import PlaceCellEncoder
from gpm_cmap.utils.metrics import AverageMeter


def hidden_decorrelation_loss(hidden_seq: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Penalise off-diagonal correlations in hidden activity.

    This is intentionally optional.  It turns the checkpoint into a
    ``grid-tuned`` navigation prior, not a pure PI reference checkpoint.
    """
    h = hidden_seq.reshape(-1, hidden_seq.shape[-1])
    h = h - h.mean(dim=0, keepdim=True)
    h = h / (h.std(dim=0, keepdim=True, unbiased=False) + eps)
    cov = (h.T @ h) / max(h.shape[0] - 1, 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    return (off_diag ** 2).mean()


def compute_hidden_diagnostics(hidden: np.ndarray, max_cov_units: int = 2048) -> Dict[str, float]:
    """Return cheap diagnostics for detecting ramp / dead-unit solutions."""
    h = np.asarray(hidden, dtype=np.float64)
    if h.ndim != 2 or h.shape[0] == 0 or h.shape[1] == 0:
        return {}
    unit_mean = h.mean(axis=0)
    unit_std = h.std(axis=0)
    active_fraction_by_unit = (h > 0).mean(axis=0)
    out = {
        "hidden_mean": float(h.mean()),
        "hidden_std": float(h.std()),
        "hidden_active_fraction": float((h > 0).mean()),
        "dead_unit_fraction": float(np.mean(active_fraction_by_unit < 1e-4)),
        "near_dead_unit_fraction": float(np.mean(unit_std < 1e-6)),
    }
    # Participation ratio from covariance eigenvalues.  Use a deterministic
    # subset for very wide models to avoid large CPU/RAM spikes during logging.
    if h.shape[1] > max_cov_units:
        rng = np.random.default_rng(0)
        idx = rng.choice(h.shape[1], size=max_cov_units, replace=False)
        hc = h[:, idx]
    else:
        hc = h
    hc = hc - hc.mean(axis=0, keepdims=True)
    cov = (hc.T @ hc) / max(hc.shape[0] - 1, 1)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.maximum(eigvals, 0.0)
    denom = float((eigvals ** 2).sum())
    out["hidden_participation_ratio"] = float((eigvals.sum() ** 2) / denom) if denom > 0 else 0.0
    # Correlation off-diagonal mean square; robust to unit scale.
    std = np.sqrt(np.diag(cov))
    valid = std > 1e-12
    if int(valid.sum()) >= 2:
        cv = cov[np.ix_(valid, valid)]
        sv = std[valid]
        corr = cv / np.outer(sv, sv)
        off = corr - np.eye(corr.shape[0])
        out["hidden_offdiag_corr_mse"] = float((off ** 2).sum() / (off.size - corr.shape[0]))
    else:
        out["hidden_offdiag_corr_mse"] = float("nan")
    return out


def train_navigation(
    model: torch.nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float = 1.0,
    loss_type: str = "ce",                   # [UPGRADE] default to CE
    whh_weight_decay: float = 1e-4,          # [NEW] separate W_hh regularisation
    hidden_decorrelation_weight: float = 0.0,
    hidden_stability_weight: float = 0.0,
) -> Dict[str, float]:
    """Train one epoch on navigation path-integration task.

    Parameters
    ----------
    loss_type : str
        "ce"  — cross-entropy / KL divergence (RECOMMENDED, reference default).
        "mse" — mean squared error (original default, weaker for grid formation).
    whh_weight_decay : float
        [NEW] L2 penalty applied ONLY to recurrent weights W_hh.
        Set to 0 to disable.  Reference default: 1e-4.
    """
    model.train()
    total_meter = AverageMeter()
    task_meter = AverageMeter()
    recurrent_meter = AverageMeter()
    decorrelation_meter = AverageMeter()
    stability_meter = AverageMeter()
    grad_meter = AverageMeter()

    for batch in train_loader:
        velocities    = batch["velocities"].to(device)
        place_targets = batch["place_targets"].to(device)
        init_place    = batch["init_place"].to(device)   # [FIX] PC(pos[1]), not PC(pos[2])!
        outputs       = model(velocities, init_place=init_place)

        # ---- task loss ----
        if loss_type == "ce":
            # [UPGRADE] Reference-style cross-entropy:
            #   targets are soft probabilities from place cell Gaussians,
            #   predictions go through softmax then cross-entropy
            pred = outputs["place_pred"].reshape(-1, outputs["place_pred"].shape[-1])
            target = place_targets.reshape(-1, place_targets.shape[-1])
            log_pred = F.log_softmax(pred, dim=-1)
            task_loss = -(target * log_pred).sum(-1).mean()
        else:
            task_loss = F.mse_loss(outputs["place_pred"], place_targets)

        recurrent_penalty = torch.zeros((), device=device)
        decorrelation_penalty = torch.zeros((), device=device)
        stability_penalty = torch.zeros((), device=device)

        # ---- W_hh regularisation (reference approach) ----
        # [UPGRADE] Only penalise recurrent weight, not all parameters
        if whh_weight_decay > 0 and hasattr(model, "recurrent_l2"):
            recurrent_penalty = whh_weight_decay * model.recurrent_l2()

        # ---- optional hidden decorrelation (512 grid-tuned objective) ----
        if hidden_decorrelation_weight > 0:
            decorrelation_penalty = (
                hidden_decorrelation_weight * hidden_decorrelation_loss(outputs["hidden_seq"])
            )

        if hidden_stability_weight > 0 and "hidden_seq" in outputs:
            # Encourage near-isometric short-time dynamics instead of memorising
            # sequence length/padding. This is a weak form of action-dynamics
            # regularisation and does not replace explicit equivariance losses.
            h = outputs["hidden_seq"]
            dh = h[:, 1:] - h[:, :-1]
            stability_penalty = hidden_stability_weight * (dh.pow(2).mean())

        loss = task_loss + recurrent_penalty + decorrelation_penalty + stability_penalty
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite navigation loss: {float(loss.detach())}")

        optimizer.zero_grad()
        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        else:
            squared = [
                parameter.grad.detach().pow(2).sum()
                for parameter in model.parameters()
                if parameter.grad is not None
            ]
            grad_norm = torch.sqrt(torch.stack(squared).sum()) if squared else torch.zeros(())
        if not torch.isfinite(torch.as_tensor(grad_norm)):
            raise FloatingPointError(f"Non-finite navigation gradient norm: {grad_norm}")
        optimizer.step()

        batch_size = velocities.size(0)
        total_meter.update(loss.item(), batch_size)
        task_meter.update(task_loss.item(), batch_size)
        recurrent_meter.update(recurrent_penalty.item(), batch_size)
        decorrelation_meter.update(decorrelation_penalty.item(), batch_size)
        stability_meter.update(stability_penalty.item(), batch_size)
        grad_meter.update(float(torch.as_tensor(grad_norm).item()), batch_size)

    return {
        "loss": total_meter.avg,
        "total_loss": total_meter.avg,
        "task_loss": task_meter.avg,
        "recurrent_regularizer": recurrent_meter.avg,
        "decorrelation_regularizer": decorrelation_meter.avg,
        "stability_regularizer": stability_meter.avg,
        "gradient_norm": grad_meter.avg,
    }


@torch.no_grad()
def evaluate_navigation_loss(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Cheap validation pass used for checkpoint selection every epoch.

    Grid maps, SACs and hidden-state matrices are intentionally omitted.  The
    expensive scientific analysis runs only at the configured cadence and
    after a checkpoint has been locked.
    """
    model.eval()
    loss_meter = AverageMeter()
    entropy_meter = AverageMeter()
    n_place = None
    for batch in data_loader:
        velocities = batch["velocities"].to(device)
        target = batch["place_targets"].to(device)
        init_place = batch["init_place"].to(device)
        outputs = model(velocities, init_place=init_place)
        pred_flat = outputs["place_pred"].reshape(-1, outputs["place_pred"].shape[-1])
        target_flat = target.reshape(-1, target.shape[-1])
        if not torch.isfinite(pred_flat).all() or not torch.isfinite(target_flat).all():
            raise FloatingPointError("Non-finite navigation validation tensor")
        row_sums = target_flat.sum(dim=-1)
        if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-3):
            raise ValueError("Navigation place-cell targets must sum to one")
        if float(target_flat.min()) < -1e-6:
            raise ValueError("Navigation place-cell targets must be non-negative")
        loss = -(target_flat * F.log_softmax(pred_flat, dim=-1)).sum(-1).mean()
        entropy = -(target_flat * torch.log(target_flat.clamp_min(1e-30))).sum(-1).mean()
        loss_meter.update(loss.item(), velocities.size(0))
        entropy_meter.update(entropy.item(), velocities.size(0))
        n_place = int(pred_flat.shape[-1])
    if n_place is None:
        raise RuntimeError("Navigation validation loader is empty")
    uniform_loss = float(np.log(n_place))
    denominator = uniform_loss - entropy_meter.avg
    return {
        "loss": loss_meter.avg,
        "place_target_entropy": entropy_meter.avg,
        "uniform_loss": uniform_loss,
        "loss_excess_over_floor": loss_meter.avg - entropy_meter.avg,
        "frac_loss_captured": (
            (uniform_loss - loss_meter.avg) / denominator
            if denominator > 1e-12 else float("nan")
        ),
    }


@torch.no_grad()
def evaluate_navigation(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    arena_size: float = 2.2,
    grid_bins: int = 32,
    grid_smooth_sigma: float = 0.0,
    place_encoder: Optional[PlaceCellEncoder] = None,
    include_google_ref_score: bool = False,
    include_hidden_diagnostics: bool = False,
    grid_burn_in_steps: int = 0,
    grid_min_step_displacement: float = 0.0,
) -> Dict[str, object]:
    """Evaluate navigation model.

    Returns dict with: loss, positions, hidden, headings, grid_results,
    mean_grid_score, max_grid_score, frac_grid_above_03,
    and (if place_encoder given) path_integration_error,
    path_integration_error_relative, decoded_positions.
    """
    model.eval()
    meter = AverageMeter()
    all_positions: List[np.ndarray] = []
    all_hidden: List[np.ndarray] = []
    all_headings: List[np.ndarray] = []
    all_trajectory_ids: List[np.ndarray] = []
    all_time_indices: List[np.ndarray] = []
    all_step_displacements: List[np.ndarray] = []
    all_place_pred: List[torch.Tensor] = []
    # [FIX-M6] Targets are needed to compute H(p), the cross-entropy floor.
    all_place_targets: List[torch.Tensor] = []
    trajectory_offset = 0

    for batch in data_loader:
        velocities = batch["velocities"].to(device)
        place_targets = batch["place_targets"].to(device)
        init_place = batch["init_place"].to(device)       # [FIX] PC(pos[1])
        outputs = model(velocities, init_place=init_place)
        # [FIX] Use CE loss for val (same as training), not MSE
        pred = outputs["place_pred"].reshape(-1, outputs["place_pred"].shape[-1])
        target = place_targets.reshape(-1, place_targets.shape[-1])
        loss = -(target * F.log_softmax(pred, dim=-1)).sum(-1).mean()
        meter.update(loss.item(), velocities.size(0))

        batch_positions = batch["positions"]
        batch_size, seq_len = batch_positions.shape[:2]
        all_positions.append(batch_positions.reshape(-1, 2).cpu().numpy())
        all_hidden.append(
            outputs["hidden_seq"].reshape(-1, outputs["hidden_seq"].shape[-1]).cpu().numpy()
        )
        all_headings.append(batch["headings"].reshape(-1).cpu().numpy())
        all_trajectory_ids.append(
            np.repeat(np.arange(trajectory_offset, trajectory_offset + batch_size), seq_len)
        )
        all_time_indices.append(np.tile(np.arange(seq_len), batch_size))
        all_step_displacements.append(
            torch.linalg.vector_norm(batch["velocities"], dim=-1).reshape(-1).cpu().numpy()
        )
        trajectory_offset += batch_size
        all_place_pred.append(
            outputs["place_pred"].reshape(-1, outputs["place_pred"].shape[-1]).cpu()
        )
        all_place_targets.append(target.detach().cpu())

    positions = np.concatenate(all_positions, axis=0)
    hidden = np.concatenate(all_hidden, axis=0)
    headings = np.concatenate(all_headings, axis=0)
    trajectory_ids = np.concatenate(all_trajectory_ids, axis=0)
    time_indices = np.concatenate(all_time_indices, axis=0)
    step_displacements = np.concatenate(all_step_displacements, axis=0)
    place_pred_all = torch.cat(all_place_pred, dim=0)

    # [FIX-M6] Report the cross-entropy against its theoretical FLOOR.
    #
    # The place-cell target is a soft distribution, so the minimum achievable
    # cross-entropy is not 0 but H(p), the entropy of the target itself. With
    # the DoG encoder the target is nearly uniform (the DoG shift-and-normalise
    # step lifts every cell above zero), so for 512 cells H(p) ~= 6.14 versus
    # ln(512) = 6.24 for a uniform prediction: the ENTIRE learnable signal is
    # only ~0.10 nats. A raw val_loss of 6.15 therefore looks like "barely
    # learned anything" while actually capturing ~92% of the achievable
    # reduction. Without this normalisation the loss number is uninterpretable
    # and invites the wrong conclusion that navigation pretraining failed.
    target_entropy = float(
        (-(place_targets_cat := torch.cat(all_place_targets, dim=0))
         * torch.log(place_targets_cat.clamp_min(1e-30))).sum(-1).mean().item()
    )
    n_place = int(place_pred_all.shape[-1])
    uniform_loss = float(np.log(n_place))
    denom = uniform_loss - target_entropy

    # --- Grid score analysis ---
    # The torus paper restricted awake analyses to moving periods.  A dedicated
    # long-rollout topology evaluation can also omit early recurrent burn-in;
    # ordinary validation retains the historical all-sample behaviour because
    # both optional thresholds default to zero.
    grid_mask = np.ones(hidden.shape[0], dtype=bool)
    if int(grid_burn_in_steps) > 0:
        grid_mask &= time_indices >= int(grid_burn_in_steps)
    if float(grid_min_step_displacement) > 0:
        grid_mask &= step_displacements > float(grid_min_step_displacement)
    if int(grid_mask.sum()) < 10:
        raise ValueError(
            "Fewer than ten samples remain for grid-score analysis after "
            "burn-in/movement filtering"
        )
    grid_results = batch_grid_scores(
        positions[grid_mask],
        hidden[grid_mask],
        bins=grid_bins,
        arena_size=arena_size,
        smooth_sigma=grid_smooth_sigma,
        include_google_ref=include_google_ref_score,
    )    
    grid_scores = [float(item["grid_score"]) for item in grid_results]
    mean_grid_score = float(np.mean(grid_scores))
    max_grid_score = float(np.max(grid_scores))
    frac_above_03 = float(np.mean([s > 0.3 for s in grid_scores]))

    # T8: report the conservative (min-max) gridness in parallel so claims can
    # cite the stricter metric without re-running analysis. Each unit dict now
    # carries grid_score_conservative (NaN where not applicable); we aggregate
    # over the finite values only.
    cons_scores = [
        float(item.get("grid_score_conservative", float("nan")))
        for item in grid_results
    ]
    finite_cons = [s for s in cons_scores if np.isfinite(s)]
    mean_grid_score_conservative = float(np.mean(finite_cons)) if finite_cons else float("nan")
    max_grid_score_conservative = float(np.max(finite_cons)) if finite_cons else float("nan")
    frac_above_03_conservative = (
        float(np.mean([s > 0.3 for s in finite_cons])) if finite_cons else float("nan")
    )

    google_scores = [
        float(item.get("grid_score_google", float("nan")))
        for item in grid_results
    ]
    finite_google = [s for s in google_scores if np.isfinite(s)]

    # [FIX-M6] Validate the target really is a normalised distribution before
    # interpreting its entropy as a loss floor (fail loud rather than silently
    # reporting a meaningless "%captured").
    tgt_sums = place_targets_cat.sum(-1)
    if not torch.allclose(tgt_sums, torch.ones_like(tgt_sums), atol=1e-3):
        raise ValueError(
            "place_targets rows must sum to 1 to interpret H(p) as a CE floor; "
            f"observed sum range [{float(tgt_sums.min()):.6f}, {float(tgt_sums.max()):.6f}]."
        )
    if float(place_targets_cat.min()) < -1e-6:
        raise ValueError(
            f"place_targets must be non-negative; min={float(place_targets_cat.min()):.6g}."
        )

    result: Dict[str, object] = {
        "loss": meter.avg,
        # Cross-entropy floor diagnostics (see comment above).
        "place_target_entropy": target_entropy,
        "uniform_loss": uniform_loss,
        "loss_excess_over_floor": float(meter.avg - target_entropy),
        "frac_loss_captured": float((uniform_loss - meter.avg) / denom) if denom > 1e-12 else float("nan"),
        "positions": positions,
        "hidden": hidden,
        "headings": headings,
        "trajectory_ids": trajectory_ids,
        "time_indices": time_indices,
        "step_displacements": step_displacements,
        "grid_sample_count": int(grid_mask.sum()),
        "grid_burn_in_steps": int(grid_burn_in_steps),
        "grid_min_step_displacement": float(grid_min_step_displacement),
        "grid_results": grid_results,
        "mean_grid_score": mean_grid_score,
        "max_grid_score": max_grid_score,
        "frac_grid_above_03": frac_above_03,
        "mean_grid_score_conservative": mean_grid_score_conservative,
        "max_grid_score_conservative": max_grid_score_conservative,
        "frac_grid_above_03_conservative": frac_above_03_conservative,
        "mean_grid_score_google": float(np.mean(finite_google)) if finite_google else float("nan"),
        "max_grid_score_google": float(np.max(finite_google)) if finite_google else float("nan"),
        "frac_grid_above_03_google": (
            float(np.mean([s > 0.3 for s in finite_google])) if finite_google else float("nan")
        ),
    }

    if include_hidden_diagnostics:
        result.update(compute_hidden_diagnostics(hidden))

    # --- Path integration error ---
    if place_encoder is not None:
        place_prob = F.softmax(place_pred_all, dim=-1)
        decoded_positions = place_encoder.decode(place_prob).numpy()
        errors = np.linalg.norm(decoded_positions - positions, axis=-1)
        pi_error_mean = float(np.mean(errors))
        pi_error_relative = pi_error_mean / arena_size
        result["path_integration_error"] = pi_error_mean
        result["path_integration_error_relative"] = pi_error_relative
        result["decoded_positions"] = decoded_positions

    return result
