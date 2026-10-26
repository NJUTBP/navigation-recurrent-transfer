"""Weight construction and diagnostics for the four-factor recurrent experiment."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from gpm_cmap.utils.checkpointing import sha256_file

HH_ORIGINS = ("navigation", "matched_random")
IH_ORIGINS = ("navigation", "matched_random")
TRAINABILITY = ("frozen", "trainable")


@dataclass(frozen=True)
class FactorialCell:
    hh_origin: str
    ih_origin: str
    hh_trainability: str
    ih_trainability: str

    def __post_init__(self) -> None:
        if self.hh_origin not in HH_ORIGINS:
            raise ValueError(f"invalid W_hh origin: {self.hh_origin}")
        if self.ih_origin not in IH_ORIGINS:
            raise ValueError(f"invalid W_ih origin: {self.ih_origin}")
        if self.hh_trainability not in TRAINABILITY:
            raise ValueError(f"invalid W_hh trainability: {self.hh_trainability}")
        if self.ih_trainability not in TRAINABILITY:
            raise ValueError(f"invalid W_ih trainability: {self.ih_trainability}")

    @property
    def label(self) -> str:
        return (
            f"Ohh-{self.hh_origin}__Oih-{self.ih_origin}__"
            f"Thh-{self.hh_trainability}__Tih-{self.ih_trainability}"
        )

    def as_dict(self) -> dict[str, str]:
        return {
            "hh_origin": self.hh_origin,
            "ih_origin": self.ih_origin,
            "hh_trainability": self.hh_trainability,
            "ih_trainability": self.ih_trainability,
            "cell": self.label,
        }


def all_factorial_cells() -> tuple[FactorialCell, ...]:
    return tuple(
        FactorialCell(hh, ih, thh, tih)
        for hh in HH_ORIGINS
        for ih in IH_ORIGINS
        for thh in TRAINABILITY
        for tih in TRAINABILITY
    )


def parse_factorial_cell(label: str) -> FactorialCell:
    fields = {}
    for part in str(label).split("__"):
        key, value = part.split("-", 1)
        fields[key] = value
    return FactorialCell(
        hh_origin=fields["Ohh"],
        ih_origin=fields["Oih"],
        hh_trainability=fields["Thh"],
        ih_trainability=fields["Tih"],
    )


def load_navigation_grid_weights(checkpoint_path: str | Path) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Navigation checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model_state", payload)
    hh_key = "grid_rnn.cell.h_proj.weight"
    ih_key = "grid_rnn.cell.in_proj.weight"
    if hh_key not in state or ih_key not in state:
        raise KeyError(f"Navigation checkpoint must contain {hh_key!r} and {ih_key!r}")
    # Preserve checkpoint dtype; promoting a 4096x4096 matrix to float64
    # needlessly doubles the formal factorial experiment's peak memory.
    hh = state[hh_key].detach().cpu().clone()
    ih = state[ih_key].detach().cpu().clone()
    if hh.ndim != 2 or hh.shape[0] != hh.shape[1]:
        raise ValueError(f"W_hh must be square; got {tuple(hh.shape)}")
    if ih.shape != (hh.shape[0], 2):
        raise ValueError(f"W_ih must have shape {(hh.shape[0], 2)}; got {tuple(ih.shape)}")
    if not torch.isfinite(hh).all() or not torch.isfinite(ih).all():
        raise FloatingPointError("Navigation weights contain NaN or Inf")
    return hh, ih, {
        "checkpoint": str(path.resolve()),
        "checkpoint_sha256": sha256_file(path),
        "hidden_dim": int(hh.shape[0]),
    }


def _spectral_radius_power(
    matrix: torch.Tensor,
    *,
    iterations: int = 16,
    seed: int = 0,
    n_probes: int = 2,
) -> float:
    """Stabilized Gelfand-sequence estimate for a non-normal matrix."""
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("spectral-radius estimate requires a square matrix")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    estimates: list[float] = []
    steps = max(4, int(iterations))
    for _ in range(max(1, int(n_probes))):
        vector = torch.randn(matrix.shape[1], dtype=matrix.dtype, generator=generator)
        vector = vector / vector.norm().clamp_min(1e-12)
        log_growth = 0.0
        valid = True
        for _step in range(steps):
            product = matrix @ vector
            norm = float(product.norm().item())
            if not np.isfinite(norm) or norm <= 1e-20:
                valid = False
                break
            log_growth += float(np.log(norm))
            vector = product / norm
        if valid:
            estimates.append(float(np.exp(log_growth / steps)))
    return float(np.median(estimates)) if estimates else 0.0


def _top_singular_values(
    matrix: torch.Tensor,
    *,
    rank: int,
    seed: int,
    power_iterations: int = 1,
) -> np.ndarray:
    """Memory-bounded randomized principal singular-value diagnostic.

    The formal matched-random construction bypasses this routine because its
    complete singular spectrum is known exactly by construction.  This path is
    retained for optional external checkpoint controls.
    """
    n_rows, n_cols = matrix.shape
    rank = max(1, min(int(rank), min(n_rows, n_cols)))
    q_dim = min(min(n_rows, n_cols), rank + min(4, rank))
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    omega = torch.randn(n_cols, q_dim, dtype=matrix.dtype, generator=generator)
    sample = matrix @ omega
    for _ in range(max(0, int(power_iterations))):
        q = torch.linalg.qr(sample, mode="reduced").Q
        q_right = torch.linalg.qr(matrix.T @ q, mode="reduced").Q
        sample = matrix @ q_right
    q = torch.linalg.qr(sample, mode="reduced").Q
    singular = torch.linalg.svdvals(q.T @ matrix)[:rank]
    return singular.detach().cpu().to(torch.float64).numpy()


def _relative_error(actual: float, target: float) -> float:
    return float(abs(actual - target) / max(abs(target), 1e-12))


def _matching_thresholds(
    tolerances: Mapping[str, float] | None,
) -> dict[str, float]:
    defaults = {
        "hh_frobenius_relative": 1e-6,
        "hh_spectral_radius_relative": 0.50,
        "ih_frobenius_relative": 1e-6,
        "preactivation_variance_relative": 0.50,
        "relu_active_fraction_absolute": 0.10,
        "hidden_norm_relative": 0.50,
        "jacobian_frobenius_relative": 0.50,
        "top_singular_rmse_relative": 0.50,
    }
    supplied = dict(tolerances or {})
    unknown = sorted(set(supplied) - set(defaults))
    if unknown:
        raise ValueError(f"Unknown matched-random tolerance keys: {unknown}")
    return {**defaults, **{key: float(value) for key, value in supplied.items()}}


def _matching_gate(
    diagnostics: Mapping[str, Any],
    *,
    tolerances: Mapping[str, float] | None,
    strict: bool,
) -> dict[str, Any]:
    """Evaluate and optionally enforce the matched-random diagnostic gate.

    The gate uses only navigation weights and synthetic probes.  It never reads
    hard-task validation or test trajectories.  A top-level
    ``passed_matching_gate`` alias is installed by the caller for the CSV/JSON
    contracts used by the experiment wrappers.
    """
    thresholds = _matching_thresholds(tolerances)
    errors = diagnostics.get("matching_error")
    if not isinstance(errors, Mapping):
        raise ValueError("Matched-random diagnostics lack matching_error")
    checks: dict[str, dict[str, Any]] = {}
    for name, threshold in thresholds.items():
        if name not in errors:
            raise ValueError(f"Matched-random diagnostics lack error metric {name!r}")
        value = float(errors[name])
        passed = bool(np.isfinite(value) and value <= threshold)
        checks[name] = {
            "value": value,
            "maximum": float(threshold),
            "passed": passed,
        }
    passed_all = all(check["passed"] for check in checks.values())
    gate = {
        "passed": bool(passed_all),
        "strict": bool(strict),
        "selection_uses_target_data": False,
        "checks": checks,
    }
    if strict and not passed_all:
        failures = {
            name: check
            for name, check in checks.items()
            if not check["passed"]
        }
        raise RuntimeError(
            "Matched-random construction failed the pre-registered weight-matching "
            f"gate: {failures}"
        )
    return gate


def _probe_dynamics_stats(
    hh: torch.Tensor,
    ih: torch.Tensor,
    probe_h: torch.Tensor,
    probe_u: torch.Tensor,
) -> dict[str, float]:
    pre = probe_h @ hh.T + probe_u @ ih.T
    activated = torch.relu(pre)
    return {
        "preactivation_variance": float(pre.var(unbiased=False).item()),
        "relu_active_fraction": float((pre > 0).to(torch.float64).mean().item()),
        "hidden_norm_mean": float(activated.norm(dim=1).mean().item()),
        "one_step_jacobian_frobenius": float(
            torch.sqrt((pre > 0).to(hh.dtype).mean(dim=0) @ hh.square().sum(dim=1)).item()
        ),
    }


def _weight_diagnostics(
    candidate_hh: torch.Tensor,
    candidate_ih: torch.Tensor,
    target_hh: torch.Tensor,
    target_ih: torch.Tensor,
    *,
    seed: int,
    singular_rank: int,
    probe_count: int,
    known_hh_singular_scale: float | None = None,
    exact_hh_similarity: bool = False,
    exact_ih_left_orthogonal: bool = False,
) -> dict[str, Any]:
    dtype = target_hh.dtype
    generator = torch.Generator(device="cpu").manual_seed(int(seed) + 991)
    hidden_dim = target_hh.shape[0]
    probe_h = torch.relu(torch.randn(probe_count, hidden_dim, generator=generator, dtype=dtype))
    probe_u = torch.randn(probe_count, 2, generator=generator, dtype=dtype)

    target_radius = _spectral_radius_power(
        target_hh, iterations=16, seed=seed, n_probes=2
    )

    def stats(hh: torch.Tensor, ih: torch.Tensor, *, candidate: bool) -> dict[str, Any]:
        dynamic = _probe_dynamics_stats(hh, ih, probe_h, probe_u)
        if candidate and exact_hh_similarity:
            radius = target_radius
        else:
            radius = _spectral_radius_power(
                hh, iterations=16, seed=seed, n_probes=2
            )
        return {
            "hh_frobenius": float(hh.norm().item()),
            "hh_spectral_radius_estimate": float(radius),
            "hh_row_norm_quantiles": np.quantile(
                hh.norm(dim=1).detach().cpu().to(torch.float64).numpy(),
                [0, .1, .25, .5, .75, .9, 1],
            ).tolist(),
            "hh_column_norm_quantiles": np.quantile(
                hh.norm(dim=0).detach().cpu().to(torch.float64).numpy(),
                [0, .1, .25, .5, .75, .9, 1],
            ).tolist(),
            "ih_frobenius": float(ih.norm().item()),
            "ih_singular_values": torch.linalg.svdvals(ih).detach().cpu().to(torch.float64).numpy().tolist(),
            **dynamic,
        }

    target = stats(target_hh, target_ih, candidate=False)
    candidate = stats(candidate_hh, candidate_ih, candidate=True)
    if exact_hh_similarity:
        singular_error = 0.0
        target["hh_top_singular_values"] = None
        candidate["hh_top_singular_values"] = None
        candidate["hh_singular_spectrum_relationship"] = (
            "exact orthogonal signed-permutation similarity; complete singular "
            "and eigenvalue spectra equal navigation W_hh"
        )
        candidate["hh_eigenvalue_spectrum_exactly_preserved"] = True
        candidate["hh_spectral_radius_exactly_preserved"] = True
    elif known_hh_singular_scale is not None:
        scale = float(known_hh_singular_scale)
        singular_error = abs(scale - 1.0)
        target["hh_top_singular_values"] = None
        candidate["hh_top_singular_values"] = None
        candidate["hh_singular_spectrum_relationship"] = (
            "known complete singular spectrum scale relative to navigation: "
            f"{scale:.12g}"
        )
        candidate["hh_singular_spectrum_scale"] = scale
    else:
        target_singular = _top_singular_values(target_hh, rank=singular_rank, seed=seed)
        candidate_singular = _top_singular_values(candidate_hh, rank=singular_rank, seed=seed + 1)
        singular_error = float(
            np.sqrt(np.mean((candidate_singular - target_singular) ** 2))
            / max(float(np.mean(target_singular)), 1e-12)
        )
        target["hh_top_singular_values"] = target_singular.tolist()
        candidate["hh_top_singular_values"] = candidate_singular.tolist()
        candidate["hh_singular_spectrum_relationship"] = "estimated independently"

    hh_frobenius_error = (
        0.0
        if exact_hh_similarity
        else _relative_error(candidate["hh_frobenius"], target["hh_frobenius"])
    )
    hh_radius_error = (
        0.0
        if exact_hh_similarity
        else _relative_error(
            candidate["hh_spectral_radius_estimate"],
            target["hh_spectral_radius_estimate"],
        )
    )
    ih_frobenius_error = (
        0.0
        if exact_ih_left_orthogonal
        else _relative_error(candidate["ih_frobenius"], target["ih_frobenius"])
    )
    error = {
        "hh_frobenius_relative": float(hh_frobenius_error),
        "hh_spectral_radius_relative": float(hh_radius_error),
        "ih_frobenius_relative": float(ih_frobenius_error),
        "preactivation_variance_relative": _relative_error(candidate["preactivation_variance"], target["preactivation_variance"]),
        "relu_active_fraction_absolute": abs(candidate["relu_active_fraction"] - target["relu_active_fraction"]),
        "hidden_norm_relative": _relative_error(candidate["hidden_norm_mean"], target["hidden_norm_mean"]),
        "jacobian_frobenius_relative": _relative_error(candidate["one_step_jacobian_frobenius"], target["one_step_jacobian_frobenius"]),
        "top_singular_rmse_relative": float(singular_error),
    }
    if exact_ih_left_orthogonal:
        candidate["ih_singular_spectrum_relationship"] = (
            "exact shared signed-permutation left transform; W_ih column Gram "
            "matrix and action-direction geometry equal navigation W_ih"
        )
        candidate["ih_column_gram_exactly_preserved"] = True
    return {"target_navigation": target, "matched_random": candidate, "matching_error": error}


def _randomization_fractions(hidden_dim: int, n_candidates: int) -> list[float]:
    """Descending sign-flip fractions with a non-degenerate low-strength anchor."""
    n_candidates = max(1, int(n_candidates))
    minimum = max(8.0 / max(int(hidden_dim), 1), 0.002)
    if n_candidates == 1:
        return [minimum]
    return [float(value) for value in np.geomspace(0.50, minimum, num=n_candidates)]


def _signed_similarity_candidate(
    nav_hh: torch.Tensor,
    nav_ih: torch.Tensor,
    *,
    seed: int,
    sign_flip_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Randomize ReLU axes while preserving all linear operator invariants.

    For Q = D P, where P is a permutation and D is a random sign matrix,
    W_hh' = Q W_hh Q^T is an exact orthogonal similarity and W_ih' = Q W_ih.
    Thus the complete eigenvalue/singular spectra, spectral radius, Frobenius
    norms, row/column norm multisets, and input column Gram matrix are exactly
    preserved.  Non-zero sign flips change the coordinatewise ReLU active-set
    geometry, so this is not a nonlinear conjugacy of the original RNN.
    """
    rng = np.random.default_rng(int(seed))
    hidden_dim = int(nav_hh.shape[0])
    permutation = torch.as_tensor(rng.permutation(hidden_dim), dtype=torch.long)
    flip_count = max(1, min(hidden_dim, int(round(float(sign_flip_fraction) * hidden_dim))))
    flipped = rng.choice(hidden_dim, size=flip_count, replace=False)
    signs_np = np.ones(hidden_dim, dtype=np.float64)
    signs_np[flipped] = -1.0
    signs = torch.as_tensor(signs_np, dtype=nav_hh.dtype)

    hh = nav_hh.index_select(0, permutation).index_select(1, permutation)
    hh = signs[:, None] * hh * signs[None, :]
    ih = nav_ih.index_select(0, permutation)
    ih = signs[:, None] * ih
    realized_fraction = float(flip_count / hidden_dim)
    return hh.contiguous(), ih.contiguous(), {
        "construction": "relu_axis_randomized_signed_similarity",
        "seed": int(seed),
        "requested_sign_flip_fraction": float(sign_flip_fraction),
        "sign_flip_count": int(flip_count),
        "realized_sign_flip_fraction": realized_fraction,
        "linear_similarity_transform": True,
        "nonlinear_relu_conjugacy": False,
        "relu_axis_randomized": True,
        "eigenvalue_spectrum_exactly_preserved": True,
        "spectral_radius_exactly_preserved": True,
        "singular_spectrum_exactly_preserved": True,
        "frobenius_norm_exactly_preserved": True,
        "row_column_norm_multisets_exactly_preserved": True,
        "input_column_gram_exactly_preserved": True,
    }


def make_matched_random_weights(
    nav_hh: torch.Tensor,
    nav_ih: torch.Tensor,
    *,
    seed: int,
    iterations: int = 8,
    singular_rank: int = 16,
    probe_count: int = 64,
    tolerances: Mapping[str, float] | None = None,
    strict: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Construct a ReLU-axis-randomized, linearly invariant control.

    Candidate selection uses only synthetic probes and never hard-task
    validation/test trajectories.  Candidates are ordered from stronger to
    weaker sign randomization.  The strongest candidate that passes an
    independent synthetic-probe audit is selected.
    """
    if nav_hh.ndim != 2 or nav_hh.shape[0] != nav_hh.shape[1]:
        raise ValueError("nav_hh must be square")
    if nav_ih.shape != (nav_hh.shape[0], 2):
        raise ValueError("nav_ih must have shape [H,2]")

    thresholds = _matching_thresholds(tolerances)
    generator = torch.Generator(device="cpu").manual_seed(int(seed) + 23)
    selection_probe_count = max(4, min(int(probe_count), 16))
    probe_h = torch.relu(torch.randn(
        selection_probe_count, nav_hh.shape[0], generator=generator, dtype=nav_hh.dtype
    ))
    probe_u = torch.randn(selection_probe_count, 2, generator=generator, dtype=nav_hh.dtype)
    target_stats = _probe_dynamics_stats(nav_hh, nav_ih, probe_h, probe_u)

    n_candidates = max(1, min(int(iterations), 16))
    fractions = _randomization_fractions(int(nav_hh.shape[0]), n_candidates)
    candidate_rows: list[dict[str, Any]] = []
    for index, fraction in enumerate(fractions):
        candidate_seed = int(seed) + 1009 * index
        hh, ih, metadata = _signed_similarity_candidate(
            nav_hh, nav_ih, seed=candidate_seed, sign_flip_fraction=fraction
        )
        stats = _probe_dynamics_stats(hh, ih, probe_h, probe_u)
        errors = {
            "hh_frobenius_relative": 0.0,
            "hh_spectral_radius_relative": 0.0,
            "ih_frobenius_relative": 0.0,
            "preactivation_variance_relative": _relative_error(
                stats["preactivation_variance"], target_stats["preactivation_variance"]
            ),
            "relu_active_fraction_absolute": abs(
                stats["relu_active_fraction"] - target_stats["relu_active_fraction"]
            ),
            "hidden_norm_relative": _relative_error(
                stats["hidden_norm_mean"], target_stats["hidden_norm_mean"]
            ),
            "jacobian_frobenius_relative": _relative_error(
                stats["one_step_jacobian_frobenius"],
                target_stats["one_step_jacobian_frobenius"],
            ),
            "top_singular_rmse_relative": 0.0,
        }
        normalized = [errors[name] / max(thresholds[name], 1e-12) for name in errors]
        calibration_passed = all(errors[name] <= thresholds[name] for name in errors)
        row = {
            "candidate_index": int(index),
            "candidate_seed": int(candidate_seed),
            "selection_score": float(np.mean(normalized)),
            "calibration_passed": bool(calibration_passed),
            **metadata,
            **errors,
        }
        candidate_rows.append(row)

    calibration_passed_rows = [row for row in candidate_rows if row["calibration_passed"]]
    ordered_rows = sorted(
        calibration_passed_rows,
        key=lambda row: (-float(row["realized_sign_flip_fraction"]), float(row["selection_score"])),
    )
    ordered_rows.extend(
        sorted(
            [row for row in candidate_rows if not row["calibration_passed"]],
            key=lambda row: float(row["selection_score"]),
        )
    )

    selected: tuple[torch.Tensor, torch.Tensor, dict[str, Any], dict[str, Any]] | None = None
    audit_rows: list[dict[str, Any]] = []
    best_failed: tuple[torch.Tensor, torch.Tensor, dict[str, Any], dict[str, Any]] | None = None
    best_failed_score = float("inf")
    for row in ordered_rows:
        hh, ih, metadata = _signed_similarity_candidate(
            nav_hh,
            nav_ih,
            seed=int(row["candidate_seed"]),
            sign_flip_fraction=float(row["requested_sign_flip_fraction"]),
        )
        diagnostics = _weight_diagnostics(
            hh,
            ih,
            nav_hh,
            nav_ih,
            seed=int(seed) + 211,
            singular_rank=singular_rank,
            probe_count=probe_count,
            exact_hh_similarity=True,
            exact_ih_left_orthogonal=True,
        )
        gate = _matching_gate(diagnostics, tolerances=tolerances, strict=False)
        audit_score = float(np.mean([
            check["value"] / max(check["maximum"], 1e-12)
            for check in gate["checks"].values()
        ]))
        audit_rows.append({
            "candidate_index": int(row["candidate_index"]),
            "candidate_seed": int(row["candidate_seed"]),
            "realized_sign_flip_fraction": float(metadata["realized_sign_flip_fraction"]),
            "passed": bool(gate["passed"]),
            "audit_score": audit_score,
            "checks": gate["checks"],
        })
        bundle = (hh, ih, metadata, diagnostics)
        if gate["passed"]:
            selected = bundle
            break
        if audit_score < best_failed_score:
            best_failed = bundle
            best_failed_score = audit_score

    if selected is None:
        if best_failed is None:
            raise RuntimeError("Matched-random construction generated no candidates")
        hh, ih, construction, diagnostics = best_failed
        diagnostics["construction"] = construction
        diagnostics["candidate_selection"] = {
            "n_candidates": n_candidates,
            "rows": candidate_rows,
            "independent_audit_rows": audit_rows,
            "used_source_test": False,
            "probe_scope": "synthetic training-independent probes",
            "selection_rule": "strongest sign randomization passing independent audit",
        }
        diagnostics["matching_gate"] = _matching_gate(
            diagnostics, tolerances=tolerances, strict=strict
        )
        diagnostics["passed_matching_gate"] = bool(diagnostics["matching_gate"]["passed"])
        return hh.to(nav_hh.dtype), ih.to(nav_ih.dtype), diagnostics

    hh, ih, construction, diagnostics = selected
    diagnostics["construction"] = construction
    diagnostics["candidate_selection"] = {
        "n_candidates": n_candidates,
        "rows": candidate_rows,
        "independent_audit_rows": audit_rows,
        "used_source_test": False,
        "probe_scope": "synthetic training-independent probes",
        "selection_rule": "strongest sign randomization passing independent audit",
    }
    diagnostics["matching_gate"] = _matching_gate(
        diagnostics, tolerances=tolerances, strict=strict
    )
    diagnostics["passed_matching_gate"] = bool(diagnostics["matching_gate"]["passed"])
    return hh.to(nav_hh.dtype), ih.to(nav_ih.dtype), diagnostics


def spectrum_matched_shuffled_hh(nav_hh: torch.Tensor, *, seed: int) -> torch.Tensor:
    """Signed-permutation similarity control preserving exact eigen/singular spectra."""
    matrix = nav_hh.detach().cpu()
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    permutation = torch.randperm(matrix.shape[0], generator=generator)
    signs = torch.where(
        torch.rand(matrix.shape[0], generator=generator) > 0.5,
        torch.ones(matrix.shape[0]), -torch.ones(matrix.shape[0]),
    ).to(matrix.dtype)
    # Q W Q^T for Q = diag(signs) P.
    shuffled = matrix[permutation][:, permutation]
    return signs[:, None] * shuffled * signs[None, :]


def mismatched_input_basis(nav_ih: torch.Tensor, *, seed: int) -> torch.Tensor:
    """Preserve W_ih singular values while breaking its hidden-unit basis.

    An independent signed row permutation leaves the two-column singular
    spectrum and Frobenius norm exactly unchanged.  Keeping W_hh in the
    original navigation basis therefore isolates input--recurrent basis
    alignment without changing input scale or rank.
    """
    matrix = nav_ih.detach().cpu()
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    permutation = torch.randperm(matrix.shape[0], generator=generator)
    signs = torch.where(
        torch.rand(matrix.shape[0], generator=generator) > 0.5,
        torch.ones(matrix.shape[0]),
        -torch.ones(matrix.shape[0]),
    ).to(matrix.dtype)
    return signs[:, None] * matrix[permutation]


def weight_pair_diagnostics(
    candidate_hh: torch.Tensor,
    candidate_ih: torch.Tensor,
    target_hh: torch.Tensor,
    target_ih: torch.Tensor,
    *,
    seed: int,
    singular_rank: int = 16,
    probe_count: int = 64,
    known_hh_singular_scale: float | None = None,
) -> dict[str, Any]:
    """Public diagnostics for optional non-factorial reservoir controls."""
    return _weight_diagnostics(
        candidate_hh.detach().cpu().to(target_hh.dtype),
        candidate_ih.detach().cpu().to(target_ih.dtype),
        target_hh.detach().cpu(),
        target_ih.detach().cpu(),
        seed=int(seed),
        singular_rank=int(singular_rank),
        probe_count=int(probe_count),
        known_hh_singular_scale=known_hh_singular_scale,
    )


def install_custom_weights(
    model: torch.nn.Module,
    *,
    hh: torch.Tensor,
    ih: torch.Tensor,
    hh_trainability: str,
    ih_trainability: str,
) -> None:
    if hh_trainability not in TRAINABILITY or ih_trainability not in TRAINABILITY:
        raise ValueError("Custom control trainability must be frozen or trainable")
    params = dict(model.grid_rnn.named_parameters())
    if tuple(params["cell.h_proj.weight"].shape) != tuple(hh.shape):
        raise ValueError("W_hh shape does not match target model")
    if tuple(params["cell.in_proj.weight"].shape) != tuple(ih.shape):
        raise ValueError("W_ih shape does not match target model")
    with torch.no_grad():
        params["cell.h_proj.weight"].copy_(hh.to(params["cell.h_proj.weight"].dtype))
        params["cell.in_proj.weight"].copy_(ih.to(params["cell.in_proj.weight"].dtype))
    model.set_grid_trainability(
        recurrent_trainable=hh_trainability == "trainable",
        input_trainable=ih_trainability == "trainable",
    )


def install_factorial_weights(
    model: torch.nn.Module,
    *,
    cell: FactorialCell,
    nav_hh: torch.Tensor,
    nav_ih: torch.Tensor,
    random_hh: torch.Tensor,
    random_ih: torch.Tensor,
) -> None:
    hh = nav_hh if cell.hh_origin == "navigation" else random_hh
    ih = nav_ih if cell.ih_origin == "navigation" else random_ih
    install_custom_weights(
        model,
        hh=hh,
        ih=ih,
        hh_trainability=cell.hh_trainability,
        ih_trainability=cell.ih_trainability,
    )


def factorial_contrast_table(
    rows: list[Mapping[str, Any]],
    *,
    endpoint: str,
    seed_key: str = "target_seed",
    bootstrap_samples: int = 5000,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Compute balanced 2^4 within-seed factorial contrasts and paired CIs."""
    factor_names = ("hh_origin", "ih_origin", "hh_trainability", "ih_trainability")
    positive = {
        "hh_origin": "navigation",
        "ih_origin": "navigation",
        "hh_trainability": "trainable",
        "ih_trainability": "trainable",
    }
    seeds = sorted({int(row[seed_key]) for row in rows})
    expected_cells = {cell.label for cell in all_factorial_cells()}
    per_seed: dict[int, dict[str, Mapping[str, Any]]] = {}
    for target_seed in seeds:
        selected = [row for row in rows if int(row[seed_key]) == target_seed]
        mapping = {str(row["cell"]): row for row in selected}
        if set(mapping) != expected_cells:
            missing = sorted(expected_cells - set(mapping))
            raise RuntimeError(
                f"Balanced factorial contrast requires all 16 cells for seed {target_seed}; missing={missing}"
            )
        per_seed[target_seed] = mapping
    rng = np.random.default_rng(int(seed))
    output: list[dict[str, Any]] = []
    from itertools import combinations
    for order in range(1, 5):
        for factors in combinations(factor_names, order):
            effects = []
            for target_seed in seeds:
                signed = []
                for cell in all_factorial_cells():
                    sign = 1.0
                    cell_dict = cell.as_dict()
                    for factor in factors:
                        sign *= 1.0 if cell_dict[factor] == positive[factor] else -1.0
                    signed.append(sign * float(per_seed[target_seed][cell.label][endpoint]))
                effects.append(float(np.mean(signed) * 2.0))
            effects_arr = np.asarray(effects, dtype=np.float64)
            if effects_arr.size > 1 and bootstrap_samples > 0:
                boot = np.empty(int(bootstrap_samples), dtype=np.float64)
                for index in range(int(bootstrap_samples)):
                    boot[index] = rng.choice(effects_arr, size=effects_arr.size, replace=True).mean()
                ci_low, ci_high = np.quantile(boot, [0.025, 0.975])
                # Exact sign-flip when feasible, Monte Carlo otherwise.
                if effects_arr.size <= 16:
                    values = []
                    for mask in range(1 << effects_arr.size):
                        signs = np.asarray([1 if mask & (1 << i) else -1 for i in range(effects_arr.size)])
                        values.append(abs(float(np.mean(signs * effects_arr))))
                    p_value = float((np.sum(np.asarray(values) >= abs(effects_arr.mean())) + 1) / (len(values) + 1))
                else:
                    null = np.empty(int(bootstrap_samples), dtype=np.float64)
                    for index in range(int(bootstrap_samples)):
                        signs = rng.choice([-1.0, 1.0], size=effects_arr.size)
                        null[index] = abs(float(np.mean(signs * effects_arr)))
                    p_value = float((np.sum(null >= abs(effects_arr.mean())) + 1) / (null.size + 1))
            else:
                ci_low = ci_high = float(effects_arr.mean())
                p_value = None
            output.append({
                "endpoint": endpoint,
                "contrast": " × ".join(factors),
                "order": order,
                "n_target_seeds": len(seeds),
                "mean_effect": float(effects_arr.mean()),
                "sd_effect": float(effects_arr.std(ddof=1)) if effects_arr.size > 1 else 0.0,
                "ci95_low": float(ci_low),
                "ci95_high": float(ci_high),
                "paired_sign_flip_p": p_value,
                "per_seed_effects": effects,
                "claim_scope": "conditional on the analyzed navigation source",
            })
    return output


def paired_condition_comparisons(
    rows: list[Mapping[str, Any]],
    *,
    endpoint: str,
    comparisons: list[Mapping[str, str]],
    seed_key: str = "target_seed",
    bootstrap_samples: int = 5000,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Paired target-seed comparisons for confirmatory key conditions."""
    by_seed: dict[int, dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        by_seed.setdefault(int(row[seed_key]), {})[str(row["cell"])] = row
    rng = np.random.default_rng(int(seed))
    output: list[dict[str, Any]] = []
    for comparison in comparisons:
        name = str(comparison["name"])
        left = str(comparison["left"])
        right = str(comparison["right"])
        missing = {
            target_seed: [label for label in (left, right) if label not in mapping]
            for target_seed, mapping in by_seed.items()
            if left not in mapping or right not in mapping
        }
        if missing:
            raise RuntimeError(
                f"Confirmatory comparison {name!r} lacks paired cells: {missing}"
            )
        differences = np.asarray(
            [
                float(mapping[left][endpoint]) - float(mapping[right][endpoint])
                for _target_seed, mapping in sorted(by_seed.items())
            ],
            dtype=np.float64,
        )
        mean_difference = float(differences.mean())
        if differences.size > 1 and int(bootstrap_samples) > 0:
            bootstrap = np.empty(int(bootstrap_samples), dtype=np.float64)
            for index in range(int(bootstrap_samples)):
                bootstrap[index] = rng.choice(
                    differences, size=differences.size, replace=True
                ).mean()
            ci_low, ci_high = np.quantile(bootstrap, [0.025, 0.975])
            if differences.size <= 16:
                null = []
                for mask in range(1 << differences.size):
                    signs = np.asarray(
                        [1.0 if mask & (1 << i) else -1.0 for i in range(differences.size)]
                    )
                    null.append(abs(float(np.mean(signs * differences))))
                p_value = float(
                    (np.sum(np.asarray(null) >= abs(mean_difference)) + 1)
                    / (len(null) + 1)
                )
            else:
                null = np.empty(int(bootstrap_samples), dtype=np.float64)
                for index in range(int(bootstrap_samples)):
                    signs = rng.choice([-1.0, 1.0], size=differences.size)
                    null[index] = abs(float(np.mean(signs * differences)))
                p_value = float(
                    (np.sum(null >= abs(mean_difference)) + 1) / (null.size + 1)
                )
        else:
            ci_low = ci_high = mean_difference
            p_value = None
        sd = float(differences.std(ddof=1)) if differences.size > 1 else 0.0
        output.append({
            "endpoint": endpoint,
            "comparison": name,
            "left_condition": left,
            "right_condition": right,
            "n_target_seeds": int(differences.size),
            "mean_paired_difference": mean_difference,
            "sd_paired_difference": sd,
            "standardized_paired_effect_dz": (
                mean_difference / sd if sd > 1e-12 else None
            ),
            "ci95_low": float(ci_low),
            "ci95_high": float(ci_high),
            "paired_sign_flip_p": p_value,
            "per_seed_differences": differences.tolist(),
            "claim_scope": "conditional on the analyzed navigation source",
        })

    # Holm adjustment is restricted to this endpoint/comparison family.
    valid = [
        (index, float(row["paired_sign_flip_p"]))
        for index, row in enumerate(output)
        if row["paired_sign_flip_p"] is not None
    ]
    ordered = sorted(valid, key=lambda item: item[1])
    adjusted: dict[int, float] = {}
    running = 0.0
    family_size = len(ordered)
    for rank, (index, p_value) in enumerate(ordered):
        candidate = min(1.0, (family_size - rank) * p_value)
        running = max(running, candidate)
        adjusted[index] = running
    for index, row in enumerate(output):
        row["holm_adjusted_p"] = adjusted.get(index)
    return output
