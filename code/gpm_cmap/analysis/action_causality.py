"""Direct target-RNN action interventions for upgrade stage I3B."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from gpm_cmap.analysis.source_reduced_v3 import wrap_angles
from gpm_cmap.data.graph_tasks.pathint_episodes import episode_fingerprint

ACTION_TRANSFORMS = ("correct", "global_shuffle", "reverse", "sign_flip", "rotate_90", "axis_swap")


def transform_action_sequence(actions: np.ndarray, mode: str, *, seed: int = 0) -> np.ndarray:
    values = np.asarray(actions, dtype=np.float64).copy()
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError(f"actions must have shape [T,2]; got {values.shape}")
    if mode == "correct":
        return values
    if mode == "global_shuffle":
        return values[np.random.default_rng(int(seed)).permutation(values.shape[0])]
    if mode == "reverse":
        return values[::-1].copy()
    if mode == "sign_flip":
        return -values
    if mode == "rotate_90":
        rotation = np.asarray([[0.0, -1.0], [1.0, 0.0]])
        return values @ rotation.T
    if mode == "axis_swap":
        return values[:, ::-1]
    raise ValueError(f"Unknown action transform: {mode}")


def phase_of_hidden(source_model: Any, hidden: np.ndarray) -> np.ndarray:
    selected = np.asarray(source_model.selected_unit_indices, dtype=np.int64)
    hidden = np.asarray(hidden, dtype=np.float64)
    if hidden.ndim == 1:
        hidden = hidden.reshape(1, -1)
    if selected.size == 0 or int(selected.max()) >= hidden.shape[1]:
        raise ValueError(
            f"Source selected units exceed target hidden dimension: max={selected.max() if selected.size else None}, "
            f"target={hidden.shape[1]}"
        )
    return np.asarray(source_model.phase(hidden[:, selected]), dtype=np.float64)


def source_phase_delta(source_model: Any, theta: np.ndarray, action: np.ndarray) -> np.ndarray:
    return np.asarray(
        source_model.phase_dynamics.predict_delta(
            np.asarray(theta, dtype=np.float64).reshape(-1, 2),
            np.asarray(action, dtype=np.float64).reshape(-1, 2),
        ),
        dtype=np.float64,
    )


def recurrent_step(model: torch.nn.Module, hidden: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    model.eval()
    return model.grid_rnn.cell(action, hidden)


def circular_rmse(actual: np.ndarray, expected: np.ndarray) -> float:
    residual = wrap_angles(np.asarray(actual) - np.asarray(expected))
    return float(np.sqrt(np.mean(np.sum(residual * residual, axis=-1))))


def circular_r2(actual: np.ndarray, expected: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    residual = np.sum(wrap_angles(actual - expected) ** 2)
    center = np.arctan2(np.mean(np.sin(actual), axis=0), np.mean(np.cos(actual), axis=0))
    total = np.sum(wrap_angles(actual - center) ** 2)
    return float(1.0 - residual / max(total, 1e-12))


def phase_direction_labels(delta: np.ndarray, action_matrix: np.ndarray) -> np.ndarray:
    inverse = np.linalg.pinv(np.asarray(action_matrix, dtype=np.float64))
    velocity = np.asarray(delta, dtype=np.float64) @ inverse.T
    absolute = np.abs(velocity)
    axis = np.argmax(absolute, axis=1)
    sign = (velocity[np.arange(velocity.shape[0]), axis] < 0).astype(np.int64)
    labels = axis * 2 + sign
    labels[np.linalg.norm(velocity, axis=1) <= 1e-8] = 4
    return labels


def action_direction_labels(action: np.ndarray) -> np.ndarray:
    action = np.asarray(action, dtype=np.float64)
    absolute = np.abs(action)
    axis = np.argmax(absolute, axis=1)
    sign = (action[np.arange(action.shape[0]), axis] < 0).astype(np.int64)
    labels = axis * 2 + sign
    labels[np.linalg.norm(action, axis=1) <= 1e-8] = 4
    return labels


def _single_step_hidden_jacobian(model: torch.nn.Module, hidden: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    cell = model.grid_rnn.cell
    if getattr(model.grid_rnn, "cell_type", None) == "rnn":
        pre = cell.in_proj(action) + cell.h_proj(hidden)
        mask = (pre > 0).to(cell.in_proj.weight.dtype)
        return mask.unsqueeze(-1) * cell.in_proj.weight.unsqueeze(0)
    rows = []
    for index in range(hidden.shape[0]):
        h = hidden[index : index + 1].detach()
        u = action[index].detach().requires_grad_(True)
        jac = torch.autograd.functional.jacobian(lambda x: cell(x.unsqueeze(0), h).squeeze(0), u)
        rows.append(jac)
    return torch.stack(rows)


def evaluate_state_matched_actions(
    model: torch.nn.Module,
    source_model: Any,
    hidden_states: np.ndarray,
    *,
    base_action_scale: float,
    gains: Sequence[float],
    device: torch.device,
    finite_difference_eps: float = 1e-4,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Inject every counterfactual action into the same recurrent states.

    The RNN cell is actually evaluated for every state/action pair.  Primary
    action-specificity nulls are sign flip, 90-degree rotation and axis swap.
    Zero action is retained as a displacement-baseline diagnostic but is not a
    confirmatory null: for an actually injected zero action, all action mappings
    are identical and would force a meaningless null/correct ratio of one.
    """
    hidden_states = np.asarray(hidden_states, dtype=np.float32)
    if hidden_states.ndim != 2:
        raise ValueError("hidden_states must have shape [N,H]")
    cardinal = np.asarray(
        [[1, 0], [-1, 0], [0, 1], [0, -1], [0, 0]], dtype=np.float64
    ) * float(base_action_scale)
    h_tensor = torch.from_numpy(hidden_states).to(device)
    theta_before = phase_of_hidden(source_model, hidden_states)
    selected = np.asarray(source_model.selected_unit_indices, dtype=np.int64)
    rows: list[dict[str, Any]] = []
    actual_all: list[np.ndarray] = []
    expected_all: list[np.ndarray] = []
    action_all: list[np.ndarray] = []
    effect_ratios: list[np.ndarray] = []
    jacobian_norms: list[np.ndarray] = []
    phase_jacobians: list[np.ndarray] = []
    primary_null_names = ("sign_flip", "rotate_90", "axis_swap")
    primary_null_error_blocks: dict[str, list[np.ndarray]] = {
        name: [] for name in primary_null_names
    }
    zero_null_error_blocks: list[np.ndarray] = []

    for gain in gains:
        for action_index, base in enumerate(cardinal):
            action = base * float(gain)
            action_batch = np.repeat(action[None, :], hidden_states.shape[0], axis=0)
            u_tensor = torch.from_numpy(action_batch.astype(np.float32)).to(device)
            with torch.no_grad():
                next_hidden = recurrent_step(model, h_tensor, u_tensor).cpu().numpy()
            theta_after = phase_of_hidden(source_model, next_hidden)
            actual_delta = wrap_angles(theta_after - theta_before)
            expected_delta = source_phase_delta(source_model, theta_before, action_batch)
            actual_all.append(actual_delta)
            expected_all.append(expected_delta)
            action_all.append(action_batch)

            null_actions = {
                "zero": np.zeros_like(action_batch),
                "sign_flip": -action_batch,
                "rotate_90": np.column_stack(
                    [-action_batch[:, 1], action_batch[:, 0]]
                ),
                "axis_swap": action_batch[:, ::-1].copy(),
            }
            null_errors_this_condition: dict[str, np.ndarray] = {}
            for null_name, null_action_batch in null_actions.items():
                null_expected = source_phase_delta(
                    source_model, theta_before, null_action_batch
                )
                null_error = np.linalg.norm(
                    wrap_angles(actual_delta - null_expected), axis=1
                )
                null_errors_this_condition[null_name] = null_error
            zero_null_error_blocks.append(null_errors_this_condition["zero"])
            null_comparison_eligible = bool(np.linalg.norm(action) > 1e-12)
            if null_comparison_eligible:
                for null_name in primary_null_names:
                    primary_null_error_blocks[null_name].append(
                        null_errors_this_condition[null_name]
                    )

            expected_norm = np.linalg.norm(expected_delta, axis=1)
            ratio = np.linalg.norm(actual_delta, axis=1) / np.maximum(
                expected_norm, 1e-12
            )
            if null_comparison_eligible:
                effect_ratios.append(ratio)

            # Hidden J_u is evaluated on the actually injected state/action.
            with torch.enable_grad():
                jac = _single_step_hidden_jacobian(model, h_tensor, u_tensor)
            selected_jac = jac[:, selected, :]
            jac_norm = (
                torch.linalg.vector_norm(selected_jac, dim=(1, 2))
                .detach()
                .cpu()
                .numpy()
            )
            jacobian_norms.append(jac_norm)

            # Phase-projected J_u: finite difference through the actual RNN and
            # frozen phase encoder.
            phase_j = np.zeros((hidden_states.shape[0], 2, 2), dtype=np.float64)
            for axis in range(2):
                plus = action_batch.copy()
                plus[:, axis] += finite_difference_eps
                minus = action_batch.copy()
                minus[:, axis] -= finite_difference_eps
                with torch.no_grad():
                    h_plus = recurrent_step(
                        model,
                        h_tensor,
                        torch.from_numpy(plus.astype(np.float32)).to(device),
                    ).cpu().numpy()
                    h_minus = recurrent_step(
                        model,
                        h_tensor,
                        torch.from_numpy(minus.astype(np.float32)).to(device),
                    ).cpu().numpy()
                phase_plus = phase_of_hidden(source_model, h_plus)
                phase_minus = phase_of_hidden(source_model, h_minus)
                phase_j[:, :, axis] = wrap_angles(phase_plus - phase_minus) / (
                    2.0 * finite_difference_eps
                )
            phase_jacobians.append(phase_j)

            for state_index in range(hidden_states.shape[0]):
                residual = wrap_angles(
                    actual_delta[state_index] - expected_delta[state_index]
                )
                correct_error = float(np.linalg.norm(residual))
                if null_comparison_eligible:
                    eligible_null_errors = [
                        float(null_errors_this_condition[name][state_index])
                        for name in primary_null_names
                    ]
                    best_null_error: float | None = min(eligible_null_errors)
                    best_ratio: float | None = best_null_error / max(
                        correct_error, 1e-12
                    )
                else:
                    best_null_error = None
                    best_ratio = None
                rows.append({
                    "state_index": state_index,
                    "gain": float(gain),
                    "action_index": action_index,
                    "action_x": float(action[0]),
                    "action_y": float(action[1]),
                    "null_comparison_eligible": null_comparison_eligible,
                    "actual_phase_dx": float(actual_delta[state_index, 0]),
                    "actual_phase_dy": float(actual_delta[state_index, 1]),
                    "source_expected_phase_dx": float(expected_delta[state_index, 0]),
                    "source_expected_phase_dy": float(expected_delta[state_index, 1]),
                    "phase_error": correct_error,
                    "gain_ratio": (
                        float(ratio[state_index])
                        if null_comparison_eligible else None
                    ),
                    "zero_null_phase_error": float(
                        null_errors_this_condition["zero"][state_index]
                    ),
                    "sign_flip_null_phase_error": float(
                        null_errors_this_condition["sign_flip"][state_index]
                    ),
                    "rotate_90_null_phase_error": float(
                        null_errors_this_condition["rotate_90"][state_index]
                    ),
                    "axis_swap_null_phase_error": float(
                        null_errors_this_condition["axis_swap"][state_index]
                    ),
                    "best_primary_null_phase_error": best_null_error,
                    "best_primary_null_to_correct_error_ratio": best_ratio,
                    # Backward-readable aliases now refer only to eligible
                    # primary geometric nulls.
                    "best_null_phase_error": best_null_error,
                    "best_null_to_correct_error_ratio": best_ratio,
                    "hidden_input_jacobian_frobenius": float(
                        jac_norm[state_index]
                    ),
                    "phase_jacobian_00": float(phase_j[state_index, 0, 0]),
                    "phase_jacobian_01": float(phase_j[state_index, 0, 1]),
                    "phase_jacobian_10": float(phase_j[state_index, 1, 0]),
                    "phase_jacobian_11": float(phase_j[state_index, 1, 1]),
                })

    actual = np.concatenate(actual_all)
    expected = np.concatenate(expected_all)
    injected = np.concatenate(action_all)
    target_labels = action_direction_labels(injected)
    decoded_labels = phase_direction_labels(
        actual, source_model.phase_dynamics.action_matrix
    )
    nonzero = target_labels != 4
    expected_step_rms = float(
        np.sqrt(np.mean(np.sum(expected[nonzero] * expected[nonzero], axis=1)))
    ) if np.any(nonzero) else 0.0
    correct_rmse = circular_rmse(actual, expected)
    primary_null_rmse = {
        name: float(
            np.sqrt(np.mean(np.concatenate(blocks) ** 2))
        )
        for name, blocks in primary_null_error_blocks.items()
        if blocks
    }
    primary_null_ratios = {
        name: float(value / max(correct_rmse, 1e-12))
        for name, value in primary_null_rmse.items()
    }
    zero_null_rmse = float(
        np.sqrt(np.mean(np.concatenate(zero_null_error_blocks) ** 2))
    ) if zero_null_error_blocks else None
    minimum_primary_ratio = (
        float(min(primary_null_ratios.values()))
        if primary_null_ratios else None
    )
    effect_ratio_values = (
        np.concatenate(effect_ratios) if effect_ratios else np.empty(0)
    )
    summary = {
        "n_states": int(hidden_states.shape[0]),
        "n_interventions": int(actual.shape[0]),
        "n_primary_null_eligible_interventions": int(
            np.sum(nonzero)
        ),
        "gains": [float(value) for value in gains],
        "base_action_scale": float(base_action_scale),
        "circular_regression_r2": circular_r2(actual, expected),
        "phase_law_rmse": correct_rmse,
        "normalized_phase_law_rmse": float(
            correct_rmse / max(expected_step_rms, 1e-12)
        ),
        "phase_angle_error_mean": float(
            np.mean(np.linalg.norm(wrap_angles(actual - expected), axis=1))
        ),
        "gain_error_mean_absolute": (
            float(np.mean(np.abs(effect_ratio_values - 1.0)))
            if effect_ratio_values.size else None
        ),
        "source_target_effect_ratio_mean": (
            float(np.mean(effect_ratio_values))
            if effect_ratio_values.size else None
        ),
        "direction_classification_accuracy": (
            float(np.mean(decoded_labels[nonzero] == target_labels[nonzero]))
            if np.any(nonzero) else None
        ),
        "hidden_input_jacobian_frobenius_mean": float(
            np.mean(np.concatenate(jacobian_norms))
        ),
        "phase_projected_jacobian_frobenius_mean": float(
            np.mean([
                np.linalg.norm(row)
                for block in phase_jacobians
                for row in block
            ])
        ),
        "state_matched_primary_null_names": list(primary_null_names),
        "state_matched_null_phase_rmse": primary_null_rmse,
        "state_matched_zero_action_null_phase_rmse": zero_null_rmse,
        "state_matched_null_to_correct_rmse_ratio": primary_null_ratios,
        "minimum_state_matched_null_to_correct_rmse_ratio": minimum_primary_ratio,
        "all_state_matched_nulls_at_least_10pct_worse": bool(
            primary_null_ratios
            and all(value >= 1.10 for value in primary_null_ratios.values())
        ),
        "zero_action_excluded_from_confirmatory_null_gate": True,
    }
    return summary, rows

def evaluate_multistep_actions(
    model: torch.nn.Module,
    source_model: Any,
    hidden_states: np.ndarray,
    *,
    action_scale: float,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    hidden = torch.from_numpy(np.asarray(hidden_states, dtype=np.float32)).to(device)
    u = np.asarray([action_scale, 0.0], dtype=np.float32)
    v = np.asarray([0.0, action_scale], dtype=np.float32)

    def rollout(sequence: np.ndarray) -> np.ndarray:
        current = hidden
        with torch.no_grad():
            for action in sequence:
                batch = torch.from_numpy(np.repeat(action[None, :], hidden.shape[0], axis=0)).to(device)
                current = recurrent_step(model, current, batch)
        return current.cpu().numpy()

    h_uv = rollout(np.asarray([u, v]))
    h_vu = rollout(np.asarray([v, u]))
    h_u = rollout(np.asarray([u]))
    h_v = rollout(np.asarray([v]))
    h_zero = rollout(np.asarray([[0.0, 0.0]], dtype=np.float32))
    h_loop = rollout(np.asarray([u, v, -u, -v]))
    h_double = rollout(np.asarray([2 * u]))
    theta0 = phase_of_hidden(source_model, hidden_states)
    theta_u = phase_of_hidden(source_model, h_u)
    theta_v = phase_of_hidden(source_model, h_v)
    theta_uv = phase_of_hidden(source_model, h_uv)
    theta_vu = phase_of_hidden(source_model, h_vu)
    theta_loop = phase_of_hidden(source_model, h_loop)
    theta_double = phase_of_hidden(source_model, h_double)
    move_u = np.linalg.norm(h_u - hidden_states, axis=1)
    move_v = np.linalg.norm(h_v - hidden_states, axis=1)
    comm_hidden = np.linalg.norm(h_uv - h_vu, axis=1)
    comm_norm = comm_hidden / np.maximum(move_u + move_v, 1e-12)
    loop_phase = np.linalg.norm(wrap_angles(theta_loop - theta0), axis=1)
    zero_move = np.linalg.norm(h_zero - hidden_states, axis=1)
    additive = np.linalg.norm(wrap_angles((theta_uv - theta0) - wrap_angles(theta_u - theta0) - wrap_angles(theta_v - theta0)), axis=1)
    gain = np.linalg.norm(wrap_angles((theta_double - theta0) - 2.0 * wrap_angles(theta_u - theta0)), axis=1)
    rows = []
    for index in range(hidden_states.shape[0]):
        rows.append({
            "state_index": index,
            "commutator_hidden": float(comm_hidden[index]),
            "normalized_commutator": float(comm_norm[index]),
            "commutator_phase": float(np.linalg.norm(wrap_angles(theta_uv[index] - theta_vu[index]))),
            "square_loop_phase_return": float(loop_phase[index]),
            "square_loop_hidden_return": float(np.linalg.norm(h_loop[index] - hidden_states[index])),
            "nonzero_action_displacement": float(0.5 * (move_u[index] + move_v[index])),
            "zero_action_displacement": float(zero_move[index]),
            "composition_additivity_error": float(additive[index]),
            "gain_linearity_error": float(gain[index]),
        })
    summary = {
        key: float(np.mean([row[key] for row in rows]))
        for key in rows[0]
        if key != "state_index"
    }
    summary["nonzero_exceeds_zero_fraction"] = float(np.mean(0.5 * (move_u + move_v) > zero_move))
    return summary, rows


def evaluate_counterfactual_sequences(
    model: torch.nn.Module,
    source_model: Any,
    episodes: Sequence[Any],
    *,
    device: torch.device,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for trajectory_index, episode in enumerate(episodes):
        start_node = torch.tensor([episode.start_node], dtype=torch.long, device=device)
        start_coords = torch.from_numpy(np.asarray(episode.start_coords, dtype=np.float32)[None]).to(device)
        original = np.asarray(episode.velocities, dtype=np.float32)
        # Start encoder is called once; every counterfactual reruns the recurrent trajectory.
        with torch.no_grad():
            if model.start_init == "coordinate":
                h0 = model.start_coord_proj(start_coords)
            else:
                h0 = model.start_proj(model.start_embed(start_node))
            if model.nonnegative:
                h0 = torch.relu(h0)
        theta0 = phase_of_hidden(source_model, h0.cpu().numpy())[0]
        expected_correct = source_model.phase_dynamics.rollout(theta0, original)[-1]
        for mode in ACTION_TRANSFORMS:
            transformed = transform_action_sequence(original, mode, seed=seed + trajectory_index * 97)
            actions_t = torch.from_numpy(transformed.astype(np.float32)[None]).to(device)
            with torch.no_grad():
                hidden_seq, hidden_final = model.grid_rnn(actions_t, h0=h0)
                logits = model.node_head(hidden_final)
                prediction = int(logits.argmax(dim=-1).item())
                probability = torch.softmax(logits, dim=-1)[0]
            hidden_sequence_np = hidden_seq[0].cpu().numpy()
            phase_sequence = phase_of_hidden(source_model, hidden_sequence_np)
            theta_final = phase_sequence[-1]
            expected_injected_sequence = source_model.phase_dynamics.rollout(
                theta0, transformed
            )[1:]
            expected_correct_sequence = source_model.phase_dynamics.rollout(
                theta0, original
            )[1:]
            expected_for_input = expected_injected_sequence[-1]
            trajectory_residual_injected = wrap_angles(
                phase_sequence - expected_injected_sequence
            )
            trajectory_residual_correct = wrap_angles(
                phase_sequence - expected_correct_sequence
            )
            rows.append({
                "trajectory_index": trajectory_index,
                "trajectory_fingerprint": episode_fingerprint(episode),
                "mode": mode,
                "path_length": int(original.shape[0]),
                "target_node": int(episode.target_node),
                "predicted_node": prediction,
                "correct": int(prediction == int(episode.target_node)),
                "target_probability": float(probability[int(episode.target_node)].item()),
                "phase_trajectory_rmse_to_source_for_injected_actions": float(
                    np.sqrt(np.mean(np.sum(trajectory_residual_injected ** 2, axis=1)))
                ),
                "phase_trajectory_rmse_to_original_correct_law": float(
                    np.sqrt(np.mean(np.sum(trajectory_residual_correct ** 2, axis=1)))
                ),
                "phase_error_to_source_for_injected_actions": float(np.linalg.norm(wrap_angles(theta_final - expected_for_input))),
                "phase_error_to_original_correct_law": float(np.linalg.norm(wrap_angles(theta_final - expected_correct))),
                "endpoint_phase_displacement": float(np.linalg.norm(wrap_angles(theta_final - theta0))),
            })
    summary: dict[str, Any] = {"modes": {}}
    for mode in ACTION_TRANSFORMS:
        selected = [row for row in rows if row["mode"] == mode]
        summary["modes"][mode] = {
            "n_trajectories": len(selected),
            "accuracy": float(np.mean([row["correct"] for row in selected])),
            "mean_target_probability": float(np.mean([row["target_probability"] for row in selected])),
            "phase_trajectory_rmse_to_source_for_injected_actions": float(np.mean([row["phase_trajectory_rmse_to_source_for_injected_actions"] for row in selected])),
            "phase_trajectory_rmse_to_original_correct_law": float(np.mean([row["phase_trajectory_rmse_to_original_correct_law"] for row in selected])),
            "phase_error_to_source_for_injected_actions": float(np.mean([row["phase_error_to_source_for_injected_actions"] for row in selected])),
            "phase_error_to_original_correct_law": float(np.mean([row["phase_error_to_original_correct_law"] for row in selected])),
        }
    # Sequence-order nulls such as shuffle/reverse can have the same net
    # displacement as the correct path. Their causal mismatch is therefore
    # evaluated over the regenerated phase trajectory, not only at the final
    # endpoint. This avoids falsely penalising an ideal additive integrator.
    correct_error = summary["modes"]["correct"][
        "phase_trajectory_rmse_to_original_correct_law"
    ]
    summary["null_ratio_metric"] = (
        "phase_trajectory_rmse_to_original_correct_law"
    )
    summary["correct_to_null_error_ratios"] = {
        mode: float(
            summary["modes"][mode][
                "phase_trajectory_rmse_to_original_correct_law"
            ]
            / max(correct_error, 1e-12)
        )
        for mode in ACTION_TRANSFORMS if mode != "correct"
    }
    endpoint_correct_error = summary["modes"]["correct"][
        "phase_error_to_original_correct_law"
    ]
    summary["endpoint_correct_to_null_error_ratios"] = {
        mode: float(
            summary["modes"][mode]["phase_error_to_original_correct_law"]
            / max(endpoint_correct_error, 1e-12)
        )
        for mode in ACTION_TRANSFORMS if mode != "correct"
    }
    return summary, rows
