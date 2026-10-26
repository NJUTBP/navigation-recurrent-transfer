"""Evaluation metrics for dedicated same-endpoint path groups (stage I3C)."""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, r2_score

from gpm_cmap.analysis.action_causality import phase_of_hidden
from gpm_cmap.analysis.source_reduced_v3 import wrap_angles
from gpm_cmap.data.graph_tasks.path_invariance import PathInvariantDataset, VARIANT_NAMES
from gpm_cmap.data.graph_tasks.pathint_episodes import episode_fingerprint


def _pairwise_mean(values: np.ndarray, *, circular: bool = False) -> float:
    values = np.asarray(values, dtype=np.float64)
    distances = []
    for i in range(values.shape[0]):
        for j in range(i + 1, values.shape[0]):
            delta = wrap_angles(values[i] - values[j]) if circular else values[i] - values[j]
            distances.append(float(np.linalg.norm(delta)))
    return float(np.mean(distances)) if distances else 0.0


def _h0_for_group(model, group, device: torch.device) -> torch.Tensor:
    first = group.variants[0].episode
    start_node = torch.tensor([first.start_node], dtype=torch.long, device=device)
    if model.start_init == "coordinate":
        coords = torch.from_numpy(first.start_coords[None].astype(np.float32)).to(device)
        h0 = model.start_coord_proj(coords)
    else:
        h0 = model.start_proj(model.start_embed(start_node))
    if model.nonnegative:
        h0 = torch.relu(h0)
    return h0


def _residualize(train_features: np.ndarray, train_covariates: np.ndarray, test_features: np.ndarray, test_covariates: np.ndarray):
    train_design = np.column_stack([np.ones(train_covariates.shape[0]), train_covariates])
    test_design = np.column_stack([np.ones(test_covariates.shape[0]), test_covariates])
    coefficients, *_ = np.linalg.lstsq(train_design, train_features, rcond=None)
    return train_features - train_design @ coefficients, test_features - test_design @ coefficients


def evaluate_path_invariance(
    model: torch.nn.Module,
    source_model: Any,
    dataset: PathInvariantDataset,
    *,
    device: torch.device,
    seed: int,
    decoder_pca_dim: int = 64,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    model.eval()
    selected = np.asarray(source_model.selected_unit_indices, dtype=np.int64)
    variant_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    decoder_final_hidden = []
    decoder_labels = []
    decoder_groups = []
    decoder_covariates = []
    time_hidden = []
    time_coords = []
    time_targets_absolute = []
    time_targets_normalized = []
    time_groups = []

    for group in dataset.groups:
        with torch.no_grad():
            h0 = _h0_for_group(model, group, device)
        theta0 = phase_of_hidden(source_model, h0.cpu().numpy())[0]
        finals = []
        phases = []
        standardized = []
        source_final_phases = []
        outputs_by_variant = {}
        for variant_index, variant in enumerate(group.variants):
            episode = variant.episode
            actions = torch.from_numpy(episode.velocities[None].astype(np.float32)).to(device)
            with torch.no_grad():
                hidden_seq, hidden_final = model.grid_rnn(actions, h0=h0)
                logits = model.node_head(hidden_final)
                probabilities = torch.softmax(logits, dim=-1)[0]
                prediction = int(probabilities.argmax().item())
            hidden_np = hidden_seq[0].cpu().numpy()
            final_np = hidden_final.cpu().numpy()[0]
            phase_seq = phase_of_hidden(source_model, hidden_np)
            final_phase = phase_seq[-1]
            ambient_final = source_model.standardized(final_np[selected][None])[0]
            source_phase_path = source_model.phase_dynamics.rollout(theta0, episode.velocities)
            finals.append(final_np)
            phases.append(final_phase)
            standardized.append(ambient_final)
            source_final_phases.append(source_phase_path[-1])
            outputs_by_variant[variant.variant_name] = {
                "hidden_seq": hidden_np,
                "phase_seq": phase_seq,
                "final_hidden": final_np,
                "final_phase": final_phase,
            }
            variant_rows.append({
                "group_id": group.group_id,
                "variant_name": variant.variant_name,
                "trajectory_fingerprint": episode_fingerprint(episode),
                "path_length": int(episode.velocities.shape[0]),
                "start_node": int(episode.start_node),
                "target_node": int(episode.target_node),
                "predicted_node": prediction,
                "correct": int(prediction == episode.target_node),
                "target_probability": float(probabilities[episode.target_node].item()),
                "endpoint_phase_0": float(final_phase[0]),
                "endpoint_phase_1": float(final_phase[1]),
                "source_expected_phase_0": float(source_phase_path[-1, 0]),
                "source_expected_phase_1": float(source_phase_path[-1, 1]),
                "source_phase_law_error": float(np.linalg.norm(wrap_angles(final_phase - source_phase_path[-1]))),
            })
            decoder_final_hidden.append(final_np)
            decoder_labels.append(variant_index)
            decoder_groups.append(group.group_id)
            decoder_covariates.append([*group.start_xy, *group.end_xy, len(variant.action_labels)])
            coords = episode.path_coords[1:]
            absolute_time = np.arange(1, hidden_np.shape[0] + 1, dtype=np.float64)
            normalized_time = absolute_time / max(hidden_np.shape[0], 1)
            time_hidden.append(hidden_np)
            time_coords.append(coords)
            time_targets_absolute.append(absolute_time)
            time_targets_normalized.append(normalized_time)
            time_groups.append(np.full(hidden_np.shape[0], group.group_id, dtype=np.int64))

        phases_np = np.asarray(phases)
        standardized_np = np.asarray(standardized)
        source_phase_np = np.asarray(source_final_phases)
        shortest = outputs_by_variant["shortest_a"]
        padded = outputs_by_variant["a_plus_zero_padding"]
        loop_values = []
        for variant in group.variants:
            if variant.loop_before_step is None:
                continue
            output = outputs_by_variant[variant.variant_name]
            before_count = int(variant.loop_before_step)
            after_count = int(variant.loop_after_step)
            phase_before = theta0 if before_count == 0 else output["phase_seq"][before_count - 1]
            phase_after = output["phase_seq"][after_count - 1]
            loop_values.append(float(np.linalg.norm(wrap_angles(phase_after - phase_before))))
        group_accuracy = float(np.mean([
            row["correct"] for row in variant_rows if row["group_id"] == group.group_id
        ]))
        group_rows.append({
            "group_id": group.group_id,
            "start_x": group.start_xy[0], "start_y": group.start_xy[1],
            "end_x": group.end_xy[0], "end_y": group.end_xy[1],
            "endpoint_accuracy": group_accuracy,
            "endpoint_phase_dispersion": _pairwise_mean(phases_np, circular=True),
            "endpoint_hidden_dispersion_source_z": _pairwise_mean(standardized_np, circular=False) / max(np.sqrt(standardized_np.shape[1]), 1.0),
            "source_matched_endpoint_phase_dispersion": _pairwise_mean(source_phase_np, circular=True),
            "loop_holonomy": float(np.mean(loop_values)),
            "zero_padding_phase_difference": float(np.linalg.norm(wrap_angles(padded["final_phase"] - shortest["final_phase"]))),
            "zero_padding_hidden_difference_source_z": float(
                np.linalg.norm(
                    source_model.standardized(padded["final_hidden"][selected][None])[0]
                    - source_model.standardized(shortest["final_hidden"][selected][None])[0]
                ) / max(np.sqrt(source_model.standardizer.ambient_dim), 1.0)
            ),
        })

    # Decoder splits are group-level so paths from one endpoint group never leak.
    groups = np.asarray(decoder_groups, dtype=np.int64)
    unique_groups = np.unique(groups)
    rng = np.random.default_rng(int(seed))
    rng.shuffle(unique_groups)
    cut = max(1, min(len(unique_groups) - 1, int(round(0.7 * len(unique_groups)))))
    train_groups = set(unique_groups[:cut].tolist())
    train_mask = np.asarray([group in train_groups for group in groups])
    final_hidden = np.asarray(decoder_final_hidden, dtype=np.float64)
    labels = np.asarray(decoder_labels, dtype=np.int64)
    covariates = np.asarray(decoder_covariates, dtype=np.float64)
    pca_dim = max(2, min(int(decoder_pca_dim), final_hidden.shape[1], int(train_mask.sum()) - 1))
    pca = PCA(n_components=pca_dim, random_state=int(seed))
    train_latent = pca.fit_transform(final_hidden[train_mask])
    test_latent = pca.transform(final_hidden[~train_mask])
    train_residual, test_residual = _residualize(
        train_latent, covariates[train_mask], test_latent, covariates[~train_mask]
    )
    classifier = LogisticRegression(max_iter=2000, random_state=int(seed))
    classifier.fit(train_residual, labels[train_mask])
    path_accuracy = float(accuracy_score(labels[~train_mask], classifier.predict(test_residual)))

    hidden_time = np.concatenate(time_hidden)
    coords_time = np.concatenate(time_coords)
    targets_time_absolute = np.concatenate(time_targets_absolute)
    targets_time_normalized = np.concatenate(time_targets_normalized)
    groups_time = np.concatenate(time_groups)
    train_time = np.asarray([group in train_groups for group in groups_time])
    time_pca_dim = max(2, min(int(decoder_pca_dim), hidden_time.shape[1], int(train_time.sum()) - 1))
    time_pca = PCA(n_components=time_pca_dim, random_state=int(seed) + 1)
    train_time_latent = time_pca.fit_transform(hidden_time[train_time])
    test_time_latent = time_pca.transform(hidden_time[~train_time])
    train_time_resid, test_time_resid = _residualize(
        train_time_latent, coords_time[train_time], test_time_latent, coords_time[~train_time]
    )
    absolute_time_decoder = Ridge(alpha=1.0)
    absolute_time_decoder.fit(
        train_time_resid, targets_time_absolute[train_time]
    )
    absolute_time_prediction = absolute_time_decoder.predict(test_time_resid)
    absolute_time_r2 = float(
        r2_score(
            targets_time_absolute[~train_time], absolute_time_prediction
        )
    )
    normalized_time_decoder = Ridge(alpha=1.0)
    normalized_time_decoder.fit(
        train_time_resid, targets_time_normalized[train_time]
    )
    normalized_time_prediction = normalized_time_decoder.predict(test_time_resid)
    normalized_time_r2 = float(
        r2_score(
            targets_time_normalized[~train_time], normalized_time_prediction
        )
    )

    summary = {
        "n_groups": len(dataset.groups),
        "n_variants": len(variant_rows),
        "endpoint_accuracy": float(np.mean([row["endpoint_accuracy"] for row in group_rows])),
        "endpoint_phase_dispersion": float(np.mean([row["endpoint_phase_dispersion"] for row in group_rows])),
        "endpoint_hidden_dispersion_source_z": float(np.mean([row["endpoint_hidden_dispersion_source_z"] for row in group_rows])),
        "source_matched_endpoint_phase_dispersion": float(np.mean([row["source_matched_endpoint_phase_dispersion"] for row in group_rows])),
        "loop_holonomy": float(np.mean([row["loop_holonomy"] for row in group_rows])),
        "zero_padding_phase_difference": float(np.mean([row["zero_padding_phase_difference"] for row in group_rows])),
        "zero_padding_hidden_difference_source_z": float(np.mean([row["zero_padding_hidden_difference_source_z"] for row in group_rows])),
        "path_identity_decoder_accuracy": path_accuracy,
        "path_identity_decoder_chance": 1.0 / len(VARIANT_NAMES),
        # The absolute step index is the confirmatory clock metric because it
        # detects longer zero-action padding.  The normalized progress decoder
        # is retained as a secondary, length-invariant diagnostic.
        "residual_time_decoder_r2": absolute_time_r2,
        "residual_time_decoder_r2_absolute": absolute_time_r2,
        "residual_time_decoder_r2_normalized": normalized_time_r2,
        "decoder_train_groups": len(train_groups),
        "decoder_test_groups": len(unique_groups) - len(train_groups),
    }
    return summary, group_rows, variant_rows
