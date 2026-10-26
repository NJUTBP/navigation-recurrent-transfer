"""CTL-01: non-navigation delayed-reconstruction pretraining.

The transferable GridRNN architecture and optimization budget match the
navigation source.  The formal target is the input from ten steps earlier.
Shorter-lag reconstruction heads provide auxiliary supervision to make that
memory objective trainable under the bias-free ReLU GridRNN; only the
``grid_rnn`` tensors are transferred downstream.

The task loss is reported as normalized MSE (NMSE): squared prediction error
divided by the squared error of an all-zero predictor on the same targets.
This normalization is required because raw displacement MSE is around 1e-4,
whereas the navigation-matched ``1e-4 * ||W_hh||^2`` penalty is around 1e-1
at 4,096 hidden units.  Without normalization, optimization primarily shrinks
``W_hh`` and a superficially small raw MSE does not demonstrate memory.

CPU smoke test (uses a new output directory):

    python -m gpm_cmap.experiments.pretrain_sequence_control \
        --config gpm_cmap/configs/sequence_control_pretrain.yaml \
        --override model.hidden_dim=64 train.epochs=1 \
                   data.steps_per_epoch=2000 train.lr=0.001 \
                   analysis.val_num_sequences=500 \
                   device=cpu runtime.allow_device_fallback=true \
                   output_dir=outputs/sequence_control_smoketest_local

A usable formal source must pass ``training_gate.json`` and use hidden size
4,096.  Merely producing a checkpoint is not sufficient.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict, Iterable

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from gpm_cmap.data.nav_env.sequence_control import (
    SequenceControlConfig,
    SequenceControlOnlineDataset,
    SequenceControlStaticDataset,
    make_delay_targets,
)
from gpm_cmap.models.sequence_control_model import SequenceControlModel
from gpm_cmap.utils import (
    configure_runtime,
    load_config,
    make_worker_init_fn,
    prepare_output_dir,
    save_config_snapshot,
    save_run_provenance,
    set_seed,
    write_json,
)
from gpm_cmap.utils.checkpointing import load_checkpoint, save_checkpoint
from gpm_cmap.utils.logging_utils import get_logger


def _apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """Apply YAML-parsed ``section.key=value`` command-line overrides."""
    import yaml

    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must have the form key=value; got {item!r}")
        key_path, raw_value = item.split("=", 1)
        if not key_path:
            raise ValueError(f"Override key is empty in {item!r}")
        value = yaml.safe_load(raw_value)
        node = cfg
        parts = key_path.split(".")
        for part in parts[:-1]:
            child = node.setdefault(part, {})
            if not isinstance(child, dict):
                raise TypeError(
                    f"Cannot descend into non-mapping override component {part!r}"
                )
            node = child
        node[parts[-1]] = value
    return cfg


def _parse_auxiliary_lags(data_cfg: Dict[str, Any]) -> tuple[int, ...]:
    delay = int(data_cfg["delay"])
    raw = data_cfg.get("auxiliary_lags", [])
    if raw in (None, [], "none"):
        return ()
    if raw == "all_before_delay":
        return tuple(range(1, delay))
    if not isinstance(raw, Iterable) or isinstance(raw, (str, bytes)):
        raise TypeError(
            "data.auxiliary_lags must be a list of integers, 'all_before_delay', "
            "or an empty value"
        )
    lags = tuple(sorted({int(value) for value in raw}))
    invalid = [lag for lag in lags if lag < 1 or lag >= delay]
    if invalid:
        raise ValueError(
            f"Auxiliary lags must be between 1 and delay-1; invalid values: {invalid}"
        )
    return lags


def build_sequence_config(cfg: dict) -> SequenceControlConfig:
    dcfg = cfg["data"]
    return SequenceControlConfig(
        seq_len=int(dcfg["seq_len"]),
        dt=float(dcfg["dt"]),
        speed_scale=float(dcfg["speed_scale"]),
        sigma_theta=float(dcfg.get("sigma_theta", 11.52)),
        min_speed=float(dcfg.get("min_speed", 0.0)),
        max_speed=float(dcfg.get("max_speed", 100.0)),
        arena_size=float(dcfg.get("arena_size", 2.2)),
        border_region=float(dcfg.get("border_region", 0.03)),
        boundary_mode=str(dcfg.get("boundary_mode", "avoid")),
        delay=int(dcfg["delay"]),
        magnitude_mode=str(dcfg.get("magnitude_mode", "navigation_matched")),
        shuffle_magnitudes=bool(dcfg.get("shuffle_magnitudes", True)),
    )


def make_loaders(cfg: dict):
    seq_cfg = build_sequence_config(cfg)
    dcfg = cfg["data"]
    train_ds = SequenceControlOnlineDataset(
        seq_cfg,
        steps_per_epoch=int(dcfg["steps_per_epoch"]),
        batch_size=int(dcfg["batch_size"]),
        seed=int(cfg["seed"]),
    )
    train_loader = DataLoader(train_ds, batch_size=None, num_workers=0)

    validation_seed = int(cfg["seed"]) + 9999
    val_ds = SequenceControlStaticDataset(
        seq_cfg,
        num_sequences=int(cfg["analysis"].get("val_num_sequences", 2000)),
        seed=validation_seed,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(dcfg["batch_size"]),
        shuffle=False,
        num_workers=0,
        worker_init_fn=make_worker_init_fn(validation_seed),
    )
    return train_loader, val_loader, seq_cfg


def build_model(
    cfg: dict, device: torch.device, auxiliary_lags: tuple[int, ...]
) -> SequenceControlModel:
    mcfg = cfg["model"]
    model = SequenceControlModel(
        input_dim=int(mcfg.get("input_dim", 2)),
        hidden_dim=int(mcfg["hidden_dim"]),
        activation=str(mcfg.get("activation", "relu")),
        nonnegative=bool(mcfg.get("nonnegative", True)),
        cell_type=str(mcfg.get("cell_type", "rnn")),
        dropout=float(mcfg.get("dropout", 0.0)),
        use_bias=bool(mcfg.get("use_bias", False)),
        primary_delay=int(cfg["data"]["delay"]),
        auxiliary_lags=auxiliary_lags,
    )
    return model.to(device)


def _lag_nmse(
    full_prediction: torch.Tensor, inputs: torch.Tensor, lag: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return differentiable NMSE, raw MSE and zero-predictor MSE."""
    target, pred_slice = make_delay_targets(inputs, lag)
    prediction = full_prediction[:, pred_slice, :]
    raw_mse = F.mse_loss(prediction, target)
    zero_mse = target.square().mean().detach().clamp_min(1e-12)
    return raw_mse / zero_mse, raw_mse, zero_mse


def _task_objective(
    model: SequenceControlModel,
    outputs: Dict[str, object],
    inputs: torch.Tensor,
    *,
    primary_weight: float,
    auxiliary_weight_per_lag: float,
    task_loss_scale: float,
) -> tuple[torch.Tensor, Dict[int, Dict[str, torch.Tensor]]]:
    if primary_weight <= 0 or auxiliary_weight_per_lag < 0 or task_loss_scale <= 0:
        raise ValueError("Task-loss weights and scale must be positive/nonnegative")

    lag_metrics: Dict[int, Dict[str, torch.Tensor]] = {}
    weighted = torch.zeros((), device=inputs.device, dtype=inputs.dtype)
    total_weight = 0.0
    for lag, full_prediction in model.predictions_by_lag(outputs).items():
        nmse, raw_mse, zero_mse = _lag_nmse(full_prediction, inputs, lag)
        lag_metrics[lag] = {
            "normalized_mse": nmse,
            "raw_mse": raw_mse,
            "zero_mse": zero_mse,
        }
        weight = primary_weight if lag == model.primary_delay else auxiliary_weight_per_lag
        if weight > 0:
            weighted = weighted + float(weight) * nmse
            total_weight += float(weight)
    if total_weight <= 0:
        raise ValueError("At least one reconstruction loss must have positive weight")
    return task_loss_scale * weighted / total_weight, lag_metrics


def run_epoch_train(
    model: SequenceControlModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    whh_weight_decay: float,
    grad_clip: float,
    primary_weight: float,
    auxiliary_weight_per_lag: float,
    task_loss_scale: float,
    log_every_steps: int,
    device: torch.device,
    logger,
    epoch: int,
) -> Dict[str, float]:
    model.train()
    sums = {
        "objective": 0.0,
        "task_objective": 0.0,
        "primary_normalized_mse": 0.0,
        "primary_raw_mse": 0.0,
        "primary_zero_mse": 0.0,
        "whh_penalty": 0.0,
    }
    n_batches = 0
    for step, batch in enumerate(loader, start=1):
        inputs = batch["inputs"].to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        task_objective, lag_metrics = _task_objective(
            model,
            outputs,
            inputs,
            primary_weight=primary_weight,
            auxiliary_weight_per_lag=auxiliary_weight_per_lag,
            task_loss_scale=task_loss_scale,
        )
        whh_penalty = float(whh_weight_decay) * model.recurrent_l2()
        objective = task_objective + whh_penalty
        if not torch.isfinite(objective):
            raise FloatingPointError(
                f"Non-finite training objective at epoch={epoch}, step={step}"
            )
        objective.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        primary = lag_metrics[model.primary_delay]
        values = {
            "objective": objective,
            "task_objective": task_objective,
            "primary_normalized_mse": primary["normalized_mse"],
            "primary_raw_mse": primary["raw_mse"],
            "primary_zero_mse": primary["zero_mse"],
            "whh_penalty": whh_penalty,
        }
        for key, value in values.items():
            sums[key] += float(value.detach().cpu())
        n_batches += 1

        if log_every_steps > 0 and (
            step == 1 or step % log_every_steps == 0
        ):
            logger.info(
                "epoch %d step %d | objective=%.5f task=%.5f primary_NMSE=%.5f W_hh_penalty=%.5f",
                epoch,
                step,
                float(objective.detach().cpu()),
                float(task_objective.detach().cpu()),
                float(primary["normalized_mse"].detach().cpu()),
                float(whh_penalty.detach().cpu()),
            )

    if n_batches == 0:
        raise RuntimeError("Training loader yielded no batches")
    metrics = {key: value / n_batches for key, value in sums.items()}
    metrics["n_batches"] = float(n_batches)
    logger.info(
        "epoch %d train | objective=%.6f primary_NMSE=%.6f primary_raw_MSE=%.8f W_hh_penalty=%.6f",
        epoch,
        metrics["objective"],
        metrics["primary_normalized_mse"],
        metrics["primary_raw_mse"],
        metrics["whh_penalty"],
    )
    return metrics


@torch.no_grad()
def run_validation(
    model: SequenceControlModel, loader: DataLoader, device: torch.device
) -> Dict[str, object]:
    model.eval()
    totals: Dict[int, Dict[str, float]] = {
        lag: {"squared_error": 0.0, "target_energy": 0.0, "n_elements": 0.0}
        for lag in (model.primary_delay, *model.auxiliary_lags)
    }
    n_sequences = 0
    for batch in loader:
        inputs = batch["inputs"].to(device)
        outputs = model(inputs)
        n_sequences += int(inputs.shape[0])
        for lag, full_prediction in model.predictions_by_lag(outputs).items():
            target, pred_slice = make_delay_targets(inputs, lag)
            prediction = full_prediction[:, pred_slice, :]
            totals[lag]["squared_error"] += float(
                (prediction - target).square().sum().detach().cpu()
            )
            totals[lag]["target_energy"] += float(
                target.square().sum().detach().cpu()
            )
            totals[lag]["n_elements"] += float(target.numel())

    if n_sequences == 0:
        raise RuntimeError("Validation loader yielded no samples")
    by_lag: Dict[str, Dict[str, float]] = {}
    for lag, values in totals.items():
        n_elements = max(values["n_elements"], 1.0)
        target_energy = max(values["target_energy"], 1e-12)
        normalized_mse = values["squared_error"] / target_energy
        by_lag[str(lag)] = {
            "normalized_mse": float(normalized_mse),
            "r2_vs_zero": float(1.0 - normalized_mse),
            "raw_mse": float(values["squared_error"] / n_elements),
            "zero_predictor_mse": float(values["target_energy"] / n_elements),
            "n_elements": int(values["n_elements"]),
        }

    primary = by_lag[str(model.primary_delay)]
    aux_values = [
        by_lag[str(lag)]["normalized_mse"] for lag in model.auxiliary_lags
    ]
    return {
        "n_sequences": int(n_sequences),
        "primary_delay": int(model.primary_delay),
        "primary": primary,
        "auxiliary_normalized_mse_mean": (
            float(sum(aux_values) / len(aux_values)) if aux_values else None
        ),
        "by_lag": by_lag,
    }


def _checkpoint_payload(
    model: SequenceControlModel,
    cfg: dict,
    *,
    epoch: int,
    validation: Dict[str, object],
    gate: Dict[str, object] | None = None,
) -> Dict[str, object]:
    return {
        "model_state": model.state_dict(),
        "config": cfg,
        "epoch": int(epoch),
        "val_mse": float(validation["primary"]["raw_mse"]),
        "val_normalized_mse": float(validation["primary"]["normalized_mse"]),
        "delay": int(model.primary_delay),
        "auxiliary_lags": list(model.auxiliary_lags),
        "pretraining_source_type": "non_navigation_delayed_reconstruction",
        "validation": validation,
        "ctl01_gate": gate,
    }


def _training_gate(
    initial_validation: Dict[str, object],
    best_validation: Dict[str, object],
    analysis_cfg: Dict[str, Any],
) -> Dict[str, object]:
    initial = float(initial_validation["primary"]["normalized_mse"])
    best = float(best_validation["primary"]["normalized_mse"])
    max_nmse = float(analysis_cfg.get("success_primary_nmse_max", 0.5))
    min_reduction = float(
        analysis_cfg.get("success_min_relative_nmse_reduction", 0.20)
    )
    relative_reduction = (initial - best) / max(initial, 1e-12)
    finite = math.isfinite(initial) and math.isfinite(best)
    passed = bool(
        finite and best <= max_nmse and relative_reduction >= min_reduction
    )
    return {
        "passed": passed,
        "initial_primary_normalized_mse": initial,
        "best_primary_normalized_mse": best,
        "best_primary_r2_vs_zero": float(1.0 - best),
        "relative_nmse_reduction": float(relative_reduction),
        "required_primary_normalized_mse_max": max_nmse,
        "required_relative_nmse_reduction_min": min_reduction,
        "meaning": (
            "PASS: the primary delay task learned beyond the zero predictor; the "
            "checkpoint may proceed to shape/provenance checks before transfer."
            if passed
            else "FAIL: do not use this checkpoint as a trained control source."
        ),
    }


def _compatibility_report(checkpoint_path: Path, production_hidden_dim: int = 4096) -> Dict[str, object]:
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    state = checkpoint.get("model_state", {})
    input_key = "grid_rnn.cell.in_proj.weight"
    recurrent_key = "grid_rnn.cell.h_proj.weight"
    input_shape = list(state[input_key].shape) if input_key in state else None
    recurrent_shape = list(state[recurrent_key].shape) if recurrent_key in state else None
    expected_input = [production_hidden_dim, 2]
    expected_recurrent = [production_hidden_dim, production_hidden_dim]
    return {
        "checkpoint": str(checkpoint_path),
        "required_keys_present": bool(input_shape is not None and recurrent_shape is not None),
        "input_weight_shape": input_shape,
        "recurrent_weight_shape": recurrent_shape,
        "production_expected_input_weight_shape": expected_input,
        "production_expected_recurrent_weight_shape": expected_recurrent,
        "production_full_transfer_shape_compatible": bool(
            input_shape == expected_input and recurrent_shape == expected_recurrent
        ),
        "note": (
            "Small smoke-test checkpoints intentionally fail the production shape check; "
            "the formal run must use model.hidden_dim=4096."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CTL-01 non-navigation delayed-reconstruction pretraining"
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--override", type=str, nargs="*", default=[])
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = _apply_overrides(cfg, args.override)
    if args.output_dir is not None:
        cfg["output_dir"] = args.output_dir

    output_dir = prepare_output_dir(cfg["output_dir"])
    save_config_snapshot(cfg, output_dir)
    logger = get_logger("pretrain_sequence_control", output_dir / "experiment.log")
    set_seed(
        int(cfg["seed"]),
        deterministic_algorithms=bool(
            cfg.get("runtime", {}).get("deterministic_algorithms", False)
        ),
    )
    device = configure_runtime(cfg)
    logger.info("Device resolved to %s", device)

    auxiliary_lags = _parse_auxiliary_lags(cfg["data"])
    train_loader, val_loader, seq_cfg = make_loaders(cfg)
    model = build_model(cfg, device, auxiliary_lags)
    n_params = sum(parameter.numel() for parameter in model.parameters())
    n_grid_rnn_params = sum(
        parameter.numel() for parameter in model.grid_rnn.parameters()
    )
    logger.info(
        "Model built: %d total params (%d in grid_rnn core); primary delay=%d; auxiliary lags=%s",
        n_params,
        n_grid_rnn_params,
        seq_cfg.delay,
        list(auxiliary_lags),
    )

    tcfg = cfg["train"]
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(tcfg["lr"]),
        weight_decay=float(tcfg.get("weight_decay", 0.0)),
    )

    initial_validation = run_validation(model, val_loader, device)
    write_json(initial_validation, output_dir / "initial_validation.json")
    logger.info(
        "epoch 0 validation | primary_NMSE=%.6f raw_MSE=%.8f zero_MSE=%.8f",
        initial_validation["primary"]["normalized_mse"],
        initial_validation["primary"]["raw_mse"],
        initial_validation["primary"]["zero_predictor_mse"],
    )
    save_checkpoint(
        _checkpoint_payload(
            model, cfg, epoch=0, validation=initial_validation, gate=None
        ),
        output_dir / "initial.pt",
    )
    save_checkpoint(
        _checkpoint_payload(
            model, cfg, epoch=0, validation=initial_validation, gate=None
        ),
        output_dir / "best_validation.pt",
    )

    best_validation = initial_validation
    best_epoch = 0
    best_nmse = float(initial_validation["primary"]["normalized_mse"])
    history: list[Dict[str, object]] = [
        {"epoch": 0, "train": None, "validation": initial_validation}
    ]

    for epoch in range(1, int(tcfg["epochs"]) + 1):
        train_metrics = run_epoch_train(
            model,
            train_loader,
            optimizer,
            whh_weight_decay=float(tcfg.get("whh_weight_decay", 0.0)),
            grad_clip=float(tcfg.get("grad_clip", 0.0)),
            primary_weight=float(tcfg.get("primary_loss_weight", 1.0)),
            auxiliary_weight_per_lag=float(
                tcfg.get("auxiliary_loss_weight_per_lag", 1.0)
            ),
            task_loss_scale=float(tcfg.get("task_loss_scale", 1.0)),
            log_every_steps=int(tcfg.get("log_every_steps", 0)),
            device=device,
            logger=logger,
            epoch=epoch,
        )
        validation = run_validation(model, val_loader, device)
        val_nmse = float(validation["primary"]["normalized_mse"])
        logger.info(
            "epoch %d validation | primary_NMSE=%.6f R2_vs_zero=%.6f raw_MSE=%.8f aux_mean_NMSE=%s",
            epoch,
            val_nmse,
            validation["primary"]["r2_vs_zero"],
            validation["primary"]["raw_mse"],
            (
                f"{validation['auxiliary_normalized_mse_mean']:.6f}"
                if validation["auxiliary_normalized_mse_mean"] is not None
                else "n/a"
            ),
        )
        history.append(
            {"epoch": epoch, "train": train_metrics, "validation": validation}
        )

        if val_nmse < best_nmse:
            best_nmse = val_nmse
            best_epoch = epoch
            best_validation = validation
            save_checkpoint(
                _checkpoint_payload(
                    model, cfg, epoch=epoch, validation=validation, gate=None
                ),
                output_dir / "best_validation.pt",
            )
            logger.info(
                "epoch %d | new best_validation.pt (primary_NMSE=%.6f)",
                epoch,
                val_nmse,
            )

    gate = _training_gate(initial_validation, best_validation, cfg["analysis"])
    write_json(gate, output_dir / "training_gate.json")

    # Attach the final gate to both transfer-relevant checkpoints.
    best_checkpoint = load_checkpoint(output_dir / "best_validation.pt", "cpu")
    best_checkpoint["ctl01_gate"] = gate
    save_checkpoint(best_checkpoint, output_dir / "best_validation.pt")
    final_validation = history[-1]["validation"]
    save_checkpoint(
        _checkpoint_payload(
            model,
            cfg,
            epoch=int(tcfg["epochs"]),
            validation=final_validation,
            gate=gate,
        ),
        output_dir / "final.pt",
    )

    compatibility = _compatibility_report(output_dir / "best_validation.pt")
    write_json(compatibility, output_dir / "checkpoint_compatibility.json")
    write_json(
        {
            "history": history,
            "best_epoch": int(best_epoch),
            "best_validation": best_validation,
            "training_gate": gate,
        },
        output_dir / "training_history.json",
    )
    save_run_provenance(
        output_dir,
        "pretrain_sequence_control",
        cfg,
        extra={
            "n_params": n_params,
            "n_grid_rnn_params": n_grid_rnn_params,
            "primary_delay": seq_cfg.delay,
            "auxiliary_lags": list(auxiliary_lags),
            "best_epoch": best_epoch,
            "best_primary_normalized_mse": best_nmse,
            "ctl01_gate_passed": gate["passed"],
            "production_full_transfer_shape_compatible": compatibility[
                "production_full_transfer_shape_compatible"
            ],
        },
    )
    logger.info(
        "Done | best_epoch=%d best_primary_NMSE=%.6f gate=%s checkpoints=%s",
        best_epoch,
        best_nmse,
        "PASS" if gate["passed"] else "FAIL",
        output_dir,
    )


if __name__ == "__main__":
    main()
