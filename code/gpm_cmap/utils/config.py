from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable

import yaml


def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_config(cfg: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


def warn_unconsumed_model_keys(
    model_cfg: Dict[str, Any],
    consumed_keys: Iterable[str],
    logger,
    experiment: str = "",
) -> list:
    """Warn loudly about config keys under ``model:`` that the experiment never
    reads (rule #12: a silently-ignored config key makes a researcher believe
    they changed the model when they did not).

    Parameters
    ----------
    model_cfg : the ``cfg["model"]`` dict.
    consumed_keys : the set of keys the driver actually forwards into the model.
    logger : a logger with ``.warning``.
    experiment : optional name for the log message.

    Returns the sorted list of unconsumed keys (also logged).
    """
    declared = set(model_cfg.keys())
    consumed = set(consumed_keys)
    unconsumed = sorted(declared - consumed)
    if unconsumed:
        logger.warning(
            "[%s] %d config key(s) under 'model:' are NOT consumed by this "
            "experiment and will have NO effect: %s. Either wire them into the "
            "model builder or remove them from the config.",
            experiment or "config-audit",
            len(unconsumed),
            unconsumed,
        )
    return unconsumed
