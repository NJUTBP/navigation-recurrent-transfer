"""Strict JSON serialization for experiment artifacts.

Experiment metadata must remain machine-readable.  Python's default JSON writer
accepts NaN/Infinity and ``default=str`` silently converts unsupported values,
both of which make downstream validation ambiguous.  This module normalizes
NumPy/Torch scalars explicitly, maps non-finite numeric values to JSON ``null``,
and writes atomically.
"""
from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def _json_value(value: Any) -> Any:
    """Convert supported scientific Python values to strict JSON values.

    Unsupported objects raise ``TypeError`` instead of being silently converted
    to a string.  Non-finite floating-point values are represented as ``None``
    because JSON has no portable NaN/Infinity representation.
    """
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, Path):
        return str(value)

    # Torch is optional for torch-free analysis utilities.
    try:
        import torch
    except ImportError:  # pragma: no cover - exercised in torch-free installs
        torch = None
    if torch is not None and isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return _json_value(value.detach().cpu().item())
        return _json_value(value.detach().cpu().tolist())

    if isinstance(value, Mapping):
        normalized = {}
        for key, item in value.items():
            if not isinstance(key, (str, int, float, bool)) and key is not None:
                raise TypeError(f"Unsupported JSON mapping key type: {type(key).__name__}")
            normalized[str(key)] = _json_value(item)
        return normalized
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]

    raise TypeError(f"Unsupported JSON value type: {type(value).__name__}")


def dumps_json(data: Any, *, indent: int = 2) -> str:
    """Return strict JSON text for supported experiment data."""
    return json.dumps(
        _json_value(data),
        indent=indent,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=False,
    )


def write_json(data: Any, path: str | Path, *, indent: int = 2) -> Path:
    """Atomically write strict JSON and return the final path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = dumps_json(data, indent=indent) + "\n"

    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise
    return path
