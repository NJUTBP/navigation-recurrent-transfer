from __future__ import annotations

import logging
from pathlib import Path


def get_logger(name: str, log_file: str | Path | None = None) -> logging.Logger:
    """Return a logger whose handlers are bound to the requested run path.

    Reusing a logger name for a second run must not keep writing to the first
    run's file. The function therefore replaces handlers whenever the resolved
    file target changes.
    """
    logger = logging.getLogger(name)
    requested_path = str(Path(log_file).resolve()) if log_file is not None else None
    if logger.handlers and getattr(logger, "_gpm_cmap_log_file", object()) == requested_path:
        return logger

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if requested_path is not None:
        log_path = Path(requested_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    logger._gpm_cmap_log_file = requested_path
    return logger
