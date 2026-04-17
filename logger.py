"""
logger.py  –  Centralised logging setup for Air Drawing System.
Call setup_logging() once at startup; then use get_logger(__name__) anywhere.
"""

from __future__ import annotations
import logging
import logging.handlers
import sys
from pathlib import Path
from config import get_config

_initialised = False

def setup_logging() -> None:
    global _initialised
    if _initialised:
        return
    cfg = get_config().log
    paths = get_config().paths

    level = getattr(logging, cfg.level.upper(), logging.INFO)
    root  = logging.getLogger()
    root.setLevel(level)

    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(name)-25s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if cfg.to_console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        root.addHandler(ch)

    if cfg.to_file:
        paths.logs_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(
            paths.runtime_log,
            maxBytes=cfg.max_bytes,
            backupCount=cfg.backup_count,
        )
        fh.setFormatter(fmt)
        root.addHandler(fh)

    _initialised = True

def get_logger(name: str) -> logging.Logger:
    """Return a named logger (call setup_logging first)."""
    return logging.getLogger(name)
