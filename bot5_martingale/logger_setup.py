"""Structured logging configuration for bot5.

Sets up:
- Console handler (coloured, concise)
- Rotating file handler (detailed)
- Trade-specific CSV logger
"""
from __future__ import annotations

import csv
import io
import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

from bot5_martingale.config import Bot5Config


def setup_logging(cfg: Bot5Config) -> logging.Logger:
    log_dir = Path(cfg.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger("bot5_martingale")
    root.setLevel(getattr(logging, cfg.log_level.upper(), logging.INFO))
    root.handlers.clear()

    fmt_console = logging.Formatter(
        "%(asctime)s [%(levelname).1s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    fmt_file = logging.Formatter(
        "%(asctime)s [%(levelname)-8s] %(name)s:%(lineno)d %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler()
    ch.setFormatter(fmt_console)
    ch.setLevel(getattr(logging, cfg.log_level.upper(), logging.INFO))
    root.addHandler(ch)

    fh = RotatingFileHandler(
        log_dir / "bot5.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    fh.setFormatter(fmt_file)
    fh.setLevel(logging.DEBUG)
    root.addHandler(fh)

    # Quiet noisy libraries
    for name in ("httpx", "urllib3", "websockets", "httpcore"):
        logging.getLogger(name).setLevel(logging.WARNING)

    return root


class TradeCSVLogger:
    """Append-only CSV for trade records."""

    COLUMNS = [
        "ts_entry", "ts_resolved", "market", "condition_id", "side",
        "level", "target_usd", "filled_usd", "price", "shares", "fee",
        "result", "pnl", "cum_pnl", "inferred",
    ]

    def __init__(self, cfg: Bot5Config):
        log_dir = Path(cfg.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        self._path = log_dir / "trades.csv"
        if not self._path.exists():
            with open(self._path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(self.COLUMNS)

    def log(self, row: dict) -> None:
        with open(self._path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.COLUMNS, extrasaction="ignore")
            writer.writerow(row)
