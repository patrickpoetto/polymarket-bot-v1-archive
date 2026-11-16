#!/usr/bin/env python3
"""Entry point for bot5_martingale.

Usage:
    # Paper mode (default):
    python -m bot5_martingale.run

    # Live mode:
    BOT5_MODE=live python -m bot5_martingale.run

    # Or use CLI flags:
    python -m bot5_martingale.run --mode paper
    python -m bot5_martingale.run --mode live --max-level 6
"""
from __future__ import annotations

import argparse
import sys

from bot5_martingale.config import Bot5Config
from bot5_martingale.bot import Bot5


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Bot5 Martingale – 15-min BTC binary-market trading bot",
    )
    p.add_argument("--mode", choices=["paper", "live"], default=None,
                   help="Override BOT5_MODE env var")
    p.add_argument("--max-level", type=int, default=None,
                   help="Override BOT5_MAX_LEVEL")
    p.add_argument("--target-profit", type=float, default=None,
                   help="Override BOT5_TARGET_PROFIT_PER_WIN")
    p.add_argument("--log-level", default=None,
                   help="Override BOT5_LOG_LEVEL (DEBUG/INFO/WARNING)")
    p.add_argument("--validate-only", action="store_true",
                   help="Validate config and exit")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Bot5Config.from_env()

    # CLI overrides
    overrides = {}
    if args.mode:
        from bot5_martingale.models import BotMode
        overrides["mode"] = BotMode(args.mode)
    if args.max_level is not None:
        overrides["max_level"] = args.max_level
    if args.target_profit is not None:
        overrides["target_profit_per_win"] = args.target_profit
    if args.log_level:
        overrides["log_level"] = args.log_level

    if overrides:
        # Rebuild with overrides (frozen dataclass, so recreate)
        d = {f.name: getattr(cfg, f.name) for f in cfg.__dataclass_fields__.values()}
        d.update(overrides)
        cfg = Bot5Config(**d)

    errors = cfg.validate()
    if errors:
        for e in errors:
            print(f"Config error: {e}", file=sys.stderr)
        sys.exit(1)

    if args.validate_only:
        print("Config valid.")
        sys.exit(0)

    bot = Bot5(cfg)
    bot.run()


if __name__ == "__main__":
    main()
