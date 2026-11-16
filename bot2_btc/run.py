#!/usr/bin/env python3
"""Entry point for Bot2 BTC.

Usage:
    python -m bot2_btc.run            # uses .env in cwd
    ./venv/bin/python -m bot2_btc.run
"""
import sys
from pathlib import Path

# Ensure project root is on sys.path when run directly
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from bot2_btc.bot import Bot2BTC, setup_logging
from bot2_btc.config import Bot2Config


def main() -> None:
    cfg = Bot2Config.from_env()
    setup_logging(cfg)

    errors = cfg.validate()
    if errors:
        for e in errors:
            print(f"[CONFIG ERROR] {e}")
        sys.exit(1)

    bot = Bot2BTC(cfg)
    bot.run()


if __name__ == "__main__":
    main()
