"""Entry point for Bot6 Market Maker."""
import logging
import os
import sys

from bot6_mm.config import Bot6Config
from bot6_mm.bot import Bot6


def setup_logging(cfg: Bot6Config) -> None:
    os.makedirs(os.path.dirname(cfg.log_file), exist_ok=True)
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(cfg.log_file, encoding="utf-8"),
    ]
    logging.basicConfig(
        level=getattr(logging, cfg.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
    )


def main() -> None:
    cfg = Bot6Config.from_env()
    setup_logging(cfg)
    bot = Bot6(cfg)
    bot.run()


if __name__ == "__main__":
    main()
