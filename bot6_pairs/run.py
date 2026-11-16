"""Run Bot6 Paired MM."""
import logging
import os
import sys

from bot6_pairs.bot import Bot6Pairs
from bot6_pairs.config import Bot6PairsConfig


def setup_logging(log_file: str, level: str = "INFO") -> None:
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )


def main() -> None:
    pcfg = Bot6PairsConfig.from_env()
    base = pcfg.base()
    setup_logging(pcfg.log_file, base.log_level)
    Bot6Pairs(pcfg).run()


if __name__ == "__main__":
    main()
