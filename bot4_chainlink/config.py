from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Bot4Config:
    # Chainlink Data Streams access (namespaced to avoid collisions)
    api_base_url: str
    price_api_base_url: str
    api_key: str
    api_secret: str
    feed_id: str

    # Local paths
    labels_csv: Path
    out_dir: Path

    # Market window
    market_window_seconds: int = 300


def load_config() -> Bot4Config:
    """
    BOT4_* env variables only, to avoid affecting other bots.
    """
    api_base_url = os.getenv("BOT4_CHAINLINK_API_BASE_URL", "https://api.dataengine.chain.link").strip()
    price_api_base_url = os.getenv(
        "BOT4_CHAINLINK_PRICE_API_BASE_URL",
        "https://priceapi.dataengine.chain.link",
    ).strip()
    api_key = os.getenv("BOT4_CHAINLINK_API_KEY", "").strip()
    api_secret = os.getenv("BOT4_CHAINLINK_API_SECRET", "").strip()
    feed_id = os.getenv("BOT4_CHAINLINK_FEED_ID", "").strip()
    labels_csv = Path(
        os.getenv(
            "BOT4_LABELS_CSV",
            "archived_data/btc_5m_outcomes_history_merged.csv",
        )
    )
    out_dir = Path(os.getenv("BOT4_OUT_DIR", "bot4_chainlink/data"))

    missing = []
    if not api_key:
        missing.append("BOT4_CHAINLINK_API_KEY")
    if not api_secret:
        missing.append("BOT4_CHAINLINK_API_SECRET")
    if not feed_id:
        missing.append("BOT4_CHAINLINK_FEED_ID")
    if missing:
        raise ValueError(f"Missing required env vars: {', '.join(missing)}")

    return Bot4Config(
        api_base_url=api_base_url,
        price_api_base_url=price_api_base_url,
        api_key=api_key,
        api_secret=api_secret,
        feed_id=feed_id,
        labels_csv=labels_csv,
        out_dir=out_dir,
    )
