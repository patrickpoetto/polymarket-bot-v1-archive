"""Market discovery for 15-min BTC markets (self-contained, no shared deps)."""
from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime
from typing import Optional

import requests
from bs4 import BeautifulSoup

from bot2_btc.config import Bot2Config
from bot2_btc.models import MarketInfo, Side

logger = logging.getLogger(__name__)

GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"
POLYMARKET_BASE = "https://polymarket.com"

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
}


# ── Timestamp helpers ────────────────────────────────────────────────────────

def current_window_start(interval: int = 900) -> int:
    return (int(time.time()) // interval) * interval


def market_slug(start_ts: int, pattern: str = "btc-updown-15m-{ts}") -> str:
    return pattern.replace("{ts}", str(start_ts))


# ── Parse helpers ────────────────────────────────────────────────────────────

def _parse_market_from_next_data(html: str, expected_slug: str = "") -> Optional[MarketInfo]:
    soup = BeautifulSoup(html, "html.parser")
    script = soup.find("script", {"id": "__NEXT_DATA__", "type": "application/json"})
    if not script or not script.string:
        return None
    try:
        data = json.loads(script.string)
    except json.JSONDecodeError:
        return None

    queries = (
        data.get("props", {})
        .get("pageProps", {})
        .get("dehydratedState", {})
        .get("queries", [])
    )

    market_raw: Optional[dict] = None
    for query in queries:
        state_data = query.get("state", {}).get("data", {})
        if "markets" in state_data and state_data["markets"]:
            market_raw = state_data["markets"][0]
            break
        if "conditionId" in state_data:
            market_raw = state_data
            break

    if not market_raw:
        return None

    token_ids_raw = market_raw.get("clobTokenIds", "[]")
    outcomes_raw = market_raw.get("outcomes", "[]")
    if isinstance(token_ids_raw, str):
        token_ids_raw = json.loads(token_ids_raw)
    if isinstance(outcomes_raw, str):
        outcomes_raw = json.loads(outcomes_raw)

    end_ts = 0
    end_date_str = market_raw.get("endDate", "")
    if end_date_str:
        try:
            end_ts = int(
                datetime.fromisoformat(end_date_str.replace("Z", "+00:00")).timestamp()
            )
        except (ValueError, OSError):
            pass

    slug = market_raw.get("groupItemTitle", "") or market_raw.get("slug", "")
    ts_match = re.search(r"(\d{10})", slug) or (
        re.search(r"(\d{10})", expected_slug) if expected_slug else None
    )
    ts_tag = int(ts_match.group(1)) if ts_match else 0

    return MarketInfo(
        condition_id=market_raw.get("conditionId", ""),
        question=market_raw.get("question", ""),
        outcomes=list(outcomes_raw),
        token_ids=list(token_ids_raw),
        end_timestamp=end_ts,
        event_slug=slug,
        timestamp_tag=ts_tag,
    )


def scrape_market_by_slug(slug: str) -> Optional[MarketInfo]:
    url = f"{POLYMARKET_BASE}/event/{slug}"
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=15)
        resp.raise_for_status()
        return _parse_market_from_next_data(resp.text, expected_slug=slug)
    except Exception as exc:
        logger.debug("scrape %s failed: %s", slug, exc)
        return None


def find_market_gamma(slug_prefix: str) -> Optional[MarketInfo]:
    try:
        resp = requests.get(
            f"{GAMMA_API}/markets",
            params={
                "slug_contains": slug_prefix,
                "closed": "false",
                "limit": 5,
                "order": "endDate",
                "ascending": "true",
            },
            headers=_HEADERS,
            timeout=10,
        )
        resp.raise_for_status()
        markets = resp.json()
        if not isinstance(markets, list) or not markets:
            return None
        m = markets[0]
        end_ts = 0
        end_date_str = m.get("endDate", "")
        if end_date_str:
            try:
                end_ts = int(
                    datetime.fromisoformat(end_date_str.replace("Z", "+00:00")).timestamp()
                )
            except (ValueError, OSError):
                pass

        token_ids = m.get("clobTokenIds", [])
        outcomes = m.get("outcomes", [])
        if isinstance(token_ids, str):
            token_ids = json.loads(token_ids)
        if isinstance(outcomes, str):
            outcomes = json.loads(outcomes)

        slug = m.get("slug", "")
        ts_match = re.search(r"(\d{10})", slug)
        ts_tag = int(ts_match.group(1)) if ts_match else 0

        return MarketInfo(
            condition_id=m.get("conditionId", ""),
            question=m.get("question", ""),
            outcomes=list(outcomes),
            token_ids=list(token_ids),
            end_timestamp=end_ts,
            event_slug=slug,
            timestamp_tag=ts_tag,
        )
    except Exception as exc:
        logger.debug("Gamma search failed: %s", exc)
        return None


# ── MarketFinder ─────────────────────────────────────────────────────────────

class MarketFinder:
    """Discover sequential 15-min BTC markets.

    Provides:
    - ``find_current()``  – the currently active (or just-started) market
    - ``find_next(start_ts)`` – the market following the one at start_ts
    """

    def __init__(self, cfg: Bot2Config):
        self.interval = cfg.market_interval_seconds
        self.pattern = cfg.market_url_pattern
        self._slug_prefix = self.pattern.split("-{")[0]  # e.g. "btc-updown-15m"

    # ------------------------------------------------------------------

    def find_by_start(self, start_ts: int) -> Optional[MarketInfo]:
        slug = market_slug(start_ts, self.pattern)
        info = scrape_market_by_slug(slug)
        if info and info.condition_id:
            return info
        return None

    def find_current(self) -> Optional[MarketInfo]:
        """Return the market that contains *now*, or the next upcoming one."""
        start = current_window_start(self.interval)

        for offset in (0, -self.interval, self.interval):
            info = self.find_by_start(start + offset)
            if info and info.condition_id:
                return info

        # Gamma fallback
        info = find_market_gamma(self._slug_prefix)
        if info and info.condition_id:
            return info

        logger.warning("No active 15m BTC market found")
        return None

    def find_next(self, current_start_ts: int) -> Optional[MarketInfo]:
        """Return the market immediately after the one at current_start_ts."""
        return self.find_by_start(current_start_ts + self.interval)

    def find_next_slug(self, current_start_ts: int) -> str:
        return market_slug(current_start_ts + self.interval, self.pattern)


# ── CLOB best-prices helper ──────────────────────────────────────────────────

def fetch_best_prices_rest(token_ids: list[str]) -> dict[str, dict]:
    """Fetch bid/ask for a list of token_ids via CLOB REST. Returns {token_id: {bid, ask}}."""
    result: dict[str, dict] = {tid: {"bid": None, "ask": None} for tid in token_ids}
    for tid in token_ids:
        try:
            resp = requests.get(f"{CLOB_API}/book", params={"token_id": tid}, timeout=5)
            resp.raise_for_status()
            data = resp.json()
            bids = data.get("bids", [])
            asks = data.get("asks", [])
            best_bid = max((float(b["price"]) for b in bids if b.get("price")), default=None)
            best_ask = min((float(a["price"]) for a in asks if a.get("price")), default=None)
            result[tid] = {"bid": best_bid, "ask": best_ask}
        except Exception as exc:
            logger.debug("REST book fetch for %s failed: %s", tid[:12], exc)
    return result
