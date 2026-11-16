"""Market discovery, resolution, and orderbook utilities for 15-min BTC markets.

Three responsibilities:
1. **Find** the currently active (or next upcoming) market.
2. **Resolve** a closed market to its winning outcome.
3. **Infer** outcome early via 99 c price heuristic.
"""
from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timezone
from typing import Optional

import requests
from bs4 import BeautifulSoup

from bot5_martingale.config import Bot5Config
from bot5_martingale.models import MarketInfo, MarketPhase, Side

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


# ======================================================================
# Timestamp helpers
# ======================================================================

def current_market_start(interval: int = 900) -> int:
    """Return the start-timestamp of the 15-min window that contains *now*."""
    return (int(time.time()) // interval) * interval


def market_slug(start_ts: int, pattern: str = "btc-updown-15m-{ts}") -> str:
    return pattern.replace("{ts}", str(start_ts))


def market_end(start_ts: int, interval: int = 900) -> int:
    return start_ts + interval


# ======================================================================
# Market discovery
# ======================================================================

def _parse_market_from_next_data(html: str, expected_slug: Optional[str] = None) -> Optional[MarketInfo]:
    """Extract market info from Polymarket's __NEXT_DATA__ JSON island."""
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

    condition_id = market_raw.get("conditionId", "")
    token_ids_raw = market_raw.get("clobTokenIds", "[]")
    outcomes_raw = market_raw.get("outcomes", "[]")
    if isinstance(token_ids_raw, str):
        token_ids_raw = json.loads(token_ids_raw)
    if isinstance(outcomes_raw, str):
        outcomes_raw = json.loads(outcomes_raw)

    end_date_str = market_raw.get("endDate", "")
    end_ts = 0
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
        condition_id=condition_id,
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


def find_market_gamma(slug_contains: str) -> Optional[MarketInfo]:
    """Find the most-recent active market via Gamma API slug search."""
    try:
        resp = requests.get(
            f"{GAMMA_API}/markets",
            params={
                "slug_contains": slug_contains,
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
        end_date_str = m.get("endDate", "")
        end_ts = 0
        if end_date_str:
            try:
                end_ts = int(datetime.fromisoformat(end_date_str.replace("Z", "+00:00")).timestamp())
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


class MarketFinder:
    """Stateful market discovery for sequential 15-min BTC markets."""

    def __init__(self, cfg: Bot5Config):
        self.cfg = cfg
        self.interval = cfg.market_interval_seconds
        self.pattern = cfg.market_url_pattern
        self._last_known_start_ts: Optional[int] = None

    def find_current(self) -> Optional[MarketInfo]:
        """Return the currently-active or next-upcoming market."""
        start = current_market_start(self.interval)
        slug = market_slug(start, self.pattern)

        info = scrape_market_by_slug(slug)
        if info and info.condition_id:
            self._last_known_start_ts = start
            return info

        # Try previous window (market might have just ended)
        prev_start = start - self.interval
        info = scrape_market_by_slug(market_slug(prev_start, self.pattern))
        if info and info.condition_id:
            self._last_known_start_ts = prev_start
            return info

        # Gamma fallback
        slug_prefix = self.pattern.split("-{")[0]  # e.g. "btc-updown-15m"
        info = find_market_gamma(slug_prefix)
        if info and info.condition_id:
            if info.timestamp_tag:
                self._last_known_start_ts = info.timestamp_tag
            return info

        logger.warning("No active 15m market found")
        return None

    def predict_next(self, current_start_ts: int) -> str:
        """Return the slug for the market after the one starting at *current_start_ts*."""
        return market_slug(current_start_ts + self.interval, self.pattern)

    def predict_next_next(self, current_start_ts: int) -> str:
        """Return the slug two markets ahead."""
        return market_slug(current_start_ts + 2 * self.interval, self.pattern)

    def fetch_market_by_start(self, start_ts: int) -> Optional[MarketInfo]:
        return scrape_market_by_slug(market_slug(start_ts, self.pattern))


# ======================================================================
# Market resolution
# ======================================================================

def resolve_via_clob(condition_id: str) -> Optional[Side]:
    """Ask CLOB API for the resolved winner.

    Returns a Side only if the market is *closed* AND exactly one token
    has ``winner == true``.  This avoids returning stale/pending data.
    """
    try:
        resp = requests.get(f"{CLOB_API}/markets/{condition_id}", timeout=10)
        resp.raise_for_status()
        data = resp.json()

        is_closed = data.get("closed", False)
        if not is_closed:
            return None

        tokens = data.get("tokens", [])
        for tok in tokens:
            winner_raw = tok.get("winner")
            if winner_raw is True or str(winner_raw).lower() == "true":
                outcome = tok.get("outcome", "")
                side = Side.UP if outcome.lower() in ("up", "yes") else Side.DOWN
                logger.debug("CLOB resolved %s -> %s", condition_id[:12], side.value)
                return side
        # closed but no winner yet (resolution propagating)
        return None
    except Exception as exc:
        logger.debug("CLOB resolve failed for %s: %s", condition_id[:12], exc)
        return None


def resolve_via_gamma(condition_id: str) -> Optional[Side]:
    """Fallback: query Gamma for outcome prices.

    IMPORTANT: The Gamma API condition_id search is unreliable and may
    return unrelated markets.  We validate the response by checking that
    the returned condition_id matches and the market is actually closed
    with a definitive price (1.0 / 0.0).
    """
    try:
        resp = requests.get(
            f"{GAMMA_API}/markets",
            params={"condition_id": condition_id, "limit": 1},
            timeout=10,
        )
        resp.raise_for_status()
        markets = resp.json()
        if not markets:
            return None
        m = markets[0]

        # Validate: condition_id must match
        returned_cid = m.get("conditionId", "")
        if returned_cid.lower() != condition_id.lower():
            logger.debug(
                "Gamma returned wrong market: asked %s, got %s",
                condition_id[:16], returned_cid[:16],
            )
            return None

        if not m.get("closed", False):
            return None

        outcomes = m.get("outcomes", [])
        prices = m.get("outcomePrices", [])
        if isinstance(outcomes, str):
            outcomes = json.loads(outcomes)
        if isinstance(prices, str):
            prices = json.loads(prices)
        if not outcomes or not prices:
            return None

        # Require definitive prices (one at ~1.0, other at ~0.0)
        float_prices = [float(p) for p in prices]
        if max(float_prices) < 0.95:
            return None

        max_idx = max(range(len(float_prices)), key=lambda i: float_prices[i])
        winner = outcomes[max_idx]
        side = Side.UP if winner.lower() in ("up", "yes") else Side.DOWN
        logger.debug("Gamma resolved %s -> %s", condition_id[:12], side.value)
        return side
    except Exception as exc:
        logger.debug("Gamma resolve failed for %s: %s", condition_id[:12], exc)
        return None


class MarketResolver:
    """Determine the winning side of a resolved market, with retries.

    Strategy: rely primarily on CLOB (authoritative, checks closed + winner).
    Only fall back to Gamma if CLOB consistently fails, and Gamma passes
    strict validation.  Waits patiently for resolution data to propagate.
    """

    def __init__(self, max_retries: int = 20, retry_delay: float = 5.0):
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def resolve(self, condition_id: str) -> Optional[Side]:
        for attempt in range(1, self.max_retries + 1):
            # Primary: CLOB (authoritative)
            result = resolve_via_clob(condition_id)
            if result is not None:
                return result

            # Gamma fallback only after several CLOB attempts (give CLOB
            # time to propagate the resolution)
            if attempt >= 4:
                result = resolve_via_gamma(condition_id)
                if result is not None:
                    return result

            if attempt < self.max_retries:
                logger.info(
                    "Resolution pending for %s (attempt %d/%d), retrying in %.0fs",
                    condition_id[:12], attempt, self.max_retries, self.retry_delay,
                )
                time.sleep(self.retry_delay)

        logger.warning("Could not resolve %s after %d attempts", condition_id[:12], self.max_retries)
        return None


# ======================================================================
# Orderbook-based inference (early / 99c heuristic)
# ======================================================================

def infer_outcome_from_prices(
    prices: dict[str, dict],
    threshold: float = 0.99,
) -> Optional[Side]:
    """If any outcome's ask or bid >= *threshold*, return that side.

    *prices* is the same dict shape produced by _fetch_books / ws_prices:
        {"Up": {"ask": 0.99, "bid": 0.98}, "Down": {"ask": 0.02, "bid": 0.01}}
    """
    for outcome_str, book in prices.items():
        bid = book.get("bid")
        ask = book.get("ask")
        # Use the strongest available signal. Preferring bid can miss valid
        # early signals when ask is already above threshold but bid is not.
        vals = [v for v in (bid, ask) if v is not None]
        price = max(vals) if vals else None
        if price is not None and price >= threshold:
            return Side.UP if outcome_str.lower() in ("up", "yes") else Side.DOWN
    return None


def market_phase(info: MarketInfo, inference_window: int = 60) -> MarketPhase:
    """Determine the current lifecycle phase of a market."""
    now = time.time()
    end = info.end_timestamp
    if end <= 0:
        return MarketPhase.UNKNOWN
    remaining = end - now
    if remaining > info.end_timestamp - (info.timestamp_tag or (end - 900)):
        return MarketPhase.PRE_OPEN
    if remaining <= 0:
        return MarketPhase.CLOSED
    if remaining <= inference_window:
        return MarketPhase.LAST_MINUTE
    return MarketPhase.OPEN
