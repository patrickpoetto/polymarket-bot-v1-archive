"""Real-time price feed via Polymarket CLOB WebSocket, with REST fallback.

Architecture:
- A background thread runs an asyncio event loop for the WS connection.
- Price snapshots are stored in a thread-safe dict.
- The main thread reads prices via ``get(token_id)``.
- When WS data is stale (no update in ``stale_seconds``), ``get()`` returns
  ``Prices(stale=True)`` so callers can fall back to REST polling.
- Re-subscription is supported: call ``subscribe(token_ids)`` at any time.
"""
from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from typing import Optional

from bot2_btc.models import Prices

logger = logging.getLogger(__name__)

_WS_RECONNECT_DELAY = 1.0   # seconds between reconnect attempts
_WS_PING_INTERVAL = 20      # seconds between keepalive pings


class PriceFeed:
    """Thread-safe WebSocket price feed."""

    def __init__(self, ws_url: str, stale_seconds: float = 3.0):
        self._ws_url = ws_url
        self._stale_seconds = stale_seconds
        # token_id → {"bid": float|None, "ask": float|None, "ts": float}
        self._prices: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._token_ids: list[str] = []
        self._resubscribe_flag = threading.Event()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    # ── Public API ───────────────────────────────────────────────────────────

    def subscribe(self, token_ids: list[str]) -> None:
        """(Re)subscribe to a new set of token IDs. Safe to call at any time."""
        with self._lock:
            self._token_ids = list(token_ids)
            for tid in token_ids:
                if tid not in self._prices:
                    self._prices[tid] = {"bid": None, "ask": None, "ts": 0.0}
        self._resubscribe_flag.set()
        logger.debug("PriceFeed: subscribing to %d tokens", len(token_ids))

    def get(self, token_id: str) -> Prices:
        """Return the latest Prices for a token. ``stale=True`` if data is old."""
        with self._lock:
            entry = self._prices.get(token_id)
        if entry is None:
            return Prices(stale=True)
        age = time.time() - entry.get("ts", 0.0)
        stale = age > self._stale_seconds or entry.get("ts", 0.0) == 0.0
        return Prices(bid=entry.get("bid"), ask=entry.get("ask"), stale=stale)

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._thread_main, daemon=True, name="bot2-price-feed"
        )
        self._thread.start()
        logger.info("PriceFeed started (ws_url=%s)", self._ws_url)

    def stop(self) -> None:
        self._stop.set()
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("PriceFeed stopped")

    # ── Internal ─────────────────────────────────────────────────────────────

    def _thread_main(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._ws_supervisor())
        finally:
            self._loop.close()

    async def _ws_supervisor(self) -> None:
        """Outer loop: reconnects on any failure."""
        while not self._stop.is_set():
            try:
                await self._ws_session()
            except Exception as exc:
                if not self._stop.is_set():
                    logger.warning("PriceFeed WS error: %s – reconnecting in %.1fs", exc, _WS_RECONNECT_DELAY)
                    await asyncio.sleep(_WS_RECONNECT_DELAY)

    async def _ws_session(self) -> None:
        """One WS connection: subscribe, receive messages until disconnect."""
        import websockets  # local import to avoid hard dep at module level

        async with websockets.connect(
            self._ws_url,
            ping_interval=_WS_PING_INTERVAL,
            ping_timeout=10,
            close_timeout=5,
        ) as ws:
            logger.debug("PriceFeed WS connected")
            self._resubscribe_flag.clear()
            await self._send_subscribe(ws)

            while not self._stop.is_set():
                # Check if a re-subscription was requested
                if self._resubscribe_flag.is_set():
                    self._resubscribe_flag.clear()
                    await self._send_subscribe(ws)

                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
                    self._handle_raw(raw)
                except asyncio.TimeoutError:
                    continue  # no message, loop again
                except Exception:
                    raise  # bubble up to supervisor

    async def _send_subscribe(self, ws) -> None:
        with self._lock:
            token_ids = list(self._token_ids)
        if not token_ids:
            return
        msg = json.dumps({"assets_ids": token_ids, "type": "market"})
        await ws.send(msg)
        logger.debug("PriceFeed subscribed to %s", [t[:12] for t in token_ids])

    def _handle_raw(self, raw: str) -> None:
        try:
            data = json.loads(raw)
        except Exception:
            return

        if isinstance(data, list):
            for item in data:
                self._process_item(item)
        elif isinstance(data, dict):
            self._process_item(data)

    def _process_item(self, item: dict) -> None:
        if not isinstance(item, dict):
            return

        asset_id = item.get("asset_id") or item.get("market")
        if not asset_id:
            return

        bid: Optional[float] = None
        ask: Optional[float] = None

        # Direct best_bid/best_ask fields
        if "best_bid" in item:
            try:
                bid = float(item["best_bid"])
            except (TypeError, ValueError):
                pass
        if "best_ask" in item:
            try:
                ask = float(item["best_ask"])
            except (TypeError, ValueError):
                pass

        # Array-based bids/asks
        bids = item.get("bids", [])
        asks = item.get("asks", [])
        if bids and bid is None:
            try:
                bid = max(
                    float(b["price"])
                    for b in bids
                    if b.get("price") and float(b.get("size", 1)) > 0
                )
            except (TypeError, ValueError):
                pass
        if asks and ask is None:
            try:
                ask = min(
                    float(a["price"])
                    for a in asks
                    if a.get("price") and float(a.get("size", 1)) > 0
                )
            except (TypeError, ValueError):
                pass

        # NOTE: "changes" events carry incremental book-level deltas (a single
        # price level added/removed), NOT the new best bid/ask.  Using min/max
        # on them corrupts the cached price (e.g. a stale ask at 0.01 being
        # cancelled would set ask→0.01 even if best ask is 0.70).
        # We intentionally skip changes here; best_bid/best_ask and full
        # bids/asks snapshots are sufficient for our latency requirements.

        if bid is not None or ask is not None:
            with self._lock:
                entry = self._prices.get(asset_id, {"bid": None, "ask": None, "ts": 0.0})
                if bid is not None:
                    entry["bid"] = bid
                if ask is not None:
                    entry["ask"] = ask
                entry["ts"] = time.time()
                self._prices[asset_id] = entry


class NullPriceFeed(PriceFeed):
    """Always returns stale prices – used when WS is disabled."""

    def __init__(self) -> None:
        super().__init__("", stale_seconds=0.0)

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def subscribe(self, token_ids: list[str]) -> None:
        pass
