"""
Bot3 BTC - Threshold strategy with low-latency focus.

Design goals:
- Fast market checks (batch orderbook call)
- Deterministic one-trade-per-market flow
- FAK exit with retry to close residual size
- Logging layout similar to bot1
"""
import logging
import asyncio
import json
import threading
import time
from collections import deque
from datetime import datetime, timezone, timedelta
from decimal import Decimal, ROUND_DOWN
from pathlib import Path
from typing import Optional

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import ApiCreds, BalanceAllowanceParams, BookParams, OrderArgs, OrderType

from shared.config import Config
from shared.market_finder import find_active_5min_market
from shared.trade_logger import TradeLogger

try:
    import websockets
except Exception:
    websockets = None


LOG_DIR = Path("logs/bot3_btc")
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "bot3_btc.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)


def _normalize_price_level(value: float) -> float:
    """Accept both decimal (0.35) and cents (35)."""
    return value / 100.0 if value > 1 else value


class Bot3BTC:
    def __init__(self):
        Config.validate()
        self.client = self._build_client()

        # Strategy config
        self.entry_capital = Config.BOT3_ENTRY_CAPITAL
        self.entry_price = _normalize_price_level(Config.BOT3_ENTRY_PRICE)
        self.take_profit = _normalize_price_level(Config.BOT3_TAKE_PROFIT)
        self.stop_loss = _normalize_price_level(Config.BOT3_STOP_LOSS)
        self.check_interval = max(0.10, float(Config.BOT3_CHECK_INTERVAL))
        self.log_every_seconds = max(1.0, float(Config.BOT3_LOG_EVERY_SECONDS))
        self.market_start_delay = max(0, int(Config.BOT3_MARKET_START_DELAY))
        self.one_trade_per_market = Config.BOT3_ONE_TRADE_PER_MARKET
        self.require_entry_crossing = Config.BOT3_REQUIRE_ENTRY_CROSSING
        self.entry_crossing_ticks = max(1, int(Config.BOT3_ENTRY_CROSSING_TICKS))
        self.trading_enabled = Config.BOT3_TRADING_ENABLED
        self.buy_buffer = max(1.0, float(Config.BOT3_BUY_BUFFER))
        self.order_wait_seconds = max(0.2, float(Config.BOT3_ORDER_WAIT_SECONDS))
        self.exit_order_wait_seconds = max(0.1, float(Config.BOT3_EXIT_ORDER_WAIT_SECONDS))
        self.order_type_default = self._resolve_order_type(Config.BOT3_ORDER_TYPE)
        self.order_type_entry = self._resolve_order_type(
            getattr(Config, "BOT3_ORDER_TYPE_ENTRY", "GTC")
        )
        self.order_type_exit = self._resolve_order_type(
            getattr(Config, "BOT3_ORDER_TYPE_EXIT", "FAK")
        )
        self.ws_enabled = Config.BOT3_USE_WEBSOCKET and websockets is not None
        self.ws_url = str(Config.BOT3_WS_URL)
        self.ws_stale_seconds = max(0.5, float(Config.BOT3_WS_STALE_SECONDS))
        self.exit_max_retries = max(1, int(Config.BOT3_EXIT_MAX_RETRIES))
        self.exit_retry_interval = max(0.10, float(Config.BOT3_EXIT_RETRY_INTERVAL))
        self.exit_balance_retry_seconds = max(0.20, float(Config.BOT3_EXIT_BALANCE_RETRY_SECONDS))
        self.exit_balance_max_wait_seconds = max(1.0, float(Config.BOT3_EXIT_BALANCE_MAX_WAIT_SECONDS))
        self.exit_require_spendable_balance = Config.BOT3_EXIT_REQUIRE_SPENDABLE_BALANCE
        # Keep a practical floor for dust residuals to avoid endless tiny retries.
        self.exit_remaining_tolerance = max(0.01, float(Config.BOT3_EXIT_REMAINING_TOLERANCE))
        self.post_entry_sl_ticks = max(0, int(Config.BOT3_POST_ENTRY_SL_TICKS))

        # Runtime state
        self.token_ids = {}  # {'Up': token_id, 'Down': token_id}
        self.outcomes = []
        self.current_market_name = ""
        self.position = None
        self.position_outcome = None
        self.position_size = 0.0
        self.position_token_id = None
        self.position_entered_at_ts = 0.0
        self.post_entry_tick_count = 0  # ticks elapsed since last entry
        self.price_history: dict[str, deque] = {}  # outcome -> deque of recent ask prices
        self.traded_in_current_market = False
        self.market_entry_locked = False
        self.failed_exit_position = None
        self.last_status_log_ts = 0.0
        self.last_entry_attempt_ts = 0.0
        self.entry_cooldown_seconds = 1.5
        self._ws_lock = threading.Lock()
        self._ws_stop_event = threading.Event()
        self._ws_thread = None
        self._ws_prices_by_token = {}
        self._ws_last_msg_ts = 0.0
        self._ws_connected = False

        self.trade_logger = TradeLogger(logs_dir="logs/bot3_btc")

        logger.info("=" * 60)
        logger.info("🤖 BOT3 BTC INIZIALIZZATO")
        logger.info("=" * 60)
        logger.info(f"💰 Capital/trade: ${self.entry_capital:.2f}")
        logger.info(f"🎯 Entry: {self.entry_price:.3f}")
        logger.info(f"🎯 TP: {self.take_profit:.3f}")
        logger.info(f"🛑 SL: {self.stop_loss:.3f}")
        logger.info(f"⚡ Check: {self.check_interval:.3f}s")
        logger.info(f"🔌 Trading enabled: {self.trading_enabled}")
        logger.info(f"📝 Order type default: {self.order_type_default}")
        logger.info(f"📝 Order type entry: {self.order_type_entry}")
        logger.info(f"📝 Order type exit: {self.order_type_exit}")
        logger.info(f"📡 WebSocket feed enabled: {self.ws_enabled}")
        logger.info(f"🔁 Exit retries: {self.exit_max_retries} every {self.exit_retry_interval:.2f}s")
        logger.info(
            f"🔄 Balance-sync retry: {self.exit_balance_retry_seconds:.2f}s up to {self.exit_balance_max_wait_seconds:.1f}s"
        )
        logger.info(f"🧾 Require spendable balance before sell: {self.exit_require_spendable_balance}")
        logger.info(
            f"⏳ Post-entry SL delay: {self.post_entry_sl_ticks} ticks "
            f"(≈{self.post_entry_sl_ticks * self.check_interval:.2f}s) | TP always active"
        )
        if not self.trading_enabled:
            logger.warning("🧪 NO-TRADE MODE ATTIVO: ordini disabilitati, solo log/CSV prezzi")
        logger.info("=" * 60)

    def _build_client(self) -> ClobClient:
        private_key = Config.PRIVATE_KEY
        if not private_key.startswith("0x"):
            private_key = "0x" + private_key

        temp_client = ClobClient(
            host="https://clob.polymarket.com",
            key=private_key,
            chain_id=Config.CHAIN_ID,
        )
        api_creds = temp_client.create_or_derive_api_creds()
        return ClobClient(
            host="https://clob.polymarket.com",
            key=private_key,
            chain_id=Config.CHAIN_ID,
            creds=api_creds,
            signature_type=2,
            funder=Config.WALLET_ADDRESS,
        )

    def _resolve_order_type(self, order_type_raw: str) -> OrderType:
        mapping = {
            "GTC": OrderType.GTC,
            "FOK": OrderType.FOK,
            "FAK": OrderType.FAK,
            "GTD": OrderType.GTD,
        }
        if order_type_raw in mapping:
            return mapping[order_type_raw]
        logger.warning(f"⚠️ BOT3_ORDER_TYPE '{order_type_raw}' non valido, uso FAK")
        return OrderType.FAK

    def _extract_trade_shares(self, payload: dict, side: str) -> Optional[float]:
        if not isinstance(payload, dict):
            return None
        side_u = side.upper()
        if side_u == "BUY":
            # BUY: shares ricevute stanno tipicamente lato taker amount.
            keys = ("takingAmount", "taking_amount", "size", "filled_size", "amount", "makingAmount", "making_amount")
        else:
            # SELL: shares vendute stanno tipicamente lato maker amount.
            keys = ("makingAmount", "making_amount", "size", "filled_size", "amount", "takingAmount", "taking_amount")

        for key in keys:
            value = payload.get(key)
            if value is None or value == "":
                continue
            try:
                shares = float(value)
                if shares > 0:
                    return shares
            except (TypeError, ValueError):
                continue
        return None

    def _submit_order(self, side: str, token_id: str, size: float, price: float, order_type: Optional[OrderType] = None, wait_seconds: Optional[float] = None):
        """Submit order and return (ok, order_id, filled_shares, raw_payload)."""
        effective_order_type = order_type or self.order_type_default
        q_price, q_size = self._quantize_order_inputs(side=side, price=price, size=size)
        if q_size <= 0:
            return False, None, None, {"error": "invalid_size_after_quantization"}

        try:
            order = OrderArgs(token_id=token_id, side=side, size=q_size, price=q_price)
            signed = self.client.create_order(order)
            result = self.client.post_order(signed, effective_order_type)
        except Exception as exc:
            logger.warning(f"⚠️ submit order failed ({side}): {exc}")
            return False, None, None, {"error": str(exc)}

        if isinstance(result, dict):
            status = str(result.get("status", "")).lower()
            order_id = result.get("orderID")
            filled = self._extract_trade_shares(result, side=side)

            # Guardrail: se l'API restituisce una metrica in unità errata, non
            # sottostimare la posizione. Su status matched ci aspettiamo ~q_size.
            if status == "matched" and filled and q_size > 0 and filled < (q_size * 0.5):
                logger.warning(
                    "⚠️ filled_shares sospette (%s %.4f vs req %.4f). Uso size richiesta.",
                    side,
                    filled,
                    q_size,
                )
                filled = q_size

            if status == "matched":
                return True, order_id, filled, result

            if status == "live" and order_id:
                time.sleep(wait_seconds if wait_seconds is not None else self.order_wait_seconds)
                try:
                    order_status = self.client.get_order(order_id)
                    os_status = str(order_status.get("status", "")).lower() if isinstance(order_status, dict) else ""
                    if os_status == "matched":
                        filled = self._extract_trade_shares(order_status, side=side) or filled
                        if filled and q_size > 0 and filled < (q_size * 0.5):
                            logger.warning(
                                "⚠️ filled_shares sospette da order_status (%s %.4f vs req %.4f). Uso size richiesta.",
                                side,
                                filled,
                                q_size,
                            )
                            filled = q_size
                        return True, order_id, filled, order_status
                    # Order still live: cancel it.
                    try:
                        self.client.cancel(order_id)
                    except Exception:
                        pass
                    # Post-cancel re-check: the order may have filled in the race window
                    # between our status check and the cancel call.
                    time.sleep(0.3)
                    try:
                        post_cancel = self.client.get_order(order_id)
                        pc_status = str(post_cancel.get("status", "")).lower() if isinstance(post_cancel, dict) else ""
                        if pc_status == "matched":
                            filled = self._extract_trade_shares(post_cancel, side=side) or filled or q_size
                            logger.info(
                                "✅ Ordine %s rilevato come matched dopo cancel (race win): %s %.4f",
                                order_id, side, filled,
                            )
                            return True, order_id, filled, post_cancel
                    except Exception:
                        pass
                    return False, order_id, filled, order_status
                except Exception:
                    try:
                        self.client.cancel(order_id)
                    except Exception:
                        pass
                    return False, order_id, filled, result

            return False, order_id, filled, result

        return False, None, None, result

    def _quantize_order_inputs(self, side: str, price: float, size: float) -> tuple[float, float]:
        """
        Fast single-pass normalization (bot1-style behavior, minimal overhead):
        - BUY: clamp quote notional to 2 decimals and derive size with 4 decimals
        - SELL: size to 4 decimals
        - price to 4 decimals
        """
        p = Decimal(str(price)).quantize(Decimal("0.0001"), rounding=ROUND_DOWN)
        s = Decimal(str(size))
        if side.upper() == "BUY":
            notional = (s * p).quantize(Decimal("0.01"), rounding=ROUND_DOWN)
            if p > 0:
                s = (notional / p).quantize(Decimal("0.0001"), rounding=ROUND_DOWN)
            else:
                s = Decimal("0")
        else:
            s = s.quantize(Decimal("0.0001"), rounding=ROUND_DOWN)
        return float(p), float(s)

    def _get_spendable_shares(self, token_id: str) -> Optional[float]:
        """Read spendable conditional token balance from CLOB balance-allowance."""
        try:
            self.client.update_balance_allowance(
                BalanceAllowanceParams(
                    asset_type="CONDITIONAL",
                    token_id=token_id,
                    signature_type=2,
                )
            )
        except Exception:
            pass
        try:
            data = self.client.get_balance_allowance(
                BalanceAllowanceParams(
                    asset_type="CONDITIONAL",
                    token_id=token_id,
                    signature_type=2,
                )
            )
            raw = data.get("balance") if isinstance(data, dict) else None
            if raw is None:
                return None
            as_int = int(str(raw))
            # CLOB API returns conditional balances in 1e6 base units.
            return as_int / 1_000_000.0
        except Exception:
            return None

    def _fetch_books(self):
        if not self.token_ids:
            return {}
        params = [BookParams(token_id=token_id) for token_id in self.token_ids.values()]
        books = self.client.get_order_books(params)
        token_to_outcome = {token_id: outcome for outcome, token_id in self.token_ids.items()}

        prices = {
            outcome: {"token_id": token_id, "ask": None, "bid": None}
            for outcome, token_id in self.token_ids.items()
        }
        unmatched_books = []

        # Prefer deterministic mapping by asset_id/token_id returned by API.
        for book in books:
            asset_id = str(getattr(book, "asset_id", "") or "")
            outcome = token_to_outcome.get(asset_id)
            if not outcome:
                unmatched_books.append(book)
                continue

            best_ask = None
            best_bid = None
            if hasattr(book, "asks") and book.asks:
                try:
                    best_ask = float(book.asks[-1].price)
                except (TypeError, ValueError):
                    best_ask = None
            if hasattr(book, "bids") and book.bids:
                try:
                    best_bid = float(book.bids[0].price)
                except (TypeError, ValueError):
                    best_bid = None

            prices[outcome]["ask"] = best_ask
            prices[outcome]["bid"] = best_bid

        # Fallback only for missing outcomes (defensive).
        if unmatched_books:
            missing_outcomes = [o for o, p in prices.items() if p["ask"] is None and p["bid"] is None]
            for outcome, book in zip(missing_outcomes, unmatched_books):
                best_ask = None
                best_bid = None
                if hasattr(book, "asks") and book.asks:
                    try:
                        best_ask = float(book.asks[-1].price)
                    except (TypeError, ValueError):
                        best_ask = None
                if hasattr(book, "bids") and book.bids:
                    try:
                        best_bid = float(book.bids[0].price)
                    except (TypeError, ValueError):
                        best_bid = None
                prices[outcome]["ask"] = best_ask
                prices[outcome]["bid"] = best_bid

        return prices

    def _has_book_data(self, prices: dict) -> bool:
        if not prices:
            return False
        for payload in prices.values():
            if payload.get("ask") is not None or payload.get("bid") is not None:
                return True
        return False

    def _store_ws_price(self, asset_id: str, best_bid: Optional[float], best_ask: Optional[float]):
        if not asset_id:
            return
        with self._ws_lock:
            snap = self._ws_prices_by_token.get(asset_id, {"bid": None, "ask": None})
            if best_bid is not None:
                snap["bid"] = best_bid
            if best_ask is not None:
                snap["ask"] = best_ask
            self._ws_prices_by_token[asset_id] = snap
            self._ws_last_msg_ts = time.time()

    def _handle_ws_event(self, payload: dict):
        # Full book snapshot payload
        asset_id = str(payload.get("asset_id", "") or "")
        if asset_id and (payload.get("bids") is not None or payload.get("asks") is not None):
            best_bid = None
            best_ask = None
            bids = payload.get("bids") or []
            asks = payload.get("asks") or []
            if bids:
                try:
                    best_bid = max(float(level.get("price")) for level in bids if level.get("price") is not None)
                except Exception:
                    best_bid = None
            if asks:
                try:
                    best_ask = min(float(level.get("price")) for level in asks if level.get("price") is not None)
                except Exception:
                    best_ask = None
            self._store_ws_price(asset_id=asset_id, best_bid=best_bid, best_ask=best_ask)

        # Incremental price change payload
        changes = payload.get("price_changes")
        if isinstance(changes, list):
            for change in changes:
                if not isinstance(change, dict):
                    continue
                cid = str(change.get("asset_id", "") or "")
                if not cid:
                    continue
                best_bid = None
                best_ask = None
                try:
                    if change.get("best_bid") is not None:
                        best_bid = float(change.get("best_bid"))
                except (TypeError, ValueError):
                    best_bid = None
                try:
                    if change.get("best_ask") is not None:
                        best_ask = float(change.get("best_ask"))
                except (TypeError, ValueError):
                    best_ask = None
                self._store_ws_price(asset_id=cid, best_bid=best_bid, best_ask=best_ask)

    async def _ws_session(self, asset_ids: list[str]):
        assert websockets is not None
        async with websockets.connect(self.ws_url, ping_interval=20, ping_timeout=20, close_timeout=5, max_size=None) as ws:
            await ws.send(
                json.dumps(
                    {
                        "type": "market",
                        "assets_ids": asset_ids,
                        "custom_feature_enabled": True,
                    }
                )
            )
            self._ws_connected = True
            while not self._ws_stop_event.is_set():
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=10)
                except asyncio.TimeoutError:
                    try:
                        await ws.send("PING")
                    except Exception:
                        raise
                    continue

                if msg in ("PONG", "PING"):
                    continue

                try:
                    decoded = json.loads(msg)
                except Exception:
                    continue

                if isinstance(decoded, list):
                    for item in decoded:
                        if isinstance(item, dict):
                            self._handle_ws_event(item)
                elif isinstance(decoded, dict):
                    self._handle_ws_event(decoded)

    def _ws_worker(self, asset_ids: list[str]):
        while not self._ws_stop_event.is_set():
            try:
                asyncio.run(self._ws_session(asset_ids))
            except Exception as exc:
                if self._ws_stop_event.is_set():
                    break
                self._ws_connected = False
                logger.warning(f"⚠️ WebSocket disconnected: {exc}. Reconnect in 1s...")
                time.sleep(1.0)
            else:
                break
        self._ws_connected = False

    def _start_ws_feed(self):
        if not self.ws_enabled:
            return
        self._stop_ws_feed()
        asset_ids = [str(token_id) for token_id in self.token_ids.values() if token_id]
        if not asset_ids:
            return
        with self._ws_lock:
            self._ws_prices_by_token = {}
            self._ws_last_msg_ts = 0.0
        self._ws_stop_event.clear()
        self._ws_thread = threading.Thread(target=self._ws_worker, args=(asset_ids,), daemon=True)
        self._ws_thread.start()
        logger.info(f"📡 WS subscribed to {len(asset_ids)} assets")

    def _stop_ws_feed(self):
        self._ws_stop_event.set()
        th = self._ws_thread
        self._ws_thread = None
        if th and th.is_alive():
            th.join(timeout=1.5)
        self._ws_connected = False

    def _fetch_ws_prices(self) -> dict:
        prices = {
            outcome: {"token_id": token_id, "ask": None, "bid": None}
            for outcome, token_id in self.token_ids.items()
        }
        if not self.token_ids:
            return prices
        now = time.time()
        with self._ws_lock:
            stale = (now - self._ws_last_msg_ts) > self.ws_stale_seconds if self._ws_last_msg_ts else True
            token_snapshot = dict(self._ws_prices_by_token)
        if stale:
            return prices
        for outcome, token_id in self.token_ids.items():
            snap = token_snapshot.get(str(token_id))
            if not snap:
                continue
            prices[outcome]["ask"] = snap.get("ask")
            prices[outcome]["bid"] = snap.get("bid")
        return prices

    def _fetch_prices(self) -> tuple[dict, str]:
        """Return (prices, source) where source is 'WS' or 'REST'."""
        if self.ws_enabled:
            ws_prices = self._fetch_ws_prices()
            if self._has_book_data(ws_prices):
                # Fill eventual gaps with REST if WS doesn't have both outcomes yet.
                if any(p.get("ask") is None and p.get("bid") is None for p in ws_prices.values()):
                    try:
                        rest_prices = self._fetch_books()
                    except Exception:
                        rest_prices = {}
                    for outcome, payload in ws_prices.items():
                        if payload.get("ask") is None and payload.get("bid") is None and outcome in rest_prices:
                            ws_prices[outcome] = rest_prices[outcome]
                return ws_prices, "WS"
        return self._fetch_books(), "REST"

    def _entry_signal(self, prices: dict):
        signals = []
        for outcome in self.outcomes:
            payload = prices.get(outcome, {})
            ask = payload.get("ask")
            if ask is None:
                continue
            if self.require_entry_crossing:
                hist = self.price_history.get(outcome)
                # Need at least `entry_crossing_ticks` previous readings, all strictly below threshold
                if hist is None or len(hist) < self.entry_crossing_ticks:
                    continue
                all_below = all(p < self.entry_price for p in hist)
                hit = all_below and ask >= self.entry_price
            else:
                hit = ask >= self.entry_price
            if hit:
                signals.append((outcome, payload.get("token_id"), ask))
        if not signals:
            return None
        signals.sort(key=lambda x: x[2], reverse=True)  # stronger breakout first
        return signals[0]

    def _choose_exit_price(self, prices: dict) -> Optional[float]:
        if not self.position_outcome:
            return None
        payload = prices.get(self.position_outcome, {})
        bid = payload.get("bid")
        ask = payload.get("ask")
        return bid if bid is not None else ask

    def enter_position(self, token_id: str, outcome: str, entry_price: float) -> bool:
        if not self.trading_enabled:
            return False
        if self.position is not None:
            return False
        if self.market_entry_locked:
            return False
        if self.one_trade_per_market and self.traded_in_current_market:
            return False
        if time.time() - self.last_entry_attempt_ts < self.entry_cooldown_seconds:
            return False

        self.last_entry_attempt_ts = time.time()
        notional = self.entry_capital * self.buy_buffer
        size = notional / entry_price
        ok, order_id, filled_shares, payload = self._submit_order(
            "BUY",
            token_id,
            size,
            entry_price,
            order_type=self.order_type_entry,
        )
        # Precision-reject fallback: retry with GTC ONLY if the order was never placed
        # (order_id is None, meaning no fill could have happened server-side).
        if not ok and order_id is None and isinstance(payload, dict):
            err = str(payload.get("error", "")).lower()
            if "invalid amounts" in err or "max accuracy" in err:
                logger.warning("⚠️ BUY rejected for precision (no order placed). Retry once with GTC.")
                ok, order_id, filled_shares, payload = self._submit_order(
                    "BUY",
                    token_id,
                    size,
                    entry_price,
                    order_type=OrderType.GTC,
                )
        if not ok:
            status = payload.get("status") if isinstance(payload, dict) else None
            err = str(payload.get("error", "")) if isinstance(payload, dict) else str(payload)
            logger.warning(
                "⚠️ ENTRY non eseguita %s @ %.3f | order_id=%s status=%s error=%s",
                outcome,
                entry_price,
                order_id,
                status,
                err,
            )
            err_lower = err.lower()
            if "request exception" in err_lower or "timeout" in err_lower:
                # Network error: order may have reached the server unbeknownst to us.
                logger.warning("🔒 Possibile ordine orfano (network error): entry bloccata per questo mercato.")
                self.market_entry_locked = True
            elif order_id is not None:
                # Order was placed (we have an ID) but fill status is uncertain.
                # Lock to prevent a second entry on this market — better to miss
                # one trade than to double up.
                logger.warning("🔒 Ordine %s piazzato ma fill incerto: entry bloccata per sicurezza.", order_id)
                self.market_entry_locked = True
            return False

        shares = filled_shares if filled_shares and filled_shares > 0 else size
        self.position = {
            "order_id": order_id,
            "entry_price": entry_price,
        }
        self.position_outcome = outcome
        self.position_token_id = token_id
        self.position_size = shares
        self.position_entered_at_ts = time.time()
        self.post_entry_tick_count = 0
        self.traded_in_current_market = True

        logger.info(f"✅ ENTRY {outcome} @ {entry_price:.3f} | shares={shares:.4f}")
        self.trade_logger.log_entry(
            outcome=outcome,
            entry_price=entry_price,
            capital=notional,
            shares=shares,
            is_recovery=False,
        )
        return True

    def _choose_exit_trigger_price(self, prices: dict) -> Optional[float]:
        """Trigger price for TP/SL checks (use ASK for consistency with backtest)."""
        if not self.position_outcome:
            return None
        payload = prices.get(self.position_outcome, {})
        ask = payload.get("ask")
        bid = payload.get("bid")
        return ask if ask is not None else bid

    def _exit_reason(self, current_trigger_price: Optional[float]) -> Optional[str]:
        if self.position is None or current_trigger_price is None:
            return None
        # TP is always active from the first tick after entry
        if current_trigger_price >= self.take_profit:
            return "take_profit"
        # SL is delayed: only active after post_entry_sl_ticks price checks
        if self.post_entry_tick_count >= self.post_entry_sl_ticks:
            if current_trigger_price <= self.stop_loss:
                return "stop_loss"
        return None

    def exit_position_with_retry(self, reason: str, prices: dict) -> bool:
        if not self.trading_enabled:
            return False
        if self.position is None or not self.position_token_id:
            return False

        remaining = self.position_size
        entry_price = self.position["entry_price"]
        # Updated only on successful sells; used for logging and P&L.
        last_exit_price: Optional[float] = None
        total_received_usdc: float = 0.0  # accurate revenue across all partial fills

        # Riconcilia una volta sola: usa saldo spendibile reale come base.
        # Questo risolve il mismatch permanente tra size calcolata e fill effettivo.
        already_reconciled = False
        if self.exit_require_spendable_balance:
            spendable_init = self._get_spendable_shares(self.position_token_id)
            if spendable_init is not None and spendable_init > self.exit_remaining_tolerance:
                if spendable_init < remaining:
                    logger.info(
                        "🔧 Riconcilio position_size: calcolato=%.4f → spendibile reale=%.4f",
                        remaining,
                        spendable_init,
                    )
                remaining = spendable_init
                already_reconciled = True

        attempt = 0
        exit_started_at = time.time()
        while True:
            if remaining <= self.exit_remaining_tolerance:
                break
            attempt += 1
            sell_size = remaining

            # In-loop balance check solo se non già riconciliato (es. balance non ancora
            # sincronizzato subito dopo l'entry). Se già riconciliato, evitiamo REST extra.
            if self.exit_require_spendable_balance and not already_reconciled:
                spendable = self._get_spendable_shares(self.position_token_id)
                if spendable is not None:
                    waited = time.time() - exit_started_at
                    if spendable <= self.exit_remaining_tolerance:
                        if waited >= self.exit_balance_max_wait_seconds:
                            # Se dopo lunga attesa non c'è nulla di spendibile, chiudiamo
                            # localmente per evitare lock infinito da mismatch size/filled.
                            logger.warning(
                                "⚠️ Nessun saldo spendibile dopo %.1fs (remaining=%.4f). Riconcilio a 0.",
                                waited,
                                remaining,
                            )
                            remaining = 0.0
                            break
                        logger.warning(
                            "⏳ Attendo sync balance token (need=%.4f, have=%.4f) [%.1fs/%.1fs]",
                            remaining,
                            spendable,
                            waited,
                            self.exit_balance_max_wait_seconds,
                        )
                        time.sleep(self.exit_balance_retry_seconds)
                        prices = self._fetch_books()
                        continue

                    sell_size = min(remaining, spendable)
                    if sell_size + self.exit_remaining_tolerance < remaining:
                        logger.warning(
                            "↘️ Exit parziale per saldo spendibile: target=%.4f spendable=%.4f",
                            remaining,
                            spendable,
                        )

            exit_price = self._choose_exit_price(prices)
            if exit_price is None:
                time.sleep(self.exit_retry_interval)
                prices = self._fetch_books()
                continue

            ok, _, filled_shares, payload = self._submit_order(
                "SELL",
                self.position_token_id,
                sell_size,
                exit_price,
                order_type=self.order_type_exit,
                wait_seconds=self.exit_order_wait_seconds,
            )

            if ok:
                sold = filled_shares if filled_shares and filled_shares > 0 else sell_size
                sold = min(sold, remaining)
                remaining -= sold
                # Track accurate revenue; freeze last_exit_price on successful fills only
                last_exit_price = exit_price
                total_received_usdc += exit_price * sold
                logger.info(f"📤 EXIT attempt {attempt}/{self.exit_max_retries}: sold {sold:.4f}, remaining {remaining:.4f}")
                if remaining <= self.exit_remaining_tolerance:
                    break
                sleep_s = self.exit_retry_interval
            else:
                err = str(payload.get("error", "")) if isinstance(payload, dict) else ""
                err_l = err.lower()
                is_balance_sync = ("not enough balance" in err_l) or ("allowance" in err_l)
                logger.warning(f"⚠️ EXIT attempt {attempt}/{self.exit_max_retries} non eseguito")

                if attempt >= self.exit_max_retries:
                    if is_balance_sync and (time.time() - exit_started_at) < self.exit_balance_max_wait_seconds:
                        logger.warning(
                            "⏳ Balance/allowance non ancora sincronizzati: estendo retry finestra (%.1fs max).",
                            self.exit_balance_max_wait_seconds,
                        )
                    else:
                        break

                sleep_s = self.exit_balance_retry_seconds if is_balance_sync else self.exit_retry_interval

            time.sleep(sleep_s)
            prices = self._fetch_books()

        closed = remaining <= self.exit_remaining_tolerance
        sold_total = self.position_size - max(0.0, remaining)
        pnl = 0.0
        pnl_pct = 0.0
        # Effective exit price: weighted average across all partial fills.
        # Falls back to current market price if no fills occurred (pure reconcilio a 0).
        if last_exit_price is None:
            last_exit_price = self._choose_exit_price(prices) or entry_price
        effective_exit_price = (total_received_usdc / sold_total) if sold_total > 0 and total_received_usdc > 0 else last_exit_price
        if sold_total > 0:
            cost_basis = entry_price * sold_total
            pnl = total_received_usdc - cost_basis
            pnl_pct = (pnl / cost_basis) * 100 if cost_basis > 0 else 0.0
            self.trade_logger.log_exit(
                outcome=self.position_outcome,
                entry_price=entry_price,
                exit_price=effective_exit_price,
                reason=reason if closed else f"{reason}_partial",
                shares=sold_total,
                pnl=pnl,
                pnl_pct=pnl_pct,
            )

        if not closed:
            logger.error(f"❌ Exit incompleta, remaining={remaining:.4f}. Blocco entry nel mercato.")
            self.failed_exit_position = {
                "outcome": self.position_outcome,
                "entry_price": entry_price,
                "remaining_shares": remaining,
                "reason": reason,
            }
            self.market_entry_locked = True
            self.position_size = remaining
        else:
            self.position_size = 0.0

        if closed:
            pnl_sign = "+" if pnl >= 0 else ""
            logger.info(
                "✅ EXIT %s @ %.3f | reason=%s | shares=%.4f | pnl=%s$%.2f (%s%.1f%%)",
                self.position_outcome or "?",
                effective_exit_price,
                reason,
                sold_total,
                pnl_sign,
                pnl,
                pnl_sign,
                pnl_pct,
            )
        self.position = None
        self.position_outcome = None
        self.position_token_id = None
        self.position_entered_at_ts = 0.0
        self.post_entry_tick_count = 0
        return closed

    def reset_for_market(self, market_data: dict):
        self._stop_ws_feed()
        self.token_ids = {
            outcome: token_id for outcome, token_id in zip(market_data["outcomes"], market_data["token_ids"])
        }
        self.outcomes = market_data["outcomes"]
        self.current_market_name = market_data["question"]
        self.position = None
        self.position_outcome = None
        self.position_token_id = None
        self.position_entered_at_ts = 0.0
        self.post_entry_tick_count = 0
        self.position_size = 0.0
        self.price_history = {}
        self.traded_in_current_market = False
        self.market_entry_locked = False
        self.failed_exit_position = None
        self.trade_logger.log_market_start(market_name=self.current_market_name)
        self._start_ws_feed()

    def run_market(self, market_data: dict):
        self.reset_for_market(market_data)
        start_ts = time.time()
        expires_at = datetime.now(timezone.utc) + timedelta(minutes=market_data["minutes_left"])
        max_duration = max(10.0, market_data["minutes_left"] * 60 - 2)

        logger.info("")
        logger.info("=" * 60)
        logger.info(f"🚀 BOT3 START | {self.current_market_name}")
        logger.info(f"👀 Outcomes: {' / '.join(self.outcomes)}")
        logger.info(f"⏱️ Check: {self.check_interval:.3f}s | crossing_ticks={self.entry_crossing_ticks}")
        if not self.trading_enabled:
            logger.warning("🧪 Mercato in NO-TRADE MODE: nessuna entry/exit verrà inviata")

        while time.time() - start_ts < max_duration:
            now = time.time()
            elapsed = now - start_ts
            in_delay = elapsed < self.market_start_delay
            time_left = (expires_at - datetime.now(timezone.utc)).total_seconds()

            try:
                prices, price_source = self._fetch_prices()
            except Exception as exc:
                logger.warning(f"⚠️ fetch prices error: {exc}")
                time.sleep(self.check_interval)
                continue

            if not in_delay and self.trading_enabled:
                if self.position is None:
                    if not self.market_entry_locked and (not self.one_trade_per_market or not self.traded_in_current_market):
                        signal = self._entry_signal(prices)
                        if signal:
                            outcome, token_id, ask = signal
                            logger.info(
                                "🎯 Entry signal %s @ %.3f | Up=%.3f Down=%.3f",
                                outcome,
                                ask,
                                prices.get("Up", {}).get("ask") if prices.get("Up", {}).get("ask") is not None else -1.0,
                                prices.get("Down", {}).get("ask") if prices.get("Down", {}).get("ask") is not None else -1.0,
                            )
                            self.enter_position(token_id, outcome, ask)
                else:
                    self.post_entry_tick_count += 1
                    trigger_price = self._choose_exit_trigger_price(prices)
                    reason = self._exit_reason(trigger_price)
                    if reason:
                        logger.info(
                            "🧭 Exit trigger=%s | reason=%s | ask=%.3f bid=%.3f | ticks_in=%d",
                            f"{trigger_price:.3f}" if trigger_price is not None else "NA",
                            reason,
                            prices.get(self.position_outcome, {}).get("ask") if prices.get(self.position_outcome, {}).get("ask") is not None else -1.0,
                            prices.get(self.position_outcome, {}).get("bid") if prices.get(self.position_outcome, {}).get("bid") is not None else -1.0,
                            self.post_entry_tick_count,
                        )
                        self.exit_position_with_retry(reason=reason, prices=prices)

            # Update price history after logic (for crossing detection)
            for outcome, payload in prices.items():
                ask = payload.get("ask")
                if ask is not None:
                    if outcome not in self.price_history:
                        self.price_history[outcome] = deque(maxlen=self.entry_crossing_ticks)
                    self.price_history[outcome].append(ask)

            if now - self.last_status_log_ts >= self.log_every_seconds:
                self.last_status_log_ts = now
                status_prices = " | ".join(
                    f"{o}:{prices[o]['ask']:.3f}" if prices.get(o, {}).get("ask") is not None else f"{o}:NA"
                    for o in self.outcomes
                )
                logger.info(
                    f"📊 [{price_source}] {status_prices} | left={max(0, int(time_left))}s | traded={self.traded_in_current_market} | pos={self.position_outcome or '-'}"
                )

            time.sleep(self.check_interval)

        logger.info("⏰ Fine mercato (buffer scadenza)")
        self._stop_ws_feed()


def find_next_market(last_timestamp: Optional[int] = None) -> dict:
    attempt = 0
    while True:
        attempt += 1
        market_data = find_active_5min_market(last_timestamp=last_timestamp) if last_timestamp else find_active_5min_market()
        if market_data and market_data.get("minutes_left", 0) >= 0.25:
            return market_data
        if attempt % 3 == 0:
            logger.info("⏳ Nessun mercato valido, riprovo tra 5s...")
        time.sleep(5)


def _run_end_of_session(bot: "Bot3BTC") -> None:
    """
    Called once at the end of every bot run (Ctrl-C or fatal error):
      Fetch real trades from Polymarket API, log the reconciliation summary
      to the session log, and save a standalone history file.
    """
    from shared.session_reconciler import fetch_activities_since, build_text_report

    # Fetch & reconcile real trade history
    logger.info("🔍 Fetch storico reale da Polymarket API...")
    try:
        wallet = Config.WALLET_ADDRESS
        if not wallet:
            logger.warning("   ⚠️ WALLET_ADDRESS non configurato, skip fetch storico")
            return

        since_ts = bot.trade_logger.session_start_ts
        logger.info(f"   Wallet  : {wallet}")
        logger.info(f"   Da      : {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(since_ts))}")

        activities = fetch_activities_since(wallet, since_ts)
        logger.info(f"   Trovate : {len(activities)} attività")

        text = build_text_report(
            activities,
            session_start_ts=since_ts,
            session_log_pnl=bot.trade_logger.session_pnl,
        )

        bot.trade_logger.log_session_summary()
        bot.trade_logger.log_real_history_summary(text)

        logger.info(f"   ✅ Riepilogo scritto nel log di sessione")
        logger.info(f"   ✅ History file: {bot.trade_logger.history_file}")

    except Exception as exc:
        logger.warning(f"   ⚠️ Errore fetch storico: {exc}", exc_info=True)

    logger.info("=" * 60)


def main():
    bot = Bot3BTC()
    last_timestamp = None

    try:
        while True:
            market_data = find_next_market(last_timestamp=last_timestamp)
            Config.BTC_5MIN_MARKET_ID = market_data["condition_id"]
            Config.TOKEN_ID = market_data["token_ids"][0]
            last_timestamp = market_data.get("timestamp")
            bot.run_market(market_data)
    except KeyboardInterrupt:
        logger.info("🛑 Bot3 fermato dall'utente")
    except Exception as exc:
        logger.error(f"❌ Errore fatale Bot3: {exc}", exc_info=True)
    finally:
        bot._stop_ws_feed()
        _run_end_of_session(bot)


if __name__ == "__main__":
    main()
