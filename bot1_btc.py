"""
Bot1 BTC — Maker-only strategy for Polymarket BTC 5-min up/down markets.

Architecture:
  1. ENTRY  — Place maker BUY LIMIT at entry_price when ask crosses above it.
              post_only=True guarantees 0 % fee.
  2. TP     — Place maker SELL LIMIT at take_profit immediately after entry fill.
              post_only=True guarantees 0 % fee.
  3. SL     — Monitor bid via WebSocket; when bid <= stop_loss:
              a) cancel TP
              b) place maker SELL at (stop_loss + offset) with post_only
              c) if not filled within timeout → taker FAK fallback

State machine per market:
  WAIT_SIGNAL → ENTRY_LIVE → TP_LIVE → DONE
                                     ↘ SL_EXIT → DONE

Price feed: WebSocket tick-by-tick stream (TickPriceFeed) with REST fallback.
"""
import asyncio
import json
import os
import time
import logging
import threading
from decimal import Decimal, ROUND_DOWN
from enum import Enum
from typing import Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta

from py_clob_client_v2 import ClobClient, OrderArgs, OrderType, OrderPayload, BalanceAllowanceParams, AssetType
from shared.config import Config
from shared.market_finder import find_active_5min_market
from shared.trade_logger import TradeLogger
from shared.market_result import check_market_winner
from shared.bot1_notifier import Bot1Notifier, CircuitBreakerConfig

try:
    from shared.redeem_relayer import RelayerPositionRedeemer
    RELAYER_AVAILABLE = True
except ImportError:
    RELAYER_AVAILABLE = False

from shared.redeem_positions import PositionRedeemer

# ── Logging ──────────────────────────────────────────────────────────────────

os.makedirs('logs/bot1_btc', exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/bot1_btc/bot1_btc.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)


# ── Data types ───────────────────────────────────────────────────────────────

class Phase(str, Enum):
    WAIT_SIGNAL = "wait_signal"
    ENTRY_LIVE  = "entry_live"
    TP_LIVE     = "tp_live"
    SL_EXIT     = "sl_exit"
    DONE        = "done"


@dataclass
class PendingOrder:
    order_id: str
    token_id: str
    outcome: str
    side: str       # "BUY" or "SELL"
    price: float
    size: float
    placed_at: float = field(default_factory=time.time)
    purpose: str = ""  # "entry", "tp", "sl_maker", "sl_taker"


# ── Quantization (matches bot6/bot3 logic) ───────────────────────────────────

def quantize(side: str, price: float, size: float) -> tuple[float, float]:
    p = Decimal(str(price)).quantize(Decimal("0.0001"), rounding=ROUND_DOWN)
    s = Decimal(str(size))
    if side.upper() == "BUY":
        notional = (s * p).quantize(Decimal("0.01"), rounding=ROUND_DOWN)
        s = (notional / p).quantize(Decimal("0.0001"), rounding=ROUND_DOWN) if p > 0 else Decimal("0")
    else:
        s = s.quantize(Decimal("0.0001"), rounding=ROUND_DOWN)
    return float(p), float(s)


# ── WebSocket price feed (unchanged) ─────────────────────────────────────────

class TickPriceFeed:
    """Real-time WebSocket price feed with tick-level notification."""

    _DEFAULT_WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"

    def __init__(self, ws_url: str = "", stale_seconds: float = 3.0):
        self._ws_url = ws_url or self._DEFAULT_WS_URL
        self._stale_seconds = stale_seconds
        self._prices: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._token_ids: list[str] = []
        self._resub = threading.Event()
        self._stop = threading.Event()
        self._tick = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def subscribe(self, token_ids: list[str]) -> None:
        with self._lock:
            self._token_ids = list(token_ids)
            for tid in token_ids:
                self._prices.setdefault(tid, {"bid": None, "ask": None, "ts": 0.0})
        self._resub.set()
        logger.debug("TickPriceFeed: subscribing to %d tokens", len(token_ids))

    def get_ask(self, token_id: str) -> tuple[Optional[float], bool]:
        with self._lock:
            entry = self._prices.get(token_id)
        if entry is None:
            return None, True
        age = time.time() - entry["ts"]
        stale = age > self._stale_seconds or entry["ts"] == 0.0
        return entry["ask"], stale

    def get_bid(self, token_id: str) -> tuple[Optional[float], bool]:
        with self._lock:
            entry = self._prices.get(token_id)
        if entry is None:
            return None, True
        age = time.time() - entry["ts"]
        stale = age > self._stale_seconds or entry["ts"] == 0.0
        return entry["bid"], stale

    def wait_for_tick(self, timeout: float = 5.0) -> bool:
        triggered = self._tick.wait(timeout)
        self._tick.clear()
        return triggered

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="bot1-tick-feed")
        self._thread.start()
        logger.info("TickPriceFeed started (url=%s)", self._ws_url)

    def stop(self) -> None:
        self._stop.set()
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("TickPriceFeed stopped")

    def _run(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._supervisor())
        finally:
            self._loop.close()

    async def _supervisor(self) -> None:
        while not self._stop.is_set():
            try:
                await self._session()
            except Exception as exc:
                if not self._stop.is_set():
                    logger.warning("TickPriceFeed WS error: %s – reconnecting in 1s", exc)
                    await asyncio.sleep(1.0)

    async def _session(self) -> None:
        import websockets
        async with websockets.connect(
            self._ws_url, ping_interval=20, ping_timeout=10, close_timeout=5,
        ) as ws:
            logger.info("WebSocket connected")
            self._resub.clear()
            await self._send_sub(ws)
            while not self._stop.is_set():
                if self._resub.is_set():
                    self._resub.clear()
                    await self._send_sub(ws)
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
                    self._handle_raw(raw)
                except asyncio.TimeoutError:
                    continue

    async def _send_sub(self, ws) -> None:
        with self._lock:
            tids = list(self._token_ids)
        if tids:
            await ws.send(json.dumps({"assets_ids": tids, "type": "market"}))

    def _handle_raw(self, raw: str) -> None:
        try:
            data = json.loads(raw)
        except Exception:
            return
        items = data if isinstance(data, list) else [data]
        for item in items:
            if isinstance(item, dict):
                self._process_item(item)

    def _process_item(self, item: dict) -> None:
        asset_id = item.get("asset_id") or item.get("market")
        if not asset_id:
            return
        bid: Optional[float] = None
        ask: Optional[float] = None
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
        bids_arr = item.get("bids", [])
        asks_arr = item.get("asks", [])
        if bids_arr and bid is None:
            try:
                bid = max(float(b["price"]) for b in bids_arr
                          if b.get("price") and float(b.get("size", 1)) > 0)
            except (TypeError, ValueError):
                pass
        if asks_arr and ask is None:
            try:
                ask = min(float(a["price"]) for a in asks_arr
                          if a.get("price") and float(a.get("size", 1)) > 0)
            except (TypeError, ValueError):
                pass
        if bid is not None or ask is not None:
            with self._lock:
                entry = self._prices.get(asset_id, {"bid": None, "ask": None, "ts": 0.0})
                if bid is not None:
                    entry["bid"] = bid
                if ask is not None:
                    entry["ask"] = ask
                entry["ts"] = time.time()
                self._prices[asset_id] = entry
            self._tick.set()


# ── Bot1 BTC — Maker strategy ───────────────────────────────────────────────

class Bot1BTC:
    def __init__(self):
        Config.validate()

        private_key = Config.PRIVATE_KEY
        if not private_key.startswith('0x'):
            private_key = '0x' + private_key

        try:
            temp_client = ClobClient(
                host="https://clob.polymarket.com",
                key=private_key,
                chain_id=Config.CHAIN_ID,
            )
            api_creds = temp_client.create_or_derive_api_key()
            self.client = ClobClient(
                host="https://clob.polymarket.com",
                key=private_key,
                chain_id=Config.CHAIN_ID,
                creds=api_creds,
                signature_type=2,
                funder=Config.WALLET_ADDRESS,
            )
            logger.info("Client autenticato con signer: %s...", Config.WALLET_ADDRESS[:10])
        except Exception as e:
            raise RuntimeError(f"Impossibile inizializzare client: {e}") from e

        # ── Trading parameters ────────────────────────────────────────────
        self.market_id = Config.BTC_5MIN_MARKET_ID
        self.token_id = Config.TOKEN_ID
        self.entry_capital = Config.BOT1_ENTRY_CAPITAL
        self.entry_price = Config.BOT1_ENTRY_PRICE
        self.stop_loss = Config.BOT1_STOP_LOSS
        self.take_profit = Config.BOT1_TAKE_PROFIT

        # Maker-specific
        self.post_only = Config.BOT1_POST_ONLY
        self.order_poll_seconds = Config.BOT1_ORDER_POLL_SECONDS
        self.entry_timeout = Config.BOT1_ENTRY_TIMEOUT_SECONDS
        self.entry_taker_fallback = Config.BOT1_ENTRY_TAKER_FALLBACK
        self.sl_maker_offset = Config.BOT1_SL_MAKER_OFFSET
        self.sl_maker_timeout = Config.BOT1_SL_MAKER_TIMEOUT

        # Both tokens (set by main loop)
        self.token_ids: dict[str, str] = {}
        self.outcomes: list[str] = []

        # Strategy flags
        self.one_trade_per_market = Config.BOT1_ONE_TRADE_PER_MARKET
        self.market_start_delay = Config.BOT1_MARKET_START_DELAY
        self.claim_interval = Config.BOT1_CLAIM_INTERVAL
        self.win_threshold = Config.BOT1_WIN_THRESHOLD

        # Recovery
        self.recovery_enabled = Config.BOT1_RECOVERY_ENABLED
        self.recovery_multiplier = Config.BOT1_RECOVERY_MULTIPLIER
        self.recovery_time_window = Config.BOT1_RECOVERY_TIME_WINDOW
        self.recovery_max_attempts = Config.BOT1_RECOVERY_MAX_ATTEMPTS
        self.recovery_entry_price = Config.BOT1_RECOVERY_ENTRY_PRICE
        self.recovery_stop_loss = Config.BOT1_RECOVERY_STOP_LOSS

        # ── Per-market mutable state (reset each market) ──────────────────
        self.phase = Phase.WAIT_SIGNAL
        self._entry_order: Optional[PendingOrder] = None
        self._tp_order: Optional[PendingOrder] = None
        self._sl_order: Optional[PendingOrder] = None
        self._active_outcome: Optional[str] = None
        self._active_token_id: Optional[str] = None
        self._entry_price_actual: float = 0.0
        self._position_shares: float = 0.0
        self._is_recovery: bool = False
        self._traded_in_current_market = False
        self._recovery_done_this_market = False
        self._sl_triggered_this_market = False
        self._failed_outcome: Optional[str] = None
        self._recovery_count = 0
        self.last_seen_price: Optional[float] = None
        self._failed_exit_position: Optional[dict] = None
        self._tp_retry_last_at: float = 0.0
        self._tp_retry_attempts: int = 0

        # Balance-driven truth tracking. These are the ONLY source of truth
        # for fills/exits — never trust order POST/poll responses for size.
        self._pre_entry_pusd: float = 0.0   # pUSD balance snapshot before BUY
        self._pre_entry_cond: float = 0.0   # CONDITIONAL token balance snapshot before BUY
        self._entry_baseline_taken: bool = False
        # When BUY fails for locked pUSD (matched orders), pause retries — avoids API spam.
        self._entry_funds_backoff_until: float = 0.0
        self._last_insufficient_funds_log: float = 0.0
        self._last_order_error: Optional[str] = None  # set by _place_limit on failure

        # Throttle CLOB balance-allowance *updates* (not reads) — prevents CF 1015 / 429.
        self._last_balance_allowance_update: float = 0.0
        self._balance_cooldown_until: float = 0.0
        self._balance_update_min_interval: float = Config.BOT1_BALANCE_UPDATE_MIN_INTERVAL
        self._balance_cooldown_on_429: float = Config.BOT1_BALANCE_COOLDOWN_ON_429

        # Claim
        self.past_market_ids: list[str] = []

        # Logging / notifications
        self.trade_logger = TradeLogger()
        self.notifier = Bot1Notifier(
            token=Config.BOT1_TELEGRAM_TOKEN,
            chat_id=Config.BOT1_TELEGRAM_CHAT_ID,
            enabled=Config.BOT1_TELEGRAM_ENABLED,
            cb_config=CircuitBreakerConfig(
                alert_daily_loss=Config.BOT1_CB_ALERT_DAILY_LOSS,
                alert_drawdown=Config.BOT1_CB_ALERT_DRAWDOWN,
                alert_streak=Config.BOT1_CB_ALERT_STREAK,
                kill_daily_loss=Config.BOT1_CB_KILL_DAILY_LOSS,
                kill_drawdown=Config.BOT1_CB_KILL_DRAWDOWN,
                kill_streak=Config.BOT1_CB_KILL_STREAK,
            ),
            target_entry=self.entry_price,
            target_tp=self.take_profit if self.take_profit else 0.99,
            target_sl=self.stop_loss,
        )

        # Redeemer
        self.redeemer = None
        self.auto_claim_enabled = False
        if RELAYER_AVAILABLE and Config.BUILDER_API_KEY:
            try:
                self.redeemer = RelayerPositionRedeemer()
                self.auto_claim_enabled = True
                logger.info("Relayer Redeemer inizializzato (gasless)")
            except Exception as e:
                logger.warning("Impossibile inizializzare Relayer: %s", e)
        if not self.redeemer:
            try:
                self.redeemer = PositionRedeemer()
                if hasattr(self.redeemer, 'eoa_address') and hasattr(self.redeemer, 'actual_wallet'):
                    if self.redeemer.eoa_address.lower() != self.redeemer.actual_wallet.lower():
                        logger.warning("PROXY WALLET: claim automatico probabilmente non funzionerà")
                        self.auto_claim_enabled = False
                    else:
                        self.auto_claim_enabled = True
                else:
                    self.auto_claim_enabled = True
            except Exception:
                self.redeemer = None
                self.auto_claim_enabled = False

        # WebSocket price feed
        self.price_feed = TickPriceFeed(
            ws_url=Config.BOT1_WS_URL,
            stale_seconds=Config.BOT1_WS_STALE_SECONDS,
        )
        self.price_feed.start()

        logger.info("=" * 60)
        logger.info("BOT1 BTC INIZIALIZZATO — MAKER STRATEGY")
        logger.info("=" * 60)
        logger.info("  Entry:     %s (maker, post_only=%s)", self.entry_price, self.post_only)
        logger.info("  TP:        %s (maker)", self.take_profit)
        logger.info("  SL:        %s (maker+taker fallback)", self.stop_loss)
        logger.info("  Capital:   $%s/trade", self.entry_capital)
        logger.info("  SL offset: %s | SL timeout: %ss", self.sl_maker_offset, self.sl_maker_timeout)
        logger.info("  Entry timeout: %ss | Taker fallback: %s", self.entry_timeout, self.entry_taker_fallback)
        logger.info("  Telegram:  %s", 'ON' if self.notifier._tg.enabled else 'OFF')

        # Polymarket V2 CTF Exchange address — the spender we care about for trading
        EXCH_V2 = "0xE111180000d2663C0091e4f400237545B87B996B"
        try:
            params = BalanceAllowanceParams(asset_type=AssetType.COLLATERAL, signature_type=2)
            self._maybe_update_balance_allowance(params)
            bal_resp = self.client.get_balance_allowance(params)
            pusd_balance = float(bal_resp.get("balance", 0)) / 1_000_000
            # V2 returns 'allowances' as dict {spender_address: raw_amount}
            allowances = bal_resp.get("allowances") or {}
            v2_raw = int(allowances.get(EXCH_V2, 0))
            v2_amount = v2_raw / 1_000_000 if v2_raw < 10**30 else float("inf")
            v2_str = "UNLIMITED" if v2_amount == float("inf") else f"${v2_amount:,.2f}"
            logger.info("  pUSD bal:  $%.2f | allowance V2: %s", pusd_balance, v2_str)
            if pusd_balance < 5:
                logger.warning("  ⚠ Balance basso: gli ordini saranno rigettati per insufficienza fondi")
            if v2_raw == 0:
                logger.warning("  ⚠ Allowance V2 = 0 → ordini falliranno al settlement")
                logger.warning("    Esegui:  venv/bin/python3 _approve_v2.py")
        except Exception as exc:
            logger.info("  Balance check skipped: %s", exc)
        logger.info("=" * 60)

        self.notifier.notify_start(
            capital=self.entry_capital,
            entry=self.entry_price,
            tp=self.take_profit if self.take_profit else 0.99,
            sl=self.stop_loss,
        )

    # ── Price helpers ─────────────────────────────────────────────────────

    def _get_ask(self, token_id: str) -> Optional[float]:
        """Best ask — WS first, REST fallback."""
        ask, stale = self.price_feed.get_ask(token_id)
        if ask is not None and not stale:
            return ask
        try:
            book = self.client.get_order_book(token_id)
            if book and hasattr(book, 'asks') and book.asks:
                return float(book.asks[-1].price)
        except Exception:
            pass
        return None

    def _get_bid(self, token_id: str) -> Optional[float]:
        """Best bid — WS first, REST fallback."""
        bid, stale = self.price_feed.get_bid(token_id)
        if bid is not None and not stale:
            return bid
        try:
            book = self.client.get_order_book(token_id)
            if book and hasattr(book, 'bids') and book.bids:
                return float(book.bids[0].price)
        except Exception:
            pass
        return None

    # ── Order primitives ──────────────────────────────────────────────────

    def _place_limit(self, token_id: str, side: str, price: float,
                     size: float, post_only: bool = False,
                     suppress_log: bool = False) -> Optional[str]:
        """Place a GTC limit order. Returns order_id or None.

        suppress_log: when True, downgrades known-transient errors (post_only
        cross, balance/allowance lag) to debug level; useful during retry loops.
        """
        q_price, q_size = quantize(side, price, size)
        if q_size <= 0:
            logger.warning("Quantized size=0 for %s %.4f×%.4f", side, price, size)
            return None
        self._last_order_error = None
        try:
            args = OrderArgs(token_id=token_id, price=q_price, size=q_size, side=side)
            signed = self.client.create_order(args)
            result = self.client.post_order(signed, OrderType.GTC, post_only=post_only)
        except Exception as exc:
            err = str(exc)
            self._last_order_error = err
            err_lower = err.lower()
            transient = (
                (post_only and ("would match" in err_lower or "post only" in err_lower))
                or "balance is not enough" in err_lower
                or "allowance" in err_lower
            )
            if transient and suppress_log:
                logger.debug("place_limit %s transient: %s", side, exc)
            elif transient and not suppress_log:
                # Known transient — log once at info level
                logger.info("place_limit %s transient: %s", side, err.split(":", 2)[-1].strip()[:80])
            else:
                logger.warning("place_limit %s failed: %s", side, exc)
            return None

        if not isinstance(result, dict):
            return None

        status = str(result.get("status", "")).lower()
        order_id = result.get("orderID") or result.get("order_id")

        if status == "matched":
            logger.info("LIMIT %s %.4f×%.2f → MATCHED immediately (id=%s)",
                        side, q_price, q_size, (order_id or "?")[:12])
            return order_id
        if status == "live" and order_id:
            logger.info("LIMIT %s %.4f×%.2f → LIVE on book (id=%s)",
                        side, q_price, q_size, order_id[:12])
            return order_id

        logger.warning("place_limit: unexpected status '%s' → %s", status, result)
        return None

    def _place_fak_sell(self, token_id: str, price: float, size: float) -> Optional[tuple[str, float]]:
        """Place a Fill-And-Kill sell. Returns (order_id, filled_size) or None."""
        q_price, q_size = quantize("SELL", price, size)
        if q_size <= 0:
            return None
        try:
            args = OrderArgs(token_id=token_id, price=q_price, size=q_size, side="SELL")
            signed = self.client.create_order(args)
            result = self.client.post_order(signed, OrderType.FAK)
        except Exception as exc:
            logger.warning("FAK SELL failed: %s", exc)
            return None

        if not isinstance(result, dict):
            return None

        order_id = result.get("orderID") or result.get("order_id")
        filled = self._extract_fill(result, "SELL")
        return order_id, (filled or 0.0)

    def _poll_order(self, order_id: str) -> tuple[float, bool]:
        """Check order status via REST. Returns (filled_size, is_done)."""
        try:
            data = self.client.get_order(order_id)
            if not data:
                return 0.0, False
            status = (data.get("status") or "").upper()
            filled = float(
                data.get("sizeMatched")
                or data.get("sizeFilled")
                or data.get("size_matched")
                or data.get("size_filled")
                or 0
            )
            done = status in (
                "MATCHED", "FILLED", "CANCELED", "CANCELLED",
                "CANCELED_WITH_PARTIAL_FILL", "CANCELLED_WITH_PARTIAL_FILL",
                "EXPIRED", "REJECTED",
            )
            if done and filled == 0:
                logger.info("poll_order %s status=%s payload=%s",
                            order_id[:12], status, data)
            return filled, done
        except Exception as exc:
            logger.debug("poll_order %s: %s", order_id[:12], exc)
            return 0.0, False

    def _cancel_order(self, order_id: str) -> bool:
        """Cancel an order. Returns True on success."""
        try:
            self.client.cancel_order(OrderPayload(orderID=order_id))
            logger.info("CANCEL %s", order_id[:12])
            return True
        except Exception as exc:
            logger.warning("cancel %s failed: %s", order_id[:12], exc)
            return False

    def _cancel_and_check(self, order_id: str) -> float:
        """Cancel an order and return final filled size (handles cancel-fill race)."""
        self._cancel_order(order_id)
        time.sleep(0.3)
        filled, _ = self._poll_order(order_id)
        return filled

    def _maybe_update_balance_allowance(self, params: BalanceAllowanceParams) -> None:
        """Call update_balance_allowance at most once per min-interval; backoff on 429/CF."""
        now = time.time()
        if now < self._balance_cooldown_until:
            return
        if now - self._last_balance_allowance_update < self._balance_update_min_interval:
            return
        try:
            self.client.update_balance_allowance(params)
            self._last_balance_allowance_update = now
        except Exception as exc:
            err = str(exc).lower()
            if any(x in err for x in ("429", "1015", "rate limit", "cloudflare", "blocked", "temporarily")):
                self._balance_cooldown_until = now + self._balance_cooldown_on_429
                logger.warning(
                    "CLOB rate limited — balance refresh paused %.0fs (%s)",
                    self._balance_cooldown_on_429, exc.__class__.__name__,
                )
            else:
                logger.debug("update_balance_allowance failed (non-fatal): %s", exc)

    def _refresh_conditional_balance(self, token_id: str) -> None:
        """Nudge the exchange backend to refresh CONDITIONAL token balance for token_id."""
        self._maybe_update_balance_allowance(
            BalanceAllowanceParams(
                asset_type=AssetType.CONDITIONAL,
                token_id=token_id,
                signature_type=2,
            )
        )

    # ── Balance-driven primitives (TRUTH SOURCE) ──────────────────────────
    #
    # The Polymarket POST/poll responses cannot be trusted for fill size:
    # they may report fake fills that get rolled back, or partial fills as
    # complete. The exchange's internal balance ledger is the only reliable
    # source. All entry/exit decisions read CONDITIONAL & COLLATERAL balances
    # directly. PnL is measured as real pUSD delta, not theoretical.

    def _get_collateral_balance(self, force_refresh: bool = True) -> float:
        """Read pUSD balance in dollars. Refresh uses throttled update_balance_allowance."""
        try:
            params = BalanceAllowanceParams(asset_type=AssetType.COLLATERAL, signature_type=2)
            if force_refresh:
                self._maybe_update_balance_allowance(params)
            bal = self.client.get_balance_allowance(params)
            return float(bal.get("balance", 0)) / 1_000_000
        except Exception as exc:
            err = str(exc).lower()
            if any(x in err for x in ("429", "1015", "rate limit", "cloudflare")):
                self._balance_cooldown_until = time.time() + self._balance_cooldown_on_429
                logger.warning("get_collateral_balance: CLOB rate limited — %s", exc.__class__.__name__)
            else:
                logger.debug("get_collateral_balance failed: %s", exc)
            return 0.0

    def _get_conditional_balance(self, token_id: str, force_refresh: bool = True) -> float:
        """Read CONDITIONAL token balance in shares for token_id."""
        if not token_id:
            return 0.0
        try:
            params = BalanceAllowanceParams(
                asset_type=AssetType.CONDITIONAL, token_id=token_id, signature_type=2,
            )
            if force_refresh:
                self._maybe_update_balance_allowance(params)
            bal = self.client.get_balance_allowance(params)
            return float(bal.get("balance", 0)) / 1_000_000
        except Exception as exc:
            err = str(exc).lower()
            if any(x in err for x in ("429", "1015", "rate limit", "cloudflare")):
                self._balance_cooldown_until = time.time() + self._balance_cooldown_on_429
                logger.warning("get_conditional_balance: CLOB rate limited — %s", exc.__class__.__name__)
            else:
                logger.debug("get_conditional_balance failed: %s", exc)
            return 0.0

    def _wait_for_balance_change(self, token_id: str, baseline: float,
                                 min_delta: float = 0.0001, timeout: float = 8.0,
                                 poll_interval: float = 1.0) -> float:
        """Poll CONDITIONAL balance until it differs from baseline by min_delta.

        Returns the absolute delta (current_balance - baseline) at the moment
        the threshold is crossed, or 0.0 on timeout. Used to confirm a fill
        is real, not just an API illusion.
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            cur = self._get_conditional_balance(token_id, force_refresh=True)
            delta = cur - baseline
            if abs(delta) >= min_delta:
                return delta
            time.sleep(poll_interval)
        return 0.0

    def _force_cancel_verify(self, order_id: str, max_wait: float = 8.0) -> tuple[bool, float]:
        """Cancel an order and BLOCK until the cancel is confirmed by the API.

        Critical for avoiding orphaned orders that fill after we've moved on.
        Returns (canceled, final_filled_size). If max_wait elapses without
        confirmation, returns (False, last_seen_filled).
        """
        try:
            self.client.cancel_order(OrderPayload(orderID=order_id))
            logger.info("CANCEL %s (verifying)", order_id[:12])
        except Exception as exc:
            logger.warning("cancel %s failed: %s", order_id[:12], exc)

        deadline = time.time() + max_wait
        last_filled = 0.0
        while time.time() < deadline:
            time.sleep(0.4)
            try:
                data = self.client.get_order(order_id)
                if not data:
                    return True, last_filled
                status = (data.get("status") or "").upper()
                filled = float(
                    data.get("sizeMatched") or data.get("size_matched")
                    or data.get("sizeFilled") or data.get("size_filled") or 0
                )
                if filled > last_filled:
                    last_filled = filled
                if status in ("CANCELED", "CANCELLED",
                              "CANCELED_WITH_PARTIAL_FILL", "CANCELLED_WITH_PARTIAL_FILL",
                              "MATCHED", "FILLED", "EXPIRED", "REJECTED"):
                    return True, last_filled
                # Still LIVE / DELAYED — re-issue cancel
                try:
                    self.client.cancel_order(OrderPayload(orderID=order_id))
                except Exception:
                    pass
            except Exception as exc:
                logger.debug("force_cancel_verify poll: %s", exc)
        logger.error("FORCE-CANCEL %s NOT CONFIRMED after %.0fs", order_id[:12], max_wait)
        return False, last_filled

    @staticmethod
    def _extract_fill(payload: dict, side: str) -> Optional[float]:
        if side.upper() == "BUY":
            keys = ("takingAmount", "taking_amount", "size", "filled_size",
                    "makingAmount", "making_amount")
        else:
            keys = ("makingAmount", "making_amount", "size", "filled_size",
                    "takingAmount", "taking_amount")
        for k in keys:
            v = payload.get(k)
            if v is None or v == "":
                continue
            try:
                f = float(v)
                if f > 0:
                    return f
            except (TypeError, ValueError):
                continue
        return None

    # ── Phase: WAIT_SIGNAL ────────────────────────────────────────────────

    def _tick_wait_signal(self, log: bool = False) -> None:
        """Monitor both tokens. When ask crosses above entry_price, place maker BUY."""
        target_entry = self.recovery_entry_price if self._is_recovery else self.entry_price
        target_outcome = None
        target_token = None

        if self._is_recovery and self._failed_outcome:
            opposite = 'Up' if self._failed_outcome == 'Down' else 'Down'
            if opposite in self.token_ids:
                target_outcome = opposite
                target_token = self.token_ids[opposite]
        else:
            for outcome, tid in self.token_ids.items():
                ask = self._get_ask(tid)
                if ask is not None:
                    if log:
                        logger.info("  %s ask: %.3f (target: %.3f)", outcome, ask, target_entry)
                    if ask > target_entry:
                        target_outcome = outcome
                        target_token = tid
                        break
            if target_outcome is None:
                return

        if self._is_recovery:
            ask = self._get_ask(target_token)
            if ask is None or ask <= target_entry:
                if log:
                    logger.info("  Recovery %s ask: %s (target: %.3f)", target_outcome, ask, target_entry)
                return

        # Single-flight invariant: never place a 2nd entry if one is already
        # live or being processed. This prevents the "double-BUY" bug where
        # an abandoned-as-fake order fills later alongside a fresh one.
        if self._entry_order is not None:
            logger.warning("Skipping entry — _entry_order already tracked (%s)",
                           self._entry_order.order_id[:12])
            return

        now = time.time()
        if now < self._entry_funds_backoff_until:
            return

        capital = self.entry_capital * (self.recovery_multiplier if self._is_recovery else 1.0)
        size = capital / target_entry

        # Capture pre-entry baselines BEFORE the order goes out. These are
        # the truth source against which the eventual fill is measured.
        self._pre_entry_pusd = self._get_collateral_balance(force_refresh=True)
        self._pre_entry_cond = self._get_conditional_balance(target_token, force_refresh=True)
        self._entry_baseline_taken = True

        logger.info(
            "ASK > entry (%.3f) on %s — placing maker BUY @ %.4f "
            "(baseline pUSD=$%.4f cond=%.4f)",
            target_entry, target_outcome, target_entry,
            self._pre_entry_pusd, self._pre_entry_cond,
        )

        order_id = self._place_limit(
            target_token, "BUY", target_entry, size, post_only=self.post_only,
        )

        if order_id:
            self._entry_funds_backoff_until = 0.0
            self._entry_order = PendingOrder(
                order_id=order_id, token_id=target_token,
                outcome=target_outcome, side="BUY",
                price=target_entry, size=size, purpose="entry",
            )
            self._active_outcome = target_outcome
            self._active_token_id = target_token
            self.phase = Phase.ENTRY_LIVE
            logger.info("Phase → ENTRY_LIVE (%s BUY @ %.4f, id=%s)",
                        target_outcome, target_entry, order_id[:12])
        else:
            err = (self._last_order_error or "").lower()
            collateral_lock = bool(err) and (
                "sum of matched orders" in err
                or ("not enough balance" in err and "matched orders" in err)
            )
            if collateral_lock:
                # pUSD largely locked in other open/matched orders — retrying every
                # WS tick hammers Polymarket and spams logs. Back off and tell user.
                self._entry_funds_backoff_until = now + 45.0
                if now - self._last_insufficient_funds_log >= 120.0:
                    self._last_insufficient_funds_log = now
                    logger.warning(
                        "BUY bloccato: pUSD libero insufficiente (spesso ~$10 già in "
                        "'matched orders'). Cancella ordini aperti su Polymarket o "
                        "attendi 45s — niente spam di retry.",
                    )
            else:
                logger.debug("Maker BUY rejected or failed — will retry next tick")

    # ── Phase: ENTRY_LIVE ─────────────────────────────────────────────────

    def _tick_entry_live(self) -> None:
        """Balance-driven entry confirmation.

        TRUTH SOURCE: CONDITIONAL token balance vs. pre-entry baseline.
        We treat poll_order's 'size_matched' as a hint only — Polymarket has
        been observed to report fake matches that don't settle. Final word
        always comes from update_balance_allowance(CONDITIONAL).
        """
        eo = self._entry_order
        if eo is None:
            self.phase = Phase.WAIT_SIGNAL
            return

        # Read real balance. If shares appeared → real fill.
        cond_now = self._get_conditional_balance(eo.token_id, force_refresh=True)
        actual_filled = max(0.0, cond_now - self._pre_entry_cond)

        # Also poll order status — used to know when to stop waiting.
        poll_filled, poll_done = self._poll_order(eo.order_id)

        # Case 1: Real fill confirmed via balance. Trust it.
        if actual_filled > 0:
            # If poll says still LIVE and partial, give it 1-2s to grow.
            if not poll_done and actual_filled < eo.size * 0.99:
                time.sleep(1.0)
                cond_now2 = self._get_conditional_balance(eo.token_id, force_refresh=True)
                grew = max(0.0, cond_now2 - self._pre_entry_cond)
                if grew > actual_filled:
                    actual_filled = grew
            # If poll is done (any status) OR fill is "complete enough" → finalize.
            # Otherwise wait next tick for more fills.
            if poll_done or actual_filled >= eo.size * 0.99:
                # Final cancel-verify if order isn't yet done in API to avoid
                # later additional fills on the same order.
                if not poll_done:
                    self._force_cancel_verify(eo.order_id, max_wait=3.0)
                    cond_final = self._get_conditional_balance(eo.token_id, force_refresh=True)
                    actual_filled = max(actual_filled, cond_final - self._pre_entry_cond)
                self._on_entry_filled(eo, actual_filled)
                return

        # Case 2: Order is "done" per poll, but balance shows nothing yet.
        # Could be cache lag (rare) OR a fake match. Wait briefly to be sure.
        if poll_done and actual_filled == 0:
            deadline = time.time() + 4.0
            while time.time() < deadline:
                time.sleep(0.5)
                cond_now = self._get_conditional_balance(eo.token_id, force_refresh=True)
                actual_filled = max(0.0, cond_now - self._pre_entry_cond)
                if actual_filled > 0:
                    self._on_entry_filled(eo, actual_filled)
                    return
            logger.warning(
                "Entry %s reported done but balance unchanged — confirmed fake fill",
                eo.order_id[:12],
            )
            self._entry_order = None
            self.phase = Phase.WAIT_SIGNAL
            return

        # Case 3: Timeout. CRITICAL: must force-cancel-verify to prevent
        # the order filling after we move on (root cause of double-BUY bug).
        elapsed = time.time() - eo.placed_at
        if elapsed > self.entry_timeout:
            logger.info("Entry timeout (%.0fs) — force cancelling + verifying", elapsed)
            canceled, _ = self._force_cancel_verify(eo.order_id, max_wait=10.0)

            # Always re-read balance: cancel may have raced with a fill.
            time.sleep(0.5)
            cond_now = self._get_conditional_balance(eo.token_id, force_refresh=True)
            actual_filled = max(0.0, cond_now - self._pre_entry_cond)

            if actual_filled > 0:
                logger.info("Captured race-fill via balance: %.4f shares", actual_filled)
                self._on_entry_filled(eo, actual_filled)
                return

            if not canceled:
                # Cancel still not confirmed → KEEP TRACKING. Do NOT reset
                # _entry_order or we risk a second concurrent BUY. Retry next tick.
                logger.error(
                    "Entry %s cancel unconfirmed — keeping order tracked, will retry",
                    eo.order_id[:12],
                )
                eo.placed_at = time.time() - self.entry_timeout + 5  # retry in 5s
                return

            logger.info("Entry not filled — back to WAIT_SIGNAL")
            self._entry_order = None
            self.phase = Phase.WAIT_SIGNAL
            return

    def _on_entry_filled(self, eo: PendingOrder, filled: float) -> None:
        """Handle a confirmed (balance-verified) entry: record position, place TP."""
        self._entry_order = None

        self._entry_price_actual = eo.price
        self._position_shares = filled  # always the real, balance-verified amount
        self._traded_in_current_market = True
        if self._is_recovery:
            self._recovery_done_this_market = True
            self._recovery_count += 1

        capital_used = eo.price * filled
        tag = "RECOVERY" if self._is_recovery else "NORMAL"
        logger.info("ENTRY FILLED [%s] %s @ %.3f — %.4f shares ($%.2f) [verified via balance]",
                     tag, eo.outcome, eo.price, filled, capital_used)

        self.trade_logger.log_entry(
            outcome=eo.outcome, entry_price=eo.price,
            capital=capital_used, shares=filled, is_recovery=self._is_recovery,
        )
        self.notifier.notify_entry(
            outcome=eo.outcome, price=eo.price,
            capital=capital_used, shares=filled, is_recovery=self._is_recovery,
        )

        # Reset TP retry counters; _tick_tp_live will attempt placement using
        # the current balance as size (NOT cached `filled` value).
        self._tp_retry_last_at = 0.0
        self._tp_retry_attempts = 0

        # Try TP placement immediately, sized off the freshly-read balance.
        if self.take_profit is not None:
            sellable = self._get_conditional_balance(eo.token_id, force_refresh=True)
            sellable = max(0.0, sellable - self._pre_entry_cond)
            if sellable > 0:
                tp_id = self._place_limit(
                    eo.token_id, "SELL", self.take_profit, sellable,
                    post_only=self.post_only, suppress_log=True,
                )
                if tp_id:
                    self._tp_order = PendingOrder(
                        order_id=tp_id, token_id=eo.token_id,
                        outcome=eo.outcome, side="SELL",
                        price=self.take_profit, size=sellable, purpose="tp",
                    )
                    logger.info("TP SELL placed @ %.4f×%.4f (maker, id=%s)",
                                self.take_profit, sellable, tp_id[:12])
                else:
                    logger.info("TP placement deferred (will retry from balance)")
            else:
                logger.info("TP deferred: conditional balance not yet visible to backend")

        self.phase = Phase.TP_LIVE
        logger.info("Phase → TP_LIVE (monitoring for SL bid <= %.3f)", self.stop_loss)

    # ── Phase: TP_LIVE ────────────────────────────────────────────────────

    def _tick_tp_live(self, log: bool = False) -> None:
        """Poll TP order; place/retry TP using balance as size; monitor bid for SL."""
        active_sl = self.recovery_stop_loss if self._is_recovery else self.stop_loss

        # Retry TP placement, sizing from CURRENT balance (not cached value).
        # This naturally handles partial entries that grow over time.
        if self._tp_order is None and self.take_profit is not None and self._active_token_id:
            now = time.time()
            interval = 1.0 if self._tp_retry_attempts < 10 else 3.0
            if now - self._tp_retry_last_at >= interval:
                self._tp_retry_last_at = now
                self._tp_retry_attempts += 1

                cur_cond = self._get_conditional_balance(self._active_token_id, force_refresh=True)
                sellable = max(0.0, cur_cond - self._pre_entry_cond)

                if sellable <= 0:
                    if self._tp_retry_attempts in (3, 10, 30):
                        logger.info("TP retry: balance still 0 (%d attempts)", self._tp_retry_attempts)
                else:
                    tp_id = self._place_limit(
                        self._active_token_id, "SELL", self.take_profit,
                        sellable, post_only=self.post_only, suppress_log=True,
                    )
                    if tp_id:
                        self._tp_order = PendingOrder(
                            order_id=tp_id, token_id=self._active_token_id,
                            outcome=self._active_outcome or "", side="SELL",
                            price=self.take_profit, size=sellable, purpose="tp",
                        )
                        # Update cached position size to reflect what we actually hold.
                        self._position_shares = sellable
                        logger.info("TP SELL placed @ %.4f×%.4f after %d retries (id=%s)",
                                    self.take_profit, sellable, self._tp_retry_attempts,
                                    tp_id[:12])
                        self._tp_retry_attempts = 0
                    elif self._tp_retry_attempts in (3, 10, 30):
                        logger.info("TP placement still pending (%d attempts)",
                                    self._tp_retry_attempts)

        # Poll TP — also verify via balance (not just poll response).
        if self._tp_order is not None:
            filled, done = self._poll_order(self._tp_order.order_id)
            if done:
                # Verify with balance: shares should be back near baseline.
                cur_cond = self._get_conditional_balance(
                    self._tp_order.token_id, force_refresh=True
                )
                sold = max(0.0, self._position_shares - max(0.0, cur_cond - self._pre_entry_cond))
                if sold > 0 or filled > 0:
                    self._on_exit_filled("take_profit", self.take_profit, max(sold, filled))
                    return

        # Monitor bid for SL trigger
        bid = self._get_bid(self._active_token_id) if self._active_token_id else None
        if bid is not None:
            self.last_seen_price = bid
            if log:
                logger.info("  Bid: %.3f | SL: %.3f | TP: %s",
                            bid, active_sl,
                            f"{self.take_profit}" if self.take_profit else "—")

        if bid is not None and bid <= active_sl:
            logger.warning("SL triggered! Bid %.3f <= SL %.3f", bid, active_sl)
            self._enter_sl_exit(bid, active_sl)

    def _enter_sl_exit(self, current_bid: float, sl_price: float) -> None:
        """Cancel TP (verified), then attempt maker SL sized off real balance."""
        # 1. Force-cancel TP and verify. If it filled in the cancel race,
        #    record exit via balance and return.
        if self._tp_order is not None:
            canceled, _tp_fill_hint = self._force_cancel_verify(
                self._tp_order.order_id, max_wait=5.0
            )
            self._tp_order = None
            cur_cond = self._get_conditional_balance(self._active_token_id, force_refresh=True)
            held = max(0.0, cur_cond - self._pre_entry_cond)
            sold_during_cancel = max(0.0, self._position_shares - held)
            if sold_during_cancel > 0.001:
                logger.info("TP filled during cancel race (%.4f shares sold)", sold_during_cancel)
                if held <= 0.001:
                    self._on_exit_filled("take_profit", self.take_profit, sold_during_cancel)
                    return
                # Partial TP fill, fall through to SL on residue
                self._position_shares = held

        # 2. Read current balance — this is what we still hold.
        cur_cond = self._get_conditional_balance(self._active_token_id, force_refresh=True)
        sellable = max(0.0, cur_cond - self._pre_entry_cond)

        if sellable <= 0.001:
            logger.info("SL: no balance to sell, finalizing exit at SL price")
            self._on_exit_filled("stop_loss", sl_price, 0.0)
            return

        # 3. Place maker SL at SL + offset, sized to real balance.
        maker_price = round(sl_price + self.sl_maker_offset, 4)
        sl_id = self._place_limit(
            self._active_token_id, "SELL", maker_price,
            sellable, post_only=self.post_only,
        )

        if sl_id:
            self._sl_order = PendingOrder(
                order_id=sl_id, token_id=self._active_token_id,
                outcome=self._active_outcome or "", side="SELL",
                price=maker_price, size=sellable, purpose="sl_maker",
            )
            self._position_shares = sellable
            logger.info("SL maker SELL placed @ %.4f×%.4f (id=%s)",
                        maker_price, sellable, sl_id[:12])
        else:
            logger.info("SL maker rejected — going straight to taker")
            self._sl_taker_fallback(current_bid)
            return

        self.phase = Phase.SL_EXIT
        logger.info("Phase → SL_EXIT (maker attempt, timeout %ss)", self.sl_maker_timeout)

    # ── Phase: SL_EXIT ────────────────────────────────────────────────────

    def _tick_sl_exit(self) -> None:
        """Poll SL order via balance; on timeout → taker fallback."""
        so = self._sl_order
        if so is None:
            bid = self._get_bid(self._active_token_id) if self._active_token_id else None
            self._sl_taker_fallback(bid)
            return

        # Balance-driven fill check: how many shares are gone since SL placed?
        cur_cond = self._get_conditional_balance(so.token_id, force_refresh=True)
        held_now = max(0.0, cur_cond - self._pre_entry_cond)
        sold = max(0.0, self._position_shares - held_now)

        # Also check API-side state to know when order is finalized.
        poll_filled, poll_done = self._poll_order(so.order_id)

        if poll_done:
            actual_filled = max(sold, poll_filled)
            if actual_filled > 0.001:
                logger.info("SL maker SELL filled (%.4f shares @ %.4f) [verified balance]",
                            actual_filled, so.price)
                self._sl_order = None
                self._on_exit_filled("stop_loss", so.price, actual_filled)
                return
            # Done with 0 fill (e.g. canceled by us before any fill)
            logger.info("SL maker order done with 0 fill — falling back to taker")
            self._sl_order = None
            bid = self._get_bid(self._active_token_id) if self._active_token_id else None
            self._sl_taker_fallback(bid, held_now)
            return

        elapsed = time.time() - so.placed_at
        if elapsed > self.sl_maker_timeout:
            logger.info("SL maker timeout (%.0fs) — force cancel + taker fallback", elapsed)
            self._force_cancel_verify(so.order_id, max_wait=5.0)
            time.sleep(0.3)
            cur_cond = self._get_conditional_balance(so.token_id, force_refresh=True)
            held_now = max(0.0, cur_cond - self._pre_entry_cond)
            sold = max(0.0, self._position_shares - held_now)

            if sold > 0.001:
                logger.info("SL maker partial fill captured: %.4f shares", sold)

            self._sl_order = None
            if held_now <= 0.001:
                self._on_exit_filled("stop_loss", so.price, sold)
                return

            bid = self._get_bid(self._active_token_id) if self._active_token_id else None
            self._sl_taker_fallback(bid, held_now)

    def _sl_taker_fallback(self, bid: Optional[float], remaining_shares: float = 0) -> None:
        """Emergency taker sell at current bid. Sized to real balance, looped if partial."""
        # Always re-read balance — that's the truth of how much we still hold.
        cur_cond = self._get_conditional_balance(self._active_token_id, force_refresh=True)
        shares = max(0.0, cur_cond - self._pre_entry_cond)

        if shares <= 0.001:
            self._on_exit_filled("stop_loss", self.stop_loss, 0)
            return

        sell_price = bid if bid and bid > 0 else self.stop_loss
        logger.warning("SL taker fallback: FAK SELL %.4f shares @ %.4f", shares, sell_price)

        # First attempt at bid (or SL price)
        total_sold = 0.0
        last_price = sell_price
        for attempt, price in enumerate([sell_price, max(0.05, sell_price - 0.02), 0.01]):
            cur_cond = self._get_conditional_balance(self._active_token_id, force_refresh=True)
            shares_left = max(0.0, cur_cond - self._pre_entry_cond)
            if shares_left <= 0.001:
                break
            res = self._place_fak_sell(self._active_token_id, price, shares_left)
            time.sleep(1.0)
            cur_cond_after = self._get_conditional_balance(
                self._active_token_id, force_refresh=True
            )
            sold_now = max(0.0, shares_left - max(0.0, cur_cond_after - self._pre_entry_cond))
            if sold_now > 0:
                total_sold += sold_now
                last_price = price
                logger.info("FAK attempt %d @ %.4f: sold %.4f", attempt + 1, price, sold_now)
            elif res is None:
                logger.debug("FAK attempt %d @ %.4f: place failed", attempt + 1, price)

        if total_sold > 0:
            self._on_exit_filled("stop_loss", last_price, total_sold)
            return

        # Critical: before declaring failure, wait + re-verify balance.
        # Polymarket V2 has a documented lag between rejecting an order on
        # "balance locked" grounds and actually settling it once the lock
        # releases. We've observed FAKs that 400'd then settled 2-4s later.
        logger.warning("All FAK attempts returned no balance change — verifying once more")
        for wait_s in (2.0, 3.0, 3.0):
            time.sleep(wait_s)
            cur_cond = self._get_conditional_balance(
                self._active_token_id, force_refresh=True
            )
            held_now = max(0.0, cur_cond - self._pre_entry_cond)
            sold_late = max(0.0, shares - held_now)
            if sold_late > 0.001:
                # An earlier-rejected order actually settled. Compute the real
                # average exit price from pUSD delta, since we don't know which
                # of our retries got executed.
                pusd_now = self._get_collateral_balance(force_refresh=True)
                cash_in = pusd_now - self._pre_entry_pusd + (
                    self._entry_price_actual * self._position_shares
                )  # money received = (now - pre) + entry capital
                avg_exit = (cash_in / sold_late) if sold_late > 0 else sell_price
                logger.info(
                    "Late-fill detected: %.4f shares sold, avg exit $%.4f "
                    "(reconstructed from pUSD delta)",
                    sold_late, avg_exit,
                )
                self._on_exit_filled("stop_loss", avg_exit, sold_late)
                return

        # Verified failure — really stuck. Let market resolution handle it.
        logger.error("SL taker fallback failed — position left for market resolution")
        self._failed_exit_position = {
            'entry_price': self._entry_price_actual,
            'outcome': self._active_outcome,
            'shares': shares,
            'attempted_reason': 'stop_loss',
            'attempted_price': sell_price,
        }
        self._sl_order = None
        self.phase = Phase.DONE

    # ── Exit handling ─────────────────────────────────────────────────────

    def _on_exit_filled(self, reason: str, exit_price: float, filled: float) -> None:
        """Record completed trade. PnL measured as REAL pUSD delta, not theory.

        Also detects residual unsold shares and logs them so they're not
        silently lost. (They will be claimed at market resolution.)
        """
        entry_price = self._entry_price_actual
        outcome = self._active_outcome or "?"

        # Settle a moment, then read REAL balances.
        time.sleep(0.8)
        pusd_now = self._get_collateral_balance(force_refresh=True)
        cond_now = self._get_conditional_balance(self._active_token_id, force_refresh=True)
        residue = max(0.0, cond_now - self._pre_entry_cond)

        # Real PnL from balance delta. residue is still locked in shares
        # (will pay 0 or 1 at market resolution); for now value it at exit_price.
        pusd_delta = pusd_now - self._pre_entry_pusd
        residue_value = residue * exit_price  # rough mark-to-market
        pnl_real = pusd_delta + residue_value

        # Theoretical for comparison
        shares = filled if filled > 0 else self._position_shares
        pnl_theory = (exit_price - entry_price) * shares
        pnl_pct = (pnl_real / (entry_price * self._position_shares) * 100
                   if entry_price and self._position_shares else 0)

        if residue > 0.01:
            logger.warning(
                "EXIT [%s] %s: %.4f shares sold @ %.3f, RESIDUE %.4f shares left "
                "(will resolve at market end)",
                reason, outcome, shares, exit_price, residue,
            )
            self._failed_exit_position = {
                'entry_price': entry_price,
                'outcome': outcome,
                'shares': residue,
                'attempted_reason': reason,
                'attempted_price': exit_price,
            }

        logger.info(
            "EXIT [%s] %s: entry %.3f → exit %.3f | PnL REAL $%.2f (theory $%.2f, %.1f%%) "
            "| pUSD: $%.4f → $%.4f (Δ%+.4f)",
            reason, outcome, entry_price, exit_price, pnl_real, pnl_theory, pnl_pct,
            self._pre_entry_pusd, pusd_now, pusd_delta,
        )

        if reason == "stop_loss" and self.recovery_enabled:
            self._sl_triggered_this_market = True
            self._failed_outcome = outcome

        self.trade_logger.log_exit(
            outcome=outcome, entry_price=entry_price,
            exit_price=exit_price, reason=reason,
            shares=shares, pnl=pnl_real, pnl_pct=pnl_pct,
        )
        self.notifier.notify_exit(
            outcome=outcome, entry_price=entry_price,
            exit_price=exit_price, reason=reason,
            pnl=pnl_real, pnl_pct=pnl_pct, shares=shares,
        )

        self._entry_order = None
        self._tp_order = None
        self._sl_order = None
        self.phase = Phase.DONE

    # ── Per-market reset ──────────────────────────────────────────────────

    def reset_for_market(self) -> None:
        self.phase = Phase.WAIT_SIGNAL
        self._entry_order = None
        self._tp_order = None
        self._sl_order = None
        self._active_outcome = None
        self._active_token_id = None
        self._entry_price_actual = 0.0
        self._position_shares = 0.0
        self._is_recovery = False
        self._traded_in_current_market = False
        self._recovery_done_this_market = False
        self._sl_triggered_this_market = False
        self._failed_outcome = None
        self._recovery_count = 0
        self.last_seen_price = None
        self._failed_exit_position = None
        self._tp_retry_last_at = 0.0
        self._tp_retry_attempts = 0
        self._pre_entry_pusd = 0.0
        self._pre_entry_cond = 0.0
        self._entry_baseline_taken = False

    def cancel_all_pending(self) -> None:
        """Force-cancel-verify all live orders at market end.

        Uses balance deltas (not API responses) to detect any race-fills.
        This prevents orphaned orders that fill after the market loop ends.
        """
        for label, order in [("entry", self._entry_order),
                             ("tp", self._tp_order),
                             ("sl", self._sl_order)]:
            if order is None:
                continue
            logger.info("Force-cancelling %s order %s at market end",
                        label, order.order_id[:12])
            self._force_cancel_verify(order.order_id, max_wait=8.0)

            if label == "entry" and self._entry_baseline_taken:
                # Check if any shares appeared during cancel race
                cur_cond = self._get_conditional_balance(order.token_id, force_refresh=True)
                race_fill = max(0.0, cur_cond - self._pre_entry_cond)
                if race_fill > 0.001:
                    logger.info("Entry race-fill captured: %.4f shares", race_fill)
                    self._on_entry_filled(order, race_fill)
            elif label == "tp" and self._entry_baseline_taken:
                cur_cond = self._get_conditional_balance(order.token_id, force_refresh=True)
                held = max(0.0, cur_cond - self._pre_entry_cond)
                sold = max(0.0, self._position_shares - held)
                if sold > 0.001:
                    self._on_exit_filled("take_profit", order.price, sold)

    # ── Main per-market trading loop ──────────────────────────────────────

    def run_market(self, market_data: dict) -> bool:
        """Run the maker strategy for one market. Returns False if circuit breaker kills."""
        market_expires_at = datetime.now(timezone.utc) + timedelta(minutes=market_data['minutes_left'])
        market_start_time = time.time()
        max_duration = market_data['minutes_left'] * 60 - 5

        start_time = time.time()
        trading_started = False
        last_log_time = 0.0
        last_minute_log = 0.0

        while time.time() - start_time < max_duration:
            self.price_feed.wait_for_tick(timeout=self.order_poll_seconds)
            now = time.time()
            elapsed = now - market_start_time
            in_delay = elapsed < self.market_start_delay
            time_left = (market_expires_at - datetime.now(timezone.utc)).total_seconds()
            should_log = (now - last_log_time >= 10.0)

            if in_delay:
                if should_log and not trading_started:
                    last_log_time = now
                    logger.info("DELAY: %ds remaining...", int(self.market_start_delay - elapsed))
                continue
            if not trading_started:
                logger.info("TRADING ATTIVATO")
                trading_started = True

            if should_log:
                last_log_time = now

            # ── State machine tick ────────────────────────────────────

            if self.phase == Phase.WAIT_SIGNAL:
                if self._failed_exit_position is not None:
                    if should_log:
                        logger.warning("Posizione non chiusa, attendendo risoluzione mercato")
                    continue

                # Recovery check
                if (self.recovery_enabled and self._sl_triggered_this_market
                        and not self._recovery_done_this_market
                        and self._recovery_count < self.recovery_max_attempts
                        and time_left <= self.recovery_time_window):
                    if not self._is_recovery:
                        self._is_recovery = True
                        logger.info("RECOVERY mode attivato (ultimi %.0fs)", time_left)
                    self._tick_wait_signal(log=should_log)
                elif self.one_trade_per_market and self._traded_in_current_market:
                    if should_log:
                        logger.info("Trade completato, attendendo scadenza mercato...")
                else:
                    self._is_recovery = False
                    self._tick_wait_signal(log=should_log)

            elif self.phase == Phase.ENTRY_LIVE:
                self._tick_entry_live()

            elif self.phase == Phase.TP_LIVE:
                self._tick_tp_live(log=should_log)

            elif self.phase == Phase.SL_EXIT:
                self._tick_sl_exit()

            elif self.phase == Phase.DONE:
                if self.one_trade_per_market and not (
                    self.recovery_enabled and self._sl_triggered_this_market
                    and not self._recovery_done_this_market
                ):
                    if should_log:
                        logger.info("Trade completato, attendendo scadenza mercato...")
                else:
                    self.phase = Phase.WAIT_SIGNAL

            # Circuit breaker
            if self.notifier.killed:
                logger.critical("CIRCUIT BREAKER KILL: %s", self.notifier.kill_reason)
                self.cancel_all_pending()
                self.notifier.notify_stop(reason=f"CIRCUIT BREAKER: {self.notifier.kill_reason}")
                return False

            self.notifier.maybe_send_hourly()

            if now - last_minute_log >= 60.0:
                last_minute_log = now
                logger.info("Tempo rimasto: %.1f min | Phase: %s", time_left / 60, self.phase.value)

        # Market ending
        logger.warning("Mercato in scadenza!")
        self.cancel_all_pending()

        # Handle positions left open at market end
        position_active = (self.phase in (Phase.TP_LIVE, Phase.SL_EXIT)
                           or self._failed_exit_position is not None)
        if position_active:
            self._handle_market_end_position()

        return True

    def _handle_market_end_position(self) -> None:
        """Log open position at market resolution. PnL = REAL pUSD delta."""
        entry_price = self._entry_price_actual
        outcome = self._active_outcome or ""
        shares = self._position_shares

        if self._failed_exit_position:
            entry_price = self._failed_exit_position['entry_price']
            outcome = self._failed_exit_position.get('outcome', '')
            shares = self._failed_exit_position.get('shares', shares)

        # CRITICAL: re-read balances. Whatever happened, the truth is here.
        # The "failed exit" may actually have settled late, or the resolution
        # claim may have credited us already.
        pusd_now = self._get_collateral_balance(force_refresh=True)
        cond_now = self._get_conditional_balance(
            self._active_token_id, force_refresh=True
        ) if self._active_token_id else 0.0
        residue = max(0.0, cond_now - self._pre_entry_cond)

        capital_invested = entry_price * shares if entry_price else 0
        pusd_delta = pusd_now - self._pre_entry_pusd if self._entry_baseline_taken else 0.0

        if residue <= 0.001:
            # Position fully closed — PnL is just the pUSD delta. The
            # "failed_exit" was actually settled by a late fill; we just
            # didn't see it at the time.
            pnl = pusd_delta
            exit_price = (capital_invested + pnl) / shares if shares else 0
            did_win = None
            logger.info(
                "Market resolved: position was actually closed (late settlement). "
                "Real PnL $%.2f (pUSD: $%.4f → $%.4f)",
                pnl, self._pre_entry_pusd, pusd_now,
            )
        else:
            # Genuinely stuck — residue will resolve on chain.
            # Mark-to-market: residue × resolution_value. Best estimate is
            # last_seen_price (or bid). Win/loss is detected by claim worker.
            last_price = self.last_seen_price
            if last_price is None and self._active_token_id:
                last_price = self._get_bid(self._active_token_id)
            est_resolution = (1.00 if last_price is not None and last_price >= self.win_threshold
                              else (last_price if last_price is not None else 0.0))
            residue_value = residue * est_resolution
            pnl = pusd_delta + residue_value
            exit_price = est_resolution
            did_win = "probable" if est_resolution >= 0.99 else None
            logger.info(
                "Market resolved with residue %.4f shares ~$%.2f at est. %.3f. "
                "PnL $%.2f (delta pUSD %+.2f + residue value %+.2f)",
                residue, residue_value, est_resolution, pnl,
                pusd_delta, residue_value,
            )

        pnl_pct = (pnl / capital_invested * 100) if capital_invested else 0

        self.trade_logger.log_exit(
            outcome=outcome, entry_price=entry_price,
            exit_price=exit_price, reason="market_resolved",
            shares=shares, pnl=pnl, pnl_pct=pnl_pct, did_win=did_win,
        )
        self.notifier.notify_exit(
            outcome=outcome, entry_price=entry_price,
            exit_price=exit_price, reason="market_resolved",
            pnl=pnl, pnl_pct=pnl_pct, shares=shares,
        )
        logger.info("Market resolved: PnL $%.2f", pnl)

    # ── Claims ────────────────────────────────────────────────────────────

    def try_claim_positions(self) -> int:
        if not self.redeemer or not self.auto_claim_enabled:
            return 0
        if not self.past_market_ids:
            return 0
        logger.info("CLAIM: verifico %d mercati passati", len(self.past_market_ids))
        try:
            return self.redeemer.redeem_all_positions(self.past_market_ids)
        except Exception as e:
            logger.error("Errore claim: %s", e)
            return 0


# ── Claim worker ─────────────────────────────────────────────────────────────

def periodic_claim_worker(bot: Bot1BTC, interval_minutes: int = 30):
    logger.info("Claim worker avviato (ogni %d min)", interval_minutes)
    while True:
        try:
            time.sleep(interval_minutes * 60)
            bot.try_claim_positions()
        except Exception as e:
            logger.error("Errore claim worker: %s", e)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    try:
        logger.info("=" * 60)
        logger.info("BOT1 BTC — Maker Strategy")
        logger.info("=" * 60)

        # Find initial market
        logger.info("Cerco mercati BTC 5min attivi...")
        market_data = None
        attempt = 0
        while market_data is None:
            attempt += 1
            logger.info("Tentativo %d...", attempt)
            market_data = find_active_5min_market()
            if market_data:
                condition_id = market_data['condition_id']
                Config.BTC_5MIN_MARKET_ID = condition_id
                Config.TOKEN_ID = market_data['token_ids'][0]
                logger.info("Mercato trovato: %s", market_data['question'])
                logger.info("Tempo rimasto: %d min", int(market_data['minutes_left']))
            else:
                logger.info("Nessun mercato trovato, riprovo tra 5s...")
                time.sleep(5)

        # Initialize bot
        bot = Bot1BTC()

        # Check existing redeemable positions
        try:
            if bot.auto_claim_enabled and hasattr(bot.redeemer, 'get_redeemable_condition_ids'):
                existing = bot.redeemer.get_redeemable_condition_ids()
                if existing:
                    bot.past_market_ids.extend(existing)
                    logger.info("Aggiunti %d mercati da reclamare", len(existing))
        except Exception as e:
            logger.warning("Errore recupero posizioni: %s", e)

        bot.try_claim_positions()

        # Start claim worker
        if bot.auto_claim_enabled:
            claim_thread = threading.Thread(
                target=periodic_claim_worker,
                args=(bot, bot.claim_interval),
                daemon=True,
            )
            claim_thread.start()
            logger.info("Claim worker avviato (ogni %d min)", bot.claim_interval)

        # ── Infinite market loop ──────────────────────────────────────────
        while True:
            # Skip nearly-expired markets
            if market_data['minutes_left'] < 0.4:
                logger.warning("Mercato troppo vicino alla scadenza (%.1f min)",
                               market_data['minutes_left'])
                cid = Config.BTC_5MIN_MARKET_ID
                if cid not in bot.past_market_ids:
                    bot.past_market_ids.append(cid)
                market_data = _find_next_market(market_data)
                continue

            # Reset bot state for new market
            bot.reset_for_market()
            bot.token_ids = {
                outcome: tid
                for outcome, tid in zip(market_data['outcomes'], market_data['token_ids'])
            }
            bot.outcomes = market_data['outcomes']
            bot.price_feed.subscribe(list(bot.token_ids.values()))

            bot.trade_logger.log_market_start(market_name=market_data['question'])
            bot.notifier.record_market()

            logger.info("Avvio trading (maker) per: %s", market_data['question'])
            logger.info("Tempo: %d min | Token: %s",
                        int(market_data['minutes_left']),
                        ' e '.join(bot.outcomes))

            if bot.recovery_enabled:
                logger.info("Recovery: ON (ultimi %ds, %sx capitale)",
                            bot.recovery_time_window, bot.recovery_multiplier)

            # Run the market
            try:
                alive = bot.run_market(market_data)
                if not alive:
                    return  # circuit breaker killed
            except KeyboardInterrupt:
                logger.info("Bot fermato dall'utente")
                bot.cancel_all_pending()
                bot.notifier.notify_stop(reason="manual")
                raise

            # Add to claim list
            cid = Config.BTC_5MIN_MARKET_ID
            if cid not in bot.past_market_ids:
                bot.past_market_ids.append(cid)

            # Find next market
            market_data = _find_next_market(market_data)

    except KeyboardInterrupt:
        logger.info("Shutdown.")
    except Exception as e:
        logger.error("Errore fatale: %s", e, exc_info=True)


def _find_next_market(prev_market: dict) -> dict:
    """Block until the next 5-min market is found."""
    logger.info("Cerco prossimo mercato...")
    attempt = 0
    last_ts = prev_market.get('timestamp')
    while True:
        attempt += 1
        if attempt > 1:
            logger.info("Tentativo %d...", attempt)
        if attempt <= 2 and last_ts:
            md = find_active_5min_market(last_timestamp=last_ts)
        else:
            md = find_active_5min_market()
        if md:
            Config.BTC_5MIN_MARKET_ID = md['condition_id']
            Config.TOKEN_ID = md['token_ids'][0]
            logger.info("Nuovo mercato: %s (%d min)",
                        md['question'], int(md['minutes_left']))
            return md
        if attempt <= 2:
            time.sleep(3)
        else:
            time.sleep(10)


if __name__ == '__main__':
    main()
