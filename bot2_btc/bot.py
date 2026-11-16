"""Bot2 BTC – straddle entry + intramarket inventory (skeleton).

Mantenuto dal bot storico:
──────────────────────────
• Discovery mercato, WebSocket/REST, executor, Telegram, state JSON.
• PRE-MARKET: compra N Up e N Down (ciascuno ≤ max_entry_price), con retry e sync balance.
• TRANSITION: all'apertura del mercato K, pre-pos in background su K+1 e gestione di K.

Rimosso (da reimplementare sotto forma di strategia intramarket):
──────────────────────────────────────────────────────────────────
exit-low, rebuy loop, late reversal verso risoluzione.

Fase `BOTH_OPEN`: strategia ibrida inventario (`_tick_straddle_strategy`): primi minuti
bias vendita sul braccio forte; in chiusura (o spread direzionale largo) vendita sul debole;
re-entry su compressione dello spread. Parametri `BOT2_INV_*` in env.
"""
from __future__ import annotations

import json
import logging
import signal
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from dataclasses import asdict

import requests

from bot2_btc.config import Bot2Config
from bot2_btc.executor import BaseExecutor, create_executor
from bot2_btc.market_finder import MarketFinder, fetch_best_prices_rest
from bot2_btc.models import (
    BotPhase, MarketInfo, MarketSession, OrderRequest, OrderResult,
    OrderStatus, Position, Prices, SessionPhase, Side,
)
from bot2_btc.notifier import TelegramNotifier, create_notifier
from bot2_btc.price_feed import NullPriceFeed, PriceFeed

logger = logging.getLogger(__name__)


# ── Logging setup ─────────────────────────────────────────────────────────────

def setup_logging(cfg: Bot2Config) -> None:
    log_dir = Path(cfg.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    level = getattr(logging, cfg.log_level.upper(), logging.INFO)
    fmt = "%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s – %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(level=level, format=fmt, datefmt=datefmt)

    from logging.handlers import RotatingFileHandler
    fh = RotatingFileHandler(
        log_dir / "bot2.log", maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    fh.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    fh.setLevel(level)
    logging.getLogger().addHandler(fh)

    if cfg.log_per_run_file:
        run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_fh = logging.FileHandler(log_dir / f"bot2_{run_tag}.log", encoding="utf-8")
        run_fh.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
        run_fh.setLevel(level)
        logging.getLogger().addHandler(run_fh)

    for noisy in ("urllib3", "requests", "websockets", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# ── Bot ───────────────────────────────────────────────────────────────────────

class Bot2BTC:
    """Main bot class.  Call ``run()`` to start; handles SIGINT/SIGTERM."""

    def __init__(self, cfg: Bot2Config):
        self.cfg = cfg
        self._executor: BaseExecutor = create_executor(cfg)
        self._notifier: TelegramNotifier = create_notifier(cfg)
        self._finder = MarketFinder(cfg)
        self._feed: PriceFeed = (
            PriceFeed(cfg.ws_url, stale_seconds=cfg.ws_stale_seconds)
            if cfg.ws_enabled else NullPriceFeed()
        )
        self._rest_session = requests.Session()

        # State
        self._phase = BotPhase.STARTING
        self._current_market: Optional[MarketInfo] = None   # market being managed now
        self._next_market: Optional[MarketInfo] = None      # next market to pre-position on
        self._session: Optional[MarketSession] = None       # active strategy session
        self._market_index: int = 0
        self._cum_pnl: float = 0.0
        self._day_pnl: float = 0.0
        self._day_markets: int = 0
        self._day_key: str = ""

        # Pre-position tracking
        self._pre_pos_done: bool = False           # both sides filled for next market
        self._pre_pos_pending: list[str] = []      # order IDs awaiting fill
        self._next_session: Optional[MarketSession] = None  # session for next market (bg pre-pos)

        self._running = threading.Event()
        self._state_lock = threading.Lock()
        if self.cfg.state_reset_on_start:
            self._reset_state()
        else:
            self._load_state()

        # Telegram commands
        self._notifier.register_command("status", self._cmd_status)
        self._notifier.register_command("pnl", self._cmd_pnl)
        self._notifier.register_command("stop", self._cmd_stop)
        # Intramarket strategy: per-leg cooldown timestamps
        self._last_strategy_log_ts: float = 0.0
        self._inv_last_ts_up: float = 0.0
        self._inv_last_ts_down: float = 0.0
        self._stop_requested_via_telegram: bool = False

    # ── Entry point ───────────────────────────────────────────────────────────

    def run(self) -> None:
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)
        self._running.set()

        mode = "PAPER" if self.cfg.paper_mode else "LIVE"
        logger.info("=" * 60)
        logger.info("Bot2 BTC starting – mode: %s", mode)
        self._log_effective_config()
        logger.info(
            "entry_shares=%.0f  max_entry=%.2f  | inv_enabled=%s phase_split=%.2f",
            self.cfg.entry_shares, self.cfg.max_entry_price,
            self.cfg.inv_enabled, self.cfg.inv_phase_split,
        )
        balance_usd = self._executor.get_balance_usd()
        positions_value, n_positions = self._executor.get_open_positions_value()
        portfolio_usd = balance_usd + positions_value
        logger.info("Cash: $%.2f | Open positions: $%.2f (%d) | Portfolio: $%.2f",
                    balance_usd, positions_value, n_positions, portfolio_usd)
        logger.info("=" * 60)

        self._feed.start()
        self._notifier.notify_start(mode, balance_usd, positions_value, n_positions)

        try:
            self._main_loop()
        except Exception as exc:
            logger.exception("Unexpected error in main loop: %s", exc)
            self._notifier.notify_error(str(exc))
        finally:
            self._shutdown()

    def _handle_signal(self, signum, frame) -> None:
        logger.info("Shutdown signal received (%s)", signum)
        self._running.clear()

    def _wallet_snapshot(self) -> tuple[float, float, int]:
        """Return (cash_usd, positions_value, n_positions). Non-blocking best-effort."""
        cash = self._executor.get_balance_usd()
        pos_val, n_pos = self._executor.get_open_positions_value()
        return cash, pos_val, n_pos

    def _shutdown(self) -> None:
        logger.info("Shutting down…")
        self._feed.stop()
        self._save_state()
        cash, pos_val, n_pos = self._wallet_snapshot()
        portfolio = cash + pos_val
        logger.info("Final wallet – Cash: $%.2f | Positions: $%.2f (%d) | Portfolio: $%.2f",
                    cash, pos_val, n_pos, portfolio)
        self._notifier.notify_stop("graceful shutdown", cash, pos_val, n_pos)
        if self._stop_requested_via_telegram:
            self._notifier.send("✅ <b>STOP confirmed</b>: Bot2 is fully stopped.")
        self._notifier.shutdown()

    def _log_effective_config(self) -> None:
        """Log effective runtime config at startup (sensitive fields masked)."""
        cfg_map = asdict(self.cfg)
        redacted_keys = {
            "private_key",
            "wallet_address",
            "telegram_bot_token",
            "telegram_chat_id",
        }
        logger.info("Effective config (startup):")
        for key in sorted(cfg_map.keys()):
            val = cfg_map[key]
            if key in redacted_keys and val:
                shown = str(val)
                val = f"{shown[:4]}***{shown[-4:]}" if len(shown) > 10 else "***"
            logger.info("  %s=%s", key, val)

    # ── Main loop ─────────────────────────────────────────────────────────────

    def _main_loop(self) -> None:
        self._phase = BotPhase.AWAITING_NEXT_MARKET

        while self._running.is_set():
            now = time.time()
            try:
                self._tick(now)
            except Exception as exc:
                logger.exception("Tick error: %s", exc)
            time.sleep(self.cfg.tick_interval)

    def _tick(self, now: float) -> None:
        # ── Daily reset check ──────────────────────────────────────────
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today != self._day_key:
            if self._day_key:
                self._notifier.notify_daily_summary(
                    self._day_key, self._day_markets, self._day_pnl, self._cum_pnl
                )
            self._day_key = today
            self._day_pnl = 0.0
            self._day_markets = 0

        # ── Phase dispatch ─────────────────────────────────────────────
        if self._phase in (BotPhase.AWAITING_NEXT_MARKET, BotPhase.STARTING):
            self._tick_awaiting(now)

        elif self._phase == BotPhase.PRE_POSITIONING:
            self._tick_pre_positioning(now)

        elif self._phase == BotPhase.POSITIONED:
            self._tick_positioned(now)

        elif self._phase == BotPhase.MANAGING:
            self._tick_managing(now)

    # ── Phase: AWAITING NEXT MARKET ───────────────────────────────────────────

    def _tick_awaiting(self, now: float) -> None:
        """Find the current market and the next one; set up for pre-positioning."""
        current = self._finder.find_current()
        if not current or not current.condition_id:
            logger.debug("No active market found, retrying in 5s…")
            time.sleep(5)
            return

        # The current market is already running – we skip it for strategy entry.
        # We target the NEXT market for pre-positioning.
        self._current_market = current
        logger.info(
            "Found current market: %s (ends in %.0fs)",
            current.question[:70], current.end_timestamp - now,
        )

        next_m = self._finder.find_next(current.timestamp_tag)
        if not next_m or not next_m.condition_id:
            logger.info("Next market not yet available; will retry…")
            # Check if we're close to the next window
            next_start = current.timestamp_tag + self.cfg.market_interval_seconds
            wait = next_start - now - 30  # try 30s before expected start
            if wait > 0:
                logger.info("Sleeping %.0fs until next market probe", min(wait, 60))
                time.sleep(min(wait, 60))
            return

        self._next_market = next_m
        self._pre_pos_done = False
        logger.info(
            "Next market found: %s (starts in %.0fs)",
            next_m.question[:70], next_m.start_timestamp - now,
        )

        # Subscribe WS feed to both current and next tokens
        all_tokens = list(current.token_ids) + list(next_m.token_ids)
        self._feed.subscribe(all_tokens)

        # Determine if we should pre-position now
        time_to_next = next_m.start_timestamp - now
        if time_to_next <= self.cfg.pre_position_seconds_before_start:
            logger.info("Within pre-position window (%.0fs to start) – entering PRE_POSITIONING", time_to_next)
            self._phase = BotPhase.PRE_POSITIONING
        else:
            # Sleep until we're just inside the pre-position window (min 5s sleep to avoid busy loop)
            sleep_for = max(5.0, time_to_next - self.cfg.pre_position_seconds_before_start - 1)
            logger.info(
                "%.0fs until next market – sleeping %.0fs",
                time_to_next, sleep_for,
            )
            time.sleep(sleep_for)

    # ── Phase: PRE_POSITIONING ────────────────────────────────────────────────

    # ── Pre-position price helper ─────────────────────────────────────────────

    def _prepos_limit_price(self, ask: Optional[float]) -> float:
        """Return an aggressive limit price for a pre-position BUY.

        Formula:  min(ask + buffer, max_entry_price)

        The buffer crosses the spread if the ask ticks up a notch between
        price snapshot and order submission.  The hard cap is max_entry_price
        (default 0.52) so we never pay more than that per share.
        """
        base = ask if ask else self.cfg.max_entry_price
        buf = self.cfg.pre_pos_price_buffer
        return round(min(base + buf, self.cfg.max_entry_price), 4)

    def _tick_pre_positioning(self, now: float) -> None:
        """Place initial straddle orders for next market."""
        market = self._next_market
        if not market:
            self._phase = BotPhase.AWAITING_NEXT_MARKET
            return

        logger.info("Pre-positioning on: %s", market.question[:70])

        # Get current best prices for both sides
        prices = self._get_prices_for_market(market)
        up_ask = prices.get(Side.UP.value, {}).get("ask")
        down_ask = prices.get(Side.DOWN.value, {}).get("ask")

        up_price = self._prepos_limit_price(up_ask)
        down_price = self._prepos_limit_price(down_ask)

        logger.info(
            "Entry prices: UP=%.4f  DOWN=%.4f  shares=%.0f  cost=~$%.2f",
            up_price, down_price, self.cfg.entry_shares,
            self.cfg.entry_shares * (up_price + down_price),
        )
        self._notifier.notify_pre_position(
            self._market_index + 1, market.end_timestamp,
            up_price, down_price, self.cfg.entry_shares,
        )

        self._market_index += 1
        session = MarketSession(market=market, market_index=self._market_index)
        session.up_position = Position(Side.UP)
        session.down_position = Position(Side.DOWN)

        # Place both orders simultaneously to avoid one blocking the other.
        # In live mode, each buy may spend up to order_timeout_seconds polling;
        # sequential placement meant DOWN was placed ~12s late with a stale price.
        up_result, down_result = self._place_prepos_pair(
            market, up_price, down_price, "pre-pos"
        )

        self._record_fill(session, up_result, Side.UP)
        self._record_fill(session, down_result, Side.DOWN)

        # Reconcile against actual on-chain balances (live mode only).
        self._sync_spendable(session, Side.UP)
        self._sync_spendable(session, Side.DOWN)

        # Warm up CTF allowance for both tokens early so the on-chain approval
        # is confirmed well before EXIT LOW fires (on-chain tx needs ~10-30s).
        self._warm_allowances(market)

        # Retry any side that didn't reach target before the market opens.
        self._complete_prepos_with_retry(session, market)

        self._session = session
        self._save_state()

        if session.up_position.shares > 0 or session.down_position.shares > 0:
            logger.info(
                "Pre-position filled: UP=%.0f@%.4f  DOWN=%.0f@%.4f",
                session.up_position.shares, session.up_position.avg_entry_price,
                session.down_position.shares, session.down_position.avg_entry_price,
            )
            self._phase = BotPhase.POSITIONED
        else:
            logger.warning("Pre-position failed (no fills); retrying in 15s")
            time.sleep(15)

    def _place_prepos_pair(
        self,
        market: MarketInfo,
        up_price: float,
        down_price: float,
        label_prefix: str,
    ) -> tuple[OrderResult, OrderResult]:
        """Place UP and DOWN pre-position buys in parallel.

        Both orders are submitted concurrently so that their fill-polling
        windows overlap.  This prevents the sequential pattern where UP's
        ~12-second poll delays DOWN's placement until the ask has moved.
        """
        results: dict[str, OrderResult] = {}

        def _buy_side(side: Side, price: float) -> tuple[str, OrderResult]:
            r = self._place_buy(
                market, side, self.cfg.entry_shares,
                price, f"{label_prefix} {side.value}",
            )
            return side.value, r

        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="prepos") as pool:
            futures = {
                pool.submit(_buy_side, Side.UP, up_price): Side.UP,
                pool.submit(_buy_side, Side.DOWN, down_price): Side.DOWN,
            }
            for fut in as_completed(futures):
                try:
                    key, result = fut.result()
                    results[key] = result
                except Exception as exc:
                    side = futures[fut]
                    logger.error("pre-pos %s thread error: %s", side.value, exc)
                    results[side.value] = OrderResult(
                        request=OrderRequest(
                            market=market, side=side,
                            shares=self.cfg.entry_shares, limit_price=0,
                        ),
                        status=OrderStatus.REJECTED,
                        error=str(exc),
                    )

        return results.get(Side.UP.value), results.get(Side.DOWN.value)

    def _record_fill(
        self, session: MarketSession, result: Optional[OrderResult], side: Side
    ) -> None:
        """Record a pre-pos fill, replacing any prior tracking for this side."""
        if result is None:
            logger.warning("[#%d] pre-pos %s: no result (thread error)", session.market_index, side.value)
            return
        if result.ok and result.filled_shares > 0:
            pos = session.position_for(side)
            pos.shares = result.filled_shares
            pos.avg_entry_price = result.filled_price
            pos.order_id = result.order_id
            # Partial GTC fill: cancel the remainder to avoid a ghost resting order
            if result.filled_shares < self.cfg.entry_shares * 0.99 and result.order_id:
                cancelled = self._executor.cancel_order(result.order_id)
                logger.info(
                    "[#%d] pre-pos PARTIAL %s: %.4f/%.0f shares filled – cancel remainder %s",
                    session.market_index, side.value,
                    result.filled_shares, self.cfg.entry_shares,
                    "OK" if cancelled else "failed",
                )
            else:
                self._notifier.notify_position_filled(
                    side.value, result.filled_shares, result.filled_price,
                    session.market_index, session.market.end_timestamp,
                )
                logger.info(
                    "[#%d] pre-pos FILL %s: %.4f shares @ %.4f",
                    session.market_index, side.value,
                    result.filled_shares, result.filled_price,
                )
        else:
            logger.warning(
                "[#%d] pre-pos %s: status=%s error=%s",
                session.market_index, side.value, result.status, result.error or "—",
            )

    def _record_fill_additive(
        self, session: MarketSession, result: Optional[OrderResult], side: Side
    ) -> None:
        """Record a pre-pos retry fill, *adding* to existing position (weighted avg price)."""
        if result is None:
            return
        if result.ok and result.filled_shares > 0:
            pos = session.position_for(side)
            # Partial GTC retry fill: cancel remainder to prevent ghost resting orders
            if result.filled_shares < self.cfg.entry_shares * 0.99 and result.order_id:
                cancelled = self._executor.cancel_order(result.order_id)
                logger.info(
                    "[#%d] pre-pos retry PARTIAL %s: %.4f shares – cancel remainder %s",
                    session.market_index, side.value,
                    result.filled_shares, "OK" if cancelled else "failed",
                )
            total = pos.shares + result.filled_shares
            if total > 0:
                pos.avg_entry_price = (
                    pos.avg_entry_price * pos.shares
                    + result.filled_price * result.filled_shares
                ) / total
            pos.shares = total
            logger.info(
                "[#%d] pre-pos retry FILL %s: +%.4f@%.4f → total=%.4f@%.4f",
                session.market_index, side.value,
                result.filled_shares, result.filled_price,
                pos.shares, pos.avg_entry_price,
            )
        else:
            logger.warning(
                "[#%d] pre-pos retry %s: status=%s error=%s",
                session.market_index, side.value, result.status, result.error or "—",
            )

    def _complete_prepos_with_retry(
        self, session: MarketSession, market: MarketInfo, max_attempts: int = 3  # noqa: ARG002
    ) -> None:
        """Keep retrying until both sides are filled or market is about to open.

        Runs a time-based loop (not a fixed attempt count) so it searches for
        entry throughout the entire pre-positioning window (up to
        ``pre_position_seconds_before_start`` seconds = 800 s by default).

        When the ask for a missing side is above the cap:
        - waits ``prepos_cap_wait_seconds`` (default 10 s) then re-checks
        - logs a condensed status every ``prepos_cap_log_interval`` s (default 60 s)
          to avoid spam
        """
        min_secs = self.cfg.prepos_min_seconds_before_start
        cap_wait = self.cfg.prepos_cap_wait_seconds
        log_interval = getattr(self.cfg, "prepos_cap_log_interval", 60)
        last_log_t: float = 0.0
        attempt = 0

        while True:
            time_left = market.start_timestamp - time.time()
            if time_left < min_secs:
                logger.info(
                    "[#%d] Pre-pos search stopped: %.0fs to open (min=%ds)",
                    session.market_index, time_left, min_secs,
                )
                break

            up_deficit = max(0.0, self.cfg.entry_shares - session.up_position.shares)
            down_deficit = max(0.0, self.cfg.entry_shares - session.down_position.shares)

            if up_deficit < 1.0 and down_deficit < 1.0:
                logger.info("[#%d] Pre-pos complete: both sides at target", session.market_index)
                break

            prices = self._get_prices_for_market(market)
            up_ask = prices.get(Side.UP.value, {}).get("ask")
            down_ask = prices.get(Side.DOWN.value, {}).get("ask")

            needs_up = up_deficit >= 1.0
            needs_down = down_deficit >= 1.0

            up_above_cap = bool(needs_up and up_ask and up_ask > self.cfg.max_entry_price)
            down_above_cap = bool(needs_down and down_ask and down_ask > self.cfg.max_entry_price)

            # When both missing sides are above cap just wait quietly
            if up_above_cap and down_above_cap:
                now_t = time.time()
                if now_t - last_log_t >= log_interval:
                    logger.warning(
                        "[#%d] Waiting for price ≤ cap=%.2f "
                        "(UP ask=%.4f  DOWN ask=%.4f  %.0fs to open)",
                        session.market_index, self.cfg.max_entry_price,
                        up_ask or 0, down_ask or 0, time_left,
                    )
                    last_log_t = now_t
                time.sleep(min(cap_wait, max(1.0, time_left - min_secs)))
                continue

            if up_above_cap:
                needs_up = False
            if down_above_cap:
                needs_down = False

            # At least one side is actionable
            attempt += 1
            now_t = time.time()
            if now_t - last_log_t >= log_interval or attempt <= 3:
                logger.warning(
                    "[#%d] Pre-pos attempt #%d "
                    "(UP deficit=%.0f ask=%s  DOWN deficit=%.0f ask=%s  %.0fs to open)",
                    session.market_index, attempt,
                    up_deficit, f"{up_ask:.4f}" if up_ask else "?",
                    down_deficit, f"{down_ask:.4f}" if down_ask else "?",
                    time_left,
                )
                if up_above_cap:
                    logger.warning("[#%d] UP ask=%.4f > cap=%.4f – skipping", session.market_index, up_ask, self.cfg.max_entry_price)
                if down_above_cap:
                    logger.warning("[#%d] DOWN ask=%.4f > cap=%.4f – skipping", session.market_index, down_ask, self.cfg.max_entry_price)
                last_log_t = now_t

            if needs_up and needs_down:
                up_r, down_r = self._place_prepos_pair(
                    market,
                    self._prepos_limit_price(up_ask),
                    self._prepos_limit_price(down_ask),
                    f"prepos-r{attempt}",
                )
                self._record_fill_additive(session, up_r, Side.UP)
                self._record_fill_additive(session, down_r, Side.DOWN)
                self._sync_spendable(session, Side.UP)
                self._sync_spendable(session, Side.DOWN)
            elif needs_up or needs_down:
                side = Side.UP if needs_up else Side.DOWN
                ask = up_ask if needs_up else down_ask
                deficit = up_deficit if needs_up else down_deficit
                result = self._place_buy(
                    market, side, deficit,
                    self._prepos_limit_price(ask),
                    f"prepos-r{attempt}",
                )
                self._record_fill_additive(session, result, side)
                self._sync_spendable(session, side)
                # The above-cap side: wait before next round so it has time to come down
                if up_above_cap or down_above_cap:
                    time.sleep(min(cap_wait, max(1.0, time_left - min_secs)))

        up_ok = session.up_position.shares >= self.cfg.entry_shares * 0.9
        down_ok = session.down_position.shares >= self.cfg.entry_shares * 0.9
        if not up_ok or not down_ok:
            logger.warning(
                "[#%d] Pre-pos ended with partial fill: UP=%.0f/%.0f  DOWN=%.0f/%.0f",
                session.market_index,
                session.up_position.shares, self.cfg.entry_shares,
                session.down_position.shares, self.cfg.entry_shares,
            )

    # ── On-chain position reconciliation ─────────────────────────────────────

    def _warm_allowances(self, market: "MarketInfo") -> None:
        """Fire warm_up_allowance for both UP and DOWN tokens (no-op in paper mode)."""
        if not hasattr(self._executor, "warm_up_allowance"):
            return
        for side in (Side.UP, Side.DOWN):
            token_id = market.token_id_for(side)
            if token_id:
                self._executor.warm_up_allowance(token_id)

    def _sync_spendable(
        self, session: MarketSession, side: Side, *, skip_zero_guard: bool = False
    ) -> None:
        """Reconcile *pos.shares* against the actual on-chain balance.

        ``skip_zero_guard=True`` disables the propagation-delay guard and should
        be used before SELL operations: if the CLOB reports 0 shares at that
        point it is a real empty balance, not a transient post-fill lag.
        Leave it at False (default) after BUY fills where the balance API lags.
        """
        token_id = session.market.token_id_for(side)
        if not token_id:
            return
        actual = self._executor.get_spendable_shares(token_id)
        if actual is None:
            return  # paper mode or transient API error – trust internal tracking
        if not skip_zero_guard and actual < 1.0:
            logger.debug(
                "[#%d] Position sync %s: actual_onchain=%.4f < 1 – skipping (propagation delay)",
                session.market_index, side.value, actual,
            )
            return
        pos = session.position_for(side)
        if abs(actual - pos.shares) > 0.01:
            logger.warning(
                "[#%d] Position sync %s: tracked=%.4f actual_onchain=%.4f – correcting",
                session.market_index, side.value, pos.shares, actual,
            )
            pos.shares = actual

    # ── Phase: POSITIONED (waiting for market start) ──────────────────────────

    def _tick_positioned(self, now: float) -> None:
        # Prefer _next_market (set during pre-positioning flow).
        # Fall back to _current_market only after session-close transition where
        # _next_market has been cleared but the new session hasn't started yet.
        market = self._next_market or self._current_market
        if not market:
            self._phase = BotPhase.AWAITING_NEXT_MARKET
            return

        # Check if market has started
        if now >= market.start_timestamp:
            logger.info(
                "[#%d] Market STARTED: %s",
                self._market_index, market.question[:70],
            )
            self._on_market_start(market, now)
            return

        time_to_start = market.start_timestamp - now

        # ── Safety abort: incomplete straddle too close to market open ────
        if time_to_start <= self.cfg.prepos_abort_seconds and self._session is not None:
            session = self._session
            up_shares = session.up_position.shares
            down_shares = session.down_position.shares
            has_up = up_shares > 0
            has_down = down_shares > 0
            if has_up != has_down:  # exactly one side filled (XOR)
                filled_side = Side.UP if has_up else Side.DOWN
                filled_shares = up_shares if has_up else down_shares
                filled_pos = session.position_for(filled_side)
                logger.warning(
                    "[#%d] SAFETY ABORT: only %s filled (%.2f sh) at T-%.1fs – selling and skipping market",
                    session.market_index, filled_side.value, filled_shares, time_to_start,
                )
                prices = self._get_prices_for_market(market)
                bid = prices.get(filled_side.value, {}).get("bid") or 0.01
                notional = filled_shares * bid
                if notional >= 1.0:
                    result = self._place_sell(
                        market, filled_side, filled_shares, bid, "abort-prepos",
                    )
                    if result.ok:
                        logger.info(
                            "[#%d] Abort sell: %.2f %s @ %.4f",
                            session.market_index, result.filled_shares or filled_shares,
                            filled_side.value, result.filled_price or bid,
                        )
                    else:
                        logger.error(
                            "[#%d] Abort sell FAILED: %s – position may remain open",
                            session.market_index, result.error or result.status,
                        )
                else:
                    logger.warning(
                        "[#%d] Abort sell skipped: notional $%.4f < $1 – writing off %.2f sh",
                        session.market_index, notional, filled_shares,
                    )
                filled_pos.shares = 0.0
                self._notifier.notify_error(
                    f"[#{session.market_index}] Incomplete straddle aborted at T-{time_to_start:.1f}s "
                    f"(only {filled_side.value} filled). Market skipped."
                )
                self._session = None
                self._next_market = None
                self._phase = BotPhase.AWAITING_NEXT_MARKET
                return
            elif not has_up and not has_down:
                logger.warning(
                    "[#%d] SAFETY ABORT: no fills at T-%.1fs – skipping market",
                    session.market_index, time_to_start,
                )
                self._session = None
                self._next_market = None
                self._phase = BotPhase.AWAITING_NEXT_MARKET
                return

        if int(time_to_start) % 30 == 0:
            logger.debug("Positioned; %.0fs until market start", time_to_start)

    def _on_market_start(self, market: MarketInfo, now: float) -> None:
        """Transition: the pre-positioned market is now live.

        Steps:
        1. Pre-position on the NEXT market (K+2).
        2. Start managing the current session (K+1).
        """
        self._current_market = market
        self._phase = BotPhase.MANAGING

        if self._session:
            self._session.phase = SessionPhase.BOTH_OPEN
            self._check_partial_prepos(self._session)
        else:
            # Edge case: session was lost – skip managing this market
            logger.warning("Session lost at market start; will re-discover")
            self._phase = BotPhase.AWAITING_NEXT_MARKET
            return

        # Subscribe WS feed to the live market tokens
        self._feed.subscribe(list(market.token_ids))

        logger.info("[#%d] Entering MANAGING phase for %s", self._market_index, market.question[:70])

        self._inv_last_ts_up = 0.0
        self._inv_last_ts_down = 0.0
        self._last_strategy_log_ts = 0.0

        # Pre-position on the NEXT market in a background thread so we don't delay
        # managing the current one.
        next_market = self._finder.find_next(market.timestamp_tag)
        if next_market and next_market.condition_id:
            self._next_market = next_market
            self._pre_pos_done = False
            t = threading.Thread(
                target=self._background_pre_position,
                args=(next_market,),
                daemon=True,
                name="bot2-next-prepos",
            )
            t.start()
        else:
            logger.info("Next market (K+2) not yet available; will probe later")
            self._next_market = None
            self._pre_pos_done = False

    def _check_partial_prepos(self, session: MarketSession) -> None:
        """Sincronizza i balance; se lo straddle non è completo, appiattisce e passa a DONE."""
        self._sync_spendable(session, Side.UP, skip_zero_guard=True)
        self._sync_spendable(session, Side.DOWN, skip_zero_guard=True)
        up_ok = session.up_position.shares > 0
        down_ok = session.down_position.shares > 0

        logger.info(
            "[#%d] Positions at market open: UP=%.4f@%.4f  DOWN=%.4f@%.4f",
            session.market_index,
            session.up_position.shares, session.up_position.avg_entry_price,
            session.down_position.shares, session.down_position.avg_entry_price,
        )

        if up_ok and down_ok:
            session.strat_ref_up = session.up_position.shares
            session.strat_ref_down = session.down_position.shares
            session.strat_turnover_shares = 0.0
            return

        if not up_ok and not down_ok:
            logger.warning(
                "[#%d] Market opened with ZERO on both sides – idle until session close",
                session.market_index,
            )
            session.phase = SessionPhase.DONE
            return

        # Un solo lato: vendi tutto a bid (come safety abort) e chiudi la parte attiva
        filled_side = Side.UP if up_ok else Side.DOWN
        pos = session.position_for(filled_side)
        prices = self._get_prices_for_market(session.market)
        bid = prices.get(filled_side.value, {}).get("bid") or self.cfg.sell_price_floor
        logger.warning(
            "[#%d] Partial straddle at open – flattening %s only (%.2f sh)",
            session.market_index, filled_side.value, pos.shares,
        )
        if pos.shares > 0 and bid > 0:
            result = self._place_sell(
                session.market, filled_side, pos.shares, bid, "partial-open-flatten",
            )
            if result.ok:
                filled = result.filled_shares or pos.shares
                px = result.filled_price or bid
                session.realised_pnl += px * filled - pos.avg_entry_price * filled
        pos.shares = 0.0
        session.phase = SessionPhase.DONE

    def _background_pre_position(self, market: MarketInfo) -> None:
        """Run pre-positioning for market K+2 in background while managing K+1."""
        logger.info("[background] Pre-positioning on: %s", market.question[:70])
        prices = self._get_prices_for_market(market)
        up_ask = prices.get(Side.UP.value, {}).get("ask")
        down_ask = prices.get(Side.DOWN.value, {}).get("ask")
        up_price = self._prepos_limit_price(up_ask)
        down_price = self._prepos_limit_price(down_ask)

        with self._state_lock:
            self._market_index += 1
            market_index = self._market_index
        session = MarketSession(market=market, market_index=market_index)
        session.up_position = Position(Side.UP)
        session.down_position = Position(Side.DOWN)

        up_result, down_result = self._place_prepos_pair(
            market, up_price, down_price, "bg-prepos"
        )

        self._record_fill(session, up_result, Side.UP)
        self._record_fill(session, down_result, Side.DOWN)

        # Reconcile against on-chain balances.
        self._sync_spendable(session, Side.UP)
        self._sync_spendable(session, Side.DOWN)

        # Warm up CTF allowance early (background thread, non-blocking).
        self._warm_allowances(market)

        # Retry any unfilled side while time remains before market open.
        self._complete_prepos_with_retry(session, market)

        with self._state_lock:
            self._next_market = market
            # Store the session under a temporary key; it'll become active when the
            # current market ends.
            self._next_session = session
            self._pre_pos_done = True

        self._save_state()
        logger.info("[background] Next market pre-positioned: %s", market.question[:70])

    # ── Phase: MANAGING ───────────────────────────────────────────────────────

    def _tick_managing(self, now: float) -> None:
        session = self._session
        if session is None:
            self._phase = BotPhase.AWAITING_NEXT_MARKET
            return

        market = session.market
        remaining = market.end_timestamp - now

        # ── Market ended ──────────────────────────────────────────────
        if remaining <= 0:
            self._close_session(session, now)
            return

        # ── Get current prices ─────────────────────────────────────────
        up_prices = self._get_prices(market, Side.UP)
        down_prices = self._get_prices(market, Side.DOWN)
        price_map = {Side.UP: up_prices, Side.DOWN: down_prices}

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "[#%d] %.0fs left | UP bid=%.3f ask=%.3f | DOWN bid=%.3f ask=%.3f | phase=%s",
                session.market_index, remaining,
                up_prices.bid or 0, up_prices.ask or 0,
                down_prices.bid or 0, down_prices.ask or 0,
                session.phase.value,
            )

        if session.phase == SessionPhase.BOTH_OPEN:
            self._tick_straddle_strategy(session, price_map, remaining, now)
        # DONE: partial straddle già appiattito; attendi solo fine mercato

        if not self._pre_pos_done and self._next_market is None:
            self._probe_next_market(now)

    def _tick_straddle_strategy(
        self,
        session: MarketSession,
        price_map: dict[Side, Prices],
        remaining: float,
        now: float,
    ) -> None:
        """Strategia inventario ibrida: trim forte / debole + re-entry su spread stretto."""
        _ = remaining
        cfg = self.cfg
        up_px, dn_px = price_map[Side.UP], price_map[Side.DOWN]
        if up_px.stale or dn_px.stale:
            return
        ub, db = up_px.bid, dn_px.bid
        if ub is None or db is None:
            return

        u_sh = session.up_position.shares
        d_sh = session.down_position.shares
        if u_sh <= 0 and d_sh <= 0:
            return

        if (
            session.strat_ref_up <= 0.0
            and session.strat_ref_down <= 0.0
            and u_sh > 0
            and d_sh > 0
        ):
            session.strat_ref_up = u_sh
            session.strat_ref_down = d_sh

        if cfg.inv_mtm_log_interval > 0 and (
            now - self._last_strategy_log_ts >= cfg.inv_mtm_log_interval
        ):
            self._last_strategy_log_ts = now
            mtm = u_sh * ub + d_sh * db - session.total_cost + session.realised_pnl
            logger.info(
                "[#%d] INV MTM~=%.2f | UP %.2f@%.3f | DOWN %.2f@%.3f | realised=%.2f | turn=%.2f",
                session.market_index, mtm, u_sh, ub, d_sh, db,
                session.realised_pnl, session.strat_turnover_shares,
            )

        if not cfg.inv_enabled:
            return

        total_s = max(1.0, float(session.market.end_timestamp - session.market.start_timestamp))
        elapsed = max(0.0, now - session.market.start_timestamp)
        early = elapsed < total_s * cfg.inv_phase_split
        spread = abs(ub - db)
        late_mode = (not early) or (spread >= cfg.inv_directional_spread)
        strong = Side.UP if ub >= db else Side.DOWN
        weak = strong.opposite()

        # Endgame safety: if market is strongly directional near expiry,
        # force-unwind the weak leg even when turnover/cooldown caps are hit.
        if (
            cfg.inv_force_flatten_seconds > 0
            and remaining <= cfg.inv_force_flatten_seconds
            and spread >= cfg.inv_directional_spread
        ):
            if self._inv_try_trim_sell(session, weak, price_map, now, "force-endgame", ignore_limits=True):
                return

        if self._inv_turnover_budget_left(session) <= 0:
            return

        if self._inv_try_rebuy(session, price_map, now):
            return

        strong_bid = ub if strong is Side.UP else db
        if not late_mode and strong_bid < cfg.inv_early_strong_min_bid:
            return

        if late_mode:
            self._inv_try_trim_sell(session, weak, price_map, now, "late-weak")
        else:
            self._inv_try_trim_sell(session, strong, price_map, now, "early-strong")

    # ── Intramarket inventory helpers ───────────────────────────────────────────

    def _inv_turnover_budget_left(self, session: MarketSession) -> float:
        cap = self.cfg.inv_max_turnover_shares
        if cap <= 0:
            return 1e18
        return max(0.0, cap - session.strat_turnover_shares)

    def _inv_cooldown_ok(self, side: Side, now: float) -> bool:
        last = self._inv_last_ts_up if side is Side.UP else self._inv_last_ts_down
        return (now - last) >= self.cfg.inv_cooldown_seconds

    def _inv_mark_cooldown(self, side: Side, now: float) -> None:
        if side is Side.UP:
            self._inv_last_ts_up = now
        else:
            self._inv_last_ts_down = now

    def _inv_rebuy_deficit(self, session: MarketSession, side: Side) -> float:
        ref = session.strat_ref_up if side is Side.UP else session.strat_ref_down
        if ref <= 0:
            return 0.0
        pos = session.position_for(side)
        return max(0.0, ref - pos.shares)

    def _inv_apply_sell_fill(
        self, session: MarketSession, side: Side, result: OrderResult, limit_fallback: float
    ) -> None:
        pos = session.position_for(side)
        filled = result.filled_shares or 0.0
        if filled <= 0:
            return
        px = result.filled_price if result.filled_price else limit_fallback
        session.realised_pnl += filled * (px - pos.avg_entry_price)
        pos.shares = max(0.0, pos.shares - filled)
        session.strat_turnover_shares += filled

    def _inv_apply_buy_fill(self, session: MarketSession, side: Side, result: OrderResult) -> None:
        pos = session.position_for(side)
        filled = result.filled_shares or 0.0
        if filled <= 0:
            return
        px = result.filled_price or 0.0
        total = pos.shares + filled
        if total > 0:
            pos.avg_entry_price = (
                pos.avg_entry_price * pos.shares + px * filled
            ) / total
        pos.shares = total
        session.strat_turnover_shares += filled

    def _inv_try_trim_sell(
        self,
        session: MarketSession,
        side: Side,
        price_map: dict[Side, Prices],
        now: float,
        label: str,
        ignore_limits: bool = False,
    ) -> bool:
        cfg = self.cfg
        pos = session.position_for(side)
        if pos.shares <= 0 or pos.avg_entry_price <= 0:
            return False
        if not ignore_limits and not self._inv_cooldown_ok(side, now):
            return False

        budget = self._inv_turnover_budget_left(session)
        if not ignore_limits and budget <= 0:
            return False

        bid = price_map[side].bid
        if bid is None or bid <= 0:
            return False

        if ignore_limits:
            tier = pos.shares
        else:
            tier = min(pos.shares * cfg.inv_trim_fraction, pos.shares, budget)
        if cfg.inv_max_shares_per_trade > 0:
            tier = min(tier, cfg.inv_max_shares_per_trade)
        tier = round(tier, 4)
        if tier <= 0:
            return False

        limit = max(cfg.sell_price_floor, bid - cfg.inv_sell_slippage)
        if tier * limit < cfg.inv_min_sell_notional_usd:
            return False

        self._sync_spendable(session, side, skip_zero_guard=True)
        pos = session.position_for(side)
        if ignore_limits:
            tier = min(tier, pos.shares)
        else:
            tier = min(tier, pos.shares, budget)
        if tier <= 0:
            return False

        for attempt in range(max(1, cfg.inv_sell_max_retries)):
            result = self._place_sell(
                session.market, side, tier, limit, f"inv-{label}-{attempt + 1}",
            )
            if result.ok and (result.filled_shares or 0) > 0:
                self._inv_apply_sell_fill(session, side, result, limit)
                self._inv_mark_cooldown(side, now)
                logger.info(
                    "[#%d] INV SELL %s %.4f sh @ ~%.4f (%s) | left=%.4f | realised=%.2f",
                    session.market_index, side.value,
                    result.filled_shares or tier, result.filled_price or limit,
                    label, pos.shares, session.realised_pnl,
                )
                return True
            time.sleep(0.2)
        return False

    def _inv_try_rebuy(
        self, session: MarketSession, price_map: dict[Side, Prices], now: float
    ) -> bool:
        cfg = self.cfg
        if cfg.inv_reentry_fraction <= 0:
            return False

        ub, db = price_map[Side.UP].bid, price_map[Side.DOWN].bid
        if ub is None or db is None:
            return False
        if abs(ub - db) >= cfg.inv_reentry_spread_max:
            return False
        if ub < cfg.inv_reentry_both_min_bid or db < cfg.inv_reentry_both_min_bid:
            return False

        for side in (Side.UP, Side.DOWN):
            deficit = self._inv_rebuy_deficit(session, side)
            if deficit < 1e-6:
                continue
            if not self._inv_cooldown_ok(side, now):
                continue

            budget = self._inv_turnover_budget_left(session)
            if budget <= 0:
                return False

            want = deficit * cfg.inv_reentry_fraction
            if cfg.inv_max_rebuy_shares_per_trade > 0:
                want = min(want, cfg.inv_max_rebuy_shares_per_trade)
            want = min(want, deficit, budget)
            want = round(want, 4)
            if want <= 0:
                continue

            ask = price_map[side].ask
            if ask is None or ask <= 0:
                continue
            limit = self._prepos_limit_price(ask)
            if want * limit < cfg.inv_min_sell_notional_usd:
                continue

            result = self._place_buy(
                session.market, side, want, limit, "inv-reentry",
            )
            if result.ok and (result.filled_shares or 0) > 0:
                self._inv_apply_buy_fill(session, side, result)
                self._inv_mark_cooldown(side, now)
                logger.info(
                    "[#%d] INV RE-BUY %s +%.4f @ ~%.4f | turn=%.2f",
                    session.market_index, side.value,
                    result.filled_shares or want, result.filled_price or limit,
                    session.strat_turnover_shares,
                )
                self._sync_spendable(session, side)
                return True
        return False

    # ── Session close ─────────────────────────────────────────────────────────

    def _close_session(self, session: MarketSession, now: float) -> None:
        logger.info("[#%d] Market ENDED – closing session", session.market_index)
        session.phase = SessionPhase.DONE

        # Stima settlement solo su up_position / down_position (straddle + eventuali trade futuri)
        settlement_pnl = 0.0
        up_prices = self._get_prices(session.market, Side.UP)
        dn_prices = self._get_prices(session.market, Side.DOWN)
        up_bid = up_prices.bid or 0.5
        dn_bid = dn_prices.bid or 0.5

        if up_bid >= 0.80:
            up_est, dn_est = 1.0, 0.0
        elif dn_bid >= 0.80:
            up_est, dn_est = 0.0, 1.0
        elif up_bid + dn_bid > 0:
            total = up_bid + dn_bid
            up_est = round(up_bid / total, 4)
            dn_est = round(dn_bid / total, 4)
        else:
            up_est = dn_est = 0.5

        def _side_est(side: Side) -> float:
            return up_est if side is Side.UP else dn_est

        for side in (Side.UP, Side.DOWN):
            pos = session.position_for(side)
            if pos.shares <= 0:
                continue
            est = _side_est(side)
            p_pnl = est * pos.shares - pos.cost_basis
            settlement_pnl += p_pnl
            logger.info(
                "[#%d] Settlement est. %s: %.0f sh @ %.2f → pnl %.2f",
                session.market_index, side.value, pos.shares, est, p_pnl,
            )

        pnl = session.realised_pnl + settlement_pnl
        self._cum_pnl += pnl
        self._day_pnl += pnl
        self._day_markets += 1

        cash, pos_val, n_pos = self._wallet_snapshot()
        self._notifier.notify_session_close(
            session.market_index, session.market.end_timestamp,
            pnl, self._cum_pnl, cash, pos_val, n_pos,
        )
        logger.info(
            "[#%d] Session PnL: %.2f (realised=%.2f + settlement_est=%.2f) | Cum PnL: %.2f | Cash: $%.2f | Portfolio: $%.2f",
            session.market_index, pnl, session.realised_pnl, settlement_pnl, self._cum_pnl,
            cash, cash + pos_val,
        )

        # Transition to the next pre-positioned market if ready
        next_session = self._next_session
        if next_session and self._next_market:
            logger.info("Activating next session: %s", self._next_market.question[:70])
            self._session = next_session
            self._current_market = self._next_market
            self._next_market = None
            self._next_session = None
            self._pre_pos_done = False
            # If the market has already started (normal case: start == now), go
            # directly to MANAGING instead of POSITIONED so we don't lose a tick.
            if now >= self._current_market.start_timestamp:
                self._session.phase = SessionPhase.BOTH_OPEN
                self._feed.subscribe(list(self._current_market.token_ids))
                self._phase = BotPhase.MANAGING
                logger.info(
                    "[#%d] Market already live – entering MANAGING directly",
                    self._session.market_index,
                )
                # Verify actual on-chain balances — same check as _on_market_start.
                # Critical: without this, pos.shares from pre-pos tracking may not
                # reflect the real on-chain state and any sell attempt will fail.
                self._check_partial_prepos(self._session)
            else:
                self._phase = BotPhase.POSITIONED
        else:
            # No pre-positioned market ready; re-discover
            logger.info("No next session ready – re-discovering markets")
            self._session = None
            self._current_market = None
            self._next_market = None
            self._pre_pos_done = False
            self._phase = BotPhase.AWAITING_NEXT_MARKET

        self._save_state()

    # ── Next market probe ─────────────────────────────────────────────────────

    def _probe_next_market(self, now: float) -> None:
        if self._current_market is None:
            return
        # Only probe once we're > 50% through the market
        elapsed = now - self._current_market.start_timestamp
        if elapsed < self.cfg.market_interval_seconds * 0.5:
            return
        next_m = self._finder.find_next(self._current_market.timestamp_tag)
        if next_m and next_m.condition_id:
            self._next_market = next_m
            logger.info("Probed next market: %s", next_m.question[:70])
            # Kick off background pre-position
            t = threading.Thread(
                target=self._background_pre_position, args=(next_m,), daemon=True
            )
            t.start()

    # ── Price helpers ─────────────────────────────────────────────────────────

    def _get_prices(self, market: MarketInfo, side: Side) -> Prices:
        """Get best prices for one side, WS-first then REST fallback."""
        token_id = market.token_id_for(side)
        if not token_id:
            return Prices(stale=True)

        p = self._feed.get(token_id)
        if not p.stale:
            return p

        # REST fallback
        try:
            book = fetch_best_prices_rest([token_id])
            entry = book.get(token_id, {})
            return Prices(bid=entry.get("bid"), ask=entry.get("ask"))
        except Exception:
            return Prices(stale=True)

    def _get_prices_for_market(self, market: MarketInfo) -> dict[str, dict]:
        """Get bid/ask for all tokens of a market as {outcome: {bid, ask}}."""
        token_map = market.token_id_by_side
        token_ids = list(token_map.values())

        # Try WS first
        result: dict[str, dict] = {}
        all_fresh = True
        for outcome, tid in token_map.items():
            p = self._feed.get(tid)
            if p.stale:
                all_fresh = False
                break
            result[outcome] = {"bid": p.bid, "ask": p.ask}

        if all_fresh:
            return result

        # REST fallback
        try:
            rest = fetch_best_prices_rest(token_ids)
            return {outcome: rest.get(tid, {"bid": None, "ask": None})
                    for outcome, tid in token_map.items()}
        except Exception:
            return {outcome: {"bid": None, "ask": None} for outcome in token_map}

    # ── Order wrappers ────────────────────────────────────────────────────────

    def _place_buy(
        self, market: MarketInfo, side: Side, shares: float, limit_price: float, label: str
    ) -> OrderResult:
        req = OrderRequest(
            market=market, side=side, shares=shares,
            limit_price=limit_price, order_label=label,
        )
        logger.debug("BUY %s %.0f@%.4f (%s)", side.value, shares, limit_price, label)
        return self._executor.buy(req)

    def _place_sell(
        self, market: MarketInfo, side: Side, shares: float, limit_price: float, label: str
    ) -> OrderResult:
        req = OrderRequest(
            market=market, side=side, shares=shares,
            limit_price=limit_price, order_label=label,
        )
        logger.debug("SELL %s %.0f@%.4f (%s)", side.value, shares, limit_price, label)
        return self._executor.sell(req)

    # ── State persistence ─────────────────────────────────────────────────────

    def _save_state(self) -> None:
        with self._state_lock:
            path = Path(self.cfg.state_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            state = {
                "market_index": self._market_index,
                "cum_pnl": self._cum_pnl,
                "day_pnl": self._day_pnl,
                "day_markets": self._day_markets,
                "day_key": self._day_key,
                "phase": self._phase.value,
                "session": self._serialize_session(self._session),
            }
            try:
                tmp = path.with_suffix(".tmp")
                tmp.write_text(json.dumps(state, indent=2))
                tmp.replace(path)
            except Exception as exc:
                logger.warning("State save failed: %s", exc)

    def _load_state(self) -> None:
        path = Path(self.cfg.state_file)
        if not path.exists():
            return
        try:
            state = json.loads(path.read_text())
            self._market_index = state.get("market_index", 0)
            self._cum_pnl = state.get("cum_pnl", 0.0)
            self._day_pnl = state.get("day_pnl", 0.0)
            self._day_markets = state.get("day_markets", 0)
            self._day_key = state.get("day_key", "")
            logger.info(
                "State loaded: market_index=%d cum_pnl=%.2f",
                self._market_index, self._cum_pnl,
            )
        except Exception as exc:
            logger.warning("State load failed: %s", exc)

    def _reset_state(self) -> None:
        """Reset persisted state and start this run from a clean baseline."""
        self._market_index = 0
        self._cum_pnl = 0.0
        self._day_pnl = 0.0
        self._day_markets = 0
        self._day_key = ""
        self._phase = BotPhase.STARTING
        self._session = None
        self._next_session = None
        self._save_state()
        logger.info("State reset on start: market_index=0 cum_pnl=0.00")

    @staticmethod
    def _serialize_session(session: Optional[MarketSession]) -> Optional[dict]:
        if session is None:
            return None
        return {
            "market_index": session.market_index,
            "phase": session.phase.value,
            "realised_pnl": session.realised_pnl,
        }

    # ── Telegram command handlers ─────────────────────────────────────────────

    def _cmd_status(self) -> str:
        mode = "PAPER" if self.cfg.paper_mode else "LIVE"
        session_info = "none"
        if self._session:
            s = self._session
            session_info = (
                f"#{s.market_index} {s.phase.value} | "
                f"UP={s.up_position.shares:.0f}sh DOWN={s.down_position.shares:.0f}sh"
            )
        return (
            f"<b>Bot2 Status</b>\n"
            f"Mode: {mode} | Phase: {self._phase.value}\n"
            f"Session: {session_info}\n"
            f"Cum PnL: {self._cum_pnl:+.2f}\n"
            f"Balance: ${self._executor.get_balance_usd():.2f}"
        )

    def _cmd_pnl(self) -> str:
        return (
            f"<b>Bot2 PnL</b>\n"
            f"Day: {self._day_pnl:+.2f} ({self._day_markets} markets)\n"
            f"Cumulative: {self._cum_pnl:+.2f}"
        )

    def _cmd_stop(self) -> str:
        if not self._running.is_set():
            return "Bot2 is already stopping/stopped."
        self._stop_requested_via_telegram = True
        self._running.clear()
        return "🛑 Stop requested via Telegram. Waiting for graceful shutdown confirmation..."
