"""Main bot orchestrator.

Lifecycle loop:
    1.  Discover the current 15-min market (K).
    2.  Wait for K to end and resolve via CLOB API.
    3.  Record K's outcome; settle any trade on K (update progression + lock).
    4.  Place one order for K+1: side from strategy.predict() (lag2 + lock),
        size from progression.effective_size().  Both are fully determined
        because K's settlement just ran.  K+1 has been live for ~2-5 min
        but 50/51c prices are still available.
    5.  Move to watch K+1 and repeat.
"""
from __future__ import annotations

import logging
import signal
import time
from datetime import datetime, timezone
from typing import Optional

from bot5_martingale.config import Bot5Config
from bot5_martingale.executor import BaseExecutor, PaperExecutor, create_executor
from bot5_martingale.logger_setup import TradeCSVLogger, setup_logging
from bot5_martingale.market_finder import (
    MarketFinder,
    MarketResolver,
    infer_outcome_from_prices,
    market_end,
    market_slug,
)
from bot5_martingale.models import (
    BotMode,
    MarketInfo,
    OrderRequest,
    OrderResult,
    OrderStatus,
    Side,
    TradeRecord,
    TradeResult,
)
from bot5_martingale.notifier import TelegramNotifier, create_notifier
from bot5_martingale.progression import MartingaleProgression
from bot5_martingale.state import StateManager
from bot5_martingale.strategy import Lag2LockStrategy

logger = logging.getLogger(__name__)


class Bot5:
    """Top-level controller for the martingale trading bot."""

    def __init__(self, cfg: Bot5Config):
        self.cfg = cfg
        self._root_logger = setup_logging(cfg)
        self.csv_logger = TradeCSVLogger(cfg)

        self.strategy = Lag2LockStrategy(
            lag=cfg.lag,
            lock_after_n=cfg.lock_after_n_losses,
            lock_mode=cfg.lock_mode,
        )
        self.progression = MartingaleProgression(
            ladder=cfg.ladder,
            max_level=cfg.max_level,
            target_profit_per_market=cfg.target_profit_per_win,
            max_entry_price=cfg.max_entry_price,
        )
        self.finder = MarketFinder(cfg)
        self.resolver = MarketResolver()
        self.executor: BaseExecutor = create_executor(cfg)
        self.state_mgr = StateManager(cfg.state_file)
        self.notifier: TelegramNotifier = create_notifier(cfg)

        self._running = False
        self._current_day: str = ""

        # Active tracked objects
        self._watched_market: Optional[MarketInfo] = None
        self._watched_start_ts: int = 0
        self._placed_orders: dict[str, _PendingTrade] = {}  # condition_id → pending
        self._early_signal_side: Optional[Side] = None
        self._early_signal_since: float = 0.0
        self._early_order_placed_for_cid: Optional[str] = None

        self._restore_state()
        self._register_commands()

    # ==================================================================
    # State persistence helpers
    # ==================================================================

    def _restore_state(self) -> None:
        data = self.state_mgr.load()
        self.strategy.restore(data.strategy)
        self.progression.restore(data.progression)
        self._watched_start_ts = data.last_resolved_start_ts
        logger.info(
            "Restored: idx=%d level=%d pnl=%.2f",
            data.market_sequential_index,
            self.progression.level,
            data.total_pnl,
        )

    def _persist(self) -> None:
        d = self.state_mgr.data
        d.strategy = self.strategy.snapshot()
        d.progression = self.progression.snapshot()
        d.last_resolved_start_ts = self._watched_start_ts
        self.state_mgr.save()

    # ==================================================================
    # Telegram commands
    # ==================================================================

    def _register_commands(self) -> None:
        self.notifier.register_command("status", self._cmd_status)
        self.notifier.register_command("pnl", self._cmd_pnl)
        self.notifier.register_command("streak", self._cmd_streak)
        self.notifier.register_command("stop_bot", self._cmd_stop)
        self.notifier.register_command("start_bot", lambda: "Bot is already running")

    def _cmd_status(self) -> str:
        d = self.state_mgr.data
        return (
            f"<b>Bot5 Status</b>\n"
            f"Mode: {self.cfg.mode.value}\n"
            f"Running: {self._running}\n"
            f"Market idx: {d.market_sequential_index}\n"
            f"Progression: level {self.progression.level}\n"
            f"Locked: {self.strategy.is_locked} ({self.strategy.locked_side})\n"
            f"Trades: {d.total_trades} | W: {d.total_wins} | L: {d.total_losses}"
        )

    def _cmd_pnl(self) -> str:
        d = self.state_mgr.data
        avg = d.total_pnl / max(d.total_trades, 1)
        return (
            f"<b>PnL Report</b>\n"
            f"Total PnL: {d.total_pnl:+.2f}\n"
            f"Avg/trade: {avg:+.2f}\n"
            f"Daily PnL: {d.daily.pnl:+.2f}\n"
            f"Stopouts: {d.total_stopouts}"
        )

    def _cmd_streak(self) -> str:
        return (
            f"<b>Streak</b>\n"
            f"Level: {self.progression.level}\n"
            f"Consecutive losses: {self.strategy.consecutive_losses}\n"
            f"Locked: {self.strategy.is_locked} side={self.strategy.locked_side}\n"
            f"Cum spend this cycle: ${self.progression.cumulative_spend:.2f}"
        )

    def _cmd_stop(self) -> str:
        self._running = False
        return "Stop requested. Bot will halt after current tick."

    # ==================================================================
    # Main loop
    # ==================================================================

    def run(self) -> None:
        self._running = True
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

        mode_str = self.cfg.mode.value.upper()
        logger.info("=" * 60)
        logger.info("BOT5 MARTINGALE – %s MODE", mode_str)
        logger.info("=" * 60)
        logger.info("Strategy: lag=%d lock_after=%d lock_mode=%s",
                     self.cfg.lag, self.cfg.lock_after_n_losses, self.cfg.lock_mode)
        logger.info("Ladder: %s  max_level=%d", self.cfg.ladder, self.cfg.max_level)
        logger.info("Target profit/win: $%.2f  max_entry_price: %.4f",
                     self.cfg.target_profit_per_win, self.cfg.max_entry_price)
        logger.info("=" * 60)
        self.notifier.notify_start(mode_str)

        try:
            self._loop()
        except Exception as exc:
            logger.exception("Fatal error: %s", exc)
            self.notifier.notify_error(str(exc))
        finally:
            self._persist()
            stop_reason = "manual"
            if self.cfg.state_reset_on_stop:
                self.state_mgr.reset_all()
                stop_reason = "manual + state reset"
                logger.info("State reset on stop enabled: state file reinitialized")
            self.notifier.notify_stop(stop_reason)
            self.notifier.shutdown()
            logger.info("Bot5 shut down")

    def _loop(self) -> None:
        while self._running:
            try:
                self._check_daily_rollover()
                self._tick()
            except KeyboardInterrupt:
                self._running = False
                break
            except Exception as exc:
                logger.error("Tick error: %s", exc, exc_info=True)
                self.notifier.notify_error(f"Tick error: {exc}")
                time.sleep(5)

            time.sleep(self.cfg.tick_interval)

    def _handle_signal(self, signum, frame) -> None:
        logger.info("Signal %d received, shutting down gracefully", signum)
        self._running = False

    # ==================================================================
    # Core tick
    # ==================================================================

    def _tick(self) -> None:
        # 1. Ensure we have a market to watch
        if self._watched_market is None:
            self._find_and_watch_market()
            if self._watched_market is None:
                return

        now = time.time()
        end_ts = self._watched_market.end_timestamp

        # 2. Market still running – check for 99c inference in last minute
        if now < end_ts:
            remaining = end_ts - now
            if self.cfg.early_signal_enabled and remaining <= self.cfg.early_signal_window_seconds:
                self._try_early_operational_signal(now)
            if remaining <= self.cfg.inference_window_seconds:
                self._try_inference()
            return

        # 3. Market ended – resolve it
        self._resolve_current_market()

    # ==================================================================
    # Market discovery
    # ==================================================================

    def _find_and_watch_market(self) -> None:
        if self._watched_start_ts:
            # _watched_start_ts already points to the NEXT market's start
            # (it was incremented at the end of _resolve_current_market)
            info = self.finder.fetch_market_by_start(self._watched_start_ts)
            if info and info.condition_id:
                self._set_watched(info, self._watched_start_ts)
                return

        info = self.finder.find_current()
        if info and info.condition_id:
            start_ts = info.timestamp_tag or (info.end_timestamp - self.cfg.market_interval_seconds)
            self._set_watched(info, start_ts)

    def _set_watched(self, info: MarketInfo, start_ts: int) -> None:
        self._watched_market = info
        self._watched_start_ts = start_ts
        self._early_signal_side = None
        self._early_signal_since = 0.0
        self._early_order_placed_for_cid = None
        logger.info(
            "Watching market %s | ends in %.0fs | idx=%d",
            info.question[:60],
            info.seconds_left,
            self.state_mgr.data.market_sequential_index,
        )

    # ==================================================================
    # 99c inference
    # ==================================================================

    def _try_inference(self) -> None:
        if self._watched_market is None:
            return
        prices = self.executor.get_best_prices(self._watched_market)
        inferred = infer_outcome_from_prices(prices, threshold=self.cfg.inference_price_threshold)
        if inferred is None:
            return

        logger.info(
            "99c inference: %s for %s (%.0fs before end)",
            inferred.value,
            self._watched_market.question[:50],
            self._watched_market.seconds_left,
        )

        # Tag the pending trade with the inferred result (informational only)
        cid = self._watched_market.condition_id
        pending = self._placed_orders.get(cid)
        if pending is not None and pending.result is None:
            pending.inferred_outcome = inferred
            pending.inferred_won = (inferred == pending.side)

    def _try_early_operational_signal(self, now_ts: float) -> None:
        """Operational early signal near market end.

        If one side stays above threshold for N consecutive seconds,
        we place the next market order using a projected post-settlement state.
        Official resolution still remains authoritative for final accounting.
        """
        if self._watched_market is None:
            return
        if self._early_order_placed_for_cid == self._watched_market.condition_id:
            return

        prices = self.executor.get_best_prices(self._watched_market)
        candidate = infer_outcome_from_prices(
            prices, threshold=self.cfg.early_signal_price_threshold
        )
        if candidate is None:
            self._early_signal_side = None
            self._early_signal_since = 0.0
            return

        if candidate != self._early_signal_side:
            self._early_signal_side = candidate
            self._early_signal_since = now_ts
            return

        if (now_ts - self._early_signal_since) < float(self.cfg.early_signal_confirm_seconds):
            return

        projected = self._project_next_trade_from_inference(candidate)
        if projected is None:
            return
        prediction, size, level = projected

        idx = self.state_mgr.data.market_sequential_index + 1
        target_start = self._watched_start_ts + self.cfg.market_interval_seconds
        target_market = self.finder.fetch_market_by_start(target_start)
        if target_market is None:
            return
        if target_market.condition_id in self._placed_orders:
            self._early_order_placed_for_cid = self._watched_market.condition_id
            return

        logger.info(
            "EARLY signal confirmed (%s >= %.2f for %ds): placing projected order for idx=%d",
            candidate.value,
            self.cfg.early_signal_price_threshold,
            self.cfg.early_signal_confirm_seconds,
            idx,
        )
        self._submit_trade_order(
            idx=idx,
            target_market=target_market,
            prediction=prediction,
            size=size,
            level=level,
            tag="early_signal",
            early=True,
        )
        self._early_order_placed_for_cid = self._watched_market.condition_id

    def _project_next_trade_from_inference(self, inferred_outcome: Side) -> Optional[tuple[Side, float, int]]:
        """Project side/size for next market using inferred current outcome.

        This does not mutate live state; it only mirrors strategy/progression transitions.
        """
        if self._watched_market is None:
            return None

        tmp_strategy = Lag2LockStrategy(
            lag=self.cfg.lag,
            lock_after_n=self.cfg.lock_after_n_losses,
            lock_mode=self.cfg.lock_mode,
        )
        tmp_strategy.restore(self.strategy.snapshot())
        tmp_strategy.record_outcome(inferred_outcome)

        tmp_prog = MartingaleProgression(
            ladder=self.cfg.ladder,
            max_level=self.cfg.max_level,
            target_profit_per_market=self.cfg.target_profit_per_win,
            max_entry_price=self.cfg.max_entry_price,
        )
        tmp_prog.restore(self.progression.snapshot())

        pending = self._placed_orders.get(self._watched_market.condition_id)
        if pending is not None and pending.fill_result is not None and pending.fill_result.is_filled:
            won = (pending.side == inferred_outcome)
            if won:
                tmp_prog.on_win(pending.fill_result.filled_size, pending.fill_result.filled_shares)
            else:
                tmp_prog.on_loss(pending.fill_result.filled_size)
            tmp_strategy.report_result(pending.side, won)

        next_idx = self.state_mgr.data.market_sequential_index + 1
        pred = tmp_strategy.predict(next_idx)
        if pred is None:
            return None
        return pred, tmp_prog.effective_size(self.cfg.max_entry_price), tmp_prog.level

    # ==================================================================
    # Resolution + settlement
    # ==================================================================

    def _resolve_current_market(self) -> None:
        if self._watched_market is None:
            return

        cid = self._watched_market.condition_id
        outcome = self.resolver.resolve(cid)
        if outcome is None:
            # Fallback: orderbook heuristic
            prices = self.executor.get_best_prices(self._watched_market)
            outcome = infer_outcome_from_prices(prices, threshold=0.90)
            if outcome is None:
                logger.warning("Cannot resolve %s yet, will retry", cid[:12])
                return

        idx = self.state_mgr.data.market_sequential_index
        logger.info(
            "Resolved market #%d: %s -> %s",
            idx, self._watched_market.question[:50], outcome.value,
        )

        # Record in strategy
        self.strategy.record_outcome(outcome)

        # Settle any trade we had on this market
        self._settle_trade(cid, outcome, idx)

        # Advance index
        self.state_mgr.data.market_sequential_index = idx + 1

        # Place order for the NEXT market (K+1) – side and size are now
        # fully determined because K's settlement just updated lock + progression.
        self._place_next_trade()

        # Move to next market
        self._watched_start_ts += self.cfg.market_interval_seconds
        self._watched_market = None
        self._persist()

    def _settle_trade(self, condition_id: str, resolved_outcome: Side, market_idx: int) -> None:
        pending = self._placed_orders.pop(condition_id, None)
        if pending is None:
            logger.debug("No trade to settle for %s", condition_id[:16])
            return
        if pending.fill_result is None:
            logger.warning("Trade for %s had no fill result", condition_id[:16])
            return

        fill = pending.fill_result
        if not fill.is_filled:
            logger.warning("Trade for %s not filled (status=%s)", condition_id[:16], fill.status.value)
            return

        total_cost = fill.filled_size
        total_shares = fill.filled_shares
        won = (pending.side == resolved_outcome)
        avg_price = total_cost / max(total_shares, 1e-9)

        if won:
            payout = total_shares  # each share pays $1
            net = self.progression.on_win(total_cost, payout)
            self.strategy.report_result(pending.side, True)
            # Paper: credit the win
            if isinstance(self.executor, PaperExecutor):
                self.executor.credit_win(condition_id)
        else:
            stopout = self.progression.on_loss(total_cost)
            self.strategy.report_result(pending.side, False)
            payout = 0.0
            net = -total_cost
            # Paper: debit the loss
            if isinstance(self.executor, PaperExecutor):
                self.executor.debit_loss(condition_id)
            if stopout:
                logger.warning("STOP-OUT at level %d", self.cfg.max_level)
                self.state_mgr.record_stopout()
                self.notifier.notify_stopout(self.cfg.max_level, self.state_mgr.data.total_pnl + net)

        rec = TradeRecord(
            market_question=pending.market.question if pending.market else "",
            condition_id=condition_id,
            side=pending.side,
            progression_level=pending.level,
            target_size_usd=pending.target_size,
            filled_size_usd=total_cost,
            filled_price=avg_price,
            filled_shares=total_shares,
            fee=fill.fee,
            result=TradeResult.WIN if won else TradeResult.LOSS,
            pnl=net,
            ts_entry=fill.ts,
            ts_resolved=time.time(),
            inferred=pending.inferred_won is not None,
        )
        self.state_mgr.record_trade(rec)
        self.csv_logger.log({
            "ts_entry": rec.ts_entry,
            "ts_resolved": rec.ts_resolved,
            "market": rec.market_question[:80],
            "condition_id": rec.condition_id[:16],
            "side": rec.side.value,
            "level": rec.progression_level,
            "target_usd": f"{rec.target_size_usd:.2f}",
            "filled_usd": f"{rec.filled_size_usd:.2f}",
            "price": f"{rec.filled_price:.4f}",
            "shares": f"{rec.filled_shares:.2f}",
            "fee": f"{rec.fee:.4f}",
            "result": rec.result.value,
            "pnl": f"{rec.pnl:.2f}",
            "cum_pnl": f"{self.state_mgr.data.total_pnl:.2f}",
            "inferred": rec.inferred,
        })

        result_str = "WIN" if won else "LOSS"
        logger.info(
            "[%s] level=%d side=%s cost=$%.2f shares=%.2f pnl=%+.2f cum_pnl=%+.2f",
            result_str, pending.level, pending.side.value,
            total_cost, total_shares, net, self.state_mgr.data.total_pnl,
        )
        self.notifier.notify_result(
            won=won, pnl=net, level=pending.level,
            cum_pnl=self.state_mgr.data.total_pnl,
            total_trades=self.state_mgr.data.total_trades,
            idx=market_idx,
            side=pending.side.value,
        )

    # ==================================================================
    # Order placement
    # ==================================================================

    def _place_next_trade(self) -> None:
        """Place a single order for the NEXT market (idx) after resolution.

        At call time, market_sequential_index has already been incremented
        to the next market, and both strategy (lock state) and progression
        (level, cumulative spend) reflect the just-settled trade's result.
        This guarantees that side and size are fully correct.

        The target market has already started (~2-5 min ago), but at 50/51c
        prices should still be available.
        """
        idx = self.state_mgr.data.market_sequential_index
        target_start = self._watched_start_ts + self.cfg.market_interval_seconds

        prediction = self.strategy.predict(idx)
        if prediction is None:
            logger.debug("Not enough history to predict idx=%d, skipping", idx)
            return

        target_market = self.finder.fetch_market_by_start(target_start)
        if target_market is None:
            logger.warning("Market for ts=%d (idx=%d) not found", target_start, idx)
            return
        if target_market.condition_id in self._placed_orders:
            return

        size = self.progression.effective_size(self.cfg.max_entry_price)
        level = self.progression.level
        self._submit_trade_order(
            idx=idx,
            target_market=target_market,
            prediction=prediction,
            size=size,
            level=level,
            tag=f"level_{level}",
        )

    def _submit_trade_order(
        self,
        *,
        idx: int,
        target_market: MarketInfo,
        prediction: Side,
        size: float,
        level: int,
        tag: str,
        early: bool = False,
    ) -> None:
        req = OrderRequest(
            market=target_market,
            side=prediction,
            size_usd=size,
            limit_price=self.cfg.max_entry_price,
            tag=tag,
        )

        prefix = "EARLY " if early else ""
        logger.info(
            "%sPlacing order: %s $%.2f @ %.4f for market ts=%d (idx=%d, level=%d)",
            prefix, prediction.value, size, self.cfg.max_entry_price,
            target_market.timestamp_tag, idx, level,
        )
        self.notifier.notify_order(
            prediction.value, size, self.cfg.max_entry_price,
            target_market.question[:60], level, idx, early=early,
        )

        result = self.executor.place_order(req)
        if not result.is_filled:
            for attempt in range(1, self.cfg.order_max_retries + 1):
                time.sleep(self.cfg.order_poll_interval)
                logger.info("Retry %d/%d for idx=%d", attempt, self.cfg.order_max_retries, idx)
                result = self.executor.place_order(req)
                if result.is_filled:
                    break

        self._placed_orders[target_market.condition_id] = _PendingTrade(
            side=prediction,
            level=level,
            target_size=size,
            market=target_market,
            fill_result=result if result.is_filled else None,
        )

        if result.is_filled:
            logger.info(
                "Order filled: %s %.2f shares @ %.4f (order %s)",
                prediction.value, result.filled_shares,
                result.filled_price, result.order_id or "n/a",
            )
            self.notifier.notify_fill(
                prediction.value, result.filled_shares,
                result.filled_price, result.order_id or "", idx,
            )
        else:
            logger.error("Order NOT filled after retries: %s", result.status.value)
            self.notifier.notify_error(f"Order failed for idx={idx}: {result.error or result.status.value}")

    # ==================================================================
    # Daily rollover
    # ==================================================================

    def _check_daily_rollover(self) -> None:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today != self._current_day:
            if self._current_day:
                d = self.state_mgr.data.daily
                self.notifier.notify_daily_summary(
                    self._current_day, d.trades, d.wins,
                    d.losses, d.pnl, self.state_mgr.data.total_pnl,
                )
            self._current_day = today
            self.state_mgr.reset_daily(today)


# ======================================================================
# Internal helper: pending trade tracker
# ======================================================================

class _PendingTrade:
    __slots__ = (
        "side", "level", "target_size", "market",
        "fill_result", "inferred_outcome", "inferred_won",
    )

    def __init__(
        self,
        side: Side,
        level: int,
        target_size: float,
        market: Optional[MarketInfo] = None,
        fill_result: Optional[OrderResult] = None,
    ):
        self.side = side
        self.level = level
        self.target_size = target_size
        self.market = market
        self.fill_result = fill_result
        self.inferred_outcome: Optional[Side] = None
        self.inferred_won: Optional[bool] = None

    @property
    def result(self) -> Optional[bool]:
        return self.inferred_won
