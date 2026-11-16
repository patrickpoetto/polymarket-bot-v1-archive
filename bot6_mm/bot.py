"""Bot6 – delta-neutral market maker on Polymarket BTC 5-min binary markets.

Strategy:
  1. For every active market: place GTC limit BUY and SELL on both UP and DOWN.
  2. Quote around the current mid at ±half_spread.
  3. When a BUY fills → immediately place a SELL at avg_cost + 2×half_spread.
  4. When a SELL fills → record realized PnL, re-place BUY if room remains.
  5. Reprice open orders whenever mid moves by ≥ mm_requote_threshold.
  6. Regime filter: mid < stop_buying_below → no BUYs on that side; mid ≥ stop_selling_above
     continuously for winner_hold_confirm_seconds → no SELLs (hold to $1). Hold clears if
     mid drops below winner_hold_reset_below (hysteresis vs brief 0.70 spikes).
  7. N seconds before market end → cancel all GTC, hold inventory to resolution.
  8. At end: FAK-sell the side with lower mid (loser), winner inventory settles at 1.00.
  9. Skewed SELL size: overweight vs target and/or weak mid → larger GTC sell clips (capped) to
     reduce loser inventory during MM; endgame urgency bumps mult; derisk FAK trims scale up too.
"""
from __future__ import annotations

import logging
import signal
import time
from dataclasses import dataclass
from typing import Optional

from bot2_btc.market_finder import MarketFinder
from bot2_btc.models import MarketInfo, Side
from bot2_btc.models import Prices
from bot2_btc.price_feed import PriceFeed
from bot6_mm.config import Bot6Config
from bot6_mm.executor import BaseExecutor, LiveExecutor, PaperExecutor
from bot6_mm.models import (
    ActiveOrder,
    MarketSession,
    OrderSide,
    SideInventory,
)
from bot6_mm.notifier import Notifier

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InvMgmtSnapshot:
    """Per-tick caps / blocks from UP_mid − DN_mid (continuous inventory mgmt)."""
    gap: float
    up_cap: float
    dn_cap: float
    block_up_buy: bool
    block_dn_buy: bool
    disfavored_up: bool
    disfavored_dn: bool


def _token(mkt: MarketInfo, side: Side) -> str:
    """Return the token_id for the given side."""
    tid = mkt.token_id_for(side)
    if tid is None:
        raise ValueError(f"No token_id for side {side} in market {mkt.condition_id}")
    return tid


def _market_window_bounds(mkt: MarketInfo, interval_seconds: float) -> tuple[float, float]:
    """Unix (start, end) for the market window from slug ``timestamp_tag``.

    API ``endDate`` can be a later resolution time; use slug + interval for trading bounds.
    """
    iv = float(interval_seconds)
    tag = getattr(mkt, "timestamp_tag", 0) or 0
    if tag > 0:
        start = float(tag)
        end = start + iv
        api_end = float(mkt.end_timestamp)
        if api_end > 0 and abs(api_end - end) > 120:
            logger.warning(
                "window ends unix=%.0f (slug) vs API endDate=%.0f (Δ%.0fs) — using slug",
                end,
                api_end,
                api_end - end,
            )
        return start, end
    if mkt.end_timestamp > 0:
        end = float(mkt.end_timestamp)
        return end - iv, end
    now = time.time()
    logger.warning("Market missing timestamp_tag and end_timestamp; using now as window start")
    return now, now + iv


class Bot6:
    def __init__(self, cfg: Bot6Config):
        self.cfg = cfg
        self._executor: BaseExecutor = (
            PaperExecutor(cfg) if cfg.paper_mode else LiveExecutor(cfg)
        )
        self._notifier = Notifier(cfg)
        self._price_feed = PriceFeed(cfg.ws_url, stale_seconds=cfg.price_stale_seconds)
        self._market_finder = MarketFinder(cfg)
        self._session: Optional[MarketSession] = None
        self._market_index = 0
        self._stop = False
        self._cum_pnl = 0.0

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def run(self) -> None:
        logger.info(
            "Bot6 MM starting (paper=%s) interval=%ds pattern=%s inv_mgmt=%s",
            self.cfg.paper_mode,
            self.cfg.market_interval_seconds,
            self.cfg.market_url_pattern,
            self.cfg.mm_inv_mgmt_enabled,
        )
        signal.signal(signal.SIGINT,  self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

        balance = self._executor.get_balance_usd()
        self._notifier.notify_start(balance, self.cfg.paper_mode)
        self._price_feed.start()

        while not self._stop:
            try:
                self._find_and_run_market()
            except Exception as exc:
                logger.exception("Bot6 outer loop error: %s", exc)
                time.sleep(5)

        self._price_feed.stop()
        balance = self._executor.get_balance_usd()
        self._notifier.notify_stop(balance, self._cum_pnl)
        logger.info("Bot6 stopped. Cumulative PnL: $%.4f", self._cum_pnl)

    def _handle_signal(self, signum, _frame) -> None:
        logger.info("Bot6 received signal %s – shutting down", signum)
        self._stop = True
        if self._session:
            self._cancel_all_orders(self._session)

    # ── Market discovery ──────────────────────────────────────────────────────

    def _find_and_run_market(self) -> None:
        logger.info("Bot6 searching for current %ds BTC market…", self.cfg.market_interval_seconds)
        current = self._market_finder.find_current()
        if current is None:
            logger.warning("No current market found – retrying in 30s")
            time.sleep(30)
            return

        # Default: trade ``find_current()`` — the active/upcoming slug from Polymarket.
        # Using find_next(current) here skips that entire window and sleeps until the
        # following one (bad right after a session closes on the same boundary).
        if self.cfg.mm_prefer_next_window:
            nxt = self._market_finder.find_next(current.start_timestamp)
            target = nxt if nxt is not None else current
            logger.info("BOT6_MM_PREFER_NEXT_WINDOW=1 → targeting next slug after current")
        else:
            target = current
        self._market_index += 1
        self._run_market(target)

    # ── Per-market loop ───────────────────────────────────────────────────────

    def _run_market(self, mkt: MarketInfo) -> None:
        up_tid = _token(mkt, Side.UP)
        dn_tid = _token(mkt, Side.DOWN)

        session = MarketSession(market_info=mkt, market_index=self._market_index)
        self._session = session

        self._price_feed.subscribe([up_tid, dn_tid])
        logger.info("[#%d] Starting MM on %s", session.market_index, mkt.question)

        market_start, market_end = _market_window_bounds(mkt, self.cfg.market_interval_seconds)
        now = time.time()
        if market_start > now + 2:
            wait = market_start - now
            logger.info("[#%d] Waiting %.0fs for market start…", session.market_index, wait)
            self._sleep_until(market_start)

        self._notifier.notify_session_start(session)

        # Main tick loop
        while not self._stop:
            now = time.time()
            time_left = market_end - now

            if time_left <= 0:
                break

            up_prices = self._get_prices(up_tid)
            dn_prices = self._get_prices(dn_tid)

            # Simulate paper fills each tick
            if self.cfg.paper_mode and isinstance(self._executor, PaperExecutor):
                if up_prices.bid and up_prices.ask:
                    self._executor.simulate_fills(up_tid, up_prices.bid, up_prices.ask)
                if dn_prices.bid and dn_prices.ask:
                    self._executor.simulate_fills(dn_tid, dn_prices.bid, dn_prices.ask)

            # Poll GTC orders for new fills
            self._poll_fills(session)

            # Bias flags every tick (winner-hold uses wall-clock streak, not requote cadence)
            if not session.quoting_stopped:
                self._update_bias_flags(session, up_prices, dn_prices)

            # Keep endgame de-risk active until market end (also after quoting is stopped).
            self._endgame_derisk(session, up_prices, dn_prices, up_tid, dn_tid, time_left)

            if self.cfg.mm_inv_mgmt_enabled and not session.quoting_stopped:
                self._inv_mgmt_taker_trim(session, up_prices, dn_prices, up_tid, dn_tid)

            if time_left <= self.cfg.stop_quoting_seconds:
                if not session.quoting_stopped:
                    logger.info("[#%d] Stopping quotes – %.0fs left", session.market_index, time_left)
                    self._cancel_all_orders(session)
                    session.quoting_stopped = True
            else:
                self._update_quotes(session, up_prices, dn_prices, up_tid, dn_tid, time_left)

            time.sleep(self.cfg.tick_interval)

        self._close_session(session)

    # ── Regime filter (stop-buy loser / hold winner with confirmation) ─────────

    def _winner_hold_active(self, session: MarketSession, side: Side) -> bool:
        return session.up_winner_hold if side == Side.UP else session.dn_winner_hold

    def _inv_mgmt_snapshot(
        self,
        up_prices: Prices,
        dn_prices: Prices,
        up_inv: float,
        dn_inv: float,
    ) -> Optional[InvMgmtSnapshot]:
        if not self.cfg.mm_inv_mgmt_enabled:
            return None
        up_mid = up_prices.mid
        dn_mid = dn_prices.mid
        if up_mid is None or dn_mid is None:
            return None

        gap = up_mid - dn_mid
        max_i = self.cfg.mm_max_inventory
        ng = self.cfg.mm_inv_neutral_gap
        slope = self.cfg.mm_inv_cap_slope
        stop_gap = self.cfg.mm_inv_stop_buy_gap
        neutral_cap = (
            self.cfg.mm_inv_neutral_max_cap
            if self.cfg.mm_inv_neutral_max_cap > 0
            else max_i
        )

        if gap > ng:
            excess = gap - ng
            up_cap = max_i
            dn_cap = max(0.0, max_i - slope * excess)
            block_dn = gap >= stop_gap
            block_up = False
            if block_dn:
                dn_cap = min(dn_cap, dn_inv)
            return InvMgmtSnapshot(
                gap=gap,
                up_cap=up_cap,
                dn_cap=dn_cap,
                block_up_buy=block_up,
                block_dn_buy=block_dn,
                disfavored_up=False,
                disfavored_dn=True,
            )
        if gap < -ng:
            excess = -gap - ng
            dn_cap = max_i
            up_cap = max(0.0, max_i - slope * excess)
            block_up = (-gap) >= stop_gap
            block_dn = False
            if block_up:
                up_cap = min(up_cap, up_inv)
            return InvMgmtSnapshot(
                gap=gap,
                up_cap=up_cap,
                dn_cap=dn_cap,
                block_up_buy=block_up,
                block_dn_buy=block_dn,
                disfavored_up=True,
                disfavored_dn=False,
            )
        return InvMgmtSnapshot(
            gap=gap,
            up_cap=neutral_cap,
            dn_cap=neutral_cap,
            block_up_buy=False,
            block_dn_buy=False,
            disfavored_up=False,
            disfavored_dn=False,
        )

    def _inv_mgmt_taker_trim(
        self,
        session: MarketSession,
        up_prices: Prices,
        dn_prices: Prices,
        up_tid: str,
        dn_tid: str,
    ) -> None:
        trim_gap = self.cfg.mm_inv_taker_trim_gap
        if trim_gap >= 0.99:
            return
        snap = self._inv_mgmt_snapshot(
            up_prices, dn_prices, session.up_inv.shares, session.dn_inv.shares
        )
        if snap is None:
            return
        if abs(snap.gap) < trim_gap:
            return
        now = time.time()
        if (now - session.last_inv_mgmt_trim_at) < self.cfg.mm_inv_trim_interval:
            return

        if snap.gap > 0 and session.dn_inv.shares > 0.01:
            loser_side, loser_tid, loser_inv = Side.DOWN, dn_tid, session.dn_inv
            bid = dn_prices.bid
        elif snap.gap < 0 and session.up_inv.shares > 0.01:
            loser_side, loser_tid, loser_inv = Side.UP, up_tid, session.up_inv
            bid = up_prices.bid
        else:
            return

        if bid is None or bid <= 0:
            return

        buy_key = session.order_key(loser_side, OrderSide.BUY)
        pending_buy = session.active_orders.pop(buy_key, None)
        if pending_buy is not None:
            self._executor.cancel_order(pending_buy.order_id)

        floor = max(self.cfg.exit_sell_floor, bid)
        filled = self._executor.place_fak_sell(loser_tid, floor, loser_inv.shares, loser_side)
        if filled and filled > 0:
            session.last_inv_mgmt_trim_at = now
            loser_inv.on_sell_fill(floor, filled)
            logger.info(
                "[#%d] INV MGMT TRIM %s: FAK %.2f @%.4f (gap=%.3f)",
                session.market_index,
                loser_side.value,
                filled,
                floor,
                snap.gap,
            )

    def _compute_inventory_targets(
        self,
        session: MarketSession,
        up_prices: Prices,
        dn_prices: Prices,
        time_left: float,
    ) -> None:
        up_mid = up_prices.mid
        dn_mid = dn_prices.mid
        if up_mid is None or dn_mid is None:
            session.up_target_inventory = self.cfg.mm_target_total_inventory / 2.0
            session.dn_target_inventory = self.cfg.mm_target_total_inventory / 2.0
            return

        # Implied P(Up) from relative mids (handles sum ≠ 1); smooth with EMA.
        denom = max(up_mid + dn_mid, 1e-6)
        raw_p = up_mid / denom
        raw_p = max(0.02, min(0.98, raw_p))

        if session.ema_p_up is None:
            session.ema_p_up = raw_p
        else:
            a = self.cfg.mm_target_ema_alpha
            session.ema_p_up = (a * raw_p) + ((1.0 - a) * session.ema_p_up)

        p_up = session.ema_p_up
        in_endgame = time_left <= self.cfg.mm_endgame_seconds
        max_skew = self.cfg.mm_max_skew_endgame if in_endgame else self.cfg.mm_max_skew_normal
        min_p = 1.0 - max_skew
        p_up = max(min_p, min(max_skew, p_up))

        total = min(self.cfg.mm_target_total_inventory, 2 * self.cfg.mm_max_inventory)
        up_t = max(0.0, min(self.cfg.mm_max_inventory, total * p_up))
        dn_t = max(0.0, min(self.cfg.mm_max_inventory, total - up_t))

        # Winner-hold: stop building the likely loser; give remaining budget to winner side.
        hold_cap = self.cfg.mm_hold_opponent_max_target
        if session.up_winner_hold and not session.dn_winner_hold:
            dn_t = min(dn_t, hold_cap)
            up_t = max(0.0, min(self.cfg.mm_max_inventory, total - dn_t))
        elif session.dn_winner_hold and not session.up_winner_hold:
            up_t = min(up_t, hold_cap)
            dn_t = max(0.0, min(self.cfg.mm_max_inventory, total - up_t))

        session.up_target_inventory = up_t
        session.dn_target_inventory = dn_t

    def _update_bias_flags(
        self,
        session: MarketSession,
        up_prices: Prices,
        dn_prices: Prices,
    ) -> None:
        """Update winner-hold streaks and flags (confirmed high mid, hysteresis on exit)."""
        confirm = self.cfg.mm_winner_hold_confirm_seconds
        hi = self.cfg.mm_stop_selling_above
        lo = self.cfg.mm_winner_hold_reset_below
        now = time.time()

        if confirm <= 0.0:
            # Feature off: clear state
            if session.up_winner_hold or session.dn_winner_hold:
                session.up_winner_hold = session.dn_winner_hold = False
            session.up_strong_since = session.dn_strong_since = None
            return

        def _one(side: Side, mid: Optional[float]) -> None:
            if mid is None:
                return
            if side == Side.UP:
                hold_attr, since_attr = "up_winner_hold", "up_strong_since"
            else:
                hold_attr, since_attr = "dn_winner_hold", "dn_strong_since"

            hold = getattr(session, hold_attr)
            since = getattr(session, since_attr)

            if hold:
                if mid < lo:
                    setattr(session, hold_attr, False)
                    setattr(session, since_attr, None)
                    cooldown_attr = "up_buy_cooldown_until" if side == Side.UP else "dn_buy_cooldown_until"
                    setattr(session, cooldown_attr, now + self.cfg.mm_winner_hold_buy_cooldown_seconds)
                    logger.info(
                        "[#%d] WINNER HOLD OFF %s (mid=%.4f < reset %.4f) – SELLs allowed again",
                        session.market_index, side.value, mid, lo,
                    )
                    sk = session.order_key(side, OrderSide.SELL)
                    if sk in session.active_orders:
                        self._executor.cancel_order(session.active_orders[sk].order_id)
                        del session.active_orders[sk]
                return

            # Not in hold: build streak only while mid ≥ hi
            if mid >= hi:
                if since is None:
                    setattr(session, since_attr, now)
                    logger.debug("[#%d] strong %s streak start @ mid=%.4f", session.market_index, side.value, mid)
                elif now - since >= confirm:
                    setattr(session, hold_attr, True)
                    setattr(session, since_attr, None)
                    logger.info(
                        "[#%d] WINNER HOLD ON %s (mid ≥ %.4f for %.0fs) – stop SELL quotes",
                        session.market_index, side.value, hi, confirm,
                    )
                    sk = session.order_key(side, OrderSide.SELL)
                    if sk in session.active_orders:
                        self._executor.cancel_order(session.active_orders[sk].order_id)
                        del session.active_orders[sk]
            else:
                if since is not None:
                    setattr(session, since_attr, None)

        _one(Side.UP, up_prices.mid)
        _one(Side.DOWN, dn_prices.mid)

    def _endgame_derisk(
        self,
        session: MarketSession,
        up_prices: Prices,
        dn_prices: Prices,
        up_tid: str,
        dn_tid: str,
        time_left: float,
    ) -> None:
        """Trim loser inventory in endgame when divergence is clear."""
        if time_left > self.cfg.mm_endgame_seconds:
            return

        up_mid = up_prices.mid
        dn_mid = dn_prices.mid
        if up_mid is None or dn_mid is None:
            return

        if abs(up_mid - dn_mid) < self.cfg.mm_endgame_divergence_threshold:
            return
        if (time.time() - session.last_endgame_derisk_at) < self.cfg.mm_endgame_derisk_interval:
            return

        if up_mid > dn_mid:
            loser_side, loser_mid, loser_tid, loser_inv = Side.DOWN, dn_mid, dn_tid, session.dn_inv
        else:
            loser_side, loser_mid, loser_tid, loser_inv = Side.UP, up_mid, up_tid, session.up_inv

        if loser_mid < self.cfg.mm_endgame_derisk_min_mid:
            return

        excess = loser_inv.shares - self.cfg.mm_endgame_loser_max_inventory
        if excess <= 0.01:
            return

        # Cancel pending BUY on loser before trimming.
        loser_buy_key = session.order_key(loser_side, OrderSide.BUY)
        loser_buy = session.active_orders.get(loser_buy_key)
        if loser_buy is not None:
            self._executor.cancel_order(loser_buy.order_id)
            del session.active_orders[loser_buy_key]

        trim_size = min(
            excess,
            max(
                self.cfg.mm_order_size,
                self.cfg.mm_order_size * self.cfg.mm_endgame_derisk_trim_mult,
            ),
            self.cfg.mm_skew_sell_cap,
        )
        floor = max(self.cfg.exit_sell_floor, min(loser_mid, 0.99))
        filled = self._executor.place_fak_sell(loser_tid, floor, trim_size, loser_side)
        if filled and filled > 0:
            session.last_endgame_derisk_at = time.time()
            loser_inv.on_sell_fill(floor, filled)
            logger.info(
                "[#%d] ENDGAME DERISK %s: sold %.2f @%.4f (excess %.2f -> target %.2f) │ total_realized=$%.4f",
                session.market_index,
                loser_side.value,
                filled,
                floor,
                excess,
                self.cfg.mm_endgame_loser_max_inventory,
                session.total_realized_pnl,
            )

    def _compute_skew_sell_shares(
        self,
        inv: SideInventory,
        target: float,
        time_left: float,
        weak_abs: bool,
        weak_rel: bool,
    ) -> float:
        """GTC sell size: larger clips when overweight vs target and/or mid is disfavored."""
        base = self.cfg.mm_order_size
        cap = self.cfg.mm_skew_sell_cap
        if cap <= 0:
            cap = max(base, inv.shares)

        mult = 1.0
        if inv.shares > target + self.cfg.mm_target_deadband:
            mult = max(mult, self.cfg.mm_overweight_sell_mult)
        if weak_abs or weak_rel:
            mult = max(mult, self.cfg.mm_disfavored_mid_sell_mult)
        if time_left <= self.cfg.mm_endgame_seconds:
            if inv.shares > target + self.cfg.mm_target_deadband or weak_abs or weak_rel:
                mult *= self.cfg.mm_endgame_sell_urgency_mult

        raw = base * mult
        sell_shares = min(inv.shares, cap, raw)
        # Always offer at least one normal clip when we have inventory (unless dust).
        sell_shares = max(sell_shares, min(inv.shares, base))
        if inv.shares < base:
            sell_shares = inv.shares
        return float(round(sell_shares, 4))

    def _inventory_price_skew(self, inv_shares: float, target: float) -> float:
        """Return signed price skew from inventory imbalance.

        Positive return means the side is overweight and should be sold down:
        - BUY quote is shifted down (less aggressive)
        - SELL quote is shifted down (more aggressive)
        """
        step = self.cfg.mm_inventory_skew_step
        cap = max(0.0, self.cfg.mm_inventory_skew_max)
        if step <= 0 or cap <= 0:
            return 0.0
        imbalance = inv_shares - target
        units = imbalance / max(self.cfg.mm_order_size, 1.0)
        raw = units * step
        return max(-cap, min(cap, raw))

    # ── Quoting ───────────────────────────────────────────────────────────────

    def _update_quotes(
        self,
        session: MarketSession,
        up_prices: Prices,
        dn_prices: Prices,
        up_tid: str,
        dn_tid: str,
        time_left: float,
    ) -> None:
        now = time.time()
        if now - session.last_requote_time < self.cfg.mm_requote_interval:
            return

        self._compute_inventory_targets(session, up_prices, dn_prices, time_left)

        inv_snap = self._inv_mgmt_snapshot(
            up_prices,
            dn_prices,
            session.up_inv.shares,
            session.dn_inv.shares,
        )
        unwind_gap = (
            self.cfg.mm_inv_stop_buy_gap
            if self.cfg.mm_inv_mgmt_enabled
            else self.cfg.mm_loser_unwind_gap
        )

        for side, prices, token_id, inv in [
            (Side.UP,   up_prices, up_tid, session.up_inv),
            (Side.DOWN, dn_prices, dn_tid, session.dn_inv),
        ]:
            mid = prices.mid
            if mid is None:
                continue
            if mid < self.cfg.mm_min_price or mid > self.cfg.mm_max_price:
                continue

            last_mid   = session.last_up_mid if side == Side.UP else session.last_dn_mid
            mid_moved  = last_mid is None or abs(mid - last_mid) >= self.cfg.mm_requote_threshold

            # ── BUY order ──────────────────────────────────────────────
            buy_key      = session.order_key(side, OrderSide.BUY)
            existing_buy = session.active_orders.get(buy_key)
            winner_hold  = self._winner_hold_active(session, side)
            buy_cooldown_until = session.up_buy_cooldown_until if side == Side.UP else session.dn_buy_cooldown_until
            in_buy_cooldown = now < buy_cooldown_until
            in_endgame = time_left <= self.cfg.mm_endgame_seconds
            if in_endgame:
                stop_buy = (
                    self.cfg.mm_endgame_stop_buy_below > 0
                    and mid < self.cfg.mm_endgame_stop_buy_below
                )
            else:
                stop_buy = (
                    self.cfg.mm_stop_buying_below > 0
                    and mid < self.cfg.mm_stop_buying_below
                )
            side_cap = self.cfg.mm_max_inventory
            inv_block_buy = False
            inv_disfavored = False
            if inv_snap is not None:
                if side == Side.UP:
                    side_cap = min(side_cap, inv_snap.up_cap)
                    inv_block_buy = inv_snap.block_up_buy
                    inv_disfavored = inv_snap.disfavored_up
                else:
                    side_cap = min(side_cap, inv_snap.dn_cap)
                    inv_block_buy = inv_snap.block_dn_buy
                    inv_disfavored = inv_snap.disfavored_dn

            other_mid = dn_prices.mid if side == Side.UP else up_prices.mid
            weak_abs = (not in_endgame) and (mid < self.cfg.mm_weak_side_threshold)
            weak_rel = (
                (not in_endgame)
                and (other_mid is not None)
                and (mid + self.cfg.mm_weak_side_relative_gap < other_mid)
            )
            # In endgame, if this side is clearly behind, freeze BUYs to avoid
            # re-accumulating loser inventory right after derisk trims.
            if (
                in_endgame
                and (other_mid is not None)
                and ((other_mid - mid) >= self.cfg.mm_endgame_loser_stop_buy_gap)
            ):
                stop_buy = True
            loser_endgame = (
                in_endgame
                and (other_mid is not None)
                and ((other_mid - mid) >= self.cfg.mm_endgame_loser_stop_buy_gap)
            )
            loser_pressure = inv_disfavored or (
                (other_mid is not None)
                and ((other_mid - mid) >= unwind_gap)
            )
            if weak_abs or weak_rel:
                side_cap = min(side_cap, self.cfg.mm_weak_side_max_inventory)

            if inv_block_buy:
                stop_buy = True

            target = session.up_target_inventory if side == Side.UP else session.dn_target_inventory
            skew_px = self._inventory_price_skew(inv.shares, target)
            buy_price = round(mid - self.cfg.mm_half_spread - skew_px, 4)
            sell_price = round(mid + self.cfg.mm_half_spread - skew_px, 4)
            buy_price = max(self.cfg.mm_min_price, min(self.cfg.mm_max_price, buy_price))
            sell_price = max(self.cfg.mm_min_price, min(self.cfg.mm_max_price, sell_price))
            buy_limit = min(side_cap, target + self.cfg.mm_target_deadband)
            buy_headroom = max(0.0, buy_limit - inv.shares)
            buy_clip = min(self.cfg.mm_order_size, buy_headroom)

            # When a side is held as "likely winner", we do not accumulate more
            # inventory on it (prevents ending with a big 50/50 at resolution).
            if (
                buy_clip > 0.01
                and (not stop_buy)
                and (not winner_hold)
                and (not in_buy_cooldown)
            ):
                if existing_buy is None:
                    order = self._executor.place_gtc_buy(token_id, buy_price, buy_clip, side)
                    if order:
                        session.active_orders[buy_key] = order
                        logger.debug(
                            "[#%d] QUOTE BUY %s @%.4f ×%.2f (headroom %.2f)",
                            session.market_index,
                            side.value,
                            buy_price,
                            buy_clip,
                            buy_headroom,
                        )
                elif mid_moved and abs(existing_buy.price - buy_price) >= self.cfg.mm_requote_threshold:
                    self._executor.cancel_order(existing_buy.order_id)
                    del session.active_orders[buy_key]
                    order = self._executor.place_gtc_buy(token_id, buy_price, buy_clip, side)
                    if order:
                        session.active_orders[buy_key] = order
            else:
                if existing_buy:
                    if winner_hold:
                        logger.info(
                            "[#%d] STOP BUY %s due to WINNER HOLD",
                            session.market_index, side.value,
                        )
                    elif in_buy_cooldown:
                        logger.info(
                            "[#%d] STOP BUY %s due to HOLD cooldown (%.0fs left)",
                            session.market_index,
                            side.value,
                            max(0.0, buy_cooldown_until - now),
                        )
                    elif inv_block_buy:
                        logger.info(
                            "[#%d] STOP BUY %s (inv mgmt: behind by %.3f, gap=%.3f)",
                            session.market_index,
                            side.value,
                            (other_mid - mid) if other_mid is not None else 0.0,
                            inv_snap.gap if inv_snap else 0.0,
                        )
                    elif stop_buy:
                        logger.info(
                            "[#%d] STOP BUY %s (mid=%.4f < %.4f%s)",
                            session.market_index,
                            side.value,
                            mid,
                            (self.cfg.mm_endgame_stop_buy_below if in_endgame else self.cfg.mm_stop_buying_below),
                            ", endgame" if in_endgame else "",
                        )
                    elif inv.shares >= side_cap and side_cap < self.cfg.mm_max_inventory:
                        logger.info(
                            "[#%d] STOP BUY %s due to weak-side cap %.2f (mid=%.4f)",
                            session.market_index,
                            side.value,
                            side_cap,
                            mid,
                        )
                    elif inv.shares >= buy_limit:
                        logger.info(
                            "[#%d] STOP BUY %s due to target cap %.2f (target %.2f, inv %.2f)",
                            session.market_index,
                            side.value,
                            buy_limit,
                            target,
                            inv.shares,
                        )
                    self._executor.cancel_order(existing_buy.order_id)
                    del session.active_orders[buy_key]

            # ── SELL order (only when we hold inventory) ───────────────
            sell_key      = session.order_key(side, OrderSide.SELL)
            existing_sell = session.active_orders.get(sell_key)
            winner_hold = self._winner_hold_active(session, side)

            if winner_hold and existing_sell:
                self._executor.cancel_order(existing_sell.order_id)
                del session.active_orders[sell_key]
                existing_sell = None

            # Normal mode: protect spread capture using avg_cost + spread.
            target_sell = round(max(inv.avg_cost + 2 * self.cfg.mm_half_spread, sell_price), 4)
            # Unwind mode: when this side is clearly behind, prioritize inventory reduction
            # over waiting for avg+spread fills that may never come.
            if loser_pressure:
                bid = prices.bid
                if inv_disfavored and bid is not None and bid > 0:
                    unwind_sell = round(max(bid, self.cfg.mm_min_price), 4)
                else:
                    unwind_sell = round(max(sell_price, mid + self.cfg.mm_loser_unwind_sell_edge), 4)
                target_sell = min(target_sell, unwind_sell)
            target_sell = min(target_sell, self.cfg.mm_max_price)
            sell_shares = self._compute_skew_sell_shares(
                inv, target, time_left, weak_abs, weak_rel
            )

            sell_only = self.cfg.mm_endgame_loser_only_unwind and loser_endgame
            if inv.shares > 0 and not winner_hold:
                if existing_sell is not None:
                    price_changed = abs(existing_sell.price - target_sell) >= self.cfg.mm_requote_threshold
                    size_changed = abs(existing_sell.size - sell_shares) > 0.01
                    if price_changed or size_changed:
                        self._executor.cancel_order(existing_sell.order_id)
                        del session.active_orders[sell_key]
                        existing_sell = None
                if existing_sell is None:
                    order = self._executor.place_gtc_sell(token_id, target_sell, sell_shares, side)
                    if order:
                        session.active_orders[sell_key] = order
                        logger.debug(
                            "[#%d] QUOTE SELL %s @%.4f ×%.2f (skew clip)",
                            session.market_index,
                            side.value,
                            target_sell,
                            sell_shares,
                        )
                # May have already removed BUY above (stop_buy / caps); use dict truth, not stale ref.
                if sell_only:
                    cur_buy = session.active_orders.get(buy_key)
                    if cur_buy is not None:
                        self._executor.cancel_order(cur_buy.order_id)
                        session.active_orders.pop(buy_key, None)
            elif inv.shares == 0 and existing_sell:
                self._executor.cancel_order(existing_sell.order_id)
                del session.active_orders[sell_key]

        if up_prices.mid:
            session.last_up_mid = up_prices.mid
        if dn_prices.mid:
            session.last_dn_mid = dn_prices.mid
        session.last_requote_time = now

    # ── Fill polling ──────────────────────────────────────────────────────────

    def _poll_fills(self, session: MarketSession) -> None:
        mkt = session.market_info
        up_tid = _token(mkt, Side.UP)
        dn_tid = _token(mkt, Side.DOWN)

        for key, order in list(session.active_orders.items()):
            filled, done = self._executor.check_order(order)
            if filled > order.size_filled + 0.0001:
                newly = filled - order.size_filled
                order.size_filled = filled
                token_id = up_tid if order.side == Side.UP else dn_tid
                self._on_fill(session, order, newly, token_id)
            if done:
                session.active_orders.pop(key, None)

    def _on_fill(self, session: MarketSession, order: ActiveOrder, shares: float, token_id: str) -> None:
        inv = session.up_inv if order.side == Side.UP else session.dn_inv

        if order.order_side == OrderSide.BUY:
            inv.on_buy_fill(order.price, shares)
            logger.info(
                "[#%d] FILL BUY  %s @%.4f ×%.2f │ inv=%.2f avg=%.4f",
                session.market_index, order.side.value, order.price, shares,
                inv.shares, inv.avg_cost,
            )
            # Immediately place sell at avg_cost + full round-trip spread
            sell_price = round(inv.avg_cost + 2 * self.cfg.mm_half_spread, 4)
            sell_price = min(sell_price, self.cfg.mm_max_price)
            sell_key   = session.order_key(order.side, OrderSide.SELL)
            if (
                sell_key not in session.active_orders
                and not session.quoting_stopped
                and not self._winner_hold_active(session, order.side)
            ):
                sell_order = self._executor.place_gtc_sell(token_id, sell_price, shares, order.side)
                if sell_order:
                    session.active_orders[sell_key] = sell_order

        else:  # SELL fill
            pnl_this = (order.price - inv.avg_cost) * shares
            inv.on_sell_fill(order.price, shares)
            logger.info(
                "[#%d] FILL SELL %s @%.4f ×%.2f │ pnl=+$%.4f │ total_realized=$%.4f",
                session.market_index, order.side.value, order.price, shares,
                pnl_this, session.total_realized_pnl,
            )

    # ── Cancel all orders ─────────────────────────────────────────────────────

    def _cancel_all_orders(self, session: MarketSession) -> None:
        for order in list(session.active_orders.values()):
            self._executor.cancel_order(order.order_id)
        session.active_orders.clear()
        logger.info("[#%d] All GTC orders cancelled", session.market_index)

    # ── Session close ─────────────────────────────────────────────────────────

    def _close_session(self, session: MarketSession) -> None:
        if session.session_closed:
            return
        session.session_closed = True

        self._cancel_all_orders(session)

        mkt    = session.market_info
        up_tid = _token(mkt, Side.UP)
        dn_tid = _token(mkt, Side.DOWN)

        up_prices = self._get_prices(up_tid)
        dn_prices = self._get_prices(dn_tid)

        up_mid = up_prices.mid or 0.5
        dn_mid = dn_prices.mid or 0.5

        # The side with LOWER bid is likely the loser – FAK-sell it
        if up_mid < dn_mid:
            loser_side, loser_tid, loser_inv = Side.UP,   up_tid, session.up_inv
            winner_inv                        = session.dn_inv
        else:
            loser_side, loser_tid, loser_inv = Side.DOWN, dn_tid, session.dn_inv
            winner_inv                        = session.up_inv

        loser_proceeds = 0.0
        if loser_inv.shares > 0.01:
            logger.info("[#%d] FAK-selling loser %s (%.2f shares)", session.market_index, loser_side.value, loser_inv.shares)
            filled = self._executor.place_fak_sell(loser_tid, self.cfg.exit_sell_floor, loser_inv.shares, loser_side)
            loser_proceeds = (filled or 0.0) * self.cfg.exit_sell_floor

        winner_resolution = winner_inv.shares * 1.0
        winner_cost       = winner_inv.shares * winner_inv.avg_cost
        loser_cost        = loser_inv.shares  * loser_inv.avg_cost

        session_pnl = (
            session.total_realized_pnl
            + winner_resolution - winner_cost
            - loser_cost + loser_proceeds
        )
        self._cum_pnl += session_pnl

        balance = self._executor.get_balance_usd()
        self._notifier.notify_session_close(session, session_pnl, self._cum_pnl, balance)

        logger.info(
            "[#%d] Session closed │ realized=$%.4f │ winner_inv=%.2f @avg=%.4f │ "
            "loser_inv=%.2f @avg=%.4f │ session_pnl=$%.4f │ cum_pnl=$%.4f",
            session.market_index,
            session.total_realized_pnl,
            winner_inv.shares, winner_inv.avg_cost,
            loser_inv.shares,  loser_inv.avg_cost,
            session_pnl, self._cum_pnl,
        )

    # ── Price helper ──────────────────────────────────────────────────────────

    def _get_prices(self, token_id: str) -> Prices:
        p = self._price_feed.get(token_id)
        if not p.stale:
            return p
        # REST fallback
        try:
            import requests
            r = requests.get(
                f"{self.cfg.clob_host}/book",
                params={"token_id": token_id},
                timeout=3,
            )
            data = r.json()
            bids = data.get("bids", [])
            asks = data.get("asks", [])
            bid = max((float(b["price"]) for b in bids if float(b.get("size", 0)) > 0), default=None)
            ask = min((float(a["price"]) for a in asks if float(a.get("size", 0)) > 0), default=None)
            return Prices(bid=bid, ask=ask, stale=False)
        except Exception:
            return Prices(stale=True)

    def _sleep_until(self, ts: float) -> None:
        while not self._stop and time.time() < ts - 0.1:
            time.sleep(min(1.0, ts - time.time()))
