"""Paired MM bot: matched UP+DN inventory, OSL blocks loser-side maker BUYs."""
from __future__ import annotations

import logging
import signal
import time
from typing import Optional

from bot2_btc.market_finder import MarketFinder
from bot2_btc.models import MarketInfo, Prices, Side
from bot2_btc.price_feed import PriceFeed
from bot6_mm.bot import _market_window_bounds, _token
from bot6_mm.executor import BaseExecutor, LiveExecutor, PaperExecutor
from bot6_mm.models import OrderSide
from bot6_mm.notifier import Notifier
from bot6_pairs.config import Bot6PairsConfig
from bot6_pairs.models import PairArmState, PairedSession

logger = logging.getLogger(__name__)


def _favored_side(up_mid: float, dn_mid: float, gap: float) -> Optional[Side]:
    if up_mid >= dn_mid + gap:
        return Side.UP
    if dn_mid >= up_mid + gap:
        return Side.DOWN
    return None


def _loser_side(up_mid: float, dn_mid: float, gap: float) -> Optional[Side]:
    fav = _favored_side(up_mid, dn_mid, gap)
    if fav == Side.UP:
        return Side.DOWN
    if fav == Side.DOWN:
        return Side.UP
    return None


class Bot6Pairs:
    def __init__(self, pairs_cfg: Bot6PairsConfig):
        self.pcfg = pairs_cfg
        self.cfg = pairs_cfg.base()
        self._executor: BaseExecutor = (
            PaperExecutor(self.cfg) if self.cfg.paper_mode else LiveExecutor(self.cfg)
        )
        self._notifier = Notifier(self.cfg)
        self._price_feed = PriceFeed(self.cfg.ws_url, stale_seconds=self.cfg.price_stale_seconds)
        self._market_finder = MarketFinder(self.cfg)
        self._session: Optional[PairedSession] = None
        self._market_index = 0
        self._stop = False
        self._cum_pnl = 0.0

    def run(self) -> None:
        logger.info(
            "Bot6Pairs v2 paper=%s bid_sum<=%.2f max_pair_cost=%.2f orphan=%.0fs OSL=%s",
            self.cfg.paper_mode,
            self.pcfg.bid_sum_trigger,
            self.pcfg.max_pair_cost,
            self.pcfg.orphan_timeout_s,
            self.pcfg.opposite_side_lock,
        )
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)
        balance = self._executor.get_balance_usd()
        self._notifier.notify_start(balance, self.cfg.paper_mode)
        self._price_feed.start()
        while not self._stop:
            try:
                self._find_and_run_market()
            except Exception as exc:
                logger.exception("Bot6Pairs loop error: %s", exc)
                time.sleep(5)
        self._price_feed.stop()
        self._notifier.notify_stop(self._executor.get_balance_usd(), self._cum_pnl)

    def _handle_signal(self, signum, _frame) -> None:
        logger.info("Bot6Pairs signal %s", signum)
        self._stop = True
        if self._session:
            self._cancel_arm(self._session)

    def _find_and_run_market(self) -> None:
        current = self._market_finder.find_current()
        if current is None:
            time.sleep(30)
            return
        self._market_index += 1
        self._run_market(current)

    def _run_market(self, mkt: MarketInfo) -> None:
        up_tid = _token(mkt, Side.UP)
        dn_tid = _token(mkt, Side.DOWN)
        session = PairedSession(market_info=mkt, market_index=self._market_index)
        self._session = session
        self._price_feed.subscribe([up_tid, dn_tid])
        market_start, market_end = _market_window_bounds(mkt, self.cfg.market_interval_seconds)
        now = time.time()
        if market_start > now + 2:
            self._sleep_until(market_start)

        while not self._stop:
            now = time.time()
            time_left = market_end - now
            if time_left <= 0:
                break
            up_p = self._get_prices(up_tid)
            dn_p = self._get_prices(dn_tid)
            if self.cfg.paper_mode and isinstance(self._executor, PaperExecutor):
                if up_p.bid and up_p.ask:
                    self._executor.simulate_fills(up_tid, up_p.bid, up_p.ask)
                if dn_p.bid and dn_p.ask:
                    self._executor.simulate_fills(dn_tid, dn_p.bid, dn_p.ask)

            self._poll_arm(session, up_tid, dn_tid)
            elapsed = now - market_start

            if time_left <= self.pcfg.stop_quoting_seconds:
                if not session.quoting_stopped:
                    self._cancel_arm(session)
                    session.quoting_stopped = True
            elif elapsed <= self.pcfg.trade_window_seconds:
                if not self.pcfg.acquire_pair_at_bid:
                    self._rebalance(session, up_p, dn_p, up_tid, dn_tid)
                if now - session.last_requote_at >= self.pcfg.requote_interval:
                    self._maybe_arm_pair(session, up_p, dn_p, up_tid, dn_tid)
                    session.last_requote_at = now

            time.sleep(self.cfg.tick_interval)

        self._close_session(session, up_p, dn_p, up_tid, dn_tid)

    def _buy_blocked(self, side: Side, up_mid: float, dn_mid: float) -> bool:
        if not self.pcfg.opposite_side_lock:
            return False
        los = _loser_side(up_mid, dn_mid, self.pcfg.osl_gap)
        return los == side

    def _pair_buy_prices(
        self, up_p: Prices, dn_p: Prices, bid_sum: float
    ) -> tuple[float, float]:
        ub, ua = up_p.bid or 0.0, up_p.ask or 0.0
        db, da = dn_p.bid or 0.0, dn_p.ask or 0.0
        cheap = bid_sum <= self.pcfg.bid_sum_trigger + 1e-9
        if cheap and self.pcfg.quote_at_bid_on_trigger:
            return max(0.01, ub), max(0.01, db)
        half = self.pcfg.aggressive_half_spread if cheap else self.pcfg.half_spread
        up_mid, dn_mid = up_p.mid or 0.5, dn_p.mid or 0.5
        return max(0.01, up_mid - half), max(0.01, dn_mid - half)

    def _side_is_loser(self, side: Side, up_mid: float, dn_mid: float) -> bool:
        return _loser_side(up_mid, dn_mid, self.pcfg.osl_gap) == side

    def _maybe_arm_pair(
        self,
        session: PairedSession,
        up_p: Prices,
        dn_p: Prices,
        up_tid: str,
        dn_tid: str,
    ) -> None:
        if session.arm is not None:
            self._resolve_orphan(session, up_p, dn_p, up_tid, dn_tid)
            return
        up_mid, dn_mid = up_p.mid, dn_p.mid
        if up_mid is None or dn_mid is None or up_p.bid is None or dn_p.bid is None:
            return
        if up_p.ask is None or dn_p.ask is None:
            return
        bid_sum = up_p.bid + dn_p.bid
        ask_sum = up_p.ask + dn_p.ask
        if bid_sum > self.pcfg.bid_sum_trigger + 1e-9:
            return
        if ask_sum > self.pcfg.max_ask_sum:
            return
        if time.time() - session.last_trigger_at < self.pcfg.trigger_cooldown_s:
            return
        if session.n_triggers >= self.pcfg.max_triggers_per_market:
            return
        if (
            self.pcfg.acquire_pair_at_bid
            and bid_sum <= self.pcfg.max_pair_cost + 1e-9
            and not self._buy_blocked(Side.UP, up_mid, dn_mid)
            and not self._buy_blocked(Side.DOWN, up_mid, dn_mid)
            and session.paired_shares() + self.pcfg.order_size <= self.pcfg.max_pairs
        ):
            self._executor.place_gtc_buy(up_tid, up_p.bid, self.pcfg.order_size, Side.UP)
            self._executor.place_gtc_buy(dn_tid, dn_p.bid, self.pcfg.order_size, Side.DOWN)
            session.pairs_opened += 1
            session.n_triggers += 1
            session.last_trigger_at = time.time()
            logger.info(
                "[#%d] pair @ bid_sum=%.3f (UP bid %.2f + DN bid %.2f)",
                session.market_index,
                bid_sum,
                up_p.bid,
                dn_p.bid,
            )
            return
        up_px, dn_px = self._pair_buy_prices(up_p, dn_p, bid_sum)
        if up_px + dn_px > self.pcfg.max_pair_cost:
            return
        if session.paired_shares() + self.pcfg.order_size > self.pcfg.max_pairs:
            return
        if self._buy_blocked(Side.UP, up_mid, dn_mid):
            up_px = 0.0
        if self._buy_blocked(Side.DOWN, up_mid, dn_mid):
            dn_px = 0.0
        if up_px <= 0 and dn_px <= 0:
            session.osl_blocks += 1
            return
        arm = PairArmState(started_at=time.time())
        if up_px > 0:
            arm.up_order = self._executor.place_gtc_buy(up_tid, up_px, self.pcfg.order_size, Side.UP)
        else:
            arm.up_filled = True
        if dn_px > 0:
            arm.dn_order = self._executor.place_gtc_buy(dn_tid, dn_px, self.pcfg.order_size, Side.DOWN)
        else:
            arm.dn_filled = True
        session.arm = arm
        session.n_triggers += 1
        session.last_trigger_at = time.time()
        logger.info(
            "[#%d] armed pair UP@%.2f DN@%.2f bid_sum=%.2f ask_sum=%.2f",
            session.market_index,
            up_px,
            dn_px,
            bid_sum,
            ask_sum,
        )

    def _poll_arm(self, session: PairedSession, up_tid: str, dn_tid: str) -> None:
        arm = session.arm
        if arm is None:
            return
        for label, order, side, inv in (
            ("UP", arm.up_order, Side.UP, session.up_inv),
            ("DN", arm.dn_order, Side.DOWN, session.dn_inv),
        ):
            if order is None:
                continue
            filled, done = self._executor.check_order(order)
            delta = filled - order.size_filled
            if delta > 0:
                inv.on_buy_fill(order.price, delta)
                order.size_filled = filled
                if side == Side.UP:
                    arm.up_filled = True
                else:
                    arm.dn_filled = True
                logger.info("[#%d] %s pair-leg fill %.2f×%.2f", session.market_index, label, order.price, delta)
            if done and order.size_filled < order.size - 1e-6:
                self._executor.cancel_order(order.order_id)

    def _resolve_orphan(
        self,
        session: PairedSession,
        up_p: Prices,
        dn_p: Prices,
        up_tid: str,
        dn_tid: str,
    ) -> None:
        arm = session.arm
        if arm is None:
            return
        if arm.up_filled and arm.dn_filled:
            session.pairs_opened += 1
            logger.info(
                "[#%d] pair complete #%d inv=%.0f/%.0f",
                session.market_index,
                session.pairs_opened,
                session.up_inv.shares,
                session.dn_inv.shares,
            )
            session.arm = None
            session.last_trigger_at = time.time()
            return
        elapsed = time.time() - arm.started_at
        up_mid, dn_mid = up_p.mid or 0.5, dn_p.mid or 0.5
        if not arm.up_filled and not arm.dn_filled and elapsed >= self.pcfg.order_timeout_s:
            self._cancel_arm(session)
            session.last_trigger_at = time.time()
            return
        if elapsed < self.pcfg.orphan_timeout_s:
            return
        if arm.up_filled and not arm.dn_filled and session.up_inv.shares > 0:
            if self._side_is_loser(Side.UP, up_mid, dn_mid) or not self.pcfg.hold_winner_orphan:
                if self.pcfg.orphan_sell_back:
                    bid = up_p.bid or self.pcfg.exit_sell_floor
                    sh = min(self.pcfg.order_size, session.up_inv.shares)
                    self._executor.place_fak_sell(up_tid, max(bid, self.pcfg.exit_sell_floor), sh, Side.UP)
                    session.up_inv.shares = max(0.0, session.up_inv.shares - sh)
                    session.orphan_sellbacks += 1
                    logger.warning("[#%d] orphan sellback UP %.2f sh", session.market_index, sh)
        elif arm.dn_filled and not arm.up_filled and session.dn_inv.shares > 0:
            if self._side_is_loser(Side.DOWN, up_mid, dn_mid) or not self.pcfg.hold_winner_orphan:
                if self.pcfg.orphan_sell_back:
                    bid = dn_p.bid or self.pcfg.exit_sell_floor
                    sh = min(self.pcfg.order_size, session.dn_inv.shares)
                    self._executor.place_fak_sell(dn_tid, max(bid, self.pcfg.exit_sell_floor), sh, Side.DOWN)
                    session.dn_inv.shares = max(0.0, session.dn_inv.shares - sh)
                    session.orphan_sellbacks += 1
                    logger.warning("[#%d] orphan sellback DN %.2f sh", session.market_index, sh)
        self._cancel_arm(session)
        session.last_trigger_at = time.time()

    def _rebalance(
        self,
        session: PairedSession,
        up_p: Prices,
        dn_p: Prices,
        up_tid: str,
        dn_tid: str,
    ) -> None:
        up_mid, dn_mid = up_p.mid, dn_p.mid
        if up_mid is None or dn_mid is None:
            return
        diff = session.up_inv.shares - session.dn_inv.shares
        if abs(diff) < 0.5:
            return
        if diff > 0:
            if self._buy_blocked(Side.DOWN, up_mid, dn_mid):
                session.osl_blocks += 1
                bid = up_p.bid or self.pcfg.exit_sell_floor
                sh = min(diff, self.pcfg.order_size)
                self._executor.place_fak_sell(up_tid, max(bid, self.pcfg.exit_sell_floor), sh, Side.UP)
                session.up_inv.shares = max(0.0, session.up_inv.shares - sh)
                session.rebalances += 1
                return
            bid_sum = (up_p.bid or 0) + (dn_p.bid or 0)
            dn_px, _ = self._pair_buy_prices(up_p, dn_p, bid_sum)
            if dn_p.ask is not None and dn_p.ask <= dn_px + 1e-6:
                o = self._executor.place_gtc_buy(dn_tid, dn_px, min(diff, self.pcfg.order_size), Side.DOWN)
                if o:
                    session.rebalances += 1
        else:
            need = -diff
            if self._buy_blocked(Side.UP, up_mid, dn_mid):
                session.osl_blocks += 1
                bid = dn_p.bid or self.pcfg.exit_sell_floor
                sh = min(need, self.pcfg.order_size)
                self._executor.place_fak_sell(dn_tid, max(bid, self.pcfg.exit_sell_floor), sh, Side.DOWN)
                session.dn_inv.shares = max(0.0, session.dn_inv.shares - sh)
                session.rebalances += 1
                return
            bid_sum = (up_p.bid or 0) + (dn_p.bid or 0)
            up_px, _ = self._pair_buy_prices(up_p, dn_p, bid_sum)
            if up_p.ask is not None and up_p.ask <= up_px + 1e-6:
                self._executor.place_gtc_buy(up_tid, up_px, min(need, self.pcfg.order_size), Side.UP)
                session.rebalances += 1

    def _cancel_arm(self, session: PairedSession) -> None:
        arm = session.arm
        if not arm:
            return
        for o in (arm.up_order, arm.dn_order):
            if o:
                self._executor.cancel_order(o.order_id)
        session.arm = None

    def _close_session(
        self,
        session: PairedSession,
        up_p: Prices,
        dn_p: Prices,
        up_tid: str,
        dn_tid: str,
    ) -> None:
        self._cancel_arm(session)
        up_mid, dn_mid = up_p.mid or 0.0, dn_p.mid or 0.0
        los = _loser_side(up_mid, dn_mid, self.pcfg.osl_gap)
        diff = session.up_inv.shares - session.dn_inv.shares
        if diff > 0.5 and los == Side.UP:
            bid = up_p.bid or self.pcfg.exit_sell_floor
            self._executor.place_fak_sell(up_tid, max(bid, self.pcfg.exit_sell_floor), diff, Side.UP)
            session.up_inv.shares = max(0.0, session.up_inv.shares - diff)
        elif diff < -0.5 and los == Side.DOWN:
            bid = dn_p.bid or self.pcfg.exit_sell_floor
            self._executor.place_fak_sell(dn_tid, max(bid, self.pcfg.exit_sell_floor), -diff, Side.DOWN)
            session.dn_inv.shares = max(0.0, session.dn_inv.shares + diff)  # diff<0 → trim DN excess
        session.session_closed = True
        logger.info(
            "[#%d] closed pairs=%d paired=%.0f up=%.0f dn=%.0f OSL=%d orphan=%d reb=%d",
            session.market_index,
            session.pairs_opened,
            session.paired_shares(),
            session.up_inv.shares,
            session.dn_inv.shares,
            session.osl_blocks,
            session.orphan_sellbacks,
            session.rebalances,
        )

    def _get_prices(self, token_id: str) -> Prices:
        p = self._price_feed.get_prices(token_id)
        if p and p.bid and p.ask:
            return p
        return self._price_feed.get_prices_rest(token_id)

    def _sleep_until(self, ts: float) -> None:
        while not self._stop and time.time() < ts:
            time.sleep(0.5)
