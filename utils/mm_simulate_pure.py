"""
PURE 2-sided market making simulator.

Strategy:
  For EACH SIDE (UP, DOWN), continuously maintain:
    - Maker BUY  at BBO_bid (or BBO_bid - offset)
    - Maker SELL at BBO_ask (or BBO_ask + offset)  — only if we have inventory > 0

  When BUY fills (BBO_bid drops past our level):
    - Add 1 share to inventory
    - Place new SELL at BBO_ask + sell_offset

  When SELL fills (BBO_ask rises past our level):
    - Remove 1 share from inventory
    - Captured spread on this round trip

Cap inventory per side at max_inventory.
At end of market: hold residual to expiry (winner pays $1).

This is the classic "scalping the spread" strategy.
"""
from __future__ import annotations
import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from utils.mm_analyze_5m import parse_market_csv, PRICES_5M_ROOT


# ── Maker rebate (Polymarket-style: fee ∝ p(1−p), rebate = share of fee) ───────
#
# Per maker fill at price p for s shares:
#   fee_equiv  = fee_rate * p * (1 - p) * s
#   rebate     = rebate_pct * fee_equiv
#
# Keep trading P&L "pure" (cash + payoff − taker fees); add rebate_base a posteriori.


def maker_rebate_factor(fee_rate: float, rebate_pct: float) -> float:
    return rebate_pct * fee_rate


def maker_rebate_per_fill(price: float, shares: float, fee_rate: float, rebate_pct: float) -> float:
    """Rebate $ for one maker fill at ``price`` for ``shares`` shares."""
    p = float(price)
    s = float(shares)
    return maker_rebate_factor(fee_rate, rebate_pct) * p * (1.0 - p) * s


def maker_rebate_base_per_fill(price: float, shares: float) -> float:
    """Price-weighted base before ``fee_rate`` and ``rebate_pct`` (sum across fills)."""
    p = float(price)
    s = float(shares)
    return p * (1.0 - p) * s


def maker_rebate_from_base(rebate_base: float, fee_rate: float, rebate_pct: float) -> float:
    return maker_rebate_factor(fee_rate, rebate_pct) * float(rebate_base)


def estimate_rebate_base_from_averages(
    up_buys: int,
    up_sells: int,
    dn_buys: int,
    dn_sells: int,
    avg_up_buy: float,
    avg_up_sell: float,
    avg_dn_buy: float,
    avg_dn_sell: float,
    fill_size: float,
) -> float:
    """Approximate ``rebate_base`` when only per-side counts and average prices exist."""
    base = 0.0
    if up_buys > 0 and avg_up_buy > 0:
        base += up_buys * maker_rebate_base_per_fill(avg_up_buy, fill_size)
    if up_sells > 0 and avg_up_sell > 0:
        base += up_sells * maker_rebate_base_per_fill(avg_up_sell, fill_size)
    if dn_buys > 0 and avg_dn_buy > 0:
        base += dn_buys * maker_rebate_base_per_fill(avg_dn_buy, fill_size)
    if dn_sells > 0 and avg_dn_sell > 0:
        base += dn_sells * maker_rebate_base_per_fill(avg_dn_sell, fill_size)
    return base


def apply_rebates_to_df(
    df: pd.DataFrame,
    fee_rate: float = 0.07,
    rebate_pct: float = 0.20,
    fill_size: float = 1.0,
) -> pd.DataFrame:
    """Add ``rebate_total``, ``pnl_pure``, ``pnl`` from stored or estimated rebate base."""
    out = df.copy()
    factor = maker_rebate_factor(fee_rate, rebate_pct)
    if "rebate_base" in out.columns:
        base = out["rebate_base"]
    else:
        base = out.apply(
            lambda r: estimate_rebate_base_from_averages(
                int(r.get("up_buys", 0)),
                int(r.get("up_sells", 0)),
                int(r.get("dn_buys", 0)),
                int(r.get("dn_sells", 0)),
                float(r.get("avg_up_buy", 0)),
                float(r.get("avg_up_sell", 0)),
                float(r.get("avg_dn_buy", 0)),
                float(r.get("avg_dn_sell", 0)),
                fill_size,
            ),
            axis=1,
        )
    out["rebate_total"] = factor * base
    if "cash" in out.columns and "payoff" in out.columns and "fees_total" in out.columns:
        out["pnl_pure"] = out["cash"] + out["payoff"] - out["fees_total"]
    elif "pnl" in out.columns and "rebate_total" in out.columns:
        # Legacy rows: pnl already included old rebate — prefer explicit columns if present
        out["pnl_pure"] = out["pnl"] - out["rebate_total"]
    else:
        out["pnl_pure"] = out.get("pnl", 0.0)
    out["pnl"] = out["pnl_pure"] + out["rebate_total"]
    if "maker_fills" not in out.columns:
        out["maker_fills"] = (
            out.get("up_buys", 0) + out.get("up_sells", 0)
            + out.get("dn_buys", 0) + out.get("dn_sells", 0)
        )
    return out


@dataclass(frozen=True)
class PureMMPolicy:
    tick: float = 0.01
    fill_size: float = 1.0

    # Bid placement: BBO_bid - bid_offset_ticks * tick
    bid_offset_ticks: int = 1
    # Ask placement: BBO_ask + ask_offset_ticks * tick (typically 1 = 1c above ask)
    # OR sell at buy_price + min_spread_ticks * tick (require minimum spread profit)
    ask_offset_ticks: int = 1
    min_spread_ticks: int = 2  # require sell price >= avg_buy + this many ticks

    max_inventory_per_side: float = 5.0
    trade_window_seconds: float = 300.0

    # Initial pre-roll: don't trade for first observe_seconds (let market stabilize)
    observe_seconds: float = 0.0

    # End-of-market cleanup
    close_at_seconds: float = 0.0  # 0 = hold to expiry
    close_mode: str = "hold"        # "hold", "taker_close" (sell residual at bid)

    # Trend filter: stop buying on a side if its BBO_bid dropped > trend_drop_thr
    # over last trend_lookback_s seconds.
    trend_lookback_s: float = 10.0
    trend_drop_thr: float = 0.02   # 0 = disabled
    # When long inventory and trend is bad, FORCE TAKER SELL (stop-loss on inv)
    inv_stop_loss: bool = False
    inv_stop_loss_drop_thr: float = 0.04  # if BID dropped this much vs avg_buy, taker sell residual
    # Even more aggressive: when trend is bad, immediately taker-flatten the inventory
    trend_taker_flatten: bool = False

    # ============== CONTINUOUS INVENTORY MGMT (full operativity) ==============
    # Every tick: skew caps / spreads / unwind from UP_mid vs DN_mid. No observe, no skip.
    inv_mgmt_enabled: bool = False
    inv_neutral_gap: float = 0.02          # |up_mid - dn_mid| below → symmetric caps
    inv_neutral_max_cap: float = 0.0       # if >0: cap BOTH sides to this when neutral (limits 50/50 buildup)
    inv_stop_buy_gap: float = 0.04         # disfavored: no new BUYs when behind by this
    inv_loser_cap_slope: float = 80.0      # loser cap = max(0, max_inv - slope * excess_gap)
    inv_taker_trim_gap: float = 0.07       # taker-sell all disfavored inv at bid
    inv_winner_min_spread_ticks: int = 2
    inv_loser_min_spread_ticks: int = 0    # 0 = sell at bid (unwind), not avg+spread
    inv_aggressive_unwind: bool = True     # disfavored SELL targets bid, not avg+spread

    # ============== ASYMMETRIC MM (momentum-biased) ==============
    # If asymm_enabled: after momentum_observe_s seconds, identify "suspected winner"
    # via mid-price trend. Apply different caps & spreads to W vs L side.
    asymm_enabled: bool = False
    momentum_observe_s: float = 30.0
    momentum_window_s: float = 20.0     # rolling window for mid-price trend
    momentum_threshold: float = 0.005   # min mid difference to declare a winner
    asymm_cap_winner: float = 10.0
    asymm_cap_loser: float = 1.0
    asymm_min_spread_w_ticks: int = 5   # higher = sell less, accumulate more on winner
    asymm_min_spread_l_ticks: int = 1   # standard for loser
    # Re-evaluate momentum decision every N seconds (0 = never, fixed at first decision)
    asymm_reevaluate_s: float = 0.0
    # If True: when signal is uncertain (None), DON'T trade at all. If False: do symmetric MM.
    asymm_skip_uncertain: bool = False
    # Signal type: "momentum" (default) or "mid_diff" (absolute UP_mid - DN_mid at decision)
    asymm_signal_type: str = "momentum"
    # Spread gate: only apply asymm if observed avg_spread <= asymm_max_spread (= TIGHT book)
    asymm_max_spread: float = 1.0   # 1.0 = no filter; lower (e.g., 0.011) = only tight books
    # Strength gate: only apply asymm if signal strength >= this (mid distance from 0.5)
    asymm_min_strength: float = 0.0  # 0 = no filter; higher = only strong signals
    # When asymm commits, taker-liquidate any inventory on suspected loser (clears observation residual)
    asymm_liquidate_loser_on_commit: bool = True
    # During the observation phase, DON'T trade at all (= cap=0 both sides)
    asymm_observe_no_trade: bool = True

    fee_rate: float = 0.07
    rebate_pct: float = 0.20


@dataclass
class PureMMResult:
    market_name: str
    date: str
    up_buys: int
    up_sells: int
    dn_buys: int
    dn_sells: int
    up_round_trips: int
    dn_round_trips: int
    avg_up_buy: float
    avg_up_sell: float
    avg_dn_buy: float
    avg_dn_sell: float
    avg_up_spread: float
    avg_dn_spread: float
    final_up_inv: float
    final_dn_inv: float
    cash: float
    rebate_base: float       # Σ p(1−p)×shares over maker fills (rebate applied a posteriori)
    rebate_total: float      # rebate_pct × fee_rate × rebate_base
    fees_total: float
    payoff: float
    pnl_pure: float          # cash + payoff − fees (no rebate)
    pnl: float               # pnl_pure + rebate_total
    maker_fills: int         # maker buy + sell count
    winner: str
    suspected_winner: str = "NONE"   # asymm only: what we predicted
    final_loser_inv: float = 0.0     # inv on side that lost at expiry
    final_winner_inv: float = 0.0


def simulate_pure_mm(
    wide: pd.DataFrame, cfg: PureMMPolicy, market_name: str, date: str
) -> Optional[PureMMResult]:
    if wide is None or len(wide) < 30:
        return None
    wide = wide.copy()
    for col in ("up_bid", "up_ask", "dn_bid", "dn_ask"):
        wide[col] = wide[col].ffill()
    wide = wide.dropna(subset=["up_bid", "up_ask", "dn_bid", "dn_ask"]).reset_index(drop=True)
    if len(wide) < 30:
        return None

    up_bid = wide["up_bid"].to_numpy(dtype=np.float64)
    up_ask = wide["up_ask"].to_numpy(dtype=np.float64)
    dn_bid = wide["dn_bid"].to_numpy(dtype=np.float64)
    dn_ask = wide["dn_ask"].to_numpy(dtype=np.float64)
    t_ms = wide["t_ms"].to_numpy(dtype=np.int64)
    n = len(wide)
    t_start = int(t_ms[0])
    EPS = 1e-9
    tick = cfg.tick
    bid_offset = cfg.bid_offset_ticks * tick
    ask_offset = cfg.ask_offset_ticks * tick
    min_spread = cfg.min_spread_ticks * tick
    obs_ms = int(cfg.observe_seconds * 1000)
    win_ms = int(cfg.trade_window_seconds * 1000)
    close_ms = int(cfg.close_at_seconds * 1000) if cfg.close_at_seconds > 0 else None

    final_up_mid = (up_bid[-1] + up_ask[-1]) * 0.5
    final_dn_mid = (dn_bid[-1] + dn_ask[-1]) * 0.5
    if final_up_mid > final_dn_mid:
        winner = "UP"
    elif final_dn_mid > final_up_mid:
        winner = "DOWN"
    else:
        winner = "TIE"

    # State
    cash = 0.0
    rebate_base = 0.0
    fees_total = 0.0

    up_inv = 0.0
    dn_inv = 0.0
    up_avg_buy = 0.0  # running avg buy price
    dn_avg_buy = 0.0

    up_buys = 0
    up_sells = 0
    dn_buys = 0
    dn_sells = 0
    up_round_trips = 0
    dn_round_trips = 0
    sum_up_spread = 0.0
    sum_dn_spread = 0.0
    sum_up_buy = 0.0
    sum_up_sell = 0.0
    sum_dn_buy = 0.0
    sum_dn_sell = 0.0

    prev_ub = up_bid[0]
    prev_ua = up_ask[0]
    prev_db = dn_bid[0]
    prev_da = dn_ask[0]

    # Trend tracker: rolling lookback of BBO_bid for each side
    lookback_ms = int(cfg.trend_lookback_s * 1000)
    # We'll do a simple linear scan back; it's fast enough.

    def trend_ok_to_buy(idx: int, side: str) -> bool:
        """Return True if BBO_bid hasn't dropped more than trend_drop_thr in lookback_s."""
        if cfg.trend_drop_thr <= 0:
            return True
        cur_bid = up_bid[idx] if side == "UP" else dn_bid[idx]
        target_ms = t_ms[idx] - lookback_ms
        # binary search back
        j = idx
        while j > 0 and t_ms[j] >= target_ms:
            j -= 1
        past_bid = up_bid[j] if side == "UP" else dn_bid[j]
        if past_bid <= 0:
            return True
        return (past_bid - cur_bid) <= cfg.trend_drop_thr + EPS

    def stop_loss_check(idx: int, side: str, avg_buy: float) -> bool:
        """Return True if we should taker-sell residual inventory due to bid collapse."""
        if not cfg.inv_stop_loss:
            return False
        if avg_buy <= 0:
            return False
        cur_bid = up_bid[idx] if side == "UP" else dn_bid[idx]
        return (avg_buy - cur_bid) >= cfg.inv_stop_loss_drop_thr - EPS

    def inv_mgmt_state(idx: int) -> tuple[float, float, float, float, bool, bool, float]:
        """Caps, min-spreads, buy-block flags, mid gap (UP_mid - DN_mid)."""
        max_i = cfg.max_inventory_per_side
        up_ms = min_spread
        dn_ms = min_spread
        block_up_buy = False
        block_dn_buy = False
        up_mid = (up_bid[idx] + up_ask[idx]) * 0.5
        dn_mid = (dn_bid[idx] + dn_ask[idx]) * 0.5
        gap = up_mid - dn_mid

        if not cfg.inv_mgmt_enabled:
            return max_i, max_i, up_ms, dn_ms, block_up_buy, block_dn_buy, gap

        ng = cfg.inv_neutral_gap
        slope = cfg.inv_loser_cap_slope
        if gap > ng:
            excess = gap - ng
            up_cap = max_i
            dn_cap = max(0.0, max_i - slope * excess)
            up_ms = cfg.inv_winner_min_spread_ticks * tick
            dn_ms = cfg.inv_loser_min_spread_ticks * tick
            if gap >= cfg.inv_stop_buy_gap:
                block_dn_buy = True
                dn_cap = min(dn_cap, dn_inv)
        elif gap < -ng:
            excess = -gap - ng
            dn_cap = max_i
            up_cap = max(0.0, max_i - slope * excess)
            up_ms = cfg.inv_loser_min_spread_ticks * tick
            dn_ms = cfg.inv_winner_min_spread_ticks * tick
            if (-gap) >= cfg.inv_stop_buy_gap:
                block_up_buy = True
                up_cap = min(up_cap, up_inv)
        else:
            neutral_cap = cfg.inv_neutral_max_cap if cfg.inv_neutral_max_cap > 0 else max_i
            up_cap = neutral_cap
            dn_cap = neutral_cap
        return up_cap, dn_cap, up_ms, dn_ms, block_up_buy, block_dn_buy, gap

    def taker_trim_disfavored(idx: int, gap: float) -> None:
        nonlocal cash, fees_total, up_inv, dn_inv, up_avg_buy, dn_avg_buy
        if not cfg.inv_mgmt_enabled or gap < cfg.inv_taker_trim_gap - EPS:
            return
        if gap > 0 and dn_inv > 0:
            db = dn_bid[idx]
            if db > 0:
                rev = db * dn_inv
                fee = cfg.fee_rate * db * (1 - db) * dn_inv
                cash += rev
                fees_total += fee
                dn_inv = 0.0
                dn_avg_buy = 0.0
        elif gap < 0 and up_inv > 0:
            ub = up_bid[idx]
            if ub > 0:
                rev = ub * up_inv
                fee = cfg.fee_rate * ub * (1 - ub) * up_inv
                cash += rev
                fees_total += fee
                up_inv = 0.0
                up_avg_buy = 0.0

    def sell_target(
        side: str,
        idx: int,
        avg_buy: float,
        min_spread_side: float,
        disfavored: bool,
        prev_ask: float,
    ) -> float:
        ask_off = cfg.ask_offset_ticks * tick
        target = max(prev_ask + ask_off, avg_buy + min_spread_side)
        if cfg.inv_mgmt_enabled and disfavored and cfg.inv_aggressive_unwind:
            bid = up_bid[idx] if side == "UP" else dn_bid[idx]
            if bid > 0:
                target = min(target, max(bid, tick))
        return target

    # Asymmetric MM state
    asymm_observe_ms = int(cfg.momentum_observe_s * 1000)
    asymm_window_ms = int(cfg.momentum_window_s * 1000)
    asymm_reeval_ms = int(cfg.asymm_reevaluate_s * 1000) if cfg.asymm_reevaluate_s > 0 else 0
    last_asymm_eval_ms = -1
    suspected_winner = None  # None | "UP" | "DOWN"

    def evaluate_winner(idx: int) -> Optional[str]:
        """Return 'UP' or 'DOWN' based on signal, OR None if uncertain or regime filter fails."""
        # Regime filter: only consider asymm if observed avg_spread is TIGHT
        if cfg.asymm_max_spread < 1.0:
            up_sp = (up_ask[:idx+1] - up_bid[:idx+1]).mean()
            dn_sp = (dn_ask[:idx+1] - dn_bid[:idx+1]).mean()
            avg_sp_so_far = (up_sp + dn_sp) / 2.0
            if avg_sp_so_far > cfg.asymm_max_spread + EPS:
                return None  # thin book = noise regime → don't trust signal

        up_mid_now = (up_bid[idx] + up_ask[idx]) * 0.5
        dn_mid_now = (dn_bid[idx] + dn_ask[idx]) * 0.5

        # Strength filter: only commit if mid is far from 0.5
        if cfg.asymm_min_strength > 0:
            strength = max(abs(up_mid_now - 0.5), abs(dn_mid_now - 0.5))
            if strength < cfg.asymm_min_strength - EPS:
                return None

        if cfg.asymm_signal_type == "mid_diff":
            # Simple: which mid is higher = winning candidate
            if up_mid_now - dn_mid_now > cfg.momentum_threshold:
                return "UP"
            if dn_mid_now - up_mid_now > cfg.momentum_threshold:
                return "DOWN"
            return None
        else:
            # momentum signal: rate of change over window
            target_ms = t_ms[idx] - asymm_window_ms
            j = idx
            while j > 0 and t_ms[j] >= target_ms:
                j -= 1
            up_mid_then = (up_bid[j] + up_ask[j]) * 0.5
            dn_mid_then = (dn_bid[j] + dn_ask[j]) * 0.5
            up_change = up_mid_now - up_mid_then
            dn_change = dn_mid_now - dn_mid_then
            if up_change - dn_change > cfg.momentum_threshold:
                return "UP"
            if dn_change - up_change > cfg.momentum_threshold:
                return "DOWN"
            return None

    closed = False
    for i in range(1, n):
        elapsed = t_ms[i] - t_start
        if elapsed > win_ms:
            break
        if elapsed < obs_ms:
            prev_ub = up_bid[i]
            prev_ua = up_ask[i]
            prev_db = dn_bid[i]
            prev_da = dn_ask[i]
            continue

        # End-of-market close
        if close_ms is not None and not closed and elapsed >= close_ms:
            if cfg.close_mode == "taker_close":
                ub_now = up_bid[i]; db_now = dn_bid[i]
                if up_inv > 0 and ub_now > 0:
                    rev = ub_now * up_inv
                    fee = cfg.fee_rate * ub_now * (1 - ub_now) * up_inv
                    cash += rev; fees_total += fee; up_inv = 0.0
                if dn_inv > 0 and db_now > 0:
                    rev = db_now * dn_inv
                    fee = cfg.fee_rate * db_now * (1 - db_now) * dn_inv
                    cash += rev; fees_total += fee; dn_inv = 0.0
            closed = True
            break

        ub = up_bid[i]; ua = up_ask[i]
        db = dn_bid[i]; da = dn_ask[i]

        up_cap, dn_cap, up_minspread, dn_minspread, block_up_buy, block_dn_buy, mid_gap = (
            inv_mgmt_state(i)
        )
        disfavored_up = mid_gap < -cfg.inv_neutral_gap if cfg.inv_mgmt_enabled else False
        disfavored_dn = mid_gap > cfg.inv_neutral_gap if cfg.inv_mgmt_enabled else False
        taker_trim_disfavored(i, mid_gap)

        # Asymmetric MM: evaluate winner candidate (skipped when inv_mgmt drives caps)
        if cfg.inv_mgmt_enabled:
            pass  # caps already set from mids
        elif cfg.asymm_enabled and elapsed >= asymm_observe_ms:
            if suspected_winner is None or (asymm_reeval_ms > 0 and t_ms[i] - last_asymm_eval_ms >= asymm_reeval_ms):
                suspected_winner = evaluate_winner(i)
                last_asymm_eval_ms = t_ms[i]

        # Compute side-specific caps and min_spreads (asymm path; inv_mgmt already set)
        if cfg.inv_mgmt_enabled:
            pass
        elif cfg.asymm_enabled and suspected_winner is not None:
            if suspected_winner == "UP":
                up_cap = cfg.asymm_cap_winner
                dn_cap = cfg.asymm_cap_loser
                up_minspread = cfg.asymm_min_spread_w_ticks * tick
                dn_minspread = cfg.asymm_min_spread_l_ticks * tick
            else:  # DOWN
                up_cap = cfg.asymm_cap_loser
                dn_cap = cfg.asymm_cap_winner
                up_minspread = cfg.asymm_min_spread_l_ticks * tick
                dn_minspread = cfg.asymm_min_spread_w_ticks * tick
            # On first commit: LIQUIDATE existing inventory on suspected loser side via taker
            # to clear inventory accumulated during the observation phase.
            if cfg.asymm_liquidate_loser_on_commit:
                if suspected_winner == "UP" and dn_inv > 0 and db > 0:
                    rev = db * dn_inv
                    fee = cfg.fee_rate * db * (1 - db) * dn_inv
                    cash += rev; fees_total += fee; dn_inv = 0.0; dn_avg_buy = 0.0
                elif suspected_winner == "DOWN" and up_inv > 0 and ub > 0:
                    rev = ub * up_inv
                    fee = cfg.fee_rate * ub * (1 - ub) * up_inv
                    cash += rev; fees_total += fee; up_inv = 0.0; up_avg_buy = 0.0
        elif cfg.asymm_enabled and cfg.asymm_skip_uncertain and elapsed >= asymm_observe_ms:
            # Asymm enabled, signal evaluated, no commit: skip trading entirely
            up_cap = 0.0
            dn_cap = 0.0
            up_minspread = min_spread
            dn_minspread = min_spread
        elif cfg.asymm_enabled and cfg.asymm_observe_no_trade and elapsed < asymm_observe_ms:
            # During observation phase with no-trade option: don't accumulate
            up_cap = 0.0
            dn_cap = 0.0
            up_minspread = min_spread
            dn_minspread = min_spread
        else:
            up_cap = cfg.max_inventory_per_side
            dn_cap = cfg.max_inventory_per_side
            up_minspread = min_spread
            dn_minspread = min_spread

        # Inventory stop-loss check (taker close if BID collapsed too far)
        if up_inv > 0 and stop_loss_check(i, "UP", up_avg_buy):
            if ub > 0:
                rev = ub * up_inv
                fee = cfg.fee_rate * ub * (1 - ub) * up_inv
                cash += rev; fees_total += fee; up_inv = 0.0; up_avg_buy = 0.0
        if dn_inv > 0 and stop_loss_check(i, "DOWN", dn_avg_buy):
            if db > 0:
                rev = db * dn_inv
                fee = cfg.fee_rate * db * (1 - db) * dn_inv
                cash += rev; fees_total += fee; dn_inv = 0.0; dn_avg_buy = 0.0

        # Trend-triggered taker sell: if trend bad AND have inventory, flatten now
        if cfg.trend_taker_flatten:
            if up_inv > 0 and not trend_ok_to_buy(i, "UP"):
                if ub > 0:
                    rev = ub * up_inv
                    fee = cfg.fee_rate * ub * (1 - ub) * up_inv
                    cash += rev; fees_total += fee; up_inv = 0.0; up_avg_buy = 0.0
            if dn_inv > 0 and not trend_ok_to_buy(i, "DOWN"):
                if db > 0:
                    rev = db * dn_inv
                    fee = cfg.fee_rate * db * (1 - db) * dn_inv
                    cash += rev; fees_total += fee; dn_inv = 0.0; dn_avg_buy = 0.0

        # ---- UP side ----
        # BUY fill: place buy at prev_ub - bid_offset. Fill when ub <= our_buy_px.
        # Skip buy if trend filter says no.
        if up_inv < up_cap - EPS and trend_ok_to_buy(i, "UP") and not block_up_buy:
            buy_px = prev_ub - bid_offset
            if buy_px > 0 and ub < buy_px - EPS:
                fill_px = buy_px
                cost = fill_px * cfg.fill_size
                cash -= cost
                up_inv += cfg.fill_size
                rebate_base += maker_rebate_base_per_fill(fill_px, cfg.fill_size)
                up_buys += 1
                sum_up_buy += fill_px
                # Update running avg buy (weighted)
                if up_inv > 0:
                    up_avg_buy = ((up_avg_buy * (up_inv - cfg.fill_size)) + fill_px * cfg.fill_size) / up_inv

        # SELL fill: place sell at prev_ua + ask_offset OR avg_buy + min_spread (whichever higher)
        # Only if we have inventory
        if up_inv > 0:
            target_sell_px = sell_target("UP", i, up_avg_buy, up_minspread, disfavored_up, prev_ua)
            # Fill when ua >= our_sell_px (= ask rose past our level)
            if target_sell_px < 1.0 and ua > target_sell_px + EPS:
                fill_px = target_sell_px
                rev = fill_px * cfg.fill_size
                cash += rev
                up_inv -= cfg.fill_size
                rebate_base += maker_rebate_base_per_fill(fill_px, cfg.fill_size)
                up_sells += 1
                sum_up_sell += fill_px
                up_round_trips += 1
                sum_up_spread += (fill_px - up_avg_buy)
                if up_inv <= EPS:
                    up_avg_buy = 0.0

        # ---- DN side (mirror) ----
        if dn_inv < dn_cap - EPS and trend_ok_to_buy(i, "DOWN") and not block_dn_buy:
            buy_px = prev_db - bid_offset
            if buy_px > 0 and db < buy_px - EPS:
                fill_px = buy_px
                cost = fill_px * cfg.fill_size
                cash -= cost
                dn_inv += cfg.fill_size
                rebate_base += maker_rebate_base_per_fill(fill_px, cfg.fill_size)
                dn_buys += 1
                sum_dn_buy += fill_px
                if dn_inv > 0:
                    dn_avg_buy = ((dn_avg_buy * (dn_inv - cfg.fill_size)) + fill_px * cfg.fill_size) / dn_inv

        if dn_inv > 0:
            target_sell_px = sell_target("DOWN", i, dn_avg_buy, dn_minspread, disfavored_dn, prev_da)
            if target_sell_px < 1.0 and da > target_sell_px + EPS:
                fill_px = target_sell_px
                rev = fill_px * cfg.fill_size
                cash += rev
                dn_inv -= cfg.fill_size
                rebate_base += maker_rebate_base_per_fill(fill_px, cfg.fill_size)
                dn_sells += 1
                sum_dn_sell += fill_px
                dn_round_trips += 1
                sum_dn_spread += (fill_px - dn_avg_buy)
                if dn_inv <= EPS:
                    dn_avg_buy = 0.0

        prev_ub = ub; prev_ua = ua
        prev_db = db; prev_da = da

    # Payoff
    if winner == "UP":
        payoff = up_inv * 1.0
    elif winner == "DOWN":
        payoff = dn_inv * 1.0
    else:
        payoff = 0.5 * (up_inv + dn_inv)

    rebate_total = maker_rebate_from_base(rebate_base, cfg.fee_rate, cfg.rebate_pct)
    pnl_pure = cash + payoff - fees_total
    pnl = pnl_pure + rebate_total
    maker_fills = up_buys + up_sells + dn_buys + dn_sells

    if winner == "UP":
        final_winner_inv, final_loser_inv = up_inv, dn_inv
    elif winner == "DOWN":
        final_winner_inv, final_loser_inv = dn_inv, up_inv
    else:
        final_winner_inv = final_loser_inv = 0.5 * (up_inv + dn_inv)

    return PureMMResult(
        market_name=market_name, date=date,
        up_buys=up_buys, up_sells=up_sells,
        dn_buys=dn_buys, dn_sells=dn_sells,
        up_round_trips=up_round_trips, dn_round_trips=dn_round_trips,
        avg_up_buy=sum_up_buy / up_buys if up_buys else 0.0,
        avg_up_sell=sum_up_sell / up_sells if up_sells else 0.0,
        avg_dn_buy=sum_dn_buy / dn_buys if dn_buys else 0.0,
        avg_dn_sell=sum_dn_sell / dn_sells if dn_sells else 0.0,
        avg_up_spread=sum_up_spread / up_round_trips if up_round_trips else 0.0,
        avg_dn_spread=sum_dn_spread / dn_round_trips if dn_round_trips else 0.0,
        final_up_inv=up_inv, final_dn_inv=dn_inv,
        cash=cash, rebate_base=rebate_base, rebate_total=rebate_total,
        fees_total=fees_total, payoff=payoff, pnl_pure=pnl_pure, pnl=pnl,
        maker_fills=maker_fills, winner=winner,
        suspected_winner=str(suspected_winner) if suspected_winner else "NONE",
        final_loser_inv=final_loser_inv, final_winner_inv=final_winner_inv,
    )


_CFG: Optional[PureMMPolicy] = None


def _init_worker(cfg_dict: dict) -> None:
    global _CFG
    _CFG = PureMMPolicy(**cfg_dict)


def _process_one(path_str: str) -> Optional[dict]:
    cfg = _CFG or PureMMPolicy()
    p = Path(path_str)
    wide = parse_market_csv(p)
    if wide is None:
        return None
    res = simulate_pure_mm(wide, cfg, p.stem, p.parent.parent.name)
    return asdict(res) if res else None


def _collect(root: Path, max_days: Optional[int]) -> list[str]:
    days = sorted([d for d in root.iterdir() if d.is_dir()])
    if max_days:
        days = days[-max_days:]
    paths = []
    for d in days:
        csv_dir = d / "csv"
        if csv_dir.exists():
            paths.extend(str(p) for p in sorted(csv_dir.glob("*.csv")))
    return paths


def print_summary(df: pd.DataFrame, cfg: PureMMPolicy) -> None:
    n = len(df)
    days = df["date"].nunique() if "date" in df.columns else 1
    if "pnl_pure" not in df.columns:
        df = apply_rebates_to_df(df, cfg.fee_rate, cfg.rebate_pct, cfg.fill_size)
    pnl_pure = df["pnl_pure"]
    rebate = df["rebate_total"]
    pnl = df["pnl"]

    print("\n" + "=" * 100)
    print("PURE 2-SIDED MARKET MAKING")
    print("=" * 100)
    print(f"Markets: {n}    Days: {days}\n")

    print(f"Params: bid_off={cfg.bid_offset_ticks}t ask_off={cfg.ask_offset_ticks}t "
          f"min_spread={cfg.min_spread_ticks}t  max_inv={cfg.max_inventory_per_side}")
    print(f"        observe={cfg.observe_seconds}s close_at={cfg.close_at_seconds}s mode={cfg.close_mode}")
    print(f"        fee_rate={cfg.fee_rate} rebate_pct={cfg.rebate_pct} "
          f"(rebate factor={maker_rebate_factor(cfg.fee_rate, cfg.rebate_pct):.4f})\n")

    print(f"=== HEADLINE (pure = trading only; net = pure + maker rebates) ===")
    print(f"  Total P&L pure:   ${pnl_pure.sum():>10.2f}")
    print(f"  Total rebates:    ${rebate.sum():>10.2f}")
    print(f"  Total P&L net:    ${pnl.sum():>10.2f}")
    print(f"  Avg pure/market:  ${pnl_pure.mean():>10.4f}")
    print(f"  Avg rebate/market:${rebate.mean():>10.4f}")
    print(f"  Avg net/market:   ${pnl.mean():>10.4f}")
    print(f"  Median net:       ${pnl.median():>10.4f}")
    print(f"  Per day (net):    ${pnl.sum()/days:>10.2f}")
    print(f"  Markets net>0:    {int((pnl>0).sum())}/{n} ({(pnl>0).mean():.1%})")
    print(f"  Days net positive:{int((df.groupby('date')['pnl'].sum()>0).sum())}/{days}")

    print(f"\n=== ACTIVITY ===")
    print(f"  UP buys:  avg={df['up_buys'].mean():.1f}/m  total={df['up_buys'].sum()}")
    print(f"  UP sells: avg={df['up_sells'].mean():.1f}/m  total={df['up_sells'].sum()}")
    print(f"  DN buys:  avg={df['dn_buys'].mean():.1f}/m  total={df['dn_buys'].sum()}")
    print(f"  DN sells: avg={df['dn_sells'].mean():.1f}/m  total={df['dn_sells'].sum()}")
    mf = df["maker_fills"] if "maker_fills" in df.columns else (
        df["up_buys"] + df["up_sells"] + df["dn_buys"] + df["dn_sells"]
    )
    print(f"  Maker fills: avg={mf.mean():.1f}/m  total={int(mf.sum())}")
    print(f"  UP round trips: avg={df['up_round_trips'].mean():.1f}/m  total={df['up_round_trips'].sum()}")
    print(f"  DN round trips: avg={df['dn_round_trips'].mean():.1f}/m  total={df['dn_round_trips'].sum()}")

    print(f"\n=== ECONOMICS ===")
    rt_total = df["up_round_trips"].sum() + df["dn_round_trips"].sum()
    if rt_total > 0:
        avg_spread = (df["avg_up_spread"] * df["up_round_trips"] + df["avg_dn_spread"] * df["dn_round_trips"]).sum() / rt_total
        print(f"  Avg captured spread per round trip: ${avg_spread:.4f}")
    print(f"  Avg final UP inventory: {df['final_up_inv'].mean():.2f}")
    print(f"  Avg final DN inventory: {df['final_dn_inv'].mean():.2f}")
    if "final_loser_inv" in df.columns:
        print(f"  Avg loser-side residual:  {df['final_loser_inv'].mean():.2f}")
        print(f"  Avg winner-side residual: {df['final_winner_inv'].mean():.2f}")
    print(f"  Total fees paid (taker): ${df['fees_total'].sum():.2f}")
    if "rebate_base" in df.columns and mf.sum() > 0:
        avg_base_per_fill = df["rebate_base"].sum() / mf.sum()
        print(f"  Avg rebate base/fill:   {avg_base_per_fill:.4f}  "
              f"(→ ${avg_base_per_fill * maker_rebate_factor(cfg.fee_rate, cfg.rebate_pct):.4f} rebate @ p)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-days", type=int, default=None)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--out", type=str, default="utils/mm_pure.parquet")
    ap.add_argument("--bid-off", type=int, default=1)
    ap.add_argument("--ask-off", type=int, default=1)
    ap.add_argument("--min-spread", type=int, default=2)
    ap.add_argument("--max-inv", type=float, default=5.0)
    ap.add_argument("--observe", type=float, default=0.0)
    ap.add_argument("--close-at", type=float, default=0.0)
    ap.add_argument("--close-mode", type=str, default="hold")
    ap.add_argument("--trend-lookback", type=float, default=10.0)
    ap.add_argument("--trend-thr", type=float, default=0.02)
    ap.add_argument("--inv-stop", action="store_true")
    ap.add_argument("--inv-stop-thr", type=float, default=0.04)
    ap.add_argument("--trend-flatten", action="store_true")
    ap.add_argument("--inv-mgmt", action="store_true", help="continuous mid-based inventory mgmt (no skip)")
    ap.add_argument("--inv-neutral-gap", type=float, default=0.02)
    ap.add_argument("--inv-neutral-cap", type=float, default=0.0,
                    help="max cap per side when neutral (0=full max_inv)")
    ap.add_argument("--inv-stop-buy-gap", type=float, default=0.04)
    ap.add_argument("--inv-cap-slope", type=float, default=80.0)
    ap.add_argument("--inv-taker-gap", type=float, default=0.07)
    ap.add_argument("--inv-winner-spread", type=int, default=2)
    ap.add_argument("--inv-loser-spread", type=int, default=0)
    ap.add_argument("--asymm", action="store_true", help="enable asymmetric MM (momentum-biased)")
    ap.add_argument("--asymm-observe", type=float, default=30.0)
    ap.add_argument("--asymm-window", type=float, default=20.0)
    ap.add_argument("--asymm-thr", type=float, default=0.005)
    ap.add_argument("--asymm-cap-w", type=float, default=10.0)
    ap.add_argument("--asymm-cap-l", type=float, default=1.0)
    ap.add_argument("--asymm-spread-w", type=int, default=5)
    ap.add_argument("--asymm-spread-l", type=int, default=1)
    ap.add_argument("--asymm-reeval", type=float, default=0.0)
    ap.add_argument("--asymm-skip-uncertain", action="store_true")
    ap.add_argument("--asymm-signal", type=str, default="momentum", choices=["momentum","mid_diff"])
    ap.add_argument("--asymm-max-spread", type=float, default=1.0)
    ap.add_argument("--asymm-min-strength", type=float, default=0.0)
    ap.add_argument("--no-asymm-liquidate", action="store_true", help="disable taker liquidation on commit")
    ap.add_argument("--asymm-observe-mm", action="store_true", help="do symmetric MM during observation phase")
    args = ap.parse_args()

    cfg = PureMMPolicy(
        bid_offset_ticks=args.bid_off,
        ask_offset_ticks=args.ask_off,
        min_spread_ticks=args.min_spread,
        max_inventory_per_side=args.max_inv,
        observe_seconds=args.observe,
        close_at_seconds=args.close_at,
        close_mode=args.close_mode,
        trend_lookback_s=args.trend_lookback,
        trend_drop_thr=args.trend_thr,
        inv_stop_loss=args.inv_stop,
        inv_stop_loss_drop_thr=args.inv_stop_thr,
        trend_taker_flatten=args.trend_flatten,
        inv_mgmt_enabled=args.inv_mgmt,
        inv_neutral_gap=args.inv_neutral_gap,
        inv_neutral_max_cap=args.inv_neutral_cap,
        inv_stop_buy_gap=args.inv_stop_buy_gap,
        inv_loser_cap_slope=args.inv_cap_slope,
        inv_taker_trim_gap=args.inv_taker_gap,
        inv_winner_min_spread_ticks=args.inv_winner_spread,
        inv_loser_min_spread_ticks=args.inv_loser_spread,
        asymm_enabled=args.asymm,
        momentum_observe_s=args.asymm_observe,
        momentum_window_s=args.asymm_window,
        momentum_threshold=args.asymm_thr,
        asymm_cap_winner=args.asymm_cap_w,
        asymm_cap_loser=args.asymm_cap_l,
        asymm_min_spread_w_ticks=args.asymm_spread_w,
        asymm_min_spread_l_ticks=args.asymm_spread_l,
        asymm_reevaluate_s=args.asymm_reeval,
        asymm_skip_uncertain=args.asymm_skip_uncertain,
        asymm_signal_type=args.asymm_signal,
        asymm_max_spread=args.asymm_max_spread,
        asymm_min_strength=args.asymm_min_strength,
        asymm_liquidate_loser_on_commit=not args.no_asymm_liquidate,
        asymm_observe_no_trade=not args.asymm_observe_mm,
    )

    paths = _collect(PRICES_5M_ROOT, args.max_days)
    print(f"[pure_mm] paths: {len(paths)}", flush=True)

    cfg_dict = asdict(cfg)
    t0 = time.time()
    results = []
    n_done = 0
    with ProcessPoolExecutor(
        max_workers=args.workers, initializer=_init_worker, initargs=(cfg_dict,)
    ) as ex:
        futs = [ex.submit(_process_one, p) for p in paths]
        for fut in as_completed(futs):
            n_done += 1
            r = fut.result()
            if r is not None:
                results.append(r)
            if n_done % 5000 == 0 or n_done == len(paths):
                print(f"  {n_done}/{len(paths)} valid={len(results)} ({time.time()-t0:.1f}s)", flush=True)

    df = pd.DataFrame(results)
    print(f"\n[pure_mm] valid: {len(df)} ({time.time()-t0:.1f}s)")
    if df.empty:
        return 0
    if not args.no_save:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(args.out, compression="zstd")

    print_summary(df, cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
