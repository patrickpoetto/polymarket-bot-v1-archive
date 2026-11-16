"""
HYBRID SWING strategy: pair acquisition + sell-buyback swing on loser.

Phases:
  1. PAIR ACQUISITION (t=0 to init_max_s):
     - Place maker buy at BBO_bid on UP and DN.
     - If after `init_aggressive_after_s` not filled: cancel and place AGGRESSIVE
       (= maker buy at BBO_ask - 1c, basically last-tick before becoming taker).
     - If after `init_taker_after_s` still not filled: TAKER buy at BBO_ask.
     - If init_max_pair_cost exceeded: SKIP market (track count).

  2. DIRECTION OBSERVATION (after pair complete, observe for direction_observe_s):
     - Sample (UP_mid - DN_mid) at end of observation window.
     - If |delta| > direction_threshold: declare loser side.
     - Else: skip swing, hold to expiry (= just original pair, P&L = 0 + rebates).

  3. SWING SELL (on declared loser):
     - Method A (taker): sell loser at current BID immediately. Pay taker fee.
     - Method B (maker_chase): place maker sell at BBO_ask, chase down for `swing_sell_chase_s`,
       fallback to taker if not filled.

  4. SWING BUY-BACK (on same side):
     - Place maker buy at current BID - swing_buy_offset_ticks * tick.
     - Wait for fill (BBO_bid drops past our level) for `swing_buy_chase_s`.
     - If not filled: GIVE UP (we have only the winner leg).

  5. HOLD TO EXPIRY:
     - Compute payoff: UP_shares × 1 (if UP wins) + DN_shares × 1 (if DN wins).
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


@dataclass(frozen=True)
class HybridPolicy:
    tick: float = 0.01
    fill_size: float = 1.0

    # Phase 1: Pair acquisition
    init_max_window_s: float = 90.0
    init_aggressive_after_s: float = 30.0
    init_taker_after_s: float = 60.0
    init_max_pair_cost: float = 1.05  # skip swing if pair > this
    init_maker_offset_ticks: int = 1  # 1 = bid - 1c (standard), 0 = join at current bid

    # Phase 2: Direction
    direction_observe_s: float = 30.0
    direction_threshold: float = 0.05  # |UP_mid - 0.5| > thr

    # Phase 3-4: Swing
    do_swing: bool = True
    swing_sell_method: str = "taker"   # "taker" or "maker"
    swing_sell_chase_s: float = 30.0
    swing_buy_offset_ticks: int = 2
    swing_buy_chase_s: float = 90.0

    # End of window: cancel orders
    trade_window_seconds: float = 300.0

    fee_rate: float = 0.07
    rebate_pct: float = 0.20


@dataclass
class HybridResult:
    market_name: str
    date: str
    initial_pair_cost: float
    init_up_method: str   # "maker", "aggressive", "taker", "none"
    init_dn_method: str
    pair_acquired: bool
    direction: str        # "UP", "DOWN", "NEUTRAL", "NA"
    swing_attempted: bool
    swing_sell_filled: bool
    swing_sell_price: float
    swing_buy_filled: bool
    swing_buy_price: float
    swing_cash_gain: float  # sell_price - buy_price (per share)
    up_shares: float
    dn_shares: float
    cash_spent: float       # net cash out (= - cash_balance)
    rebate_total: float
    fees_total: float
    payoff: float
    pnl: float
    winner: str


def _ffill(arr: np.ndarray) -> np.ndarray:
    out = arr.copy()
    last = np.nan
    for i in range(len(out)):
        if np.isnan(out[i]):
            out[i] = last
        else:
            last = out[i]
    return out


def simulate_hybrid(
    wide: pd.DataFrame, cfg: HybridPolicy, market_name: str, date: str
) -> Optional[HybridResult]:
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
    rebate_factor = cfg.rebate_pct * cfg.fee_rate

    final_up_mid = (up_bid[-1] + up_ask[-1]) * 0.5
    final_dn_mid = (dn_bid[-1] + dn_ask[-1]) * 0.5
    if final_up_mid > final_dn_mid:
        winner = "UP"
    elif final_dn_mid > final_up_mid:
        winner = "DOWN"
    else:
        winner = "TIE"

    # State
    up_shares = 0.0
    dn_shares = 0.0
    cash = 0.0  # negative when we buy, positive when we sell
    rebate_total = 0.0
    fees_total = 0.0

    init_up_method = "none"
    init_dn_method = "none"
    init_up_filled = False
    init_dn_filled = False

    # Aggressive flag: once true, place maker at BBO_ask - 1c (pay 1c more)
    up_aggressive = False
    dn_aggressive = False

    # ---- Phase 1: Pair acquisition ----
    init_max_ms = int(cfg.init_max_window_s * 1000)
    aggressive_ms = int(cfg.init_aggressive_after_s * 1000)
    taker_ms = int(cfg.init_taker_after_s * 1000)

    prev_ub = up_bid[0]
    prev_db = dn_bid[0]
    pair_complete_idx = None

    for i in range(1, n):
        elapsed = t_ms[i] - t_start
        if elapsed > init_max_ms:
            break
        ub = up_bid[i]
        ua = up_ask[i]
        db = dn_bid[i]
        da = dn_ask[i]

        # If both filled, we're done with phase 1
        if init_up_filled and init_dn_filled:
            pair_complete_idx = i
            break

        # UP side
        if not init_up_filled:
            if elapsed >= taker_ms:
                # Taker buy at current ASK
                if ua > 0:
                    fill_px = ua
                    cost = fill_px * cfg.fill_size
                    fee = cfg.fee_rate * fill_px * (1 - fill_px) * cfg.fill_size
                    cash -= cost
                    fees_total += fee
                    up_shares += cfg.fill_size
                    init_up_method = "taker"
                    init_up_filled = True
            elif elapsed >= aggressive_ms:
                # Aggressive maker: place at BBO_ask - 1c (= 1c above bid)
                # Fill when BBO_ask drops past our level (= someone takers our bid level?)
                # Simplification: fill if BBO_ask drops to our level (= prev_BBO_ask - 1c)
                target_px = ua - tick
                if target_px > 0:
                    # If next-tick BBO_ask is <= target, we filled at target
                    # (the ask dropped to our maker bid = match)
                    if ub > target_px - EPS:  # bid at or above our level = our bid is in front of book
                        # Aggressive maker bid above current bid: post_only requires we're not crossing ask
                        # We'd be at target_px = ua - 1c, just below ask
                        # Fill when ask drops to target_px
                        # We approximate: if the new ASK is <= target_px, fill at target_px
                        if ua <= target_px + EPS:
                            fill_px = target_px
                            cost = fill_px * cfg.fill_size
                            cash -= cost
                            up_shares += cfg.fill_size
                            rebate_total += rebate_factor * fill_px * (1 - fill_px) * cfg.fill_size
                            init_up_method = "aggressive"
                            init_up_filled = True
                up_aggressive = True
            else:
                # Standard maker: place at BBO_bid - offset
                # Fill when BBO_bid drops past our level
                target_px = prev_ub - cfg.init_maker_offset_ticks * tick
                if ub < target_px - EPS and target_px > 0:
                    fill_px = target_px
                    cost = fill_px * cfg.fill_size
                    cash -= cost
                    up_shares += cfg.fill_size
                    rebate_total += rebate_factor * fill_px * (1 - fill_px) * cfg.fill_size
                    init_up_method = "maker"
                    init_up_filled = True

        # DN side (mirror)
        if not init_dn_filled:
            if elapsed >= taker_ms:
                if da > 0:
                    fill_px = da
                    cost = fill_px * cfg.fill_size
                    fee = cfg.fee_rate * fill_px * (1 - fill_px) * cfg.fill_size
                    cash -= cost
                    fees_total += fee
                    dn_shares += cfg.fill_size
                    init_dn_method = "taker"
                    init_dn_filled = True
            elif elapsed >= aggressive_ms:
                target_px = da - tick
                if target_px > 0:
                    if db > target_px - EPS:
                        if da <= target_px + EPS:
                            fill_px = target_px
                            cost = fill_px * cfg.fill_size
                            cash -= cost
                            dn_shares += cfg.fill_size
                            rebate_total += rebate_factor * fill_px * (1 - fill_px) * cfg.fill_size
                            init_dn_method = "aggressive"
                            init_dn_filled = True
                dn_aggressive = True
            else:
                target_px = prev_db - cfg.init_maker_offset_ticks * tick
                if db < target_px - EPS and target_px > 0:
                    fill_px = target_px
                    cost = fill_px * cfg.fill_size
                    cash -= cost
                    dn_shares += cfg.fill_size
                    rebate_total += rebate_factor * fill_px * (1 - fill_px) * cfg.fill_size
                    init_dn_method = "maker"
                    init_dn_filled = True

        prev_ub = ub
        prev_db = db

    # Final taker fallback after init_max_window if still not both filled
    if not (init_up_filled and init_dn_filled):
        # Force taker on missing side at last seen ASK
        last_idx = min(int(init_max_ms / 1000 * 10), n - 1)
        if last_idx >= n: last_idx = n - 1
        if not init_up_filled:
            ua = up_ask[last_idx]
            if ua > 0:
                fill_px = ua
                cost = fill_px * cfg.fill_size
                fee = cfg.fee_rate * fill_px * (1 - fill_px) * cfg.fill_size
                cash -= cost
                fees_total += fee
                up_shares += cfg.fill_size
                init_up_method = "taker_force"
                init_up_filled = True
        if not init_dn_filled:
            da = dn_ask[last_idx]
            if da > 0:
                fill_px = da
                cost = fill_px * cfg.fill_size
                fee = cfg.fee_rate * fill_px * (1 - fill_px) * cfg.fill_size
                cash -= cost
                fees_total += fee
                dn_shares += cfg.fill_size
                init_dn_method = "taker_force"
                init_dn_filled = True

    pair_acquired = init_up_filled and init_dn_filled
    initial_pair_cost = -cash if pair_acquired else 0.0

    # ---- Phase 2: Direction observation ----
    direction = "NA"
    swing_attempted = False
    swing_sell_filled = False
    swing_sell_price = 0.0
    swing_buy_filled = False
    swing_buy_price = 0.0
    swing_cash_gain = 0.0

    if not pair_acquired or not cfg.do_swing or initial_pair_cost > cfg.init_max_pair_cost + EPS:
        # skip swing
        direction = "SKIP"
    else:
        # Find direction observation start: end of init window
        obs_start_idx = None
        for i in range(n):
            if t_ms[i] - t_start >= int(cfg.init_max_window_s * 1000) - EPS:
                obs_start_idx = i
                break
        if obs_start_idx is None:
            obs_start_idx = max(1, int(cfg.init_max_window_s / 0.1))
        if obs_start_idx >= n:
            obs_start_idx = n - 1

        obs_end_ms = (t_ms[obs_start_idx] - t_start) + int(cfg.direction_observe_s * 1000)
        obs_end_idx = obs_start_idx
        for i in range(obs_start_idx, n):
            if (t_ms[i] - t_start) >= obs_end_ms:
                obs_end_idx = i
                break

        up_mid = (up_bid[obs_end_idx] + up_ask[obs_end_idx]) * 0.5
        dn_mid = (dn_bid[obs_end_idx] + dn_ask[obs_end_idx]) * 0.5
        delta = up_mid - dn_mid

        if delta > cfg.direction_threshold + EPS:
            direction = "UP"  # UP is winner, DN is loser
            loser = "DN"
        elif delta < -cfg.direction_threshold - EPS:
            direction = "DOWN"  # DN is winner, UP is loser
            loser = "UP"
        else:
            direction = "NEUTRAL"
            loser = None

        if loser is not None:
            swing_attempted = True
            sell_start_idx = obs_end_idx

            # ---- Phase 3: Swing SELL on loser ----
            sell_chase_end_ms = (t_ms[sell_start_idx] - t_start) + int(cfg.swing_sell_chase_s * 1000)
            sell_idx = None
            sell_px = 0.0
            sell_method = cfg.swing_sell_method

            if sell_method == "taker":
                # Immediate taker sell at current BID
                if loser == "UP" and up_bid[sell_start_idx] > 0:
                    sell_px = up_bid[sell_start_idx]
                    sell_idx = sell_start_idx
                elif loser == "DN" and dn_bid[sell_start_idx] > 0:
                    sell_px = dn_bid[sell_start_idx]
                    sell_idx = sell_start_idx
                if sell_idx is not None:
                    revenue = sell_px * cfg.fill_size
                    fee = cfg.fee_rate * sell_px * (1 - sell_px) * cfg.fill_size
                    cash += revenue
                    fees_total += fee
                    if loser == "UP":
                        up_shares -= cfg.fill_size
                    else:
                        dn_shares -= cfg.fill_size
                    swing_sell_filled = True
                    swing_sell_price = sell_px
            elif sell_method == "maker":
                # Place maker sell at BBO_ask, chase down (re-place if BBO_ask drops)
                # Fill when our level "transacts" (we approximate: BBO_ask rises past our level
                # or our level holds while bid catches up)
                # Simplification: maker sell at BBO_ask fills if next-tick BBO_ask <= prev_BBO_ask
                # AND BBO_bid rose to or above our level (someone takered)
                # Even simpler heuristic: assume we get the avg of BBO_ask over chase window
                # Actually: simpler approximation: each tick during chase, our maker sell
                # is at current BBO_ask. If BBO_bid rises such that someone could take our ask, we fill.
                # We model: each tick during chase, if prev_BBO_ask was eaten (= new ask > prev ask), fill at prev ask.
                # In a CRASHING market this is rare → mostly not filled → fallback to taker at end.
                prev_a = up_ask[sell_start_idx] if loser == "UP" else dn_ask[sell_start_idx]
                for j in range(sell_start_idx + 1, n):
                    if (t_ms[j] - t_start) > sell_chase_end_ms:
                        break
                    cur_a = up_ask[j] if loser == "UP" else dn_ask[j]
                    if cur_a > prev_a + EPS:
                        # ask level was eaten upward → our sell at prev_a filled
                        sell_px = prev_a
                        sell_idx = j
                        break
                    prev_a = cur_a
                if sell_idx is not None:
                    revenue = sell_px * cfg.fill_size
                    cash += revenue
                    rebate_total += rebate_factor * sell_px * (1 - sell_px) * cfg.fill_size
                    if loser == "UP":
                        up_shares -= cfg.fill_size
                    else:
                        dn_shares -= cfg.fill_size
                    swing_sell_filled = True
                    swing_sell_price = sell_px
                else:
                    # Fallback to taker at end of chase window
                    end_chase_idx = min(sell_start_idx + int(cfg.swing_sell_chase_s * 10), n - 1)
                    bid_at = up_bid[end_chase_idx] if loser == "UP" else dn_bid[end_chase_idx]
                    if bid_at > 0:
                        sell_px = bid_at
                        sell_idx = end_chase_idx
                        revenue = sell_px * cfg.fill_size
                        fee = cfg.fee_rate * sell_px * (1 - sell_px) * cfg.fill_size
                        cash += revenue
                        fees_total += fee
                        if loser == "UP":
                            up_shares -= cfg.fill_size
                        else:
                            dn_shares -= cfg.fill_size
                        swing_sell_filled = True
                        swing_sell_price = sell_px

            # ---- Phase 4: Swing BUY-BACK ----
            if swing_sell_filled and sell_idx is not None:
                buy_chase_end_ms = (t_ms[sell_idx] - t_start) + int(cfg.swing_buy_chase_s * 1000)
                # Place maker buy at current BID - swing_buy_offset_ticks * tick
                # Re-quote each tick (= follow BID down). Fill when BBO_bid drops past our level.
                prev_b = up_bid[sell_idx] if loser == "UP" else dn_bid[sell_idx]
                target_px = prev_b - cfg.swing_buy_offset_ticks * tick
                buy_idx = None
                buy_px = 0.0
                for j in range(sell_idx + 1, n):
                    if (t_ms[j] - t_start) > buy_chase_end_ms:
                        break
                    cur_b = up_bid[j] if loser == "UP" else dn_bid[j]
                    cur_a = up_ask[j] if loser == "UP" else dn_ask[j]
                    # We chase down: re-set target to current_BID - offset
                    # Fill when cur_b <= our target (= bid dropped to our level)
                    # Use the running target which "follows" the BID.
                    # Simplest: each tick, target = max(prev_target, cur_b - offset*tick)
                    # No wait: we want target to FOLLOW DOWN, so target = cur_b - offset*tick (re-place)
                    # Fill if cur_b <= prev_target - EPS (bid dropped past our existing order at prev_target)
                    if cur_b <= target_px - EPS and cur_b > 0:
                        # Our existing buy at target_px got filled
                        buy_px = target_px
                        buy_idx = j
                        break
                    # Else, re-set target to follow BID down
                    new_target = cur_b - cfg.swing_buy_offset_ticks * tick
                    if new_target < target_px:
                        target_px = new_target

                if buy_idx is not None and buy_px > 0:
                    cost = buy_px * cfg.fill_size
                    cash -= cost
                    rebate_total += rebate_factor * buy_px * (1 - buy_px) * cfg.fill_size
                    if loser == "UP":
                        up_shares += cfg.fill_size
                    else:
                        dn_shares += cfg.fill_size
                    swing_buy_filled = True
                    swing_buy_price = buy_px
                    swing_cash_gain = swing_sell_price - swing_buy_price
                else:
                    # Fallback: ALWAYS TAKER buy back at current ASK (lock in pair completion)
                    # Even if at higher price than sell — better than partial (= naked winner bet)
                    end_idx = min(sell_idx + int(cfg.swing_buy_chase_s * 10), n - 1)
                    cur_ask = up_ask[end_idx] if loser == "UP" else dn_ask[end_idx]
                    if cur_ask > 0:
                        cost = cur_ask * cfg.fill_size
                        fee = cfg.fee_rate * cur_ask * (1 - cur_ask) * cfg.fill_size
                        cash -= cost
                        fees_total += fee
                        if loser == "UP":
                            up_shares += cfg.fill_size
                        else:
                            dn_shares += cfg.fill_size
                        swing_buy_filled = True
                        swing_buy_price = cur_ask
                        swing_cash_gain = swing_sell_price - swing_buy_price

    # ---- Phase 5: End-of-market payoff ----
    if winner == "UP":
        payoff = up_shares * 1.0
    elif winner == "DOWN":
        payoff = dn_shares * 1.0
    else:
        payoff = 0.5 * (up_shares + dn_shares)

    pnl = cash + payoff + rebate_total - fees_total

    return HybridResult(
        market_name=market_name, date=date,
        initial_pair_cost=initial_pair_cost,
        init_up_method=init_up_method, init_dn_method=init_dn_method,
        pair_acquired=pair_acquired,
        direction=direction,
        swing_attempted=swing_attempted,
        swing_sell_filled=swing_sell_filled, swing_sell_price=swing_sell_price,
        swing_buy_filled=swing_buy_filled, swing_buy_price=swing_buy_price,
        swing_cash_gain=swing_cash_gain,
        up_shares=up_shares, dn_shares=dn_shares,
        cash_spent=-cash if cash < 0 else 0.0,
        rebate_total=rebate_total, fees_total=fees_total,
        payoff=payoff, pnl=pnl, winner=winner,
    )


_CFG: Optional[HybridPolicy] = None


def _init_worker(cfg_dict: dict) -> None:
    global _CFG
    _CFG = HybridPolicy(**cfg_dict)


def _process_one(path_str: str) -> Optional[dict]:
    cfg = _CFG or HybridPolicy()
    p = Path(path_str)
    wide = parse_market_csv(p)
    if wide is None:
        return None
    res = simulate_hybrid(wide, cfg, p.stem, p.parent.parent.name)
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


def print_summary(df: pd.DataFrame, cfg: HybridPolicy) -> None:
    n = len(df)
    days = df["date"].nunique() if "date" in df.columns else 1
    pnl = df["pnl"]

    print("\n" + "=" * 100)
    print("HYBRID SWING STRATEGY")
    print("=" * 100)
    print(f"Markets: {n}    Days: {days}\n")

    print(f"Init params: max_win={cfg.init_max_window_s}s aggr_after={cfg.init_aggressive_after_s}s "
          f"taker_after={cfg.init_taker_after_s}s max_pair=${cfg.init_max_pair_cost}")
    print(f"Direction: observe={cfg.direction_observe_s}s thr={cfg.direction_threshold}")
    print(f"Swing: do={cfg.do_swing} sell_method={cfg.swing_sell_method} "
          f"buy_offset={cfg.swing_buy_offset_ticks}t buy_chase={cfg.swing_buy_chase_s}s\n")

    print(f"=== HEADLINE ===")
    print(f"  Total P&L:        ${pnl.sum():>10.2f}")
    print(f"  Avg P&L / market: ${pnl.mean():>10.4f}")
    print(f"  Median P&L:       ${pnl.median():>10.4f}")
    print(f"  Per day:          ${pnl.sum()/days:>10.2f}")
    print(f"  Markets P&L>0:    {int((pnl>0).sum())}/{n} ({(pnl>0).mean():.1%})")
    print(f"  Days positive:    {int((df.groupby('date')['pnl'].sum()>0).sum())}/{days}")

    pair_ok = df[df["pair_acquired"]]
    print(f"\n=== PAIR ACQUISITION ===")
    print(f"  Pair acquired:    {len(pair_ok)} ({len(pair_ok)/n:.1%})")
    if len(pair_ok):
        print(f"  Avg init pair $:  ${pair_ok['initial_pair_cost'].mean():.4f}")
        print(f"  Median init pair $: ${pair_ok['initial_pair_cost'].median():.4f}")
        # Methods
        up_methods = pair_ok["init_up_method"].value_counts(normalize=True)
        dn_methods = pair_ok["init_dn_method"].value_counts(normalize=True)
        print(f"  UP methods: {dict(up_methods.round(3))}")
        print(f"  DN methods: {dict(dn_methods.round(3))}")

    direction_counts = df["direction"].value_counts()
    print(f"\n=== DIRECTION ===")
    for d, c in direction_counts.items():
        print(f"  {d}: {c} ({c/n:.1%})")

    swing_attempt = df[df["swing_attempted"]]
    swing_complete = df[df["swing_buy_filled"]]
    print(f"\n=== SWING ===")
    print(f"  Attempts:   {len(swing_attempt)} ({len(swing_attempt)/n:.1%})")
    print(f"  Sell filled:    {int(df['swing_sell_filled'].sum())}")
    print(f"  Buy-back filled: {int(df['swing_buy_filled'].sum())}")
    if len(swing_complete):
        print(f"  Avg sell $:  ${swing_complete['swing_sell_price'].mean():.4f}")
        print(f"  Avg buy $:   ${swing_complete['swing_buy_price'].mean():.4f}")
        print(f"  Avg cash gain (sell-buy) /share: ${swing_complete['swing_cash_gain'].mean():.4f}")
        print(f"  Avg P&L on completed swings: ${swing_complete['pnl'].mean():.4f}")

    print(f"\n=== P&L BREAKDOWN ===")
    print(f"  Pair NOT acquired (skipped):     {int((~df['pair_acquired']).sum())} avg P&L=${df[~df['pair_acquired']]['pnl'].mean():.4f}" if (~df['pair_acquired']).any() else "  Pair always acquired")
    print(f"  Pair OK + no swing (NEUTRAL/SKIP): {int((df['pair_acquired'] & ~df['swing_attempted']).sum())} avg P&L=${df[df['pair_acquired'] & ~df['swing_attempted']]['pnl'].mean():.4f}" if (df['pair_acquired'] & ~df['swing_attempted']).any() else "")
    s_ok = df[df['swing_attempted'] & df['swing_buy_filled']]
    s_partial = df[df['swing_attempted'] & df['swing_sell_filled'] & ~df['swing_buy_filled']]
    s_failed = df[df['swing_attempted'] & ~df['swing_sell_filled']]
    if len(s_ok):
        print(f"  Swing FULL (sell+buy):          {len(s_ok)} avg P&L=${s_ok['pnl'].mean():.4f}")
    if len(s_partial):
        print(f"  Swing PARTIAL (sell only):      {len(s_partial)} avg P&L=${s_partial['pnl'].mean():.4f}")
    if len(s_failed):
        print(f"  Swing FAILED (no sell):         {len(s_failed)} avg P&L=${s_failed['pnl'].mean():.4f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-days", type=int, default=None)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--out", type=str, default="utils/mm_hybrid_swing.parquet")
    ap.add_argument("--init-max", type=float, default=90.0)
    ap.add_argument("--aggr-after", type=float, default=30.0)
    ap.add_argument("--taker-after", type=float, default=60.0)
    ap.add_argument("--init-max-pair", type=float, default=1.05)
    ap.add_argument("--observe", type=float, default=30.0)
    ap.add_argument("--thr", type=float, default=0.05)
    ap.add_argument("--swing", type=int, default=1)
    ap.add_argument("--sell-method", type=str, default="taker", choices=["taker", "maker"])
    ap.add_argument("--buy-offset", type=int, default=2)
    ap.add_argument("--buy-chase", type=float, default=90.0)
    ap.add_argument("--init-offset", type=int, default=1)
    args = ap.parse_args()

    cfg = HybridPolicy(
        init_max_window_s=args.init_max,
        init_aggressive_after_s=args.aggr_after,
        init_taker_after_s=args.taker_after,
        init_max_pair_cost=args.init_max_pair,
        direction_observe_s=args.observe,
        direction_threshold=args.thr,
        do_swing=bool(args.swing),
        swing_sell_method=args.sell_method,
        swing_buy_offset_ticks=args.buy_offset,
        swing_buy_chase_s=args.buy_chase,
        init_maker_offset_ticks=args.init_offset,
    )

    paths = _collect(PRICES_5M_ROOT, args.max_days)
    print(f"[hybrid_swing] paths: {len(paths)}", flush=True)

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
    print(f"\n[hybrid_swing] valid: {len(df)} ({time.time()-t0:.1f}s)")
    if df.empty:
        return 0

    if not args.no_save:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(args.out, compression="zstd")

    print_summary(df, cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
