"""
Paired market making: keep UP+DN inventory matched; open pairs when cost < $1.

Core rules
----------
1. Pair cost = price_up + price_dn on simultaneous maker BUYs. Open only if <= max_pair_cost.
2. Inventory: always drive toward up_inv == dn_inv (paired shares).
3. OSL (opposite-side lock): block BUY on the disfavored (loser) mid; only buy the
   favored leg when completing a pair (never accumulate loser via maker BUY).
4. After any SELL on one leg, immediately try to BUY the other to restore pairs
   (rebalance) if ask_sum allows.
5. Optional: sell both legs when bid_sum >= min_bid_sum_sell (round-trip).
6. Rebate on maker fills; pnl_pure vs pnl net like mm_simulate_pure.

Hold paired inventory to expiry → payoff = pairs * $1 per share.
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
from utils.mm_simulate_pure import (
    apply_rebates_to_df,
    maker_rebate_base_per_fill,
    maker_rebate_from_base,
)


@dataclass(frozen=True)
class PairedMMPolicy:
    tick: float = 0.01
    fill_size: float = 10.0

    bid_offset_ticks: int = 0
    half_spread: float = 0.01
    # When bid_sum <= bid_sum_trigger, quote with aggressive_half_spread (often 0 = at bid)
    bid_sum_trigger: float = 0.97
    quote_at_bid_on_trigger: bool = True
    aggressive_half_spread: float = 0.0

    max_pair_cost: float = 0.99
    max_ask_sum: float = 1.02

    max_pairs: float = 50.0
    trade_window_seconds: float = 280.0
    max_triggers_per_market: int = 40
    trigger_cooldown_s: float = 3.0
    order_timeout_s: float = 15.0

    opposite_side_lock: bool = True
    osl_gap: float = 0.01

    rebalance_after_sell: bool = True
    min_bid_sum_sell: float = 0.0

    orphan_sell_back: bool = True
    orphan_timeout_s: float = 8.0
    # If single fill is on winner side, try to complete pair until timeout (no early sellback)
    hold_winner_orphan: bool = True
    # When bid_sum <= trigger, acquire both legs at bid (structural arb; maker rebate)
    acquire_pair_at_bid: bool = True

    fee_rate: float = 0.07
    rebate_pct: float = 0.20


@dataclass
class PairedMMResult:
    market_name: str
    date: str
    pairs_opened: int
    pairs_closed_rt: int
    orphan_sellbacks: int
    rebalances: int
    osl_blocks: int
    pair_buys: int
    up_buys: int
    dn_buys: int
    up_sells: int
    dn_sells: int
    avg_pair_open_cost: float
    final_pairs: float
    final_up_inv: float
    final_dn_inv: float
    final_unpaired: float
    cash: float
    rebate_base: float
    rebate_total: float
    fees_total: float
    payoff: float
    pnl_pure: float
    pnl: float
    winner: str


def _favored_side(up_mid: float, dn_mid: float, gap: float) -> Optional[str]:
    """Higher mid = favored / winner side."""
    if up_mid >= dn_mid + gap:
        return "UP"
    if dn_mid >= up_mid + gap:
        return "DOWN"
    return None


def _loser_side(up_mid: float, dn_mid: float, gap: float) -> Optional[str]:
    """Lower mid = loser — OSL blocks maker BUY here."""
    fav = _favored_side(up_mid, dn_mid, gap)
    if fav == "UP":
        return "DOWN"
    if fav == "DOWN":
        return "UP"
    return None


def _expiry_payoff(up_inv: float, dn_inv: float, winner: str) -> float:
    if winner == "UP":
        return up_inv
    if winner == "DOWN":
        return dn_inv
    return (up_inv + dn_inv) * 0.5


def _pair_buy_prices(
    cfg: PairedMMPolicy,
    ub: float,
    ua: float,
    db: float,
    da: float,
    bid_sum: float,
    bid_off: float,
) -> tuple[float, float]:
    """Quote prices for pair BUYs; aggressive at bid when book is cheap."""
    cheap = cfg.bid_sum_trigger <= 0 or bid_sum <= cfg.bid_sum_trigger + 1e-9
    if cheap and cfg.quote_at_bid_on_trigger:
        up_px = max(cfg.tick, ub - bid_off)
        dn_px = max(cfg.tick, db - bid_off)
        return up_px, dn_px
    half = cfg.aggressive_half_spread if cheap else cfg.half_spread
    up_mid = (ub + ua) * 0.5
    dn_mid = (db + da) * 0.5
    up_px = max(cfg.tick, up_mid - half) - bid_off
    dn_px = max(cfg.tick, dn_mid - half) - bid_off
    return up_px, dn_px


def _side_is_loser(side: str, up_mid: float, dn_mid: float, gap: float) -> bool:
    return _loser_side(up_mid, dn_mid, gap) == side


def simulate_paired_mm(
    wide: pd.DataFrame, cfg: PairedMMPolicy, market_name: str, date: str
) -> Optional[PairedMMResult]:
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
    t_start = int(t_ms[0])
    EPS = 1e-9
    bid_off = cfg.bid_offset_ticks * cfg.tick
    half = cfg.half_spread
    win_ms = int(cfg.trade_window_seconds * 1000)
    orphan_ms = int(cfg.orphan_timeout_s * 1000)
    order_ms = int(cfg.order_timeout_s * 1000)
    cooldown_ms = int(cfg.trigger_cooldown_s * 1000)

    final_up_mid = (up_bid[-1] + up_ask[-1]) * 0.5
    final_dn_mid = (dn_bid[-1] + dn_ask[-1]) * 0.5
    if final_up_mid > final_dn_mid:
        winner = "UP"
    elif final_dn_mid > final_up_mid:
        winner = "DOWN"
    else:
        winner = "TIE"

    cash = 0.0
    fees = 0.0
    rebate_base = 0.0
    up_inv = 0.0
    dn_inv = 0.0

    pairs_opened = 0
    pairs_closed_rt = 0
    orphan_sellbacks = 0
    rebalances = 0
    osl_blocks = 0
    pair_buys = 0
    up_buys = dn_buys = up_sells = dn_sells = 0
    sum_pair_cost = 0.0

    arm_t = -1
    arm_up_px = arm_dn_px = 0.0
    arm_up_filled = arm_dn_filled = False
    last_trigger_ms = -10**9
    n_triggers = 0

    def pairs_count() -> float:
        return min(up_inv, dn_inv)

    def add_rebate_buy(px: float, sh: float) -> None:
        nonlocal rebate_base
        rebate_base += maker_rebate_base_per_fill(px, sh)

    def taker_sell(side: str, idx: int, sh: float, buy_px: float) -> None:
        nonlocal cash, fees, up_inv, dn_inv
        if sh <= EPS:
            return
        inv = up_inv if side == "UP" else dn_inv
        sh = min(sh, inv)
        if sh <= EPS:
            return
        bid = up_bid[idx] if side == "UP" else dn_bid[idx]
        if bid <= 0:
            return
        cash += bid * sh
        fees += cfg.fee_rate * bid * (1 - bid) * sh
        if side == "UP":
            up_inv = max(0.0, up_inv - sh)
        else:
            dn_inv = max(0.0, dn_inv - sh)

    def maker_buy(side: str, px: float, sh: float, *, taker: bool = False) -> None:
        nonlocal cash, fees, up_inv, dn_inv, up_buys, dn_buys, pair_buys
        if sh <= EPS or px <= 0:
            return
        cash -= px * sh
        if taker:
            fees += cfg.fee_rate * px * (1 - px) * sh
        else:
            add_rebate_buy(px, sh)
        if side == "UP":
            up_inv += sh
            up_buys += 1
        else:
            dn_inv += sh
            dn_buys += 1
        pair_buys += 1

    def maker_sell(side: str, px: float, sh: float) -> None:
        nonlocal cash, up_inv, dn_inv, up_sells, dn_sells
        if sh <= EPS or px <= 0:
            return
        cash += px * sh
        add_rebate_buy(px, sh)
        if side == "UP":
            up_inv = max(0.0, up_inv - sh)
            up_sells += 1
        else:
            dn_inv = max(0.0, dn_inv - sh)
            dn_sells += 1

    def buy_blocked(side: str, idx: int) -> bool:
        if not cfg.opposite_side_lock:
            return False
        up_mid = (up_bid[idx] + up_ask[idx]) * 0.5
        dn_mid = (dn_bid[idx] + dn_ask[idx]) * 0.5
        los = _loser_side(up_mid, dn_mid, cfg.osl_gap)
        if los == side:
            return True
        return False

    def try_open_pair(idx: int, up_px: float, dn_px: float) -> None:
        nonlocal arm_t, arm_up_px, arm_dn_px, arm_up_filled, arm_dn_filled
        if pairs_count() + cfg.fill_size > cfg.max_pairs + EPS:
            return
        if up_px + dn_px > cfg.max_pair_cost + EPS:
            return
        if up_ask[idx] + dn_ask[idx] > cfg.max_ask_sum + EPS:
            return
        arm_t = t_ms[idx]
        arm_up_px = up_px if not buy_blocked("UP", idx) else 0.0
        arm_dn_px = dn_px if not buy_blocked("DOWN", idx) else 0.0
        arm_up_filled = False
        arm_dn_filled = False
        if arm_up_px <= 0 and arm_dn_px <= 0:
            arm_t = -1

    def resolve_armed(idx: int) -> None:
        nonlocal arm_t, arm_up_filled, arm_dn_filled, pairs_opened, sum_pair_cost
        nonlocal orphan_sellbacks, last_trigger_ms
        if arm_t < 0:
            return
        if arm_up_filled and arm_dn_filled and arm_up_px > 0 and arm_dn_px > 0:
            pairs_opened += 1
            sum_pair_cost += (arm_up_px + arm_dn_px) * cfg.fill_size
            arm_t = -1
            last_trigger_ms = t_ms[idx]
            return
        elapsed = t_ms[idx] - arm_t
        up_mid = (up_bid[idx] + up_ask[idx]) * 0.5
        dn_mid = (dn_bid[idx] + dn_ask[idx]) * 0.5

        # Neither leg filled → cancel after order_timeout
        if not arm_up_filled and not arm_dn_filled and elapsed >= order_ms:
            arm_t = -1
            last_trigger_ms = t_ms[idx]
            return

        if not arm_up_filled and not arm_dn_filled:
            return
        if elapsed < orphan_ms:
            return

        # Orphan: sell back if filled leg is loser; else hold winner to expiry
        if arm_up_filled and not arm_dn_filled:
            if _side_is_loser("UP", up_mid, dn_mid, cfg.osl_gap):
                if cfg.orphan_sell_back:
                    taker_sell("UP", idx, cfg.fill_size, arm_up_px)
                    orphan_sellbacks += 1
            elif not cfg.hold_winner_orphan and cfg.orphan_sell_back:
                taker_sell("UP", idx, cfg.fill_size, arm_up_px)
                orphan_sellbacks += 1
        elif arm_dn_filled and not arm_up_filled:
            if _side_is_loser("DOWN", up_mid, dn_mid, cfg.osl_gap):
                if cfg.orphan_sell_back:
                    taker_sell("DOWN", idx, cfg.fill_size, arm_dn_px)
                    orphan_sellbacks += 1
            elif not cfg.hold_winner_orphan and cfg.orphan_sell_back:
                taker_sell("DOWN", idx, cfg.fill_size, arm_dn_px)
                orphan_sellbacks += 1
        arm_t = -1
        arm_up_filled = arm_dn_filled = False
        last_trigger_ms = t_ms[idx]

    def trim_winner_excess(idx: int, excess_side: str, sh: float) -> None:
        """OSL rebalance: sell excess on favored leg instead of buying loser."""
        nonlocal rebalances
        if sh <= EPS:
            return
        taker_sell(excess_side, idx, sh, 0.0)
        rebalances += 1

    def rebalance_pair(idx: int) -> None:
        nonlocal rebalances, osl_blocks
        if not cfg.rebalance_after_sell:
            return
        ua, da = up_ask[idx], dn_ask[idx]
        ub, db = up_bid[idx], dn_bid[idx]
        up_mid = (ub + ua) * 0.5
        dn_mid = (db + da) * 0.5
        los = _loser_side(up_mid, dn_mid, cfg.osl_gap)

        if abs(up_inv - dn_inv) < cfg.fill_size * 0.5:
            return
        if up_inv > dn_inv + EPS:
            need = up_inv - dn_inv
            if buy_blocked("DOWN", idx):
                osl_blocks += 1
                trim_winner_excess(idx, "UP", min(need, cfg.fill_size))
                return
            if ua + da > cfg.max_ask_sum + EPS:
                return
            dn_px, up_ref = _pair_buy_prices(cfg, ub, ua, db, da, ub + db, bid_off)
            if dn_px > 0 and da <= dn_px + EPS and dn_px + up_ref <= cfg.max_pair_cost + EPS:
                maker_buy("DOWN", dn_px, min(need, cfg.fill_size))
                rebalances += 1
        elif dn_inv > up_inv + EPS:
            need = dn_inv - up_inv
            if buy_blocked("UP", idx):
                osl_blocks += 1
                trim_winner_excess(idx, "DOWN", min(need, cfg.fill_size))
                return
            if ua + da > cfg.max_ask_sum + EPS:
                return
            up_px, dn_ref = _pair_buy_prices(cfg, ub, ua, db, da, ub + db, bid_off)
            if up_px > 0 and ua <= up_px + EPS and up_px + dn_ref <= cfg.max_pair_cost + EPS:
                maker_buy("UP", up_px, min(need, cfg.fill_size))
                rebalances += 1

    for i in range(1, len(wide)):
        elapsed = t_ms[i] - t_start
        ub, ua = up_bid[i], up_ask[i]
        db, da = dn_bid[i], dn_ask[i]
        up_mid = (ub + ua) * 0.5
        dn_mid = (db + da) * 0.5
        bid_sum = ub + db
        ask_sum = ua + da
        fav = _favored_side(up_mid, dn_mid, cfg.osl_gap)

        if arm_t >= 0:
            # Maker BUY fills when ask crosses our bid (same as Bot6 paper executor)
            if not arm_up_filled and arm_up_px > 0 and ua <= arm_up_px + EPS:
                if buy_blocked("UP", i):
                    osl_blocks += 1
                else:
                    maker_buy("UP", arm_up_px, cfg.fill_size)
                    arm_up_filled = True
            if not arm_dn_filled and arm_dn_px > 0 and da <= arm_dn_px + EPS:
                if buy_blocked("DOWN", i):
                    osl_blocks += 1
                else:
                    maker_buy("DOWN", arm_dn_px, cfg.fill_size)
                    arm_dn_filled = True
            resolve_armed(i)

        if elapsed > win_ms:
            continue

        # Round-trip sell both legs
        if cfg.min_bid_sum_sell > 0 and bid_sum >= cfg.min_bid_sum_sell - EPS:
            p = pairs_count()
            if p >= cfg.fill_size - EPS:
                sell_up = min(cfg.fill_size, up_inv, p)
                sell_dn = min(cfg.fill_size, dn_inv, p)
                if ub > 0 and sell_up > 0:
                    maker_sell("UP", ub, sell_up)
                if db > 0 and sell_dn > 0:
                    maker_sell("DOWN", db, sell_dn)
                if sell_up > 0 and sell_dn > 0:
                    pairs_closed_rt += 1
                rebalance_pair(i)

        if not cfg.acquire_pair_at_bid and arm_t < 0:
            rebalance_pair(i)

        if arm_t < 0 and elapsed <= win_ms and n_triggers < cfg.max_triggers_per_market:
            if t_ms[i] - last_trigger_ms >= cooldown_ms:
                trigger_ok = bid_sum <= cfg.bid_sum_trigger + EPS
                if (
                    trigger_ok
                    and bid_sum <= cfg.max_pair_cost + EPS
                    and pairs_count() + cfg.fill_size <= cfg.max_pairs + EPS
                ):
                    blocked_u = buy_blocked("UP", i)
                    blocked_d = buy_blocked("DOWN", i)
                    if (
                        cfg.acquire_pair_at_bid
                        and ub > 0
                        and db > 0
                        and not blocked_u
                        and not blocked_d
                    ):
                        maker_buy("UP", ub, cfg.fill_size)
                        maker_buy("DOWN", db, cfg.fill_size)
                        pairs_opened += 1
                        sum_pair_cost += bid_sum * cfg.fill_size
                        n_triggers += 1
                        last_trigger_ms = t_ms[i]
                    elif ask_sum <= cfg.max_ask_sum + EPS:
                        up_px, dn_px = _pair_buy_prices(cfg, ub, ua, db, da, bid_sum, bid_off)
                        if (
                            up_px > 0
                            and dn_px > 0
                            and up_px + dn_px <= cfg.max_pair_cost + EPS
                        ):
                            try_open_pair(i, up_px, dn_px)
                            if arm_t >= 0:
                                n_triggers += 1
                                if not arm_up_filled and arm_up_px > 0 and ua <= arm_up_px + EPS:
                                    if not blocked_u:
                                        maker_buy("UP", arm_up_px, cfg.fill_size)
                                        arm_up_filled = True
                                if not arm_dn_filled and arm_dn_px > 0 and da <= arm_dn_px + EPS:
                                    if not blocked_d:
                                        maker_buy("DOWN", arm_dn_px, cfg.fill_size)
                                        arm_dn_filled = True
                                resolve_armed(i)

    # Liquidate unpaired LOSER at end; keep unpaired winner for expiry
    last_i = len(wide) - 1
    up_mid_f = (up_bid[last_i] + up_ask[last_i]) * 0.5
    dn_mid_f = (dn_bid[last_i] + dn_ask[last_i]) * 0.5
    los_end = _loser_side(up_mid_f, dn_mid_f, cfg.osl_gap)
    if up_inv > dn_inv + EPS:
        excess = up_inv - dn_inv
        if los_end == "UP":
            taker_sell("UP", last_i, excess, 0)
        # else excess UP is winner — hold to expiry
    elif dn_inv > up_inv + EPS:
        excess = dn_inv - up_inv
        if los_end == "DOWN":
            taker_sell("DOWN", last_i, excess, 0)

    paired = pairs_count()
    payoff = _expiry_payoff(up_inv, dn_inv, winner)

    rebate_total = maker_rebate_from_base(rebate_base, cfg.fee_rate, cfg.rebate_pct)
    pnl_pure = cash + payoff - fees
    pnl = pnl_pure + rebate_total

    return PairedMMResult(
        market_name=market_name,
        date=date,
        pairs_opened=pairs_opened,
        pairs_closed_rt=pairs_closed_rt,
        orphan_sellbacks=orphan_sellbacks,
        rebalances=rebalances,
        osl_blocks=osl_blocks,
        pair_buys=pair_buys,
        up_buys=up_buys,
        dn_buys=dn_buys,
        up_sells=up_sells,
        dn_sells=dn_sells,
        avg_pair_open_cost=(
            sum_pair_cost / (pairs_opened * cfg.fill_size) if pairs_opened else 0.0
        ),
        final_pairs=paired,
        final_up_inv=up_inv,
        final_dn_inv=dn_inv,
        final_unpaired=abs(up_inv - dn_inv),
        cash=cash,
        rebate_base=rebate_base,
        rebate_total=rebate_total,
        fees_total=fees,
        payoff=payoff,
        pnl_pure=pnl_pure,
        pnl=pnl,
        winner=winner,
    )


_CFG: Optional[PairedMMPolicy] = None


def _init_worker(cfg_dict: dict) -> None:
    global _CFG
    _CFG = PairedMMPolicy(**cfg_dict)


def _process_one(path_str: str) -> Optional[dict]:
    cfg = _CFG or PairedMMPolicy()
    p = Path(path_str)
    wide = parse_market_csv(p)
    if wide is None:
        return None
    res = simulate_paired_mm(wide, cfg, p.stem, p.parent.parent.name)
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


def print_summary(df: pd.DataFrame, cfg: PairedMMPolicy) -> None:
    n = len(df)
    days = df["date"].nunique() if "date" in df.columns else 1
    print("\n" + "=" * 100)
    print("PAIRED MM (matched UP+DN, cost < $1, OSL)")
    print("=" * 100)
    print(f"Markets: {n}   Days: {days}")
    print(
        f"bid_sum_trigger={cfg.bid_sum_trigger} max_pair_cost={cfg.max_pair_cost} "
        f"orphan_timeout={cfg.orphan_timeout_s}s osl_gap={cfg.osl_gap} "
        f"fill_size={cfg.fill_size}"
    )
    print(f"\n=== HEADLINE ===")
    print(f"  Total PnL pure:     ${df['pnl_pure'].sum():>10.2f}")
    print(f"  Total rebates:      ${df['rebate_total'].sum():>10.2f}")
    print(f"  Total PnL net:      ${df['pnl'].sum():>10.2f}")
    print(f"  Avg net / market:   ${df['pnl'].mean():>10.4f}")
    print(f"  Median net:         ${df['pnl'].median():>10.4f}")
    print(f"  Per day (net):      ${df['pnl'].sum() / days:>10.2f}")
    print(f"  Markets net > 0:    {(df['pnl'] > 0).sum()}/{n} ({(df['pnl'] > 0).mean():.1%})")
    print(f"\n=== PAIRS ===")
    print(f"  Pairs opened:       {df['pairs_opened'].sum():.0f}  (avg {df['pairs_opened'].mean():.2f}/m)")
    print(f"  Pairs held @ expiry: {df['final_pairs'].sum():.0f} sh (avg {df['final_pairs'].mean():.2f}/m)")
    print(f"  Avg open cost:      {df.loc[df['pairs_opened'] > 0, 'avg_pair_open_cost'].mean() if (df['pairs_opened'] > 0).any() else 0:.4f}")
    print(f"  Avg final unpaired: {df['final_unpaired'].mean():.3f}")
    print(f"  Orphan sellbacks:   {df['orphan_sellbacks'].sum():.0f}")
    print(f"  Rebalances:         {df['rebalances'].sum():.0f}")
    print(f"  OSL blocks:         {df['osl_blocks'].sum():.0f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-days", type=int, default=30)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--out", type=str, default="utils/mm_paired.parquet")
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--bid-sum-trigger", type=float, default=0.98)
    ap.add_argument("--max-pair-cost", type=float, default=0.99)
    ap.add_argument("--max-ask-sum", type=float, default=1.02)
    ap.add_argument("--orphan-timeout", type=float, default=8.0)
    ap.add_argument("--fill-size", type=float, default=10.0)
    ap.add_argument("--max-pairs", type=float, default=50.0)
    ap.add_argument("--osl-gap", type=float, default=0.01)
    ap.add_argument("--no-osl", action="store_true")
    ap.add_argument("--no-bid-quote", action="store_true", help="disable quote-at-bid on trigger")
    ap.add_argument("--min-bid-sell", type=float, default=0.0)
    ap.add_argument("--sweep", action="store_true", help="grid search bid_sum_trigger x max_pair_cost")
    args = ap.parse_args()

    cfg = PairedMMPolicy(
        bid_sum_trigger=args.bid_sum_trigger,
        quote_at_bid_on_trigger=not args.no_bid_quote,
        max_pair_cost=args.max_pair_cost,
        max_ask_sum=args.max_ask_sum,
        orphan_timeout_s=args.orphan_timeout,
        fill_size=args.fill_size,
        max_pairs=args.max_pairs,
        opposite_side_lock=not args.no_osl,
        osl_gap=args.osl_gap,
        min_bid_sum_sell=args.min_bid_sell,
    )
    paths = _collect(PRICES_5M_ROOT, args.max_days)

    def _run_one(policy: PairedMMPolicy) -> pd.DataFrame:
        print(f"[paired_mm] paths={len(paths)}", flush=True)
        t0 = time.time()
        results: list[dict] = []
        with ProcessPoolExecutor(
            max_workers=args.workers, initializer=_init_worker, initargs=(asdict(policy),)
        ) as ex:
            futs = [ex.submit(_process_one, p) for p in paths]
            for j, fut in enumerate(as_completed(futs), 1):
                r = fut.result()
                if r is not None:
                    results.append(r)
                if j % 5000 == 0 or j == len(paths):
                    print(f"  {j}/{len(paths)} valid={len(results)} ({time.time() - t0:.1f}s)", flush=True)
        return pd.DataFrame(results)

    if args.sweep:
        rows = []
        for bst in (0.96, 0.97, 0.98, 0.99):
            for mpc in (0.98, 0.99, 1.00):
                pol = PairedMMPolicy(
                    bid_sum_trigger=bst,
                    quote_at_bid_on_trigger=not args.no_bid_quote,
                    max_pair_cost=mpc,
                    max_ask_sum=args.max_ask_sum,
                    orphan_timeout_s=args.orphan_timeout,
                    fill_size=args.fill_size,
                    max_pairs=args.max_pairs,
                    opposite_side_lock=not args.no_osl,
                    osl_gap=args.osl_gap,
                    min_bid_sum_sell=args.min_bid_sell,
                )
                df = _run_one(pol)
                if df.empty:
                    continue
                rows.append({
                    "bid_sum_trigger": bst,
                    "max_pair_cost": mpc,
                    "avg_pnl": df["pnl"].mean(),
                    "total_pnl": df["pnl"].sum(),
                    "pairs": df["pairs_opened"].sum(),
                    "pairs_held": df["final_pairs"].sum(),
                    "orphans": df["orphan_sellbacks"].sum(),
                    "unpaired": df["final_unpaired"].mean(),
                })
        sweep_df = pd.DataFrame(rows).sort_values("avg_pnl", ascending=False)
        print("\n=== SWEEP (top 8 by avg pnl/market) ===")
        print(sweep_df.head(8).to_string(index=False))
        return 0

    print(f"[paired_mm] paths={len(paths)}", flush=True)
    df = _run_one(cfg)
    if df.empty:
        print("No results")
        return 1
    if not args.no_save:
        df.to_parquet(args.out, index=False)
        print(f"Saved {args.out}")
    print_summary(df, cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
