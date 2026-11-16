"""
Pair-Atomic MM with strict orphan management.

Idea: trigger on low bid_sum (symmetric liquidity retraction), place pair maker buys.
If both fill: pair acquired below $1.
If only one fills: SELL BACK orphan immediately (taker) — lock small loss instead of
big adverse expiry.

This explicitly tries to ELIMINATE the adverse selection on single-leg fills.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
import os, time
from typing import Optional

import numpy as np
import pandas as pd

from utils.mm_analyze_5m import parse_market_csv, PRICES_5M_ROOT


@dataclass(frozen=True)
class PairAtomicPolicy:
    tick: float = 0.01
    fill_size: float = 1.0

    # Trigger: only place pair when bid_sum < bid_sum_trigger
    bid_sum_trigger: float = 0.97

    # Place maker buys at BBO_bid - bid_offset_ticks on each side
    bid_offset_ticks: int = 1

    # If only one leg fills within orphan_timeout_s, cancel the other and SELL BACK orphan
    orphan_timeout_s: float = 3.0
    # If neither fills within order_timeout_s, cancel both
    order_timeout_s: float = 30.0

    # Cooldown between trigger events (avoid retriggering immediately after a failed cycle)
    trigger_cooldown_s: float = 5.0

    # Max trigger events per market
    max_triggers_per_market: int = 50

    # Trade window
    trade_window_seconds: float = 270.0  # stop new triggers at 270s, give 30s for cleanup

    # End-of-market: liquidate any remaining pair via taker if requested
    final_taker_close: bool = False

    fee_rate: float = 0.07
    rebate_pct: float = 0.20


@dataclass
class PairAtomicResult:
    market_name: str
    date: str
    n_triggers: int
    n_both_fill: int
    n_orphan_only: int
    n_neither: int
    avg_pair_cost: float            # avg cost when both fill
    avg_orphan_loss: float          # avg loss when orphan event
    pairs_acquired: int             # = n_both_fill at end (kept to expiry)
    cash: float
    rebate_total: float
    fees_total: float
    payoff: float
    pnl: float
    winner: str


def simulate_pair_atomic(
    wide: pd.DataFrame, cfg: PairAtomicPolicy, market_name: str, date: str
) -> Optional[PairAtomicResult]:
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
    rebate_factor = cfg.rebate_pct * cfg.fee_rate
    bid_offset = cfg.bid_offset_ticks * cfg.tick
    orphan_timeout_ms = int(cfg.orphan_timeout_s * 1000)
    order_timeout_ms = int(cfg.order_timeout_s * 1000)
    cooldown_ms = int(cfg.trigger_cooldown_s * 1000)
    win_ms = int(cfg.trade_window_seconds * 1000)

    # Determine winner
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
    rebate = 0.0
    pair_count = 0  # accumulated pairs (= 1 UP + 1 DN each)

    n_triggers = 0
    n_both_fill = 0
    n_orphan_only = 0
    n_neither = 0
    sum_pair_cost = 0.0
    sum_orphan_loss = 0.0

    last_trigger_ms = -10**9
    state = "idle"  # "idle" or "armed" (orders out)
    arm_t = 0
    up_buy_px = 0.0
    dn_buy_px = 0.0
    prev_ub = up_bid[0]
    prev_db = dn_bid[0]
    up_filled = False
    dn_filled = False

    for i in range(1, n):
        elapsed = t_ms[i] - t_start

        ub = up_bid[i]; ua = up_ask[i]
        db = dn_bid[i]; da = dn_ask[i]
        bid_sum = ub + db

        if state == "armed":
            time_armed = t_ms[i] - arm_t

            # Check for fills (BBO crosses our level)
            if not up_filled and ub < up_buy_px - EPS and up_buy_px > 0:
                # UP filled at our maker price
                fee_inst = 0.0  # maker fill no fee
                cash -= up_buy_px * cfg.fill_size
                rebate += rebate_factor * up_buy_px * (1 - up_buy_px) * cfg.fill_size
                up_filled = True
            if not dn_filled and db < dn_buy_px - EPS and dn_buy_px > 0:
                cash -= dn_buy_px * cfg.fill_size
                rebate += rebate_factor * dn_buy_px * (1 - dn_buy_px) * cfg.fill_size
                dn_filled = True

            # Both fill: pair acquired
            if up_filled and dn_filled:
                pair_count += 1
                n_both_fill += 1
                sum_pair_cost += up_buy_px + dn_buy_px
                state = "idle"
                last_trigger_ms = t_ms[i]
                up_filled = False; dn_filled = False
                prev_ub = ub; prev_db = db
                continue

            # Orphan timeout: cancel non-filled, SELL BACK orphan
            if (up_filled != dn_filled) and time_armed >= orphan_timeout_ms:
                # We have one leg filled, sell back via taker at current BID
                if up_filled:
                    if ub > 0:
                        sell_px = ub
                        cash += sell_px * cfg.fill_size
                        fees += cfg.fee_rate * sell_px * (1 - sell_px) * cfg.fill_size
                        loss = up_buy_px - sell_px  # we bought at up_buy_px, sold at ub
                        sum_orphan_loss += loss + cfg.fee_rate * sell_px * (1 - sell_px)
                    n_orphan_only += 1
                else:
                    if db > 0:
                        sell_px = db
                        cash += sell_px * cfg.fill_size
                        fees += cfg.fee_rate * sell_px * (1 - sell_px) * cfg.fill_size
                        loss = dn_buy_px - sell_px
                        sum_orphan_loss += loss + cfg.fee_rate * sell_px * (1 - sell_px)
                    n_orphan_only += 1
                state = "idle"
                last_trigger_ms = t_ms[i]
                up_filled = False; dn_filled = False
                prev_ub = ub; prev_db = db
                continue

            # Neither fills: cancel both at order_timeout
            if (not up_filled) and (not dn_filled) and time_armed >= order_timeout_ms:
                n_neither += 1
                state = "idle"
                last_trigger_ms = t_ms[i]
                prev_ub = ub; prev_db = db
                continue

            prev_ub = ub; prev_db = db
            continue

        # state == idle: look for trigger
        if elapsed > win_ms:
            prev_ub = ub; prev_db = db
            continue
        if t_ms[i] - last_trigger_ms < cooldown_ms:
            prev_ub = ub; prev_db = db
            continue
        if n_triggers >= cfg.max_triggers_per_market:
            prev_ub = ub; prev_db = db
            continue

        if bid_sum <= cfg.bid_sum_trigger + EPS and ub > 0 and db > 0:
            # Place pair maker buys
            up_buy_px = ub - bid_offset
            dn_buy_px = db - bid_offset
            if up_buy_px > 0 and dn_buy_px > 0:
                state = "armed"
                arm_t = t_ms[i]
                up_filled = False; dn_filled = False
                n_triggers += 1

        prev_ub = ub; prev_db = db

    # End of market: final taker close if requested AND we have orphan inventory
    # (in our model, orphans are sold immediately on timeout, so should be 0 here)

    # Hold pairs to expiry: each pair pays $1 (one side wins)
    if winner in ("UP", "DOWN"):
        payoff = float(pair_count) * cfg.fill_size * 1.0
    else:
        payoff = float(pair_count) * cfg.fill_size * 0.5

    pnl = cash + payoff + rebate - fees

    return PairAtomicResult(
        market_name=market_name, date=date,
        n_triggers=n_triggers,
        n_both_fill=n_both_fill,
        n_orphan_only=n_orphan_only,
        n_neither=n_neither,
        avg_pair_cost=sum_pair_cost / n_both_fill if n_both_fill else 0.0,
        avg_orphan_loss=sum_orphan_loss / n_orphan_only if n_orphan_only else 0.0,
        pairs_acquired=pair_count,
        cash=cash, rebate_total=rebate, fees_total=fees,
        payoff=payoff, pnl=pnl, winner=winner,
    )


_CFG: Optional[PairAtomicPolicy] = None


def _init_worker(cfg_dict: dict) -> None:
    global _CFG
    _CFG = PairAtomicPolicy(**cfg_dict)


def _process_one(path_str: str) -> Optional[dict]:
    cfg = _CFG or PairAtomicPolicy()
    p = Path(path_str)
    wide = parse_market_csv(p)
    if wide is None:
        return None
    res = simulate_pair_atomic(wide, cfg, p.stem, p.parent.parent.name)
    return asdict(res) if res else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-days", type=int, default=30)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--bid-sum", type=float, default=0.97)
    ap.add_argument("--bid-off", type=int, default=1)
    ap.add_argument("--orphan-timeout", type=float, default=3.0)
    ap.add_argument("--order-timeout", type=float, default=30.0)
    ap.add_argument("--cooldown", type=float, default=5.0)
    ap.add_argument("--max-triggers", type=int, default=50)
    args = ap.parse_args()

    cfg = PairAtomicPolicy(
        bid_sum_trigger=args.bid_sum,
        bid_offset_ticks=args.bid_off,
        orphan_timeout_s=args.orphan_timeout,
        order_timeout_s=args.order_timeout,
        trigger_cooldown_s=args.cooldown,
        max_triggers_per_market=args.max_triggers,
    )

    days = sorted([d for d in PRICES_5M_ROOT.iterdir() if d.is_dir()])
    if args.max_days:
        days = days[-args.max_days:]
    paths = []
    for d in days:
        csv_dir = d / "csv"
        if csv_dir.exists():
            paths.extend(str(p) for p in sorted(csv_dir.glob("*.csv")))

    print(f"[pair_atomic] paths: {len(paths)}  bid_sum<{cfg.bid_sum_trigger}  bid_off={cfg.bid_offset_ticks}t  orphan_to={cfg.orphan_timeout_s}s")
    cfg_dict = asdict(cfg)
    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker, initargs=(cfg_dict,)) as ex:
        futs = [ex.submit(_process_one, p) for p in paths]
        for f in as_completed(futs):
            r = f.result()
            if r:
                results.append(r)
    df = pd.DataFrame(results)
    print(f"[pair_atomic] valid: {len(df)}  ({time.time()-t0:.1f}s)\n")
    if df.empty:
        return 0
    n_days = df["date"].nunique()

    print("=" * 100)
    print(f"PAIR-ATOMIC MM   bid_sum<{cfg.bid_sum_trigger}  bid_off={cfg.bid_offset_ticks}t  orphan_to={cfg.orphan_timeout_s}s  cooldown={cfg.trigger_cooldown_s}s")
    print("=" * 100)
    pnl = df["pnl"]
    print(f"Markets: {len(df)}  Days: {n_days}\n")
    print(f"=== HEADLINE ===")
    print(f"  Total P&L:        ${pnl.sum():>10.2f}")
    print(f"  Avg P&L / market: ${pnl.mean():>10.4f}")
    print(f"  Median P&L:       ${pnl.median():>10.4f}")
    print(f"  Per day:          ${pnl.sum()/n_days:>10.2f}")
    print(f"  Markets P&L>0:    {int((pnl>0).sum())}/{len(df)} ({(pnl>0).mean():.1%})")
    print(f"  Days positive:    {int((df.groupby('date')['pnl'].sum()>0).sum())}/{n_days}")

    print(f"\n=== TRIGGER OUTCOMES ===")
    print(f"  Total triggers:   {df['n_triggers'].sum():>7}  ({df['n_triggers'].mean():.2f}/market)")
    print(f"  Both filled:      {df['n_both_fill'].sum():>7}  ({df['n_both_fill'].sum()/max(df['n_triggers'].sum(),1):.1%})")
    print(f"  Orphan only:      {df['n_orphan_only'].sum():>7}  ({df['n_orphan_only'].sum()/max(df['n_triggers'].sum(),1):.1%})")
    print(f"  Neither:          {df['n_neither'].sum():>7}  ({df['n_neither'].sum()/max(df['n_triggers'].sum(),1):.1%})")

    print(f"\n=== PAIR ECONOMICS ===")
    if df['n_both_fill'].sum() > 0:
        avg_pair = (df['avg_pair_cost'] * df['n_both_fill']).sum() / df['n_both_fill'].sum()
        print(f"  Avg pair acquisition cost: ${avg_pair:.4f}")
        print(f"  Pair gain per pair:        ${1.0 - avg_pair:.4f}  + rebates")
    if df['n_orphan_only'].sum() > 0:
        avg_orphan_loss = (df['avg_orphan_loss'] * df['n_orphan_only']).sum() / df['n_orphan_only'].sum()
        print(f"  Avg orphan loss per orphan: ${avg_orphan_loss:.4f}")
    print(f"  Total rebates:    ${df['rebate_total'].sum():.2f}")
    print(f"  Total fees:       ${df['fees_total'].sum():.2f}")
    print(f"  Total pairs held to expiry: {df['pairs_acquired'].sum()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
