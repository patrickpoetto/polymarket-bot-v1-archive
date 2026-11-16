"""
Cross-leg arbitrage simulator.

Strategy: when UP_ask + DN_ask < threshold (e.g., 0.99), TAKER buy both legs.
Each pair pays $1 at expiry; cost is ask_sum < 1; net profit per pair = 1 - ask_sum - fees.

Realistic assumptions:
  - Latency: only fill on opps that persist for at least N ticks (100ms each)
  - Liquidity: max fill_size per opp (we cannot move the market)
  - Cooldown: after detecting an opp, wait C seconds before re-detecting
  - Fees: standard taker fees apply on both legs

Edge cases:
  - If only one leg fills (the other moves away mid-execution), we have residual single leg.
    Modeled as: assume both fill simultaneously (ideal). Conservative variant: skip if
    bid_sum at time of detection > 1 (which would indicate one leg about to fill).
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
class ArbPolicy:
    threshold: float = 0.99       # take when ask_sum < threshold
    fill_size: float = 1.0
    persist_ticks: int = 1        # opp must persist this many ticks before we can hit (latency)
    cooldown_s: float = 0.5       # cooldown between trades
    max_trades_per_market: int = 100
    fee_rate: float = 0.07
    # Optional: skip very deep arbs that are likely stale orders
    max_arb_depth: float = 0.10   # ignore opps with profit > this (likely error/stale)
    # Time gating: only scan within window
    earliest_seconds: float = 0.0   # only after this many seconds in market
    latest_seconds: float = 300.0   # only before this many seconds (after expiry irrelevant)


@dataclass
class ArbResult:
    market_name: str
    date: str
    n_opps_detected: int
    n_trades_executed: int
    total_cost: float
    total_payoff: float
    total_fees: float
    total_profit: float
    avg_ask_sum: float
    winner: str


def simulate_arb(wide: pd.DataFrame, cfg: ArbPolicy, market_name: str, date: str) -> Optional[ArbResult]:
    if wide is None or len(wide) < 30:
        return None
    wide = wide.copy()
    for col in ("up_bid", "up_ask", "dn_bid", "dn_ask"):
        wide[col] = wide[col].ffill()
    wide = wide.dropna(subset=["up_bid", "up_ask", "dn_bid", "dn_ask"]).reset_index(drop=True)
    if len(wide) < 30:
        return None

    ua = wide["up_ask"].to_numpy(dtype=np.float64)
    da = wide["dn_ask"].to_numpy(dtype=np.float64)
    ub = wide["up_bid"].to_numpy(dtype=np.float64)
    db = wide["dn_bid"].to_numpy(dtype=np.float64)
    t_ms = wide["t_ms"].to_numpy(dtype=np.int64)
    n = len(wide)

    final_up_mid = (ub[-1] + ua[-1]) * 0.5
    final_dn_mid = (db[-1] + da[-1]) * 0.5
    if final_up_mid > final_dn_mid:
        winner = "UP"
    elif final_dn_mid > final_up_mid:
        winner = "DOWN"
    else:
        winner = "TIE"

    cooldown_ms = int(cfg.cooldown_s * 1000)
    earliest_ms = int(cfg.earliest_seconds * 1000)
    latest_ms = int(cfg.latest_seconds * 1000)
    t_start = int(t_ms[0])
    last_trade_ms = -10_000_000_000
    n_opps_detected = 0
    n_trades_executed = 0
    total_cost = 0.0
    total_fees = 0.0
    total_payoff = 0.0
    sum_ask_sum = 0.0

    # Track persistence: count consecutive ticks where ask_sum < threshold
    persist_count = 0
    for i in range(n):
        elapsed = t_ms[i] - t_start
        if elapsed < earliest_ms or elapsed > latest_ms:
            persist_count = 0
            continue
        ask_sum = ua[i] + da[i]
        if ua[i] <= 0 or da[i] <= 0:
            persist_count = 0
            continue
        if ask_sum < cfg.threshold:
            persist_count += 1
            n_opps_detected += 1
            arb_depth = 1.0 - ask_sum
            if arb_depth > cfg.max_arb_depth:
                # Skip overly deep arbs (likely stale)
                continue
            if persist_count < cfg.persist_ticks:
                continue
            if t_ms[i] - last_trade_ms < cooldown_ms:
                continue
            if n_trades_executed >= cfg.max_trades_per_market:
                continue
            # Execute taker buy on both legs
            cost_up = ua[i] * cfg.fill_size
            cost_dn = da[i] * cfg.fill_size
            fee_up = cfg.fee_rate * ua[i] * (1 - ua[i]) * cfg.fill_size
            fee_dn = cfg.fee_rate * da[i] * (1 - da[i]) * cfg.fill_size
            total_cost += cost_up + cost_dn
            total_fees += fee_up + fee_dn
            sum_ask_sum += ask_sum
            n_trades_executed += 1
            last_trade_ms = t_ms[i]
            persist_count = 0
        else:
            persist_count = 0

    # Each pair pays $1 at expiry (one side wins)
    if winner in ("UP", "DOWN"):
        total_payoff = float(n_trades_executed) * cfg.fill_size * 1.0
    else:
        total_payoff = float(n_trades_executed) * cfg.fill_size * 0.5  # tie

    total_profit = total_payoff - total_cost - total_fees
    avg_ask_sum = sum_ask_sum / n_trades_executed if n_trades_executed else 0.0

    return ArbResult(
        market_name=market_name, date=date,
        n_opps_detected=n_opps_detected,
        n_trades_executed=n_trades_executed,
        total_cost=total_cost,
        total_payoff=total_payoff,
        total_fees=total_fees,
        total_profit=total_profit,
        avg_ask_sum=avg_ask_sum,
        winner=winner,
    )


_CFG: Optional[ArbPolicy] = None


def _init_worker(cfg_dict: dict) -> None:
    global _CFG
    _CFG = ArbPolicy(**cfg_dict)


def _process_one(path_str: str) -> Optional[dict]:
    cfg = _CFG or ArbPolicy()
    p = Path(path_str)
    wide = parse_market_csv(p)
    if wide is None:
        return None
    res = simulate_arb(wide, cfg, p.stem, p.parent.parent.name)
    return asdict(res) if res else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-days", type=int, default=30)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--threshold", type=float, default=0.99)
    ap.add_argument("--fill-size", type=float, default=1.0)
    ap.add_argument("--persist", type=int, default=1)
    ap.add_argument("--cooldown", type=float, default=0.5)
    ap.add_argument("--max-trades", type=int, default=100)
    ap.add_argument("--max-depth", type=float, default=0.10)
    ap.add_argument("--earliest", type=float, default=0.0)
    ap.add_argument("--latest", type=float, default=300.0)
    args = ap.parse_args()

    cfg = ArbPolicy(
        threshold=args.threshold,
        fill_size=args.fill_size,
        persist_ticks=args.persist,
        cooldown_s=args.cooldown,
        max_trades_per_market=args.max_trades,
        max_arb_depth=args.max_depth,
        earliest_seconds=args.earliest,
        latest_seconds=args.latest,
    )

    days = sorted([d for d in PRICES_5M_ROOT.iterdir() if d.is_dir()])
    if args.max_days:
        days = days[-args.max_days:]
    paths = []
    for d in days:
        csv_dir = d / "csv"
        if csv_dir.exists():
            paths.extend(str(p) for p in sorted(csv_dir.glob("*.csv")))

    print(f"[arb] paths: {len(paths)}  threshold={cfg.threshold}  size={cfg.fill_size}  persist={cfg.persist_ticks}t cooldown={cfg.cooldown_s}s max_depth={cfg.max_arb_depth}")
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
    print(f"[arb] valid: {len(df)}  ({time.time()-t0:.1f}s)\n")
    if df.empty:
        return 0
    n_days = df["date"].nunique()

    print("=" * 90)
    print(f"CROSS-LEG ARB  threshold={cfg.threshold}  fill_size={cfg.fill_size}  persist={cfg.persist_ticks}  cooldown={cfg.cooldown_s}s  max_depth={cfg.max_arb_depth}")
    print("=" * 90)
    print(f"\n=== ACTIVITY ===")
    print(f"  Markets with detected opps:    {(df['n_opps_detected']>0).sum()}/{len(df)} ({(df['n_opps_detected']>0).mean():.1%})")
    print(f"  Total opps detected:           {df['n_opps_detected'].sum()}")
    print(f"  Total trades executed:         {df['n_trades_executed'].sum()}")
    print(f"  Avg ask_sum at trade:          {df.loc[df['n_trades_executed']>0, 'avg_ask_sum'].mean():.4f}")
    print(f"  Trades per active market:      {df.loc[df['n_trades_executed']>0, 'n_trades_executed'].mean():.1f}")

    print(f"\n=== ECONOMICS ===")
    print(f"  Total cost:                    ${df['total_cost'].sum():.2f}")
    print(f"  Total payoff:                  ${df['total_payoff'].sum():.2f}")
    print(f"  Total fees:                    ${df['total_fees'].sum():.2f}")
    print(f"  Total profit:                  ${df['total_profit'].sum():.2f}")
    print(f"  Per day:                       ${df['total_profit'].sum()/n_days:.2f}")
    if df['n_trades_executed'].sum() > 0:
        print(f"  Avg profit per pair:           ${df['total_profit'].sum()/df['n_trades_executed'].sum():.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
