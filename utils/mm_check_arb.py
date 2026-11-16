"""
Check for cross-leg arbitrage opportunities.

Arb #1: ask_sum = UP_ask + DN_ask < 1
  If we taker-buy both at ASK, total cost < $1, payoff = $1 (one wins).
  Profit per pair (before fees) = 1 - ask_sum.
  Fees: fee_rate * UP_ask*(1-UP_ask) + fee_rate * DN_ask*(1-DN_ask).

Arb #2: bid_sum > 1 (sell both at bid IF you have inventory)
  Less actionable since requires holding shares.
"""
from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import os, time
import numpy as np
import pandas as pd

from utils.mm_analyze_5m import parse_market_csv, PRICES_5M_ROOT

FEE_RATE = 0.07
TICK = 0.01


def _scan_market(path_str: str) -> dict | None:
    p = Path(path_str)
    wide = parse_market_csv(p)
    if wide is None or len(wide) < 30:
        return None
    for col in ("up_bid", "up_ask", "dn_bid", "dn_ask"):
        wide[col] = wide[col].ffill()
    wide = wide.dropna(subset=["up_bid", "up_ask", "dn_bid", "dn_ask"])
    if len(wide) < 30:
        return None

    ua = wide["up_ask"].to_numpy()
    da = wide["dn_ask"].to_numpy()
    ub = wide["up_bid"].to_numpy()
    db = wide["dn_bid"].to_numpy()
    ask_sum = ua + da
    bid_sum = ub + db

    n = len(wide)
    arb_ticks_98 = int((ask_sum < 0.98).sum())
    arb_ticks_99 = int((ask_sum < 0.99).sum())
    arb_ticks_995 = int((ask_sum < 0.995).sum())

    bid_arb_101 = int((bid_sum > 1.01).sum())
    bid_arb_1005 = int((bid_sum > 1.005).sum())

    min_ask_sum = float(ask_sum.min())
    max_bid_sum = float(bid_sum.max())

    # Estimate net profit if we took ALL ask_sum < 0.99 opps
    profit_pre_fee = 0.0
    fees = 0.0
    nopps = 0
    for i in range(n):
        if ask_sum[i] < 0.99 and ua[i] > 0 and da[i] > 0:
            profit_pre_fee += 1.0 - ask_sum[i]
            fees += FEE_RATE * ua[i] * (1 - ua[i]) + FEE_RATE * da[i] * (1 - da[i])
            nopps += 1

    return dict(
        market=p.stem,
        date=p.parent.parent.name,
        n_ticks=n,
        arb_ticks_98=arb_ticks_98,
        arb_ticks_99=arb_ticks_99,
        arb_ticks_995=arb_ticks_995,
        min_ask_sum=min_ask_sum,
        max_bid_sum=max_bid_sum,
        bid_arb_101=bid_arb_101,
        bid_arb_1005=bid_arb_1005,
        profit_pre_fee_99=profit_pre_fee,
        fees_99=fees,
        net_99=profit_pre_fee - fees,
        nopps_99=nopps,
    )


def main(max_days=30):
    days = sorted([d for d in PRICES_5M_ROOT.iterdir() if d.is_dir()])
    if max_days:
        days = days[-max_days:]
    paths = []
    for d in days:
        csv_dir = d / "csv"
        if csv_dir.exists():
            paths.extend(str(p) for p in sorted(csv_dir.glob("*.csv")))

    print(f"[arb_check] paths: {len(paths)}")
    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 4) - 1)) as ex:
        futs = [ex.submit(_scan_market, p) for p in paths]
        for f in as_completed(futs):
            r = f.result()
            if r:
                results.append(r)

    df = pd.DataFrame(results)
    n_days = df["date"].nunique()
    print(f"\nMarkets: {len(df)}  Days: {n_days}  ({time.time()-t0:.1f}s)\n")

    print("=== ASK ARB (UP_ask + DN_ask < threshold) ===")
    print(f"  Min ask_sum across all markets:     {df['min_ask_sum'].min():.4f}")
    print(f"  Median min ask_sum per market:      {df['min_ask_sum'].median():.4f}")
    print(f"  Markets with ask_sum < 0.995:       {(df['arb_ticks_995']>0).sum()} ({(df['arb_ticks_995']>0).mean():.1%})")
    print(f"  Markets with ask_sum < 0.99:        {(df['arb_ticks_99']>0).sum()} ({(df['arb_ticks_99']>0).mean():.1%})")
    print(f"  Markets with ask_sum < 0.98:        {(df['arb_ticks_98']>0).sum()} ({(df['arb_ticks_98']>0).mean():.1%})")

    print("\n=== BID ARB (UP_bid + DN_bid > threshold) — requires inventory! ===")
    print(f"  Max bid_sum across all markets:     {df['max_bid_sum'].max():.4f}")
    print(f"  Median max bid_sum per market:      {df['max_bid_sum'].median():.4f}")
    print(f"  Markets with bid_sum > 1.005:       {(df['bid_arb_1005']>0).sum()} ({(df['bid_arb_1005']>0).mean():.1%})")
    print(f"  Markets with bid_sum > 1.01:        {(df['bid_arb_101']>0).sum()} ({(df['bid_arb_101']>0).mean():.1%})")

    print("\n=== POTENTIAL ASK-ARB PROFIT (ask_sum<0.99, taker both) ===")
    print(f"  Total opps:                ${df['nopps_99'].sum():.0f}")
    print(f"  Total profit pre-fee:      ${df['profit_pre_fee_99'].sum():.2f}")
    print(f"  Total fees if taker:       ${df['fees_99'].sum():.2f}")
    print(f"  Total net (per share unit): ${df['net_99'].sum():.2f}")
    print(f"  Per day:                   ${df['net_99'].sum()/n_days:.2f}")

    print("\n=== Per-market opportunity counts ===")
    print(df[['arb_ticks_995','arb_ticks_99','arb_ticks_98']].describe().to_string())


if __name__ == "__main__":
    main(max_days=30)
