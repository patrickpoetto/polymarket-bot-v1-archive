"""
Analizza la distribuzione di bid_sum (UP_bid + DN_bid) nel tempo,
per capire quanto spesso il mercato offre 'coppie sotto $1'.
"""
from __future__ import annotations
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import pandas as pd

from utils.mm_analyze_5m import parse_market_csv, PRICES_5M_ROOT


def market_bidsum_stats(path_str: str) -> dict | None:
    p = Path(path_str)
    wide = parse_market_csv(p)
    if wide is None or len(wide) < 30:
        return None
    for col in ("up_bid", "dn_bid"):
        wide[col] = wide[col].ffill()
    wide = wide.dropna(subset=["up_bid", "dn_bid"]).reset_index(drop=True)
    if len(wide) < 30:
        return None

    t_ms = wide["t_ms"].to_numpy()
    elapsed = (t_ms - t_ms[0]) / 1000.0
    bid_sum = wide["up_bid"].to_numpy() + wide["dn_bid"].to_numpy()

    return {
        "market": p.stem, "date": p.parent.parent.name,
        "n_ticks": len(wide),
        "duration": float(elapsed[-1]),
        "open_bidsum": float(bid_sum[0]),
        "min_bidsum": float(bid_sum.min()),
        "max_bidsum": float(bid_sum.max()),
        "mean_bidsum": float(bid_sum.mean()),
        "median_bidsum": float(np.median(bid_sum)),
        "pct_below_100": float((bid_sum < 1.00).mean() * 100),
        "pct_below_99":  float((bid_sum < 0.99).mean() * 100),
        "pct_below_98":  float((bid_sum < 0.98).mean() * 100),
        "pct_below_97":  float((bid_sum < 0.97).mean() * 100),
        "pct_below_95":  float((bid_sum < 0.95).mean() * 100),
        "pct_below_90":  float((bid_sum < 0.90).mean() * 100),
        # Time spent below thresholds (in seconds, considering tick gaps)
        "secs_below_99": float(((bid_sum[1:] < 0.99) * np.diff(elapsed)).sum()) if len(elapsed) > 1 else 0.0,
        "secs_below_98": float(((bid_sum[1:] < 0.98) * np.diff(elapsed)).sum()) if len(elapsed) > 1 else 0.0,
        "secs_below_97": float(((bid_sum[1:] < 0.97) * np.diff(elapsed)).sum()) if len(elapsed) > 1 else 0.0,
        "secs_below_95": float(((bid_sum[1:] < 0.95) * np.diff(elapsed)).sum()) if len(elapsed) > 1 else 0.0,
    }


def main() -> int:
    days = sorted([d for d in PRICES_5M_ROOT.iterdir() if d.is_dir()])[-30:]
    paths = []
    for d in days:
        csv_dir = d / "csv"
        if csv_dir.exists():
            paths.extend(str(p) for p in sorted(csv_dir.glob("*.csv")))

    print(f"Markets: {len(paths)}  Days: {len(days)}", flush=True)
    t0 = time.time()
    results = []
    n_done = 0
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 4) - 1)) as ex:
        futs = [ex.submit(market_bidsum_stats, p) for p in paths]
        for fut in as_completed(futs):
            n_done += 1
            r = fut.result()
            if r is not None:
                results.append(r)
            if n_done % 10000 == 0:
                print(f"  {n_done}/{len(paths)} ({time.time()-t0:.1f}s)", flush=True)

    df = pd.DataFrame(results)
    print(f"\nValid markets: {len(df)} ({time.time()-t0:.1f}s)\n")

    print("=" * 80)
    print("BID-SUM DISTRIBUTION ACROSS MARKETS")
    print("=" * 80)
    print(f"\nOPEN bid_sum (at t=0):")
    print(f"  Mean:   ${df['open_bidsum'].mean():.4f}")
    print(f"  Median: ${df['open_bidsum'].median():.4f}")
    print(f"  Min:    ${df['open_bidsum'].min():.4f}")
    print(f"  Max:    ${df['open_bidsum'].max():.4f}")
    print(f"  % open < $1.00: {(df['open_bidsum'] < 1.00).mean()*100:.1f}%")
    print(f"  % open < $0.99: {(df['open_bidsum'] < 0.99).mean()*100:.1f}%")
    print(f"  % open < $0.98: {(df['open_bidsum'] < 0.98).mean()*100:.1f}%")

    print(f"\nMIN bid_sum during market:")
    print(f"  Mean:   ${df['min_bidsum'].mean():.4f}")
    print(f"  Median: ${df['min_bidsum'].median():.4f}")
    print(f"  P25:    ${df['min_bidsum'].quantile(0.25):.4f}")
    print(f"  P75:    ${df['min_bidsum'].quantile(0.75):.4f}")
    print(f"  Min:    ${df['min_bidsum'].min():.4f}")

    print(f"\nFRACTION OF TIME bid_sum < threshold (mean across markets):")
    print(f"  < $1.00: {df['pct_below_100'].mean():.1f}%")
    print(f"  < $0.99: {df['pct_below_99'].mean():.1f}%")
    print(f"  < $0.98: {df['pct_below_98'].mean():.1f}%")
    print(f"  < $0.97: {df['pct_below_97'].mean():.1f}%")
    print(f"  < $0.95: {df['pct_below_95'].mean():.1f}%")
    print(f"  < $0.90: {df['pct_below_90'].mean():.1f}%")

    print(f"\nMARKETS WHERE bid_sum WAS BELOW threshold AT LEAST ONCE:")
    print(f"  Ever < $1.00: {(df['min_bidsum'] < 1.00).mean()*100:.1f}%")
    print(f"  Ever < $0.99: {(df['min_bidsum'] < 0.99).mean()*100:.1f}%")
    print(f"  Ever < $0.98: {(df['min_bidsum'] < 0.98).mean()*100:.1f}%")
    print(f"  Ever < $0.97: {(df['min_bidsum'] < 0.97).mean()*100:.1f}%")
    print(f"  Ever < $0.95: {(df['min_bidsum'] < 0.95).mean()*100:.1f}%")
    print(f"  Ever < $0.90: {(df['min_bidsum'] < 0.90).mean()*100:.1f}%")

    print(f"\nTIME SPENT BELOW threshold (mean seconds per market, out of 300s):")
    print(f"  Below $0.99: {df['secs_below_99'].mean():.1f}s ({df['secs_below_99'].mean()/300*100:.1f}%)")
    print(f"  Below $0.98: {df['secs_below_98'].mean():.1f}s ({df['secs_below_98'].mean()/300*100:.1f}%)")
    print(f"  Below $0.97: {df['secs_below_97'].mean():.1f}s ({df['secs_below_97'].mean()/300*100:.1f}%)")
    print(f"  Below $0.95: {df['secs_below_95'].mean():.1f}s ({df['secs_below_95'].mean()/300*100:.1f}%)")

    Path("utils/mm_bidsum_dist.parquet").parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet("utils/mm_bidsum_dist.parquet", compression="zstd")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
