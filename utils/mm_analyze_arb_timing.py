"""Distribution of cross-leg arb opportunities by time-in-market and depth."""
from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import os, time
import numpy as np
import pandas as pd

from utils.mm_analyze_5m import parse_market_csv, PRICES_5M_ROOT

THRESHOLD = 0.99


def _scan(path_str: str):
    p = Path(path_str)
    wide = parse_market_csv(p)
    if wide is None or len(wide) < 30:
        return None
    for col in ("up_bid", "up_ask", "dn_bid", "dn_ask"):
        wide[col] = wide[col].ffill()
    wide = wide.dropna(subset=["up_bid", "up_ask", "dn_bid", "dn_ask"]).reset_index(drop=True)
    if len(wide) < 30:
        return None
    ua = wide["up_ask"].to_numpy(); da = wide["dn_ask"].to_numpy()
    t_ms = wide["t_ms"].to_numpy()
    t_start = int(t_ms[0])
    ask_sum = ua + da
    rows = []
    persist = 0
    for i in range(len(wide)):
        if ua[i] <= 0 or da[i] <= 0:
            persist = 0; continue
        if ask_sum[i] < THRESHOLD:
            persist += 1
            elapsed = (t_ms[i] - t_start) / 1000.0
            rows.append((elapsed, ask_sum[i], persist, ua[i], da[i]))
        else:
            persist = 0
    return rows


def main(max_days=30):
    days = sorted([d for d in PRICES_5M_ROOT.iterdir() if d.is_dir()])[-max_days:]
    paths = []
    for d in days:
        csv_dir = d / "csv"
        if csv_dir.exists():
            paths.extend(str(p) for p in sorted(csv_dir.glob("*.csv")))

    print(f"[arb_timing] paths: {len(paths)}")
    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 4) - 1)) as ex:
        futs = [ex.submit(_scan, p) for p in paths]
        for f in as_completed(futs):
            r = f.result()
            if r:
                rows.extend(r)

    df = pd.DataFrame(rows, columns=["elapsed_s", "ask_sum", "persist", "up_ask", "dn_ask"])
    print(f"\nTotal opp ticks: {len(df):,}  ({time.time()-t0:.1f}s)\n")

    print("=== ASK_SUM DISTRIBUTION ===")
    print(df["ask_sum"].describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99]).to_string())

    print("\n=== TIME IN MARKET (when do opps happen?) ===")
    bins = [0, 30, 60, 120, 180, 240, 270, 290, 300]
    df["bucket"] = pd.cut(df["elapsed_s"], bins=bins, right=False)
    print(df.groupby("bucket")["ask_sum"].agg(["count", "mean"]).to_string())
    print(f"\nMean elapsed at opp: {df['elapsed_s'].mean():.1f}s")
    print(f"Median elapsed at opp: {df['elapsed_s'].median():.1f}s")

    print("\n=== PERSISTENCE DISTRIBUTION (how long do opps last?) ===")
    print(df["persist"].describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99]).to_string())
    pct_persist = (df.groupby("persist").size() / len(df)).cumsum()
    print("\nCumul. % of opps that last >= N ticks (= 100ms each):")
    for p in [1, 2, 3, 5, 10, 20, 50]:
        n_atleast = (df["persist"] >= p).sum()
        print(f"  >= {p:>3} ticks ({p*100}ms):  {n_atleast:>7}  ({n_atleast/len(df):.1%})")

    print("\n=== ARB DEPTH DISTRIBUTION (1 - ask_sum) ===")
    df["depth"] = 1.0 - df["ask_sum"]
    print(df["depth"].describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99]).to_string())
    print(f"\nDepth bands:")
    for lo, hi in [(0.001, 0.01), (0.01, 0.02), (0.02, 0.03), (0.03, 0.05), (0.05, 0.10), (0.10, 0.50), (0.50, 1.0)]:
        sub = df[(df["depth"] >= lo) & (df["depth"] < hi)]
        print(f"  {lo:.3f} - {hi:.3f}:  {len(sub):>7} ({len(sub)/len(df):.1%})  avg={sub['depth'].mean():.4f}")


if __name__ == "__main__":
    main(max_days=30)
