"""
Combina il filtro early-direction signal con la STATIC LADDER strategy.

Per ogni mercato:
  - Calcola delta = up_mid_T - up_mid_0  per T in (3, 5, 10, 15)s
  - Bucketizza per |delta| (oscillating vs directional)
  - Misura: % paired, pair cost paired, P&L paired, P&L unpaired, P&L total
  - Trova la threshold ottimale che massimizza P&L netto.

Risultato target: trovare il filtro che skippa quasi tutti gli unpaired
mantenendo quasi tutti i paired.
"""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd

from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--static-parquet", type=str, default="utils/mm_static.parquet")
    ap.add_argument("--signal-parquet", type=str, default="utils/mm_early_signal.parquet")
    args = ap.parse_args()

    static = pd.read_parquet(args.static_parquet)
    signal = pd.read_parquet(args.signal_parquet)

    print(f"Static: {len(static)} markets")
    print(f"Signal: {len(signal)} markets")

    df = static.merge(signal[["market_name", "date", "delta_t3", "delta_t5", "delta_t10",
                              "delta_t15", "range_t3", "range_t5", "range_t10", "range_t15"]],
                      on=["market_name", "date"], how="inner")
    print(f"Merged: {len(df)} markets\n")

    n_days = df["date"].nunique()

    # Distribuzione paired/unpaired
    paired = df["both_sides_filled"]
    print(f"Overall: {paired.sum()}/{len(df)} paired ({paired.mean():.1%})")
    print(f"Pair cost paired: ${df.loc[paired, 'pair_cost'].mean():.4f}")
    print(f"P&L paired: ${df.loc[paired, 'pnl'].sum():.2f} (avg ${df.loc[paired, 'pnl'].mean():.4f})")
    print(f"P&L unpaired: ${df.loc[~paired, 'pnl'].sum():.2f} (avg ${df.loc[~paired, 'pnl'].mean():.4f})")
    print()

    print("=" * 100)
    print("FILTER ANALYSIS: P(paired | |delta_tT| < threshold)")
    print("=" * 100)
    print(f"  N total markets / day: {len(df)/n_days:.1f}")
    print()
    for T in (3, 5, 10, 15):
        col = f"delta_t{T}"
        print(f"\n--- Filter on {col} (skip if |delta| > threshold) ---")
        print(f"  {'thresh':<10}{'kept':>8}{'%kept':>8}{'paired%':>10}{'pair_cost':>12}"
              f"{'$/m kept':>12}{'$/m paired':>12}{'$/d kept':>12}")
        for th in (0.005, 0.010, 0.015, 0.020, 0.030, 0.050, 0.100, 0.200, 1.0):
            mask = df[col].abs() <= th
            sub = df[mask]
            n = len(sub)
            if n == 0:
                continue
            paired_mask = sub["both_sides_filled"]
            n_paired = paired_mask.sum()
            paired_pct = n_paired / n if n > 0 else 0
            pair_cost = sub.loc[paired_mask, "pair_cost"].mean() if n_paired else float("nan")
            pnl_kept = sub["pnl"].sum()
            pnl_per_market = sub["pnl"].mean()
            pnl_per_paired = sub.loc[paired_mask, "pnl"].mean() if n_paired else float("nan")
            pnl_per_day = pnl_kept / n_days
            th_str = f"≤{th*100:.1f}c" if th < 1 else "ALL"
            print(f"  {th_str:<10}{n:>8}{n/len(df):>7.0%}"
                  f"{paired_pct:>10.0%}{pair_cost:>12.4f}{pnl_per_market:>12.4f}"
                  f"{pnl_per_paired:>12.4f}{pnl_per_day:>12.2f}")

    # Combined filter: small |delta| AND moderate range (= oscillating)
    print()
    print("=" * 100)
    print("COMBINED FILTER: |delta_t10| < th_d AND range_t10 > th_r (oscillating)")
    print("=" * 100)
    print(f"  {'th_d':<8}{'th_r':<8}{'kept':>8}{'%kept':>8}{'paired%':>10}"
          f"{'pair_cost':>12}{'$/m':>10}{'$/d':>10}")
    for th_d in (0.005, 0.010, 0.020, 0.030):
        for th_r in (0.01, 0.02, 0.03, 0.05):
            mask = (df["delta_t10"].abs() <= th_d) & (df["range_t10"] >= th_r)
            sub = df[mask]
            n = len(sub)
            if n < 50:
                continue
            paired_mask = sub["both_sides_filled"]
            n_paired = paired_mask.sum()
            paired_pct = n_paired / n if n > 0 else 0
            pair_cost = sub.loc[paired_mask, "pair_cost"].mean() if n_paired else float("nan")
            pnl_kept = sub["pnl"].sum()
            pnl_per_day = pnl_kept / n_days
            th_d_str = f"{th_d*100:.1f}c"
            th_r_str = f">{th_r*100:.0f}c"
            print(f"  {th_d_str:<8}{th_r_str:<8}{n:>8}{n/len(df):>7.0%}"
                  f"{paired_pct:>10.0%}{pair_cost:>12.4f}{sub['pnl'].mean():>10.4f}{pnl_per_day:>10.2f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
