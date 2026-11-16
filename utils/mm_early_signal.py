"""
Quanto è affidabile il segnale direzionale dei primi N secondi?

Per ciascun mercato 5-min:
  - up_mid_t0  = mid all'apertura (~0.50)
  - up_mid_tN  = mid a t=N secondi (3, 5, 10, 15)
  - delta = up_mid_tN - up_mid_t0
  - winner finale (UP / DOWN)

Misure aggregate:
  - P(UP vince | delta > +threshold)  — probabilità che UP vinca dato un
    movimento positivo di mid nel primo N secondi
  - P(UP vince | delta < -threshold)
  - Counts in ogni bucket

Se P(UP wins | delta > +c) > 0.55 per qualche c, allora il segnale è
sufficiente per una strategia maker direzionale single-leg.
"""

from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from utils.mm_analyze_5m import parse_market_csv, PRICES_5M_ROOT  # type: ignore


# Tempi di osservazione (secondi)
OBS_TIMES = (3, 5, 10, 15, 30)


def market_metrics(wide: pd.DataFrame, market_name: str, date: str) -> Optional[dict]:
    if wide is None or len(wide) < 30:
        return None
    wide = wide.copy()
    for col in ("up_bid", "up_ask", "dn_bid", "dn_ask"):
        wide[col] = wide[col].ffill()
    wide = wide.dropna(subset=["up_bid", "up_ask", "dn_bid", "dn_ask"]).reset_index(drop=True)
    if len(wide) < 30:
        return None

    up_mid = (wide["up_bid"] + wide["up_ask"]) * 0.5
    dn_mid = (wide["dn_bid"] + wide["dn_ask"]) * 0.5
    t_ms = wide["t_ms"].to_numpy(dtype=np.int64)
    t_start = int(t_ms[0])
    elapsed_s = (t_ms - t_start) / 1000.0

    up_t0 = float(up_mid.iloc[0])

    final_up = float(up_mid.iloc[-1])
    final_dn = float(dn_mid.iloc[-1])
    if final_up > final_dn:
        winner = "UP"
    elif final_dn > final_up:
        winner = "DOWN"
    else:
        winner = "TIE"

    out = {"market_name": market_name, "date": date, "up_t0": up_t0, "winner": winner}

    for T in OBS_TIMES:
        # trova l'indice del primo tick con elapsed >= T
        idx = np.searchsorted(elapsed_s, T)
        if idx >= len(elapsed_s):
            idx = len(elapsed_s) - 1
        up_mid_T = float(up_mid.iloc[idx])
        out[f"up_mid_t{T}"] = up_mid_T
        out[f"delta_t{T}"] = up_mid_T - up_t0
        # range nei primi T secondi
        sub_up = up_mid.iloc[: idx + 1]
        out[f"range_t{T}"] = float(sub_up.max() - sub_up.min()) if len(sub_up) >= 2 else 0.0
        out[f"std_t{T}"] = float(sub_up.std()) if len(sub_up) >= 2 else 0.0

    return out


def _process_one(path_str: str) -> Optional[dict]:
    p = Path(path_str)
    wide = parse_market_csv(p)
    if wide is None:
        return None
    return market_metrics(wide, p.stem, p.parent.parent.name)


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


def analyze(df: pd.DataFrame) -> None:
    n = len(df)
    print(f"Markets: {n}    Winner UP: {(df['winner']=='UP').mean():.1%}")
    print()

    print("=" * 88)
    print("DIRECTION SIGNAL: P(UP wins | delta_tN > threshold)")
    print("=" * 88)
    thresholds = (0.005, 0.010, 0.015, 0.020, 0.030, 0.050)

    for T in OBS_TIMES:
        col = f"delta_t{T}"
        print(f"\n--- t={T}s ---")
        print(f"  {'thresh':<10}{'count up':>12}{'P(up wins|delta>+)':>22}"
              f"{'count dn':>12}{'P(dn wins|delta<-)':>22}")
        for th in thresholds:
            up_mask = df[col] > th
            dn_mask = df[col] < -th
            up_n = int(up_mask.sum())
            dn_n = int(dn_mask.sum())
            up_winrate = (df[up_mask]["winner"] == "UP").mean() if up_n > 0 else float("nan")
            dn_winrate = (df[dn_mask]["winner"] == "DOWN").mean() if dn_n > 0 else float("nan")
            print(f"  ±{th*100:>4.1f}c    {up_n:>12}{up_winrate:>22.2%}{dn_n:>12}{dn_winrate:>22.2%}")

    # Bucket: |delta| < threshold (neutral/oscillating)
    print()
    print("=" * 88)
    print("NEUTRAL ZONE: P(UP wins | |delta_tN| < threshold)")
    print("=" * 88)
    for T in OBS_TIMES:
        col = f"delta_t{T}"
        print(f"\n--- t={T}s ---")
        print(f"  {'thresh':<10}{'count':>10}{'P(up wins)':>14}{'P(dn wins)':>14}")
        for th in (0.005, 0.010, 0.020):
            mask = df[col].abs() < th
            n_n = int(mask.sum())
            if n_n == 0:
                print(f"  <{th*100:>4.1f}c     {n_n:>10}{'-':>14}{'-':>14}")
                continue
            up_winrate = (df[mask]["winner"] == "UP").mean()
            dn_winrate = (df[mask]["winner"] == "DOWN").mean()
            print(f"  <{th*100:>4.1f}c     {n_n:>10}{up_winrate:>14.2%}{dn_winrate:>14.2%}")

    # Range filter
    print()
    print("=" * 88)
    print("VOLATILITY FILTER: distribution of range_tN (cents)")
    print("=" * 88)
    for T in OBS_TIMES:
        col = f"range_t{T}"
        if col not in df.columns:
            continue
        rg = df[col] * 100  # in cents
        q = rg.quantile([0.25, 0.50, 0.75, 0.90, 0.99]).values
        print(
            f"  t={T:>3}s  median={q[1]:.2f}c  p25={q[0]:.2f}c  p75={q[2]:.2f}c  "
            f"p90={q[3]:.2f}c  p99={q[4]:.2f}c"
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-days", type=int, default=None)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--out", type=str, default="utils/mm_early_signal.parquet")
    args = ap.parse_args()

    paths = _collect(PRICES_5M_ROOT, args.max_days)
    print(f"[signal] paths: {len(paths)}", flush=True)
    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_process_one, p) for p in paths]
        for n, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            if r:
                results.append(r)
            if n % 2000 == 0 or n == len(paths):
                print(f"  {n}/{len(paths)} valid={len(results)} ({time.time()-t0:.1f}s)", flush=True)

    df = pd.DataFrame(results)
    print(f"\n[signal] valid: {len(df)} ({time.time()-t0:.1f}s)\n")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, compression="zstd")

    analyze(df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
