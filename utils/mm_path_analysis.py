"""
Analizza il "path" del prezzo nei mercati BTC 5-min per capire se sono
oscillanti (visita di entrambi i lati profondi) o direzionali (una singola
gamba che corre verso 0/1).

Per ciascun mercato calcola:
  - up_mid_min, up_mid_max, dn_mid_min, dn_mid_max
  - up_mid_start, up_mid_end
  - "deep_visit_both": l'up_mid è sceso sotto 0.30 E ha superato 0.70
    (cioè ha oscillato tra le due metà profonde)
  - "directional_strength": (range - |end - start|) / range
    -> vicino a 0 = puro trend, vicino a 1 = pura oscillazione
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from utils.mm_analyze_5m import parse_market_csv, PRICES_5M_ROOT  # type: ignore


def path_metrics(wide: pd.DataFrame, market_name: str, date: str) -> Optional[dict]:
    if wide is None or len(wide) < 10:
        return None
    wide = wide.copy()
    for col in ("up_bid", "up_ask", "dn_bid", "dn_ask"):
        wide[col] = wide[col].ffill()
    wide = wide.dropna(subset=["up_bid", "up_ask", "dn_bid", "dn_ask"]).reset_index(
        drop=True
    )
    if len(wide) < 10:
        return None

    up_mid = ((wide["up_bid"] + wide["up_ask"]) * 0.5).to_numpy()
    dn_mid = ((wide["dn_bid"] + wide["dn_ask"]) * 0.5).to_numpy()

    up_min, up_max = float(up_mid.min()), float(up_mid.max())
    dn_min, dn_max = float(dn_mid.min()), float(dn_mid.max())
    up_start, up_end = float(up_mid[0]), float(up_mid[-1])

    up_range = up_max - up_min
    up_directional = abs(up_end - up_start)
    # rapporto di "non-directionalità": range non spiegato dal trend
    if up_range > 1e-6:
        oscillation_ratio = max(0.0, 1.0 - up_directional / up_range)
    else:
        oscillation_ratio = 0.0

    # Ha visitato sia <0.30 che >0.70?
    deep_visit_low = up_min < 0.30
    deep_visit_high = up_max > 0.70
    deep_visit_both = bool(deep_visit_low and deep_visit_high)

    # Per ciascun livello: up_mid è sceso sotto?
    levels = (0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05)
    up_visits = [bool(up_min <= L) for L in levels]
    dn_visits = [bool(dn_min <= L) for L in levels]
    # entrambi visitati?
    both_visits = [a and b for a, b in zip(up_visits, dn_visits)]

    winner = "UP" if up_end > dn_mid[-1] else "DOWN" if dn_mid[-1] > up_end else "TIE"

    return {
        "market_name": market_name,
        "date": date,
        "up_min": up_min,
        "up_max": up_max,
        "dn_min": dn_min,
        "dn_max": dn_max,
        "up_start": up_start,
        "up_end": up_end,
        "up_range": up_range,
        "up_directional": up_directional,
        "oscillation_ratio": oscillation_ratio,
        "deep_visit_low": deep_visit_low,
        "deep_visit_high": deep_visit_high,
        "deep_visit_both": deep_visit_both,
        "winner": winner,
        # un campo per ogni livello
        **{f"up_visit_le_{int(L*100):02d}": v for L, v in zip(levels, up_visits)},
        **{f"dn_visit_le_{int(L*100):02d}": v for L, v in zip(levels, dn_visits)},
        **{f"both_visit_le_{int(L*100):02d}": v for L, v in zip(levels, both_visits)},
    }


def _process_one(path_str: str) -> Optional[dict]:
    p = Path(path_str)
    wide = parse_market_csv(p)
    if wide is None:
        return None
    return path_metrics(wide, p.stem, p.parent.parent.name)


def _collect(root: Path, max_days: Optional[int]) -> list[str]:
    days = sorted([d for d in root.iterdir() if d.is_dir()])
    if max_days:
        days = days[-max_days:]
    paths: list[str] = []
    for d in days:
        csv_dir = d / "csv"
        if csv_dir.exists():
            paths.extend(str(p) for p in sorted(csv_dir.glob("*.csv")))
    return paths


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-days", type=int, default=None)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--out", type=str, default="utils/mm_path_5m.parquet")
    args = ap.parse_args()

    paths = _collect(PRICES_5M_ROOT, args.max_days)
    print(f"[path] paths: {len(paths)}", flush=True)
    t0 = time.time()
    results: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_process_one, p) for p in paths]
        for n, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            if r:
                results.append(r)
            if n % 1000 == 0 or n == len(paths):
                el = time.time() - t0
                print(f"  {n}/{len(paths)} valid={len(results)} ({el:.1f}s)", flush=True)
    df = pd.DataFrame(results)
    print(f"[path] valid: {len(df)} ({time.time()-t0:.1f}s)")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, compression="zstd")

    print("\n" + "=" * 78)
    print("PATH ANALYSIS")
    print("=" * 78)
    n = max(1, len(df))
    print(f"Markets: {len(df)}    Days: {df['date'].nunique()}")
    print()
    print(f"oscillation_ratio (1 - |end-start|/range):")
    print(f"  mean={df['oscillation_ratio'].mean():.3f}  "
          f"median={df['oscillation_ratio'].median():.3f}")
    q = df['oscillation_ratio'].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).values
    print(f"  p10={q[0]:.3f}  p25={q[1]:.3f}  p50={q[2]:.3f}  "
          f"p75={q[3]:.3f}  p90={q[4]:.3f}")
    print()

    # Soglie di oscillation
    for thresh in (0.3, 0.5, 0.7):
        c = int((df['oscillation_ratio'] >= thresh).sum())
        print(f"  Markets oscillation_ratio >= {thresh}: {c} ({c/n:.1%})")
    print()

    print(f"deep_visit_low  (up_min < 0.30):     "
          f"{int(df['deep_visit_low'].sum())} ({df['deep_visit_low'].mean():.1%})")
    print(f"deep_visit_high (up_max > 0.70):     "
          f"{int(df['deep_visit_high'].sum())} ({df['deep_visit_high'].mean():.1%})")
    print(f"deep_visit_both (both <0.30 & >0.70):"
          f"{int(df['deep_visit_both'].sum())} ({df['deep_visit_both'].mean():.1%})")
    print()

    # Per livello: % markets where BOTH up and dn visited <= L
    print("=== % markets where BOTH up_min ≤ L AND dn_min ≤ L ===")
    print("(richiesto perchè la ladder fillerebbe entrambi i lati a quel livello)")
    print(f"  {'level':<8}{'% UP visit':>12}{'% DN visit':>12}{'% BOTH':>12}")
    for L_int in (50, 45, 40, 35, 30, 25, 20, 15, 10, 5):
        up_col = f"up_visit_le_{L_int:02d}"
        dn_col = f"dn_visit_le_{L_int:02d}"
        both_col = f"both_visit_le_{L_int:02d}"
        u = df[up_col].mean()
        d = df[dn_col].mean()
        b = df[both_col].mean()
        print(f"  {L_int/100:<8.2f}{u:>12.1%}{d:>12.1%}{b:>12.1%}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
