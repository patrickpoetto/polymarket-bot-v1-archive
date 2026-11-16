"""
Test della HP di liquidità: i prezzi sono RUMORE in low-vol, INFORMATIVI in high-vol.

Proxy di regime di liquidità (senza dati di volume esplicito):
  - Spread medio: ask - bid (wide = thin book)
  - BBO tick rate: BBO changes / second
  - Price impact follow-through: dopo un move di X cent, dopo Y secondi è ancora lì?

Per ogni mercato, calcoliamo:
  1. Proxy di liquidità (avg spread, tick rate)
  2. Per ogni movimento "significativo" del mid (>= 5c in 20s):
     - Persistence ratio: (mid 30s dopo - mid pre-move) / move iniziale
     - = 1.0 → mantenuto integralmente (= informato)
     - = 0.0 → completamente revertito (= noise)
     - < 0 → over-reversion
  3. Predict winner basandoci sul mid finale-30s: accuracy correla col proxy?

Output: stats aggregate per "tier" di liquidità.
"""
from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import os, time
from typing import Optional

import numpy as np
import pandas as pd

from utils.mm_analyze_5m import parse_market_csv, PRICES_5M_ROOT


def _analyze_market(path_str: str) -> Optional[dict]:
    p = Path(path_str)
    wide = parse_market_csv(p)
    if wide is None or len(wide) < 30:
        return None
    for col in ("up_bid", "up_ask", "dn_bid", "dn_ask"):
        wide[col] = wide[col].ffill()
    wide = wide.dropna(subset=["up_bid", "up_ask", "dn_bid", "dn_ask"]).reset_index(drop=True)
    if len(wide) < 100:
        return None

    ub = wide["up_bid"].to_numpy()
    ua = wide["up_ask"].to_numpy()
    db = wide["dn_bid"].to_numpy()
    da = wide["dn_ask"].to_numpy()
    t_ms = wide["t_ms"].to_numpy()
    n = len(wide)
    duration_s = (t_ms[-1] - t_ms[0]) / 1000.0
    if duration_s < 60:
        return None

    # Proxies di liquidità
    up_spread = ua - ub
    dn_spread = da - db
    avg_spread = float(((up_spread + dn_spread) / 2).mean())

    # BBO change rate
    bbo_changes = int(((ub[1:] != ub[:-1]) | (ua[1:] != ua[:-1]) |
                        (db[1:] != db[:-1]) | (da[1:] != da[:-1])).sum())
    tick_rate = bbo_changes / duration_s

    # Mid prices
    up_mid = (ub + ua) / 2.0
    dn_mid = (db + da) / 2.0

    # Final winner
    if up_mid[-1] > dn_mid[-1]:
        winner = "UP"
    elif dn_mid[-1] > up_mid[-1]:
        winner = "DOWN"
    else:
        winner = "TIE"

    # Find "big moves": find segments where mid moves >= 0.05 in 20s
    persistence_samples = []
    # Find largest move in the market overall (proxy)
    # Use sliding window
    n_persist_30s_idx = max(1, int(30 / (duration_s / n)))  # approx ticks for 30s

    # Big move detection (for UP only — DN is mirror)
    for i in range(0, n, 50):  # sample every 50 ticks to save time
        if i + 200 >= n:
            break
        # Look at change over ~20s (assume ~10 ticks/sec)
        target_ms = t_ms[i] + 20_000
        j = i
        while j < n and t_ms[j] < target_ms:
            j += 1
        if j >= n:
            break
        move = up_mid[j] - up_mid[i]
        if abs(move) < 0.05:
            continue
        # Check persistence at t+50s (30s later)
        target_ms2 = t_ms[j] + 30_000
        k = j
        while k < n and t_ms[k] < target_ms2:
            k += 1
        if k >= n:
            break
        post_move = up_mid[k] - up_mid[i]
        persistence = post_move / move
        persistence_samples.append(persistence)

    persist_mean = float(np.mean(persistence_samples)) if persistence_samples else None
    persist_median = float(np.median(persistence_samples)) if persistence_samples else None
    n_big_moves = len(persistence_samples)

    # Did momentum signal at t=30s correctly predict winner?
    # Simple: at 30s, which side had higher mid?
    target_30s = t_ms[0] + 30_000
    idx_30 = 0
    while idx_30 < n and t_ms[idx_30] < target_30s:
        idx_30 += 1
    if idx_30 >= n:
        idx_30 = n - 1
    if up_mid[idx_30] > dn_mid[idx_30]:
        signal_30s = "UP"
    elif dn_mid[idx_30] > up_mid[idx_30]:
        signal_30s = "DOWN"
    else:
        signal_30s = "TIE"
    signal_correct = (signal_30s == winner and winner != "TIE")

    # Signal strength at 30s (how far from 0.5)
    signal_strength_30s = abs(up_mid[idx_30] - 0.5)

    # Final price range across market
    mid_range = float(up_mid.max() - up_mid.min())

    return {
        "market": p.stem,
        "date": p.parent.parent.name,
        "avg_spread": avg_spread,
        "tick_rate": tick_rate,
        "n_big_moves": n_big_moves,
        "persist_mean": persist_mean,
        "persist_median": persist_median,
        "winner": winner,
        "signal_30s": signal_30s,
        "signal_correct": signal_correct,
        "signal_strength_30s": signal_strength_30s,
        "mid_range": mid_range,
        "duration_s": duration_s,
    }


def main(max_days=30):
    days = sorted([d for d in PRICES_5M_ROOT.iterdir() if d.is_dir()])[-max_days:]
    paths = []
    for d in days:
        csv_dir = d / "csv"
        if csv_dir.exists():
            paths.extend(str(p) for p in sorted(csv_dir.glob("*.csv")))

    print(f"[vol_regime] paths: {len(paths)}")
    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 4) - 1)) as ex:
        futs = [ex.submit(_analyze_market, p) for p in paths]
        for f in as_completed(futs):
            r = f.result()
            if r is not None:
                results.append(r)
    df = pd.DataFrame(results)
    print(f"\nValid markets: {len(df)} ({time.time()-t0:.1f}s)\n")
    if df.empty:
        return

    # Filter to clear winners
    clear = df[df["winner"].isin(["UP", "DOWN"])].copy()
    print(f"Clear-winner markets: {len(clear)}\n")

    # Baseline signal accuracy at 30s
    base_acc = clear["signal_correct"].mean()
    print(f"=== BASELINE: signal accuracy at 30s = {base_acc:.1%}\n")

    # === TIER BY SPREAD ===
    print("=== TIER BY AVG SPREAD (proxy: wide = thin liquidity) ===")
    clear["spread_tier"] = pd.qcut(clear["avg_spread"], q=4, labels=["Q1_tight","Q2","Q3","Q4_wide"], duplicates="drop")
    print(f"{'tier':<12}{'n':>6}{'spread':>10}{'tick_rate':>12}{'signal_acc':>12}{'mid_range':>12}{'persist_med':>14}")
    for tier in clear["spread_tier"].cat.categories:
        sub = clear[clear["spread_tier"]==tier]
        print(f"{tier:<12}{len(sub):>6}{sub['avg_spread'].mean():>10.4f}"
              f"{sub['tick_rate'].mean():>12.2f}{sub['signal_correct'].mean()*100:>11.1f}%"
              f"{sub['mid_range'].mean():>12.4f}{sub['persist_median'].mean():>14.4f}")

    print("\n=== TIER BY TICK RATE (BBO changes/s) ===")
    clear["tick_tier"] = pd.qcut(clear["tick_rate"], q=4, labels=["Q1_slow","Q2","Q3","Q4_fast"], duplicates="drop")
    print(f"{'tier':<12}{'n':>6}{'tick_rate':>12}{'spread':>10}{'signal_acc':>12}{'persist_med':>14}")
    for tier in clear["tick_tier"].cat.categories:
        sub = clear[clear["tick_tier"]==tier]
        print(f"{tier:<12}{len(sub):>6}{sub['tick_rate'].mean():>12.2f}"
              f"{sub['avg_spread'].mean():>10.4f}{sub['signal_correct'].mean()*100:>11.1f}%"
              f"{sub['persist_median'].mean():>14.4f}")

    print("\n=== TIER BY PERSISTENCE (mean reversion proxy) ===")
    has_persist = clear[clear["persist_median"].notna()].copy()
    print(f"  Markets with valid persistence: {len(has_persist)}")
    if len(has_persist) >= 100:
        has_persist["persist_tier"] = pd.qcut(has_persist["persist_median"], q=4,
                                              labels=["Q1_revert","Q2","Q3","Q4_persist"], duplicates="drop")
        print(f"{'tier':<14}{'n':>6}{'persist':>10}{'spread':>10}{'signal_acc':>12}")
        for tier in has_persist["persist_tier"].cat.categories:
            sub = has_persist[has_persist["persist_tier"]==tier]
            print(f"{tier:<14}{len(sub):>6}{sub['persist_median'].mean():>10.4f}"
                  f"{sub['avg_spread'].mean():>10.4f}{sub['signal_correct'].mean()*100:>11.1f}%")

    print("\n=== CROSSED: signal_strength × spread ===")
    # When signal is STRONG (mid far from 0.5), is it more reliable in tight or wide spread?
    clear["strength_tier"] = pd.qcut(clear["signal_strength_30s"], q=4,
                                     labels=["Q1_weak","Q2","Q3","Q4_strong"], duplicates="drop")
    print(f"{'strength':<14}{'spread_tier':<14}{'n':>6}{'acc':>8}")
    for strength in clear["strength_tier"].cat.categories:
        for spread in clear["spread_tier"].cat.categories:
            sub = clear[(clear["strength_tier"]==strength) & (clear["spread_tier"]==spread)]
            if len(sub) < 10:
                continue
            print(f"{strength:<14}{spread:<14}{len(sub):>6}{sub['signal_correct'].mean()*100:>7.1f}%")

    df.to_parquet("utils/mm_volume_regime.parquet", compression="zstd")
    print(f"\nSaved to utils/mm_volume_regime.parquet")


if __name__ == "__main__":
    main(max_days=30)
