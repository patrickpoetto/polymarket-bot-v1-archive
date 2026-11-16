"""
Static maker LIMIT-BUY at price X for entire 5m market window.

Strategy:
  Place a maker bid at $X on UP. Fill when UP_ask touches X (= our maker is hit).
  Place a maker bid at $X on DOWN. Fill when DN_ask touches X.
  Hold to expiry. Get $1 per share if that side wins.

This is the TRUE 'buy at $0.98 in general' strategy.
For each market, for each threshold X:
  - On UP side: did UP_ask ever <= X during market? if yes, we fill at X. Did UP win?
  - On DOWN side: same.

We treat fills on UP and DN as INDEPENDENT trades (we'd run both legs).

Also test variants:
  - Different sizes
  - Different time windows for the limit order to be active (e.g. only first 60s, or first 200s, or any time)
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

from utils.mm_analyze_5m import parse_market_csv, PRICES_5M_ROOT


THRESHOLDS = [0.85, 0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]
# Time windows during which the limit order is ACTIVE (in seconds from market start)
# (start_s, end_s). None=any time.
ACTIVE_WINDOWS = [
    (0, 300, "any"),         # entire market
    (0, 60, "0-60s"),        # only first minute
    (0, 120, "0-120s"),
    (60, 300, "60-300s"),    # after first minute
    (120, 300, "120-300s"),  # last 3 minutes
    (180, 300, "last_120s"),
    (240, 300, "last_60s"),
    (270, 300, "last_30s"),
]
SIZE = 100.0
FEE_RATE = 0.07
REBATE_PCT = 0.20  # maker rebate


def analyze_market(wide: pd.DataFrame, market: str, date: str) -> list[dict]:
    if wide is None or len(wide) < 30:
        return []
    wide = wide.copy()
    for col in ("up_bid", "up_ask", "dn_bid", "dn_ask"):
        wide[col] = wide[col].ffill()
    wide = wide.dropna(subset=["up_bid", "up_ask", "dn_bid", "dn_ask"]).reset_index(drop=True)
    if len(wide) < 30:
        return []

    up_ask = wide["up_ask"].to_numpy(dtype=np.float64)
    dn_ask = wide["dn_ask"].to_numpy(dtype=np.float64)
    up_bid = wide["up_bid"].to_numpy(dtype=np.float64)
    dn_bid = wide["dn_bid"].to_numpy(dtype=np.float64)
    t_ms = wide["t_ms"].to_numpy(dtype=np.int64)
    n = len(wide)
    t_start = int(t_ms[0])

    final_up_mid = (up_bid[-1] + up_ask[-1]) * 0.5
    final_dn_mid = (dn_bid[-1] + dn_ask[-1]) * 0.5
    if final_up_mid > final_dn_mid:
        winner = "UP"
    elif final_dn_mid > final_up_mid:
        winner = "DOWN"
    else:
        winner = "TIE"

    rows = []
    for thr in THRESHOLDS:
        for win_start, win_end, win_label in ACTIVE_WINDOWS:
            start_ms = t_start + win_start * 1000
            end_ms = t_start + win_end * 1000

            for side, ask_arr in (("UP", up_ask), ("DOWN", dn_ask)):
                # Realistic maker post_only logic:
                # 1. Wait for ASK to FIRST be > thr (= post_only valid: we're below ask)
                # 2. After that, when ASK touches/crosses thr (= our maker bid is hit), fill at thr
                # If ASK never exceeds thr in window, we can't place the order → no fill
                placed = False
                fill_idx = None
                for i in range(n):
                    tm = t_ms[i]
                    if tm < start_ms or tm > end_ms:
                        continue
                    a = ask_arr[i]
                    if not placed:
                        # waiting for ASK > thr to be able to post_only
                        if a > thr + 1e-9:
                            placed = True
                        continue
                    # placed: looking for ask to touch/cross thr
                    if a <= thr + 1e-9 and a > 0:
                        fill_idx = i
                        break
                if fill_idx is None:
                    continue
                # Fill price = thr (our maker at thr is hit)
                fill_px = thr
                won = (winner == side)
                tte_s = (t_ms[-1] - t_ms[fill_idx]) / 1000.0
                t_from_start = (t_ms[fill_idx] - t_start) / 1000.0
                rows.append({
                    "market": market, "date": date,
                    "threshold": thr, "win_label": win_label,
                    "side": side, "fill_px": fill_px,
                    "fill_at_s": t_from_start, "tte_s": tte_s,
                    "won": won, "winner": winner,
                    "actual_ask": float(ask_arr[fill_idx]),
                })
    return rows


def _process_one(path_str: str) -> list[dict]:
    p = Path(path_str)
    wide = parse_market_csv(p)
    if wide is None:
        return []
    return analyze_market(wide, p.stem, p.parent.parent.name)


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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-days", type=int, default=30)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    args = ap.parse_args()

    paths = _collect(PRICES_5M_ROOT, args.max_days)
    print(f"[static_buy] paths: {len(paths)}", flush=True)

    t0 = time.time()
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_process_one, p) for p in paths]
        n_done = 0
        for fut in as_completed(futs):
            n_done += 1
            rows.extend(fut.result())
            if n_done % 5000 == 0 or n_done == len(paths):
                print(f"  {n_done}/{len(paths)} ({time.time()-t0:.1f}s)", flush=True)

    df = pd.DataFrame(rows)
    print(f"[static_buy] obs rows: {len(df)} ({time.time()-t0:.1f}s)")
    if df.empty:
        return 0

    n_days = df["date"].nunique()
    print(f"\n  Days: {n_days}\n")

    print("=" * 130)
    print("Static maker LIMIT-BUY at $X for given window. Fill at X. Hold to expiry.")
    print("Per (threshold, window): N fills (= opportunities), win rate, EV per fill, $/day")
    print("=" * 130)
    print(f"{'thr':<7}{'window':<14}{'fills':>8}{'fills/d':>10}{'win%':>8}"
          f"{'EV$/fill':>10}{'EV-fee':>10}{'$/day':>10}{'$/day-fee':>11}{'avg_tte':>9}")

    grp = df.groupby(["threshold", "win_label"], sort=False)
    summary = []
    for (thr, label), g in grp:
        n_fills = len(g)
        per_day = n_fills / max(1, n_days)
        win_rate = g["won"].mean()
        # P&L per fill (we ALWAYS pay 'thr' — maker fill price)
        # win = +(1 - thr) * SIZE ; lose = -thr * SIZE
        ev = win_rate * (1 - thr) * SIZE - (1 - win_rate) * thr * SIZE
        # Maker rebate: rebate = REBATE_PCT * fee_rate * thr * (1-thr) * SIZE
        rebate = REBATE_PCT * FEE_RATE * thr * (1 - thr) * SIZE
        ev_after = ev + rebate  # maker = positive rebate, no fee paid
        daily = ev * per_day
        daily_after = ev_after * per_day
        avg_tte = g["tte_s"].mean()
        print(f"{thr:<7}{label:<14}{n_fills:>8}{per_day:>10.1f}"
              f"{win_rate*100:>7.2f}%{ev:>10.3f}{ev_after:>10.3f}"
              f"{daily:>10.2f}{daily_after:>11.2f}{avg_tte:>9.1f}")
        summary.append({
            "thr": thr, "window": label, "fills": n_fills,
            "fills/d": per_day, "win%": win_rate * 100,
            "ev$/f": ev, "ev-fee": ev_after,
            "$/d": daily, "$/d-fee": daily_after,
            "avg_tte_s": avg_tte,
        })

    print("\n" + "=" * 130)
    print("TOP 15 by $/day (after maker rebate):")
    print("=" * 130)
    sdf = pd.DataFrame(summary)
    sdf2 = sdf[sdf["fills/d"] >= 5].sort_values("$/d-fee", ascending=False).head(15)
    if len(sdf2):
        print(sdf2.to_string(index=False))

    print("\n" + "=" * 130)
    print("Specific case: thr=0.98, window=any (your original question)")
    print("=" * 130)
    sub = df[(df["threshold"] == 0.98) & (df["win_label"] == "any")]
    if len(sub):
        n = len(sub)
        wins = sub[sub["won"]]
        losses = sub[~sub["won"]]
        wr = sub["won"].mean()
        print(f"  Total fills: {n} ({n/n_days:.1f}/day)")
        print(f"  Wins:  {len(wins)} ({wr*100:.2f}%)")
        print(f"  Loss:  {len(losses)} ({(1-wr)*100:.2f}%)")
        print(f"  Avg fill time (s from start): {sub['fill_at_s'].mean():.1f}s")
        print(f"  Avg time-to-expiry at fill:   {sub['tte_s'].mean():.1f}s")
        print(f"  Per-trade EV (no fee):        ${wr * (1-0.98) * SIZE - (1-wr) * 0.98 * SIZE:.3f}")
        rebate = REBATE_PCT * FEE_RATE * 0.98 * 0.02 * SIZE
        print(f"  Maker rebate:                 ${rebate:.4f} per trade")
        print(f"  Daily $:                      ${(wr * (1-0.98) * SIZE - (1-wr) * 0.98 * SIZE) * (n/n_days):.2f}")
        print(f"  Daily $ + rebate:             ${((wr * (1-0.98) * SIZE - (1-wr) * 0.98 * SIZE) + rebate) * (n/n_days):.2f}")

        if len(losses) > 0:
            print(f"\n  Sample losers (first 20):")
            print(losses[["market", "date", "side", "fill_at_s", "tte_s", "winner"]].head(20).to_string(index=False))
            print(f"\n  Loss distribution: avg fill_at_s={losses['fill_at_s'].mean():.1f}s")
            # Histogram of fill_at_s for losses
            import numpy as np
            bins = [0, 30, 60, 120, 180, 240, 300]
            hist, _ = np.histogram(losses["fill_at_s"], bins=bins)
            print(f"  Loss timing histogram:")
            for i, h in enumerate(hist):
                print(f"    [{bins[i]:>3}-{bins[i+1]:>3}s]: {h:>3} losses")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
