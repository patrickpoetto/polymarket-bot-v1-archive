"""
Analyze: 'buy at $0.98' strategy properly.

Strategy concept (taker-buy):
  At time T (= expiry - X seconds), look at both sides' ASK.
  If one side's ASK is in a narrow band [low, high] (e.g. [0.97, 0.99]),
  buy 100 shares at that ask. Did that side win?

Run for multiple (band, time-to-expiry) combinations.
Compute: opportunities, win rate, avg fill price, EV per opp, $ per day, with/without taker fee.
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


# Bands [low, high] for ASK (we buy taker at the ask if it's in this band)
BANDS = [
    (0.95, 0.96),
    (0.96, 0.97),
    (0.97, 0.98),
    (0.98, 0.99),
    (0.99, 0.995),
    (0.995, 1.00),
    (0.95, 0.99),  # wide band
    (0.97, 0.99),
]
# Sample times = expiry - X seconds
TTE_S = [120, 60, 30, 15, 10, 5, 2]
SIZE = 100.0
FEE_RATE = 0.07


def _state_at(t_ms_arr: np.ndarray, vals: np.ndarray, target_ms: int) -> Optional[float]:
    # Last value with t_ms <= target_ms
    idx = int(np.searchsorted(t_ms_arr, target_ms, side="right")) - 1
    if idx < 0:
        return None
    return float(vals[idx])


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
    t_end = int(t_ms[-1])

    final_up_mid = (up_bid[-1] + up_ask[-1]) * 0.5
    final_dn_mid = (dn_bid[-1] + dn_ask[-1]) * 0.5
    if final_up_mid > final_dn_mid:
        winner = "UP"
    elif final_dn_mid > final_up_mid:
        winner = "DOWN"
    else:
        winner = "TIE"

    rows = []
    for tte in TTE_S:
        target_ms = t_end - tte * 1000
        ua = _state_at(t_ms, up_ask, target_ms)
        da = _state_at(t_ms, dn_ask, target_ms)
        if ua is None or da is None:
            continue
        # For each band, check if either side's ask falls in band; if yes, that's an opportunity
        for band_lo, band_hi in BANDS:
            for side, ask in (("UP", ua), ("DOWN", da)):
                if band_lo - 1e-9 <= ask <= band_hi + 1e-9:
                    won = (winner == side)
                    rows.append({
                        "market": market, "date": date,
                        "tte_s": tte,
                        "band_lo": band_lo, "band_hi": band_hi,
                        "side": side, "ask": ask, "won": won, "winner": winner,
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
    print(f"[high_price] paths: {len(paths)}", flush=True)

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
    print(f"[high_price] obs rows: {len(df)} ({time.time()-t0:.1f}s)")
    if df.empty:
        return 0

    n_days = df["date"].nunique()
    print(f"\n  Days: {n_days}")
    print(f"  Note: if BOTH sides are in a band at same time, both count as opportunities (rare).\n")

    print("=" * 130)
    print("Per (band, tte) — TAKER buy at the ask in band, 100 shares.")
    print("opps = # of times any side was in band at exactly time T-tte")
    print("=" * 130)
    print(f"{'band':<14}{'tte_s':<7}{'opps':>8}{'opps/day':>10}{'win%':>9}{'avg_px':>10}"
          f"{'EV$/opp':>10}{'EV-fee':>10}{'$/day':>10}{'$/day-fee':>11}")

    grp = df.groupby(["band_lo", "band_hi", "tte_s"], sort=False)
    summary = []
    for (lo, hi, tte), g in grp:
        n_opps = len(g)
        opps_per_day = n_opps / max(1, n_days)
        win_rate = g["won"].mean()
        avg_px = g["ask"].mean()
        pnl_per_opp = np.where(
            g["won"].to_numpy(),
            (1 - g["ask"].to_numpy()) * SIZE,
            -g["ask"].to_numpy() * SIZE,
        )
        ev = pnl_per_opp.mean()
        fee_per_opp = FEE_RATE * g["ask"].to_numpy() * (1 - g["ask"].to_numpy()) * SIZE
        ev_after = (pnl_per_opp - fee_per_opp).mean()
        daily = ev * opps_per_day
        daily_after = ev_after * opps_per_day
        band_label = f"[{lo:.3f},{hi:.3f}]"
        print(f"{band_label:<14}{tte:<7}{n_opps:>8}{opps_per_day:>10.1f}"
              f"{win_rate*100:>8.2f}%{avg_px:>10.4f}"
              f"{ev:>10.3f}{ev_after:>10.3f}{daily:>10.2f}{daily_after:>11.2f}")
        summary.append({
            "band": band_label, "tte": tte, "opps": n_opps,
            "opps/day": opps_per_day, "win%": win_rate * 100,
            "avg_px": avg_px, "ev$/opp": ev, "ev-fee": ev_after,
            "$/day": daily, "$/day-fee": daily_after,
        })

    # Show only the most promising (after fees, with reasonable opps/day)
    print("\n" + "=" * 130)
    print("TOP opportunities by daily $ (after fees), opps>=10/day:")
    print("=" * 130)
    sdf = pd.DataFrame(summary)
    sdf2 = sdf[sdf["opps/day"] >= 10].sort_values("$/day-fee", ascending=False).head(15)
    if len(sdf2):
        print(sdf2.to_string(index=False))

    # Detailed examples for the most relevant case: band [0.97, 0.99], tte=30
    sub = df[(df["band_lo"] == 0.97) & (df["band_hi"] == 0.99) & (df["tte_s"] == 30)]
    if len(sub):
        print("\n" + "=" * 130)
        print(f"Detail @ band [0.97, 0.99], tte=30s")
        print("=" * 130)
        n = len(sub)
        wins = sub[sub["won"]]
        losses = sub[~sub["won"]]
        print(f"  Total: {n} opportunities ({n/n_days:.1f}/day)")
        print(f"  Wins:  {len(wins)} ({len(wins)/n*100:.2f}%)  avg_px=${wins['ask'].mean():.4f}")
        print(f"  Loss:  {len(losses)} ({len(losses)/n*100:.2f}%) avg_px=${losses['ask'].mean():.4f}")
        if len(losses):
            print(f"  Sample losers (first 15):")
            print(losses[["market", "date", "side", "ask", "winner"]].head(15).to_string(index=False))
            # If we always paid avg_px and won at win_rate, what's the simple math?
            avg = sub["ask"].mean()
            wr = sub["won"].mean()
            ev_calc = wr * (1 - avg) - (1 - wr) * avg
            print(f"  Per-share EV (no fee): {ev_calc*100:.3f}c    Per-trade(100): ${ev_calc*100:.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
