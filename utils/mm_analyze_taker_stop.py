"""
TAKER buy at $0.98 (~within band) + STOP-LOSS at $X.

Strategy:
  - Watch market. First time ASK <= TAKER_TRIGGER_HIGH and ASK >= TAKER_TRIGGER_LOW (band):
    - Taker buy 100 shares at the actual ASK price (pay current ask).
  - After fill, monitor BID:
    - If BID drops to <= STOP_LOSS, taker-sell 100 shares at the actual BID (lock loss).
    - Otherwise hold to expiry: get $1 if side wins, $0 if loses.

Compare with maker case:
  - For taker, we trigger on ASK first reaching the band, regardless of direction.
  - This includes cases where price is rising through $0.98 toward $1 (= bullish momentum).
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


# Trigger ASK band — when ASK first enters this band, we taker-buy
TRIGGER_BANDS = [
    (0.97, 0.98),
    (0.98, 0.99),
    (0.97, 0.99),
    (0.95, 0.97),
    (0.95, 0.98),
    (0.92, 0.95),
    (0.94, 0.96),
    (0.96, 0.98),
]
# Stop-loss BID levels (= BID at which we sell to lock loss). None = no stop.
STOP_LOSS_LEVELS = [None, 0.90, 0.85, 0.80, 0.75, 0.70, 0.60, 0.50]
SIZE = 100.0
FEE_RATE = 0.07


def _won(winner: str, side: str) -> bool:
    return winner == side


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
    for trig_lo, trig_hi in TRIGGER_BANDS:
        for stop_loss in STOP_LOSS_LEVELS:
            for side, ask_arr, bid_arr in (
                ("UP", up_ask, up_bid),
                ("DOWN", dn_ask, dn_bid),
            ):
                # 1. Find first index where ask is in trigger band → taker buy at ask
                buy_idx = None
                for i in range(n):
                    a = ask_arr[i]
                    if trig_lo - 1e-9 <= a <= trig_hi + 1e-9 and a > 0:
                        buy_idx = i
                        break
                if buy_idx is None:
                    continue
                buy_px = float(ask_arr[buy_idx])
                t_from_start = (t_ms[buy_idx] - t_start) / 1000.0

                # 2. Monitor BID for stop-loss after buy
                stop_idx = None
                if stop_loss is not None:
                    for j in range(buy_idx + 1, n):
                        b = bid_arr[j]
                        if 0 < b <= stop_loss + 1e-9:
                            stop_idx = j
                            break

                won = _won(winner, side)
                if stop_idx is not None:
                    sell_px = float(bid_arr[stop_idx])
                    pnl_per_share = (sell_px - buy_px)
                    fees = FEE_RATE * buy_px * (1 - buy_px) + FEE_RATE * sell_px * (1 - sell_px)
                    outcome = "STOP"
                else:
                    pnl_per_share = (1.0 if won else 0.0) - buy_px
                    fees = FEE_RATE * buy_px * (1 - buy_px)  # only buy fee, no settlement fee
                    outcome = "WIN" if won else "LOSS_HOLD"

                pnl_after_fees = SIZE * pnl_per_share - SIZE * fees
                pnl_no_fees = SIZE * pnl_per_share

                rows.append({
                    "market": market, "date": date,
                    "trig_lo": trig_lo, "trig_hi": trig_hi,
                    "stop_loss": -1.0 if stop_loss is None else stop_loss,
                    "side": side, "buy_px": buy_px,
                    "sell_px": float(bid_arr[stop_idx]) if stop_idx is not None else 0.0,
                    "won": won, "winner": winner,
                    "outcome": outcome,
                    "buy_at_s": t_from_start,
                    "pnl_no_fees": pnl_no_fees,
                    "pnl_after_fees": pnl_after_fees,
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
    print(f"[taker_stop] paths: {len(paths)}", flush=True)

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
    print(f"[taker_stop] obs rows: {len(df)} ({time.time()-t0:.1f}s)")
    if df.empty:
        return 0

    n_days = df["date"].nunique()
    print(f"\n  Days: {n_days}\n")

    # Aggregate per (band, stop_loss)
    print("=" * 140)
    print("TAKER buy first time ASK in trigger band; STOP-LOSS at BID level")
    print("=" * 140)
    print(f"{'band':<14}{'stop':<7}{'fills':>7}{'fills/d':>9}{'win%':>8}"
          f"{'stop%':>8}{'lossH%':>8}{'avg_buy':>9}{'avg_sell':>10}"
          f"{'EV-net':>10}{'$/day':>10}")

    grp = df.groupby(["trig_lo", "trig_hi", "stop_loss"], sort=False)
    summary = []
    for (lo, hi, sl), g in grp:
        n_f = len(g)
        per_d = n_f / max(1, n_days)
        wr = g["won"].mean()
        outcomes = g["outcome"].value_counts(normalize=True)
        stop_pct = outcomes.get("STOP", 0.0) * 100
        loss_hold_pct = outcomes.get("LOSS_HOLD", 0.0) * 100
        avg_buy = g["buy_px"].mean()
        avg_sell = g[g["outcome"] == "STOP"]["sell_px"].mean() if (g["outcome"] == "STOP").any() else 0.0
        ev_net = g["pnl_after_fees"].mean()
        daily = ev_net * per_d
        sl_label = "none" if sl < 0 else f"${sl:.2f}"
        band_label = f"[{lo:.2f},{hi:.2f}]"
        print(f"{band_label:<14}{sl_label:<7}{n_f:>7}{per_d:>9.1f}"
              f"{wr*100:>7.2f}%{stop_pct:>7.1f}%{loss_hold_pct:>7.1f}%"
              f"{avg_buy:>9.4f}{avg_sell:>10.4f}{ev_net:>10.3f}{daily:>10.2f}")
        summary.append({
            "band": band_label, "stop": sl_label, "fills/d": per_d,
            "win%": wr * 100, "stop%": stop_pct, "loss_hold%": loss_hold_pct,
            "avg_buy": avg_buy, "avg_sell": avg_sell,
            "ev_net": ev_net, "$/day": daily,
        })

    print("\n" + "=" * 140)
    print("TOP 15 by $/day (after fees), fills/d >= 5:")
    print("=" * 140)
    sdf = pd.DataFrame(summary)
    sdf2 = sdf[sdf["fills/d"] >= 5].sort_values("$/day", ascending=False).head(15)
    if len(sdf2):
        print(sdf2.to_string(index=False))

    # Specific case: trigger [0.97, 0.98], various stop-losses
    print("\n" + "=" * 140)
    print("Detail: trigger band [0.97, 0.98] with various stop-losses")
    print("=" * 140)
    for sl in STOP_LOSS_LEVELS:
        sl_val = -1.0 if sl is None else sl
        sub = df[(df["trig_lo"] == 0.97) & (df["trig_hi"] == 0.98) & (df["stop_loss"] == sl_val)]
        if len(sub) == 0: continue
        wr = sub["won"].mean()
        avg_buy = sub["buy_px"].mean()
        outcomes = sub["outcome"].value_counts()
        ev_net = sub["pnl_after_fees"].mean()
        daily = ev_net * len(sub) / n_days
        sl_label = "none" if sl is None else f"${sl:.2f}"
        print(f"\n  Stop={sl_label}: {len(sub)} fills ({len(sub)/n_days:.1f}/d)")
        print(f"    Win at expiry:   {outcomes.get('WIN', 0)} ({outcomes.get('WIN', 0)/len(sub)*100:.2f}%)")
        print(f"    Stop triggered:  {outcomes.get('STOP', 0)} ({outcomes.get('STOP', 0)/len(sub)*100:.2f}%)")
        print(f"    Loss hold:       {outcomes.get('LOSS_HOLD', 0)} ({outcomes.get('LOSS_HOLD', 0)/len(sub)*100:.2f}%)")
        print(f"    Avg buy:    ${avg_buy:.4f}")
        if (sub['outcome'] == 'STOP').any():
            print(f"    Avg sell on stop: ${sub[sub['outcome'] == 'STOP']['sell_px'].mean():.4f}")
        print(f"    EV/trade after fees: ${ev_net:.3f}")
        print(f"    $/day:               ${daily:.2f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
