"""
Replay tick-by-tick di UN solo mercato per capire ESATTAMENTE quando e come
si attivano i fill nel simulator BBO-follow su entrambi i lati.

Verifica empirica della tua tesi 'in qualsiasi mercato c'è almeno un tick di
pullback per fillare maker, e l'altro lato scende quanto basta'.

Uso: python -m utils.mm_replay_one <market_csv_path>
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

from utils.mm_analyze_5m import parse_market_csv, PRICES_5M_ROOT


def replay_static(path: Path, levels=(0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10),
                  size=10, cap=200, close_at=15.0) -> None:
    wide = parse_market_csv(path)
    if wide is None:
        print(f"Could not parse {path}"); return
    for col in ("up_bid", "up_ask", "dn_bid", "dn_ask"):
        wide[col] = wide[col].ffill()
    wide = wide.dropna().reset_index(drop=True)

    up_bid = wide["up_bid"].values
    up_ask = wide["up_ask"].values
    dn_bid = wide["dn_bid"].values
    dn_ask = wide["dn_ask"].values
    t_ms = wide["t_ms"].values
    t0 = t_ms[0]
    elapsed = (t_ms - t0) / 1000.0

    up_mid = (up_bid + up_ask) / 2
    dn_mid = (dn_bid + dn_ask) / 2

    print(f"Market: {path.stem} ({path.parent.parent.name})")
    print(f"Open:  UP={up_mid[0]:.3f}  DN={dn_mid[0]:.3f}")
    print(f"Close: UP={up_mid[-1]:.3f}  DN={dn_mid[-1]:.3f}")
    print(f"UP range: {up_mid.min():.3f} - {up_mid.max():.3f}")
    print(f"DN range: {dn_mid.min():.3f} - {dn_mid.max():.3f}")

    if up_mid[-1] > dn_mid[-1]:
        winner = "UP"
    elif dn_mid[-1] > up_mid[-1]:
        winner = "DOWN"
    else:
        winner = "TIE"
    print(f"Winner: {winner}\n")
    print(f"Strategy: STATIC LADDER levels={levels}, size={size}, cap={cap}, close_at={close_at}s\n")

    # Simulate
    up_filled = [False] * len(levels)
    dn_filled = [False] * len(levels)
    up_fills = []
    dn_fills = []
    up_shares = 0
    dn_shares = 0
    EPS = 1e-9

    prev_ub = up_bid[0]
    prev_db = dn_bid[0]

    for i in range(1, len(wide)):
        ub = up_bid[i]; db = dn_bid[i]
        for k, lvl in enumerate(levels):
            if up_filled[k]: continue
            if prev_ub >= lvl - EPS and ub < lvl - EPS and up_shares < cap - EPS:
                up_fills.append((elapsed[i], lvl, prev_ub, ub))
                up_shares += size
                up_filled[k] = True
        for k, lvl in enumerate(levels):
            if dn_filled[k]: continue
            if prev_db >= lvl - EPS and db < lvl - EPS and dn_shares < cap - EPS:
                dn_fills.append((elapsed[i], lvl, prev_db, db))
                dn_shares += size
                dn_filled[k] = True
        prev_ub = ub
        prev_db = db

    print(f"=== UP ladder fills ===")
    for t, lvl, old_b, new_b in up_fills:
        print(f"  t={t:6.1f}s  lvl@{lvl:.2f}  (UP_bid {old_b:.3f}→{new_b:.3f})")
    print(f"  Total UP shares (pre-close): {up_shares}  avg_px=${sum(f[1] for f in up_fills)/len(up_fills) if up_fills else 0:.3f}")

    print(f"\n=== DN ladder fills ===")
    for t, lvl, old_b, new_b in dn_fills:
        print(f"  t={t:6.1f}s  lvl@{lvl:.2f}  (DN_bid {old_b:.3f}→{new_b:.3f})")
    print(f"  Total DN shares (pre-close): {dn_shares}  avg_px=${sum(f[1] for f in dn_fills)/len(dn_fills) if dn_fills else 0:.3f}")

    # Close at t=close_at
    idx_close = int(np.searchsorted(t_ms, t0 + int(close_at * 1000)))
    idx_close = min(max(idx_close, 0), len(wide) - 1)
    ub_close = up_bid[idx_close]
    db_close = dn_bid[idx_close]
    print(f"\n=== Close at t={close_at}s ===")
    print(f"  At t={elapsed[idx_close]:.1f}s: UP_bid={ub_close:.3f}, DN_bid={db_close:.3f}")

    up_excess = max(0, up_shares - dn_shares)
    dn_excess = max(0, dn_shares - up_shares)
    sl_recovery = 0
    if up_excess > 0:
        sl_recovery += ub_close * up_excess
        up_shares -= up_excess
        print(f"  TAKER SELL {up_excess} UP excess @ ${ub_close:.3f} → recovery ${ub_close*up_excess:.2f}")
    if dn_excess > 0:
        sl_recovery += db_close * dn_excess
        dn_shares -= dn_excess
        print(f"  TAKER SELL {dn_excess} DN excess @ ${db_close:.3f} → recovery ${db_close*dn_excess:.2f}")
    if up_excess == 0 and dn_excess == 0:
        print(f"  No residual to close (both sides equal)")

    pairs = min(up_shares, dn_shares)
    avg_up = sum(f[1] for f in up_fills)/len(up_fills) if up_fills else 0
    avg_dn = sum(f[1] for f in dn_fills)/len(dn_fills) if dn_fills else 0
    cash = sum(f[1] for f in up_fills) * size + sum(f[1] for f in dn_fills) * size

    pair_payoff = pairs * 1.0
    pnl = pair_payoff + sl_recovery - cash

    print(f"\n=== Outcome ===")
    print(f"Pre-close: UP={up_shares + up_excess}, DN={dn_shares + dn_excess}")
    print(f"Post-close: UP={up_shares}, DN={dn_shares}, Pairs={pairs}")
    print(f"Avg UP fill: ${avg_up:.3f}")
    print(f"Avg DN fill: ${avg_dn:.3f}")
    print(f"Pair cost (avg_up + avg_dn): ${avg_up + avg_dn:.3f}")
    print(f"Total cash spent (maker): ${cash:.2f}")
    print(f"Recovery (taker close):   ${sl_recovery:.2f}")
    print(f"Pair payoff (pairs × $1): ${pair_payoff:.2f}")
    print(f"P&L: ${pnl:+.2f}")


def replay(path: Path, cap: int = 10, fill_size: int = 5, bbo_offset: int = 1,
           tick: float = 0.01) -> None:
    wide = parse_market_csv(path)
    if wide is None:
        print(f"Could not parse {path}"); return
    for col in ("up_bid", "up_ask", "dn_bid", "dn_ask"):
        wide[col] = wide[col].ffill()
    wide = wide.dropna().reset_index(drop=True)

    up_bid = wide["up_bid"].values
    up_ask = wide["up_ask"].values
    dn_bid = wide["dn_bid"].values
    dn_ask = wide["dn_ask"].values
    t_ms = wide["t_ms"].values
    t0 = t_ms[0]
    elapsed = (t_ms - t0) / 1000.0

    up_mid = (up_bid + up_ask) / 2
    dn_mid = (dn_bid + dn_ask) / 2

    print(f"Market: {path.stem} ({path.parent.parent.name})")
    print(f"Open:  UP={up_mid[0]:.3f}  DN={dn_mid[0]:.3f}")
    print(f"Close: UP={up_mid[-1]:.3f}  DN={dn_mid[-1]:.3f}")
    print(f"UP range: {up_mid.min():.3f} - {up_mid.max():.3f}  ({(up_mid.max()-up_mid.min())*100:.1f}c)")
    print(f"DN range: {dn_mid.min():.3f} - {dn_mid.max():.3f}  ({(dn_mid.max()-dn_mid.min())*100:.1f}c)")

    if up_mid[-1] > dn_mid[-1]:
        winner = "UP"
    elif dn_mid[-1] > up_mid[-1]:
        winner = "DOWN"
    else:
        winner = "TIE"
    print(f"Winner: {winner}\n")

    # Replay con BBO-1 maker su entrambi i lati, cap basso
    up_shares = 0
    dn_shares = 0
    up_fills = []
    dn_fills = []
    offset = bbo_offset * tick
    prev_ub = up_bid[0]
    prev_db = dn_bid[0]

    for i in range(1, len(wide)):
        ub = up_bid[i]; db = dn_bid[i]
        # UP fill check
        if up_shares < cap and ub < prev_ub - 1e-9:
            order_px = prev_ub - offset
            if ub < order_px - 1e-9:
                up_fills.append((elapsed[i], order_px, prev_ub, ub))
                up_shares += fill_size
        if dn_shares < cap and db < prev_db - 1e-9:
            order_px = prev_db - offset
            if db < order_px - 1e-9:
                dn_fills.append((elapsed[i], order_px, prev_db, db))
                dn_shares += fill_size
        prev_ub = ub
        prev_db = db

    print(f"=== UP fills (cap {cap}, BBO-{bbo_offset}) ===")
    for t, px, old_b, new_b in up_fills:
        print(f"  t={t:6.1f}s  fill@{px:.3f}  (UP_bid {old_b:.3f}→{new_b:.3f})")
    print(f"  Total UP shares: {up_shares}")
    print(f"  Avg UP price:    ${np.mean([f[1] for f in up_fills]) if up_fills else 0:.3f}")

    print(f"\n=== DN fills (cap {cap}, BBO-{bbo_offset}) ===")
    for t, px, old_b, new_b in dn_fills:
        print(f"  t={t:6.1f}s  fill@{px:.3f}  (DN_bid {old_b:.3f}→{new_b:.3f})")
    print(f"  Total DN shares: {dn_shares}")
    print(f"  Avg DN price:    ${np.mean([f[1] for f in dn_fills]) if dn_fills else 0:.3f}")

    pairs = min(up_shares, dn_shares)
    up_res = up_shares - pairs
    dn_res = dn_shares - pairs
    avg_up = np.mean([f[1] for f in up_fills]) if up_fills else 0
    avg_dn = np.mean([f[1] for f in dn_fills]) if dn_fills else 0
    pair_cost = avg_up + avg_dn

    cash = up_shares * avg_up + dn_shares * avg_dn
    pair_payoff = pairs * 1.0
    if winner == "UP":
        res_payoff = up_res * 1.0
    elif winner == "DOWN":
        res_payoff = dn_res * 1.0
    else:
        res_payoff = 0
    pnl = pair_payoff + res_payoff - cash

    print(f"\n=== Outcome ===")
    print(f"Pairs:        {pairs}  @ ${pair_cost:.3f}/pair")
    print(f"UP residual:  {up_res} share @ ${avg_up:.3f}")
    print(f"DN residual:  {dn_res} share @ ${avg_dn:.3f}")
    print(f"Cash spent:   ${cash:.2f}")
    print(f"Pair payoff:  ${pair_payoff:.2f}  (sempre)")
    print(f"Resid payoff: ${res_payoff:.2f}  (winner={winner})")
    print(f"P&L:          ${pnl:+.3f}")


def main() -> int:
    mode = "static"
    args = sys.argv[1:]
    if args and args[0] in ("--bbo", "--static"):
        mode = args[0][2:]
        args = args[1:]
    if args:
        path = Path(args[0])
    else:
        days = sorted([d for d in PRICES_5M_ROOT.iterdir() if d.is_dir()])
        latest = days[-1]
        files = sorted((latest / "csv").glob("*.csv"))
        path = files[0]

    if mode == "bbo":
        replay(path)
    else:
        replay_static(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
