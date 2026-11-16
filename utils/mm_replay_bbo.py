"""
Replay tick-by-tick di UN solo mercato con BBO-follow su entrambi i lati.

Per ogni tick:
  - L'order maker UP è a (UP_bid - offset). L'order maker DN è a (DN_bid - offset).
  - Quando il bid scende, l'ordine vecchio fillа. Re-piazza al nuovo BBO-1.
  - Cap inventory = max_inv_per_side.

A close_at_seconds: cancella ordini, taker-sell residual al bid corrente.
Hold pairs a expiry.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

from utils.mm_analyze_5m import parse_market_csv, PRICES_5M_ROOT


def replay_bbo(path: Path, offset_ticks=1, fill_size=5, max_inv=200,
               trade_window=300, close_at=None, tick=0.01, verbose=True) -> dict:
    wide = parse_market_csv(path)
    if wide is None:
        if verbose: print(f"Could not parse {path}")
        return {}
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

    if verbose:
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
    if verbose:
        print(f"Winner: {winner}\n")
        print(f"BBO-follow: offset={offset_ticks}  fill_size={fill_size}  max_inv={max_inv}")
        print(f"Window: {trade_window}s  close_at: {close_at}s\n")

    offset = offset_ticks * tick
    EPS = 1e-9
    up_shares = 0.0
    dn_shares = 0.0
    up_fills = []  # (t, fill_px)
    dn_fills = []
    cash = 0.0
    rebate = 0.0
    rebate_factor = 0.20 * 0.07

    prev_ub = up_bid[0]
    prev_db = dn_bid[0]

    win_ms = int(trade_window * 1000)
    close_ms = int(close_at * 1000) if close_at else None

    sl_recovery = 0.0
    sl_taker_fees = 0.0
    closed = False

    for i in range(1, len(wide)):
        elapsed_ms = t_ms[i] - t0
        if elapsed_ms > win_ms:
            break

        # Close-at trigger
        if close_ms is not None and not closed and elapsed_ms >= close_ms:
            ub_close = up_bid[i]
            db_close = dn_bid[i]
            up_excess = max(0.0, up_shares - dn_shares)
            dn_excess = max(0.0, dn_shares - up_shares)
            if up_excess > 0 and ub_close > 0:
                sl_recovery += ub_close * up_excess
                sl_taker_fees += 0.07 * ub_close * (1 - ub_close) * up_excess
                up_shares -= up_excess
                if verbose:
                    print(f"  CLOSE t={elapsed[i]:.1f}s: TAKER SELL {up_excess:.0f} UP @ ${ub_close:.3f}")
            if dn_excess > 0 and db_close > 0:
                sl_recovery += db_close * dn_excess
                sl_taker_fees += 0.07 * db_close * (1 - db_close) * dn_excess
                dn_shares -= dn_excess
                if verbose:
                    print(f"  CLOSE t={elapsed[i]:.1f}s: TAKER SELL {dn_excess:.0f} DN @ ${db_close:.3f}")
            closed = True
            break  # cancel ladders

        ub = up_bid[i]
        db = dn_bid[i]

        # UP fill check: if UP_bid dropped, our old maker at (prev_ub - offset) fills
        if up_shares < max_inv - EPS and ub < prev_ub - EPS:
            old_order_px = prev_ub - offset
            if old_order_px > 0 and ub < old_order_px - EPS:
                fill_px = old_order_px
                sz = min(fill_size, max_inv - up_shares)
                cash += fill_px * sz
                up_shares += sz
                rebate += rebate_factor * fill_px * (1 - fill_px) * sz
                up_fills.append((elapsed[i], fill_px))

        if dn_shares < max_inv - EPS and db < prev_db - EPS:
            old_order_px = prev_db - offset
            if old_order_px > 0 and db < old_order_px - EPS:
                fill_px = old_order_px
                sz = min(fill_size, max_inv - dn_shares)
                cash += fill_px * sz
                dn_shares += sz
                rebate += rebate_factor * fill_px * (1 - fill_px) * sz
                dn_fills.append((elapsed[i], fill_px))

        prev_ub = ub
        prev_db = db

    pairs = min(up_shares, dn_shares)
    up_res = up_shares - pairs
    dn_res = dn_shares - pairs

    if winner == "UP":
        residual_payoff = up_res * 1.0
    elif winner == "DOWN":
        residual_payoff = dn_res * 1.0
    else:
        residual_payoff = 0.5 * (up_res + dn_res)

    pair_payoff = pairs * 1.0
    pnl = pair_payoff + residual_payoff - cash + rebate + sl_recovery - sl_taker_fees

    avg_up = sum(f[1] for f in up_fills) / len(up_fills) if up_fills else 0.0
    avg_dn = sum(f[1] for f in dn_fills) / len(dn_fills) if dn_fills else 0.0

    if verbose:
        print(f"=== UP fills ({len(up_fills)} total) ===")
        for t, px in up_fills[:30]:
            print(f"  t={t:6.1f}s  fill@${px:.3f}")
        if len(up_fills) > 30:
            print(f"  ... and {len(up_fills)-30} more")
        print(f"  Total UP shares: {up_shares + up_res if not closed else up_shares}, avg ${avg_up:.3f}")

        print(f"\n=== DN fills ({len(dn_fills)} total) ===")
        for t, px in dn_fills[:30]:
            print(f"  t={t:6.1f}s  fill@${px:.3f}")
        if len(dn_fills) > 30:
            print(f"  ... and {len(dn_fills)-30} more")
        print(f"  Total DN shares: {dn_shares + dn_res if not closed else dn_shares}, avg ${avg_dn:.3f}")

        print(f"\n=== Outcome ===")
        print(f"Pairs: {pairs:.0f}  Pair cost: ${avg_up + avg_dn:.3f}")
        print(f"Up residual: {up_res:.0f}  Dn residual: {dn_res:.0f}")
        print(f"Cash spent (maker buys): ${cash:.2f}")
        print(f"Rebate earned:           ${rebate:.4f}")
        print(f"Pair payoff:             ${pair_payoff:.2f}")
        print(f"Residual payoff:         ${residual_payoff:.2f}")
        print(f"SL recovery (close):     ${sl_recovery:.2f}")
        print(f"SL taker fees:           ${sl_taker_fees:.4f}")
        print(f"P&L:                     ${pnl:+.3f}")

    return {
        "pnl": pnl, "pairs": pairs, "pair_cost": avg_up + avg_dn,
        "up_shares": up_shares, "dn_shares": dn_shares,
        "up_res": up_res, "dn_res": dn_res,
        "n_up_fills": len(up_fills), "n_dn_fills": len(dn_fills),
        "winner": winner,
    }


def main() -> int:
    args = sys.argv[1:]
    close_at = None
    for i, a in enumerate(args):
        if a == "--close" and i + 1 < len(args):
            close_at = float(args[i + 1])
    paths = [a for a in args if not a.startswith("--") and not (a.replace(".", "").replace("-", "").isdigit())]
    if not paths:
        days = sorted([d for d in PRICES_5M_ROOT.iterdir() if d.is_dir()])
        latest = days[-1]
        files = sorted((latest / "csv").glob("*.csv"))
        paths = [str(files[0])]

    for p in paths:
        replay_bbo(Path(p), close_at=close_at)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
