"""
Calcolo dell'EV teorico esatto della strategia ladder, per livello,
usando i dati storici reali dei 66 giorni × 17.807 mercati.

Per ogni mercato m e ogni livello L, ricostruiamo cosa succederebbe se
piazzassimo una ladder con `size` shares al livello L su entrambi i lati:

  - both_filled  : P&L = size × (1 - 2L) + rebate (deterministico)
  - only_up_filled: residuo UP (size shares).
        payoff = size × $1 se UP vince, $0 altrimenti
        profit = size × (winner_payoff - L) + rebate parziale
  - only_dn_filled: simmetrico
  - none_filled  : 0

Aggrega per livello e per portfolio (somma dei livelli).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--path-parquet",
        type=str,
        default="utils/mm_path_5m.parquet",
        help="Output di mm_path_analysis.py",
    )
    ap.add_argument("--size", type=float, default=20.0,
                    help="Shares per level su ciascun lato.")
    ap.add_argument("--levels", type=str,
                    default="0.50,0.45,0.40,0.35,0.30,0.25,0.20,0.15,0.10,0.05")
    ap.add_argument("--rebate-pct", type=float, default=0.20)
    ap.add_argument("--fee-rate", type=float, default=0.07)
    args = ap.parse_args()

    df = pd.read_parquet(args.path_parquet)
    n = len(df)
    print(f"Loaded {n} markets")
    print(f"Winner: UP={(df['winner']=='UP').mean():.1%}  "
          f"DOWN={(df['winner']=='DOWN').mean():.1%}  "
          f"TIE={(df['winner']=='TIE').mean():.1%}")
    print()

    levels = [float(x) for x in args.levels.split(",")]
    size = args.size

    rebate_per_share = lambda px: args.rebate_pct * args.fee_rate * px * (1.0 - px)

    print("=" * 100)
    print(f"EV PER LEVEL (size={size} shares per side per level)")
    print("=" * 100)
    print(
        f"{'L':<6}{'P(both)':>9}{'P(up only)':>12}{'P(dn only)':>12}"
        f"{'P(uponly|UPwon)':>17}{'P(dnonly|DNwon)':>17}"
        f"{'EV_pair':>10}{'EV_singleUP':>13}{'EV_singleDN':>13}{'EV_total':>11}"
    )

    win_up_mask = df["winner"] == "UP"
    win_dn_mask = df["winner"] == "DOWN"

    total_ev = 0.0
    for L in levels:
        L_int = int(round(L * 100))
        up_col = f"up_visit_le_{L_int:02d}"
        dn_col = f"dn_visit_le_{L_int:02d}"
        if up_col not in df.columns:
            print(f"  Missing column {up_col}, skipping level {L}")
            continue
        up_vis = df[up_col].astype(bool)
        dn_vis = df[dn_col].astype(bool)
        both = up_vis & dn_vis
        only_up = up_vis & ~dn_vis
        only_dn = dn_vis & ~up_vis

        p_both = both.mean()
        p_only_up = only_up.mean()
        p_only_dn = only_dn.mean()

        # Conditional: dato only_up_filled, qual è P(UP wins)?
        n_only_up = int(only_up.sum())
        n_only_dn = int(only_dn.sum())
        p_up_won_given_only_up = (
            (only_up & win_up_mask).sum() / max(1, n_only_up) if n_only_up else 0.0
        )
        p_dn_won_given_only_dn = (
            (only_dn & win_dn_mask).sum() / max(1, n_only_dn) if n_only_dn else 0.0
        )

        # EV pair: deterministico (sempre +(1-2L) per share quando both filla)
        ev_pair_per_market = (
            p_both * size * (1.0 - 2.0 * L) + p_both * 2.0 * size * rebate_per_share(L)
        )

        # EV single-leg UP: solo se UP visited but not DN.
        # Con prob (p_up_won_given_only_up): UP vince → payoff = size, profit = size*(1-L)
        # Else: UP perde → payoff = 0, profit = -size * L
        # In entrambi i casi contribuisce il rebate (filla maker su UP).
        ev_single_up_per_market = (
            p_only_up * size * (
                p_up_won_given_only_up * (1.0 - L)
                + (1.0 - p_up_won_given_only_up) * (-L)
            )
            + p_only_up * size * rebate_per_share(L)
        )
        ev_single_dn_per_market = (
            p_only_dn * size * (
                p_dn_won_given_only_dn * (1.0 - L)
                + (1.0 - p_dn_won_given_only_dn) * (-L)
            )
            + p_only_dn * size * rebate_per_share(L)
        )

        ev_total = ev_pair_per_market + ev_single_up_per_market + ev_single_dn_per_market
        total_ev += ev_total

        print(
            f"{L:<6.2f}{p_both:>9.2%}{p_only_up:>12.2%}{p_only_dn:>12.2%}"
            f"{p_up_won_given_only_up:>17.2%}{p_dn_won_given_only_dn:>17.2%}"
            f"{ev_pair_per_market:>10.4f}{ev_single_up_per_market:>13.4f}"
            f"{ev_single_dn_per_market:>13.4f}{ev_total:>11.4f}"
        )

    print("=" * 100)
    print(f"TOTAL EV per market (sum over all levels): ${total_ev:>8.4f}")
    print(f"Per day (×288): ${total_ev * 288:>8.2f}")
    print(f"Markets analysed: {n}")
    print()

    # Anche: solo i livelli con EV positivo
    print("Vediamo solo combinazioni di livelli a EV positivo...")
    positive_levels = []
    for L in levels:
        L_int = int(round(L * 100))
        up_col = f"up_visit_le_{L_int:02d}"
        dn_col = f"dn_visit_le_{L_int:02d}"
        if up_col not in df.columns:
            continue
        up_vis = df[up_col].astype(bool)
        dn_vis = df[dn_col].astype(bool)
        both = up_vis & dn_vis
        only_up = up_vis & ~dn_vis
        only_dn = dn_vis & ~up_vis
        p_both = both.mean()
        p_only_up = only_up.mean()
        p_only_dn = only_dn.mean()
        n_only_up = int(only_up.sum())
        n_only_dn = int(only_dn.sum())
        p_up_w = (only_up & win_up_mask).sum() / max(1, n_only_up)
        p_dn_w = (only_dn & win_dn_mask).sum() / max(1, n_only_dn)
        ev = (
            p_both * size * (1.0 - 2.0 * L)
            + p_both * 2.0 * size * rebate_per_share(L)
            + p_only_up * size * (p_up_w * (1.0 - L) + (1.0 - p_up_w) * (-L))
            + p_only_dn * size * (p_dn_w * (1.0 - L) + (1.0 - p_dn_w) * (-L))
            + (p_only_up + p_only_dn) * size * rebate_per_share(L)
        )
        if ev > 0:
            positive_levels.append((L, ev))
    print(f"Livelli profittevoli individualmente: {[(f'{L:.2f}', f'${e:.4f}') for L, e in positive_levels]}")
    if positive_levels:
        print(f"Sum EV su soli livelli positivi: ${sum(e for _, e in positive_levels):.4f}/market")
        print(f"Sum × 288 markets/day: ${sum(e for _, e in positive_levels) * 288:.2f}/day")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
