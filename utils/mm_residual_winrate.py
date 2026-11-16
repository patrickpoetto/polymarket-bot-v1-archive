"""
Misura empiricamente: quando il BBO-follow simulator finisce con un residuo
single-leg sul lato X, X effettivamente perde quanto spesso?

Se il lato del residuo perde ~50% delle volte, il residuo è un asset random
con EV neutrale (~+1c/share dopo fee/rebate al momento del fill).
Se perde >>50%, c'è un bias strutturale (loser-side concentration).

Questo è IL test che decide se la strategia BBO-1 pair MM è profittevole.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="utils/mm_bbo_7d.parquet")
    args = ap.parse_args()

    df = pd.read_parquet(args.parquet)
    n = len(df)
    print(f"Loaded {n} markets")
    print()

    # Winner distribution
    print("=== WINNER DISTRIBUTION ===")
    print(df["winner"].value_counts())
    print()

    # Residual concentration: in ogni mercato il residuo è su UP o DN
    df["residual_side"] = "NONE"
    df.loc[df["up_residual"] > df["dn_residual"], "residual_side"] = "UP"
    df.loc[df["dn_residual"] > df["up_residual"], "residual_side"] = "DOWN"

    print("=== RESIDUAL SIDE DISTRIBUTION ===")
    print(df["residual_side"].value_counts())
    print()

    # Test critico: se residuo è su UP, quanto spesso UP vince?
    print("=" * 72)
    print("TEST CRITICO: P(side wins | residual on side)")
    print("=" * 72)
    has_res = df["residual_side"] != "NONE"
    df_res = df[has_res].copy()
    print(f"Markets with residual: {len(df_res)} ({len(df_res)/n:.1%})")
    print()

    for side in ("UP", "DOWN"):
        mask = df_res["residual_side"] == side
        sub = df_res[mask]
        if len(sub) == 0:
            continue
        wins = (sub["winner"] == side).sum()
        print(
            f"  Residual on {side}: {len(sub)} markets    "
            f"{side} wins: {wins} ({wins/len(sub):.1%})"
        )

    print()
    print("Se P(side wins | residual side) ≈ 50%, residuo è random → strategia OK")
    print("Se P(side wins | residual side) << 50%, c'è bias loser → strategia bad")
    print()

    # Additional: P(side wins | side has more fills)
    print("=" * 72)
    print("ALTERNATIVA: P(side wins | side has more fills)")
    print("=" * 72)
    df["fill_diff"] = df["n_up_fills"] - df["n_dn_fills"]
    print(f"Fill diff stats: mean={df['fill_diff'].mean():.2f}  "
          f"std={df['fill_diff'].std():.2f}")
    print()
    df["which_filled_more"] = "TIE"
    df.loc[df["fill_diff"] > 0, "which_filled_more"] = "UP"
    df.loc[df["fill_diff"] < 0, "which_filled_more"] = "DOWN"
    print(df["which_filled_more"].value_counts())
    print()

    for side in ("UP", "DOWN"):
        mask = df["which_filled_more"] == side
        sub = df[mask]
        if len(sub) == 0:
            continue
        wins = (sub["winner"] == side).sum()
        print(
            f"  More fills on {side}: {len(sub)} markets    "
            f"{side} wins: {wins} ({wins/len(sub):.1%})"
        )

    # Per i markets dove il residuo è sul perdente vs vincente, qual è il P&L tipico?
    print()
    print("=" * 72)
    print("P&L per residual-side outcome (markets with residual)")
    print("=" * 72)
    for side in ("UP", "DOWN"):
        mask = df_res["residual_side"] == side
        sub = df_res[mask]
        if len(sub) == 0:
            continue
        won = sub[sub["winner"] == side]
        lost = sub[sub["winner"] != side]
        print(f"\n  Residual on {side}:")
        print(f"    Won  ({len(won)}):  avg P&L = ${won['pnl'].mean():.4f}  "
              f"avg residual = {won[f'{side.lower()}_residual'].mean() if side=='UP' else won['dn_residual'].mean():.2f}")
        print(f"    Lost ({len(lost)}): avg P&L = ${lost['pnl'].mean():.4f}  "
              f"avg residual = {lost[f'{side.lower()}_residual'].mean() if side=='UP' else lost['dn_residual'].mean():.2f}")
    print()

    # Counterfactual: se i residui fossero random 50/50, quale sarebbe il P&L?
    print("=" * 72)
    print("COUNTERFACTUAL: P&L se residui fossero random 50/50")
    print("=" * 72)
    # Per ogni market: pnl_corrente = payoff_pairs + payoff_residual - cash_spent + rebate
    # Counterfactual residual payoff = 0.5 * (up_residual + dn_residual) * $1
    # (in realtà uno dei due è 0, quindi è 0.5 × residuo_size)
    df["residual_size"] = df["up_residual"] + df["dn_residual"]
    df["cf_residual_payoff"] = 0.5 * df["residual_size"]
    df["cf_pnl"] = (
        df["payoff_pairs"]
        + df["cf_residual_payoff"]
        - df["cash_spent"]
        + df["rebate_total"]
        - df["taker_fees"]
    )

    print(f"  Avg actual P&L:           ${df['pnl'].mean():.4f}")
    print(f"  Avg counterfactual (CF):  ${df['cf_pnl'].mean():.4f}")
    print(f"  Diff (actual - CF):       ${(df['pnl']-df['cf_pnl']).mean():.4f}")
    print(f"  → Bias loss: ${df['pnl'].mean() - df['cf_pnl'].mean():.2f}/market dovuto al bias loser")
    print()
    print(f"  Days observed: {df['date'].nunique()}")
    print(f"  Per day actual:           ${df['pnl'].sum()/df['date'].nunique():.2f}")
    print(f"  Per day counterfactual:   ${df['cf_pnl'].sum()/df['date'].nunique():.2f}")
    print(f"  Per day gain if random:   ${(df['cf_pnl']-df['pnl']).sum()/df['date'].nunique():.2f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
