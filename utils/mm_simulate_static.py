"""
STATIC LADDER su ENTRAMBI i lati: i maker buy NON seguono il BBO.
Sono piazzati a livelli fissi (es. 0.45, 0.40, 0.35, 0.30, ...) e fillanano
solo quando il bid del rispettivo lato tocca quel livello.

Vantaggio rispetto al BBO-follow:
  - Se un lato sale, i nostri ordini RIMANGONO ai livelli bassi (non ci
    seguono al rialzo). Quindi NON compriamo MAI a prezzo alto su quel lato.
  - Se un lato scende, fillanamo progressivamente a prezzi bassi.
  - In mercati oscillanti dove ENTRAMBI i lati toccano prezzi bassi in
    momenti diversi, otteniamo pairs a costo < $1.

Limite intrinseco:
  - In mercati direzionali, solo UN lato tocca i livelli bassi → fillanamo
    solo il PERDENTE. Loss su residual.

Lo scopo del simulator è misurare:
  - Quanti mercati hanno entrambi i lati che toccano i livelli (= profit potential)
  - Pair cost medio quando si formano pairs
  - P&L netto per mercato
"""
from __future__ import annotations
import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from utils.mm_analyze_5m import parse_market_csv, PRICES_5M_ROOT


@dataclass(frozen=True)
class StaticPolicy:
    # Livelli statici per i maker buy (price units, in [0,1])
    levels: tuple[float, ...] = (0.49, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10)
    size_per_level: float = 5.0

    # Cap totale per lato (somma di tutti i fills sul lato)
    max_inv_per_side: float = 100.0

    # Re-arm levels: se un livello viene fillato e poi il bid risale sopra
    # il livello, posso rifillare al livello? Default: NO (un fill per livello).
    refill: bool = False

    # Window operativo: ordini attivi per i primi N secondi.
    # 0 o >300 = tutto il mercato
    trade_window_seconds: float = 300.0

    # STOP-LOSS continuo (intra-market): se il bid di un lato scende sotto
    # questa soglia, taker-vendi le shares di quel lato. 0 = disabilitato.
    stop_loss_threshold: float = 0.0
    stop_loss_residual_only: bool = True

    # END-OF-WINDOW close: a t=close_at_seconds, taker-vendi i residual netti
    # al bid corrente. 0 = disabilitato.
    close_at_seconds: float = 0.0
    # Se True, chiude SOLO il residual; se False, chiude TUTTO (pairs incluse).
    close_residual_only: bool = True

    # SMART CANCEL: se un lato ha N share in più dell'altro, blocca i nuovi
    # fill su quel lato (= cancel della ladder su quel lato).
    # Così limitiamo l'accumulo unilaterale = riduzione perdite unpaired.
    skew_cancel_threshold: int = 0  # 0 = disabilitato, e.g. 10 = block side leading by 10+ shares

    fee_rate: float = 0.07
    rebate_pct: float = 0.20
    tick: float = 0.01


@dataclass
class StaticResult:
    market_name: str
    date: str
    n_up_fills: int
    n_dn_fills: int
    up_shares: float
    dn_shares: float
    pairs: float
    up_residual: float
    dn_residual: float
    avg_up_price: float
    avg_dn_price: float
    pair_cost: float
    cash_spent: float
    rebate_total: float
    payoff_pairs: float
    payoff_residual: float
    pnl: float
    winner: str
    up_levels_touched: int
    dn_levels_touched: int
    both_sides_filled: bool
    sl_triggered: bool = False
    sl_recovery: float = 0.0


def simulate_static(
    wide: pd.DataFrame, cfg: StaticPolicy, market_name: str, date: str
) -> Optional[StaticResult]:
    if wide is None or len(wide) < 30:
        return None
    wide = wide.copy()
    for col in ("up_bid", "up_ask", "dn_bid", "dn_ask"):
        wide[col] = wide[col].ffill()
    wide = wide.dropna(subset=["up_bid", "up_ask", "dn_bid", "dn_ask"]).reset_index(drop=True)
    if len(wide) < 30:
        return None

    up_bid = wide["up_bid"].to_numpy(dtype=np.float64)
    up_ask = wide["up_ask"].to_numpy(dtype=np.float64)
    dn_bid = wide["dn_bid"].to_numpy(dtype=np.float64)
    dn_ask = wide["dn_ask"].to_numpy(dtype=np.float64)
    t_ms = wide["t_ms"].to_numpy(dtype=np.int64)
    n = len(wide)
    t_start = int(t_ms[0])
    EPS = 1e-9

    rebate_factor = cfg.rebate_pct * cfg.fee_rate
    levels = cfg.levels

    # Per ogni livello, traccia: fillato? (per re-arm logic)
    up_filled = [False] * len(levels)
    dn_filled = [False] * len(levels)

    up_shares = 0.0
    dn_shares = 0.0
    cash_spent = 0.0
    rebate_total = 0.0
    n_up_fills = 0
    n_dn_fills = 0
    sum_up_px = 0.0
    sum_dn_px = 0.0
    sum_up_sz = 0.0
    sum_dn_sz = 0.0

    prev_ub = up_bid[0]
    prev_db = dn_bid[0]

    win_ms = int(cfg.trade_window_seconds * 1000.0)
    close_ms = int(cfg.close_at_seconds * 1000.0) if cfg.close_at_seconds > 0 else None

    # Stop-loss state
    sl_triggered_up = False
    sl_triggered_dn = False
    sl_recovery = 0.0
    sl_taker_fees = 0.0
    sl_any = False
    close_done = False

    # Compute closing prices
    final_up_mid = (up_bid[-1] + up_ask[-1]) * 0.5
    final_dn_mid = (dn_bid[-1] + dn_ask[-1]) * 0.5
    if final_up_mid > final_dn_mid:
        winner = "UP"
    elif final_dn_mid > final_up_mid:
        winner = "DOWN"
    else:
        winner = "TIE"

    for i in range(1, n):
        elapsed_ms = t_ms[i] - t_start
        if elapsed_ms > win_ms:
            break

        # CLOSE-AT: at the configured time, cancel all ladders and taker-close residual
        if close_ms is not None and not close_done and elapsed_ms >= close_ms:
            ub_close = up_bid[i]
            db_close = dn_bid[i]
            if cfg.close_residual_only:
                up_excess = max(0.0, up_shares - dn_shares)
                dn_excess = max(0.0, dn_shares - up_shares)
            else:
                up_excess = up_shares
                dn_excess = dn_shares
            if up_excess > 0 and ub_close > 0:
                sell_px = ub_close
                sl_recovery += sell_px * up_excess
                sl_taker_fees += cfg.fee_rate * sell_px * (1.0 - sell_px) * up_excess
                up_shares -= up_excess
                sl_any = True
            if dn_excess > 0 and db_close > 0:
                sell_px = db_close
                sl_recovery += sell_px * dn_excess
                sl_taker_fees += cfg.fee_rate * sell_px * (1.0 - sell_px) * dn_excess
                dn_shares -= dn_excess
                sl_any = True
            close_done = True
            # Cancel all ladders → no more fills
            break

        ub = up_bid[i]
        db = dn_bid[i]

        # SMART CANCEL: blocca fill sul lato che ha già più share dell'altro
        sct = cfg.skew_cancel_threshold
        block_up = sct > 0 and (up_shares - dn_shares) >= sct
        block_dn = sct > 0 and (dn_shares - up_shares) >= sct

        # Check UP fills (skip entirely if block_up)
        for k, lvl in enumerate(levels):
            if block_up:
                break
            if up_filled[k] and not cfg.refill:
                continue
            # Se refill=True, ri-arm il livello quando il bid è risalito sopra lvl + 1c
            if cfg.refill and up_filled[k] and prev_ub >= lvl + cfg.tick:
                up_filled[k] = False
            if up_filled[k]:
                continue
            # Fill triggers when bid crosses level downwards
            if prev_ub >= lvl - EPS and ub < lvl - EPS and up_shares < cfg.max_inv_per_side - EPS:
                sz = min(cfg.size_per_level, cfg.max_inv_per_side - up_shares)
                if sz > 0:
                    cash_spent += lvl * sz
                    up_shares += sz
                    rebate_total += rebate_factor * lvl * (1.0 - lvl) * sz
                    sum_up_px += lvl * sz
                    sum_up_sz += sz
                    n_up_fills += 1
                up_filled[k] = True

        for k, lvl in enumerate(levels):
            if block_dn:
                break
            if dn_filled[k] and not cfg.refill:
                continue
            if cfg.refill and dn_filled[k] and prev_db >= lvl + cfg.tick:
                dn_filled[k] = False
            if dn_filled[k]:
                continue
            if prev_db >= lvl - EPS and db < lvl - EPS and dn_shares < cfg.max_inv_per_side - EPS:
                sz = min(cfg.size_per_level, cfg.max_inv_per_side - dn_shares)
                if sz > 0:
                    cash_spent += lvl * sz
                    dn_shares += sz
                    rebate_total += rebate_factor * lvl * (1.0 - lvl) * sz
                    sum_dn_px += lvl * sz
                    sum_dn_sz += sz
                    n_dn_fills += 1
                dn_filled[k] = True

        # Stop-loss check (taker-sell on the side that's crashing)
        if cfg.stop_loss_threshold > 0.0:
            # UP side stop-loss
            if not sl_triggered_up and up_shares > 0 and ub > 0 and ub < cfg.stop_loss_threshold - EPS:
                # Determine quantity to sell: if residual_only, sell only excess over dn_shares
                if cfg.stop_loss_residual_only:
                    qty = max(0.0, up_shares - dn_shares)
                else:
                    qty = up_shares
                if qty > 0:
                    # taker sell at current bid
                    sell_px = ub
                    sl_recovery += sell_px * qty
                    # taker fee (on sale, polymarket fee = feeRate × p × (1-p) per share)
                    sl_taker_fees += cfg.fee_rate * sell_px * (1.0 - sell_px) * qty
                    up_shares -= qty
                    sl_triggered_up = True
                    sl_any = True
            if not sl_triggered_dn and dn_shares > 0 and db > 0 and db < cfg.stop_loss_threshold - EPS:
                if cfg.stop_loss_residual_only:
                    qty = max(0.0, dn_shares - up_shares)
                else:
                    qty = dn_shares
                if qty > 0:
                    sell_px = db
                    sl_recovery += sell_px * qty
                    sl_taker_fees += cfg.fee_rate * sell_px * (1.0 - sell_px) * qty
                    dn_shares -= qty
                    sl_triggered_dn = True
                    sl_any = True

        prev_ub = ub
        prev_db = db

    # If close_at was configured but never reached (market shorter than close_at),
    # do nothing - we hold what we have.
    pairs = min(up_shares, dn_shares)
    up_res = up_shares - pairs
    dn_res = dn_shares - pairs

    avg_up = sum_up_px / sum_up_sz if sum_up_sz > 0 else 0.0
    avg_dn = sum_dn_px / sum_dn_sz if sum_dn_sz > 0 else 0.0
    pair_cost = avg_up + avg_dn if (sum_up_sz > 0 and sum_dn_sz > 0) else float("nan")

    payoff_pairs = pairs * 1.0
    if winner == "UP":
        payoff_residual = up_res * 1.0
    elif winner == "DOWN":
        payoff_residual = dn_res * 1.0
    else:
        payoff_residual = 0.5 * (up_res + dn_res)

    pnl = payoff_pairs + payoff_residual - cash_spent + rebate_total + sl_recovery - sl_taker_fees

    up_levels_touched = sum(1 for f in up_filled if f)
    dn_levels_touched = sum(1 for f in dn_filled if f)

    return StaticResult(
        market_name=market_name, date=date,
        n_up_fills=n_up_fills, n_dn_fills=n_dn_fills,
        up_shares=up_shares, dn_shares=dn_shares,
        pairs=pairs, up_residual=up_res, dn_residual=dn_res,
        avg_up_price=avg_up, avg_dn_price=avg_dn,
        pair_cost=pair_cost,
        cash_spent=cash_spent, rebate_total=rebate_total,
        payoff_pairs=payoff_pairs, payoff_residual=payoff_residual,
        pnl=pnl, winner=winner,
        up_levels_touched=up_levels_touched,
        dn_levels_touched=dn_levels_touched,
        both_sides_filled=(up_levels_touched > 0 and dn_levels_touched > 0),
        sl_triggered=sl_any,
        sl_recovery=sl_recovery,
    )


_GLOBAL_CFG: Optional[StaticPolicy] = None


def _init_worker(cfg_dict: dict) -> None:
    global _GLOBAL_CFG
    _GLOBAL_CFG = StaticPolicy(**cfg_dict)


def _process_one(path_str: str) -> Optional[dict]:
    cfg = _GLOBAL_CFG or StaticPolicy()
    p = Path(path_str)
    wide = parse_market_csv(p)
    if wide is None:
        return None
    res = simulate_static(wide, cfg, p.stem, p.parent.parent.name)
    return asdict(res) if res else None


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


def print_summary(df: pd.DataFrame, cfg: StaticPolicy) -> None:
    print("\n" + "=" * 88)
    print("STATIC LADDER SUMMARY")
    print("=" * 88)
    n = len(df)
    days = df["date"].nunique() if "date" in df.columns else 1
    print(f"Markets: {n}    Days: {days}    Levels: {cfg.levels}    Size: {cfg.size_per_level}")
    print(f"Window: {cfg.trade_window_seconds}s    Refill: {cfg.refill}    Cap/side: {cfg.max_inv_per_side}\n")

    pnl = df["pnl"]
    print(f"=== HEADLINE ===")
    print(f"  Total P&L:           ${pnl.sum():>10.2f}")
    print(f"  Avg P&L / market:    ${pnl.mean():>10.4f}")
    print(f"  Median P&L:          ${pnl.median():>10.4f}")
    print(f"  P&L > 0:             {int((pnl>0).sum())} ({(pnl>0).mean():.1%})")
    print(f"  Per day:             ${pnl.sum()/days:>10.2f}")
    print(f"  Days positive:       {int((df.groupby('date')['pnl'].sum()>0).sum())}/{days}")

    print(f"\n=== FILL STATS ===")
    print(f"  Markets with ≥1 fill ANY side: {int((df['n_up_fills'] + df['n_dn_fills'] > 0).sum())} ({((df['n_up_fills']+df['n_dn_fills']>0).mean()):.1%})")
    print(f"  Markets with both sides filled: {int(df['both_sides_filled'].sum())} ({df['both_sides_filled'].mean():.1%})")
    print(f"  Markets with ONLY UP fills:    {int(((df['n_up_fills']>0)&(df['n_dn_fills']==0)).sum())}")
    print(f"  Markets with ONLY DN fills:    {int(((df['n_dn_fills']>0)&(df['n_up_fills']==0)).sum())}")
    print(f"  Markets with NO fills:         {int(((df['n_up_fills']==0)&(df['n_dn_fills']==0)).sum())}")

    print(f"\n=== AVG PER MARKET ===")
    print(f"  UP fills:            {df['n_up_fills'].mean():.2f}")
    print(f"  DN fills:            {df['n_dn_fills'].mean():.2f}")
    print(f"  UP shares:           {df['up_shares'].mean():.2f}")
    print(f"  DN shares:           {df['dn_shares'].mean():.2f}")
    print(f"  Pairs:               {df['pairs'].mean():.2f}")
    print(f"  UP residual:         {df['up_residual'].mean():.2f}")
    print(f"  DN residual:         {df['dn_residual'].mean():.2f}")
    paired = df[df["both_sides_filled"]]
    if len(paired):
        print(f"  Pair cost (paired):  ${paired['pair_cost'].mean():.4f}    (n={len(paired)})")

    # Profit decomposition for paired markets
    print(f"\n=== PAIRED MARKETS ONLY ({int(df['both_sides_filled'].sum())} markets) ===")
    if len(paired):
        print(f"  Total P&L:           ${paired['pnl'].sum():.2f}")
        print(f"  Avg P&L / market:    ${paired['pnl'].mean():.4f}")
        print(f"  Median P&L:          ${paired['pnl'].median():.4f}")
        print(f"  P&L > 0:             {int((paired['pnl']>0).sum())} ({(paired['pnl']>0).mean():.1%})")
        print(f"  Per day:             ${paired['pnl'].sum()/days:.2f}")

    print(f"\n=== UNPAIRED (one side only) ({int(((df['n_up_fills']>0)&(df['n_dn_fills']==0)).sum() + ((df['n_dn_fills']>0)&(df['n_up_fills']==0)).sum())} markets) ===")
    unpaired = df[(((df['n_up_fills']>0)&(df['n_dn_fills']==0)) | ((df['n_dn_fills']>0)&(df['n_up_fills']==0)))]
    if len(unpaired):
        print(f"  Total P&L:           ${unpaired['pnl'].sum():.2f}")
        print(f"  Avg P&L / market:    ${unpaired['pnl'].mean():.4f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-days", type=int, default=None)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--out", type=str, default="utils/mm_static.parquet")
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--levels", type=str, default="0.45,0.40,0.35,0.30,0.25,0.20,0.15,0.10",
                    help="Comma-separated price levels")
    ap.add_argument("--size", type=float, default=5.0)
    ap.add_argument("--cap", type=float, default=100.0)
    ap.add_argument("--refill", action="store_true")
    ap.add_argument("--window", type=float, default=300.0)
    ap.add_argument("--stop-loss", type=float, default=0.0,
                    help="Stop-loss threshold (taker sell when bid < this)")
    ap.add_argument("--sl-mode", choices=("residual", "all"), default="residual",
                    help="Stop-loss applies to: residual only or all shares of side")
    ap.add_argument("--close-at", type=float, default=0.0,
                    help="End-of-window taker close at t=N seconds (0=disabled)")
    ap.add_argument("--close-mode", choices=("residual", "all"), default="residual")
    ap.add_argument("--skew-cancel", type=int, default=0,
                    help="Block fills on the side leading by N+ shares (0=disabled)")
    args = ap.parse_args()

    levels = tuple(float(x) for x in args.levels.split(",") if x.strip())
    cfg = StaticPolicy(
        levels=levels, size_per_level=args.size,
        max_inv_per_side=args.cap, refill=args.refill,
        trade_window_seconds=args.window,
        stop_loss_threshold=args.stop_loss,
        stop_loss_residual_only=(args.sl_mode == "residual"),
        close_at_seconds=args.close_at,
        close_residual_only=(args.close_mode == "residual"),
        skew_cancel_threshold=args.skew_cancel,
    )

    paths = _collect(PRICES_5M_ROOT, args.max_days)
    print(f"[static] paths: {len(paths)}", flush=True)
    print(f"[static] cfg: levels={levels}  size={args.size}  cap={args.cap}  refill={args.refill}  window={args.window}", flush=True)

    cfg_dict = asdict(cfg)
    t0 = time.time()
    results = []
    n_done = 0
    with ProcessPoolExecutor(
        max_workers=args.workers, initializer=_init_worker, initargs=(cfg_dict,)
    ) as ex:
        futs = [ex.submit(_process_one, p) for p in paths]
        for fut in as_completed(futs):
            n_done += 1
            r = fut.result()
            if r is not None:
                results.append(r)
            if n_done % 5000 == 0 or n_done == len(paths):
                print(f"  {n_done}/{len(paths)} valid={len(results)} ({time.time()-t0:.1f}s)", flush=True)

    df = pd.DataFrame(results)
    print(f"\n[static] valid: {len(df)} ({time.time()-t0:.1f}s)")
    if df.empty:
        return 0

    if not args.no_save:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(args.out, compression="zstd")

    print_summary(df, cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
