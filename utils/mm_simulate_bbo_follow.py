"""
BBO-following maker simulator: l'interpretazione PULITA della strategia
descritta dall'utente.

Per ciascun mercato 5-min:
  - In ogni tick: maintain un maker UP a BBO_up - 1 tick e un maker DN a
    BBO_dn - 1 tick. Cancella e re-place quando BBO si muove.
  - Quando un lato filla, immediatamente piazza un nuovo ordine sulla
    stessa gamba al nuovo BBO-1 (continua ad accumulare).
  - Per il "catch-up" sul lato mancante (single-leg), opzionalmente
    aggiungiamo ordini "shallow ladder" a BBO-2, BBO-5, BBO-10 ticks
    per catturare drop più ampi.
  - Fill model: back-of-queue (fill solo se BBO[i] < order_px - EPS).
  - Cap inventario per lato (default 200 shares per side).

Obiettivo principale: misurare l'EMPIRICAL `avg_pair_cost` con questa
strategia. Confronto con la soglia di redditività (~ 1 - rebate ≈ 0.992).

Uso:
    python -m utils.mm_simulate_bbo_follow --max-days 7 --no-save
    python -m utils.mm_simulate_bbo_follow --max-days 7 --no-save --catchup-ladder
    python -m utils.mm_simulate_bbo_follow --tick 0.01 --fill-size 5
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from utils.mm_analyze_5m import parse_market_csv, PRICES_5M_ROOT  # type: ignore


@dataclass(frozen=True)
class BBOPolicy:
    tick: float = 0.01
    fill_size: float = 5.0  # shares per fill
    max_inv_per_side: float = 200.0
    bbo_offset_ticks: int = 1  # place at BBO - N*tick

    # "Catchup ladder": quando un lato è in single-leg deficit, aggiungi
    # ordini extra a BBO - 2*tick, BBO - 5*tick, BBO - 10*tick.
    catchup_ladder: bool = False

    # Skew limit: non piazzare ordini sul lato X se shares_X > shares_other
    # + skew_limit_shares. Forza il "completamento coppia". 0 = off.
    skew_limit_shares: float = 0.0

    # Trade window: trada solo se elapsed_seconds <= trade_window_seconds.
    # Dopo, cancella tutti gli ordini e tieni l'inventario fino a expiry.
    # 0 = off (trada per l'intero mercato).
    trade_window_seconds: float = 0.0

    # Hedge a fine mercato: hold (let residuals expire) o taker_close.
    hedge_mode: str = "hold"
    hedge_seconds: float = 25.0

    # Fees
    fee_rate: float = 0.07
    rebate_pct: float = 0.20


@dataclass
class BBOResult:
    market_name: str
    date: str

    n_ticks: int
    n_up_fills: int
    n_dn_fills: int
    up_shares: float
    dn_shares: float
    pairs: float
    up_residual: float
    dn_residual: float

    avg_up_buy: float
    avg_dn_buy: float
    avg_pair_cost: float
    pair_cost_vs_avg_bidsum: float  # avg_pair_cost - avg_bid_sum (AS bias)

    cash_spent: float
    rebate_total: float
    taker_fees: float

    payoff_pairs: float
    payoff_residual: float
    pnl: float
    pnl_per_pair: float

    avg_bid_sum: float
    winner: str
    final_up_mid: float
    final_dn_mid: float

    # Per-fill tracking
    sum_up_buy_px_at_fill: float  # sum of prices at fill
    sum_dn_buy_px_at_fill: float
    sum_avg_bid_at_fill_up: float  # comparison: avg(bid) at moment of UP fill
    sum_avg_bid_at_fill_dn: float


def simulate_bbo(
    wide: pd.DataFrame,
    cfg: BBOPolicy,
    market_name: str,
    date: str,
) -> Optional[BBOResult]:
    if wide is None or len(wide) < 10:
        return None

    wide = wide.copy()
    for col in ("up_bid", "up_ask", "dn_bid", "dn_ask"):
        wide[col] = wide[col].ffill()
    wide = wide.dropna(subset=["up_bid", "up_ask", "dn_bid", "dn_ask"]).reset_index(drop=True)
    if len(wide) < 10:
        return None

    up_bid = wide["up_bid"].to_numpy(dtype=np.float64)
    up_ask = wide["up_ask"].to_numpy(dtype=np.float64)
    dn_bid = wide["dn_bid"].to_numpy(dtype=np.float64)
    dn_ask = wide["dn_ask"].to_numpy(dtype=np.float64)
    t_left = wide["t_left"].to_numpy(dtype=np.float64)
    t_ms = wide["t_ms"].to_numpy(dtype=np.int64)
    n = len(wide)
    t_start_ms = int(t_ms[0])

    EPS = 1e-9
    tick = cfg.tick
    offset = cfg.bbo_offset_ticks * tick
    max_inv = cfg.max_inv_per_side
    fill_size = cfg.fill_size
    rebate_factor = cfg.rebate_pct * cfg.fee_rate

    # Stato:
    # Per ogni lato manteniamo una LISTA di livelli con ordini aperti.
    # In modalità non-catchup: 1 ordine per lato, a BBO - offset.
    # In modalità catchup: fino a 4 ordini per lato (BBO-1, BBO-2, BBO-5, BBO-10).
    if cfg.catchup_ladder:
        catchup_offsets = [1, 2, 5, 10]
    else:
        catchup_offsets = [cfg.bbo_offset_ticks]
    catchup_offsets_arr = np.array(catchup_offsets, dtype=np.float64) * tick
    K = len(catchup_offsets)

    # Per ogni livello-tier, il prezzo dell'ordine corrente (-1 se nessuno)
    up_order_px = np.full(K, -1.0, dtype=np.float64)
    dn_order_px = np.full(K, -1.0, dtype=np.float64)

    up_shares = 0.0
    dn_shares = 0.0
    cash_spent = 0.0
    rebate_total = 0.0
    taker_fees = 0.0
    n_up_fills = 0
    n_dn_fills = 0
    sum_up_px = 0.0
    sum_dn_px = 0.0
    sum_up_sz = 0.0
    sum_dn_sz = 0.0

    # Per AS-bias measurement
    sum_avg_bid_at_fill_up = 0.0
    sum_avg_bid_at_fill_dn = 0.0
    n_avg_bid_at_fill_up = 0
    n_avg_bid_at_fill_dn = 0

    # avg bid sum across whole market (for reference)
    bid_sum_sum = 0.0

    # Window taker-close flag: ad fine trade window, chiudi i residui taker
    window_taker_done = False

    prev_up_bid = up_bid[0]
    prev_dn_bid = dn_bid[0]

    for i in range(n):
        ub = up_bid[i]
        db = dn_bid[i]
        bid_sum_sum += ub + db

        # ---- 0) Trade window check: dopo trade_window_seconds, cancella tutto
        elapsed_s = (t_ms[i] - t_start_ms) / 1000.0
        window_blocked = (
            cfg.trade_window_seconds > 0 and elapsed_s > cfg.trade_window_seconds
        )

        # Quando entriamo nello stato window_blocked per la prima volta E
        # hedge_mode è taker_close, chiudi i residui ora (mid-market) invece
        # di portarli a expiry.
        if (
            window_blocked
            and not window_taker_done
            and cfg.hedge_mode == "taker_close"
        ):
            window_taker_done = True
            pairs_now = min(up_shares, dn_shares)
            up_res_now = up_shares - pairs_now
            dn_res_now = dn_shares - pairs_now
            if up_res_now > 0:
                px = float(dn_ask[i])
                if px > 0 and not np.isnan(px):
                    cash_spent += px * up_res_now
                    taker_fees += cfg.fee_rate * px * (1.0 - px) * up_res_now
                    dn_shares += up_res_now
            if dn_res_now > 0:
                px = float(up_ask[i])
                if px > 0 and not np.isnan(px):
                    cash_spent += px * dn_res_now
                    taker_fees += cfg.fee_rate * px * (1.0 - px) * dn_res_now
                    up_shares += dn_res_now

        # ---- 1) Update ordini: cancel & re-place a BBO - offset_k * tick ----
        # Skew check: blocca il lato pieno se shares_X > shares_Y + limit
        skew_block_up = (
            cfg.skew_limit_shares > 0
            and (up_shares - dn_shares) >= cfg.skew_limit_shares - EPS
        ) or window_blocked
        skew_block_dn = (
            cfg.skew_limit_shares > 0
            and (dn_shares - up_shares) >= cfg.skew_limit_shares - EPS
        ) or window_blocked

        # Up side
        if up_shares < max_inv - EPS and not skew_block_up:
            for k_idx, off in enumerate(catchup_offsets_arr):
                target = ub - off
                if target > 0 and target < up_ask[i] - EPS:
                    up_order_px[k_idx] = target
                else:
                    up_order_px[k_idx] = -1.0
        else:
            up_order_px[:] = -1.0

        if dn_shares < max_inv - EPS and not skew_block_dn:
            for k_idx, off in enumerate(catchup_offsets_arr):
                target = db - off
                if target > 0 and target < dn_ask[i] - EPS:
                    dn_order_px[k_idx] = target
                else:
                    dn_order_px[k_idx] = -1.0
        else:
            dn_order_px[:] = -1.0

        # ---- 2) Check fills: BBO è sceso strict-below qualche order_px? ----
        # Per ogni livello, fill se prev_bid >= order_px (era abbastanza alto)
        # AND ub < order_px - EPS (ora è sotto).
        # Order è stato piazzato al tick precedente al prezzo prev_bid - offset
        # (oppure ora a target = ub - offset se ub < prev_bid; ma per fill,
        # serve che il livello abbia avuto tempo di essere in book).

        # In questo modello semplice: facciamo cancel & replace ogni tick, quindi
        # un order può fillare solo se BBO è sceso STRICT NEL TICK CORRENTE
        # rispetto al precedente, e il prezzo precedente del nostro order
        # (= prev_bid - offset) è sopra la nuova BBO.

        # Per il fill check, usiamo lo stato di skew DEL TICK PRECEDENTE
        # (cioè il blocco era effettivo prima di questo tick): skipping
        # logic se eravamo già bloccati. Per semplicità qui ricontrolliamo
        # con lo skew CORRENTE (dopo eventuali aggiornamenti). Se l'ordine
        # NON era piazzato al tick i-1 (era cancellato), il fill non avviene.
        prev_up_blocked = (
            cfg.skew_limit_shares > 0 and (up_shares - dn_shares) >= cfg.skew_limit_shares - EPS
        ) or window_blocked
        prev_dn_blocked = (
            cfg.skew_limit_shares > 0 and (dn_shares - up_shares) >= cfg.skew_limit_shares - EPS
        ) or window_blocked

        if i > 0 and ub < prev_up_bid - EPS and up_shares < max_inv - EPS and not prev_up_blocked:
            for k_idx, off in enumerate(catchup_offsets_arr):
                old_order_px = prev_up_bid - off
                if old_order_px > 0 and ub < old_order_px - EPS and up_shares < max_inv - EPS:
                    fill_px = old_order_px
                    sz = min(fill_size, max_inv - up_shares)
                    cash_spent += fill_px * sz
                    up_shares += sz
                    rebate_total += rebate_factor * fill_px * (1.0 - fill_px) * sz
                    sum_up_px += fill_px * sz
                    sum_up_sz += sz
                    n_up_fills += 1
                    sum_avg_bid_at_fill_up += (prev_up_bid + ub) * 0.5
                    n_avg_bid_at_fill_up += 1

        if i > 0 and db < prev_dn_bid - EPS and dn_shares < max_inv - EPS and not prev_dn_blocked:
            for k_idx, off in enumerate(catchup_offsets_arr):
                old_order_px = prev_dn_bid - off
                if old_order_px > 0 and db < old_order_px - EPS and dn_shares < max_inv - EPS:
                    fill_px = old_order_px
                    sz = min(fill_size, max_inv - dn_shares)
                    cash_spent += fill_px * sz
                    dn_shares += sz
                    rebate_total += rebate_factor * fill_px * (1.0 - fill_px) * sz
                    sum_dn_px += fill_px * sz
                    sum_dn_sz += sz
                    n_dn_fills += 1
                    sum_avg_bid_at_fill_dn += (prev_dn_bid + db) * 0.5
                    n_avg_bid_at_fill_dn += 1

        prev_up_bid = ub
        prev_dn_bid = db

    # End-of-market: pair up
    pairs = min(up_shares, dn_shares)
    up_res = up_shares - pairs
    dn_res = dn_shares - pairs

    final_up_mid = (up_bid[-1] + up_ask[-1]) * 0.5
    final_dn_mid = (dn_bid[-1] + dn_ask[-1]) * 0.5
    if final_up_mid > final_dn_mid:
        winner = "UP"
    elif final_dn_mid > final_up_mid:
        winner = "DOWN"
    else:
        winner = "TIE"

    if cfg.hedge_mode == "taker_close":
        if up_res > 0:
            px = float(dn_ask[-1])
            if px > 0:
                cash_spent += px * up_res
                taker_fees += cfg.fee_rate * px * (1 - px) * up_res
                dn_shares += up_res
        if dn_res > 0:
            px = float(up_ask[-1])
            if px > 0:
                cash_spent += px * dn_res
                taker_fees += cfg.fee_rate * px * (1 - px) * dn_res
                up_shares += dn_res
        pairs = min(up_shares, dn_shares)
        up_res = up_shares - pairs
        dn_res = dn_shares - pairs

    payoff_pairs = pairs * 1.0
    if winner == "UP":
        payoff_residual = up_res * 1.0
    elif winner == "DOWN":
        payoff_residual = dn_res * 1.0
    else:
        payoff_residual = 0.5 * (up_res + dn_res)

    pnl = payoff_pairs + payoff_residual - cash_spent + rebate_total - taker_fees

    avg_up = sum_up_px / sum_up_sz if sum_up_sz > 0 else float("nan")
    avg_dn = sum_dn_px / sum_dn_sz if sum_dn_sz > 0 else float("nan")

    if pairs > 0 and sum_up_sz > 0 and sum_dn_sz > 0:
        avg_pair_cost = avg_up + avg_dn
        pnl_per_pair = pnl / pairs
    else:
        avg_pair_cost = float("nan")
        pnl_per_pair = float("nan")

    avg_bid_sum = bid_sum_sum / n if n > 0 else float("nan")
    pair_cost_vs_bidsum = (avg_pair_cost - avg_bid_sum) if not np.isnan(avg_pair_cost) else float("nan")

    return BBOResult(
        market_name=market_name,
        date=date,
        n_ticks=n,
        n_up_fills=n_up_fills,
        n_dn_fills=n_dn_fills,
        up_shares=up_shares,
        dn_shares=dn_shares,
        pairs=pairs,
        up_residual=up_res,
        dn_residual=dn_res,
        avg_up_buy=avg_up,
        avg_dn_buy=avg_dn,
        avg_pair_cost=avg_pair_cost,
        pair_cost_vs_avg_bidsum=pair_cost_vs_bidsum,
        cash_spent=cash_spent,
        rebate_total=rebate_total,
        taker_fees=taker_fees,
        payoff_pairs=payoff_pairs,
        payoff_residual=payoff_residual,
        pnl=pnl,
        pnl_per_pair=pnl_per_pair,
        avg_bid_sum=avg_bid_sum,
        winner=winner,
        final_up_mid=final_up_mid,
        final_dn_mid=final_dn_mid,
        sum_up_buy_px_at_fill=sum_up_px,
        sum_dn_buy_px_at_fill=sum_dn_px,
        sum_avg_bid_at_fill_up=sum_avg_bid_at_fill_up / max(1, n_avg_bid_at_fill_up),
        sum_avg_bid_at_fill_dn=sum_avg_bid_at_fill_dn / max(1, n_avg_bid_at_fill_dn),
    )


_GLOBAL_CFG: Optional[BBOPolicy] = None


def _init_worker(cfg_dict: dict) -> None:
    global _GLOBAL_CFG
    _GLOBAL_CFG = BBOPolicy(**cfg_dict)


def _process_one(path_str: str) -> Optional[dict]:
    cfg = _GLOBAL_CFG or BBOPolicy()
    p = Path(path_str)
    wide = parse_market_csv(p)
    if wide is None:
        return None
    res = simulate_bbo(wide, cfg, p.stem, p.parent.parent.name)
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-days", type=int, default=None)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--out", type=str, default="utils/mm_bbo_follow_5m.parquet")
    ap.add_argument("--tick", type=float, default=0.01)
    ap.add_argument("--fill-size", type=float, default=5.0)
    ap.add_argument("--max-inv", type=float, default=200.0)
    ap.add_argument("--bbo-offset", type=int, default=1)
    ap.add_argument("--catchup-ladder", action="store_true")
    ap.add_argument("--skew-limit", type=float, default=0.0,
                    help="Max single-side share imbalance before blocking that side. 0=off.")
    ap.add_argument("--trade-window", type=float, default=0.0,
                    help="Trade only in first N seconds of market. 0=off.")
    ap.add_argument("--hedge-mode", choices=("hold", "taker_close"), default="hold")
    args = ap.parse_args()

    cfg = BBOPolicy(
        tick=args.tick,
        fill_size=args.fill_size,
        max_inv_per_side=args.max_inv,
        bbo_offset_ticks=args.bbo_offset,
        catchup_ladder=args.catchup_ladder,
        skew_limit_shares=args.skew_limit,
        trade_window_seconds=args.trade_window,
        hedge_mode=args.hedge_mode,
    )

    paths = _collect(PRICES_5M_ROOT, args.max_days)
    print(f"[bbo] paths: {len(paths)}", flush=True)
    print(f"[bbo] cfg: tick={cfg.tick}  fill_size={cfg.fill_size}  "
          f"max_inv={cfg.max_inv_per_side}  bbo_offset={cfg.bbo_offset_ticks}  "
          f"catchup={cfg.catchup_ladder}  hedge={cfg.hedge_mode}", flush=True)

    cfg_dict = asdict(cfg)
    t0 = time.time()
    results = []
    n_done = 0
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker,
                              initargs=(cfg_dict,)) as ex:
        futs = [ex.submit(_process_one, p) for p in paths]
        for fut in as_completed(futs):
            n_done += 1
            r = fut.result()
            if r is not None:
                results.append(r)
            if n_done % 1000 == 0 or n_done == len(paths):
                el = time.time() - t0
                rate = n_done / el if el > 0 else 0
                print(f"  {n_done}/{len(paths)} valid={len(results)} rate={rate:.1f}/s", flush=True)

    df = pd.DataFrame(results)
    print(f"\n[bbo] valid: {len(df)} ({time.time()-t0:.1f}s)")
    if df.empty:
        return 0

    if not args.no_save:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(args.out, compression="zstd")
        print(f"[bbo] saved -> {args.out}")

    print_summary(df, cfg)
    return 0


def print_summary(df: pd.DataFrame, cfg: BBOPolicy) -> None:
    print("\n" + "=" * 80)
    print("BBO-FOLLOW SIMULATION SUMMARY")
    print("=" * 80)
    n = max(1, len(df))
    pnl = df["pnl"]
    print(f"Markets: {len(df)}  Days: {df['date'].nunique()}")
    print(f"Config: bbo_offset={cfg.bbo_offset_ticks}t  fill_size={cfg.fill_size}  "
          f"catchup={cfg.catchup_ladder}  hedge={cfg.hedge_mode}")
    print()

    print("=== HEADLINE ===")
    print(f"  Total P&L:          ${pnl.sum():>10.2f}")
    print(f"  Avg P&L / market:   ${pnl.mean():>10.4f}")
    print(f"  Median P&L:         ${pnl.median():>10.4f}")
    print(f"  P&L > 0:            {int((pnl>0).sum())} ({(pnl>0).mean():.1%})")
    print(f"  Per day:            ${pnl.sum()/df['date'].nunique():>10.2f}")
    print()

    print("=== ECONOMICS (la verità sul pair cost) ===")
    valid = df.dropna(subset=["avg_pair_cost"])
    if len(valid):
        print(f"  Markets with both fills: {len(valid)} ({len(valid)/n:.1%})")
        print(f"  Avg up_buy_price:        ${valid['avg_up_buy'].mean():.6f}")
        print(f"  Avg dn_buy_price:        ${valid['avg_dn_buy'].mean():.6f}")
        print(f"  Avg pair_cost:           ${valid['avg_pair_cost'].mean():.6f}")
        print(f"  Median pair_cost:        ${valid['avg_pair_cost'].median():.6f}")
        print(f"  Avg bid_sum (whole mkt): ${valid['avg_bid_sum'].mean():.6f}")
        print(f"  AS-bias (cost - bidsum): ${valid['pair_cost_vs_avg_bidsum'].mean():.6f}")
        print()
        print(f"  Avg pnl_per_pair:        ${valid['pnl_per_pair'].mean():.6f}")
        print(f"  Median pnl_per_pair:     ${valid['pnl_per_pair'].median():.6f}")
        print(f"  Avg rebate_total:        ${valid['rebate_total'].mean():.6f}")
    print()

    print("=== INVENTORY ===")
    print(f"  Avg pairs:           {df['pairs'].mean():.2f}  median {df['pairs'].median():.2f}")
    print(f"  Avg up_residual:     {df['up_residual'].mean():.2f}")
    print(f"  Avg dn_residual:     {df['dn_residual'].mean():.2f}")
    print(f"  Avg up_fills count:  {df['n_up_fills'].mean():.2f}")
    print(f"  Avg dn_fills count:  {df['n_dn_fills'].mean():.2f}")
    print()

    print("=== PAIR COST DISTRIBUTION ===")
    valid = df.dropna(subset=["avg_pair_cost"])
    if len(valid):
        q = valid["avg_pair_cost"].quantile([0.05, 0.25, 0.50, 0.75, 0.95]).values
        print(f"  p05={q[0]:.4f}  p25={q[1]:.4f}  p50={q[2]:.4f}  p75={q[3]:.4f}  p95={q[4]:.4f}")
        for thresh in (0.99, 0.995, 1.0, 1.005, 1.01):
            c = int((valid["avg_pair_cost"] <= thresh).sum())
            print(f"  Markets with pair_cost <= {thresh}:  {c} ({c/len(valid):.1%})")


if __name__ == "__main__":
    raise SystemExit(main())
