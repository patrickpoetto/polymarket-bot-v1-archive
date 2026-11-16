"""
BBO-follow GATED su bid_sum. Idea:
  - Piazziamo maker BUY su entrambi i lati a (BBO_bid - offset)
  - I fill sono PERMESSI solo quando bid_sum CORRENTE <= bid_sum_threshold
    (cioè il mercato sta offrendo coppie complessivamente sotto soglia)
  - Quando bid_sum > soglia, sono "cancellati" (no fill).
  - Cap inventory per side.
  - Optional close_at: cancella tutto e taker-sell residual.
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
class GatedPolicy:
    tick: float = 0.01
    bbo_offset_ticks: int = 1
    fill_size: float = 5.0
    max_inv_per_side: float = 200.0
    trade_window_seconds: float = 300.0
    close_at_seconds: float = 0.0
    bid_sum_gate: float = 0.99  # only allow fills when bid_sum <= this
    fee_rate: float = 0.07
    rebate_pct: float = 0.20


@dataclass
class GatedResult:
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
    sl_recovery: float
    sl_taker_fees: float
    payoff_pairs: float
    payoff_residual: float
    pnl: float
    winner: str
    both_sides_filled: bool
    secs_gate_open: float


def simulate_gated(
    wide: pd.DataFrame, cfg: GatedPolicy, market_name: str, date: str
) -> Optional[GatedResult]:
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

    offset = cfg.bbo_offset_ticks * cfg.tick
    rebate_factor = cfg.rebate_pct * cfg.fee_rate

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

    sl_recovery = 0.0
    sl_taker_fees = 0.0
    closed = False
    secs_gate_open = 0.0

    prev_ub = up_bid[0]
    prev_db = dn_bid[0]
    prev_t = t_ms[0]

    win_ms = int(cfg.trade_window_seconds * 1000.0)
    close_ms = int(cfg.close_at_seconds * 1000.0) if cfg.close_at_seconds > 0 else None

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

        if close_ms is not None and not closed and elapsed_ms >= close_ms:
            ub_close = up_bid[i]
            db_close = dn_bid[i]
            up_excess = max(0.0, up_shares - dn_shares)
            dn_excess = max(0.0, dn_shares - up_shares)
            if up_excess > 0 and ub_close > 0:
                sl_recovery += ub_close * up_excess
                sl_taker_fees += cfg.fee_rate * ub_close * (1 - ub_close) * up_excess
                up_shares -= up_excess
            if dn_excess > 0 and db_close > 0:
                sl_recovery += db_close * dn_excess
                sl_taker_fees += cfg.fee_rate * db_close * (1 - db_close) * dn_excess
                dn_shares -= dn_excess
            closed = True
            break

        ub = up_bid[i]
        db = dn_bid[i]

        # GATE: only allow fills when bid_sum at PREV tick (when our maker was
        # sitting at prev_ub-offset and prev_db-offset) was <= threshold.
        bid_sum_prev = prev_ub + prev_db
        gate_open_prev = bid_sum_prev <= cfg.bid_sum_gate

        # Track time gate is open
        dt = (t_ms[i] - prev_t) / 1000.0
        if gate_open_prev:
            secs_gate_open += dt

        # UP fill
        if gate_open_prev and up_shares < cfg.max_inv_per_side - EPS and ub < prev_ub - EPS:
            old_order_px = prev_ub - offset
            if old_order_px > 0 and ub < old_order_px - EPS:
                fill_px = old_order_px
                sz = min(cfg.fill_size, cfg.max_inv_per_side - up_shares)
                cash_spent += fill_px * sz
                up_shares += sz
                rebate_total += rebate_factor * fill_px * (1 - fill_px) * sz
                sum_up_px += fill_px * sz
                sum_up_sz += sz
                n_up_fills += 1

        if gate_open_prev and dn_shares < cfg.max_inv_per_side - EPS and db < prev_db - EPS:
            old_order_px = prev_db - offset
            if old_order_px > 0 and db < old_order_px - EPS:
                fill_px = old_order_px
                sz = min(cfg.fill_size, cfg.max_inv_per_side - dn_shares)
                cash_spent += fill_px * sz
                dn_shares += sz
                rebate_total += rebate_factor * fill_px * (1 - fill_px) * sz
                sum_dn_px += fill_px * sz
                sum_dn_sz += sz
                n_dn_fills += 1

        prev_ub = ub
        prev_db = db
        prev_t = t_ms[i]

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

    return GatedResult(
        market_name=market_name, date=date,
        n_up_fills=n_up_fills, n_dn_fills=n_dn_fills,
        up_shares=up_shares, dn_shares=dn_shares,
        pairs=pairs, up_residual=up_res, dn_residual=dn_res,
        avg_up_price=avg_up, avg_dn_price=avg_dn,
        pair_cost=pair_cost,
        cash_spent=cash_spent, rebate_total=rebate_total,
        sl_recovery=sl_recovery, sl_taker_fees=sl_taker_fees,
        payoff_pairs=payoff_pairs, payoff_residual=payoff_residual,
        pnl=pnl, winner=winner,
        both_sides_filled=(n_up_fills > 0 and n_dn_fills > 0),
        secs_gate_open=secs_gate_open,
    )


_GLOBAL_CFG: Optional[GatedPolicy] = None


def _init_worker(cfg_dict: dict) -> None:
    global _GLOBAL_CFG
    _GLOBAL_CFG = GatedPolicy(**cfg_dict)


def _process_one(path_str: str) -> Optional[dict]:
    cfg = _GLOBAL_CFG or GatedPolicy()
    p = Path(path_str)
    wide = parse_market_csv(p)
    if wide is None:
        return None
    res = simulate_gated(wide, cfg, p.stem, p.parent.parent.name)
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


def print_summary(df: pd.DataFrame, cfg: GatedPolicy) -> None:
    n = len(df)
    days = df["date"].nunique() if "date" in df.columns else 1
    pnl = df["pnl"]

    print("\n" + "=" * 88)
    print("BBO-FOLLOW GATED ON BID_SUM SUMMARY")
    print("=" * 88)
    print(f"Markets: {n}    Days: {days}    BBO_offset: {cfg.bbo_offset_ticks}t    "
          f"size: {cfg.fill_size}    cap: {cfg.max_inv_per_side}    "
          f"window: {cfg.trade_window_seconds}s    close: {cfg.close_at_seconds}s    "
          f"GATE: bid_sum <= ${cfg.bid_sum_gate}\n")

    print(f"=== HEADLINE ===")
    print(f"  Total P&L:        ${pnl.sum():>10.2f}")
    print(f"  Avg P&L / market: ${pnl.mean():>10.4f}")
    print(f"  Median P&L:       ${pnl.median():>10.4f}")
    print(f"  Per day:          ${pnl.sum()/days:>10.2f}")
    print(f"  Markets P&L>0:    {int((pnl>0).sum())}/{n} ({(pnl>0).mean():.1%})")
    print(f"  Days positive:    {int((df.groupby('date')['pnl'].sum()>0).sum())}/{days}")

    paired = df[df["both_sides_filled"]]
    unpaired_one = df[(~df["both_sides_filled"]) & (df["n_up_fills"] + df["n_dn_fills"] > 0)]
    untouched = df[(df["n_up_fills"] == 0) & (df["n_dn_fills"] == 0)]
    print(f"\n=== FILL STATS ===")
    print(f"  Both sides filled:      {len(paired)} ({len(paired)/n:.1%})")
    print(f"  Only ONE side filled:   {len(unpaired_one)} ({len(unpaired_one)/n:.1%})")
    print(f"  Untouched (gate never): {len(untouched)} ({len(untouched)/n:.1%})")
    if len(paired):
        print(f"  Pair cost (paired):     ${paired['pair_cost'].mean():.4f}")
    print(f"  Avg gate-open time:     {df['secs_gate_open'].mean():.1f}s of 300s")
    print(f"  Avg UP fills:           {df['n_up_fills'].mean():.2f}")
    print(f"  Avg DN fills:           {df['n_dn_fills'].mean():.2f}")
    print(f"  Avg pairs (per market): {df['pairs'].mean():.2f}")

    if len(paired):
        print(f"\n=== PAIRED ({len(paired)}): ===")
        print(f"  Total P&L:        ${paired['pnl'].sum():.2f}")
        print(f"  Avg P&L / market: ${paired['pnl'].mean():.4f}")
        print(f"  Per day:          ${paired['pnl'].sum()/days:.2f}")
    if len(unpaired_one):
        print(f"\n=== ONE-SIDE-ONLY ({len(unpaired_one)}): ===")
        print(f"  Total P&L:        ${unpaired_one['pnl'].sum():.2f}")
        print(f"  Avg P&L / market: ${unpaired_one['pnl'].mean():.4f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-days", type=int, default=None)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--out", type=str, default="utils/mm_bbo_gated.parquet")
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--bbo-offset", type=int, default=1)
    ap.add_argument("--size", type=float, default=5.0)
    ap.add_argument("--cap", type=float, default=200.0)
    ap.add_argument("--window", type=float, default=300.0)
    ap.add_argument("--close-at", type=float, default=0.0)
    ap.add_argument("--gate", type=float, default=0.99)
    args = ap.parse_args()

    cfg = GatedPolicy(
        bbo_offset_ticks=args.bbo_offset, fill_size=args.size,
        max_inv_per_side=args.cap,
        trade_window_seconds=args.window,
        close_at_seconds=args.close_at,
        bid_sum_gate=args.gate,
    )

    paths = _collect(PRICES_5M_ROOT, args.max_days)
    print(f"[bbo_gated] paths: {len(paths)}", flush=True)

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
    print(f"\n[bbo_gated] valid: {len(df)} ({time.time()-t0:.1f}s)")
    if df.empty:
        return 0

    if not args.no_save:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(args.out, compression="zstd")

    print_summary(df, cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
