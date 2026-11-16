"""
PAIR COMPLETION strategy.

Idea (esattamente la tua):
  1. Piazza maker BUY su entrambi i lati a (BBO_bid - offset) — caccia il primo fill.
  2. Quando una gamba si fillа (es. UP @ P_up), CANCELLA il maker su UP.
  3. RI-PIAZZA il maker su DN ad un prezzo TARGET = (target_pair_cost - P_up).
     - Es: P_up=$0.48, target=$0.96 → maker DN a $0.48 (deve scendere lì per fillare).
  4. Aspetta T secondi che il second leg si fillа.
  5. Se non fillа entro T → TAKER close su DN (compra al ask) per chiudere la coppia.
     Costo: ask_DN al momento. Pair cost = P_up + ask_DN. Possibilmente > $1 = perdita.
  6. Se fillа → pair completata a target.

Inventory: cap sul massimo numero di "round" attivi.
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
class CompletionPolicy:
    tick: float = 0.01
    bbo_offset_ticks: int = 1
    fill_size: float = 5.0
    max_open_rounds: int = 3            # max round simultanei in attesa di completion
    max_completed_pairs: int = 50       # cap totale pair completate
    trade_window_seconds: float = 300.0
    completion_target: float = 0.97     # target pair cost per round completata
    completion_timeout_s: float = 30.0  # se second leg non fillа entro T sec → taker close
    abort_above_pair_cost: float = 1.05 # se costo pair forzato (P1+ask) > soglia, aborta? (taker anyway)
    fee_rate: float = 0.07
    rebate_pct: float = 0.20


@dataclass
class CompletionResult:
    market_name: str
    date: str
    n_rounds_started: int
    n_rounds_maker_completed: int      # pair fillata interamente da maker (target)
    n_rounds_taker_completed: int      # second leg via taker (timeout)
    n_rounds_unclosed: int             # window finita prima di completare (residual = single leg)
    pairs: float
    up_residual: float                 # shares ancora unmatched al fine market
    dn_residual: float
    avg_pair_cost: float
    cash_spent: float                  # cash totale (sia maker che taker buy)
    rebate_total: float
    taker_fees_total: float
    payoff_pairs: float
    payoff_residual: float
    pnl: float
    winner: str


def simulate_completion(
    wide: pd.DataFrame, cfg: CompletionPolicy, market_name: str, date: str
) -> Optional[CompletionResult]:
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

    # Open rounds tracking:
    # round = {leg1_side: "UP"|"DN", leg1_px: float, leg1_size: float, started_at_ms: int,
    #          target_px2: float}
    open_rounds: list[dict] = []

    # Stats
    n_rounds_started = 0
    n_rounds_maker_completed = 0
    n_rounds_taker_completed = 0
    pairs_total = 0.0
    cash_spent = 0.0
    rebate_total = 0.0
    taker_fees_total = 0.0
    sum_pair_cost_weighted = 0.0
    sum_pair_size = 0.0

    # Residual single-leg held to expiry
    up_residual = 0.0
    dn_residual = 0.0

    prev_ub = up_bid[0]
    prev_db = dn_bid[0]

    win_ms = int(cfg.trade_window_seconds * 1000.0)

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

        ub = up_bid[i]
        ua = up_ask[i]
        db = dn_bid[i]
        da = dn_ask[i]

        # ============ COMPLETION CHECK for open rounds ============
        # For each open round, check if leg2 maker fills (target_px2 reachable via bid drop)
        rounds_to_remove = []
        for r in open_rounds:
            leg2 = "DN" if r["leg1_side"] == "UP" else "UP"
            target = r["target_px2"]
            elapsed_round_s = (t_ms[i] - r["started_at_ms"]) / 1000.0

            # Check timeout → taker close
            if elapsed_round_s >= cfg.completion_timeout_s:
                # Taker buy second leg at ASK
                if leg2 == "DN":
                    fill_px2 = da
                else:
                    fill_px2 = ua
                if fill_px2 <= 0 or fill_px2 >= 1:
                    rounds_to_remove.append(r)
                    continue
                sz = r["leg1_size"]
                cash_spent += fill_px2 * sz
                taker_fee = cfg.fee_rate * fill_px2 * (1 - fill_px2) * sz
                taker_fees_total += taker_fee
                pair_cost = r["leg1_px"] + fill_px2
                pairs_total += sz
                sum_pair_cost_weighted += pair_cost * sz
                sum_pair_size += sz
                n_rounds_taker_completed += 1
                rounds_to_remove.append(r)
                continue

            # Check maker fill on second leg
            if leg2 == "DN":
                # Maker buy DN at target. Fills when DN_bid drops to target.
                if db <= target + EPS and prev_db > target + EPS:
                    fill_px2 = target
                    sz = r["leg1_size"]
                    cash_spent += fill_px2 * sz
                    rebate_total += rebate_factor * fill_px2 * (1 - fill_px2) * sz
                    pair_cost = r["leg1_px"] + fill_px2
                    pairs_total += sz
                    sum_pair_cost_weighted += pair_cost * sz
                    sum_pair_size += sz
                    n_rounds_maker_completed += 1
                    rounds_to_remove.append(r)
                    continue
            else:
                if ub <= target + EPS and prev_ub > target + EPS:
                    fill_px2 = target
                    sz = r["leg1_size"]
                    cash_spent += fill_px2 * sz
                    rebate_total += rebate_factor * fill_px2 * (1 - fill_px2) * sz
                    pair_cost = r["leg1_px"] + fill_px2
                    pairs_total += sz
                    sum_pair_cost_weighted += pair_cost * sz
                    sum_pair_size += sz
                    n_rounds_maker_completed += 1
                    rounds_to_remove.append(r)
                    continue

        for r in rounds_to_remove:
            open_rounds.remove(r)

        # ============ NEW LEG-1 FILL CHECK ============
        # Only initiate new round if we have capacity
        # If pairs total already >= cap, stop
        if pairs_total >= cfg.max_completed_pairs - EPS:
            prev_ub = ub
            prev_db = db
            continue

        if len(open_rounds) >= cfg.max_open_rounds:
            prev_ub = ub
            prev_db = db
            continue

        # UP leg1 fill check
        if ub < prev_ub - EPS:
            old_order_px = prev_ub - offset
            if old_order_px > 0 and ub < old_order_px - EPS:
                fill_px = old_order_px
                # Check that this is "cheap enough" for round
                # Required: target_px2 = completion_target - fill_px
                target_px2 = cfg.completion_target - fill_px
                if target_px2 > 0 and target_px2 < 1:
                    sz = cfg.fill_size
                    cash_spent += fill_px * sz
                    rebate_total += rebate_factor * fill_px * (1 - fill_px) * sz
                    open_rounds.append({
                        "leg1_side": "UP",
                        "leg1_px": fill_px,
                        "leg1_size": sz,
                        "started_at_ms": int(t_ms[i]),
                        "target_px2": target_px2,
                    })
                    n_rounds_started += 1

        # DN leg1 fill check
        if db < prev_db - EPS and len(open_rounds) < cfg.max_open_rounds and pairs_total < cfg.max_completed_pairs - EPS:
            old_order_px = prev_db - offset
            if old_order_px > 0 and db < old_order_px - EPS:
                fill_px = old_order_px
                target_px2 = cfg.completion_target - fill_px
                if target_px2 > 0 and target_px2 < 1:
                    sz = cfg.fill_size
                    cash_spent += fill_px * sz
                    rebate_total += rebate_factor * fill_px * (1 - fill_px) * sz
                    open_rounds.append({
                        "leg1_side": "DN",
                        "leg1_px": fill_px,
                        "leg1_size": sz,
                        "started_at_ms": int(t_ms[i]),
                        "target_px2": target_px2,
                    })
                    n_rounds_started += 1

        prev_ub = ub
        prev_db = db

    # End of window: any open rounds that aren't closed → residual
    n_rounds_unclosed = len(open_rounds)
    for r in open_rounds:
        if r["leg1_side"] == "UP":
            up_residual += r["leg1_size"]
        else:
            dn_residual += r["leg1_size"]

    avg_pair_cost = sum_pair_cost_weighted / sum_pair_size if sum_pair_size > 0 else float("nan")
    payoff_pairs = pairs_total * 1.0
    if winner == "UP":
        payoff_residual = up_residual * 1.0
    elif winner == "DOWN":
        payoff_residual = dn_residual * 1.0
    else:
        payoff_residual = 0.5 * (up_residual + dn_residual)

    pnl = payoff_pairs + payoff_residual - cash_spent + rebate_total - taker_fees_total

    return CompletionResult(
        market_name=market_name, date=date,
        n_rounds_started=n_rounds_started,
        n_rounds_maker_completed=n_rounds_maker_completed,
        n_rounds_taker_completed=n_rounds_taker_completed,
        n_rounds_unclosed=n_rounds_unclosed,
        pairs=pairs_total,
        up_residual=up_residual, dn_residual=dn_residual,
        avg_pair_cost=avg_pair_cost,
        cash_spent=cash_spent, rebate_total=rebate_total,
        taker_fees_total=taker_fees_total,
        payoff_pairs=payoff_pairs, payoff_residual=payoff_residual,
        pnl=pnl, winner=winner,
    )


_GLOBAL_CFG: Optional[CompletionPolicy] = None


def _init_worker(cfg_dict: dict) -> None:
    global _GLOBAL_CFG
    _GLOBAL_CFG = CompletionPolicy(**cfg_dict)


def _process_one(path_str: str) -> Optional[dict]:
    cfg = _GLOBAL_CFG or CompletionPolicy()
    p = Path(path_str)
    wide = parse_market_csv(p)
    if wide is None:
        return None
    res = simulate_completion(wide, cfg, p.stem, p.parent.parent.name)
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


def print_summary(df: pd.DataFrame, cfg: CompletionPolicy) -> None:
    n = len(df)
    days = df["date"].nunique() if "date" in df.columns else 1
    pnl = df["pnl"]

    print("\n" + "=" * 88)
    print("PAIR COMPLETION STRATEGY SUMMARY")
    print("=" * 88)
    print(f"Markets: {n}    Days: {days}    offset: {cfg.bbo_offset_ticks}t    "
          f"size: {cfg.fill_size}    open_rounds: {cfg.max_open_rounds}    "
          f"target: ${cfg.completion_target}    timeout: {cfg.completion_timeout_s}s    "
          f"window: {cfg.trade_window_seconds}s\n")

    print(f"=== HEADLINE ===")
    print(f"  Total P&L:        ${pnl.sum():>10.2f}")
    print(f"  Avg P&L / market: ${pnl.mean():>10.4f}")
    print(f"  Median P&L:       ${pnl.median():>10.4f}")
    print(f"  Per day:          ${pnl.sum()/days:>10.2f}")
    print(f"  Markets P&L>0:    {int((pnl>0).sum())}/{n} ({(pnl>0).mean():.1%})")
    print(f"  Days positive:    {int((df.groupby('date')['pnl'].sum()>0).sum())}/{days}")

    nrs = df["n_rounds_started"].sum()
    nm = df["n_rounds_maker_completed"].sum()
    nt = df["n_rounds_taker_completed"].sum()
    nu = df["n_rounds_unclosed"].sum()
    print(f"\n=== ROUND STATS ===")
    print(f"  Total rounds started:   {int(nrs)}    Avg/market: {nrs/n:.2f}")
    print(f"  Maker-completed:        {int(nm)} ({nm/max(1,nrs):.1%})")
    print(f"  Taker-completed:        {int(nt)} ({nt/max(1,nrs):.1%})")
    print(f"  Unclosed (window end):  {int(nu)} ({nu/max(1,nrs):.1%})")
    print(f"  Avg pair cost:          ${df['avg_pair_cost'].mean():.4f}")
    print(f"  Total pairs completed:  {df['pairs'].sum():.0f}")

    # Decompose P&L
    pair_pnl = df["payoff_pairs"].sum() - (df["cash_spent"].sum() - df["payoff_residual"].sum())
    print(f"\n=== P&L DECOMP ===")
    print(f"  Pair payoff:        ${df['payoff_pairs'].sum():>10.2f}")
    print(f"  Residual payoff:    ${df['payoff_residual'].sum():>10.2f}")
    print(f"  Cash spent:         ${df['cash_spent'].sum():>10.2f}")
    print(f"  Rebate total:       ${df['rebate_total'].sum():>10.2f}")
    print(f"  Taker fees:         ${df['taker_fees_total'].sum():>10.2f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-days", type=int, default=None)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--out", type=str, default="utils/mm_completion.parquet")
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--bbo-offset", type=int, default=1)
    ap.add_argument("--size", type=float, default=5.0)
    ap.add_argument("--max-rounds", type=int, default=3)
    ap.add_argument("--max-pairs", type=int, default=50)
    ap.add_argument("--target", type=float, default=0.97)
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--window", type=float, default=300.0)
    args = ap.parse_args()

    cfg = CompletionPolicy(
        bbo_offset_ticks=args.bbo_offset, fill_size=args.size,
        max_open_rounds=args.max_rounds,
        max_completed_pairs=args.max_pairs,
        completion_target=args.target,
        completion_timeout_s=args.timeout,
        trade_window_seconds=args.window,
    )

    paths = _collect(PRICES_5M_ROOT, args.max_days)
    print(f"[completion] paths: {len(paths)}", flush=True)

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
    print(f"\n[completion] valid: {len(df)} ({time.time()-t0:.1f}s)")
    if df.empty:
        return 0

    if not args.no_save:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(args.out, compression="zstd")

    print_summary(df, cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
