"""
ADAPTIVE V2 — versione pulita senza swap.

Logica:
1. OBSERVE primi observe_seconds, calcola delta(up_mid).
2. Se |delta| > threshold → favored = UP o DN. Altrimenti NEUTRAL (entrambi BBO-follow).
3. Maker BUY favored @ favored_bid - offset_ticks (BBO-follow): fill su ogni dip del favored.
4. Maker BUY loser @ MIN(target_los, los_bid - offset):
     - target_los = pair_target - avg_fav
     - Se target_los >= los_bid - offset: piazza a los_bid - offset (BBO-follow su loser, sempre piazzabile)
     - Se target_los < los_bid - offset: piazza a target_los (aspetta che bid scenda al target)
   Pair cost = avg_fav + fill_px_loser <= pair_target sempre.
5. NEUTRAL mode: BBO-follow su entrambi (no targeting).
6. Stop-loss a t=stop_loss_seconds: taker-close residual al bid corrente.
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
class Adaptive2Policy:
    tick: float = 0.01
    bbo_offset_ticks: int = 1
    fill_size: float = 5.0
    max_inv_per_side: float = 100.0
    observe_seconds: float = 10.0
    direction_threshold: float = 0.02
    pair_target: float = 0.97
    trade_window_seconds: float = 300.0
    stop_loss_seconds: float = 270.0
    allow_neutral_mode: bool = True
    # Early exit: if we have accumulated >= early_exit_fav_shares on favored
    # and still 0 loser fills, the favored is the actual loser → taker-close.
    early_exit_fav_shares: float = 0.0   # 0 = disabled
    early_exit_min_seconds: float = 30.0
    fee_rate: float = 0.07
    rebate_pct: float = 0.20


@dataclass
class Adaptive2Result:
    market_name: str
    date: str
    observed_delta: float
    initial_favored: str
    n_fav_fills: int
    n_los_fills: int
    fav_shares: float
    los_shares: float
    pairs: float
    avg_fav_price: float
    avg_los_price: float
    avg_pair_cost: float
    cash_spent: float
    rebate_total: float
    taker_fees: float
    sl_recovery: float
    payoff_pairs: float
    payoff_residual: float
    pnl: float
    winner: str
    guess_correct: bool


def simulate_adaptive2(
    wide: pd.DataFrame, cfg: Adaptive2Policy, market_name: str, date: str
) -> Optional[Adaptive2Result]:
    if wide is None or len(wide) < 50:
        return None
    wide = wide.copy()
    for col in ("up_bid", "up_ask", "dn_bid", "dn_ask"):
        wide[col] = wide[col].ffill()
    wide = wide.dropna(subset=["up_bid", "up_ask", "dn_bid", "dn_ask"]).reset_index(drop=True)
    if len(wide) < 50:
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

    up_mid_open = (up_bid[0] + up_ask[0]) * 0.5
    final_up_mid = (up_bid[-1] + up_ask[-1]) * 0.5
    final_dn_mid = (dn_bid[-1] + dn_ask[-1]) * 0.5
    if final_up_mid > final_dn_mid:
        winner = "UP"
    elif final_dn_mid > final_up_mid:
        winner = "DOWN"
    else:
        winner = "TIE"

    favored: Optional[str] = None
    initial_favored = "?"
    observed_delta = 0.0

    # On UP-side and DN-side inventory tracked separately (no swap)
    up_shares = 0.0
    dn_shares = 0.0
    sum_up_px = 0.0
    sum_dn_px = 0.0
    n_up_fills = 0
    n_dn_fills = 0

    cash_spent = 0.0
    rebate_total = 0.0
    taker_fees = 0.0
    sl_recovery = 0.0

    prev_ub = up_bid[0]
    prev_db = dn_bid[0]

    win_ms = int(cfg.trade_window_seconds * 1000.0)
    obs_ms = int(cfg.observe_seconds * 1000.0)
    sl_ms = int(cfg.stop_loss_seconds * 1000.0)
    closed = False

    # Per-tick maker prices (where our orders sit at start of tick i, having been placed
    # in tick i-1 based on the prices at end of tick i-1)
    maker_up_px = -1.0   # -1 = no maker
    maker_dn_px = -1.0

    for i in range(1, n):
        elapsed_ms = t_ms[i] - t_start
        if elapsed_ms > win_ms:
            break

        ub = up_bid[i]
        ua = up_ask[i]
        db = dn_bid[i]
        da = dn_ask[i]
        up_mid = (ub + ua) * 0.5

        # PHASE 1: Observation
        if favored is None:
            if elapsed_ms >= obs_ms:
                observed_delta = up_mid - up_mid_open
                if observed_delta > cfg.direction_threshold:
                    favored = "UP"
                elif observed_delta < -cfg.direction_threshold:
                    favored = "DN"
                else:
                    favored = "NEUTRAL" if cfg.allow_neutral_mode else "UP"
                initial_favored = favored
            prev_ub = ub
            prev_db = db
            continue

        # EARLY EXIT: if we have lots of fav shares and no los fills, we're wrong
        if (not closed and cfg.early_exit_fav_shares > 0
                and elapsed_ms >= cfg.early_exit_min_seconds * 1000.0
                and favored in ("UP", "DN")):
            if favored == "UP":
                fav_inv = up_shares
                los_inv = dn_shares
            else:
                fav_inv = dn_shares
                los_inv = up_shares
            if fav_inv >= cfg.early_exit_fav_shares - EPS and los_inv < EPS:
                # Taker-close fav side at current bid
                if favored == "UP":
                    avg_up_pre = sum_up_px / up_shares if up_shares > 0 else 0.0
                    sl_recovery += ub * up_shares
                    taker_fees += cfg.fee_rate * ub * (1 - ub) * up_shares
                    sum_up_px -= avg_up_pre * up_shares
                    up_shares = 0.0
                else:
                    avg_dn_pre = sum_dn_px / dn_shares if dn_shares > 0 else 0.0
                    sl_recovery += db * dn_shares
                    taker_fees += cfg.fee_rate * db * (1 - db) * dn_shares
                    sum_dn_px -= avg_dn_pre * dn_shares
                    dn_shares = 0.0
                closed = True
                break

        # PHASE 7: Stop-loss / final close
        if not closed and elapsed_ms >= sl_ms:
            paired = min(up_shares, dn_shares)
            up_excess = max(0.0, up_shares - paired)
            dn_excess = max(0.0, dn_shares - paired)
            if up_excess > 0 and ub > 0 and up_shares > 0:
                avg_up_pre = sum_up_px / up_shares
                sl_recovery += ub * up_excess
                taker_fees += cfg.fee_rate * ub * (1 - ub) * up_excess
                sum_up_px -= avg_up_pre * up_excess
                up_shares -= up_excess
            if dn_excess > 0 and db > 0 and dn_shares > 0:
                avg_dn_pre = sum_dn_px / dn_shares
                sl_recovery += db * dn_excess
                taker_fees += cfg.fee_rate * db * (1 - db) * dn_excess
                sum_dn_px -= avg_dn_pre * dn_excess
                dn_shares -= dn_excess
            closed = True
            break

        # === FILL CHECKS based on prices placed at end of last tick ===
        # If our maker_up_px was set, fill if ub <= maker_up_px (price crossed it down)
        if maker_up_px > 0 and up_shares < cfg.max_inv_per_side - EPS:
            if ub <= maker_up_px - EPS:
                fill_px = maker_up_px
                sz = min(cfg.fill_size, cfg.max_inv_per_side - up_shares)
                cash_spent += fill_px * sz
                up_shares += sz
                sum_up_px += fill_px * sz
                rebate_total += rebate_factor * fill_px * (1 - fill_px) * sz
                n_up_fills += 1
        if maker_dn_px > 0 and dn_shares < cfg.max_inv_per_side - EPS:
            if db <= maker_dn_px - EPS:
                fill_px = maker_dn_px
                sz = min(cfg.fill_size, cfg.max_inv_per_side - dn_shares)
                cash_spent += fill_px * sz
                dn_shares += sz
                sum_dn_px += fill_px * sz
                rebate_total += rebate_factor * fill_px * (1 - fill_px) * sz
                n_dn_fills += 1

        # === COMPUTE AVG FAV / TARGET ===
        # Identify favored bid and loser bid based on current label
        if favored == "UP":
            fav_shares_curr = up_shares
            sum_fav_curr = sum_up_px
            los_shares_curr = dn_shares
            fav_bid = ub
            los_bid = db
        elif favored == "DN":
            fav_shares_curr = dn_shares
            sum_fav_curr = sum_dn_px
            los_shares_curr = up_shares
            fav_bid = db
            los_bid = ub
        else:  # NEUTRAL
            fav_bid = ub
            los_bid = db
            fav_shares_curr = 0
            sum_fav_curr = 0
            los_shares_curr = 0

        # === RE-PLACE MAKER ORDERS based on current state ===
        # Favored maker: always at BBO-1 (BBO-follow). Always placeable (< current ask).
        new_fav_px = fav_bid - offset
        if new_fav_px <= 0 or new_fav_px >= 1:
            new_fav_px = -1

        # Loser maker:
        if favored == "NEUTRAL":
            new_los_px = los_bid - offset
        else:
            if fav_shares_curr > 0:
                avg_fav = sum_fav_curr / fav_shares_curr
                target_los = cfg.pair_target - avg_fav
                bbo_los_max = los_bid - offset  # BBO-follow on loser side
                # Buy at the LOWER of (target, BBO-1). Always <= ask, hence placeable maker.
                new_los_px = min(target_los, bbo_los_max)
                if new_los_px <= 0 or new_los_px >= 1:
                    new_los_px = -1
            else:
                # No favored fill yet → don't place on loser (single-leg-on-favored phase)
                new_los_px = -1

        if new_los_px <= 0 or new_los_px >= 1:
            new_los_px = -1

        # Update our maker prices for NEXT tick's fill check
        if favored == "UP":
            maker_up_px = new_fav_px
            maker_dn_px = new_los_px
        elif favored == "DN":
            maker_dn_px = new_fav_px
            maker_up_px = new_los_px
        else:  # NEUTRAL
            maker_up_px = new_fav_px  # BBO-follow up
            maker_dn_px = new_los_px  # BBO-follow dn

        prev_ub = ub
        prev_db = db

    pairs = min(up_shares, dn_shares)
    up_residual = up_shares - pairs
    dn_residual = dn_shares - pairs

    avg_up = sum_up_px / up_shares if up_shares > 0 else 0.0
    avg_dn = sum_dn_px / dn_shares if dn_shares > 0 else 0.0
    avg_pair_cost = (avg_up + avg_dn) if (up_shares > 0 and dn_shares > 0) else float("nan")

    payoff_pairs = pairs * 1.0
    if winner == "UP":
        payoff_residual = up_residual * 1.0
    elif winner == "DOWN":
        payoff_residual = dn_residual * 1.0
    else:
        payoff_residual = 0.5 * (up_residual + dn_residual)

    pnl = (payoff_pairs + payoff_residual + sl_recovery + rebate_total
           - cash_spent - taker_fees)

    if initial_favored == "UP":
        guess_correct = (winner == "UP")
    elif initial_favored == "DN":
        guess_correct = (winner == "DOWN")
    else:
        guess_correct = False

    if favored == "UP" or favored is None or favored == "NEUTRAL":
        avg_fav_price = avg_up
        avg_los_price = avg_dn
        n_fav_fills_o = n_up_fills
        n_los_fills_o = n_dn_fills
        fav_shares_o = up_shares
        los_shares_o = dn_shares
    else:
        avg_fav_price = avg_dn
        avg_los_price = avg_up
        n_fav_fills_o = n_dn_fills
        n_los_fills_o = n_up_fills
        fav_shares_o = dn_shares
        los_shares_o = up_shares

    return Adaptive2Result(
        market_name=market_name, date=date,
        observed_delta=observed_delta,
        initial_favored=initial_favored,
        n_fav_fills=n_fav_fills_o,
        n_los_fills=n_los_fills_o,
        fav_shares=fav_shares_o,
        los_shares=los_shares_o,
        pairs=pairs,
        avg_fav_price=avg_fav_price,
        avg_los_price=avg_los_price,
        avg_pair_cost=avg_pair_cost,
        cash_spent=cash_spent, rebate_total=rebate_total,
        taker_fees=taker_fees, sl_recovery=sl_recovery,
        payoff_pairs=payoff_pairs, payoff_residual=payoff_residual,
        pnl=pnl, winner=winner,
        guess_correct=guess_correct,
    )


_GLOBAL_CFG: Optional[Adaptive2Policy] = None


def _init_worker(cfg_dict: dict) -> None:
    global _GLOBAL_CFG
    _GLOBAL_CFG = Adaptive2Policy(**cfg_dict)


def _process_one(path_str: str) -> Optional[dict]:
    cfg = _GLOBAL_CFG or Adaptive2Policy()
    p = Path(path_str)
    wide = parse_market_csv(p)
    if wide is None:
        return None
    res = simulate_adaptive2(wide, cfg, p.stem, p.parent.parent.name)
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


def print_summary(df: pd.DataFrame, cfg: Adaptive2Policy) -> None:
    n = len(df)
    days = df["date"].nunique() if "date" in df.columns else 1
    pnl = df["pnl"]

    print("\n" + "=" * 92)
    print("ADAPTIVE V2 — SUMMARY")
    print("=" * 92)
    print(f"Markets: {n}    Days: {days}    obs={cfg.observe_seconds}s    "
          f"thresh={cfg.direction_threshold}    target=${cfg.pair_target}    "
          f"SL@{cfg.stop_loss_seconds}s    cap={cfg.max_inv_per_side}    size={cfg.fill_size}\n")

    print(f"=== HEADLINE ===")
    print(f"  Total P&L:        ${pnl.sum():>10.2f}")
    print(f"  Avg P&L / market: ${pnl.mean():>10.4f}")
    print(f"  Median P&L:       ${pnl.median():>10.4f}")
    print(f"  Per day:          ${pnl.sum()/days:>10.2f}")
    print(f"  Markets P&L>0:    {int((pnl>0).sum())}/{n} ({(pnl>0).mean():.1%})")
    print(f"  Days positive:    {int((df.groupby('date')['pnl'].sum()>0).sum())}/{days}")

    print(f"\n=== BY INITIAL GUESS ===")
    for g in ("UP", "DN", "NEUTRAL"):
        sub = df[df["initial_favored"] == g]
        if len(sub) == 0:
            continue
        correct = sub["guess_correct"].sum()
        print(f"  {g:>8}: n={len(sub):4d} ({len(sub)/n:.1%})  "
              f"correct={correct} ({correct/max(1,len(sub)):.1%})  "
              f"avg P&L=${sub['pnl'].mean():+.4f}  "
              f"total=${sub['pnl'].sum():+.2f}")

    correct = df[df["guess_correct"]]
    wrong = df[(~df["guess_correct"]) & (df["initial_favored"].isin(["UP","DN"]))]
    if len(correct):
        print(f"\n=== DIRECTIONAL OUTCOMES ===")
        print(f"  CORRECT  guess: n={len(correct):4d}  avg P&L=${correct['pnl'].mean():+.4f}  "
              f"total=${correct['pnl'].sum():+.2f}")
    if len(wrong):
        print(f"  WRONG    guess: n={len(wrong):4d}  avg P&L=${wrong['pnl'].mean():+.4f}  "
              f"total=${wrong['pnl'].sum():+.2f}")

    valid_pair = df[(df["pairs"] > 0) & df["avg_pair_cost"].notna()]
    print(f"\n=== PAIR STATS ===")
    print(f"  Markets with ≥1 pair: {len(valid_pair)} ({len(valid_pair)/n:.1%})")
    if len(valid_pair):
        print(f"  Avg pair cost:        ${valid_pair['avg_pair_cost'].mean():.4f}")
        print(f"  Median pair cost:     ${valid_pair['avg_pair_cost'].median():.4f}")
        print(f"  Avg pairs/market:     {df['pairs'].mean():.2f}")
        print(f"  Total pairs:          {df['pairs'].sum():.0f}")
    print(f"\n=== ROUND STATS ===")
    print(f"  Avg favored fills:    {df['n_fav_fills'].mean():.2f}")
    print(f"  Avg loser   fills:    {df['n_los_fills'].mean():.2f}")
    print(f"  Avg fav shares:       {df['fav_shares'].mean():.2f}")
    print(f"  Avg los shares:       {df['los_shares'].mean():.2f}")

    print(f"\n=== P&L DECOMP ===")
    print(f"  Pair payoff:          ${df['payoff_pairs'].sum():>10.2f}")
    print(f"  Residual payoff:      ${df['payoff_residual'].sum():>10.2f}")
    print(f"  SL taker recovery:    ${df['sl_recovery'].sum():>10.2f}")
    print(f"  Cash spent:           ${df['cash_spent'].sum():>10.2f}")
    print(f"  Rebate total:         ${df['rebate_total'].sum():>10.2f}")
    print(f"  Taker fees:           ${df['taker_fees'].sum():>10.2f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-days", type=int, default=None)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--out", type=str, default="utils/mm_adaptive_v2.parquet")
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--observe", type=float, default=10.0)
    ap.add_argument("--threshold", type=float, default=0.02)
    ap.add_argument("--target", type=float, default=0.97)
    ap.add_argument("--stop-loss", type=float, default=270.0)
    ap.add_argument("--bbo-offset", type=int, default=1)
    ap.add_argument("--size", type=float, default=5.0)
    ap.add_argument("--cap", type=float, default=100.0)
    ap.add_argument("--no-neutral", action="store_true")
    ap.add_argument("--early-exit-shares", type=float, default=0.0)
    ap.add_argument("--early-exit-min", type=float, default=30.0)
    args = ap.parse_args()

    cfg = Adaptive2Policy(
        bbo_offset_ticks=args.bbo_offset,
        fill_size=args.size,
        max_inv_per_side=args.cap,
        observe_seconds=args.observe,
        direction_threshold=args.threshold,
        pair_target=args.target,
        stop_loss_seconds=args.stop_loss,
        allow_neutral_mode=not args.no_neutral,
        early_exit_fav_shares=args.early_exit_shares,
        early_exit_min_seconds=args.early_exit_min,
    )

    paths = _collect(PRICES_5M_ROOT, args.max_days)
    print(f"[adaptive_v2] paths: {len(paths)}", flush=True)

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
    print(f"\n[adaptive_v2] valid: {len(df)} ({time.time()-t0:.1f}s)")
    if df.empty:
        return 0

    if not args.no_save:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(args.out, compression="zstd")

    print_summary(df, cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
