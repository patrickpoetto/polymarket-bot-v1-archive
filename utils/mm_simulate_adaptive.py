"""
ADAPTIVE PAIR strategy.

Esattamente come descritto:
  1. Osserva il mercato per `observe_seconds` (es. 10s).
  2. Scegli un lato "ritenuto vincente" (favored) basato sul delta_mid.
     - Se |delta| < direction_threshold → "NEUTRAL" (no preferenza, piazza su entrambi)
  3. Phase OPEN: piazza maker BUY SOLO sul lato favored (post_only) a BBO-1c.
  4. Quando fillа il leg1 a P1: piazza maker BUY sull'altro lato a target_pair_cost - P1.
  5. Phase ADAPTIVE: ad ogni tick ricalcoliamo i max-price ammessi
     su ciascun lato in modo che la pair cost media RESTI ≤ target.
     Regola ferrea: NON piazzo a un prezzo che porterebbe pair_cost > target.
  6. Fine window: residuo single-leg → hold to expiry (no taker fallback).

Variabili di stato:
  up_shares, dn_shares
  up_cash, dn_cash (totale speso, NON medio)
  avg_up = up_cash / up_shares se >0 else 0
  pair_cost_running = avg_up + avg_dn (con la convenzione che 0 se uno dei due lati è vuoto)

Logica price caps:
  Entrambi vuoti (no_inv):
    - Se favored=UP: max_up = initial_max_price (es. 0.55), max_dn = -1 (no fill)
    - Se favored=DN: simmetrico
    - Se NEUTRAL: max_up = max_dn = initial_max_price (tipo 0.51)
  Solo UP ha shares:
    - max_up = avg_up   (averaging down only)
    - max_dn = target - avg_up   (per chiudere coppia a target)
  Solo DN ha shares: simmetrico
  Entrambi hanno shares:
    - Per UP: vincolo che (up_cash + p*sz)/(up_shares+sz) + avg_dn ≤ target
              → p ≤ ((target - avg_dn)*(up_shares+sz) - up_cash) / sz
    - Per DN: simmetrico
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
class AdaptivePolicy:
    tick: float = 0.01
    bbo_offset_ticks: int = 1
    fill_size: float = 5.0
    max_inv_per_side: float = 100.0
    trade_window_seconds: float = 300.0

    observe_seconds: float = 10.0
    direction_threshold: float = 0.01

    target_pair_cost: float = 0.98
    initial_max_price: float = 0.51  # safety price cap when no inventory yet
    max_skew: float = 0.0   # 0 = disabled. If >0, block new fills on side X when X_shares - Y_shares >= max_skew
    skip_neutral: bool = False  # if True, don't trade markets where signal is NEUTRAL
    bid_sum_max_at_obs: float = 0.0  # if >0, only trade if bid_sum at obs end <= this

    # End-of-window forced completion modes
    close_at_seconds: float = 0.0   # 0 = disabled. else cancel orders at this time
    close_mode: str = "hold"        # "hold" | "taker_close" (sell excess) | "taker_balance" (buy missing)
    close_taker_max_pair_cost: float = 1.10  # only taker_balance if forced pair_cost <= this

    fee_rate: float = 0.07
    rebate_pct: float = 0.20


@dataclass
class AdaptiveResult:
    market_name: str
    date: str
    favored: str
    n_up_fills: int
    n_dn_fills: int
    up_shares: float
    dn_shares: float
    up_cash: float
    dn_cash: float
    avg_up_price: float
    avg_dn_price: float
    pairs: float
    up_residual: float
    dn_residual: float
    pair_cost_avg: float
    cash_total: float
    rebate_total: float
    taker_close_pnl: float          # net contribution from end-window taker actions
    payoff_pairs: float
    payoff_residual: float
    pnl: float
    winner: str


def simulate_adaptive(
    wide: pd.DataFrame, cfg: AdaptivePolicy, market_name: str, date: str
) -> Optional[AdaptiveResult]:
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
    size = cfg.fill_size
    target = cfg.target_pair_cost
    obs_ms = int(cfg.observe_seconds * 1000)
    win_ms = int(cfg.trade_window_seconds * 1000)

    initial_up_mid = (up_bid[0] + up_ask[0]) * 0.5

    up_shares = 0.0
    dn_shares = 0.0
    up_cash = 0.0
    dn_cash = 0.0
    n_up_fills = 0
    n_dn_fills = 0
    rebate_total = 0.0
    favored = None  # "UP", "DOWN", "NEUTRAL"
    closed = False
    taker_close_pnl = 0.0
    close_ms = int(cfg.close_at_seconds * 1000.0) if cfg.close_at_seconds > 0 else None

    prev_ub = up_bid[0]
    prev_db = dn_bid[0]

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

        # End-of-window forced completion (close_at)
        # ACCOUNTING:
        #   up_cash, dn_cash  -> cumulative cash spent buying shares (maker + close-taker BUYS).
        #                        NEVER reduced on sales.
        #   taker_close_pnl   -> net cash flow from close-taker actions:
        #                        positive for sells (revenue - fee),
        #                        negative for buys (-fee only; the cost goes into up/dn_cash).
        if close_ms is not None and not closed and elapsed_ms >= close_ms:
            ub_now = up_bid[i]
            ua_now = up_ask[i]
            db_now = dn_bid[i]
            da_now = dn_ask[i]
            if cfg.close_mode == "taker_close":
                # SELL excess on the heavy side at current bid
                up_excess = max(0.0, up_shares - dn_shares)
                dn_excess = max(0.0, dn_shares - up_shares)
                if up_excess > 0 and ub_now > 0:
                    revenue = ub_now * up_excess
                    fee = cfg.fee_rate * ub_now * (1 - ub_now) * up_excess
                    taker_close_pnl += revenue - fee
                    up_shares -= up_excess
                    # NOTE: do NOT reduce up_cash. The cash was already spent.
                if dn_excess > 0 and db_now > 0:
                    revenue = db_now * dn_excess
                    fee = cfg.fee_rate * db_now * (1 - db_now) * dn_excess
                    taker_close_pnl += revenue - fee
                    dn_shares -= dn_excess
            elif cfg.close_mode == "taker_balance":
                # BUY missing side at ask to complete pairs
                up_short = max(0.0, dn_shares - up_shares)
                dn_short = max(0.0, up_shares - dn_shares)
                if up_short > 0 and ua_now > 0:
                    avg_dn = dn_cash / dn_shares if dn_shares > 0 else 0.0
                    proj_pair_cost = avg_dn + ua_now
                    if proj_pair_cost <= cfg.close_taker_max_pair_cost + EPS:
                        cost = ua_now * up_short
                        fee = cfg.fee_rate * ua_now * (1 - ua_now) * up_short
                        # cost is registered in up_cash (cumulative); only fee in taker_close_pnl
                        taker_close_pnl -= fee
                        up_cash += cost
                        up_shares += up_short
                if dn_short > 0 and da_now > 0:
                    avg_up = up_cash / up_shares if up_shares > 0 else 0.0
                    proj_pair_cost = avg_up + da_now
                    if proj_pair_cost <= cfg.close_taker_max_pair_cost + EPS:
                        cost = da_now * dn_short
                        fee = cfg.fee_rate * da_now * (1 - da_now) * dn_short
                        taker_close_pnl -= fee
                        dn_cash += cost
                        dn_shares += dn_short
            closed = True
            break  # cancel all maker orders

        # Phase 1: determine favored side at observe_seconds
        if favored is None and elapsed_ms >= obs_ms:
            up_mid_now = (up_bid[i] + up_ask[i]) * 0.5
            delta = up_mid_now - initial_up_mid
            bid_sum_now = up_bid[i] + dn_bid[i]
            if delta > cfg.direction_threshold + EPS:
                favored = "UP"
            elif delta < -cfg.direction_threshold - EPS:
                favored = "DOWN"
            else:
                favored = "NEUTRAL"
            # Filter: skip neutral if requested
            if cfg.skip_neutral and favored == "NEUTRAL":
                favored = "SKIP"
            # Filter: skip if bid_sum too high at obs (= market expensive)
            if cfg.bid_sum_max_at_obs > 0 and bid_sum_now > cfg.bid_sum_max_at_obs + EPS:
                favored = "SKIP"

        # Compute current avg prices (0 if no shares)
        avg_up = up_cash / up_shares if up_shares > 0 else 0.0
        avg_dn = dn_cash / dn_shares if dn_shares > 0 else 0.0

        # Compute max-price caps for new fills
        # Only place orders if favored is determined (or before observe period for NEUTRAL?)
        # Actually: don't trade before observation period ends.
        if favored is None:
            prev_ub = up_bid[i]
            prev_db = dn_bid[i]
            continue

        if favored == "SKIP":
            prev_ub = up_bid[i]
            prev_db = dn_bid[i]
            continue

        if up_shares < EPS and dn_shares < EPS:
            # No inventory yet
            if favored == "UP":
                max_up = cfg.initial_max_price
                max_dn = -1.0
            elif favored == "DOWN":
                max_up = -1.0
                max_dn = cfg.initial_max_price
            else:  # NEUTRAL
                max_up = cfg.initial_max_price
                max_dn = cfg.initial_max_price
        elif up_shares > EPS and dn_shares < EPS:
            # Only UP, need DN to close. Allow further UP only if averaging down.
            max_up = avg_up  # strict averaging down
            max_dn = target - avg_up
        elif dn_shares > EPS and up_shares < EPS:
            max_up = target - avg_dn
            max_dn = avg_dn
        else:
            # Both sides have inventory
            # New UP price constraint:
            #   (up_cash + p*sz) / (up_shares+sz) + avg_dn ≤ target
            # → p ≤ ((target - avg_dn)*(up_shares+sz) - up_cash) / sz
            denom_up = max(size, EPS)
            max_up = ((target - avg_dn) * (up_shares + size) - up_cash) / denom_up
            denom_dn = max(size, EPS)
            max_dn = ((target - avg_up) * (dn_shares + size) - dn_cash) / denom_dn

        # Cap at inventory limit
        if up_shares >= cfg.max_inv_per_side - EPS:
            max_up = -1.0
        if dn_shares >= cfg.max_inv_per_side - EPS:
            max_dn = -1.0

        # Skew cap: block fills on the heavy side when imbalance exceeds threshold
        if cfg.max_skew > 0:
            if (up_shares - dn_shares) >= cfg.max_skew - EPS:
                max_up = -1.0
            if (dn_shares - up_shares) >= cfg.max_skew - EPS:
                max_dn = -1.0

        # Floor: max prices must be > 0 to be valid
        if max_up > 1.0 - EPS:
            max_up = 1.0 - cfg.tick
        if max_dn > 1.0 - EPS:
            max_dn = 1.0 - cfg.tick

        ub = up_bid[i]
        db = dn_bid[i]

        # UP fill check: maker at prev_ub - offset, fills if ub drops past it
        # AND if old_order_px <= max_up (we wouldn't have placed above max)
        if max_up > 0 and ub < prev_ub - EPS:
            old_order_px = prev_ub - offset
            if old_order_px > 0 and old_order_px <= max_up + EPS and ub < old_order_px - EPS:
                fill_px = old_order_px
                sz = min(size, cfg.max_inv_per_side - up_shares)
                up_cash += fill_px * sz
                up_shares += sz
                rebate_total += rebate_factor * fill_px * (1 - fill_px) * sz
                n_up_fills += 1

        if max_dn > 0 and db < prev_db - EPS:
            old_order_px = prev_db - offset
            if old_order_px > 0 and old_order_px <= max_dn + EPS and db < old_order_px - EPS:
                fill_px = old_order_px
                sz = min(size, cfg.max_inv_per_side - dn_shares)
                dn_cash += fill_px * sz
                dn_shares += sz
                rebate_total += rebate_factor * fill_px * (1 - fill_px) * sz
                n_dn_fills += 1

        prev_ub = ub
        prev_db = db

    pairs = min(up_shares, dn_shares)
    up_res = up_shares - pairs
    dn_res = dn_shares - pairs

    avg_up = up_cash / up_shares if up_shares > 0 else 0.0
    avg_dn = dn_cash / dn_shares if dn_shares > 0 else 0.0

    if pairs > 0:
        pair_cost_avg = avg_up + avg_dn
    else:
        pair_cost_avg = float("nan")

    payoff_pairs = pairs * 1.0
    if winner == "UP":
        payoff_residual = up_res * 1.0
    elif winner == "DOWN":
        payoff_residual = dn_res * 1.0
    else:
        payoff_residual = 0.5 * (up_res + dn_res)

    cash_total = up_cash + dn_cash
    pnl = payoff_pairs + payoff_residual - cash_total + rebate_total + taker_close_pnl

    return AdaptiveResult(
        market_name=market_name, date=date,
        favored=favored or "NONE",
        n_up_fills=n_up_fills, n_dn_fills=n_dn_fills,
        up_shares=up_shares, dn_shares=dn_shares,
        up_cash=up_cash, dn_cash=dn_cash,
        avg_up_price=avg_up, avg_dn_price=avg_dn,
        pairs=pairs, up_residual=up_res, dn_residual=dn_res,
        pair_cost_avg=pair_cost_avg,
        cash_total=cash_total, rebate_total=rebate_total,
        taker_close_pnl=taker_close_pnl,
        payoff_pairs=payoff_pairs, payoff_residual=payoff_residual,
        pnl=pnl, winner=winner,
    )


_GLOBAL_CFG: Optional[AdaptivePolicy] = None


def _init_worker(cfg_dict: dict) -> None:
    global _GLOBAL_CFG
    _GLOBAL_CFG = AdaptivePolicy(**cfg_dict)


def _process_one(path_str: str) -> Optional[dict]:
    cfg = _GLOBAL_CFG or AdaptivePolicy()
    p = Path(path_str)
    wide = parse_market_csv(p)
    if wide is None:
        return None
    res = simulate_adaptive(wide, cfg, p.stem, p.parent.parent.name)
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


def print_summary(df: pd.DataFrame, cfg: AdaptivePolicy) -> None:
    n = len(df)
    days = df["date"].nunique() if "date" in df.columns else 1
    pnl = df["pnl"]

    print("\n" + "=" * 90)
    print("ADAPTIVE PAIR STRATEGY SUMMARY")
    print("=" * 90)
    print(f"Markets: {n}    Days: {days}    obs: {cfg.observe_seconds}s    "
          f"thresh: {cfg.direction_threshold:.3f}    target: ${cfg.target_pair_cost}    "
          f"init_max: ${cfg.initial_max_price}    cap: {cfg.max_inv_per_side}\n")

    print(f"=== HEADLINE ===")
    print(f"  Total P&L:        ${pnl.sum():>10.2f}")
    print(f"  Avg P&L / market: ${pnl.mean():>10.4f}")
    print(f"  Median P&L:       ${pnl.median():>10.4f}")
    print(f"  Per day:          ${pnl.sum()/days:>10.2f}")
    print(f"  Markets P&L>0:    {int((pnl>0).sum())}/{n} ({(pnl>0).mean():.1%})")
    print(f"  Days positive:    {int((df.groupby('date')['pnl'].sum()>0).sum())}/{days}")
    if "taker_close_pnl" in df.columns:
        print(f"  Close mode:       {cfg.close_mode}  close_at: {cfg.close_at_seconds}s  "
              f"max_pair: ${cfg.close_taker_max_pair_cost}")
        print(f"  Taker close P&L:  ${df['taker_close_pnl'].sum():.2f} (avg ${df['taker_close_pnl'].mean():.4f}/m)")

    paired = df[(df["pairs"] > 0)]
    untouched = df[(df["n_up_fills"] == 0) & (df["n_dn_fills"] == 0)]
    one_only = df[((df["n_up_fills"] > 0) ^ (df["n_dn_fills"] > 0))]
    print(f"\n=== FILL STATS ===")
    print(f"  Markets with pairs (>=1):  {len(paired)} ({len(paired)/n:.1%})")
    print(f"  Markets only one side:     {len(one_only)} ({len(one_only)/n:.1%})")
    print(f"  Markets untouched:         {len(untouched)} ({len(untouched)/n:.1%})")
    if len(paired):
        print(f"  Pair cost avg (paired):    ${paired['pair_cost_avg'].mean():.4f}")
        print(f"  Avg pairs (paired):        {paired['pairs'].mean():.2f}")
    print(f"  Avg pairs (all):           {df['pairs'].mean():.2f}")
    print(f"  Avg up_shares (all):       {df['up_shares'].mean():.2f}")
    print(f"  Avg dn_shares (all):       {df['dn_shares'].mean():.2f}")

    print(f"\n=== BY FAVORED SIDE ===")
    for fav in ["UP", "DOWN", "NEUTRAL", "SKIP"]:
        sub = df[df["favored"] == fav]
        if len(sub) == 0:
            continue
        winner_match = (sub["favored"] == sub["winner"]).mean() * 100
        print(f"  favored={fav:<8} n={len(sub):4d} ({len(sub)/n:.1%}) "
              f"$/m={sub['pnl'].mean():+.4f}  pair_c={sub['pair_cost_avg'].mean():.4f}  "
              f"hit_rate={winner_match:.1f}%")

    if len(paired):
        print(f"\n=== PAIRED ({len(paired)}): ===")
        print(f"  Total P&L:        ${paired['pnl'].sum():.2f}")
        print(f"  Avg P&L / market: ${paired['pnl'].mean():.4f}")
        print(f"  Per day:          ${paired['pnl'].sum()/days:.2f}")
    if len(one_only):
        print(f"\n=== ONE-SIDE-ONLY ({len(one_only)}): ===")
        print(f"  Total P&L:        ${one_only['pnl'].sum():.2f}")
        print(f"  Avg P&L / market: ${one_only['pnl'].mean():.4f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-days", type=int, default=None)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--out", type=str, default="utils/mm_adaptive.parquet")
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--bbo-offset", type=int, default=1)
    ap.add_argument("--size", type=float, default=5.0)
    ap.add_argument("--cap", type=float, default=100.0)
    ap.add_argument("--window", type=float, default=300.0)
    ap.add_argument("--observe", type=float, default=10.0)
    ap.add_argument("--threshold", type=float, default=0.01)
    ap.add_argument("--target", type=float, default=0.98)
    ap.add_argument("--init-max", type=float, default=0.51)
    ap.add_argument("--skew", type=float, default=0.0)
    ap.add_argument("--skip-neutral", action="store_true")
    ap.add_argument("--bidsum-max", type=float, default=0.0)
    ap.add_argument("--close-at", type=float, default=0.0)
    ap.add_argument("--close-mode", type=str, default="hold",
                    choices=["hold", "taker_close", "taker_balance"])
    ap.add_argument("--close-max-pair", type=float, default=1.10)
    args = ap.parse_args()

    cfg = AdaptivePolicy(
        bbo_offset_ticks=args.bbo_offset, fill_size=args.size,
        max_inv_per_side=args.cap,
        trade_window_seconds=args.window,
        observe_seconds=args.observe,
        direction_threshold=args.threshold,
        target_pair_cost=args.target,
        initial_max_price=args.init_max,
        max_skew=args.skew,
        skip_neutral=args.skip_neutral,
        bid_sum_max_at_obs=args.bidsum_max,
        close_at_seconds=args.close_at,
        close_mode=args.close_mode,
        close_taker_max_pair_cost=args.close_max_pair,
    )

    paths = _collect(PRICES_5M_ROOT, args.max_days)
    print(f"[adaptive] paths: {len(paths)}", flush=True)

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
    print(f"\n[adaptive] valid: {len(df)} ({time.time()-t0:.1f}s)")
    if df.empty:
        return 0

    if not args.no_save:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(args.out, compression="zstd")

    print_summary(df, cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
