"""
Strategia ibrida basata su segnale direzionale precoce:

  1) Per i primi `observe_seconds`, NESSUN ordine.
  2) A t=observe_seconds, computa delta = up_mid - up_mid_0.
  3) Se delta > +trigger: MODE = "UP_directional"
     - Place maker buys SOLO su UP (BBO-1). DN non si tocca.
     - Hold to expiry. EV = P(UP wins) × $1 − avg_fill.
  4) Se delta < −trigger: MODE = "DN_directional"
     - Place maker buys SOLO su DN. UP non si tocca.
  5) Altrimenti |delta| ≤ trigger: MODE = "neutral"
     - Pair MM: maker buys su entrambi i lati (BBO-1).
     - Cap inventory low, hold or close.
  6) Trade window: max trade fino a t=trade_window_seconds, poi hold.

Hold-to-expiry per direzionali è la strategia più semplice. Per neutrali si
applica la BBO-follow + skew-limit + finestra come nel test precedente.
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
class HybridPolicy:
    tick: float = 0.01
    fill_size: float = 5.0
    bbo_offset_ticks: int = 1

    # Segnale direzionale
    observe_seconds: float = 10.0
    direction_trigger: float = 0.015  # in mid-price units (1.5c)

    # Inventory cap PER LATO
    max_inv_favored: float = 100.0   # cap sul lato favorito (BBO-follow)
    max_inv_loser: float = 100.0     # cap sul lato sfavorito (deep ladder)
    max_inv_neutral: float = 50.0    # cap su ciascun lato in modalità neutral

    # Modalità per il lato sfavorito (in direzionale):
    #   "ladder":      buy maker a livelli fissi
    #   "bbo":         buy maker a BBO - bbo_offset_loser_ticks
    #   "skip":        non comprare lo sfavorito, single-leg solo favorito
    #   "skip_market": SKIP totale del mercato direzionale (no orders entrambi i lati)
    loser_mode: str = "ladder"
    bbo_offset_loser_ticks: int = 5
    # Livelli ladder per il lato sfavorito (in price units)
    loser_ladder_levels: tuple[float, ...] = (0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10)
    loser_size_per_level: float = 15.0

    # Trade window: dopo questo, cancella ordini, hold
    trade_window_seconds: float = 60.0

    # Hedge: hold o taker_close a fine window
    hedge_mode: str = "hold"

    # Fees
    fee_rate: float = 0.07
    rebate_pct: float = 0.20


@dataclass
class HybridResult:
    market_name: str
    date: str
    mode: str  # "UP_dir" / "DN_dir" / "neutral"
    delta_obs: float

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

    cash_spent: float
    rebate_total: float
    taker_fees: float

    payoff_pairs: float
    payoff_residual: float
    pnl: float

    winner: str
    final_up_mid: float


def simulate_hybrid(
    wide: pd.DataFrame, cfg: HybridPolicy, market_name: str, date: str
) -> Optional[HybridResult]:
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
    tick = cfg.tick
    fav_offset = cfg.bbo_offset_ticks * tick
    los_bbo_offset = cfg.bbo_offset_loser_ticks * tick
    rebate_factor = cfg.rebate_pct * cfg.fee_rate
    fill_size = cfg.fill_size

    up_mid_arr = (up_bid + up_ask) * 0.5
    up_t0 = float(up_mid_arr[0])

    obs_idx = int(np.searchsorted(t_ms, t_start + int(cfg.observe_seconds * 1000.0)))
    obs_idx = min(max(obs_idx, 1), n - 1)
    delta = float(up_mid_arr[obs_idx] - up_t0)

    # Identify favored / loser sides
    if delta > cfg.direction_trigger + EPS:
        mode = "UP_dir"
        favored = "UP"
        if cfg.loser_mode == "skip_market":
            cap_fav = 0.0
            cap_los = 0.0
        else:
            cap_fav = cfg.max_inv_favored
            cap_los = cfg.max_inv_loser
    elif delta < -cfg.direction_trigger - EPS:
        mode = "DN_dir"
        favored = "DN"
        if cfg.loser_mode == "skip_market":
            cap_fav = 0.0
            cap_los = 0.0
        else:
            cap_fav = cfg.max_inv_favored
            cap_los = cfg.max_inv_loser
    else:
        mode = "neutral"
        favored = "NONE"
        cap_fav = cfg.max_inv_neutral
        cap_los = cfg.max_inv_neutral

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

    prev_up_bid = up_bid[obs_idx]
    prev_dn_bid = dn_bid[obs_idx]

    # Track ladder fills: each level fills at most once unless re-armed below current bid
    # Ladder is for the LOSER side in directional mode.
    if mode == "neutral":
        ladder_levels_up: tuple[float, ...] = ()
        ladder_levels_dn: tuple[float, ...] = ()
    elif mode == "UP_dir":
        # UP is favored (BBO-follow), DN is loser (ladder)
        ladder_levels_up = ()
        ladder_levels_dn = cfg.loser_ladder_levels if cfg.loser_mode == "ladder" else ()
    else:  # DN_dir
        ladder_levels_up = cfg.loser_ladder_levels if cfg.loser_mode == "ladder" else ()
        ladder_levels_dn = ()
    up_ladder_armed = [True] * len(ladder_levels_up)
    dn_ladder_armed = [True] * len(ladder_levels_dn)

    def is_favored(side: str) -> bool:
        return (mode == "UP_dir" and side == "UP") or (mode == "DN_dir" and side == "DN")

    def is_loser(side: str) -> bool:
        return (mode == "UP_dir" and side == "DN") or (mode == "DN_dir" and side == "UP")

    trade_end_target_ms = t_start + int(cfg.trade_window_seconds * 1000.0)

    for i in range(obs_idx, n):
        elapsed_ms = t_ms[i] - t_start
        in_window = elapsed_ms <= int(cfg.trade_window_seconds * 1000.0)

        ub = up_bid[i]
        db = dn_bid[i]

        if in_window and i > obs_idx:
            # ===== UP side =====
            if mode == "neutral" or is_favored("UP"):
                cap_up_curr = cap_fav if (mode == "neutral" or is_favored("UP")) else 0.0
                if cap_up_curr > 0 and up_shares < cap_up_curr - EPS and ub < prev_up_bid - EPS:
                    old_order_px = prev_up_bid - fav_offset
                    if old_order_px > 0 and ub < old_order_px - EPS:
                        fill_px = old_order_px
                        sz = min(fill_size, cap_up_curr - up_shares)
                        cash_spent += fill_px * sz
                        up_shares += sz
                        rebate_total += rebate_factor * fill_px * (1.0 - fill_px) * sz
                        sum_up_px += fill_px * sz
                        sum_up_sz += sz
                        n_up_fills += 1
            elif is_loser("UP"):
                if cfg.loser_mode == "ladder":
                    # Fill ladder levels on UP side as UP_bid drops
                    for k, lvl in enumerate(ladder_levels_up):
                        if not up_ladder_armed[k]:
                            continue
                        # fill if UP_bid drops below level (we sat at level as buy-maker)
                        if prev_up_bid >= lvl - EPS and ub < lvl - EPS:
                            sz = min(cfg.loser_size_per_level, cap_los - up_shares)
                            if sz <= 0:
                                up_ladder_armed[k] = False
                                continue
                            fill_px = lvl
                            cash_spent += fill_px * sz
                            up_shares += sz
                            rebate_total += rebate_factor * fill_px * (1.0 - fill_px) * sz
                            sum_up_px += fill_px * sz
                            sum_up_sz += sz
                            n_up_fills += 1
                            up_ladder_armed[k] = False
                elif cfg.loser_mode == "bbo":
                    if up_shares < cap_los - EPS and ub < prev_up_bid - EPS:
                        old_order_px = prev_up_bid - los_bbo_offset
                        if old_order_px > 0 and ub < old_order_px - EPS:
                            fill_px = old_order_px
                            sz = min(fill_size, cap_los - up_shares)
                            cash_spent += fill_px * sz
                            up_shares += sz
                            rebate_total += rebate_factor * fill_px * (1.0 - fill_px) * sz
                            sum_up_px += fill_px * sz
                            sum_up_sz += sz
                            n_up_fills += 1
                # "skip" mode: no orders on UP loser side

            # ===== DN side =====
            if mode == "neutral" or is_favored("DN"):
                cap_dn_curr = cap_fav if (mode == "neutral" or is_favored("DN")) else 0.0
                if cap_dn_curr > 0 and dn_shares < cap_dn_curr - EPS and db < prev_dn_bid - EPS:
                    old_order_px = prev_dn_bid - fav_offset
                    if old_order_px > 0 and db < old_order_px - EPS:
                        fill_px = old_order_px
                        sz = min(fill_size, cap_dn_curr - dn_shares)
                        cash_spent += fill_px * sz
                        dn_shares += sz
                        rebate_total += rebate_factor * fill_px * (1.0 - fill_px) * sz
                        sum_dn_px += fill_px * sz
                        sum_dn_sz += sz
                        n_dn_fills += 1
            elif is_loser("DN"):
                if cfg.loser_mode == "ladder":
                    for k, lvl in enumerate(ladder_levels_dn):
                        if not dn_ladder_armed[k]:
                            continue
                        if prev_dn_bid >= lvl - EPS and db < lvl - EPS:
                            sz = min(cfg.loser_size_per_level, cap_los - dn_shares)
                            if sz <= 0:
                                dn_ladder_armed[k] = False
                                continue
                            fill_px = lvl
                            cash_spent += fill_px * sz
                            dn_shares += sz
                            rebate_total += rebate_factor * fill_px * (1.0 - fill_px) * sz
                            sum_dn_px += fill_px * sz
                            sum_dn_sz += sz
                            n_dn_fills += 1
                            dn_ladder_armed[k] = False
                elif cfg.loser_mode == "bbo":
                    if dn_shares < cap_los - EPS and db < prev_dn_bid - EPS:
                        old_order_px = prev_dn_bid - los_bbo_offset
                        if old_order_px > 0 and db < old_order_px - EPS:
                            fill_px = old_order_px
                            sz = min(fill_size, cap_los - dn_shares)
                            cash_spent += fill_px * sz
                            dn_shares += sz
                            rebate_total += rebate_factor * fill_px * (1.0 - fill_px) * sz
                            sum_dn_px += fill_px * sz
                            sum_dn_sz += sz
                            n_dn_fills += 1

        prev_up_bid = ub
        prev_dn_bid = db

    # Determine winner
    final_up_mid = (up_bid[-1] + up_ask[-1]) * 0.5
    final_dn_mid = (dn_bid[-1] + dn_ask[-1]) * 0.5
    if final_up_mid > final_dn_mid:
        winner = "UP"
    elif final_dn_mid > final_up_mid:
        winner = "DOWN"
    else:
        winner = "TIE"

    # Pairs and residuals
    pairs = min(up_shares, dn_shares)
    up_res = up_shares - pairs
    dn_res = dn_shares - pairs

    if cfg.hedge_mode == "taker_close" and (up_res > 0 or dn_res > 0):
        idx_close = int(np.searchsorted(t_ms, trade_end_target_ms))
        idx_close = min(max(idx_close, 0), n - 1)
        if up_res > 0:
            px = float(dn_ask[idx_close])
            if px > 0 and not np.isnan(px):
                cash_spent += px * up_res
                taker_fees += cfg.fee_rate * px * (1.0 - px) * up_res
                dn_shares += up_res
        if dn_res > 0:
            px = float(up_ask[idx_close])
            if px > 0 and not np.isnan(px):
                cash_spent += px * dn_res
                taker_fees += cfg.fee_rate * px * (1.0 - px) * dn_res
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
    if sum_up_sz > 0 and sum_dn_sz > 0:
        avg_pair_cost = avg_up + avg_dn
    else:
        avg_pair_cost = float("nan")

    return HybridResult(
        market_name=market_name,
        date=date,
        mode=mode,
        delta_obs=delta,
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
        cash_spent=cash_spent,
        rebate_total=rebate_total,
        taker_fees=taker_fees,
        payoff_pairs=payoff_pairs,
        payoff_residual=payoff_residual,
        pnl=pnl,
        winner=winner,
        final_up_mid=final_up_mid,
    )


_GLOBAL_CFG: Optional[HybridPolicy] = None


def _init_worker(cfg_dict: dict) -> None:
    global _GLOBAL_CFG
    _GLOBAL_CFG = HybridPolicy(**cfg_dict)


def _process_one(path_str: str) -> Optional[dict]:
    cfg = _GLOBAL_CFG or HybridPolicy()
    p = Path(path_str)
    wide = parse_market_csv(p)
    if wide is None:
        return None
    res = simulate_hybrid(wide, cfg, p.stem, p.parent.parent.name)
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
    ap.add_argument("--out", type=str, default="utils/mm_hybrid.parquet")
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--observe", type=float, default=10.0)
    ap.add_argument("--trigger", type=float, default=0.015)
    ap.add_argument("--bbo-offset", type=int, default=1)
    ap.add_argument("--fill-size", type=float, default=5.0)
    ap.add_argument("--max-inv-fav", type=float, default=100.0)
    ap.add_argument("--max-inv-los", type=float, default=100.0)
    ap.add_argument("--max-inv-neu", type=float, default=50.0)
    ap.add_argument("--loser-mode", choices=("ladder", "bbo", "skip", "skip_market"),
                    default="ladder")
    ap.add_argument("--bbo-offset-loser", type=int, default=5)
    ap.add_argument("--loser-levels", type=str, default="0.40,0.35,0.30,0.25,0.20,0.15,0.10",
                    help="Comma-separated price levels for loser ladder")
    ap.add_argument("--loser-size", type=float, default=15.0)
    ap.add_argument("--trade-window", type=float, default=300.0)
    ap.add_argument("--hedge-mode", choices=("hold", "taker_close"), default="hold")
    args = ap.parse_args()

    levels = tuple(float(x) for x in args.loser_levels.split(",") if x.strip())
    cfg = HybridPolicy(
        bbo_offset_ticks=args.bbo_offset,
        fill_size=args.fill_size,
        observe_seconds=args.observe,
        direction_trigger=args.trigger,
        max_inv_favored=args.max_inv_fav,
        max_inv_loser=args.max_inv_los,
        max_inv_neutral=args.max_inv_neu,
        loser_mode=args.loser_mode,
        bbo_offset_loser_ticks=args.bbo_offset_loser,
        loser_ladder_levels=levels,
        loser_size_per_level=args.loser_size,
        trade_window_seconds=args.trade_window,
        hedge_mode=args.hedge_mode,
    )

    paths = _collect(PRICES_5M_ROOT, args.max_days)
    print(f"[hybrid] paths: {len(paths)}", flush=True)
    print(f"[hybrid] cfg: observe={cfg.observe_seconds}s  trigger={cfg.direction_trigger:.3f}  "
          f"fav_offset={cfg.bbo_offset_ticks}  cap_fav={cfg.max_inv_favored}  "
          f"loser_mode={cfg.loser_mode}  cap_los={cfg.max_inv_loser}  "
          f"loser_levels={cfg.loser_ladder_levels}  loser_size={cfg.loser_size_per_level}  "
          f"cap_neu={cfg.max_inv_neutral}  window={cfg.trade_window_seconds}s  "
          f"hedge={cfg.hedge_mode}", flush=True)

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
            if n_done % 2000 == 0 or n_done == len(paths):
                el = time.time() - t0
                print(f"  {n_done}/{len(paths)} valid={len(results)} ({el:.1f}s)", flush=True)

    df = pd.DataFrame(results)
    print(f"\n[hybrid] valid: {len(df)} ({time.time()-t0:.1f}s)")
    if df.empty:
        return 0

    if not args.no_save:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(args.out, compression="zstd")

    print_summary(df, cfg)
    return 0


def print_summary(df: pd.DataFrame, cfg: HybridPolicy) -> None:
    print("\n" + "=" * 88)
    print("HYBRID STRATEGY SUMMARY")
    print("=" * 88)
    n = max(1, len(df))
    pnl = df["pnl"]
    days = df["date"].nunique() if "date" in df.columns else 1

    print(f"Markets: {len(df)}  Days: {days}")
    print(f"Mode distribution:")
    for mode in ("UP_dir", "DN_dir", "neutral"):
        c = (df["mode"] == mode).sum()
        print(f"  {mode}: {c} ({c/n:.1%})")
    print()

    print("=== HEADLINE ===")
    print(f"  Total P&L:          ${pnl.sum():>10.2f}")
    print(f"  Avg P&L / market:   ${pnl.mean():>10.4f}")
    print(f"  Median P&L:         ${pnl.median():>10.4f}")
    print(f"  P&L > 0:            {int((pnl>0).sum())} ({(pnl>0).mean():.1%})")
    print(f"  Per day:            ${pnl.sum()/days:>10.2f}")
    print(f"  Days positive:      {int((df.groupby('date')['pnl'].sum()>0).sum())}/{days}")
    print()

    print("=== PER MODE ===")
    for mode in ("UP_dir", "DN_dir", "neutral"):
        sub = df[df["mode"] == mode]
        if len(sub) == 0:
            continue
        # for direzionali, mostra anche win-rate
        if mode == "UP_dir":
            wr = (sub["winner"] == "UP").mean()
        elif mode == "DN_dir":
            wr = (sub["winner"] == "DOWN").mean()
        else:
            wr = float("nan")
        print(f"\n  {mode}: {len(sub)} markets    P(side wins)={wr:.2%}")
        print(f"    Total P&L:           ${sub['pnl'].sum():.2f}")
        print(f"    Avg P&L:             ${sub['pnl'].mean():.4f}")
        print(f"    Median P&L:          ${sub['pnl'].median():.4f}")
        print(f"    Avg pairs:           {sub['pairs'].mean():.2f}")
        print(f"    Avg up_shares:       {sub['up_shares'].mean():.2f}")
        print(f"    Avg dn_shares:       {sub['dn_shares'].mean():.2f}")
        print(f"    Avg cash_spent:      ${sub['cash_spent'].mean():.2f}")
        print(f"    Avg rebate:          ${sub['rebate_total'].mean():.4f}")
        if not np.isnan(sub["avg_pair_cost"].mean()):
            print(f"    Avg pair_cost:       ${sub['avg_pair_cost'].mean():.4f}")
        print(f"    Avg payoff_pairs:    ${sub['payoff_pairs'].mean():.4f}")
        print(f"    Avg payoff_residual: ${sub['payoff_residual'].mean():.4f}")


if __name__ == "__main__":
    raise SystemExit(main())
