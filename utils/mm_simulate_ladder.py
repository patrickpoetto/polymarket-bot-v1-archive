"""
Ladder MM simulator per Polymarket BTC 5-min.

A differenza di `mm_simulate_5m.py` (che piazza al BBO e chasa), questo
simulatore piazza un *ladder* di buy maker a prezzi FISSI sotto il mid
iniziale, sia su UP che su DN, e lascia che la volatilità del mercato
fila gli ordini.

Idea: con range del mid medio = 60c in 5 min, molti livelli vengono
visitati. Una ladder a 0.49, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15,
0.10, 0.05 su entrambi i lati cattura fill su dip senza dover chase.
Se in un singolo mercato il prezzo oscilla (es. 0.50 -> 0.30 -> 0.70 ->
0.50 -> 0.99), entrambi i lati possono fillare a livelli profondi e il
costo medio della coppia scende ben sotto 1.

Limiti del modello:
- Fill model back-of-queue (drop strict), come per il simulator base.
- Nessun impatto del nostro ordine sulle dinamiche di mercato.
- Nessuna competizione di coda esplicita.
- Single-fill per livello (FIXED ladder; opzionale refill).
- I livelli sono FISSI a prezzi assoluti (es. 0.49 sempre), non
  relativi al mid corrente.

Uso:
    python -m utils.mm_simulate_ladder --max-days 1
    python -m utils.mm_simulate_ladder --hedge-mode hold
    python -m utils.mm_simulate_ladder --levels 0.49,0.45,0.40,0.35,0.30,0.25,0.20,0.15,0.10,0.05
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from utils.mm_analyze_5m import parse_market_csv, PRICES_5M_ROOT  # type: ignore


DEFAULT_LEVELS = (
    0.49, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05,
)


# ---------------------------------------------------------------------------
#  Policy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LadderPolicy:
    levels: tuple = field(default_factory=lambda: DEFAULT_LEVELS)
    size_per_level: float = 20.0
    refill_on_fill: bool = False  # se True: ri-piazza al livello dopo fill

    # Inventory balancing: se |up_shares - dn_shares| > skew_max_shares,
    # cancella temporaneamente la ladder sul lato più pesante (re-armata
    # quando lo skew rientra, indipendentemente da refill_on_fill).
    skew_max_shares: float = 0.0  # 0 = disabilitato

    # Momentum-based cancel: monitora la variazione del mid_price su una
    # finestra retrospettiva di `momentum_lookback_seconds` secondi.
    # Se mid_X scende di più di `momentum_threshold`, cancella la ladder
    # sul lato X (= il lato che sta perdendo). Ri-arma quando la momentum
    # rientra entro la soglia.
    momentum_lookback_seconds: float = 30.0
    momentum_threshold: float = 0.0  # 0 = disabilitato; tipico 0.03-0.10

    # Early-direction filter: dopo `early_decision_seconds`, se up_mid è
    # sopra `early_decision_threshold` (sopra 0.5), cancella DEFINITIVAMENTE
    # la ladder DN per il resto del mercato. Simmetrico se up_mid sotto.
    # 0 = disabilitato.
    early_decision_seconds: float = 0.0
    early_decision_threshold: float = 0.55

    # Hedge dei residui a fine mercato
    hedge_mode: str = "hold"  # "hold" | "taker_close"
    hedge_seconds: float = 25.0

    # Fees & rebate (Polymarket Crypto)
    fee_rate: float = 0.07
    rebate_pct: float = 0.20


# ---------------------------------------------------------------------------
#  Result
# ---------------------------------------------------------------------------


@dataclass
class LadderResult:
    market_name: str
    date: str
    duration_s: float
    n_ticks: int

    n_up_fills: int
    n_dn_fills: int
    up_shares_final: float
    dn_shares_final: float
    pairs_final: float
    up_residual: float
    dn_residual: float

    avg_up_buy_price: float
    avg_dn_buy_price: float
    avg_pair_cost_maker: float
    true_pair_cost: float

    cash_spent: float
    cash_spent_maker: float
    rebate_total: float
    taker_fees_paid: float

    payoff_pairs: float
    payoff_residual: float
    pnl: float
    pnl_per_pair: float

    capital_used_max: float
    fills_per_level_up: list  # number of fills at each level
    fills_per_level_dn: list

    winner: str
    final_up_mid: float
    final_dn_mid: float
    up_min_bid: float       # minimo bid raggiunto (per debug)
    dn_min_bid: float


# ---------------------------------------------------------------------------
#  Simulator
# ---------------------------------------------------------------------------


def simulate_ladder(
    wide: pd.DataFrame,
    cfg: LadderPolicy,
    market_name: str,
    date: str,
) -> Optional[LadderResult]:
    if wide is None or len(wide) < 10:
        return None

    wide = wide.copy()
    for col in ("up_bid", "up_ask", "dn_bid", "dn_ask"):
        wide[col] = wide[col].ffill()
    wide = wide.dropna(subset=["up_bid", "up_ask", "dn_bid", "dn_ask"]).reset_index(
        drop=True
    )
    if len(wide) < 10:
        return None

    up_bid = wide["up_bid"].to_numpy(dtype=np.float64)
    up_ask = wide["up_ask"].to_numpy(dtype=np.float64)
    dn_bid = wide["dn_bid"].to_numpy(dtype=np.float64)
    dn_ask = wide["dn_ask"].to_numpy(dtype=np.float64)
    t_left = wide["t_left"].to_numpy(dtype=np.float64)
    t_ms = wide["t_ms"].to_numpy(dtype=np.int64)
    n = len(wide)

    levels = np.array(sorted(cfg.levels, reverse=True), dtype=np.float64)  # alta -> bassa
    K = len(levels)

    # Quantità ancora "presente" come ordine maker su ciascun livello.
    up_open = np.full(K, cfg.size_per_level, dtype=np.float64)
    dn_open = np.full(K, cfg.size_per_level, dtype=np.float64)
    fills_per_level_up = [0] * K
    fills_per_level_dn = [0] * K

    up_shares = 0.0
    dn_shares = 0.0
    cash_spent = 0.0
    cash_spent_maker = 0.0
    rebate_total = 0.0
    taker_fees_paid = 0.0

    sum_up_px = 0.0
    sum_up_sz = 0.0
    sum_dn_px = 0.0
    sum_dn_sz = 0.0

    capital_used_max = 0.0

    EPS = 1e-9

    up_min_bid = float(up_bid[0])
    dn_min_bid = float(dn_bid[0])

    fee_rate = cfg.fee_rate
    rebate_factor = cfg.rebate_pct * cfg.fee_rate
    skew_cap = cfg.skew_max_shares
    mom_thresh = cfg.momentum_threshold
    mom_lookback_s = cfg.momentum_lookback_seconds

    # Ladder in stato "cancellato" per skew (può essere ri-armata).
    up_skew_blocked = False
    dn_skew_blocked = False
    up_mom_blocked = False
    dn_mom_blocked = False

    # Pre-compute mid prices per momentum
    up_mid = (up_bid + up_ask) * 0.5
    dn_mid = (dn_bid + dn_ask) * 0.5

    # Per ogni tick i, troviamo l'indice del primo tick j tale che
    # (t_ms[i] - t_ms[j]) / 1000 <= mom_lookback_s.
    # Calcoliamo lazy in loop (doubly-pointer).
    mom_j = 0  # cursore della finestra

    # Early decision flag: una volta deciso, tagliamo permanentemente uno dei lati
    early_done = False
    early_perm_block_up = False
    early_perm_block_dn = False

    # Tempo iniziale del mercato (per calcolare elapsed)
    t_start_ms = int(t_ms[0])
    early_thresh_ms = cfg.early_decision_seconds * 1000.0

    for i in range(n):
        ub = up_bid[i]
        db = dn_bid[i]
        if ub < up_min_bid:
            up_min_bid = float(ub)
        if db < dn_min_bid:
            dn_min_bid = float(db)

        # ---- (-2) Early direction: a t=early_decision_seconds, decidi lato ----
        if (
            cfg.early_decision_seconds > 0
            and not early_done
            and (t_ms[i] - t_start_ms) >= early_thresh_ms
        ):
            early_done = True
            cur_up_mid = up_mid[i]
            if cur_up_mid >= cfg.early_decision_threshold:
                # UP sembra vincere -> ladder DN sarebbe sul perdente
                dn_open[:] = 0.0
                early_perm_block_dn = True
            elif cur_up_mid <= (1.0 - cfg.early_decision_threshold):
                # DN sembra vincere -> ladder UP sarebbe sul perdente
                up_open[:] = 0.0
                early_perm_block_up = True
            # altrimenti: nessuno blocco, mercato ancora ambiguo

        # Hard re-zero per lati permanentemente bloccati (early decision)
        if early_perm_block_up:
            up_open[:] = 0.0
        if early_perm_block_dn:
            dn_open[:] = 0.0

        # ---- (-1) Momentum check: cancella ladder sul lato in calo ------
        if mom_thresh > 0:
            window_ms = mom_lookback_s * 1000.0
            while mom_j < i and (t_ms[i] - t_ms[mom_j]) > window_ms:
                mom_j += 1
            up_change = up_mid[i] - up_mid[mom_j]
            dn_change = dn_mid[i] - dn_mid[mom_j]

            if up_change < -mom_thresh:
                if not up_mom_blocked:
                    up_open[:] = 0.0
                    up_mom_blocked = True
            else:
                if up_mom_blocked and not early_perm_block_up:
                    for k in range(K):
                        if (fills_per_level_up[k] == 0 or cfg.refill_on_fill) and levels[k] < up_bid[i] - EPS:
                            up_open[k] = cfg.size_per_level
                    up_mom_blocked = False

            if dn_change < -mom_thresh:
                if not dn_mom_blocked:
                    dn_open[:] = 0.0
                    dn_mom_blocked = True
            else:
                if dn_mom_blocked and not early_perm_block_dn:
                    for k in range(K):
                        if (fills_per_level_dn[k] == 0 or cfg.refill_on_fill) and levels[k] < dn_bid[i] - EPS:
                            dn_open[k] = cfg.size_per_level
                    dn_mom_blocked = False

        # ---- 0) Skew check: cancella/riarma ladder dei lati ---------------
        if skew_cap > 0:
            diff = up_shares - dn_shares
            if diff > skew_cap + EPS:
                if not up_skew_blocked:
                    up_open[:] = 0.0
                    up_skew_blocked = True
                if dn_skew_blocked and not early_perm_block_dn:
                    for k in range(K):
                        if (fills_per_level_dn[k] == 0 or cfg.refill_on_fill) and levels[k] < dn_bid[i] - EPS:
                            dn_open[k] = cfg.size_per_level
                    dn_skew_blocked = False
            elif -diff > skew_cap + EPS:
                if not dn_skew_blocked:
                    dn_open[:] = 0.0
                    dn_skew_blocked = True
                if up_skew_blocked and not early_perm_block_up:
                    for k in range(K):
                        if (fills_per_level_up[k] == 0 or cfg.refill_on_fill) and levels[k] < up_bid[i] - EPS:
                            up_open[k] = cfg.size_per_level
                    up_skew_blocked = False
            else:
                if up_skew_blocked and not early_perm_block_up:
                    for k in range(K):
                        if (fills_per_level_up[k] == 0 or cfg.refill_on_fill) and levels[k] < up_bid[i] - EPS:
                            up_open[k] = cfg.size_per_level
                    up_skew_blocked = False
                if dn_skew_blocked and not early_perm_block_dn:
                    for k in range(K):
                        if (fills_per_level_dn[k] == 0 or cfg.refill_on_fill) and levels[k] < dn_bid[i] - EPS:
                            dn_open[k] = cfg.size_per_level
                    dn_skew_blocked = False

        # ---- 1) Fills su UP: ogni livello con ordine attivo e level > ub fila ----
        for k in range(K):
            if up_open[k] > 0 and levels[k] > ub + EPS:
                size = up_open[k]
                px = levels[k]
                cash_spent += px * size
                cash_spent_maker += px * size
                rebate_total += rebate_factor * px * (1.0 - px) * size
                up_shares += size
                sum_up_px += px * size
                sum_up_sz += size
                fills_per_level_up[k] += 1
                if cfg.refill_on_fill:
                    up_open[k] = cfg.size_per_level
                else:
                    up_open[k] = 0.0

        # ---- 2) Fills su DN ----
        for k in range(K):
            if dn_open[k] > 0 and levels[k] > db + EPS:
                size = dn_open[k]
                px = levels[k]
                cash_spent += px * size
                cash_spent_maker += px * size
                rebate_total += rebate_factor * px * (1.0 - px) * size
                dn_shares += size
                sum_dn_px += px * size
                sum_dn_sz += size
                fills_per_level_dn[k] += 1
                if cfg.refill_on_fill:
                    dn_open[k] = cfg.size_per_level
                else:
                    dn_open[k] = 0.0

        # ---- 3) Hedge dei residui negli ultimi `hedge_seconds` -------------
        if t_left[i] <= cfg.hedge_seconds and (
            i == 0 or t_left[i - 1] > cfg.hedge_seconds
        ):
            pairs_now = min(up_shares, dn_shares)
            up_res = up_shares - pairs_now
            dn_res = dn_shares - pairs_now

            if cfg.hedge_mode == "taker_close":
                if up_res > 0:
                    px = float(dn_ask[i])
                    if not np.isnan(px) and px > 0:
                        cash_spent += px * up_res
                        taker_fees_paid += fee_rate * px * (1.0 - px) * up_res
                        dn_shares += up_res
                if dn_res > 0:
                    px = float(up_ask[i])
                    if not np.isnan(px) and px > 0:
                        cash_spent += px * dn_res
                        taker_fees_paid += fee_rate * px * (1.0 - px) * dn_res
                        up_shares += dn_res
                # cancella tutti gli ordini residui ladder (smetti di accumulare)
                up_open[:] = 0.0
                dn_open[:] = 0.0
            elif cfg.hedge_mode == "hold":
                # cancella ladder ma tieni residui a expiry
                up_open[:] = 0.0
                dn_open[:] = 0.0

        # ---- 4) Track capital used ----------------------------------------
        cap_now = (
            up_shares * (up_bid[i] + up_ask[i]) * 0.5
            + dn_shares * (dn_bid[i] + dn_ask[i]) * 0.5
        )
        if cap_now > capital_used_max:
            capital_used_max = cap_now

    # ---- End of market ---------------------------------------------------
    pairs_final = min(up_shares, dn_shares)
    up_res = up_shares - pairs_final
    dn_res = dn_shares - pairs_final

    final_up_bid = float(up_bid[-1])
    final_up_ask = float(up_ask[-1])
    final_dn_bid = float(dn_bid[-1])
    final_dn_ask = float(dn_ask[-1])
    final_up_mid = (final_up_bid + final_up_ask) * 0.5
    final_dn_mid = (final_dn_bid + final_dn_ask) * 0.5

    if final_up_mid > final_dn_mid:
        winner = "UP"
    elif final_dn_mid > final_up_mid:
        winner = "DOWN"
    else:
        winner = "TIE"

    payoff_pairs = pairs_final * 1.0
    if winner == "UP":
        payoff_residual = up_res * 1.0
    elif winner == "DOWN":
        payoff_residual = dn_res * 1.0
    else:
        payoff_residual = 0.5 * (up_res + dn_res)

    pnl = payoff_pairs + payoff_residual - cash_spent + rebate_total - taker_fees_paid

    avg_up_px = sum_up_px / sum_up_sz if sum_up_sz > 0 else float("nan")
    avg_dn_px = sum_dn_px / sum_dn_sz if sum_dn_sz > 0 else float("nan")
    if pairs_final > 0:
        if not (np.isnan(avg_up_px) or np.isnan(avg_dn_px)):
            avg_pair_cost_maker = avg_up_px + avg_dn_px
        else:
            avg_pair_cost_maker = float("nan")
        true_pair_cost = cash_spent / pairs_final
        pnl_per_pair = pnl / pairs_final
    else:
        avg_pair_cost_maker = float("nan")
        true_pair_cost = float("nan")
        pnl_per_pair = float("nan")

    duration_s = float((t_ms[-1] - t_ms[0]) / 1000.0)

    return LadderResult(
        market_name=market_name,
        date=date,
        duration_s=duration_s,
        n_ticks=n,
        n_up_fills=int(sum(fills_per_level_up)),
        n_dn_fills=int(sum(fills_per_level_dn)),
        up_shares_final=up_shares,
        dn_shares_final=dn_shares,
        pairs_final=pairs_final,
        up_residual=up_res,
        dn_residual=dn_res,
        avg_up_buy_price=avg_up_px,
        avg_dn_buy_price=avg_dn_px,
        avg_pair_cost_maker=avg_pair_cost_maker,
        true_pair_cost=true_pair_cost,
        cash_spent=cash_spent,
        cash_spent_maker=cash_spent_maker,
        rebate_total=rebate_total,
        taker_fees_paid=taker_fees_paid,
        payoff_pairs=payoff_pairs,
        payoff_residual=payoff_residual,
        pnl=pnl,
        pnl_per_pair=pnl_per_pair,
        capital_used_max=capital_used_max,
        fills_per_level_up=fills_per_level_up,
        fills_per_level_dn=fills_per_level_dn,
        winner=winner,
        final_up_mid=final_up_mid,
        final_dn_mid=final_dn_mid,
        up_min_bid=up_min_bid,
        dn_min_bid=dn_min_bid,
    )


# ---------------------------------------------------------------------------
#  Driver
# ---------------------------------------------------------------------------


_GLOBAL_CFG: Optional[LadderPolicy] = None


def _init_worker(cfg_dict: dict) -> None:
    global _GLOBAL_CFG
    levels = tuple(cfg_dict.pop("levels"))
    _GLOBAL_CFG = LadderPolicy(levels=levels, **cfg_dict)


def _process_one_csv(path_str: str) -> Optional[dict]:
    cfg = _GLOBAL_CFG or LadderPolicy()
    path = Path(path_str)
    wide = parse_market_csv(path)
    if wide is None:
        return None
    res = simulate_ladder(wide, cfg, path.stem, path.parent.parent.name)
    return asdict(res) if res is not None else None


def _collect_paths(
    root: Path, max_days: Optional[int], days: Optional[list[str]]
) -> list[str]:
    all_days = sorted([d for d in root.iterdir() if d.is_dir()])
    if days is not None:
        wanted = set(days)
        all_days = [d for d in all_days if d.name in wanted]
    if max_days is not None:
        all_days = all_days[-max_days:]
    paths: list[str] = []
    for day in all_days:
        csv_dir = day / "csv"
        if not csv_dir.exists():
            continue
        paths.extend(str(p) for p in sorted(csv_dir.glob("*.csv")))
    return paths


def _parse_levels(s: str) -> tuple:
    return tuple(float(x) for x in s.split(","))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-days", type=int, default=None)
    ap.add_argument("--days", nargs="*", default=None)
    ap.add_argument("--max-markets", type=int, default=None)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--out", type=str, default="utils/mm_ladder_5m.parquet")
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--levels", type=str,
                    default=",".join(f"{x:g}" for x in DEFAULT_LEVELS),
                    help="Comma-separated price levels (per side).")
    ap.add_argument("--size", type=float, default=20.0,
                    help="Shares per level.")
    ap.add_argument("--refill", action="store_true",
                    help="Re-arm a level after fill (continuous ladder).")
    ap.add_argument("--skew-max", type=float, default=0.0,
                    help="Max share imbalance |up - dn| before cancelling the heavier ladder.")
    ap.add_argument("--mom-thresh", type=float, default=0.0,
                    help="Momentum threshold (mid drop in lookback window) to cancel that side's ladder.")
    ap.add_argument("--mom-lookback", type=float, default=30.0,
                    help="Momentum lookback window in seconds.")
    ap.add_argument("--early-seconds", type=float, default=0.0,
                    help="Seconds after start to make 'early direction' decision (0 disabled).")
    ap.add_argument("--early-thresh", type=float, default=0.55,
                    help="Threshold on up_mid to lock direction (>thresh = expect UP wins).")
    ap.add_argument("--hedge-mode", choices=("hold", "taker_close"), default="hold")
    ap.add_argument("--hedge-seconds", type=float, default=25.0)
    args = ap.parse_args()

    cfg = LadderPolicy(
        levels=_parse_levels(args.levels),
        size_per_level=args.size,
        refill_on_fill=args.refill,
        skew_max_shares=args.skew_max,
        momentum_threshold=args.mom_thresh,
        momentum_lookback_seconds=args.mom_lookback,
        early_decision_seconds=args.early_seconds,
        early_decision_threshold=args.early_thresh,
        hedge_mode=args.hedge_mode,
        hedge_seconds=args.hedge_seconds,
    )

    paths = _collect_paths(PRICES_5M_ROOT, args.max_days, args.days)
    if args.max_markets is not None:
        paths = paths[: args.max_markets]
    print(f"[ladder] paths: {len(paths)}", flush=True)
    print(f"[ladder] policy: levels={cfg.levels}  size={cfg.size_per_level}  "
          f"refill={cfg.refill_on_fill}  hedge={cfg.hedge_mode}", flush=True)
    if not paths:
        print("No CSVs found.", file=sys.stderr)
        return 1

    cfg_dict = {
        "size_per_level": cfg.size_per_level,
        "refill_on_fill": cfg.refill_on_fill,
        "skew_max_shares": cfg.skew_max_shares,
        "momentum_threshold": cfg.momentum_threshold,
        "momentum_lookback_seconds": cfg.momentum_lookback_seconds,
        "early_decision_seconds": cfg.early_decision_seconds,
        "early_decision_threshold": cfg.early_decision_threshold,
        "hedge_mode": cfg.hedge_mode,
        "hedge_seconds": cfg.hedge_seconds,
        "fee_rate": cfg.fee_rate,
        "rebate_pct": cfg.rebate_pct,
        "levels": list(cfg.levels),
    }

    t0 = time.time()
    results: list[dict] = []
    n_done = 0
    with ProcessPoolExecutor(
        max_workers=args.workers, initializer=_init_worker, initargs=(cfg_dict,)
    ) as ex:
        futs = [ex.submit(_process_one_csv, p) for p in paths]
        for fut in as_completed(futs):
            n_done += 1
            r = fut.result()
            if r is not None:
                results.append(r)
            if n_done % 1000 == 0 or n_done == len(paths):
                el = time.time() - t0
                rate = n_done / el if el > 0 else 0.0
                eta = (len(paths) - n_done) / rate if rate > 0 else 0.0
                print(
                    f"  {n_done}/{len(paths)}  valid={len(results)}  "
                    f"rate={rate:.1f}/s  eta={eta:.0f}s",
                    flush=True,
                )

    df = pd.DataFrame(results)
    print(f"\n[ladder] valid: {len(df)}  ({time.time()-t0:.1f}s)")
    if df.empty:
        return 0

    if not args.no_save:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path, compression="zstd")
        print(f"[ladder] saved -> {out_path}")

    print_summary(df, cfg)
    return 0


def print_summary(df: pd.DataFrame, cfg: LadderPolicy) -> None:
    print("\n" + "=" * 78)
    print("LADDER SIMULATION SUMMARY")
    print("=" * 78)
    print(f"Markets: {len(df):>6}    Days: {df['date'].nunique() if 'date' in df.columns else '-'}")
    print(f"Levels: {cfg.levels}")
    print(f"Size/level: {cfg.size_per_level}    refill={cfg.refill_on_fill}    "
          f"hedge={cfg.hedge_mode}")
    print()

    pnl = df["pnl"]
    n_pos = int((pnl > 0).sum())
    n_neg = int((pnl < 0).sum())
    n_zer = int((pnl == 0).sum())
    n_total = max(1, len(df))

    print("=== HEADLINE ===")
    print(f"  Total P&L:          ${pnl.sum():>10.2f}")
    print(f"  Avg P&L / market:   ${pnl.mean():>10.4f}")
    print(f"  Median P&L:         ${pnl.median():>10.4f}")
    print(f"  Stddev P&L:         ${pnl.std():>10.4f}")
    print(f"  P&L > 0:            {n_pos} ({n_pos/n_total:.1%})")
    print(f"  P&L < 0:            {n_neg} ({n_neg/n_total:.1%})")
    print(f"  P&L = 0 (no fills): {n_zer} ({n_zer/n_total:.1%})")
    print()

    print("=== P&L DISTRIBUTION ($/market) ===")
    q = pnl.quantile([0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).values
    print(
        f"  p01={q[0]:.4f}  p05={q[1]:.4f}  p10={q[2]:.4f}  p25={q[3]:.4f}  "
        f"p50={q[4]:.4f}  p75={q[5]:.4f}  p90={q[6]:.4f}  p95={q[7]:.4f}  p99={q[8]:.4f}"
    )
    print(f"  min={pnl.min():.4f}  max={pnl.max():.4f}")
    print()

    print("=== FILL STATS ===")
    print(f"  Avg UP fills:           {df['n_up_fills'].mean():.2f}  "
          f"(sum: {df['n_up_fills'].sum()})")
    print(f"  Avg DN fills:           {df['n_dn_fills'].mean():.2f}")
    print(f"  Avg pairs:              {df['pairs_final'].mean():.2f}  "
          f"(median {df['pairs_final'].median():.2f})")
    print(f"  Avg residual UP:        {df['up_residual'].mean():.2f}")
    print(f"  Avg residual DOWN:      {df['dn_residual'].mean():.2f}")
    print(f"  Markets with pairs>0:   {int((df['pairs_final']>0).sum())} "
          f"({(df['pairs_final']>0).mean():.1%})")
    print()

    print("=== ECONOMICS ===")
    print(f"  Avg cash spent:         ${df['cash_spent'].mean():.4f}")
    print(f"  Avg payoff pairs:       ${df['payoff_pairs'].mean():.4f}")
    print(f"  Avg payoff residual:    ${df['payoff_residual'].mean():.4f}")
    print(f"  Avg rebate:             ${df['rebate_total'].mean():.6f}")
    print(f"  Avg taker fees:         ${df['taker_fees_paid'].mean():.6f}")
    print(f"  Avg avg_pair_cost MAKER:${df['avg_pair_cost_maker'].mean():.6f}")
    print(f"  Avg true_pair_cost:     ${df['true_pair_cost'].mean():.6f}")
    print(f"  Avg pnl_per_pair:       ${df['pnl_per_pair'].mean():.6f}  "
          f"(median {df['pnl_per_pair'].median():.6f})")
    print()

    print("=== CAPITAL ===")
    cap = df["capital_used_max"]
    q = cap.quantile([0.50, 0.75, 0.90, 0.99]).values
    print(f"  Avg cap_used_max:       ${cap.mean():.2f}")
    print(f"  Cap p50={q[0]:.2f}  p75={q[1]:.2f}  p90={q[2]:.2f}  p99={q[3]:.2f}")
    print()

    # Fills per level (averaged)
    print("=== FILLS PER LEVEL (avg per market) ===")
    K = len(cfg.levels)
    ups_per_level = np.zeros(K)
    dns_per_level = np.zeros(K)
    for _, row in df.iterrows():
        u = row["fills_per_level_up"]
        d = row["fills_per_level_dn"]
        for k in range(min(K, len(u))):
            ups_per_level[k] += u[k]
            dns_per_level[k] += d[k]
    levels_sorted = sorted(cfg.levels, reverse=True)
    n = max(1, len(df))
    print(f"  {'level':<8}{'UP fills/m':>14}{'DN fills/m':>14}")
    for k, lvl in enumerate(levels_sorted):
        print(f"  {lvl:<8.3f}{ups_per_level[k]/n:>14.4f}{dns_per_level[k]/n:>14.4f}")
    print()

    # Daily aggregate
    if "date" in df.columns:
        daily = df.groupby("date")["pnl"].agg(["sum", "count", "mean"])
        print("=== DAILY P&L ===")
        print(f"  Days: {len(daily)}")
        print(f"  Avg daily pnl:  ${daily['sum'].mean():.4f}")
        print(f"  Median daily:   ${daily['sum'].median():.4f}")
        print(f"  Best day:       ${daily['sum'].max():.4f}")
        print(f"  Worst day:      ${daily['sum'].min():.4f}")
        print(f"  Days positive:  {int((daily['sum']>0).sum())}/{len(daily)}")


if __name__ == "__main__":
    raise SystemExit(main())
