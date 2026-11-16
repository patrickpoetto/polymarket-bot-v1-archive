"""
Sequential Pair-Completion MM simulator per Polymarket BTC 5-min.

Implementa ESATTAMENTE la strategia descritta dall'utente:

  1. Piazza un maker buy near-BBO su UP e DN (al bid o bid - N tick).
  2. Quando una gamba filla a prezzo X, cancella l'altra gamba e piazza
     una "completion" buy sull'altra gamba a prezzo (target_pair_cost - X).
     Es. UP filla a 0.49 -> piazza DN a 0.99 - 0.49 = 0.50.
  3. Quando la completion filla, la coppia è chiusa con profitto:
     payoff = $1, cost = X + (target - X) = target.
     Profit = 1 - target + rebate.
  4. Se la completion non filla entro `completion_timeout_seconds`, c'è
     opzione di:
        - "hold": tieni il single-leg fino a expiry
        - "lower": abbassa la completion price seguendo il BBO che scende
                   (ma non oltre il break-even target)
        - "taker_close": chiudi la coppia da taker
  5. Dopo una pair completata, ricomincia il ciclo (piazza nuova UP+DN al BBO)
     fino a `max_pairs_per_market`.

Uso:
    python -m utils.mm_simulate_seq --max-days 7 --no-save
    python -m utils.mm_simulate_seq --target 0.98 --offset 1
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
class SeqPolicy:
    target_pair_cost: float = 0.99
    max_pairs_per_market: int = 200
    bid_offset_ticks: int = 0
    tick: float = 0.01
    completion_timeout_seconds: float = 60.0
    completion_action: str = "lower"  # "hold" | "lower" | "taker_close"
    # When 'lower': the completion price chases the bid down 1 tick at a time
    # but never above target_pair_cost - first_fill_price
    refill_after_pair: bool = True
    # Stop placing new orders when this many seconds remain
    stop_buy_seconds: float = 30.0
    # Early direction lock: a t=early_seconds, se up_mid >= early_thresh,
    # cancella PERMANENTEMENTE il lato DN (e simmetrico). 0 = disabilitato.
    early_decision_seconds: float = 0.0
    early_decision_threshold: float = 0.55
    fee_rate: float = 0.07
    rebate_pct: float = 0.20


@dataclass
class SeqResult:
    market_name: str
    date: str
    duration_s: float
    n_ticks: int
    n_pairs: int
    n_up_first_fills: int
    n_dn_first_fills: int
    n_completions: int
    n_completion_timeouts: int
    n_taker_closes: int
    up_residual: float
    dn_residual: float
    cash_spent: float
    cash_spent_maker: float
    rebate_total: float
    taker_fees_paid: float
    payoff_pairs: float
    payoff_residual: float
    pnl: float
    pnl_per_pair: float
    avg_first_fill_price: float
    avg_completion_price: float
    avg_pair_cost_maker: float
    capital_used_max: float
    winner: str


# ---------------------------------------------------------------------------


def simulate_seq(wide: pd.DataFrame, cfg: SeqPolicy, market_name: str,
                 date: str) -> Optional[SeqResult]:
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

    EPS = 1e-9
    fee_rate = cfg.fee_rate
    rebate_factor = cfg.rebate_pct * cfg.fee_rate
    offset = cfg.bid_offset_ticks * cfg.tick

    # State machine
    # state == 'IDLE'  : both UP and DN waiting for first fill
    # state == 'UP_LEG': UP filled, waiting DN completion
    # state == 'DN_LEG': DN filled, waiting UP completion
    state = "IDLE"
    up_quote = None  # active UP buy price (None if no quote)
    dn_quote = None
    first_fill_price = 0.0
    first_fill_t_ms = 0
    completion_price = 0.0

    n_pairs = 0
    n_up_first = 0
    n_dn_first = 0
    n_completions = 0
    n_timeouts = 0
    n_taker_closes = 0

    sum_first_px = 0.0
    sum_completion_px = 0.0
    n_first_seen = 0
    n_compl_seen = 0

    cash_spent = 0.0
    cash_spent_maker = 0.0
    rebate_total = 0.0
    taker_fees = 0.0
    capital_used_max = 0.0

    # residual tracking (single legs that never completed)
    up_residual = 0.0
    dn_residual = 0.0
    residual_up_avg_cost = 0.0
    residual_dn_avg_cost = 0.0
    residual_up_count = 0.0
    residual_dn_count = 0.0

    def _place_idle_quotes(i: int):
        """Place fresh maker buys near BBO on both sides (rispetta block_*)."""
        nonlocal up_quote, dn_quote
        if block_up:
            up_quote = None
        else:
            up_q = up_bid[i] - offset
            up_quote = up_q if up_q < up_ask[i] - EPS else None
        if block_dn:
            dn_quote = None
        else:
            dn_q = dn_bid[i] - offset
            dn_quote = dn_q if dn_q < dn_ask[i] - EPS else None

    def _maker_fill_check(quote: Optional[float], cur_bid: float) -> bool:
        """Restituisce True se un ordine maker buy a `quote` è stato attraversato
        dal bid corrente (back-of-queue: bid strict below quote)."""
        if quote is None:
            return False
        return cur_bid < quote - EPS

    # Early direction lock state
    early_done = False
    block_up = False  # blocca permanentemente quote UP (DN winning expected)
    block_dn = False  # blocca permanentemente quote DN (UP winning expected)
    t_start_ms = int(t_ms[0])
    early_ms = cfg.early_decision_seconds * 1000.0

    # Inizializza - se early_seconds > 0, NON piazzare alle t=0; aspetta la
    # decisione direzionale.
    if cfg.early_decision_seconds <= 0:
        _place_idle_quotes(0)

    for i in range(n):
        t_remaining = t_left[i]

        # Early direction decision (one-shot)
        if cfg.early_decision_seconds > 0 and not early_done and (t_ms[i] - t_start_ms) >= early_ms:
            early_done = True
            cur_up_mid = (up_bid[i] + up_ask[i]) * 0.5
            if cur_up_mid >= cfg.early_decision_threshold:
                block_dn = True
                if state == "DN_LEG":
                    dn_residual += 1.0
                    residual_dn_avg_cost = (
                        (residual_dn_avg_cost * residual_dn_count + first_fill_price)
                        / (residual_dn_count + 1)
                    )
                    residual_dn_count += 1
                    state = "IDLE"
                dn_quote = None
            elif cur_up_mid <= (1.0 - cfg.early_decision_threshold):
                block_up = True
                if state == "UP_LEG":
                    up_residual += 1.0
                    residual_up_avg_cost = (
                        (residual_up_avg_cost * residual_up_count + first_fill_price)
                        / (residual_up_count + 1)
                    )
                    residual_up_count += 1
                    state = "IDLE"
                up_quote = None
            # Posiziona i quote (per la prima volta se early_seconds > 0)
            if state == "IDLE" and t_remaining > cfg.stop_buy_seconds:
                _place_idle_quotes(i)

        # Hard re-zero for permanently blocked sides
        if block_up:
            up_quote = None
        if block_dn:
            dn_quote = None

        # Stop-buy: cancella tutti i quote attivi negli ultimi N secondi
        if t_remaining <= cfg.stop_buy_seconds:
            up_quote = None
            dn_quote = None

        # ---- Fill check ----
        if state == "IDLE":
            up_filled = _maker_fill_check(up_quote, up_bid[i])
            dn_filled = _maker_fill_check(dn_quote, dn_bid[i])
            if up_filled and dn_filled:
                # Entrambe nel medesimo tick: trattiamo come pair atomico al
                # somma dei due prezzi (rara ma possibile col tick discreto)
                px_up = up_quote
                px_dn = dn_quote
                cost = px_up + px_dn
                cash_spent += cost
                cash_spent_maker += cost
                rebate_total += rebate_factor * (
                    px_up * (1.0 - px_up) + px_dn * (1.0 - px_dn)
                )
                n_pairs += 1
                sum_first_px += px_up
                sum_completion_px += px_dn
                n_first_seen += 1
                n_compl_seen += 1
                state = "IDLE"
                up_quote = None
                dn_quote = None
                if cfg.refill_after_pair and n_pairs < cfg.max_pairs_per_market and t_remaining > cfg.stop_buy_seconds:
                    _place_idle_quotes(i)
            elif up_filled:
                px = up_quote
                cash_spent += px
                cash_spent_maker += px
                rebate_total += rebate_factor * px * (1.0 - px)
                first_fill_price = px
                first_fill_t_ms = int(t_ms[i])
                n_up_first += 1
                sum_first_px += px
                n_first_seen += 1
                # Cancella DN quote esistente, piazza completion DN
                up_quote = None
                completion_price = cfg.target_pair_cost - px
                # post-only safety
                if completion_price < dn_ask[i] - EPS:
                    dn_quote = completion_price
                else:
                    dn_quote = None
                state = "UP_LEG"
            elif dn_filled:
                px = dn_quote
                cash_spent += px
                cash_spent_maker += px
                rebate_total += rebate_factor * px * (1.0 - px)
                first_fill_price = px
                first_fill_t_ms = int(t_ms[i])
                n_dn_first += 1
                sum_first_px += px
                n_first_seen += 1
                dn_quote = None
                completion_price = cfg.target_pair_cost - px
                if completion_price < up_ask[i] - EPS:
                    up_quote = completion_price
                else:
                    up_quote = None
                state = "DN_LEG"

        elif state == "UP_LEG":
            # Aspettiamo DN completion
            elapsed_ms = int(t_ms[i]) - first_fill_t_ms
            timed_out = elapsed_ms >= cfg.completion_timeout_seconds * 1000

            if cfg.completion_action == "lower" and dn_quote is not None and not timed_out:
                # Abbassa la completion price seguendo il bid (1 tick alla volta)
                # MA sempre <= target - first_fill (per non perdere il discount)
                max_compl = cfg.target_pair_cost - first_fill_price
                follow_q = dn_bid[i] - offset
                new_q = min(max_compl, follow_q)
                if new_q < dn_ask[i] - EPS:
                    dn_quote = new_q

            if _maker_fill_check(dn_quote, dn_bid[i]):
                px = dn_quote
                cash_spent += px
                cash_spent_maker += px
                rebate_total += rebate_factor * px * (1.0 - px)
                sum_completion_px += px
                n_compl_seen += 1
                n_pairs += 1
                n_completions += 1
                state = "IDLE"
                up_quote = None
                dn_quote = None
                if cfg.refill_after_pair and n_pairs < cfg.max_pairs_per_market and t_remaining > cfg.stop_buy_seconds:
                    _place_idle_quotes(i)
            elif timed_out:
                # Completion timeout
                n_timeouts += 1
                if cfg.completion_action == "taker_close":
                    # Compra DN come taker al ASK
                    px = dn_ask[i]
                    if not np.isnan(px) and px > 0:
                        cash_spent += px
                        taker_fees += fee_rate * px * (1.0 - px)
                        n_taker_closes += 1
                        n_pairs += 1
                # else hold: lascia residual
                if state != "IDLE":
                    # diventa residual UP
                    if cfg.completion_action != "taker_close":
                        up_residual += 1.0
                        residual_up_avg_cost = (
                            (residual_up_avg_cost * residual_up_count + first_fill_price)
                            / (residual_up_count + 1)
                        )
                        residual_up_count += 1
                state = "IDLE"
                up_quote = None
                dn_quote = None
                # Riapri il ciclo se possibile
                if (cfg.refill_after_pair
                    and n_pairs + int(up_residual) + int(dn_residual) < cfg.max_pairs_per_market
                    and t_remaining > cfg.stop_buy_seconds):
                    _place_idle_quotes(i)

        elif state == "DN_LEG":
            elapsed_ms = int(t_ms[i]) - first_fill_t_ms
            timed_out = elapsed_ms >= cfg.completion_timeout_seconds * 1000

            if cfg.completion_action == "lower" and up_quote is not None and not timed_out:
                max_compl = cfg.target_pair_cost - first_fill_price
                follow_q = up_bid[i] - offset
                new_q = min(max_compl, follow_q)
                if new_q < up_ask[i] - EPS:
                    up_quote = new_q

            if _maker_fill_check(up_quote, up_bid[i]):
                px = up_quote
                cash_spent += px
                cash_spent_maker += px
                rebate_total += rebate_factor * px * (1.0 - px)
                sum_completion_px += px
                n_compl_seen += 1
                n_pairs += 1
                n_completions += 1
                state = "IDLE"
                up_quote = None
                dn_quote = None
                if cfg.refill_after_pair and n_pairs < cfg.max_pairs_per_market and t_remaining > cfg.stop_buy_seconds:
                    _place_idle_quotes(i)
            elif timed_out:
                n_timeouts += 1
                if cfg.completion_action == "taker_close":
                    px = up_ask[i]
                    if not np.isnan(px) and px > 0:
                        cash_spent += px
                        taker_fees += fee_rate * px * (1.0 - px)
                        n_taker_closes += 1
                        n_pairs += 1
                if state != "IDLE":
                    if cfg.completion_action != "taker_close":
                        dn_residual += 1.0
                        residual_dn_avg_cost = (
                            (residual_dn_avg_cost * residual_dn_count + first_fill_price)
                            / (residual_dn_count + 1)
                        )
                        residual_dn_count += 1
                state = "IDLE"
                up_quote = None
                dn_quote = None
                if (cfg.refill_after_pair
                    and n_pairs + int(up_residual) + int(dn_residual) < cfg.max_pairs_per_market
                    and t_remaining > cfg.stop_buy_seconds):
                    _place_idle_quotes(i)

        # Track capital
        cap_now = (n_pairs * 1.0  # locked $1 per pair (cost basis is what was paid, but use approx)
                   + up_residual * (up_bid[i] + up_ask[i]) * 0.5
                   + dn_residual * (dn_bid[i] + dn_ask[i]) * 0.5)
        if cap_now > capital_used_max:
            capital_used_max = cap_now

    # End of market: handle any active state with single-leg (state UP_LEG or DN_LEG)
    if state == "UP_LEG":
        up_residual += 1.0
        residual_up_avg_cost = (
            (residual_up_avg_cost * residual_up_count + first_fill_price)
            / (residual_up_count + 1)
        )
        residual_up_count += 1
    elif state == "DN_LEG":
        dn_residual += 1.0
        residual_dn_avg_cost = (
            (residual_dn_avg_cost * residual_dn_count + first_fill_price)
            / (residual_dn_count + 1)
        )
        residual_dn_count += 1

    final_up_mid = (up_bid[-1] + up_ask[-1]) * 0.5
    final_dn_mid = (dn_bid[-1] + dn_ask[-1]) * 0.5
    if final_up_mid > final_dn_mid:
        winner = "UP"
    elif final_dn_mid > final_up_mid:
        winner = "DOWN"
    else:
        winner = "TIE"

    payoff_pairs = float(n_pairs)
    if winner == "UP":
        payoff_residual = up_residual
    elif winner == "DOWN":
        payoff_residual = dn_residual
    else:
        payoff_residual = 0.5 * (up_residual + dn_residual)

    pnl = payoff_pairs + payoff_residual - cash_spent + rebate_total - taker_fees

    avg_first_fill = sum_first_px / n_first_seen if n_first_seen > 0 else float("nan")
    avg_compl = sum_completion_px / n_compl_seen if n_compl_seen > 0 else float("nan")
    if n_pairs > 0 and (n_first_seen > 0 and n_compl_seen > 0):
        avg_pair_cost_maker = avg_first_fill + avg_compl
    else:
        avg_pair_cost_maker = float("nan")
    pnl_per_pair = pnl / n_pairs if n_pairs > 0 else float("nan")

    duration_s = float((t_ms[-1] - t_ms[0]) / 1000.0)

    return SeqResult(
        market_name=market_name,
        date=date,
        duration_s=duration_s,
        n_ticks=n,
        n_pairs=n_pairs,
        n_up_first_fills=n_up_first,
        n_dn_first_fills=n_dn_first,
        n_completions=n_completions,
        n_completion_timeouts=n_timeouts,
        n_taker_closes=n_taker_closes,
        up_residual=up_residual,
        dn_residual=dn_residual,
        cash_spent=cash_spent,
        cash_spent_maker=cash_spent_maker,
        rebate_total=rebate_total,
        taker_fees_paid=taker_fees,
        payoff_pairs=payoff_pairs,
        payoff_residual=payoff_residual,
        pnl=pnl,
        pnl_per_pair=pnl_per_pair,
        avg_first_fill_price=avg_first_fill,
        avg_completion_price=avg_compl,
        avg_pair_cost_maker=avg_pair_cost_maker,
        capital_used_max=capital_used_max,
        winner=winner,
    )


# ---------------------------------------------------------------------------


_GLOBAL_CFG: Optional[SeqPolicy] = None


def _init_worker(cfg_dict: dict) -> None:
    global _GLOBAL_CFG
    _GLOBAL_CFG = SeqPolicy(**cfg_dict)


def _process_one(path_str: str) -> Optional[dict]:
    cfg = _GLOBAL_CFG or SeqPolicy()
    p = Path(path_str)
    wide = parse_market_csv(p)
    if wide is None:
        return None
    res = simulate_seq(wide, cfg, p.stem, p.parent.parent.name)
    return asdict(res) if res is not None else None


def _collect_paths(root: Path, max_days: Optional[int],
                   days: Optional[list]) -> list:
    all_days = sorted([d for d in root.iterdir() if d.is_dir()])
    if days is not None:
        wanted = set(days)
        all_days = [d for d in all_days if d.name in wanted]
    if max_days is not None:
        all_days = all_days[-max_days:]
    paths: list = []
    for day in all_days:
        csv_dir = day / "csv"
        if csv_dir.exists():
            paths.extend(str(p) for p in sorted(csv_dir.glob("*.csv")))
    return paths


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-days", type=int, default=None)
    ap.add_argument("--days", nargs="*", default=None)
    ap.add_argument("--max-markets", type=int, default=None)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--out", type=str, default="utils/mm_seq_5m.parquet")
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--target", type=float, default=0.99,
                    help="Target pair cost (e.g. 0.99 = 1c discount, 0.97 = 3c).")
    ap.add_argument("--offset", type=int, default=0,
                    help="Bid offset ticks (0 = at BBO, 1 = 1c below BBO, etc).")
    ap.add_argument("--max-pairs", type=int, default=200)
    ap.add_argument("--timeout", type=float, default=60.0,
                    help="Seconds to wait for completion before timeout.")
    ap.add_argument("--action", choices=("hold", "lower", "taker_close"),
                    default="lower", help="Completion timeout action.")
    ap.add_argument("--no-refill", action="store_true",
                    help="Don't restart cycle after a pair is completed.")
    ap.add_argument("--stop-buy-seconds", type=float, default=30.0)
    ap.add_argument("--early-seconds", type=float, default=0.0,
                    help="Sec after start to lock direction (0 disabled).")
    ap.add_argument("--early-thresh", type=float, default=0.55,
                    help="up_mid threshold to lock direction (UP wins side).")
    args = ap.parse_args()

    cfg = SeqPolicy(
        target_pair_cost=args.target,
        bid_offset_ticks=args.offset,
        max_pairs_per_market=args.max_pairs,
        completion_timeout_seconds=args.timeout,
        completion_action=args.action,
        refill_after_pair=not args.no_refill,
        stop_buy_seconds=args.stop_buy_seconds,
        early_decision_seconds=args.early_seconds,
        early_decision_threshold=args.early_thresh,
    )

    paths = _collect_paths(PRICES_5M_ROOT, args.max_days, args.days)
    if args.max_markets is not None:
        paths = paths[: args.max_markets]
    print(f"[seq] paths: {len(paths)}", flush=True)
    print(f"[seq] policy: target={cfg.target_pair_cost}  offset={cfg.bid_offset_ticks}  "
          f"timeout={cfg.completion_timeout_seconds}s  action={cfg.completion_action}  "
          f"refill={cfg.refill_after_pair}", flush=True)
    if not paths:
        print("No CSVs found.", file=sys.stderr)
        return 1

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
            if n_done % 1000 == 0 or n_done == len(paths):
                el = time.time() - t0
                rate = n_done / el if el > 0 else 0.0
                eta = (len(paths) - n_done) / rate if rate > 0 else 0.0
                print(f"  {n_done}/{len(paths)}  valid={len(results)}  rate={rate:.1f}/s  eta={eta:.0f}s", flush=True)

    df = pd.DataFrame(results)
    print(f"\n[seq] valid: {len(df)}  ({time.time()-t0:.1f}s)")
    if df.empty:
        return 0
    if not args.no_save:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(args.out, compression="zstd")
        print(f"[seq] saved -> {args.out}")
    print_summary(df, cfg)
    return 0


def print_summary(df: pd.DataFrame, cfg: SeqPolicy) -> None:
    n = max(1, len(df))
    pnl = df["pnl"]
    print("\n" + "=" * 78)
    print("SEQUENTIAL PAIR-COMPLETION SUMMARY")
    print("=" * 78)
    print(f"Markets: {len(df)}  Days: {df['date'].nunique() if 'date' in df.columns else '-'}")
    print(f"Policy: target={cfg.target_pair_cost}  offset={cfg.bid_offset_ticks}t  "
          f"timeout={cfg.completion_timeout_seconds}s  action={cfg.completion_action}")
    print()
    print("=== HEADLINE ===")
    print(f"  Total P&L:          ${pnl.sum():>10.2f}")
    print(f"  Avg P&L / market:   ${pnl.mean():>10.4f}")
    print(f"  Median P&L:         ${pnl.median():>10.4f}")
    print(f"  Stddev P&L:         ${pnl.std():>10.4f}")
    print(f"  P&L > 0 markets:    {int((pnl>0).sum())} ({(pnl>0).mean():.1%})")
    print(f"  P&L < 0 markets:    {int((pnl<0).sum())} ({(pnl<0).mean():.1%})")
    print(f"  P&L = 0 markets:    {int((pnl==0).sum())} ({(pnl==0).mean():.1%})")
    print()
    print("=== PAIRS ===")
    print(f"  Avg pairs/market:        {df['n_pairs'].mean():.2f}  "
          f"(median {df['n_pairs'].median():.0f})")
    print(f"  Avg completions:         {df['n_completions'].mean():.2f}")
    print(f"  Avg timeouts:            {df['n_completion_timeouts'].mean():.2f}")
    print(f"  Avg taker closes:        {df['n_taker_closes'].mean():.2f}")
    print(f"  Avg UP first fills:      {df['n_up_first_fills'].mean():.2f}")
    print(f"  Avg DN first fills:      {df['n_dn_first_fills'].mean():.2f}")
    print(f"  Markets with pairs>0:    {int((df['n_pairs']>0).sum())} ({(df['n_pairs']>0).mean():.1%})")
    print()
    print("=== ECONOMICS ===")
    print(f"  Avg cash spent:           ${df['cash_spent'].mean():.4f}")
    print(f"  Avg payoff pairs:         ${df['payoff_pairs'].mean():.4f}")
    print(f"  Avg payoff residual:      ${df['payoff_residual'].mean():.4f}")
    print(f"  Avg rebate:               ${df['rebate_total'].mean():.6f}")
    print(f"  Avg taker fees:           ${df['taker_fees_paid'].mean():.6f}")
    print(f"  Avg first fill price:     {df['avg_first_fill_price'].mean():.4f}")
    print(f"  Avg completion price:     {df['avg_completion_price'].mean():.4f}")
    print(f"  Avg pair cost MAKER:      {df['avg_pair_cost_maker'].mean():.4f}")
    print(f"  Avg pnl/pair:             ${df['pnl_per_pair'].mean():.6f}  "
          f"(median {df['pnl_per_pair'].median():.6f})")
    print()
    print("=== RESIDUALS ===")
    print(f"  Avg UP residual:        {df['up_residual'].mean():.3f}")
    print(f"  Avg DN residual:        {df['dn_residual'].mean():.3f}")
    print(f"  Markets w/ residual>0:  {int(((df['up_residual']>0)|(df['dn_residual']>0)).sum())} "
          f"({((df['up_residual']>0)|(df['dn_residual']>0)).mean():.1%})")
    print()
    if "date" in df.columns:
        daily = df.groupby("date")["pnl"].agg(["sum", "count", "mean"])
        print("=== DAILY P&L ===")
        print(f"  Days: {len(daily)}")
        print(f"  Avg daily pnl:    ${daily['sum'].mean():.2f}")
        print(f"  Median daily:     ${daily['sum'].median():.2f}")
        print(f"  Best day:         ${daily['sum'].max():.2f}")
        print(f"  Worst day:        ${daily['sum'].min():.2f}")
        print(f"  Days positive:    {int((daily['sum']>0).sum())}/{len(daily)}")


if __name__ == "__main__":
    raise SystemExit(main())
