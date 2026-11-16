"""
SIGNAL-LEAN strategy: usa il primo fill maker come segnale.

Logica:
  1. Piazza maker BUY post_only a BBO-1c su ENTRAMBI i lati.
  2. Quando uno dei due fillа (leg1 su side X a prezzo P1):
     - X è probabilmente il LOSER (bid che scende = svalutazione).
     - Cancel maker su X. Niente più accumulo su X. (Tieni 1 share residua come 'scommessa persa'.)
     - KEEP maker su Y a BBO-1c (segui BBO). Y è probabilmente il WINNER.
     - Continua ad accumulare Y fin oltre la fine della trade window (entro cap).
  3. A expiry:
     - Se Y vince: Y_shares × $1, X × $0. Profit dipende da quanto accumulato su Y.
     - Se X vince (signal wrong): X × $1, Y_shares × $0. Loss = Y_cash - X residual_payoff.

Idee chiave:
  - Sfrutta il bias adverse-selection del primo fill maker
  - Il vincitore "tira su" il suo prezzo lentamente con piccoli dip → BBO-follow su Y fillа
  - Cap su Y per limitare downside in caso di signal misfire.
  - Optional: filtro di confirmation (es. dopo leg1 aspetta K secondi e verifica trend Y).
  - Optional: bidsum filter al momento di leg1 (= il mercato è oscillante, signal poco affidabile).
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
class SignalLeanPolicy:
    tick: float = 0.01
    bbo_offset_ticks: int = 1
    fill_size: float = 1.0

    # Cap inventory on each side
    max_inv_loser: float = 1.0       # leg1 = signal trigger, kept at 1
    max_inv_winner: float = 10.0     # how many we accumulate on suspected winner

    trade_window_seconds: float = 300.0

    # Initial pre-signal phase: maker on BOTH sides at BBO-1c
    # Only activates after `pre_signal_min_seconds` (avoid pure-noise fills at t=0)
    pre_signal_min_seconds: float = 0.0
    # Cap on initial fill price (= max P1 for leg1)
    initial_max_price: float = 0.55

    # After signal: confirmation window (in seconds)
    # If 0 = immediate. Else wait this long, verify Y_bid is rising (or stable) before committing.
    confirmation_seconds: float = 0.0
    # Confirmation rule: Y_bid must be >= Y_bid at signal time + this delta
    confirmation_delta: float = 0.0

    # Bidsum gate at signal moment (skip if bidsum too high - market may be oscillating, low signal value)
    bidsum_max_at_signal: float = 0.0  # 0 = disabled

    # End-of-window force close (sell residual at bid). 0 = hold to expiry.
    close_at_seconds: float = 0.0
    close_residual_only_loser: bool = True  # only sell the LOSER residual (1 share); keep winner

    # Continuous abort: if Y_bid drops more than this BELOW Y_bid at signal time,
    # stop accumulating Y. This catches the 'wrong signal' early.
    y_drop_abort: float = 0.0   # 0 = disabled. else cents (e.g. 0.05 = abort if Y_bid -5c)
    # Stop-loss: if abort triggers AND we have residual on either side, taker-sell all at current bid
    stop_loss_taker_close: bool = False

    fee_rate: float = 0.07
    rebate_pct: float = 0.20


@dataclass
class SignalLeanResult:
    market_name: str
    date: str
    signal_side: str           # "UP", "DOWN", or "NONE" (no fill in window)
    signal_time_s: float       # time of leg1 fill
    signal_price: float        # P1 fill price
    bidsum_at_signal: float
    n_winner_fills: int
    loser_shares: float        # = 1 if signal happened, 0 otherwise
    winner_shares: float
    loser_cash: float
    winner_cash: float
    avg_winner_price: float
    cash_total: float
    rebate_total: float
    taker_close_pnl: float
    payoff_loser: float
    payoff_winner: float
    pnl: float
    winner: str
    signal_correct: bool       # True if X (loser-bet side) actually lost


def simulate_signal_lean(
    wide: pd.DataFrame, cfg: SignalLeanPolicy, market_name: str, date: str
) -> Optional[SignalLeanResult]:
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
    pre_min_ms = int(cfg.pre_signal_min_seconds * 1000.0)
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

    # State
    signal_side: Optional[str] = None  # the side that filled FIRST (= suspected loser)
    signal_idx: Optional[int] = None
    signal_time_s = 0.0
    signal_price = 0.0
    bidsum_at_signal = 0.0
    confirmed = False  # only matters if confirmation_seconds > 0

    loser_shares = 0.0
    loser_cash = 0.0
    winner_shares = 0.0
    winner_cash = 0.0
    n_winner_fills = 0
    rebate_total = 0.0
    taker_close_pnl = 0.0
    closed = False

    prev_ub = up_bid[0]
    prev_db = dn_bid[0]

    for i in range(1, n):
        elapsed_ms = t_ms[i] - t_start
        if elapsed_ms > win_ms:
            break

        # End-of-window taker close (close LOSER residual)
        if close_ms is not None and not closed and elapsed_ms >= close_ms:
            if cfg.close_residual_only_loser and signal_side is not None and loser_shares > 0:
                if signal_side == "UP":
                    bid_close = up_bid[i]
                else:
                    bid_close = dn_bid[i]
                if bid_close > 0:
                    revenue = bid_close * loser_shares
                    fee = cfg.fee_rate * bid_close * (1 - bid_close) * loser_shares
                    taker_close_pnl += revenue - fee
                    loser_shares = 0.0
            closed = True
            break

        ub = up_bid[i]
        db = dn_bid[i]

        # --- PRE-SIGNAL PHASE: maker on BOTH sides, wait for first fill ---
        if signal_side is None:
            if elapsed_ms < pre_min_ms:
                prev_ub = ub
                prev_db = db
                continue

            # Check fill on UP
            up_fill = False
            if ub < prev_ub - EPS:
                old_order_px = prev_ub - offset
                if (old_order_px > 0 and old_order_px <= cfg.initial_max_price + EPS
                        and ub < old_order_px - EPS):
                    # UP fills — UP is the SIGNAL (suspected loser)
                    fill_px = old_order_px
                    bidsum_at_signal = ub + db
                    if cfg.bidsum_max_at_signal > 0 and bidsum_at_signal > cfg.bidsum_max_at_signal + EPS:
                        # signal SKIPPED: market oscillating, low signal value
                        prev_ub = ub
                        prev_db = db
                        continue
                    signal_side = "UP"
                    signal_idx = i
                    signal_time_s = elapsed_ms / 1000.0
                    signal_price = fill_px
                    sz = min(size, cfg.max_inv_loser - loser_shares)
                    if sz > 0:
                        loser_cash += fill_px * sz
                        loser_shares += sz
                        rebate_total += rebate_factor * fill_px * (1 - fill_px) * sz
                    up_fill = True

            # Check fill on DN (only if no signal yet)
            if signal_side is None and db < prev_db - EPS:
                old_order_px = prev_db - offset
                if (old_order_px > 0 and old_order_px <= cfg.initial_max_price + EPS
                        and db < old_order_px - EPS):
                    fill_px = old_order_px
                    bidsum_at_signal = ub + db
                    if cfg.bidsum_max_at_signal > 0 and bidsum_at_signal > cfg.bidsum_max_at_signal + EPS:
                        prev_ub = ub
                        prev_db = db
                        continue
                    signal_side = "DOWN"
                    signal_idx = i
                    signal_time_s = elapsed_ms / 1000.0
                    signal_price = fill_px
                    sz = min(size, cfg.max_inv_loser - loser_shares)
                    if sz > 0:
                        loser_cash += fill_px * sz
                        loser_shares += sz
                        rebate_total += rebate_factor * fill_px * (1 - fill_px) * sz

            prev_ub = ub
            prev_db = db
            continue

        # --- POST-SIGNAL PHASE: maker only on the OPPOSITE side ---
        # signal_side == "UP" means UP is loser, accumulate DOWN
        # signal_side == "DOWN" means DN is loser, accumulate UP

        winner_side = "DOWN" if signal_side == "UP" else "UP"

        # Confirmation logic (optional)
        if cfg.confirmation_seconds > 0 and not confirmed:
            elapsed_since_signal_s = (t_ms[i] - t_ms[signal_idx]) / 1000.0
            if elapsed_since_signal_s < cfg.confirmation_seconds:
                # Don't fill on winner side until confirmation passes
                prev_ub = ub
                prev_db = db
                continue
            else:
                # Verify confirmation: winner_bid must have risen (or stayed)
                if winner_side == "UP":
                    winner_bid_at_sig = up_bid[signal_idx]
                    if ub < winner_bid_at_sig - cfg.confirmation_delta - EPS:
                        # winner_bid dropped, confirmation FAILED → abort the rest
                        prev_ub = ub
                        prev_db = db
                        signal_side = "ABORT"
                        continue
                else:
                    winner_bid_at_sig = dn_bid[signal_idx]
                    if db < winner_bid_at_sig - cfg.confirmation_delta - EPS:
                        prev_ub = ub
                        prev_db = db
                        signal_side = "ABORT"
                        continue
                confirmed = True

        # Continuous abort: if winner_side bid has dropped too much, abort
        if cfg.y_drop_abort > 0 and signal_idx is not None:
            triggered = False
            if winner_side == "UP":
                y_bid_at_sig = up_bid[signal_idx]
                if ub < y_bid_at_sig - cfg.y_drop_abort - EPS:
                    triggered = True
            else:
                y_bid_at_sig = dn_bid[signal_idx]
                if db < y_bid_at_sig - cfg.y_drop_abort - EPS:
                    triggered = True
            if triggered:
                # Optional stop-loss taker: liquidate residual at current bids
                if cfg.stop_loss_taker_close:
                    # the loser_bet has 'loser_shares' on signal_side at price signal_price
                    # the winner_bet has 'winner_shares' on winner_side at avg_winner_price
                    if signal_side == "UP" and loser_shares > 0 and ub > 0:
                        rev = ub * loser_shares; fee = cfg.fee_rate * ub * (1 - ub) * loser_shares
                        taker_close_pnl += rev - fee; loser_shares = 0.0
                    if signal_side == "DOWN" and loser_shares > 0 and db > 0:
                        rev = db * loser_shares; fee = cfg.fee_rate * db * (1 - db) * loser_shares
                        taker_close_pnl += rev - fee; loser_shares = 0.0
                    if winner_side == "UP" and winner_shares > 0 and ub > 0:
                        rev = ub * winner_shares; fee = cfg.fee_rate * ub * (1 - ub) * winner_shares
                        taker_close_pnl += rev - fee; winner_shares = 0.0
                    if winner_side == "DOWN" and winner_shares > 0 and db > 0:
                        rev = db * winner_shares; fee = cfg.fee_rate * db * (1 - db) * winner_shares
                        taker_close_pnl += rev - fee; winner_shares = 0.0
                signal_side = "ABORT"
                prev_ub = ub
                prev_db = db
                continue

        # Maker BUY on winner side at BBO-1c, fills on bid drops
        if winner_side == "UP":
            if winner_shares < cfg.max_inv_winner - EPS and ub < prev_ub - EPS:
                old_order_px = prev_ub - offset
                if old_order_px > 0 and ub < old_order_px - EPS:
                    fill_px = old_order_px
                    sz = min(size, cfg.max_inv_winner - winner_shares)
                    winner_cash += fill_px * sz
                    winner_shares += sz
                    rebate_total += rebate_factor * fill_px * (1 - fill_px) * sz
                    n_winner_fills += 1
        else:  # winner_side == "DOWN"
            if winner_shares < cfg.max_inv_winner - EPS and db < prev_db - EPS:
                old_order_px = prev_db - offset
                if old_order_px > 0 and db < old_order_px - EPS:
                    fill_px = old_order_px
                    sz = min(size, cfg.max_inv_winner - winner_shares)
                    winner_cash += fill_px * sz
                    winner_shares += sz
                    rebate_total += rebate_factor * fill_px * (1 - fill_px) * sz
                    n_winner_fills += 1

        prev_ub = ub
        prev_db = db

    # --- End of market: payoffs ---
    avg_winner_price = winner_cash / winner_shares if winner_shares > 0 else 0.0

    # Determine which inventory bet on which outcome
    if signal_side is None or signal_side == "ABORT":
        # No active position (or aborted). Loser=0 shares. Winner=0 shares.
        payoff_loser = 0.0
        payoff_winner = 0.0
        signal_correct = False
    else:
        loser_bet_side = signal_side  # we have shares on this (1)
        winner_bet_side = "DOWN" if signal_side == "UP" else "UP"
        # Loser side: pays $1 if it actually wins (SIGNAL WRONG)
        if loser_bet_side == winner:
            payoff_loser = loser_shares * 1.0
            signal_correct = False
        elif loser_bet_side != winner and winner != "TIE":
            payoff_loser = 0.0
            signal_correct = True
        else:  # TIE
            payoff_loser = 0.5 * loser_shares
            signal_correct = False
        # Winner-bet side: pays $1 if it actually wins (SIGNAL CORRECT)
        if winner_bet_side == winner:
            payoff_winner = winner_shares * 1.0
        elif winner_bet_side != winner and winner != "TIE":
            payoff_winner = 0.0
        else:
            payoff_winner = 0.5 * winner_shares

    cash_total = loser_cash + winner_cash
    pnl = payoff_loser + payoff_winner - cash_total + rebate_total + taker_close_pnl

    return SignalLeanResult(
        market_name=market_name, date=date,
        signal_side=signal_side or "NONE",
        signal_time_s=signal_time_s, signal_price=signal_price,
        bidsum_at_signal=bidsum_at_signal,
        n_winner_fills=n_winner_fills,
        loser_shares=loser_shares, winner_shares=winner_shares,
        loser_cash=loser_cash, winner_cash=winner_cash,
        avg_winner_price=avg_winner_price,
        cash_total=cash_total, rebate_total=rebate_total,
        taker_close_pnl=taker_close_pnl,
        payoff_loser=payoff_loser, payoff_winner=payoff_winner,
        pnl=pnl, winner=winner,
        signal_correct=signal_correct,
    )


_GLOBAL_CFG: Optional[SignalLeanPolicy] = None


def _init_worker(cfg_dict: dict) -> None:
    global _GLOBAL_CFG
    _GLOBAL_CFG = SignalLeanPolicy(**cfg_dict)


def _process_one(path_str: str) -> Optional[dict]:
    cfg = _GLOBAL_CFG or SignalLeanPolicy()
    p = Path(path_str)
    wide = parse_market_csv(p)
    if wide is None:
        return None
    res = simulate_signal_lean(wide, cfg, p.stem, p.parent.parent.name)
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


def print_summary(df: pd.DataFrame, cfg: SignalLeanPolicy) -> None:
    n = len(df)
    days = df["date"].nunique() if "date" in df.columns else 1
    pnl = df["pnl"]

    print("\n" + "=" * 90)
    print("SIGNAL-LEAN STRATEGY SUMMARY")
    print("=" * 90)
    print(f"Markets: {n}    Days: {days}    "
          f"max_inv_winner: {cfg.max_inv_winner}    "
          f"init_max: ${cfg.initial_max_price}    "
          f"window: {cfg.trade_window_seconds}s    close: {cfg.close_at_seconds}s\n")

    print(f"=== HEADLINE ===")
    print(f"  Total P&L:        ${pnl.sum():>10.2f}")
    print(f"  Avg P&L / market: ${pnl.mean():>10.4f}")
    print(f"  Median P&L:       ${pnl.median():>10.4f}")
    print(f"  Per day:          ${pnl.sum()/days:>10.2f}")
    print(f"  Markets P&L>0:    {int((pnl>0).sum())}/{n} ({(pnl>0).mean():.1%})")
    print(f"  Days positive:    {int((df.groupby('date')['pnl'].sum()>0).sum())}/{days}")

    has_signal = df[df["signal_side"].isin(["UP", "DOWN"])]
    no_signal = df[df["signal_side"] == "NONE"]
    aborted = df[df["signal_side"] == "ABORT"]

    print(f"\n=== SIGNAL STATS ===")
    print(f"  Signal triggered:    {len(has_signal)} ({len(has_signal)/n:.1%})")
    print(f"  No fill in window:   {len(no_signal)} ({len(no_signal)/n:.1%})")
    if len(aborted):
        print(f"  Confirmation aborted:{len(aborted)} ({len(aborted)/n:.1%})")
    if len(has_signal):
        sig_correct = has_signal["signal_correct"].mean()
        print(f"  Signal accuracy:     {sig_correct:.1%}  (fraction where loser-bet side actually lost)")
        print(f"  Avg signal time:     {has_signal['signal_time_s'].mean():.2f}s")
        print(f"  Avg signal price:    ${has_signal['signal_price'].mean():.4f}")
        print(f"  Avg bidsum @ signal: ${has_signal['bidsum_at_signal'].mean():.4f}")
        print(f"  Avg winner_shares:   {has_signal['winner_shares'].mean():.2f}")
        print(f"  Avg winner price:    ${has_signal['avg_winner_price'].mean():.4f}")
        print(f"  Avg P&L / signal:    ${has_signal['pnl'].mean():.4f}")

    if len(has_signal):
        correct = has_signal[has_signal["signal_correct"]]
        wrong = has_signal[~has_signal["signal_correct"]]
        if len(correct):
            print(f"\n=== SIGNAL CORRECT ({len(correct)}, {len(correct)/len(has_signal):.1%}) ===")
            print(f"  Avg P&L:           ${correct['pnl'].mean():.4f}")
            print(f"  Avg winner_shares: {correct['winner_shares'].mean():.2f}")
        if len(wrong):
            print(f"\n=== SIGNAL WRONG ({len(wrong)}, {len(wrong)/len(has_signal):.1%}) ===")
            print(f"  Avg P&L:           ${wrong['pnl'].mean():.4f}")
            print(f"  Avg winner_shares: {wrong['winner_shares'].mean():.2f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-days", type=int, default=None)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--out", type=str, default="utils/mm_signal_lean.parquet")
    ap.add_argument("--bbo-offset", type=int, default=1)
    ap.add_argument("--size", type=float, default=1.0)
    ap.add_argument("--max-winner", type=float, default=10.0)
    ap.add_argument("--init-max", type=float, default=0.55)
    ap.add_argument("--pre-min", type=float, default=0.0)
    ap.add_argument("--bidsum-max", type=float, default=0.0)
    ap.add_argument("--confirm", type=float, default=0.0)
    ap.add_argument("--confirm-delta", type=float, default=0.0)
    ap.add_argument("--window", type=float, default=300.0)
    ap.add_argument("--close-at", type=float, default=0.0)
    ap.add_argument("--y-drop-abort", type=float, default=0.0)
    ap.add_argument("--stop-loss", action="store_true")
    args = ap.parse_args()

    cfg = SignalLeanPolicy(
        bbo_offset_ticks=args.bbo_offset, fill_size=args.size,
        max_inv_winner=args.max_winner,
        initial_max_price=args.init_max,
        pre_signal_min_seconds=args.pre_min,
        bidsum_max_at_signal=args.bidsum_max,
        confirmation_seconds=args.confirm,
        confirmation_delta=args.confirm_delta,
        trade_window_seconds=args.window,
        close_at_seconds=args.close_at,
        y_drop_abort=args.y_drop_abort,
        stop_loss_taker_close=args.stop_loss,
    )

    paths = _collect(PRICES_5M_ROOT, args.max_days)
    print(f"[signal_lean] paths: {len(paths)}", flush=True)

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
    print(f"\n[signal_lean] valid: {len(df)} ({time.time()-t0:.1f}s)")
    if df.empty:
        return 0

    if not args.no_save:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(args.out, compression="zstd")

    print_summary(df, cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
