"""
Simulatore di market making incrementale su mercati BTC 5-min Polymarket.

Replay tick-by-tick di ogni mercato presente in `prices/5/days/<date>/csv/` con
una policy "Patient Incremental MM" parametrizzabile, e calcola il P&L
realistico tenendo conto di:

  - rebate maker (Crypto: 20% × 0.07 × p × (1-p) per share filled)
  - taker fees (se la policy chiude in taker un residuo a fine mercato)
  - payoff a expiry (1$ per ogni coppia detenuta + winner residuo)

Fill model (back-of-queue, conservativo):
  - buy maker @ price P è considerato fillato quando il book mostra
    `new_bid < P` (drop strict → il livello è stato mangiato e ci ha toccato).
  - sell maker @ price P è considerato fillato quando `new_ask > P`.

La policy default:
  - Piazza ordini buy (size = 10 share) sui due lati al best bid,
    se `bid_sum ≤ 0.99` e l'inventario sul lato è < cap.
  - Se il BBO si sposta sopra il nostro prezzo → cancella & riposizona (chase).
  - Se la window si chiude (`bid_sum > 0.99`) → cancella entrambi.
  - Negli ultimi `stop_buy_seconds` del mercato non si piazzano nuovi buy
    (si lascia che gli ordini residui filino o si cancellino naturalmente).
  - Negli ultimi `hedge_seconds`: gestione del residuale single-leg
    secondo `hedge_mode` ("hold", "taker_close", "maker_sell_aggressive").

Uso:
    python -m utils.mm_simulate_5m --max-days 1
    python -m utils.mm_simulate_5m            # tutti i 66 giorni
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


# ---------------------------------------------------------------------------
#  Policy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PolicyConfig:
    # Sizing
    order_size: float = 10.0
    max_inventory_per_side: float = 200.0

    # Quoting window
    place_when_bid_sum_le: float = 0.99
    chase_bbo_up: bool = True  # cancel & replace if BBO moves above our price
    bid_offset_ticks: int = 0  # 0=at BBO; N=N ticks below BBO (more patient)

    # Inventory balancing: non aggiungere al lato che è già più pieno dell'altro
    # di oltre `inventory_skew_max` share. 0 = disabilitato.
    inventory_skew_max: float = 0.0

    # Time gates (seconds remaining)
    stop_buy_seconds: float = 45.0   # ultimi N sec: smetti di piazzare nuovi buy
    hedge_seconds: float = 25.0      # ultimi N sec: hedge dei residui
    hedge_mode: str = "taker_close"  # "hold" | "taker_close" | "maker_sell_aggressive"

    # Fees & rebates (Polymarket Crypto)
    fee_rate: float = 0.07
    rebate_pct: float = 0.20  # for Crypto category

    @classmethod
    def from_kwargs(cls, **kw) -> "PolicyConfig":
        return cls(**kw)


# ---------------------------------------------------------------------------
#  Per-market simulation
# ---------------------------------------------------------------------------


@dataclass
class MarketResult:
    market_name: str
    date: str
    duration_s: float
    n_ticks: int

    n_buy_fills: int          # totale fill maker buy (UP + DOWN)
    n_buy_cancels: int
    up_buy_fills: int
    dn_buy_fills: int

    up_shares_final: float
    dn_shares_final: float
    pairs_final: float
    up_residual: float
    dn_residual: float

    avg_up_buy_price: float
    avg_dn_buy_price: float
    avg_pair_cost_maker: float    # avg_up + avg_dn maker-only (NaN se 0 fills)
    true_pair_cost: float         # cash_spent / pairs_final  (TUTTO incluso)
    n_taker_close_fills: int
    avg_taker_close_price: float

    cash_spent: float             # totale (maker buy + taker hedge)
    cash_spent_maker: float       # solo maker buy fills
    cash_received: float
    rebate_total: float
    taker_fees_paid: float

    payoff_pairs: float        # = pairs_final * 1.0
    payoff_residual: float     # winner residuo paga 1$/share, loser 0$
    pnl: float                 # payoff - cash_spent + cash_received + rebate - fees
    pnl_per_pair: float        # pnl / pairs_final  (NaN se pairs=0)

    capital_used_max: float    # max(up_shares*p + dn_shares*p) lungo il market (proxy)

    winner: str                # "UP" | "DOWN" | "TIE"
    final_up_mid: float
    final_dn_mid: float


def _rebate_per_share(price: float, cfg: PolicyConfig) -> float:
    """Rebate per share in USDC dato fill_price (Crypto category)."""
    return cfg.rebate_pct * cfg.fee_rate * price * (1.0 - price)


def _taker_fee_per_share(price: float, cfg: PolicyConfig) -> float:
    return cfg.fee_rate * price * (1.0 - price)


def simulate_market(
    wide: pd.DataFrame,
    cfg: PolicyConfig,
    market_name: str,
    date: str,
) -> Optional[MarketResult]:
    """Replay un singolo mercato con la policy data."""
    if wide is None or len(wide) < 10:
        return None

    # Forward-fill prezzi (gap nel logging).
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

    # Stato
    up_shares = 0.0
    dn_shares = 0.0
    up_buy_price: Optional[float] = None
    dn_buy_price: Optional[float] = None

    cash_spent = 0.0           # totale cash speso (maker buy + taker hedge)
    cash_spent_maker = 0.0     # solo maker buy fills
    cash_received = 0.0
    rebate_total = 0.0
    taker_fees_paid = 0.0

    n_up_buy_fills = 0
    n_dn_buy_fills = 0
    n_buy_cancels = 0
    sum_up_buy_px = 0.0
    sum_dn_buy_px = 0.0
    sum_up_buy_sz = 0.0
    sum_dn_buy_sz = 0.0

    n_taker_close_fills = 0
    sum_taker_close_px = 0.0
    sum_taker_close_sz = 0.0

    capital_used_max = 0.0

    EPS = 1e-9

    for i in range(n):
        # ------ 1) Resolve fills basandosi sul movimento del bid -----------
        if i > 0:
            # UP buy filled when up_bid scende sotto il nostro prezzo
            if up_buy_price is not None and up_bid[i] < up_buy_price - EPS:
                fill_size = min(
                    cfg.order_size, cfg.max_inventory_per_side - up_shares
                )
                if fill_size > 0:
                    fill_px = up_buy_price
                    cash_spent += fill_px * fill_size
                    cash_spent_maker += fill_px * fill_size
                    up_shares += fill_size
                    rebate_total += _rebate_per_share(fill_px, cfg) * fill_size
                    sum_up_buy_px += fill_px * fill_size
                    sum_up_buy_sz += fill_size
                    n_up_buy_fills += 1
                up_buy_price = None

            if dn_buy_price is not None and dn_bid[i] < dn_buy_price - EPS:
                fill_size = min(
                    cfg.order_size, cfg.max_inventory_per_side - dn_shares
                )
                if fill_size > 0:
                    fill_px = dn_buy_price
                    cash_spent += fill_px * fill_size
                    cash_spent_maker += fill_px * fill_size
                    dn_shares += fill_size
                    rebate_total += _rebate_per_share(fill_px, cfg) * fill_size
                    sum_dn_buy_px += fill_px * fill_size
                    sum_dn_buy_sz += fill_size
                    n_dn_buy_fills += 1
                dn_buy_price = None

        # ------ 2) Cancellazioni (chase BBO up + window closed) ------------
        bid_sum = up_bid[i] + dn_bid[i]

        # Chase BBO: se il nuovo best bid è SOPRA il nostro prezzo → cancella.
        if cfg.chase_bbo_up:
            if up_buy_price is not None and up_bid[i] > up_buy_price + EPS:
                up_buy_price = None
                n_buy_cancels += 1
            if dn_buy_price is not None and dn_bid[i] > dn_buy_price + EPS:
                dn_buy_price = None
                n_buy_cancels += 1

        # Window closed: bid_sum > soglia → cancella entrambi (no margin)
        if bid_sum > cfg.place_when_bid_sum_le + EPS:
            if up_buy_price is not None:
                up_buy_price = None
                n_buy_cancels += 1
            if dn_buy_price is not None:
                dn_buy_price = None
                n_buy_cancels += 1

        # ------ 3) Time gates ----------------------------------------------
        # Stop placing nuovi buy negli ultimi `stop_buy_seconds`
        stop_placing = t_left[i] <= cfg.stop_buy_seconds
        if stop_placing:
            # cancella ogni buy ancora attivo (smetti di accumulare)
            if up_buy_price is not None:
                up_buy_price = None
                n_buy_cancels += 1
            if dn_buy_price is not None:
                dn_buy_price = None
                n_buy_cancels += 1

        # Hedge dei residui single-leg negli ultimi `hedge_seconds`
        # (eseguito una volta sola, alla prima entrata in finestra).
        if (
            t_left[i] <= cfg.hedge_seconds
            and (i == 0 or t_left[i - 1] > cfg.hedge_seconds)
        ):
            pairs_now = min(up_shares, dn_shares)
            up_res = up_shares - pairs_now
            dn_res = dn_shares - pairs_now

            if cfg.hedge_mode == "taker_close":
                # Compra il lato mancante al taker per chiudere coppia.
                if up_res > 0:
                    need = up_res
                    px = dn_ask[i]
                    if not np.isnan(px) and px > 0:
                        cash_spent += px * need
                        taker_fees_paid += _taker_fee_per_share(px, cfg) * need
                        dn_shares += need
                        n_taker_close_fills += 1
                        sum_taker_close_px += px * need
                        sum_taker_close_sz += need
                if dn_res > 0:
                    need = dn_res
                    px = up_ask[i]
                    if not np.isnan(px) and px > 0:
                        cash_spent += px * need
                        taker_fees_paid += _taker_fee_per_share(px, cfg) * need
                        up_shares += need
                        n_taker_close_fills += 1
                        sum_taker_close_px += px * need
                        sum_taker_close_sz += need

            elif cfg.hedge_mode == "maker_sell_aggressive":
                # Vendi il lato in eccesso al BBO ask come maker.
                # Modello: se entro la fine vediamo l'ask scendere *strettamente*
                # sotto il nostro sell price (improbabile ma possibile), filla.
                # Per semplicità non simuliamo l'ordine maker sell qui (lo si può
                # modellare in v2). Default: tieni a expiry → "hold".
                pass  # equivale a hold

            else:  # "hold"
                pass

        # ------ 4) Place new orders se non attivi e bid_sum ok -------------
        if not stop_placing and bid_sum <= cfg.place_when_bid_sum_le + EPS:
            # Inventory skew check: non aggiungere al lato già "in vantaggio"
            # se la differenza supera inventory_skew_max.
            up_skew_block = (
                cfg.inventory_skew_max > 0
                and (up_shares - dn_shares) >= cfg.inventory_skew_max - EPS
            )
            dn_skew_block = (
                cfg.inventory_skew_max > 0
                and (dn_shares - up_shares) >= cfg.inventory_skew_max - EPS
            )
            offset = 0.01 * cfg.bid_offset_ticks
            if (
                up_buy_price is None
                and up_shares < cfg.max_inventory_per_side
                and not up_skew_block
            ):
                up_buy_price = float(up_bid[i] - offset)
            if (
                dn_buy_price is None
                and dn_shares < cfg.max_inventory_per_side
                and not dn_skew_block
            ):
                dn_buy_price = float(dn_bid[i] - offset)

        # ------ 5) Track capital used --------------------------------------
        # Stima approssimativa del capitale impegnato in shares acquistate.
        cap_now = (
            up_shares * (up_bid[i] + up_ask[i]) * 0.5
            + dn_shares * (dn_bid[i] + dn_ask[i]) * 0.5
        )
        if cap_now > capital_used_max:
            capital_used_max = cap_now

    # --------- End of market: compute payoff ------------------------------
    pairs_final = min(up_shares, dn_shares)
    up_residual = up_shares - pairs_final
    dn_residual = dn_shares - pairs_final

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
        payoff_residual = up_residual * 1.0
    elif winner == "DOWN":
        payoff_residual = dn_residual * 1.0
    else:
        payoff_residual = 0.5 * (up_residual + dn_residual)

    payoff_total = payoff_pairs + payoff_residual
    pnl = payoff_total - cash_spent + cash_received + rebate_total - taker_fees_paid

    avg_up_buy_price = (
        sum_up_buy_px / sum_up_buy_sz if sum_up_buy_sz > 0 else float("nan")
    )
    avg_dn_buy_price = (
        sum_dn_buy_px / sum_dn_buy_sz if sum_dn_buy_sz > 0 else float("nan")
    )
    if pairs_final > 0:
        if not (np.isnan(avg_up_buy_price) or np.isnan(avg_dn_buy_price)):
            avg_pair_cost_maker = avg_up_buy_price + avg_dn_buy_price
        else:
            avg_pair_cost_maker = float("nan")
        true_pair_cost = cash_spent / pairs_final
        pnl_per_pair = pnl / pairs_final
    else:
        avg_pair_cost_maker = float("nan")
        true_pair_cost = float("nan")
        pnl_per_pair = float("nan")
    avg_taker_close_price = (
        sum_taker_close_px / sum_taker_close_sz
        if sum_taker_close_sz > 0
        else float("nan")
    )

    duration_s = float((t_ms[-1] - t_ms[0]) / 1000.0)

    return MarketResult(
        market_name=market_name,
        date=date,
        duration_s=duration_s,
        n_ticks=n,
        n_buy_fills=n_up_buy_fills + n_dn_buy_fills,
        n_buy_cancels=n_buy_cancels,
        up_buy_fills=n_up_buy_fills,
        dn_buy_fills=n_dn_buy_fills,
        up_shares_final=up_shares,
        dn_shares_final=dn_shares,
        pairs_final=pairs_final,
        up_residual=up_residual,
        dn_residual=dn_residual,
        avg_up_buy_price=avg_up_buy_price,
        avg_dn_buy_price=avg_dn_buy_price,
        avg_pair_cost_maker=avg_pair_cost_maker,
        true_pair_cost=true_pair_cost,
        n_taker_close_fills=n_taker_close_fills,
        avg_taker_close_price=avg_taker_close_price,
        cash_spent=cash_spent,
        cash_spent_maker=cash_spent_maker,
        cash_received=cash_received,
        rebate_total=rebate_total,
        taker_fees_paid=taker_fees_paid,
        payoff_pairs=payoff_pairs,
        payoff_residual=payoff_residual,
        pnl=pnl,
        pnl_per_pair=pnl_per_pair,
        capital_used_max=capital_used_max,
        winner=winner,
        final_up_mid=final_up_mid,
        final_dn_mid=final_dn_mid,
    )


# ---------------------------------------------------------------------------
#  Driver
# ---------------------------------------------------------------------------


_GLOBAL_CFG: Optional[PolicyConfig] = None


def _init_worker(cfg_dict: dict) -> None:
    global _GLOBAL_CFG
    _GLOBAL_CFG = PolicyConfig(**cfg_dict)


def _process_one_csv(path_str: str) -> Optional[dict]:
    global _GLOBAL_CFG
    cfg = _GLOBAL_CFG
    if cfg is None:
        cfg = PolicyConfig()
    path = Path(path_str)
    wide = parse_market_csv(path)
    if wide is None:
        return None
    res = simulate_market(wide, cfg, path.stem, path.parent.parent.name)
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-days", type=int, default=None)
    ap.add_argument("--days", nargs="*", default=None)
    ap.add_argument("--max-markets", type=int, default=None)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--out", type=str, default="utils/mm_simulation_5m.parquet")
    ap.add_argument("--no-save", action="store_true")
    # Policy parameters
    ap.add_argument("--order-size", type=float, default=10.0)
    ap.add_argument("--max-inv", type=float, default=200.0)
    ap.add_argument("--bid-sum-le", type=float, default=0.99)
    ap.add_argument("--no-chase", action="store_true",
                    help="Disable BBO chase (default: chase ON).")
    ap.add_argument("--bid-offset-ticks", type=int, default=0,
                    help="Place buy orders N ticks (1c) BELOW BBO (default: 0=at BBO).")
    ap.add_argument("--inv-skew-max", type=float, default=0.0,
                    help="Max share imbalance up-vs-down before blocking buys "
                         "on the heavier side. 0 = disabled.")
    ap.add_argument("--stop-buy-seconds", type=float, default=45.0)
    ap.add_argument("--hedge-seconds", type=float, default=25.0)
    ap.add_argument("--hedge-mode", choices=("hold", "taker_close",
                                              "maker_sell_aggressive"),
                    default="taker_close")
    args = ap.parse_args()

    cfg = PolicyConfig(
        order_size=args.order_size,
        max_inventory_per_side=args.max_inv,
        place_when_bid_sum_le=args.bid_sum_le,
        chase_bbo_up=not args.no_chase,
        bid_offset_ticks=args.bid_offset_ticks,
        inventory_skew_max=args.inv_skew_max,
        stop_buy_seconds=args.stop_buy_seconds,
        hedge_seconds=args.hedge_seconds,
        hedge_mode=args.hedge_mode,
    )

    paths = _collect_paths(PRICES_5M_ROOT, args.max_days, args.days)
    if args.max_markets is not None:
        paths = paths[: args.max_markets]
    print(f"[mm_simulate_5m] paths to simulate: {len(paths)}", flush=True)
    print(f"[mm_simulate_5m] policy: {cfg}", flush=True)
    if not paths:
        print("No CSVs found.", file=sys.stderr)
        return 1

    t0 = time.time()
    results: list[dict] = []
    n_done = 0
    cfg_dict = asdict(cfg)
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
                elapsed = time.time() - t0
                rate = n_done / elapsed if elapsed > 0 else 0.0
                eta = (len(paths) - n_done) / rate if rate > 0 else 0.0
                print(
                    f"  {n_done}/{len(paths)}  valid={len(results)}  "
                    f"rate={rate:.1f}/s  eta={eta:.0f}s",
                    flush=True,
                )

    df = pd.DataFrame(results)
    print(f"\n[mm_simulate_5m] valid markets: {len(df)} (took {time.time()-t0:.1f}s)")
    if df.empty:
        return 0

    if not args.no_save:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path, compression="zstd")
        print(f"[mm_simulate_5m] saved to {out_path}")

    print_summary(df, cfg)
    return 0


def print_summary(df: pd.DataFrame, cfg: PolicyConfig) -> None:
    print("\n" + "=" * 78)
    print("SIMULATION SUMMARY")
    print("=" * 78)
    print(
        f"Markets simulated: {len(df):>6}    "
        f"Days: {df['date'].nunique() if 'date' in df.columns else '-'}"
    )
    print(f"Policy: order_size={cfg.order_size}  max_inv={cfg.max_inventory_per_side}  "
          f"chase={cfg.chase_bbo_up}  hedge={cfg.hedge_mode}")
    print()

    # Headline metrics
    pnl = df["pnl"]
    n_pos = int((pnl > 0).sum())
    n_neg = int((pnl < 0).sum())
    n_zer = int((pnl == 0).sum())
    pct_pos = n_pos / len(df) if len(df) else 0.0
    n_total = len(df)

    print("=== HEADLINE ===")
    print(f"  Total P&L:          ${pnl.sum():>10.2f}")
    print(f"  Avg P&L / market:   ${pnl.mean():>10.4f}")
    print(f"  Median P&L:         ${pnl.median():>10.4f}")
    print(f"  Stddev P&L:         ${pnl.std():>10.4f}")
    print(f"  P&L > 0:            {n_pos} ({pct_pos:.1%})")
    print(f"  P&L < 0:            {n_neg} ({n_neg/n_total:.1%})")
    print(f"  P&L = 0 (no fills): {n_zer} ({n_zer/n_total:.1%})")
    print()

    # Distribution
    print("=== P&L DISTRIBUTION ($/market) ===")
    q = pnl.quantile([0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).values
    print(
        f"  p01={q[0]:.4f}  p05={q[1]:.4f}  p10={q[2]:.4f}  p25={q[3]:.4f}  "
        f"p50={q[4]:.4f}  p75={q[5]:.4f}  p90={q[6]:.4f}  p95={q[7]:.4f}  p99={q[8]:.4f}"
    )
    print(f"  min={pnl.min():.4f}  max={pnl.max():.4f}")
    print()

    # Fills
    print("=== FILL STATS ===")
    print(f"  Avg buy fills/market:   {df['n_buy_fills'].mean():.2f}  "
          f"(UP {df['up_buy_fills'].mean():.2f} | DN {df['dn_buy_fills'].mean():.2f})")
    print(f"  Avg cancellations:      {df['n_buy_cancels'].mean():.2f}")
    print(f"  Avg pairs accumulated:  {df['pairs_final'].mean():.2f}")
    print(f"  Avg residual UP:        {df['up_residual'].mean():.2f}")
    print(f"  Avg residual DOWN:      {df['dn_residual'].mean():.2f}")
    print()

    # Rebate vs fees
    print("=== ECONOMICS ===")
    print(f"  Avg cash spent (tot):   ${df['cash_spent'].mean():.4f}")
    print(f"  Avg cash spent maker:   ${df['cash_spent_maker'].mean():.4f}")
    print(f"  Avg payoff (pairs):     ${df['payoff_pairs'].mean():.4f}")
    print(f"  Avg payoff (residual):  ${df['payoff_residual'].mean():.4f}")
    print(f"  Avg rebate:             ${df['rebate_total'].mean():.6f}")
    print(f"  Avg taker fees:         ${df['taker_fees_paid'].mean():.6f}")
    print(f"  Avg pnl_per_pair:       ${df['pnl_per_pair'].mean():.6f}  "
          f"(median {df['pnl_per_pair'].median():.6f})")
    print(f"  Avg pair_cost MAKER:    ${df['avg_pair_cost_maker'].mean():.6f}")
    print(f"  Avg pair_cost TRUE:     ${df['true_pair_cost'].mean():.6f}  "
          f"(includes taker hedge)")
    print(f"  Avg taker close fills:  {df['n_taker_close_fills'].mean():.2f}")
    if df['n_taker_close_fills'].sum() > 0:
        print(f"  Avg taker close price:  ${df['avg_taker_close_price'].mean():.4f}")
    print()

    # Capital
    print("=== CAPITAL ===")
    cap = df["capital_used_max"]
    print(f"  Avg capital_used_max:   ${cap.mean():.2f}")
    q = cap.quantile([0.50, 0.75, 0.90, 0.99]).values
    print(f"  Capital p50={q[0]:.2f}  p75={q[1]:.2f}  p90={q[2]:.2f}  p99={q[3]:.2f}")
    print()

    # Daily aggregate
    if "date" in df.columns:
        daily = df.groupby("date")["pnl"].agg(["sum", "count", "mean"])
        print("=== DAILY P&L ===")
        print(f"  Days:           {len(daily)}")
        print(f"  Avg daily pnl:  ${daily['sum'].mean():.4f}")
        print(f"  Median daily:   ${daily['sum'].median():.4f}")
        print(f"  Best day:       ${daily['sum'].max():.4f}")
        print(f"  Worst day:      ${daily['sum'].min():.4f}")
        print(f"  Days positive:  {int((daily['sum']>0).sum())}/{len(daily)}")


if __name__ == "__main__":
    raise SystemExit(main())
