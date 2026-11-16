"""
Market-Making analysis for 5-minute BTC Polymarket markets.

Legge i CSV in `prices/5/days/<date>/csv/` e calcola, per ciascun mercato e
poi in aggregato, le metriche strutturali necessarie per valutare la
fattibilità di una strategia di market making che accumula coppie
(buy UP maker + buy DOWN maker) hold-to-expiry per catturare il rebate
+ eventuale spread.

Output:
- DataFrame parquet con una riga per mercato (`utils/mm_analysis_5m.parquet`)
- Stampa stats aggregate principali

Uso:
    python -m utils.mm_analyze_5m --max-days 1 --workers 4
    python -m utils.mm_analyze_5m   # full
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

PRICES_5M_ROOT = Path(__file__).resolve().parent.parent / "prices" / "5" / "days"

# Cap dt fra tick consecutivi: oltre questa soglia consideriamo un buco
# nel logging (non un periodo reale di mercato fermo).
MAX_DT_MS = 5_000

# Soglia minima di durata "effettiva" sotto la quale skippiamo il mercato.
MIN_DURATION_S = 60.0


# ---------------------------------------------------------------------------
#  Parsing
# ---------------------------------------------------------------------------


def parse_market_csv(path: Path) -> Optional[pd.DataFrame]:
    """Legge un CSV e ritorna un DataFrame "wide" con colonne:
        t_ms, t_left, up_bid, up_ask, dn_bid, dn_ask
    Una riga per snapshot del book (UP + DOWN insieme).
    """
    try:
        df = pd.read_csv(
            path,
            usecols=[
                "timestamp_unix_ms",
                "outcome",
                "ask",
                "bid",
                "time_left_seconds",
            ],
        )
    except Exception:
        return None

    if df.empty:
        return None

    df["outcome"] = df["outcome"].astype("category")
    df = df.sort_values("timestamp_unix_ms")

    up = df[df["outcome"] == "Up"][
        ["timestamp_unix_ms", "time_left_seconds", "bid", "ask"]
    ].rename(columns={"bid": "up_bid", "ask": "up_ask"})
    dn = df[df["outcome"] == "Down"][
        ["timestamp_unix_ms", "time_left_seconds", "bid", "ask"]
    ].rename(columns={"bid": "dn_bid", "ask": "dn_ask"})

    if up.empty or dn.empty:
        return None

    wide = pd.merge(
        up,
        dn[["timestamp_unix_ms", "dn_bid", "dn_ask"]],
        on="timestamp_unix_ms",
        how="inner",
    )
    if wide.empty:
        return None

    wide = wide.rename(
        columns={"timestamp_unix_ms": "t_ms", "time_left_seconds": "t_left"}
    )
    wide = wide.sort_values("t_ms").reset_index(drop=True)
    return wide


# ---------------------------------------------------------------------------
#  Per-market metrics
# ---------------------------------------------------------------------------


def market_metrics(wide: pd.DataFrame, market_name: str) -> Optional[dict]:
    if wide is None or len(wide) < 10:
        return None

    # Forward-fill prezzi mancanti (succede ai bordi, near 0 o 1).
    wide = wide.copy()
    for col in ("up_bid", "up_ask", "dn_bid", "dn_ask"):
        wide[col] = wide[col].ffill()
    wide = wide.dropna(subset=["up_bid", "up_ask", "dn_bid", "dn_ask"]).reset_index(
        drop=True
    )
    if len(wide) < 10:
        return None

    # dt fra tick consecutivi, capato a MAX_DT_MS per gestire buchi.
    dt = wide["t_ms"].diff().fillna(0).clip(lower=0, upper=MAX_DT_MS).to_numpy()
    total_ms = float(dt.sum())
    if total_ms < MIN_DURATION_S * 1000:
        return None

    up_bid = wide["up_bid"].to_numpy(dtype=np.float32)
    up_ask = wide["up_ask"].to_numpy(dtype=np.float32)
    dn_bid = wide["dn_bid"].to_numpy(dtype=np.float32)
    dn_ask = wide["dn_ask"].to_numpy(dtype=np.float32)

    up_spread = up_ask - up_bid
    dn_spread = dn_ask - dn_bid
    bid_sum = up_bid + dn_bid
    ask_sum = up_ask + dn_ask
    up_mid = (up_bid + up_ask) * 0.5
    dn_mid = (dn_bid + dn_ask) * 0.5

    # Time-weighted helper.
    def twpct(mask: np.ndarray) -> float:
        return float(dt[mask].sum() / total_ms) if total_ms > 0 else 0.0

    def twavg(arr: np.ndarray) -> float:
        return float((arr * dt).sum() / total_ms) if total_ms > 0 else 0.0

    # Distribuzioni time-weighted dei valori chiave.
    pct_bid_sum_le_99 = twpct(bid_sum <= 0.990 + 1e-6)
    pct_bid_sum_le_98 = twpct(bid_sum <= 0.980 + 1e-6)
    pct_bid_sum_le_97 = twpct(bid_sum <= 0.970 + 1e-6)
    pct_bid_sum_le_95 = twpct(bid_sum <= 0.950 + 1e-6)
    pct_bid_sum_eq_100 = twpct((bid_sum > 0.999) & (bid_sum < 1.001))
    pct_bid_sum_gt_100 = twpct(bid_sum > 1.001)  # arbitraggio (raro)

    pct_up_spread_1c = twpct(up_spread <= 0.011)
    pct_up_spread_2c = twpct((up_spread > 0.011) & (up_spread <= 0.021))
    pct_up_spread_ge3c = twpct(up_spread >= 0.029)

    pct_dn_spread_1c = twpct(dn_spread <= 0.011)
    pct_dn_spread_2c = twpct((dn_spread > 0.011) & (dn_spread <= 0.021))
    pct_dn_spread_ge3c = twpct(dn_spread >= 0.029)

    avg_up_spread = twavg(up_spread)
    avg_dn_spread = twavg(dn_spread)
    avg_bid_sum = twavg(bid_sum)
    avg_ask_sum = twavg(ask_sum)
    avg_up_mid = twavg(up_mid)
    avg_dn_mid = twavg(dn_mid)

    # Conta dei "movimenti di book" (proxy per attività e fill events).
    up_bid_diff = np.diff(up_bid)
    dn_bid_diff = np.diff(dn_bid)
    up_ask_diff = np.diff(up_ask)
    dn_ask_diff = np.diff(dn_ask)

    n_up_bid_drop = int(np.sum(up_bid_diff < -1e-6))
    n_up_bid_rise = int(np.sum(up_bid_diff > 1e-6))
    n_dn_bid_drop = int(np.sum(dn_bid_diff < -1e-6))
    n_dn_bid_rise = int(np.sum(dn_bid_diff > 1e-6))
    n_up_ask_drop = int(np.sum(up_ask_diff < -1e-6))
    n_up_ask_rise = int(np.sum(up_ask_diff > 1e-6))
    n_dn_ask_drop = int(np.sum(dn_ask_diff < -1e-6))
    n_dn_ask_rise = int(np.sum(dn_ask_diff > 1e-6))

    # Ampiezza media dei movimenti (in tick = cents).
    avg_up_bid_drop = (
        float(-up_bid_diff[up_bid_diff < -1e-6].mean()) if n_up_bid_drop else 0.0
    )
    avg_dn_bid_drop = (
        float(-dn_bid_diff[dn_bid_diff < -1e-6].mean()) if n_dn_bid_drop else 0.0
    )

    # Inferenza winner: usa lo stato finale.
    last = wide.iloc[-1]
    if last["up_bid"] > last["dn_bid"]:
        winner = "UP"
    elif last["dn_bid"] > last["up_bid"]:
        winner = "DOWN"
    else:
        winner = "TIE"

    # Volatilità del mid (proxy per BTC volatility nel periodo).
    up_mid_std = float(np.std(up_mid))
    up_mid_range = float(up_mid.max() - up_mid.min())

    return {
        "market_name": market_name,
        "n_ticks": int(len(wide)),
        "duration_seconds": total_ms / 1000.0,
        "first_t_left": float(wide.iloc[0]["t_left"]),
        "last_t_left": float(last["t_left"]),
        "winner": winner,
        "up_mid_final": float((last["up_bid"] + last["up_ask"]) / 2),
        "dn_mid_final": float((last["dn_bid"] + last["dn_ask"]) / 2),
        # spreads
        "avg_up_spread": avg_up_spread,
        "avg_dn_spread": avg_dn_spread,
        "pct_up_spread_1c": pct_up_spread_1c,
        "pct_up_spread_2c": pct_up_spread_2c,
        "pct_up_spread_ge3c": pct_up_spread_ge3c,
        "pct_dn_spread_1c": pct_dn_spread_1c,
        "pct_dn_spread_2c": pct_dn_spread_2c,
        "pct_dn_spread_ge3c": pct_dn_spread_ge3c,
        # bid_sum (fattibilità window)
        "avg_bid_sum": avg_bid_sum,
        "avg_ask_sum": avg_ask_sum,
        "pct_bid_sum_le_99": pct_bid_sum_le_99,
        "pct_bid_sum_le_98": pct_bid_sum_le_98,
        "pct_bid_sum_le_97": pct_bid_sum_le_97,
        "pct_bid_sum_le_95": pct_bid_sum_le_95,
        "pct_bid_sum_eq_100": pct_bid_sum_eq_100,
        "pct_bid_sum_gt_100": pct_bid_sum_gt_100,
        # mid stats
        "avg_up_mid": avg_up_mid,
        "avg_dn_mid": avg_dn_mid,
        "up_mid_std": up_mid_std,
        "up_mid_range": up_mid_range,
        # book activity (proxy per fill rate)
        "n_up_bid_drop": n_up_bid_drop,
        "n_up_bid_rise": n_up_bid_rise,
        "n_dn_bid_drop": n_dn_bid_drop,
        "n_dn_bid_rise": n_dn_bid_rise,
        "n_up_ask_drop": n_up_ask_drop,
        "n_up_ask_rise": n_up_ask_rise,
        "n_dn_ask_drop": n_dn_ask_drop,
        "n_dn_ask_rise": n_dn_ask_rise,
        "avg_up_bid_drop_size": avg_up_bid_drop,
        "avg_dn_bid_drop_size": avg_dn_bid_drop,
    }


# ---------------------------------------------------------------------------
#  Driver
# ---------------------------------------------------------------------------


def process_one_csv(path_str: str) -> Optional[dict]:
    path = Path(path_str)
    wide = parse_market_csv(path)
    if wide is None:
        return None
    out = market_metrics(wide, path.stem)
    if out is None:
        return None
    out["date"] = path.parent.parent.name  # YYYY-MM-DD
    return out


def collect_csv_paths(
    root: Path, max_days: Optional[int] = None, days: Optional[list[str]] = None
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
        for p in sorted(csv_dir.glob("*.csv")):
            paths.append(str(p))
    return paths


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-days", type=int, default=None)
    ap.add_argument("--days", nargs="*", default=None,
                    help="Specific YYYY-MM-DD folders (overrides --max-days)")
    ap.add_argument("--max-markets", type=int, default=None)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--out", type=str, default="utils/mm_analysis_5m.parquet")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    paths = collect_csv_paths(PRICES_5M_ROOT, args.max_days, args.days)
    if args.max_markets is not None:
        paths = paths[: args.max_markets]
    print(f"[mm_analyze_5m] CSV files to process: {len(paths)}", flush=True)
    if not paths:
        print("No CSVs found.", file=sys.stderr)
        return 1

    t0 = time.time()
    results: list[dict] = []
    n_done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(process_one_csv, p) for p in paths]
        for fut in as_completed(futures):
            n_done += 1
            r = fut.result()
            if r is not None:
                results.append(r)
            if n_done % 500 == 0 or n_done == len(paths):
                elapsed = time.time() - t0
                rate = n_done / elapsed if elapsed > 0 else 0.0
                eta = (len(paths) - n_done) / rate if rate > 0 else 0.0
                print(
                    f"  {n_done}/{len(paths)}  valid={len(results)}  "
                    f"rate={rate:.1f}/s  eta={eta:.0f}s",
                    flush=True,
                )

    df = pd.DataFrame(results)
    print(f"\n[mm_analyze_5m] valid markets: {len(df)} (took {time.time()-t0:.1f}s)")
    if df.empty:
        return 0

    if not args.no_save:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path, compression="zstd")
        print(f"[mm_analyze_5m] saved to {out_path}")

    print_summary(df)
    return 0


def print_summary(df: pd.DataFrame) -> None:
    print("\n" + "=" * 78)
    print("AGGREGATE SUMMARY")
    print("=" * 78)
    print(
        f"Markets analysed: {len(df):>6}   "
        f"Days covered: {df['date'].nunique() if 'date' in df.columns else '-'}"
    )
    if "winner" in df.columns:
        win = df["winner"].value_counts()
        n_total = max(1, len(df))
        wu = int(win.get("UP", 0))
        wd = int(win.get("DOWN", 0))
        print(
            f"Winner balance:  UP={wu} ({wu/n_total:.2%})  "
            f"DOWN={wd} ({wd/n_total:.2%})  "
            f"TIE={int(win.get('TIE', 0))}"
        )
    print()

    def _row(label: str, series: pd.Series, fmt: str = "{:>7.4f}") -> None:
        q = series.quantile([0.10, 0.25, 0.50, 0.75, 0.90]).values
        line = (
            f"  {label:<28} "
            f"mean={fmt.format(series.mean()):<10}"
            f"p10={fmt.format(q[0]):<10}"
            f"p25={fmt.format(q[1]):<10}"
            f"p50={fmt.format(q[2]):<10}"
            f"p75={fmt.format(q[3]):<10}"
            f"p90={fmt.format(q[4]):<10}"
        )
        print(line)

    print("Spread averages (per share, in $):")
    _row("avg_up_spread", df["avg_up_spread"])
    _row("avg_dn_spread", df["avg_dn_spread"])
    _row("avg_bid_sum",  df["avg_bid_sum"])
    _row("avg_ask_sum",  df["avg_ask_sum"])
    print()
    print("Time-weighted % of market in each spread regime (single side):")
    _row("pct_up_spread_1c",   df["pct_up_spread_1c"])
    _row("pct_up_spread_2c",   df["pct_up_spread_2c"])
    _row("pct_up_spread_ge3c", df["pct_up_spread_ge3c"])
    print()
    print("Time-weighted % of market with bid_sum at each level:")
    _row("pct_bid_sum_le_99",  df["pct_bid_sum_le_99"])
    _row("pct_bid_sum_le_98",  df["pct_bid_sum_le_98"])
    _row("pct_bid_sum_le_97",  df["pct_bid_sum_le_97"])
    _row("pct_bid_sum_le_95",  df["pct_bid_sum_le_95"])
    _row("pct_bid_sum_eq_100", df["pct_bid_sum_eq_100"])
    _row("pct_bid_sum_gt_100", df["pct_bid_sum_gt_100"])
    print()
    print("Book activity (counts per market):")
    for c in ("n_up_bid_drop", "n_up_bid_rise", "n_dn_bid_drop", "n_dn_bid_rise"):
        _row(c, df[c], fmt="{:>7.1f}")
    print()
    print("Avg drop size when bid drops (in $):")
    _row("avg_up_bid_drop_size", df["avg_up_bid_drop_size"])
    _row("avg_dn_bid_drop_size", df["avg_dn_bid_drop_size"])
    print()
    print("Volatility / mid range:")
    _row("up_mid_std",   df["up_mid_std"])
    _row("up_mid_range", df["up_mid_range"])


if __name__ == "__main__":
    raise SystemExit(main())
