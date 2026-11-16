"""
Sweep di parametri per la strategia ibrida 'skip-directional + tight-neutral'.

Itera (observe, trigger, window, offset) e stampa una tabella ordinata per P&L.
"""
from __future__ import annotations
import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from utils.mm_simulate_hybrid import HybridPolicy, simulate_hybrid
from utils.mm_analyze_5m import parse_market_csv, PRICES_5M_ROOT


def _collect(root: Path, max_days: int | None) -> list[str]:
    days = sorted([d for d in root.iterdir() if d.is_dir()])
    if max_days:
        days = days[-max_days:]
    paths = []
    for d in days:
        csv_dir = d / "csv"
        if csv_dir.exists():
            paths.extend(str(p) for p in sorted(csv_dir.glob("*.csv")))
    return paths


def _load_one(path_str: str):
    p = Path(path_str)
    wide = parse_market_csv(p)
    if wide is None:
        return None
    return path_str, wide, p.stem, p.parent.parent.name


def load_all_wides(paths: list[str], workers: int):
    wides: dict[str, tuple[pd.DataFrame, str, str]] = {}
    print(f"[sweep] loading {len(paths)} markets...", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_load_one, p) for p in paths]
        for n, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            if r:
                path_str, wide, stem, dt = r
                wides[path_str] = (wide, stem, dt)
            if n % 5000 == 0 or n == len(paths):
                print(f"  loaded {n}/{len(paths)} ({time.time()-t0:.1f}s)", flush=True)
    print(f"[sweep] loaded {len(wides)} valid wides ({time.time()-t0:.1f}s)\n", flush=True)
    return wides


def run_one_config(wides_cached, cfg: HybridPolicy) -> dict:
    rows = []
    for path_str, (wide, stem, dt) in wides_cached.items():
        res = simulate_hybrid(wide, cfg, stem, dt)
        if res is not None:
            rows.append(asdict(res))
    df = pd.DataFrame(rows)
    if df.empty:
        return {"n": 0}

    n = len(df)
    days = df["date"].nunique() if "date" in df.columns else 1
    neu = df[df["mode"] == "neutral"]

    return {
        "n": n,
        "days": days,
        "n_neutral": len(neu),
        "pair_cost_neu": neu["avg_pair_cost"].mean() if len(neu) else float("nan"),
        "pnl_per_market": df["pnl"].mean(),
        "pnl_total": df["pnl"].sum(),
        "pnl_per_day": df["pnl"].sum() / days,
        "neu_pnl_per_market": neu["pnl"].mean() if len(neu) else float("nan"),
        "neu_pnl_per_day": neu["pnl"].sum() / days if len(neu) else float("nan"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-days", type=int, default=7)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    args = ap.parse_args()

    paths = _collect(PRICES_5M_ROOT, args.max_days)
    wides = load_all_wides(paths, args.workers)

    # Sweep grid
    OBS = (3, 5, 10, 15)
    TRIG = (0.005, 0.010, 0.020)
    WIN = (10, 15, 20, 30, 60, 120, 300)
    OFF = (1, 2, 3)
    INV = (50, 100)

    rows = []
    n_configs = len(OBS) * len(TRIG) * len(WIN) * len(OFF) * len(INV)
    print(f"[sweep] running {n_configs} configurations...\n", flush=True)
    cfg_n = 0
    for obs in OBS:
        for trig in TRIG:
            for win in WIN:
                for off in OFF:
                    for inv in INV:
                        cfg_n += 1
                        cfg = HybridPolicy(
                            tick=0.01, fill_size=5.0,
                            bbo_offset_ticks=off,
                            observe_seconds=obs,
                            direction_trigger=trig,
                            max_inv_favored=0.0,
                            max_inv_loser=0.0,
                            max_inv_neutral=inv,
                            loser_mode="skip_market",
                            trade_window_seconds=win,
                            hedge_mode="hold",
                        )
                        r = run_one_config(wides, cfg)
                        rows.append({
                            "obs": obs, "trig": trig, "win": win, "off": off, "inv": inv,
                            **r,
                        })
                        if cfg_n % 20 == 0:
                            print(f"  {cfg_n}/{n_configs}", flush=True)

    df = pd.DataFrame(rows)
    df = df.sort_values("neu_pnl_per_day", ascending=False)
    pd.set_option("display.max_rows", 60)
    pd.set_option("display.width", 200)

    print("\n=== TOP 25 BY NEUTRAL P&L / DAY ===")
    print(df[["obs", "trig", "win", "off", "inv", "n_neutral",
              "pair_cost_neu", "neu_pnl_per_market", "neu_pnl_per_day",
              "pnl_per_day"]].head(25).to_string(index=False))

    print("\n=== TOP 15 BY OVERALL P&L / DAY (incl. directional skipped) ===")
    df2 = df.sort_values("pnl_per_day", ascending=False)
    print(df2[["obs", "trig", "win", "off", "inv", "n_neutral",
               "pair_cost_neu", "neu_pnl_per_market", "neu_pnl_per_day",
               "pnl_per_day"]].head(15).to_string(index=False))

    df.to_csv("utils/mm_sweep_hybrid.csv", index=False)
    print("\n[sweep] saved utils/mm_sweep_hybrid.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
