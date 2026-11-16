#!/usr/bin/env python3
"""Backtest entry/TP levels on historical BTC 5-min up/down markets.

For every (entry, tp) pair on a configurable grid, simulates bot1's
strategy across all historical CSVs and reports win-rate, expected P&L,
and annualised projections.

Usage:
    python utils/backtest_levels.py [--days N] [--no-sl]     # grid search completo
    python utils/backtest_levels.py --top3-presets           # tabella metriche top 3
    python utils/backtest_levels.py --top4-presets          # tabella metriche top 4
    python utils/backtest_levels.py --mega-grid                  # triple post-only su tutto il corpus
    python utils/backtest_levels.py --mega-grid --days 3       # ultimi 3 giorni
    python utils/backtest_levels.py --with-fees                # griglia + fee (entry taker, TP maker, SL taker)
    python utils/backtest_levels.py --top4-presets --with-fees # preset con PnL netto
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
import time as _time
from collections import defaultdict
from dataclasses import dataclass, field
from multiprocessing import Pool, cpu_count
from typing import Optional

import numpy as np

try:
    from numba import njit

    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False

    def njit(*_args, **_kwargs):
        def _dec(f):
            return f

        return _dec


# ── Configuration ─────────────────────────────────────────────────────────────

PRICES_DIR = os.path.join(os.path.dirname(__file__), "..", "prices", "5", "days")

ENTRY_MIN = 0.55
ENTRY_MAX = 0.92
ENTRY_STEP = 0.01

TP_OFFSET_MIN = 0.03
TP_MAX = 0.99
TP_STEP = 0.01

SL_OFFSET = 0.15          # SL = entry - SL_OFFSET (set very large to disable)
MARKETS_PER_DAY = 288      # 5-min markets in 24 h

# Polymarket fee model (per share, same structure as shares * rate * p*(1-p)):
#   Taker:  rate * p * (1-p)
#   Maker rebate: rate * p * (1-p) * maker_rebate_frac
FEE_RATE = 0.07
MAKER_REBATE_FRAC = 0.20

# Top-4 (entry, TP) from full-grid ranking on the full CSV corpus (SL = entry - SL_OFFSET).
PRESET_TOP4_COMBOS: list[tuple[float, float]] = [
    (0.59, 0.98),
    (0.59, 0.97),
    (0.60, 0.98),
    (0.60, 0.97),
]


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class MarketTicks:
    """Pre-processed tick arrays for one market (one CSV)."""
    name: str
    # Parallel arrays, sorted by time.  Index i is one snapshot.
    ts: np.ndarray           # float64 – unix-ms timestamps
    up_ask: np.ndarray       # float64 – Up best-ask at each tick
    down_ask: np.ndarray     # float64 – Down best-ask at each tick
    up_bid: np.ndarray       # float64 – Up best-bid at each tick
    down_bid: np.ndarray     # float64 – Down best-bid at each tick


@dataclass
class PairResult:
    entry: float
    tp: float
    sl: float
    n_markets: int = 0       # total markets analysed
    n_entered: int = 0       # markets where entry triggered
    n_tp: int = 0            # exited via TP
    n_sl: int = 0            # exited via SL
    n_resolved_win: int = 0  # held to expiry, resolved as win (ask >= 0.90)
    n_resolved_loss: int = 0 # held to expiry, resolved as loss
    total_pnl: float = 0.0   # sum of per-trade P&L (gross or net if fees enabled)
    total_pnl_gross: float = 0.0  # before fees (only accumulated when fees on)


@dataclass
class EquityStats:
    """Per-combo path metrics (same trade order as ``run_equity_analysis``)."""
    n_trades: int = 0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    max_dd_trade_idx: int = 0
    longest_losing_streak_trades: int = 0
    longest_losing_streak_pnl: float = 0.0
    worst_streak_trades: int = 0
    worst_streak_pnl: float = 0.0
    positive_day_windows: int = 0
    total_day_windows: int = 0


# ── CSV loading ───────────────────────────────────────────────────────────────

def load_market(csv_path: str) -> Optional[MarketTicks]:
    """Load a single market CSV into numpy arrays (ask + bid)."""
    up_ask: dict[float, float] = {}
    down_ask: dict[float, float] = {}
    up_bid: dict[float, float] = {}
    down_bid: dict[float, float] = {}
    name = ""

    try:
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                t = float(row["timestamp_unix_ms"])
                outcome = row["outcome"]
                ask_val = row.get("ask", "")
                bid_val = row.get("bid", "")
                if not ask_val:
                    continue
                ask = float(ask_val)
                bid = float(bid_val) if bid_val else np.nan
                if not name:
                    name = row.get("market_name", csv_path)
                if outcome == "Up":
                    up_ask[t] = ask
                    up_bid[t] = bid
                elif outcome == "Down":
                    down_ask[t] = ask
                    down_bid[t] = bid
    except Exception:
        return None

    all_ts = sorted(set(up_ask.keys()) | set(down_ask.keys()))
    if len(all_ts) < 5:
        return None

    n = len(all_ts)
    ts_arr = np.array(all_ts, dtype=np.float64)
    up_a = np.full(n, np.nan, dtype=np.float64)
    dn_a = np.full(n, np.nan, dtype=np.float64)
    up_b = np.full(n, np.nan, dtype=np.float64)
    dn_b = np.full(n, np.nan, dtype=np.float64)

    for i, t in enumerate(all_ts):
        if t in up_ask:
            up_a[i] = up_ask[t]
            up_b[i] = up_bid[t]
        if t in down_ask:
            dn_a[i] = down_ask[t]
            dn_b[i] = down_bid[t]

    _ffill(up_a)
    _ffill(dn_a)
    _ffill(up_b)
    _ffill(dn_b)

    valid = ~(np.isnan(up_a) & np.isnan(dn_a))
    if valid.sum() < 5:
        return None

    return MarketTicks(
        name=name, ts=ts_arr[valid],
        up_ask=up_a[valid], down_ask=dn_a[valid],
        up_bid=up_b[valid], down_bid=dn_b[valid],
    )


def _ffill(arr: np.ndarray) -> None:
    """In-place forward fill for NaN values."""
    last = np.nan
    for i in range(len(arr)):
        if np.isnan(arr[i]):
            arr[i] = last
        else:
            last = arr[i]


# ── Simulation core (vectorised) ─────────────────────────────────────────────

@dataclass
class MarketProfile:
    """Pre-computed profile for fast grid search on one market."""
    entered_side: str          # 'up' or 'down' or '' if no entry at any level
    entry_price: float         # ask at entry tick (buy price)
    entry_idx: int             # tick index where entry happened
    post_entry_bids: np.ndarray   # bid values after entry (sell price for exits)
    last_bid: float            # final bid price


def _build_profile_instant(market: MarketTicks, entry_level: float) -> Optional[MarketProfile]:
    """Legacy model: enter at first tick with ask >= level (instant fill at touch)."""
    up_a = market.up_ask
    dn_a = market.down_ask
    n = len(up_a)

    entry_idx = -1
    side = ""
    entry_price = 0.0

    for i in range(n):
        u = up_a[i]
        d = dn_a[i]
        if not np.isnan(u) and u >= entry_level:
            entry_idx = i
            side = "up"
            entry_price = u
            break
        if not np.isnan(d) and d >= entry_level:
            entry_idx = i
            side = "down"
            entry_price = d
            break

    if entry_idx < 0:
        return None

    bids = market.up_bid if side == "up" else market.down_bid
    post_bids = bids[entry_idx + 1:]

    valid_bids = post_bids[~np.isnan(post_bids)]
    if len(valid_bids) == 0:
        return MarketProfile(
            entered_side=side, entry_price=entry_price, entry_idx=entry_idx,
            post_entry_bids=valid_bids, last_bid=np.nan,
        )

    return MarketProfile(
        entered_side=side, entry_price=entry_price, entry_idx=entry_idx,
        post_entry_bids=valid_bids, last_bid=float(valid_bids[-1]),
    )


def _build_profile_post_only(market: MarketTicks, entry_level: float) -> Optional[MarketProfile]:
    """Entry as resting BUY @ ``entry_level`` with post-only geometry (matches bot1 roughly).

    1. **Place** when best ask on a side is **strictly above** ``entry_level`` (no cross); try **Up**
       first each tick, then **Down** (same priority as instant profile).
    2. **Fill** at limit ``entry_level`` on the first **later** tick where best ask **<=** ``entry_level``.
       If the ask never returns to the limit before the series ends, no trade (``None``).

    After fill, use ``_sim_profile_for_tps`` (instant exits) or ``_sim_post_only_exits*``
    (``--mega-grid``: TP/SL maker-style on bids).
    """
    L = float(entry_level)
    up_a = market.up_ask
    dn_a = market.down_ask
    n = len(up_a)

    place_idx = -1
    side = ""

    for i in range(n):
        u = up_a[i]
        d = dn_a[i]
        if not np.isnan(u) and u > L:
            place_idx = i
            side = "up"
            break
        if not np.isnan(d) and d > L:
            place_idx = i
            side = "down"
            break

    if place_idx < 0:
        return None

    asks = up_a if side == "up" else dn_a
    fill_idx = -1
    for j in range(place_idx + 1, n):
        a = asks[j]
        if np.isnan(a):
            continue
        if a <= L:
            fill_idx = j
            break

    if fill_idx < 0:
        return None

    bids = market.up_bid if side == "up" else market.down_bid
    post_bids = bids[fill_idx + 1:]

    valid_bids = post_bids[~np.isnan(post_bids)]
    if len(valid_bids) == 0:
        return MarketProfile(
            entered_side=side, entry_price=L, entry_idx=fill_idx,
            post_entry_bids=valid_bids, last_bid=np.nan,
        )

    return MarketProfile(
        entered_side=side, entry_price=L, entry_idx=fill_idx,
        post_entry_bids=valid_bids, last_bid=float(valid_bids[-1]),
    )


def fee_taker_per_share(p: float) -> float:
    """Taker fee per share at price ``p`` (USD): shares * 0.07 * p*(1-p) / shares."""
    return FEE_RATE * p * (1.0 - p)


def fee_maker_rebate_per_share(p: float) -> float:
    """Maker rebate per share (credit): shares * 0.07 * p*(1-p) * 0.2 / shares."""
    return FEE_RATE * p * (1.0 - p) * MAKER_REBATE_FRAC


def net_pnl_per_share(
    gross_pnl: float,
    entry_price: float,
    exit_type: str,
    tp: float,
    sl: float,
    exit_price: float,
) -> float:
    """Gross PnL per share minus entry taker + exit leg (TP maker rebate / SL taker / expiry taker)."""
    cost = fee_taker_per_share(entry_price)
    if exit_type == "tp":
        cost -= fee_maker_rebate_per_share(tp)
    elif exit_type == "sl":
        cost += fee_taker_per_share(sl)
    else:
        # expiry / unresolved — conservative: taker sell at exit bid
        if not np.isnan(exit_price) and exit_price > 0:
            cost += fee_taker_per_share(exit_price)
    return gross_pnl - cost


def _build_profile(
    market: MarketTicks, entry_level: float, post_only_entry: bool = False,
) -> Optional[MarketProfile]:
    """Build trade profile for one market and entry grid level.

    ``post_only_entry=False`` (default): first instant touch (ask >= level).
    ``post_only_entry=True``: post-only resting bid — place when ask > level, fill when ask <= level.
    """
    if post_only_entry:
        return _build_profile_post_only(market, entry_level)
    return _build_profile_instant(market, entry_level)


def _sim_profile_for_tps(
    profile: MarketProfile, tps: np.ndarray, sl: float,
    maker_tp_fill: bool = False,
) -> list[tuple[float, str, float]]:
    """Given a market profile, return (gross_pnl, exit_type, exit_price) per TP level.

    Trigger timing uses BID. SL exit at ``sl``.
    If ``maker_tp_fill``, TP gross uses limit ``tp`` (maker), else bid at trigger (taker-ish).
    """
    n_tp = len(tps)
    results: list[Optional[tuple[float, str, float]]] = [None] * n_tp
    remaining = set(range(n_tp))
    sl_hit = False

    for bid in profile.post_entry_bids:
        if not remaining:
            break
        if bid <= sl and not sl_hit:
            sl_hit = True
            for j in list(remaining):
                results[j] = (sl - profile.entry_price, "sl", sl)
            remaining.clear()
            break
        hit = []
        for j in remaining:
            if bid >= tps[j]:
                tp_px = float(tps[j]) if maker_tp_fill else float(bid)
                results[j] = (tp_px - profile.entry_price, "tp", tp_px)
                hit.append(j)
        for j in hit:
            remaining.discard(j)

    if remaining:
        last_bid = profile.last_bid
        for j in remaining:
            if np.isnan(last_bid):
                results[j] = (-profile.entry_price, "loss", np.nan)
            elif last_bid >= 0.90:
                results[j] = (last_bid - profile.entry_price, "win", float(last_bid))
            else:
                results[j] = (last_bid - profile.entry_price, "loss", float(last_bid))

    return results  # type: ignore


def _accumulate_trade(
    pr: PairResult,
    gross_pnl: float,
    exit_type: str,
    entry_price: float,
    tp: float,
    sl: float,
    exit_price: float,
    apply_fees: bool,
) -> None:
    pr.n_entered += 1
    if apply_fees:
        pr.total_pnl_gross += gross_pnl
        pnl = net_pnl_per_share(gross_pnl, entry_price, exit_type, tp, sl, exit_price)
    else:
        pnl = gross_pnl
    pr.total_pnl += pnl
    if exit_type == "tp":
        pr.n_tp += 1
    elif exit_type == "sl":
        pr.n_sl += 1
    elif exit_type == "win":
        pr.n_resolved_win += 1
    else:
        pr.n_resolved_loss += 1


# ── Post-only TP + SL (bid-path), for --mega-grid ────────────────────────────
# Exit kinds: 0=tp, 1=sl, 2=exp_win, 3=exp_loss, 4=invalid


@njit(cache=True)
def _jit_sim_post_only_exits(bids, last_bid, ep, tp, sl):
    """SELL @ tp / SL @ sl with post-only resting: SL priority; TP arms when bid < tp."""
    if sl >= tp or sl >= ep or tp <= ep:
        return 0.0, 4
    n = bids.shape[0]
    tp_armed = False
    in_sl_maker = False
    for t in range(n):
        b = bids[t]
        if b != b:  # nan
            continue
        if in_sl_maker:
            if b >= sl:
                return sl - ep, 1
            continue
        if b <= sl:
            if b < sl:
                in_sl_maker = True
                continue
            return sl - ep, 1
        if not tp_armed:
            if b >= tp:
                return tp - ep, 0
            tp_armed = True
        else:
            if b >= tp:
                return tp - ep, 0
    lb = last_bid
    if lb != lb:  # nan last_bid
        if n > 0:
            lb = bids[n - 1]
    if lb != lb:
        return -ep, 3
    if in_sl_maker:
        return lb - ep, 1
    if tp_armed:
        if lb >= 0.90:
            return lb - ep, 2
        return lb - ep, 3
    if lb >= tp:
        return tp - ep, 0
    if lb >= 0.90:
        return lb - ep, 2
    return lb - ep, 3


@njit(cache=True)
def _jit_sim_batch(bids, last_bid, ep, tp_arr, sl_arr, out_pnl, out_kind):
    k_n = tp_arr.shape[0]
    for k in range(k_n):
        pnl, kind = _jit_sim_post_only_exits(bids, last_bid, ep, tp_arr[k], sl_arr[k])
        out_pnl[k] = pnl
        out_kind[k] = kind


def _py_sim_post_only_exits(
    bids: np.ndarray, last_bid: float, ep: float, tp: float, sl: float,
) -> tuple[float, int]:
    """Python mirror of jit sim (fallback when numba missing)."""
    if sl >= tp or sl >= ep or tp <= ep:
        return 0.0, 4
    n = len(bids)
    tp_armed = False
    in_sl_maker = False
    for t in range(n):
        b = bids[t]
        if np.isnan(b):
            continue
        if in_sl_maker:
            if b >= sl:
                return sl - ep, 1
            continue
        if b <= sl:
            if b < sl:
                in_sl_maker = True
                continue
            return sl - ep, 1
        if not tp_armed:
            if b >= tp:
                return tp - ep, 0
            tp_armed = True
        else:
            if b >= tp:
                return tp - ep, 0
    lb = last_bid
    if np.isnan(lb) and n > 0:
        lb = float(bids[-1])
    if np.isnan(lb):
        return -ep, 3
    if in_sl_maker:
        return lb - ep, 1
    if tp_armed:
        if lb >= 0.90:
            return lb - ep, 2
        return lb - ep, 3
    if lb >= tp:
        return tp - ep, 0
    if lb >= 0.90:
        return lb - ep, 2
    return lb - ep, 3


def _sim_batch_post_only_exits(
    bids: np.ndarray,
    last_bid: float,
    ep: float,
    tp_arr: np.ndarray,
    sl_arr: np.ndarray,
    out_pnl: np.ndarray,
    out_kind: np.ndarray,
) -> None:
    if _HAS_NUMBA:
        _jit_sim_batch(
            np.ascontiguousarray(bids, dtype=np.float64),
            float(last_bid) if not np.isnan(last_bid) else np.nan,
            float(ep),
            np.ascontiguousarray(tp_arr, dtype=np.float64),
            np.ascontiguousarray(sl_arr, dtype=np.float64),
            out_pnl,
            out_kind,
        )
        return
    for k in range(len(tp_arr)):
        pnl, ki = _py_sim_post_only_exits(
            bids, last_bid, ep, float(tp_arr[k]), float(sl_arr[k]),
        )
        out_pnl[k] = pnl
        out_kind[k] = ki


def generate_mega_combos(
    entries: np.ndarray,
    sl_below_min: float = 0.10,
    sl_below_max: float = 0.28,
    sl_step: float = 0.02,
) -> dict[float, list[tuple[float, float]]]:
    """For each entry, all (tp, sl) with tp on standard grid and sl = entry - delta."""
    out: dict[float, list[tuple[float, float]]] = {}
    deltas = np.arange(sl_below_min, sl_below_max + 1e-9, sl_step)
    deltas = np.round(deltas, 2)
    for entry in entries:
        e = round(float(entry), 2)
        tps = np.arange(e + TP_OFFSET_MIN, TP_MAX + 1e-9, TP_STEP)
        tps = np.round(tps, 2)
        pairs: list[tuple[float, float]] = []
        for tp in tps:
            tpv = float(tp)
            if tpv <= e + 0.001:
                continue
            for d in deltas:
                slv = round(e - float(d), 2)
                if slv < 0.05:
                    continue
                if slv >= tpv - 0.005:
                    continue
                pairs.append((tpv, slv))
        out[e] = pairs
    return out


def _mega_worker_chunk(
    args: tuple[list[MarketTicks], list[float], dict[float, list[tuple[float, float]]]],
) -> dict[tuple[float, float, float], list[float]]:
    """Parallel shard: sum stats per (entry,tp,sl). Value = [pnl, n_ent, n_tp, n_sl, win, loss]."""
    markets_chunk, entries_keys, combo_by_entry = args
    acc: dict[tuple[float, float, float], list[float]] = {}
    for mkt in markets_chunk:
        for e in entries_keys:
            profile = _build_profile_post_only(mkt, e)
            if profile is None:
                continue
            pairs = combo_by_entry.get(e)
            if not pairs:
                continue
            bids = np.asarray(profile.post_entry_bids, dtype=np.float64)
            lb = float(profile.last_bid) if not np.isnan(profile.last_bid) else np.nan
            ep = float(profile.entry_price)
            K = len(pairs)
            tp_arr = np.array([p[0] for p in pairs], dtype=np.float64)
            sl_arr = np.array([p[1] for p in pairs], dtype=np.float64)
            out_pnl = np.empty(K, dtype=np.float64)
            out_kind = np.empty(K, dtype=np.int64)
            _sim_batch_post_only_exits(bids, lb, ep, tp_arr, sl_arr, out_pnl, out_kind)
            for k in range(K):
                ki = int(out_kind[k])
                if ki == 4:
                    continue
                key = (e, round(float(tp_arr[k]), 2), round(float(sl_arr[k]), 2))
                if key not in acc:
                    acc[key] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                row = acc[key]
                row[0] += float(out_pnl[k])
                row[1] += 1.0
                if ki == 0:
                    row[2] += 1.0
                elif ki == 1:
                    row[3] += 1.0
                elif ki == 2:
                    row[4] += 1.0
                else:
                    row[5] += 1.0
    return acc


def run_mega_grid_post_only(
    csv_paths: list[str],
    markets: Optional[list[MarketTicks]] = None,
    sl_below_min: float = 0.10,
    sl_below_max: float = 0.28,
    sl_step: float = 0.02,
) -> tuple[list[PairResult], dict[float, list[tuple[float, float]]]]:
    """Full triple grid: **entry** = post-only BUY; **TP** + **SL** = post-only SELL logic on bids.

    Requires ``numba`` for reasonable runtime (``pip install numba``).
    """
    if markets is None:
        print(f"Loading {len(csv_paths)} market CSVs (mega post-only grid) ...")
        t0 = _time.time()
        n_workers_ld = min(cpu_count(), 8)
        with Pool(n_workers_ld) as pool:
            raw = pool.map(_worker_load, csv_paths)
        markets = [m for m in raw if m is not None]
        print(f"Loaded {len(markets)} valid markets in {_time.time() - t0:.1f}s\n")

    if not markets:
        return [], {}

    entries = np.arange(ENTRY_MIN, ENTRY_MAX + 1e-9, ENTRY_STEP)
    entries = np.round(entries, 2)
    combo_by_entry = generate_mega_combos(
        entries,
        sl_below_min=sl_below_min,
        sl_below_max=sl_below_max,
        sl_step=sl_step,
    )
    n_triple = sum(len(v) for v in combo_by_entry.values())
    entries_keys = [round(float(e), 2) for e in entries]
    print(
        f"Mega grid POST-ONLY (entry+TP+SL): {len(entries)} entries × "
        f"~{n_triple // max(len(entries), 1)} (tp,sl) / entry ≈ {n_triple} triples.\n"
        f"  Entry: resting BUY @ L (ask>L then fill ask<=L).\n"
        f"  TP/SL: resting SELL (bid<limit then fill bid>=limit); SL precedes TP same tick.\n"
    )
    if not _HAS_NUMBA:
        print(
            "  WARNING: numba non installato — mega grid sarà molto lenta. "
            "Consiglio: pip install numba\n"
        )

    n_m = len(markets)
    n_workers = min(cpu_count(), 8)
    chunks: list[list[MarketTicks]] = [
        markets[i::n_workers] for i in range(n_workers)
    ]
    tasks = [(ch, entries_keys, combo_by_entry) for ch in chunks if ch]

    print(f"Running mega grid ({n_workers} workers) ...")
    t0 = _time.time()
    with Pool(n_workers) as pool:
        partials = pool.map(_mega_worker_chunk, tasks)

    merged: dict[tuple[float, float, float], list[float]] = {}
    for part in partials:
        for key, val in part.items():
            if key not in merged:
                merged[key] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            for i in range(6):
                merged[key][i] += val[i]

    print(f"Finished mega grid in {_time.time() - t0:.1f}s\n")

    results: list[PairResult] = []
    for (e, tp, sl), v in merged.items():
        results.append(
            PairResult(
                entry=e,
                tp=tp,
                sl=sl,
                n_markets=n_m,
                n_entered=int(v[1]),
                n_tp=int(v[2]),
                n_sl=int(v[3]),
                n_resolved_win=int(v[4]),
                n_resolved_loss=int(v[5]),
                total_pnl=v[0],
            )
        )
    return results, combo_by_entry


def print_mega_results(results: list[PairResult], top_n: int = 40) -> None:
    """Same columns as ``print_results``; title stresses full post-only triple."""
    if not results:
        return
    results.sort(key=lambda r: r.total_pnl / max(r.n_markets, 1), reverse=True)
    top = results[:top_n]
    print("=" * 120)
    print(
        f"  TOP {top_n} (entry, TP, SL) — POST-ONLY su tutti e tre i livelli — per P&L/mkt"
    )
    print("=" * 120)
    header = (
        f"{'Entry':>6} {'TP':>6} {'SL':>6} | "
        f"{'Entered':>8} {'TP hits':>8} {'SL hits':>8} {'Exp.Win':>8} {'Exp.Loss':>8} | "
        f"{'WinRate':>8} {'AvgPnL':>10} {'PnL/mkt':>10} {'Daily$10':>10}"
    )
    print(header)
    print("-" * 120)
    for pr in top:
        n_wins = pr.n_tp + pr.n_resolved_win
        n_losses = pr.n_sl + pr.n_resolved_loss
        n_total = n_wins + n_losses
        wr = (n_wins / n_total * 100) if n_total > 0 else 0.0
        avg_pnl = (pr.total_pnl / pr.n_entered) if pr.n_entered > 0 else 0.0
        pnl_per_mkt = pr.total_pnl / pr.n_markets
        shares_per_trade = 10.0 / pr.entry
        daily_dollar = pnl_per_mkt * MARKETS_PER_DAY * shares_per_trade
        print(
            f"{pr.entry:>6.2f} {pr.tp:>6.2f} {pr.sl:>6.2f} | "
            f"{pr.n_entered:>8} {pr.n_tp:>8} {pr.n_sl:>8} {pr.n_resolved_win:>8} {pr.n_resolved_loss:>8} | "
            f"{wr:>7.1f}% {avg_pnl:>+10.4f} {pnl_per_mkt:>+10.5f} {daily_dollar:>+10.2f}"
        )
    print("=" * 120)
    print()
    print("Legenda (mega post-only):")
    print("  Entry     = BUY maker: ask>L poi fill quando ask<=L.")
    print("  TP        = SELL maker @ TP: se bid<TP resta; fill quando bid>=TP (o taker se bid>=TP al primo tick).")
    print("  SL        = se bid<=SL: SELL maker @ SL; fill quando bid>=SL; a scadenza bid se mai fill.")
    print("  Entered   = mercati con entry post-only fillata.")
    print(f"  Daily$10  = stima $10/trade × {MARKETS_PER_DAY} mercati")
    print()


# ── Grid search ──────────────────────────────────────────────────────────────

def _worker_load(csv_path: str) -> Optional[MarketTicks]:
    return load_market(csv_path)


def run_backtest(
    csv_paths: list[str],
    sl_offset: float,
    max_days: Optional[int] = None,
    post_only_entry: bool = False,
    apply_fees: bool = False,
):
    """Run full grid search across all markets.

    ``apply_fees``: entry taker + TP maker rebate + SL taker (instant entry path).
    """

    print(f"Loading {len(csv_paths)} market CSVs ...")
    t0 = _time.time()

    n_workers = min(cpu_count(), 8)
    with Pool(n_workers) as pool:
        raw = pool.map(_worker_load, csv_paths)

    markets = [m for m in raw if m is not None]
    elapsed = _time.time() - t0
    print(f"Loaded {len(markets)} valid markets in {elapsed:.1f}s\n")

    if not markets:
        print("No valid market data found.")
        return []

    entries = np.arange(ENTRY_MIN, ENTRY_MAX + 1e-9, ENTRY_STEP)
    entries = np.round(entries, 2)

    # Pre-allocate result accumulators keyed by (entry, tp)
    pair_data: dict[tuple[float, float], PairResult] = {}
    total_combos = 0

    if apply_fees:
        print(
            "Fees ON: entry TAKER, TP MAKER (fill @ TP limit), SL TAKER — "
            f"rate={FEE_RATE}, rebate={MAKER_REBATE_FRAC}\n"
        )
    print("Running grid search ...")
    t0 = _time.time()
    n_entries = len(entries)
    maker_tp = apply_fees

    for ei, entry in enumerate(entries):
        sl = round(entry - sl_offset, 2)
        tps = np.arange(entry + TP_OFFSET_MIN, TP_MAX + 1e-9, TP_STEP)
        tps = np.round(tps, 2)
        n_tp = len(tps)
        total_combos += n_tp

        # Init result objects for this entry row
        row_results = []
        for tp in tps:
            pr = PairResult(entry=entry, tp=tp, sl=sl, n_markets=len(markets))
            pair_data[(entry, tp)] = pr
            row_results.append(pr)

        # For each market, build profile once per entry level, then sweep all TPs
        for mkt in markets:
            profile = _build_profile(mkt, entry, post_only_entry=post_only_entry)
            if profile is None:
                continue

            sim_results = _sim_profile_for_tps(profile, tps, sl, maker_tp_fill=maker_tp)

            for idx, (gross, exit_type, exit_px) in enumerate(sim_results):
                _accumulate_trade(
                    row_results[idx], gross, exit_type,
                    profile.entry_price, float(tps[idx]), sl, exit_px, apply_fees,
                )

        if (ei + 1) % 10 == 0 or ei == n_entries - 1:
            pct = (ei + 1) / n_entries * 100
            print(f"  entry levels: {ei + 1}/{n_entries} ({pct:.0f}%)")

    elapsed = _time.time() - t0
    print(f"Tested {total_combos} combos in {elapsed:.1f}s\n")

    return list(pair_data.values())


def run_backtest_fixed_combos(
    csv_paths: list[str],
    sl_offset: float,
    combos: list[tuple[float, float]],
    markets: Optional[list[MarketTicks]] = None,
    post_only_entry: bool = False,
    apply_fees: bool = False,
) -> list[PairResult]:
    """Same simulation as the full grid, but only for explicit (entry, TP) pairs.

    For each distinct entry level, ``_build_profile`` is called once per market;
    TP levels for that entry share one ``_sim_profile_for_tps`` call (TP array sorted
    ascending to match grid behaviour).

    If ``markets`` is provided, CSV loading is skipped (caller already loaded).
    """
    if markets is None:
        print(f"Loading {len(csv_paths)} market CSVs (fixed combo mode) ...")
        t0 = _time.time()
        n_workers = min(cpu_count(), 8)
        with Pool(n_workers) as pool:
            raw = pool.map(_worker_load, csv_paths)
        markets = [m for m in raw if m is not None]
        print(f"Loaded {len(markets)} valid markets in {_time.time() - t0:.1f}s\n")

    if not markets or not combos:
        return []

    n_markets = len(markets)

    by_entry: dict[float, list[tuple[int, float]]] = defaultdict(list)
    for i, (e, t) in enumerate(combos):
        by_entry[e].append((i, t))

    pair_results: list[PairResult] = []
    for entry, tp in combos:
        sl = round(entry - sl_offset, 2) if sl_offset < 500 else -999.0
        pair_results.append(
            PairResult(entry=entry, tp=tp, sl=sl, n_markets=n_markets)
        )

    print("Running fixed-combo simulation ...")
    t0 = _time.time()
    for mkt in markets:
        for entry, idx_tp_list in by_entry.items():
            sl = round(entry - sl_offset, 2) if sl_offset < 500 else -999.0
            profile = _build_profile(mkt, entry, post_only_entry=post_only_entry)
            if profile is None:
                continue
            pairs = sorted(idx_tp_list, key=lambda x: x[1])
            combo_indices = [x[0] for x in pairs]
            tps = np.array([x[1] for x in pairs], dtype=np.float64)
            sim_results = _sim_profile_for_tps(
                profile, tps, sl, maker_tp_fill=apply_fees,
            )
            for k, combo_idx in enumerate(combo_indices):
                gross, exit_type, exit_px = sim_results[k]
                pr = pair_results[combo_idx]
                _accumulate_trade(
                    pr, gross, exit_type, profile.entry_price,
                    float(tps[k]), sl, exit_px, apply_fees,
                )

    print(f"Finished {len(combos)} combos × {n_markets} markets in {_time.time() - t0:.1f}s\n")
    return pair_results


def print_fixed_combo_table(
    results: list[PairResult],
    title: str,
    markets: Optional[list[MarketTicks]] = None,
    sl_offset: Optional[float] = None,
    capital: float = 10.0,
    post_only_entry: bool = False,
    apply_fees: bool = False,
) -> None:
    """Same columns as print_results; optional dollar-path equity (DD, streaks, day windows)."""
    if not results:
        return
    eq_mode = markets is not None and sl_offset is not None and sl_offset < 500
    width = 178 if eq_mode else 120
    print("=" * width)
    print(f"  {title}")
    print("=" * width)
    base = (
        f"{'#':>3} {'Entry':>6} {'TP':>6} {'SL':>6} | "
        f"{'Entered':>8} {'TP hits':>8} {'SL hits':>8} {'Exp.Win':>8} {'Exp.Loss':>8} | "
        f"{'WinRate':>8} {'AvgPnL':>10} {'PnL/mkt':>10} {'Daily$10':>10}"
    )
    if eq_mode:
        header = (
            base
            + " | "
            + f"{'Tot$':>10} {'MaxDD':>9} {'StrkMax':>14} {'StrkWorst$':>14} {'Giorni+':>12}"
        )
    else:
        header = base
    print(header)
    print("-" * width)
    for rank, pr in enumerate(results, 1):
        n_wins = pr.n_tp + pr.n_resolved_win
        n_losses = pr.n_sl + pr.n_resolved_loss
        n_total = n_wins + n_losses
        wr = (n_wins / n_total * 100) if n_total > 0 else 0.0
        avg_pnl = (pr.total_pnl / pr.n_entered) if pr.n_entered > 0 else 0.0
        pnl_per_mkt = pr.total_pnl / pr.n_markets
        shares_per_trade = capital / pr.entry
        daily_dollar = pnl_per_mkt * MARKETS_PER_DAY * shares_per_trade
        row = (
            f"{rank:>3} {pr.entry:>6.2f} {pr.tp:>6.2f} {pr.sl:>6.2f} | "
            f"{pr.n_entered:>8} {pr.n_tp:>8} {pr.n_sl:>8} {pr.n_resolved_win:>8} {pr.n_resolved_loss:>8} | "
            f"{wr:>7.1f}% {avg_pnl:>+10.4f} {pnl_per_mkt:>+10.5f} {daily_dollar:>+10.2f}"
        )
        if eq_mode and markets is not None:
            sl_v = round(pr.entry - float(sl_offset), 2)
            pnls = simulate_trades_dollar_pnls(
                markets, pr.entry, pr.tp, sl_v, capital,
                post_only_entry=post_only_entry,
                apply_fees=apply_fees,
            )
            st = compute_equity_stats_from_pnls(pnls)
            smx = f"{st.longest_losing_streak_trades}t/{st.longest_losing_streak_pnl:+.0f}$"
            sw = f"{st.worst_streak_trades}t/{st.worst_streak_pnl:+.0f}$"
            if st.total_day_windows > 0:
                dpct = 100.0 * st.positive_day_windows / st.total_day_windows
                gstr = f"{st.positive_day_windows}/{st.total_day_windows} ({dpct:.0f}%)"
            else:
                gstr = "—"
            row += (
                " | "
                f"{st.total_pnl:>+10.2f} {st.max_drawdown:>9.2f} {smx:>14} {sw:>14} {gstr:>12}"
            )
        print(row)
    print("=" * width)
    if eq_mode:
        print(
            "  StrkMax   = streak più lunga di trade con PnL<0 (n trade / somma $ in quella streak).\n"
            "  StrkWorst$ = streak con somma $ peggiore (n trade / somma $).\n"
            f"  Giorni+   = finestre consecutive di {MARKETS_PER_DAY} trade (~1 giorno di mercati 5m); "
            "PnL finestra > 0.\n"
            f"  Tot$/MaxDD/streaks usano ${capital:.0f}/trade e stesso ordine CSV dell'equity.\n"
        )
    print()


# ── Output ────────────────────────────────────────────────────────────────────

def print_results(results: list[PairResult], top_n: int = 30, fees_applied: bool = False):
    if not results:
        return

    # Sort by expected P&L per market (= total_pnl / n_markets)
    results.sort(key=lambda r: r.total_pnl / max(r.n_markets, 1), reverse=True)

    top = results[:top_n]

    print("=" * 120)
    title = f"  TOP {top_n} COMBINAZIONI (entry, TP) — ordinate per P&L atteso per mercato"
    if fees_applied:
        title += " (NET of Polymarket fees: entry taker, TP maker rebate, SL taker)"
    print(title)
    print("=" * 120)
    header = (
        f"{'Entry':>6} {'TP':>6} {'SL':>6} | "
        f"{'Entered':>8} {'TP hits':>8} {'SL hits':>8} {'Exp.Win':>8} {'Exp.Loss':>8} | "
        f"{'WinRate':>8} {'AvgPnL':>10} {'PnL/mkt':>10} {'Daily$10':>10}"
    )
    print(header)
    print("-" * 120)

    for pr in top:
        n_wins = pr.n_tp + pr.n_resolved_win
        n_losses = pr.n_sl + pr.n_resolved_loss
        n_total = n_wins + n_losses
        wr = (n_wins / n_total * 100) if n_total > 0 else 0.0
        avg_pnl = (pr.total_pnl / pr.n_entered) if pr.n_entered > 0 else 0.0
        pnl_per_mkt = pr.total_pnl / pr.n_markets
        # $10 buys (10 / entry_price) shares; daily = pnl_per_share_per_mkt * shares * 288
        shares_per_trade = 10.0 / pr.entry
        daily_dollar = pnl_per_mkt * MARKETS_PER_DAY * shares_per_trade

        print(
            f"{pr.entry:>6.2f} {pr.tp:>6.2f} {pr.sl:>6.2f} | "
            f"{pr.n_entered:>8} {pr.n_tp:>8} {pr.n_sl:>8} {pr.n_resolved_win:>8} {pr.n_resolved_loss:>8} | "
            f"{wr:>7.1f}% {avg_pnl:>+10.4f} {pnl_per_mkt:>+10.5f} {daily_dollar:>+10.2f}"
        )

    print("=" * 120)
    print()
    print("Legenda:")
    print("  Entered   = mercati dove almeno un outcome ha raggiunto il livello entry")
    print("  TP hits   = uscite via Take Profit")
    print("  SL hits   = uscite via Stop Loss")
    print("  Exp.Win   = portati a scadenza con prezzo finale >= 0.90 (probabile win)")
    print("  Exp.Loss  = portati a scadenza con prezzo finale < 0.90")
    print("  WinRate   = (TP + Exp.Win) / totale trades")
    print("  AvgPnL    = P&L medio per trade (in quote, 1 share)")
    print("  PnL/mkt   = P&L medio per mercato (inclusi quelli senza entry)")
    print(f"  Daily$10  = stima guadagno giornaliero con $10/trade x {MARKETS_PER_DAY} mercati")
    if fees_applied:
        print(
            f"  Fee model: taker {FEE_RATE}*p*(1-p); maker rebate ×{MAKER_REBATE_FRAC}; "
            "entry taker, TP @ limit maker, SL taker"
        )
    print()


# ── Equity curve / drawdown analysis ─────────────────────────────────────────

def simulate_trades_dollar_pnls(
    markets: list[MarketTicks],
    entry_level: float,
    tp: float,
    sl: float,
    capital_per_trade: float,
    post_only_entry: bool = False,
    apply_fees: bool = False,
) -> list[float]:
    """One PnL in dollars per entered market; same rules as ``run_equity_analysis``."""
    pnls: list[float] = []
    for mkt in markets:
        profile = _build_profile(mkt, entry_level, post_only_entry=post_only_entry)
        if profile is None:
            continue
        tps = np.array([tp], dtype=np.float64)
        sim = _sim_profile_for_tps(profile, tps, sl, maker_tp_fill=apply_fees)
        gross, exit_type, exit_px = sim[0]
        shares = capital_per_trade / profile.entry_price
        if apply_fees:
            net_ps = net_pnl_per_share(
                gross, profile.entry_price, exit_type, tp, sl, exit_px,
            )
            pnls.append(float(net_ps * shares))
        else:
            pnls.append(float(gross * shares))
    return pnls


def compute_equity_stats_from_pnls(pnls: list[float], trades_per_day: int = MARKETS_PER_DAY) -> EquityStats:
    """Drawdown, losing streaks, and fixed-size 'day' windows (``trades_per_day`` trades each)."""
    if not pnls:
        return EquityStats()
    arr = np.array(pnls, dtype=np.float64)
    n_trades = len(arr)
    total_pnl = float(arr.sum())
    cumulative = np.cumsum(arr)
    peak = np.maximum.accumulate(cumulative)
    drawdown = cumulative - peak
    max_dd = float(drawdown.min())
    max_dd_idx = int(np.argmin(drawdown))

    streaks: list[tuple[int, float]] = []
    cur_n = 0
    cur_loss = 0.0
    for p in pnls:
        if p < 0:
            cur_n += 1
            cur_loss += p
        else:
            if cur_n > 0:
                streaks.append((cur_n, cur_loss))
            cur_n = 0
            cur_loss = 0.0
    if cur_n > 0:
        streaks.append((cur_n, cur_loss))

    longest = max(streaks, key=lambda x: x[0]) if streaks else (0, 0.0)
    worst = min(streaks, key=lambda x: x[1]) if streaks else (0, 0.0)

    pos_days = 0
    n_windows = 0
    if n_trades >= trades_per_day:
        for d_start in range(0, n_trades - trades_per_day + 1, trades_per_day):
            d_end = min(d_start + trades_per_day, n_trades)
            d_pnl = float(arr[d_start:d_end].sum())
            n_windows += 1
            if d_pnl > 0:
                pos_days += 1

    return EquityStats(
        n_trades=n_trades,
        total_pnl=total_pnl,
        max_drawdown=max_dd,
        max_dd_trade_idx=max_dd_idx,
        longest_losing_streak_trades=longest[0],
        longest_losing_streak_pnl=float(longest[1]),
        worst_streak_trades=worst[0],
        worst_streak_pnl=float(worst[1]),
        positive_day_windows=pos_days,
        total_day_windows=n_windows,
    )


def run_equity_analysis(
    markets: list[MarketTicks],
    entry_level: float, tp: float, sl: float,
    capital_per_trade: float = 10.0,
    post_only_entry: bool = False,
    apply_fees: bool = False,
):
    """Trade-by-trade simulation for a single (entry, TP, SL) combo.

    Markets are processed in CSV order (roughly chronological).
    Prints equity curve stats, drawdown, losing streaks, bankroll sizing.
    """
    trades: list[dict] = []

    for mkt in markets:
        profile = _build_profile(mkt, entry_level, post_only_entry=post_only_entry)
        if profile is None:
            continue

        shares = capital_per_trade / profile.entry_price
        tps = np.array([tp], dtype=np.float64)
        gross, exit_type, exit_price = _sim_profile_for_tps(
            profile, tps, sl, maker_tp_fill=apply_fees,
        )[0]

        if np.isnan(exit_price) and exit_type in ("loss",):
            pnl_dollar = -capital_per_trade
        elif apply_fees:
            pnl_dollar = float(
                net_pnl_per_share(gross, profile.entry_price, exit_type, tp, sl, exit_price) * shares
            )
        else:
            pnl_dollar = float(gross * shares)

        trades.append({
            "entry": profile.entry_price,
            "exit_bid": exit_price,
            "type": exit_type,
            "pnl": pnl_dollar,
            "side": profile.entered_side,
        })

    if not trades:
        print("No trades executed.")
        return

    pnls = [float(t["pnl"]) for t in trades]
    n_trades = len(trades)
    n_wins = sum(1 for t in trades if t["type"] in ("tp", "win"))
    n_sl = sum(1 for t in trades if t["type"] == "sl")
    n_loss = sum(1 for t in trades if t["type"] == "loss")
    total_pnl = float(sum(pnls))
    win_rate = n_wins / n_trades * 100

    st = compute_equity_stats_from_pnls(pnls)
    max_dd = st.max_drawdown
    max_dd_idx = st.max_dd_trade_idx
    longest_streak = (st.longest_losing_streak_trades, st.longest_losing_streak_pnl)
    worst_streak = (st.worst_streak_trades, st.worst_streak_pnl)

    trades_per_day = MARKETS_PER_DAY
    n_days = n_trades / trades_per_day if trades_per_day > 0 else 1
    daily_pnl = total_pnl / n_days

    avg_win = np.mean([t["pnl"] for t in trades if t["pnl"] > 0]) if n_wins > 0 else 0
    avg_loss = np.mean([t["pnl"] for t in trades if t["pnl"] < 0]) if (n_sl + n_loss) > 0 else 0

    print("=" * 80)
    extra = ""
    if apply_fees:
        extra = " | FEES: entry taker, TP maker, SL taker"
    elif post_only_entry:
        extra = " | POST-ONLY entry"
    print(
        f"  EQUITY CURVE — Entry {entry_level:.2f} | TP {tp:.2f} | SL {sl:.2f} | "
        f"${capital_per_trade}/trade{extra}"
    )
    print("=" * 80)
    print()
    print(f"  Trades totali:       {n_trades}")
    print(f"  Win (TP + exp.win):  {n_wins} ({win_rate:.1f}%)")
    print(f"  SL hits:             {n_sl}")
    print(f"  Exp. loss:           {n_loss}")
    print(f"  Giorni simulati:     {n_days:.1f}")
    print()
    print(f"  P&L totale:          ${total_pnl:+.2f}")
    print(f"  P&L giornaliero:     ${daily_pnl:+.2f}")
    print(f"  Win medio:           ${avg_win:+.4f}")
    print(f"  Loss medio:          ${avg_loss:+.4f}")
    print(f"  Ratio win/loss:      {abs(avg_win/avg_loss):.2f}x" if avg_loss != 0 else "")
    print()
    print(f"  Max drawdown:        ${max_dd:.2f} (al trade #{max_dd_idx + 1})")
    print(f"  Streak persa max:    {longest_streak[0]} trades (${longest_streak[1]:+.2f})")
    print(f"  Streak peggiore $:   {worst_streak[0]} trades (${worst_streak[1]:+.2f})")
    print()

    # Bankroll sizing
    safe_1x = abs(max_dd) * 1.5
    safe_2x = abs(max_dd) * 2.0
    print(f"  --- BANKROLL CONSIGLIATO ---")
    print(f"  Conservativo (2x max DD):  ${safe_2x:.0f}")
    print(f"  Moderato (1.5x max DD):    ${safe_1x:.0f}")
    print(f"  Minimo (1x max DD):        ${abs(max_dd):.0f}")
    print()

    # Show worst 10 consecutive-day windows
    if n_trades >= trades_per_day:
        day_pnls = []
        for d_start in range(0, n_trades - trades_per_day + 1, trades_per_day):
            d_end = min(d_start + trades_per_day, n_trades)
            d_pnl = sum(pnls[d_start:d_end])
            day_pnls.append((d_start // trades_per_day + 1, d_pnl))

        day_pnls_sorted = sorted(day_pnls, key=lambda x: x[1])
        print(f"  --- PEGGIOR GIORNI ---")
        for rank, (day_num, dpnl) in enumerate(day_pnls_sorted[:5], 1):
            print(f"    #{rank}: Giorno {day_num} → ${dpnl:+.2f}")

        best = sorted(day_pnls, key=lambda x: x[1], reverse=True)
        print(f"\n  --- MIGLIOR GIORNI ---")
        for rank, (day_num, dpnl) in enumerate(best[:5], 1):
            print(f"    #{rank}: Giorno {day_num} → ${dpnl:+.2f}")

        print()
        winning_days = st.positive_day_windows
        total_days = st.total_day_windows
        if total_days > 0:
            print(f"  Giorni positivi: {winning_days}/{total_days} ({winning_days/total_days*100:.0f}%)")

    print("=" * 80)
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def _load_markets(csv_paths: list[str]) -> list[MarketTicks]:
    """Load and filter markets (shared by grid search and equity analysis)."""
    n_workers = min(cpu_count(), 8)
    with Pool(n_workers) as pool:
        raw = pool.map(_worker_load, csv_paths)
    return [m for m in raw if m is not None]


def main():
    parser = argparse.ArgumentParser(description="Backtest entry/TP levels for BTC 5-min markets")
    parser.add_argument("--days", type=int, default=None, help="Limit to last N days")
    parser.add_argument("--no-sl", action="store_true", help="Disable stop loss (hold to expiry or TP)")
    parser.add_argument("--sl-offset", type=float, default=SL_OFFSET, help=f"SL = entry - offset (default {SL_OFFSET})")
    parser.add_argument("--top", type=int, default=30, help="Show top N results (default 30)")
    parser.add_argument("--equity", action="store_true", help="Run equity/drawdown analysis for best combo")
    parser.add_argument("--entry", type=float, default=None, help="Entry level for equity analysis")
    parser.add_argument("--tp", type=float, default=None, help="TP level for equity analysis")
    parser.add_argument("--capital", type=float, default=10.0, help="Capital per trade (default $10)")
    parser.add_argument(
        "--top3-presets",
        action="store_true",
        help="Top-3 (entry,TP) dalla griglia: tabella griglia + Tot$/MaxDD/streak/giorni (stesso dataset)",
    )
    parser.add_argument(
        "--top4-presets",
        action="store_true",
        help="Top-4 (entry,TP): stessa tabella estesa di --top3-presets più una riga",
    )
    parser.add_argument(
        "--post-only-entry",
        action="store_true",
        help=(
            "Entry come maker post-only: piazza BUY al limite quando ask > entry; "
            "fill al primo tick successivo con ask <= entry. "
            "TP/SL restano sul path dei bid (non modello maker in uscita)."
        ),
    )
    parser.add_argument(
        "--mega-grid",
        action="store_true",
        help=(
            "Griglia (entry, TP, SL): ingresso post-only + uscite TP/SL post-only sui bid. "
            "Richiede numba (pip install numba)."
        ),
    )
    parser.add_argument(
        "--mega-sl-min",
        type=float,
        default=0.10,
        help="Mega-grid: SL = entry - delta, delta minimo (default 0.10)",
    )
    parser.add_argument(
        "--mega-sl-max",
        type=float,
        default=0.28,
        help="Mega-grid: delta massimo verso SL (default 0.28 → SL fino a entry-0.28)",
    )
    parser.add_argument(
        "--mega-sl-step",
        type=float,
        default=0.02,
        help="Mega-grid: passo delta SL (default 0.02)",
    )
    parser.add_argument(
        "--with-fees",
        action="store_true",
        help=(
            "Applica fee Polymarket al PnL: entry TAKER, TP MAKER (fill @ TP), SL TAKER. "
            "Formula: shares*0.07*p*(1-p); rebate TP ×0.2. Usa entry istantaneo (primo ask>=entry)."
        ),
    )
    args = parser.parse_args()

    prices_dir = os.path.abspath(PRICES_DIR)
    if not os.path.isdir(prices_dir):
        print(f"Directory not found: {prices_dir}")
        sys.exit(1)

    day_dirs = sorted(glob.glob(os.path.join(prices_dir, "*")))
    if args.days:
        day_dirs = day_dirs[-args.days:]

    csv_paths = []
    for dd in day_dirs:
        csv_dir = os.path.join(dd, "csv")
        if os.path.isdir(csv_dir):
            csv_paths.extend(sorted(glob.glob(os.path.join(csv_dir, "*.csv"))))

    sl_offset = 999.0 if args.no_sl else args.sl_offset

    if args.top3_presets and args.top4_presets:
        print("Usa solo uno tra --top3-presets e --top4-presets.", file=sys.stderr)
        sys.exit(2)

    if args.mega_grid and (args.top3_presets or args.top4_presets or args.equity):
        print("--mega-grid non è combinabile con --top3-presets / --top4-presets / --equity.", file=sys.stderr)
        sys.exit(2)

    preset_n = 4 if args.top4_presets else (3 if args.top3_presets else 0)
    if preset_n:
        combos = PRESET_TOP4_COMBOS[:preset_n]
        print(f"Found {len(csv_paths)} CSVs across {len(day_dirs)} days")
        if args.post_only_entry:
            print(
                "Entry model: POST-ONLY — resting bid @ entry (place ask>entry, fill ask<=entry). "
                "Exits: same bid walk (TP/SL not post-only)."
            )
        if args.no_sl:
            print("SL: DISABLED")
        else:
            print(f"SL: entry - {sl_offset:.2f} (fixed combos use each row's entry)")
        print(f"Fixed combos (rank order, first {preset_n}): {combos}")
        print(f"Equity columns: ${args.capital:.0f}/trade, path = simulate_trades_dollar_pnls (stesso equity)\n")
        t0 = _time.time()
        markets = _load_markets(csv_paths)
        print(f"Loaded {len(markets)} valid markets in {_time.time() - t0:.1f}s\n")
        po_entry = args.post_only_entry and not args.with_fees
        if args.with_fees and args.post_only_entry:
            print("Nota: --with-fees usa entry taker (istantaneo); --post-only-entry ignorato.\n")
        results = run_backtest_fixed_combos(
            csv_paths,
            sl_offset,
            combos,
            markets=markets,
            post_only_entry=po_entry,
            apply_fees=args.with_fees,
        )
        title = f"TOP {preset_n} PRESETS — griglia + drawdown / streak / giorni (path $)"
        if args.with_fees:
            title += " — NET FEES"
        elif args.post_only_entry:
            title += " — POST-ONLY ENTRY"
        print_fixed_combo_table(
            results,
            title,
            markets=markets,
            sl_offset=sl_offset,
            capital=args.capital,
            post_only_entry=po_entry,
            apply_fees=args.with_fees,
        )
        return

    if args.mega_grid:
        print(f"Found {len(csv_paths)} CSVs across {len(day_dirs)} days")
        mega_results, _ = run_mega_grid_post_only(
            csv_paths,
            markets=None,
            sl_below_min=args.mega_sl_min,
            sl_below_max=args.mega_sl_max,
            sl_step=args.mega_sl_step,
        )
        print_mega_results(mega_results, top_n=max(args.top, 1))
        return

    if args.equity:
        entry = args.entry if args.entry else 0.60
        tp_val = args.tp if args.tp else 0.97
        sl_val = round(entry - sl_offset, 2) if not args.no_sl else -999.0

        print(f"Found {len(csv_paths)} CSVs across {len(day_dirs)} days")
        print(f"Loading markets for equity analysis ...\n")

        t0 = _time.time()
        markets = _load_markets(csv_paths)
        print(f"Loaded {len(markets)} valid markets in {_time.time() - t0:.1f}s\n")

        po_entry = args.post_only_entry and not args.with_fees
        run_equity_analysis(
            markets, entry, tp_val, sl_val, args.capital,
            post_only_entry=po_entry,
            apply_fees=args.with_fees,
        )
        return

    print(f"Found {len(csv_paths)} CSVs across {len(day_dirs)} days")
    print(f"Grid: entry [{ENTRY_MIN:.2f} .. {ENTRY_MAX:.2f}], TP [entry+{TP_OFFSET_MIN:.2f} .. {TP_MAX:.2f}]")

    if args.no_sl:
        print("SL: DISABLED (hold to TP or expiry)")
    else:
        print(f"SL: entry - {sl_offset:.2f}")
    print()

    po_entry = args.post_only_entry and not args.with_fees
    if args.with_fees:
        if args.post_only_entry:
            print("Nota: --with-fees usa entry taker (istantaneo); --post-only-entry ignorato.\n")
        print(
            "Modello live + fee: entry TAKER (ask>=entry), TP MAKER @ TP, SL TAKER @ SL.\n"
        )
    elif args.post_only_entry:
        print(
            "Entry model: POST-ONLY — resting bid @ entry. "
            "(TP/SL sul bid path; uscite maker non simulate.)\n"
        )

    results = run_backtest(
        csv_paths,
        sl_offset=sl_offset,
        max_days=args.days,
        post_only_entry=po_entry,
        apply_fees=args.with_fees,
    )
    print_results(results, top_n=args.top, fees_applied=args.with_fees)


if __name__ == "__main__":
    main()
