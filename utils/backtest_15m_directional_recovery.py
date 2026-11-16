#!/usr/bin/env python3
"""
Backtest: directional entry near 50c + TP/SL + up to 2 recovery flips with scaled size.

Tutti i parametri di default stanno in cima (non usare .env).
Uso:
  python utils/backtest_15m_directional_recovery.py              # grid GRID in cima + report
  python utils/backtest_15m_directional_recovery.py --wide-grid  # GRID + extended_grid (più lento)
  python utils/backtest_15m_directional_recovery.py --single     # StrategyParams() default o PARAM_SINGLE
"""
from __future__ import annotations

import csv
import glob
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Iterator, Literal

# =============================================================================
# PARAMETRI (modifica qui)
# =============================================================================

PRICES_ROOT = Path(__file__).resolve().parents[1] / "prices" / "15" / "days"
DATE_START = date(2026, 3, 12)
DATE_END = date(2026, 3, 23)

# Ingresso: primo tick in cui almeno un lato ha ask in banda; se entrambi, scegli il più alto o il più basso
ENTRY_BAND_LO = 0.46
ENTRY_BAND_HI = 0.54
ENTRY_TIE_BREAK: Literal["higher", "lower"] = "higher"

# Primo leg
TP0 = 0.70
SL0 = 0.26

# Recovery leg 1 e 2 (prezzi assoluti sull’ask del lato detenuto)
TP1 = 0.92
SL1 = 0.18  # se None o <=0 → nessuno SL sul recovery (solo TP o settlement a fine serie)
TP2 = 0.88
SL2 = 0.12

MAX_RECOVERIES = 2
SIZE_BASE = 1.0
SIZE_RECOVERY_MULTS: tuple[float, ...] = (1.5, 3.0)  # 1° recovery, 2° recovery

# Filtro calma prima di aprire un recovery: finestra all’indice di ingresso
CALM_LOOKBACK_SEC = 40.0
CALM_MAX_MID_RANGE = 0.10  # max(mid_up) - min(mid_up) con mid_up = up/(up+dn)
CALM_MAX_WAIT_SEC = 120.0  # dopo SL, quanto attendere un tick “calmo” per entrare in recovery
SKIP_RECOVERY_IF_NO_CALM = True  # se False, entra subito senza filtro

# Opzionale: non aprire recovery se ask opposto > soglia (troppo caro)
MAX_OPP_ASK_FOR_RECOVERY = 0.82

# Train / validation (solo report)
TRAIN_END = date(2026, 3, 18)  # train: DATE_START..TRAIN_END, val: day after .. DATE_END

# --- Modalità --single (disabilita grid) ---
PARAM_SINGLE = None  # oppure dict sovrascrittura, es. {"TP0": 0.68, "SL0": 0.25}

# =============================================================================
# Grid (lista di dict: chiavi = nomi attributo di StrategyParams)
# =============================================================================

GRID: list[dict[str, float | bool | str | None]] = [
    # baseline
    {"TP0": 0.70, "SL0": 0.26, "TP1": 0.92, "TP2": 0.90, "SKIP_RECOVERY_IF_NO_CALM": False},
    {"TP0": 0.70, "SL0": 0.26, "TP1": 0.92, "TP2": 0.88, "SKIP_RECOVERY_IF_NO_CALM": True},
    {"TP0": 0.68, "SL0": 0.25, "TP1": 0.90, "TP2": 0.86, "SKIP_RECOVERY_IF_NO_CALM": True},
    {"TP0": 0.65, "SL0": 0.24, "TP1": 0.88, "TP2": 0.85, "SKIP_RECOVERY_IF_NO_CALM": True},
    {"TP0": 0.72, "SL0": 0.27, "TP1": 0.93, "TP2": 0.90, "SKIP_RECOVERY_IF_NO_CALM": True},
    {"TP0": 0.70, "SL0": 0.25, "TP1": 0.91, "TP2": 0.87, "MAX_OPP_ASK_FOR_RECOVERY": 0.78},
    {"TP0": 0.70, "SL0": 0.26, "TP1": 0.94, "TP2": 0.90, "MAX_OPP_ASK_FOR_RECOVERY": 0.78},
    {"TP0": 0.66, "SL0": 0.26, "TP1": 0.92, "TP2": 0.88, "SL1": 0.15, "SL2": 0.10},
    {"TP0": 0.70, "SL0": 0.26, "TP1": 0.92, "TP2": 0.88, "SL1": None, "SL2": None},
    {"TP0": 0.64, "SL0": 0.28, "TP1": 0.89, "TP2": 0.84, "SKIP_RECOVERY_IF_NO_CALM": True},
]


@dataclass
class StrategyParams:
    entry_band_lo: float = ENTRY_BAND_LO
    entry_band_hi: float = ENTRY_BAND_HI
    entry_tie_break: str = ENTRY_TIE_BREAK
    tp0: float = TP0
    sl0: float = SL0
    tp1: float = TP1
    sl1: float | None = SL1
    tp2: float = TP2
    sl2: float | None = SL2
    max_recoveries: int = MAX_RECOVERIES
    size_base: float = SIZE_BASE
    size_recovery_mults: tuple[float, ...] = SIZE_RECOVERY_MULTS
    calm_lookback_sec: float = CALM_LOOKBACK_SEC
    calm_max_mid_range: float = CALM_MAX_MID_RANGE
    calm_max_wait_sec: float = CALM_MAX_WAIT_SEC
    skip_recovery_if_no_calm: bool = SKIP_RECOVERY_IF_NO_CALM
    max_opp_ask_for_recovery: float = MAX_OPP_ASK_FOR_RECOVERY

    def tp_sl_for_leg(self, leg: int) -> tuple[float, float | None]:
        if leg == 0:
            return self.tp0, self.sl0
        if leg == 1:
            return self.tp1, self.sl1
        if leg == 2:
            return self.tp2, self.sl2
        raise ValueError(leg)

    def size_for_leg(self, leg: int) -> float:
        if leg == 0:
            return self.size_base
        if 1 <= leg <= len(self.size_recovery_mults):
            return self.size_base * self.size_recovery_mults[leg - 1]
        return self.size_base


@dataclass
class MarketSeries:
    date_str: str
    path: str
    ts: list[int]
    up: list[float]
    dn: list[float]

    def mid_up(self, i: int) -> float:
        u, d = self.up[i], self.dn[i]
        s = u + d
        return u / s if s > 1e-9 else 0.5


def daterange(a: date, b: date) -> Iterator[date]:
    d = a
    while d <= b:
        yield d
        d += timedelta(days=1)


def load_market_csv(fp: str) -> MarketSeries | None:
    by_ts: dict[int, dict[str, float]] = {}
    date_str = ""
    for part in Path(fp).parts:
        if len(part) == 10 and part[4] == "-" and part[7] == "-":
            try:
                date.fromisoformat(part)
                date_str = part
                break
            except ValueError:
                pass
    with open(fp, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                ts = int(float(row.get("timestamp_unix_ms") or ""))
                out = (row.get("outcome") or "").strip()
                ask = float(row.get("ask") or "")
            except (TypeError, ValueError):
                continue
            if out not in ("Up", "Down"):
                continue
            if ts not in by_ts:
                by_ts[ts] = {}
            by_ts[ts][out] = ask

    if not by_ts:
        return None
    ts_sorted = sorted(by_ts)
    up_list: list[float] = []
    dn_list: list[float] = []
    t_list: list[int] = []
    for t in ts_sorted:
        row = by_ts[t]
        if "Up" not in row or "Down" not in row:
            continue
        t_list.append(t)
        up_list.append(row["Up"])
        dn_list.append(row["Down"])
    if len(t_list) < 3:
        return None
    return MarketSeries(date_str=date_str, path=Path(fp).name, ts=t_list, up=up_list, dn=dn_list)


def load_all_markets(root: Path, start: date, end: date) -> list[MarketSeries]:
    out: list[MarketSeries] = []
    for d in daterange(start, end):
        ds = d.isoformat()
        pattern = str(root / ds / "csv" / "prices_*.csv")
        for fp in sorted(glob.glob(pattern)):
            m = load_market_csv(fp)
            if m:
                out.append(m)
    return out


def find_entry(m: MarketSeries, p: StrategyParams) -> tuple[int, Literal["Up", "Down"]] | None:
    lo, hi = p.entry_band_lo, p.entry_band_hi
    for i in range(len(m.ts)):
        u, d = m.up[i], m.dn[i]
        u_in = lo <= u <= hi
        d_in = lo <= d <= hi
        if not u_in and not d_in:
            continue
        if u_in and d_in:
            if p.entry_tie_break == "higher":
                side: Literal["Up", "Down"] = "Up" if u >= d else "Down"
            else:
                side = "Up" if u < d else "Down"
        elif u_in:
            side = "Up"
        else:
            side = "Down"
        return i, side
    return None


def _opp(side: Literal["Up", "Down"]) -> Literal["Up", "Down"]:
    return "Down" if side == "Up" else "Up"


def ask_at(m: MarketSeries, i: int, side: Literal["Up", "Down"]) -> float:
    return m.up[i] if side == "Up" else m.dn[i]


def calm_ok_at_index(m: MarketSeries, idx: int, p: StrategyParams) -> bool:
    if p.calm_lookback_sec <= 0 or p.calm_max_mid_range <= 0:
        return True
    t0 = m.ts[idx] - int(p.calm_lookback_sec * 1000)
    mids: list[float] = []
    for j in range(idx + 1):
        if m.ts[j] < t0:
            continue
        mids.append(m.mid_up(j))
    if len(mids) < 2:
        return True
    return max(mids) - min(mids) <= p.calm_max_mid_range


def find_recovery_entry_index(m: MarketSeries, sl_idx: int, p: StrategyParams) -> int | None:
    """Dopo SL a sl_idx, primo indice >= sl_idx dove (calma opzionale) e ask opposto ok."""
    if not p.skip_recovery_if_no_calm:
        return sl_idx
    t_limit = m.ts[sl_idx] + int(p.calm_max_wait_sec * 1000)
    for k in range(sl_idx, len(m.ts)):
        if m.ts[k] > t_limit:
            break
        if calm_ok_at_index(m, k, p):
            return k
    return None


def simulate_market(m: MarketSeries, p: StrategyParams) -> dict:
    """
    Ritorna pnl totale (unità = 1 share size base sul leg0; recovery scala),
    numero leg chiusi in TP, recovery usati, ecc.
    """
    ent = find_entry(m, p)
    if ent is None:
        return {
            "pnl": 0.0,
            "skipped": True,
            "reason": "no_entry_band",
            "recoveries": 0,
            "legs": 0,
            "won": False,
        }
    idx, side = ent
    pnl = 0.0
    recoveries = 0
    leg = 0
    max_leg = p.max_recoveries  # 0=initial, recoveries 1..max add legs

    while True:
        tp, sl = p.tp_sl_for_leg(min(leg, 2))
        sl_v = sl if sl is not None and sl > 0 else None
        size = p.size_for_leg(min(leg, 2))
        entry = ask_at(m, idx, side)
        exit_price: float | None = None
        exit_kind = "open"
        j = idx + 1
        while j < len(m.ts):
            price = ask_at(m, j, side)
            hit_tp = price >= tp
            hit_sl = sl_v is not None and price <= sl_v
            if hit_tp and hit_sl:
                # stesso tick: conservativo = SL prima (peggio per noi)
                hit_tp = False
            if hit_tp:
                exit_price = tp
                exit_kind = "tp"
                idx = j
                break
            if hit_sl:
                exit_price = sl_v
                exit_kind = "sl"
                idx = j
                break
            j += 1

        if exit_price is None:
            # settlement proxy: ultimo ask
            exit_price = ask_at(m, len(m.ts) - 1, side)
            exit_kind = "eod"
            j = len(m.ts) - 1

        pnl += size * (exit_price - entry)

        if exit_kind == "tp":
            return {
                "pnl": pnl,
                "skipped": False,
                "recoveries": recoveries,
                "legs": leg + 1,
                "won": True,
                "exit": "tp",
            }

        # SL o EOD senza TP
        if exit_kind == "eod":
            return {
                "pnl": pnl,
                "skipped": False,
                "recoveries": recoveries,
                "legs": leg + 1,
                "won": False,
                "exit": "eod",
            }

        # SL: recovery?
        if recoveries >= p.max_recoveries:
            return {
                "pnl": pnl,
                "skipped": False,
                "recoveries": recoveries,
                "legs": leg + 1,
                "won": False,
                "exit": "sl_final",
            }

        opp = _opp(side)
        rk = find_recovery_entry_index(m, idx, p)
        if rk is None:
            return {
                "pnl": pnl,
                "skipped": False,
                "recoveries": recoveries,
                "legs": leg + 1,
                "won": False,
                "exit": "sl_no_calm_recovery",
            }

        opp_ask = ask_at(m, rk, opp)
        if opp_ask > p.max_opp_ask_for_recovery:
            return {
                "pnl": pnl,
                "skipped": False,
                "recoveries": recoveries,
                "legs": leg + 1,
                "won": False,
                "exit": "sl_opp_too_rich",
            }

        side = opp
        idx = rk
        recoveries += 1
        leg += 1


def params_from_grid_row(row: dict, base: StrategyParams | None = None) -> StrategyParams:
    b = base or StrategyParams()
    d: dict = {
        "entry_band_lo": b.entry_band_lo,
        "entry_band_hi": b.entry_band_hi,
        "entry_tie_break": b.entry_tie_break,
        "tp0": b.tp0,
        "sl0": b.sl0,
        "tp1": b.tp1,
        "sl1": b.sl1,
        "tp2": b.tp2,
        "sl2": b.sl2,
        "max_recoveries": b.max_recoveries,
        "size_base": b.size_base,
        "size_recovery_mults": b.size_recovery_mults,
        "calm_lookback_sec": b.calm_lookback_sec,
        "calm_max_mid_range": b.calm_max_mid_range,
        "calm_max_wait_sec": b.calm_max_wait_sec,
        "skip_recovery_if_no_calm": b.skip_recovery_if_no_calm,
        "max_opp_ask_for_recovery": b.max_opp_ask_for_recovery,
    }
    keymap = {
        "TP0": "tp0",
        "SL0": "sl0",
        "TP1": "tp1",
        "SL1": "sl1",
        "TP2": "tp2",
        "SL2": "sl2",
        "ENTRY_BAND_LO": "entry_band_lo",
        "ENTRY_BAND_HI": "entry_band_hi",
        "ENTRY_TIE_BREAK": "entry_tie_break",
        "MAX_RECOVERIES": "max_recoveries",
        "SKIP_RECOVERY_IF_NO_CALM": "skip_recovery_if_no_calm",
        "MAX_OPP_ASK_FOR_RECOVERY": "max_opp_ask_for_recovery",
        "CALM_MAX_MID_RANGE": "calm_max_mid_range",
        "CALM_LOOKBACK_SEC": "calm_lookback_sec",
    }
    for k, v in row.items():
        dk = keymap.get(k)
        if dk is not None:
            d[dk] = v

    def fsl(x) -> float | None:
        if x is None:
            return None
        return float(x)

    return StrategyParams(
        entry_band_lo=float(d["entry_band_lo"]),
        entry_band_hi=float(d["entry_band_hi"]),
        entry_tie_break=str(d["entry_tie_break"]),
        tp0=float(d["tp0"]),
        sl0=float(d["sl0"]),
        tp1=float(d["tp1"]),
        sl1=fsl(d["sl1"]),
        tp2=float(d["tp2"]),
        sl2=fsl(d["sl2"]),
        max_recoveries=int(d["max_recoveries"]),
        size_base=float(d["size_base"]),
        size_recovery_mults=tuple(d["size_recovery_mults"])
        if isinstance(d["size_recovery_mults"], tuple)
        else SIZE_RECOVERY_MULTS,
        calm_lookback_sec=float(d["calm_lookback_sec"]),
        calm_max_mid_range=float(d["calm_max_mid_range"]),
        calm_max_wait_sec=float(d["calm_max_wait_sec"]),
        skip_recovery_if_no_calm=bool(d["skip_recovery_if_no_calm"]),
        max_opp_ask_for_recovery=float(d["max_opp_ask_for_recovery"]),
    )


def _stats_from_rows(rows: list[dict]) -> dict:
    traded = [r for r in rows if not r.get("skipped")]
    pnl_total = sum(r["pnl"] for r in rows)
    n = len(traded)
    wins = sum(1 for r in traded if r.get("won"))
    nm = len(rows)
    return {
        "n_markets": nm,
        "n_traded": n,
        "n_skipped": sum(1 for r in rows if r.get("skipped")),
        "pnl_total": pnl_total,
        "avg_per_market_all": pnl_total / nm if nm else 0.0,
        "avg_per_market_traded": pnl_total / n if n else 0.0,
        "win_rate": wins / n if n else 0.0,
        "rec0": sum(1 for r in traded if r["recoveries"] == 0),
        "rec1": sum(1 for r in traded if r["recoveries"] == 1),
        "rec2": sum(1 for r in traded if r["recoveries"] >= 2),
    }


def run_backtest(markets: list[MarketSeries], p: StrategyParams) -> dict:
    rows = [simulate_market(m, p) for m in markets]
    return _stats_from_rows(rows)


def run_backtest_train_val_all(
    markets: list[MarketSeries], p: StrategyParams, train_end: date
) -> tuple[dict, dict, dict]:
    """Una sola simulazione per mercato; aggrega train/val/all (più veloce della tripla run_backtest)."""
    rows_all: list[dict] = []
    rows_tr: list[dict] = []
    rows_va: list[dict] = []
    for m in markets:
        r = simulate_market(m, p)
        rows_all.append(r)
        try:
            d = date.fromisoformat(m.date_str)
        except ValueError:
            rows_tr.append(r)
            continue
        if d <= train_end:
            rows_tr.append(r)
        else:
            rows_va.append(r)
    return _stats_from_rows(rows_tr), _stats_from_rows(rows_va), _stats_from_rows(rows_all)


def split_by_date(markets: list[MarketSeries], train_end: date) -> tuple[list[MarketSeries], list[MarketSeries]]:
    tr, va = [], []
    for m in markets:
        try:
            d = date.fromisoformat(m.date_str)
        except ValueError:
            tr.append(m)
            continue
        if d <= train_end:
            tr.append(m)
        else:
            va.append(m)
    return tr, va


def extended_grid() -> list[dict]:
    """Aggiunge combinazioni TP0/SL0/TP1 per ricerca più fitta (solo numeri)."""
    extra: list[dict] = []
    for tp0 in (0.64, 0.66, 0.68, 0.70):
        for sl0 in (0.24, 0.26, 0.28):
            for tp1 in (0.88, 0.90, 0.92, 0.94):
                for mx in (0.78, 0.82, 0.99):
                    extra.append(
                        {
                            "TP0": tp0,
                            "SL0": sl0,
                            "TP1": tp1,
                            "TP2": 0.87,
                            "SKIP_RECOVERY_IF_NO_CALM": True,
                            "MAX_OPP_ASK_FOR_RECOVERY": mx,
                        }
                    )
    return extra


def main() -> None:
    single = "--single" in sys.argv
    wide = "--wide-grid" in sys.argv
    try:
        sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    except Exception:
        pass

    markets = load_all_markets(PRICES_ROOT, DATE_START, DATE_END)
    if not markets:
        print("Nessun mercato caricato. Verifica PRICES_ROOT e date.")
        sys.exit(1)

    train_m, val_m = split_by_date(markets, TRAIN_END)

    if single and PARAM_SINGLE:
        p = params_from_grid_row(PARAM_SINGLE)
        r_all = run_backtest(markets, p)
        print("SINGLE RUN", PARAM_SINGLE)
        print(r_all)
        return

    if single:
        p = StrategyParams()
        print("SINGLE DEFAULT StrategyParams()")
        print(run_backtest(markets, p))
        return

    grid = list(GRID)
    if wide:
        grid.extend(extended_grid())

    results: list[tuple[float, float, float, dict, dict]] = []
    for row in grid:
        try:
            p = params_from_grid_row(row)
        except Exception as e:
            print("skip bad row", row, e, flush=True)
            continue
        rt, rv, ra = run_backtest_train_val_all(markets, p, TRAIN_END)
        results.append((rv["pnl_total"], rt["pnl_total"], ra["pnl_total"], row, ra))
        if wide and len(results) % 25 == 0:
            print(f"  … grid {len(results)}/{len(grid)}", flush=True)

    results.sort(key=lambda x: x[0], reverse=True)

    print("=" * 80)
    print(
        f"Campione: {DATE_START} .. {DATE_END} │ mercati={len(markets)} │ "
        f"train={len(train_m)} ({DATE_START}..{TRAIN_END}) val={len(val_m)}"
    )
    print("Ranking per PnL **validation** (poi train, poi all). Unità = PnL con size 1 / 1.5 / 3.")
    print("=" * 80)
    for i, (pv, pt, pa, row, stats) in enumerate(results[:25]):
        print(f"\n#{i+1}  VAL pnl={pv:+.4f}  TRAIN={pt:+.4f}  ALL={pa:+.4f}")
        print(f"    row={row}")
        print(
            f"    traded={stats['n_traded']} win%={100*stats['win_rate']:.1f} "
            f"avg/traded={stats['avg_per_market_traded']:+.4f} "
            f"rec0/1/2+={stats['rec0']}/{stats['rec1']}/{stats['rec2']}"
        )

    if results:
        best_v, best_t, best_a, best_row, best_stats = results[0]
        print("\n" + "=" * 80)
        print("MIGLIORE SU VALIDATION (non implica profitto reale):")
        print(f"  row={best_row}")
        print(f"  val_pnl={best_v:+.4f} train_pnl={best_t:+.4f} all_pnl={best_a:+.4f}")
        print(f"  {best_stats}")
        print("=" * 80)

    # Sintesi matematica
    print(
        "\nNota: con recovery 1.5x e 3x, le perdite su catene SL+SL+SL pesano ~ come -1 -1.5*... -3*... "
        "in unità di prezzo; un TP alto sul recovery richiede frequenza sufficiente. "
        "Se **tutte** le combinazioni hanno val_pnl < 0, il campione non supporta il pattern."
    )


if __name__ == "__main__":
    main()
