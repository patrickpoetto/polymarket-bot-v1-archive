#!/usr/bin/env python3
"""
Generate Up/Down price charts from per-market CSV snapshots.

Outputs:
- One SVG chart per market in <prices_dir>/charts
- One HTML report in <prices_dir>/price_report.html
"""
from __future__ import annotations

import argparse
import csv
import html
from dataclasses import dataclass
from pathlib import Path

GUIDE_X_RED_SECONDS = None
GUIDE_X_GREEN_SECONDS = None

GUIDE_Y_RED_PRICE = None
GUIDE_Y_GREEN_PRICE = None

# === STRATEGY MARKERS CONFIG ===
# Accepts either [0.0-1.0] or cents [0-100] (e.g. 35 -> 0.35)
ENTRY_PRICE = 0.34
TP_PRICE = 0.48
SL_PRICE = 0.29


@dataclass
class PricePoint:
    ts_ms: int
    outcome: str
    price: float       # ask price (used for entry/exit trigger detection)
    market_name: str
    condition_id: str
    bid: float | None = None  # bid price at the same snapshot (used for realistic exit P&L)


@dataclass
class ChartArtifact:
    market_name: str
    csv_file: Path
    svg_file: Path
    samples_per_outcome: dict[str, int]
    first_ts_ms: int
    strategy: dict[str, object]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate price charts for Polymarket BTC logs.")
    parser.add_argument("--prices-dir", default="logs/bot3_btc/prices_backtest")
    parser.add_argument("--pattern", default="prices_*.csv")
    parser.add_argument("--out-dir", default=None)
    return parser.parse_args()


def load_price_csv(path: Path) -> list[PricePoint]:
    points: list[PricePoint] = []
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = set(reader.fieldnames or [])
        # Support old CSVs with "price" column and new ones with "ask"+"bid"
        has_ask_bid = "ask" in fieldnames and "bid" in fieldnames
        has_legacy_price = "price" in fieldnames
        required = {"timestamp_unix_ms", "condition_id", "market_name", "outcome"}
        if not has_ask_bid and not has_legacy_price:
            raise ValueError(f"{path.name}: missing price columns (need 'ask'+'bid' or 'price')")
        missing = required - fieldnames
        if missing:
            raise ValueError(f"{path.name}: missing columns {sorted(missing)}")
        for row in reader:
            try:
                # Ask price: prefer "ask" column, fall back to legacy "price"
                ask_raw = row.get("ask") or row.get("price") or ""
                price = float(ask_raw)
                ts_ms = int(float(row["timestamp_unix_ms"]))
            except (TypeError, ValueError):
                continue
            outcome = str(row["outcome"] or "").strip()
            if not outcome:
                continue
            bid: float | None = None
            if has_ask_bid:
                try:
                    bid_raw = row.get("bid") or ""
                    if bid_raw:
                        bid = float(bid_raw)
                except (TypeError, ValueError):
                    pass
            points.append(
                PricePoint(
                    ts_ms=ts_ms,
                    outcome=outcome,
                    price=price,
                    market_name=str(row["market_name"] or ""),
                    condition_id=str(row["condition_id"] or ""),
                    bid=bid,
                )
            )

    points.sort(key=lambda p: p.ts_ms)
    if not points:
        raise ValueError(f"{path.name}: no valid points")
    return points


def _polyline(points: list[tuple[float, float]], color: str) -> str:
    coord_str = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    return f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{coord_str}" />'


def _normalize_price_level(value: float) -> float:
    """Allow both decimal (0.35) and cents (35) inputs."""
    return (value / 100.0) if value > 1 else value


def evaluate_strategy(points: list[PricePoint]) -> dict[str, object]:
    """Simulate: first outcome that touches ENTRY, then TP/SL on that outcome."""
    entry_level = _normalize_price_level(float(ENTRY_PRICE))
    tp_level = _normalize_price_level(float(TP_PRICE))
    sl_level = _normalize_price_level(float(SL_PRICE))

    by_outcome: dict[str, list[PricePoint]] = {}
    by_ts: dict[int, dict[str, float]] = {}
    for p in points:
        by_outcome.setdefault(p.outcome, []).append(p)
        by_ts.setdefault(p.ts_ms, {})[p.outcome] = p.price
    for outcome in by_outcome:
        by_outcome[outcome].sort(key=lambda x: x.ts_ms)

    # ENTRY rule: first upward crossing of ENTRY level (prev < entry <= curr)
    entry_ts: int | None = None
    entry_outcome: str | None = None
    entry_hit_price: float | None = None
    entry_index: int | None = None
    for outcome, series in by_outcome.items():
        prev = None
        for idx, p in enumerate(series):
            if prev is not None and prev < entry_level <= p.price:
                if entry_ts is None or p.ts_ms < entry_ts or (p.ts_ms == entry_ts and p.price > (entry_hit_price or -1)):
                    entry_ts = p.ts_ms
                    entry_outcome = outcome
                    entry_hit_price = float(p.price)
                    entry_index = idx
                break
            prev = p.price

    if entry_ts is None or entry_outcome is None or entry_index is None:
        return {
            "entry_level": entry_level,
            "tp_level": tp_level,
            "sl_level": sl_level,
            "entry_ts_ms": None,
            "entry_outcome": None,
            "entry_hit_price": None,
            "exit_reason": "no_entry",
            "exit_ts_ms": None,
            "exit_level": None,
            "exit_hit_price": None,
            "pnl_pct": None,
        }

    outcome_series = by_outcome.get(entry_outcome, [])
    exit_reason = "open"
    exit_ts_ms: int | None = None
    exit_level: float | None = None
    exit_hit_price: float | None = None
    exit_bid_price: float | None = None  # actual bid at exit snapshot

    # EXIT rule:
    # - TP on upward crossing (prev < tp <= curr)  — trigger based on ASK
    # - SL on downward crossing (prev > sl >= curr) — trigger based on ASK
    prev_price = outcome_series[entry_index].price
    for p in outcome_series[entry_index + 1:]:
        if prev_price < tp_level <= p.price:
            exit_reason = "tp"
            exit_ts_ms = p.ts_ms
            exit_level = tp_level
            exit_hit_price = p.price
            exit_bid_price = p.bid  # may be None for legacy CSVs
            break
        if prev_price > sl_level >= p.price:
            exit_reason = "sl"
            exit_ts_ms = p.ts_ms
            exit_level = sl_level
            exit_hit_price = p.price
            exit_bid_price = p.bid
            break
        prev_price = p.price

    # P&L uses BID price when available (realistic sell execution).
    # Falls back to the exact level price for legacy CSVs without bid column.
    pnl_pct = None
    if exit_reason == "tp":
        effective_exit = exit_bid_price if exit_bid_price is not None else tp_level
        pnl_pct = ((effective_exit - entry_level) / entry_level) * 100.0
    elif exit_reason == "sl":
        effective_exit = exit_bid_price if exit_bid_price is not None else sl_level
        pnl_pct = ((effective_exit - entry_level) / entry_level) * 100.0

    return {
        "entry_level": entry_level,
        "tp_level": tp_level,
        "sl_level": sl_level,
        "entry_ts_ms": entry_ts,
        "entry_outcome": entry_outcome,
        "entry_hit_price": entry_hit_price,
        "exit_reason": exit_reason,
        "exit_ts_ms": exit_ts_ms,
        "exit_level": exit_level,
        "exit_hit_price": exit_hit_price,
        "pnl_pct": pnl_pct,
    }


def render_svg(points: list[PricePoint], market_name: str, strategy: dict[str, object]) -> str:
    width, height = 1200, 520
    margin_left, margin_right, margin_top, margin_bottom = 70, 30, 50, 45
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    min_ts = points[0].ts_ms
    max_ts = points[-1].ts_ms
    if max_ts == min_ts:
        max_ts += 1

    def x_scale(ts_ms: int) -> float:
        ratio = (ts_ms - min_ts) / (max_ts - min_ts)
        return margin_left + ratio * plot_w

    def y_scale(price: float) -> float:
        p = min(1.0, max(0.0, price))
        return margin_top + (1.0 - p) * plot_h

    by_outcome: dict[str, list[PricePoint]] = {}
    for p in points:
        by_outcome.setdefault(p.outcome, []).append(p)

    palette = ["#3b82f6", "#ef4444", "#22c55e", "#f59e0b"]
    outcome_colors: dict[str, str] = {}
    for idx, outcome in enumerate(sorted(by_outcome.keys())):
        outcome_colors[outcome] = palette[idx % len(palette)]

    lines: list[str] = []
    hover_dots: list[str] = []
    for outcome in sorted(by_outcome.keys()):
        series = by_outcome[outcome]
        pts = [(x_scale(p.ts_ms), y_scale(p.price)) for p in series]
        lines.append(_polyline(pts, outcome_colors[outcome]))
        for p in series:
            x = x_scale(p.ts_ms)
            y = y_scale(p.price)
            elapsed_s = (p.ts_ms - min_ts) / 1000.0
            tip = html.escape(f"{outcome} | t={elapsed_s:.1f}s | price={p.price:.3f}")
            hover_dots.append(
                f'<circle class="hover-point" data-tip="{tip}" cx="{x:.2f}" cy="{y:.2f}" r="5.0" fill="{outcome_colors[outcome]}" fill-opacity="0.02" stroke="none" pointer-events="all" />'
            )

    # Strategy markers: ENTRY on first touched outcome; EXIT on TP/SL level
    strategy_markers: list[str] = []
    entry_ts = strategy.get("entry_ts_ms")
    entry_outcome = strategy.get("entry_outcome")
    entry_hit_price = strategy.get("entry_hit_price")
    exit_reason = strategy.get("exit_reason")
    exit_ts = strategy.get("exit_ts_ms")
    exit_hit_price = strategy.get("exit_hit_price")
    if entry_ts is not None and isinstance(entry_hit_price, float):
        ex = x_scale(int(entry_ts))
        ey = y_scale(entry_hit_price)
        e_tip = html.escape(
            f"ENTRY | {entry_outcome} | t={(int(entry_ts)-min_ts)/1000.0:.1f}s | price={entry_hit_price:.3f}"
        )
        strategy_markers.append(
            f'<circle class="hover-point" data-tip="{e_tip}" cx="{ex:.2f}" cy="{ey:.2f}" r="6.0" fill="#fbbf24" stroke="#111" stroke-width="1.2" />'
        )
        strategy_markers.append(
            f'<text x="{ex+7:.2f}" y="{ey-7:.2f}" fill="#fbbf24" font-size="11">ENTRY</text>'
        )

    if exit_reason in ("tp", "sl") and exit_ts is not None and isinstance(exit_hit_price, float):
        xx = x_scale(int(exit_ts))
        yy = y_scale(exit_hit_price)
        is_tp = exit_reason == "tp"
        m_color = "#22c55e" if is_tp else "#ef4444"
        label = "TP" if is_tp else "SL"
        tip = html.escape(
            f"{label} | t={(int(exit_ts)-min_ts)/1000.0:.1f}s | price={exit_hit_price:.3f}"
        )
        strategy_markers.append(
            f'<rect class="hover-point" data-tip="{tip}" x="{xx-5:.2f}" y="{yy-5:.2f}" width="10" height="10" fill="{m_color}" stroke="#111" stroke-width="1.0" />'
        )
        strategy_markers.append(
            f'<text x="{xx+7:.2f}" y="{yy-7:.2f}" fill="{m_color}" font-size="11">{label}</text>'
        )

    y_grid = []
    for tick_i in range(0, 11):
        tick = tick_i / 10.0
        y = y_scale(tick)
        y_grid.append(f'<line x1="{margin_left}" y1="{y:.2f}" x2="{width-margin_right}" y2="{y:.2f}" stroke="#2f2f2f" />')
        y_grid.append(f'<text x="{margin_left-8}" y="{y+4:.2f}" fill="#bdbdbd" font-size="11" text-anchor="end">{tick:.2f}</text>')

    x_grid = []
    total_seconds = max(1.0, (max_ts - min_ts) / 1000.0)
    tick_step = 30 if total_seconds > 180 else 20 if total_seconds > 120 else 10
    num_ticks = int(total_seconds // tick_step) + 1
    for idx in range(num_ticks + 1):
        sec = min(total_seconds, idx * tick_step)
        x = margin_left + (sec / total_seconds) * plot_w
        x_grid.append(f'<line x1="{x:.2f}" y1="{margin_top}" x2="{x:.2f}" y2="{height-margin_bottom}" stroke="#252525" />')
        x_grid.append(f'<text x="{x:.2f}" y="{height-margin_bottom+18}" fill="#bdbdbd" font-size="11" text-anchor="middle">{int(sec)}s</text>')

    # Guide verticali configurabili (elapsed time)
    x_guides = []
    for sec, color in ((GUIDE_X_RED_SECONDS, "#ef4444"), (GUIDE_X_GREEN_SECONDS, "#22c55e")):
        if sec is None:
            continue
        sec_val = float(sec)
        if sec_val < 0 or sec_val > total_seconds:
            continue
        x = margin_left + (sec_val / total_seconds) * plot_w
        x_guides.append(
            f'<line x1="{x:.2f}" y1="{margin_top}" x2="{x:.2f}" y2="{height-margin_bottom}" '
            f'stroke="{color}" stroke-width="1.8" stroke-dasharray="6 4" />'
        )
        x_guides.append(
            f'<text x="{x+4:.2f}" y="{margin_top+14}" fill="{color}" font-size="11">{int(sec_val)}s</text>'
        )

    # Guide orizzontali configurabili (price levels)
    y_guides = []
    for level, color in ((GUIDE_Y_RED_PRICE, "#ef4444"), (GUIDE_Y_GREEN_PRICE, "#22c55e")):
        if level is None:
            continue
        level_val = _normalize_price_level(float(level))
        if level_val < 0 or level_val > 1:
            continue
        y = y_scale(level_val)
        label = f"{int(round(level_val * 100))}c"
        y_guides.append(
            f'<line x1="{margin_left}" y1="{y:.2f}" x2="{width-margin_right}" y2="{y:.2f}" '
            f'stroke="{color}" stroke-width="1.8" stroke-dasharray="6 4" />'
        )
        y_guides.append(
            f'<text x="{width-margin_right-4}" y="{y-6:.2f}" fill="{color}" font-size="11" text-anchor="end">{label}</text>'
        )

    legend_items = []
    lx = margin_left
    ly = 24
    for outcome in sorted(by_outcome.keys()):
        color = outcome_colors[outcome]
        legend_items.append(f'<rect x="{lx}" y="{ly-10}" width="12" height="12" fill="{color}" />')
        legend_items.append(f'<text x="{lx+18}" y="{ly}" fill="#e6e6e6" font-size="12">{html.escape(outcome)}</text>')
        lx += 120

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect x="0" y="0" width="{width}" height="{height}" fill="#111111"/>
  {"".join(legend_items)}
  <rect x="{margin_left}" y="{margin_top}" width="{plot_w}" height="{plot_h}" fill="#171717" stroke="#333333"/>
  {"".join(x_grid)}
  {"".join(y_grid)}
  {"".join(x_guides)}
  {"".join(y_guides)}
  {"".join(lines)}
  {"".join(strategy_markers)}
  {"".join(hover_dots)}
  <text x="{width/2:.2f}" y="{height-8}" fill="#bdbdbd" font-size="12" text-anchor="middle">Elapsed time (seconds)</text>
  <text x="18" y="{height/2:.2f}" fill="#bdbdbd" font-size="12" transform="rotate(-90, 18, {height/2:.2f})" text-anchor="middle">Best ask price</text>
</svg>"""


def save_market_chart(points: list[PricePoint], source_csv: Path, out_dir: Path) -> ChartArtifact:
    market_name = points[0].market_name
    stem = source_csv.stem
    svg_file = out_dir / f"{stem}.svg"
    strategy = evaluate_strategy(points)
    svg_file.write_text(render_svg(points, market_name, strategy), encoding="utf-8")

    counts: dict[str, int] = {}
    for p in points:
        counts[p.outcome] = counts.get(p.outcome, 0) + 1

    return ChartArtifact(
        market_name=market_name,
        csv_file=source_csv,
        svg_file=svg_file,
        samples_per_outcome=counts,
        first_ts_ms=points[0].ts_ms,
        strategy=strategy,
    )


def write_html_report(artifacts: list[ChartArtifact], report_path: Path, charts_dir: Path, title: str = "Price Charts Report") -> None:
    entry_level = _normalize_price_level(float(ENTRY_PRICE))
    tp_level = _normalize_price_level(float(TP_PRICE))
    sl_level = _normalize_price_level(float(SL_PRICE))
    entry_c = int(round(entry_level * 100))
    tp_c = int(round(tp_level * 100))
    sl_c = int(round(sl_level * 100))
    tp_pnl_pct = ((tp_level - entry_level) / entry_level) * 100.0
    sl_pnl_pct = ((sl_level - entry_level) / entry_level) * 100.0

    tp_count = sum(1 for a in artifacts if a.strategy.get("exit_reason") == "tp")
    sl_count = sum(1 for a in artifacts if a.strategy.get("exit_reason") == "sl")
    open_count = sum(1 for a in artifacts if a.strategy.get("exit_reason") == "open")
    no_entry_count = sum(1 for a in artifacts if a.strategy.get("exit_reason") == "no_entry")
    decided_count = tp_count + sl_count
    total_pnl_pct = tp_count * tp_pnl_pct + sl_count * sl_pnl_pct
    avg_pnl_pct = (total_pnl_pct / decided_count) if decided_count else 0.0

    sections = []
    for idx, item in enumerate(artifacts, start=1):
        sample_parts = ", ".join(f"{k}: {v}" for k, v in sorted(item.samples_per_outcome.items()))
        svg_inline = item.svg_file.read_text(encoding="utf-8")
        strategy = item.strategy
        reason = strategy.get("exit_reason")
        if reason == "tp":
            strategy_line = f"ENTRY on {strategy.get('entry_outcome')} -> TP"
        elif reason == "sl":
            strategy_line = f"ENTRY on {strategy.get('entry_outcome')} -> SL"
        elif reason == "open":
            strategy_line = f"ENTRY on {strategy.get('entry_outcome')} -> still open"
        else:
            strategy_line = "No entry hit"
        sections.append(
            f"""
<section class="card">
  <h2>{idx}. {html.escape(item.market_name)}</h2>
  <p><b>CSV:</b> <code>{html.escape(item.csv_file.name)}</code> | <b>Samples:</b> {html.escape(sample_parts)}</p>
  <p><b>Strategy:</b> {html.escape(strategy_line)}</p>
  <div class="chart-wrap">{svg_inline}</div>
</section>
"""
        )

    content = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>{html.escape(title)}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 0; background: #111; color: #eee; }}
    .page {{ width: min(1280px, calc(100vw - 32px)); margin: 20px auto 28px; }}
    h1 {{ margin: 0 0 8px 0; text-align: center; }}
    .meta {{ color: #bbb; margin-bottom: 18px; text-align: center; }}
    .summary {{ background: #171717; border: 1px solid #2f2f2f; border-radius: 10px; padding: 14px 16px; margin: 0 auto 18px; }}
    .summary-grid {{ display: grid; grid-template-columns: 165px 1fr 165px 1fr; gap: 10px 24px; align-items: center; }}
    .summary-label {{ color: #b8b8b8; font-weight: 600; text-align: right; }}
    .summary-value {{ color: #f3f3f3; font-variant-numeric: tabular-nums; letter-spacing: 0.1px; }}
    @media (max-width: 980px) {{
      .summary-grid {{ grid-template-columns: 180px 1fr; }}
    }}
    .card {{ background: #1a1a1a; border: 1px solid #2b2b2b; border-radius: 10px; padding: 14px; margin-bottom: 16px; }}
    code {{ background: #0f0f0f; padding: 2px 5px; border-radius: 4px; }}
    .chart-wrap svg {{ width: 100%; max-width: 1200px; height: auto; border-radius: 8px; border: 1px solid #333; display: block; }}
    .chart-tooltip {{
      position: fixed;
      z-index: 9999;
      pointer-events: none;
      background: rgba(10, 10, 10, 0.95);
      color: #f3f3f3;
      border: 1px solid #3a3a3a;
      border-radius: 6px;
      padding: 6px 8px;
      font-size: 12px;
      display: none;
      white-space: nowrap;
    }}
  </style>
</head>
<body>
  <main class="page">
  <h1>{html.escape(title)}</h1>
  <div class="meta">Charts dir: <code>{html.escape(charts_dir.as_posix())}</code> | Markets: {len(artifacts)}</div>
  <section class="summary">
    <div class="summary-grid">
      <div class="summary-label">Strategy levels</div>
      <div class="summary-value">Entry {entry_c}c, TP {tp_c}c, SL {sl_c}c</div>
      <div class="summary-label">Counts</div>
      <div class="summary-value">TP {tp_count} | SL {sl_count} | Open {open_count} | No Entry {no_entry_count}</div>
      <div class="summary-label">P/L per trade</div>
      <div class="summary-value">TP {tp_pnl_pct:+.2f}% | SL {sl_pnl_pct:+.2f}%</div>
      <div class="summary-label">Total P/L</div>
      <div class="summary-value">{total_pnl_pct:+.2f}%</div>
      <div class="summary-label">Average P/L</div>
      <div class="summary-value">{avg_pnl_pct:+.2f}% over {decided_count} trades</div>
      <div class="summary-label"></div>
      <div class="summary-value"></div>
    </div>
  </section>
  {"".join(sections)}
  <div id="chart-tooltip" class="chart-tooltip"></div>
  <script>
    (() => {{
      const tip = document.getElementById('chart-tooltip');
      if (!tip) return;
      const show = (evt, text) => {{
        tip.textContent = text || '';
        tip.style.display = 'block';
        tip.style.left = (evt.clientX + 14) + 'px';
        tip.style.top = (evt.clientY + 14) + 'px';
      }};
      const move = (evt) => {{
        if (tip.style.display === 'block') {{
          tip.style.left = (evt.clientX + 14) + 'px';
          tip.style.top = (evt.clientY + 14) + 'px';
        }}
      }};
      const hide = () => {{
        tip.style.display = 'none';
      }};
      document.querySelectorAll('.hover-point').forEach((el) => {{
        el.addEventListener('mouseenter', (evt) => show(evt, el.getAttribute('data-tip')));
        el.addEventListener('mousemove', move);
        el.addEventListener('mouseleave', hide);
      }});
    }})();
  </script>
  </main>
</body>
</html>
"""
    report_path.write_text(content, encoding="utf-8")


def main() -> int:
    import re

    args = parse_args()
    prices_dir = Path(args.prices_dir)
    if not prices_dir.exists():
        print(f"❌ Prices directory not found: {prices_dir}")
        return 1

    # CSV files live in the csv/ subfolder; fall back to prices_dir for legacy data
    csv_dir = prices_dir / "csv"
    if not csv_dir.exists() or not any(csv_dir.glob(args.pattern)):
        csv_dir = prices_dir

    out_dir = Path(args.out_dir) if args.out_dir else prices_dir / "charts"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(csv_dir.glob(args.pattern))
    if not csv_files:
        print(f"⚠️ No files matching '{args.pattern}' in {csv_dir}")
        return 0

    artifacts: list[ChartArtifact] = []
    skipped = 0
    for csv_file in csv_files:
        try:
            points = load_price_csv(csv_file)
            artifacts.append(save_market_chart(points, csv_file, out_dir))
        except Exception as exc:
            skipped += 1
            print(f"⚠️ Skipping {csv_file.name}: {exc}")

    if not artifacts:
        print("❌ No charts generated.")
        return 1

    artifacts.sort(key=lambda a: a.first_ts_ms)

    # Total report — all markets ever recorded
    total_report = prices_dir / "report_all.html"
    write_html_report(artifacts, total_report, out_dir, title="Price Charts — All Markets")
    print(f"✅ Total report  ({len(artifacts)} markets): {total_report}")

    # Session report — group CSVs by session_time prefix (YYYY-MM-DD_HH-MM-SS)
    # All CSVs from a single bot run share the same session_time prefix, so the
    # most recent group is the currently-running (or last) session.
    session_pat = re.compile(r"^prices_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})_")
    session_groups: dict[str, list[ChartArtifact]] = {}
    for a in artifacts:
        m = session_pat.match(a.csv_file.name)
        if m:
            session_groups.setdefault(m.group(1), []).append(a)

    if session_groups:
        latest_key = sorted(session_groups.keys())[-1]
        session_artifacts = session_groups[latest_key]
        session_report = prices_dir / f"report_session_{latest_key}.html"
        write_html_report(
            session_artifacts,
            session_report,
            out_dir,
            title=f"Price Charts — Session {latest_key}",
        )
        print(f"📊 Session report ({len(session_artifacts)} markets, session {latest_key}): {session_report}")

    if skipped:
        print(f"⚠️ Skipped files: {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
