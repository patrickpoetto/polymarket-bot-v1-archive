#!/usr/bin/env python3
"""
Generate bid/ask charts (Up/Down) from per-market CSV snapshots.

Expected CSV columns are compatible with shared.price_logger:
    timestamp_iso,timestamp_unix_ms,condition_id,market_name,outcome,token_id,ask,bid,time_left_seconds
"""
from __future__ import annotations

import argparse
import csv
import html
from dataclasses import dataclass
from pathlib import Path

# Optional guide lines (same style as generate_price_charts.py)
GUIDE_X_RED_SECONDS = 120
GUIDE_X_GREEN_SECONDS = None
GUIDE_Y_RED_PRICE = 0.20
GUIDE_Y_GREEN_PRICE = 0.70


@dataclass
class BookPoint:
    ts_ms: int
    outcome: str
    ask: float | None
    bid: float | None
    market_name: str


@dataclass
class ChartArtifact:
    market_name: str
    csv_file: Path
    svg_file: Path
    first_ts_ms: int
    sample_count: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate bid/ask market charts report.")
    p.add_argument("--prices-dir", required=True, help="Directory that contains csv/ and charts/")
    p.add_argument("--pattern", default="prices_*.csv")
    p.add_argument("--report-name", default="report_all.html")
    return p.parse_args()


def _parse_f(v: str | None) -> float | None:
    if v is None:
        return None
    v = str(v).strip()
    if not v:
        return None
    try:
        return float(v)
    except ValueError:
        return None


def load_points(path: Path) -> list[BookPoint]:
    pts: list[BookPoint] = []
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                ts_ms = int(float(row.get("timestamp_unix_ms", "")))
            except (TypeError, ValueError):
                continue
            outcome = (row.get("outcome") or "").strip()
            if not outcome:
                continue
            pts.append(
                BookPoint(
                    ts_ms=ts_ms,
                    outcome=outcome,
                    ask=_parse_f(row.get("ask")),
                    bid=_parse_f(row.get("bid")),
                    market_name=(row.get("market_name") or "").strip(),
                )
            )
    pts.sort(key=lambda x: x.ts_ms)
    return pts


def _polyline(points: list[tuple[float, float]], color: str, dashed: bool = False) -> str:
    if not points:
        return ""
    coords = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    dash = ' stroke-dasharray="6 4"' if dashed else ""
    return f'<polyline fill="none" stroke="{color}" stroke-width="2"{dash} points="{coords}" />'


def render_svg(points: list[BookPoint], market_name: str) -> str:
    width, height = 1200, 520
    ml, mr, mt, mb = 70, 30, 50, 45
    plot_w = width - ml - mr
    plot_h = height - mt - mb

    min_ts = points[0].ts_ms
    max_ts = points[-1].ts_ms
    if max_ts <= min_ts:
        max_ts = min_ts + 1

    def x_scale(ts: int) -> float:
        return ml + ((ts - min_ts) / (max_ts - min_ts)) * plot_w

    def y_scale(price: float) -> float:
        p = min(1.0, max(0.0, price))
        return mt + (1.0 - p) * plot_h

    series: dict[tuple[str, str], list[tuple[float, float]]] = {}
    for p in points:
        ox = p.outcome.strip().title()
        if p.ask is not None:
            series.setdefault((ox, "ask"), []).append((x_scale(p.ts_ms), y_scale(p.ask)))
        if p.bid is not None:
            series.setdefault((ox, "bid"), []).append((x_scale(p.ts_ms), y_scale(p.bid)))

    colors = {
        ("Up", "ask"): "#3b82f6",
        ("Up", "bid"): "#93c5fd",
        ("Down", "ask"): "#ef4444",
        ("Down", "bid"): "#fca5a5",
    }
    ordered_keys = [("Up", "ask"), ("Up", "bid"), ("Down", "ask"), ("Down", "bid")]

    lines: list[str] = []
    hover_points: list[str] = []
    for k in ordered_keys:
        pts = series.get(k, [])
        if pts:
            lines.append(_polyline(pts, colors[k], dashed=(k[1] == "bid")))

    # Lightweight interactivity: invisible hit-points with tooltip payload.
    for p in points:
        elapsed_s = (p.ts_ms - min_ts) / 1000.0
        outcome = p.outcome.strip().title()
        if p.ask is not None:
            x = x_scale(p.ts_ms)
            y = y_scale(p.ask)
            tip = html.escape(f"{outcome} ask | t={elapsed_s:.1f}s | price={p.ask:.3f}")
            hover_points.append(
                f'<circle class="hover-point" data-tip="{tip}" cx="{x:.2f}" cy="{y:.2f}" r="5.0" fill="{colors.get((outcome, "ask"), "#888")}" fill-opacity="0.02" stroke="none" pointer-events="all" />'
            )
        if p.bid is not None:
            x = x_scale(p.ts_ms)
            y = y_scale(p.bid)
            tip = html.escape(f"{outcome} bid | t={elapsed_s:.1f}s | price={p.bid:.3f}")
            hover_points.append(
                f'<circle class="hover-point" data-tip="{tip}" cx="{x:.2f}" cy="{y:.2f}" r="5.0" fill="{colors.get((outcome, "bid"), "#888")}" fill-opacity="0.02" stroke="none" pointer-events="all" />'
            )

    y_grid: list[str] = []
    for i in range(11):
        tick = i / 10.0
        y = y_scale(tick)
        y_grid.append(f'<line x1="{ml}" y1="{y:.2f}" x2="{width-mr}" y2="{y:.2f}" stroke="#2f2f2f" />')
        y_grid.append(f'<text x="{ml-8}" y="{y+4:.2f}" fill="#bdbdbd" font-size="11" text-anchor="end">{tick:.2f}</text>')

    x_grid: list[str] = []
    total_sec = max(1.0, (max_ts - min_ts) / 1000.0)
    tick_step = 60 if total_sec > 600 else 30 if total_sec > 180 else 10
    n_ticks = int(total_sec // tick_step) + 1
    for i in range(n_ticks + 1):
        sec = min(total_sec, i * tick_step)
        x = ml + (sec / total_sec) * plot_w
        x_grid.append(f'<line x1="{x:.2f}" y1="{mt}" x2="{x:.2f}" y2="{height-mb}" stroke="#252525" />')
        x_grid.append(f'<text x="{x:.2f}" y="{height-mb+18}" fill="#bdbdbd" font-size="11" text-anchor="middle">{int(sec)}s</text>')

    # Optional vertical guide lines (seconds from market start)
    x_guides: list[str] = []
    for sec, color in ((GUIDE_X_RED_SECONDS, "#ef4444"), (GUIDE_X_GREEN_SECONDS, "#22c55e")):
        if sec is None:
            continue
        sec_val = float(sec)
        if sec_val < 0 or sec_val > total_sec:
            continue
        x = ml + (sec_val / total_sec) * plot_w
        x_guides.append(
            f'<line x1="{x:.2f}" y1="{mt}" x2="{x:.2f}" y2="{height-mb}" '
            f'stroke="{color}" stroke-width="1.8" stroke-dasharray="6 4" />'
        )
        x_guides.append(
            f'<text x="{x+4:.2f}" y="{mt+14}" fill="{color}" font-size="11">{int(sec_val)}s</text>'
        )

    # Optional horizontal guide lines (price level, supports cents if >1)
    y_guides: list[str] = []
    for level, color in ((GUIDE_Y_RED_PRICE, "#ef4444"), (GUIDE_Y_GREEN_PRICE, "#22c55e")):
        if level is None:
            continue
        level_val = float(level)
        if level_val > 1:
            level_val = level_val / 100.0
        if level_val < 0 or level_val > 1:
            continue
        y = y_scale(level_val)
        label = f"{int(round(level_val * 100))}c"
        y_guides.append(
            f'<line x1="{ml}" y1="{y:.2f}" x2="{width-mr}" y2="{y:.2f}" '
            f'stroke="{color}" stroke-width="1.8" stroke-dasharray="6 4" />'
        )
        y_guides.append(
            f'<text x="{width-mr-4}" y="{y-6:.2f}" fill="{color}" font-size="11" text-anchor="end">{label}</text>'
        )

    legend_items = []
    legend_labels = [
        ("Up ask", colors[("Up", "ask")], False),
        ("Up bid", colors[("Up", "bid")], True),
        ("Down ask", colors[("Down", "ask")], False),
        ("Down bid", colors[("Down", "bid")], True),
    ]
    lx = ml
    ly = 24
    for label, color, dashed in legend_labels:
        dash = ' stroke-dasharray="6 4"' if dashed else ""
        legend_items.append(
            f'<line x1="{lx}" y1="{ly-5}" x2="{lx+16}" y2="{ly-5}" stroke="{color}" stroke-width="3"{dash} />'
        )
        legend_items.append(f'<text x="{lx+22}" y="{ly}" fill="#e6e6e6" font-size="12">{html.escape(label)}</text>')
        lx += 165

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect x="0" y="0" width="{width}" height="{height}" fill="#111111"/>
  {"".join(legend_items)}
  <rect x="{ml}" y="{mt}" width="{plot_w}" height="{plot_h}" fill="#171717" stroke="#333333"/>
  {"".join(x_grid)}
  {"".join(y_grid)}
  {"".join(x_guides)}
  {"".join(y_guides)}
  {"".join(lines)}
  {"".join(hover_points)}
  <text x="{width/2:.2f}" y="{height-8}" fill="#bdbdbd" font-size="12" text-anchor="middle">Elapsed time (seconds)</text>
  <text x="18" y="{height/2:.2f}" fill="#bdbdbd" font-size="12" transform="rotate(-90, 18, {height/2:.2f})" text-anchor="middle">Price</text>
</svg>"""


def save_chart(points: list[BookPoint], src_csv: Path, charts_dir: Path) -> ChartArtifact:
    svg_path = charts_dir / f"{src_csv.stem}.svg"
    market_name = points[0].market_name if points else src_csv.stem
    svg_path.write_text(render_svg(points, market_name), encoding="utf-8")
    return ChartArtifact(
        market_name=market_name,
        csv_file=src_csv,
        svg_file=svg_path,
        first_ts_ms=points[0].ts_ms,
        sample_count=len(points),
    )


def write_report(artifacts: list[ChartArtifact], report_path: Path, title: str) -> None:
    # Enforce deterministic chronological ordering (oldest -> newest).
    artifacts = sorted(artifacts, key=lambda a: a.first_ts_ms)
    sections: list[str] = []
    for i, a in enumerate(artifacts, start=1):
        svg = a.svg_file.read_text(encoding="utf-8")
        sections.append(
            f"""
<section class="card">
  <h2>{i}. {html.escape(a.market_name)}</h2>
  <p><b>CSV:</b> <code>{html.escape(a.csv_file.name)}</code> | <b>Samples:</b> {a.sample_count}</p>
  <div class="chart-wrap">{svg}</div>
</section>
"""
        )

    doc = f"""<!doctype html>
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
    <div class="meta">Markets: {len(artifacts)}</div>
    {"".join(sections)}
    <div id="chart-tooltip" class="chart-tooltip"></div>
  </main>
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
</body>
</html>"""
    report_path.write_text(doc, encoding="utf-8")


def main() -> int:
    args = parse_args()
    prices_dir = Path(args.prices_dir)
    csv_dir = prices_dir / "csv"
    charts_dir = prices_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    if not csv_dir.exists():
        print(f"⚠️ CSV dir not found: {csv_dir}")
        return 0

    csv_files = sorted(csv_dir.glob(args.pattern))
    if not csv_files:
        print(f"⚠️ No CSV files in {csv_dir}")
        return 0

    artifacts: list[ChartArtifact] = []
    skipped = 0
    for csv_file in csv_files:
        try:
            pts = load_points(csv_file)
            if not pts:
                continue
            artifacts.append(save_chart(pts, csv_file, charts_dir))
        except Exception as exc:
            skipped += 1
            print(f"⚠️ skip {csv_file.name}: {exc}")

    if not artifacts:
        print("⚠️ no charts generated")
        return 0

    artifacts.sort(key=lambda x: x.first_ts_ms)
    title = f"Market Book Charts — {prices_dir.name}m"
    report_path = prices_dir / args.report_name
    write_report(artifacts, report_path, title=title)
    print(f"✅ report: {report_path} | charts: {len(artifacts)} | skipped: {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
