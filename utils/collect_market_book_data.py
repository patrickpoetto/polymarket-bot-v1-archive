#!/usr/bin/env python3
"""
Standalone market book collector for BTC Up/Down 5m and 15m markets.

Writes CSV snapshots under:
  prices/5/csv
  prices/15/csv

And periodically regenerates HTML reports with:
  utils/generate_book_charts.py
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import requests
from bs4 import BeautifulSoup

# Ensure project root is importable when script is run directly.
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from shared.price_logger import PriceLogger

logger = logging.getLogger("book_collector")

POLYMARKET_BASE = "https://polymarket.com"
CLOB_BOOK_API = "https://clob.polymarket.com/book"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
}
NY_TZ = ZoneInfo("America/New_York")
MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}


@dataclass
class MarketConfig:
    label: str
    interval: int
    slug_pattern: str
    prices_dir: Path
    min_span_seconds: float


@dataclass
class MarketInfo:
    condition_id: str
    question: str
    outcomes: list[str]
    token_ids: list[str]
    start_ts: int
    end_ts: int


def setup_logging(
    verbose: bool,
    quiet: bool = False,
    log_file: str = "",
) -> None:
    if quiet and not verbose:
        level = logging.WARNING
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)

    if log_file:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        # With --quiet, still record INFO+ on file (heartbeats, market switches).
        fh.setLevel(logging.DEBUG if verbose else logging.INFO)
        fh.setFormatter(fmt)
        root.addHandler(fh)

    # Avoid flooding Cursor's integrated terminal (scrollback can reach GB).
    if not quiet or verbose:
        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(level)
        sh.setFormatter(fmt)
        root.addHandler(sh)
    elif not log_file:
        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(logging.WARNING)
        sh.setFormatter(fmt)
        root.addHandler(sh)

    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect BTC 5m/15m bid-ask snapshots in standalone mode.")
    p.add_argument("--poll-seconds", type=float, default=1.0)
    p.add_argument(
        "--market-probe-seconds",
        type=float,
        default=20.0,
        help="How often to re-discover market metadata (slug/condition).",
    )
    p.add_argument(
        "--report-every-seconds",
        type=float,
        default=120.0,
        help="Regenerate HTML charts (heavy). Use 0 with --no-reports for 24/7 collection.",
    )
    p.add_argument("--base-dir", default="prices")
    p.add_argument(
        "--bootstrap-only",
        action="store_true",
        help="Run migration + historical report generation, then exit.",
    )
    p.add_argument(
        "--bootstrap-reports",
        action="store_true",
        help="On startup, run reports for every day/* folder (slow, high RAM). Default: migrate only.",
    )
    p.add_argument(
        "--no-reports",
        action="store_true",
        help="Never spawn generate_book_charts.py (recommended for H24 / low memory).",
    )
    p.add_argument(
        "--log-file",
        default="",
        help="Append logs to this file; console stays quiet unless --verbose.",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Console WARNING only (use with --log-file for long runs).",
    )
    p.add_argument(
        "--heartbeat-seconds",
        type=float,
        default=300.0,
        help="Log alive status to --log-file every N seconds (0=off). Default 300.",
    )
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def _slug_from_start(start_ts: int, pattern: str) -> str:
    return pattern.replace("{ts}", str(start_ts))


def _parse_market(html_text: str, interval: int, expected_slug: str) -> Optional[MarketInfo]:
    soup = BeautifulSoup(html_text, "html.parser")
    script = soup.find("script", {"id": "__NEXT_DATA__", "type": "application/json"})
    if not script or not script.string:
        return None
    try:
        data = json.loads(script.string)
    except json.JSONDecodeError:
        return None

    queries = (
        data.get("props", {})
        .get("pageProps", {})
        .get("dehydratedState", {})
        .get("queries", [])
    )
    raw = None
    for q in queries:
        state_data = q.get("state", {}).get("data", {})
        if "markets" in state_data and state_data["markets"]:
            raw = state_data["markets"][0]
            break
        if "conditionId" in state_data:
            raw = state_data
            break
    if not raw:
        return None

    condition_id = str(raw.get("conditionId") or "")
    outcomes_raw = raw.get("outcomes", [])
    token_ids_raw = raw.get("clobTokenIds", [])
    if isinstance(outcomes_raw, str):
        outcomes_raw = json.loads(outcomes_raw)
    if isinstance(token_ids_raw, str):
        token_ids_raw = json.loads(token_ids_raw)

    end_ts = 0
    end_date = raw.get("endDate")
    if end_date:
        try:
            end_ts = int(datetime.fromisoformat(end_date.replace("Z", "+00:00")).timestamp())
        except Exception:
            end_ts = 0
    if end_ts <= 0:
        return None

    slug = str(raw.get("groupItemTitle") or raw.get("slug") or expected_slug)
    m = re.search(r"(\d{10})", slug) or re.search(r"(\d{10})", expected_slug)
    start_ts = int(m.group(1)) if m else (end_ts - interval)

    return MarketInfo(
        condition_id=condition_id,
        question=str(raw.get("question") or ""),
        outcomes=list(outcomes_raw),
        token_ids=list(token_ids_raw),
        start_ts=start_ts,
        end_ts=end_ts,
    )


def fetch_market_by_start(start_ts: int, cfg: MarketConfig, session: requests.Session) -> Optional[MarketInfo]:
    slug = _slug_from_start(start_ts, cfg.slug_pattern)
    url = f"{POLYMARKET_BASE}/event/{slug}"
    try:
        resp = session.get(url, headers=HEADERS, timeout=12)
        if resp.status_code != 200:
            return None
        return _parse_market(resp.text, cfg.interval, slug)
    except Exception:
        return None


def find_current_market(cfg: MarketConfig, session: requests.Session) -> Optional[MarketInfo]:
    now = int(time.time())
    start = (now // cfg.interval) * cfg.interval
    candidates = [start, start - cfg.interval, start + cfg.interval]
    infos: list[MarketInfo] = []
    for st in candidates:
        info = fetch_market_by_start(st, cfg, session)
        if info and info.condition_id:
            infos.append(info)
    if not infos:
        return None

    # Prefer currently running market; otherwise nearest upcoming.
    running = [x for x in infos if x.start_ts <= now < x.end_ts]
    if running:
        running.sort(key=lambda x: x.start_ts, reverse=True)
        return running[0]
    future = [x for x in infos if x.start_ts > now]
    if future:
        future.sort(key=lambda x: x.start_ts)
        return future[0]
    infos.sort(key=lambda x: x.start_ts, reverse=True)
    return infos[0]


def _best_prices_from_book(session: requests.Session, token_id: str) -> tuple[Optional[float], Optional[float]]:
    try:
        resp = session.get(CLOB_BOOK_API, params={"token_id": token_id}, timeout=5)
        resp.raise_for_status()
        book = resp.json()
    except Exception:
        return None, None

    asks = book.get("asks", []) or []
    bids = book.get("bids", []) or []
    best_ask = None
    best_bid = None
    try:
        best_ask = min((float(x.get("price")) for x in asks if x.get("price") is not None), default=None)
    except Exception:
        best_ask = None
    try:
        best_bid = max((float(x.get("price")) for x in bids if x.get("price") is not None), default=None)
    except Exception:
        best_bid = None
    return best_ask, best_bid


def collect_prices(market: MarketInfo, session: requests.Session) -> dict[str, dict]:
    snapshot: dict[str, dict] = {}
    pairs = list(zip(market.outcomes, market.token_ids))
    if not pairs:
        return snapshot
    workers = min(4, max(1, len(pairs)))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        fut_map = {
            ex.submit(_best_prices_from_book, session, token_id): (outcome, token_id)
            for outcome, token_id in pairs
        }
        for fut in as_completed(fut_map):
            outcome, token_id = fut_map[fut]
            ask, bid = fut.result()
            snapshot[outcome] = {
                "token_id": token_id,
                "ask": ask,
                "bid": bid,
            }
    return snapshot


def run_report(prices_dir: Path, verbose_report: bool = False) -> None:
    """Regenerate charts/report HTML for one day folder. Can use 100MB+ RAM if many CSVs."""
    cmd = [
        sys.executable,
        str(ROOT_DIR / "utils" / "generate_book_charts.py"),
        "--prices-dir",
        str(prices_dir),
    ]
    try:
        res = subprocess.run(
            cmd,
            cwd=str(ROOT_DIR),
            timeout=300,
            text=True,
            stdout=subprocess.PIPE if verbose_report else subprocess.DEVNULL,
            stderr=subprocess.PIPE if verbose_report else subprocess.DEVNULL,
        )
        if verbose_report and res.stdout and res.stdout.strip():
            logger.info("[report %s] %s", prices_dir.name, res.stdout.strip()[:200])
        if res.returncode != 0:
            logger.warning("[report %s] exit code %s", prices_dir.name, res.returncode)
    except subprocess.TimeoutExpired:
        logger.warning("Report generation timed out for %s", prices_dir)
    except Exception as exc:
        logger.warning("Report generation failed for %s: %s", prices_dir, exc)


def _day_key(now_ts: float) -> str:
    return datetime.fromtimestamp(now_ts, tz=NY_TZ).strftime("%Y-%m-%d")


def _day_dir(cfg: MarketConfig, day_key: str) -> Path:
    return cfg.prices_dir / "days" / day_key


def _market_label_from_market_name(market_name: str) -> str:
    rhs = market_name.split(" - ", 1)[1] if " - " in market_name else market_name
    safe = "".join(ch if ch.isalnum() else "_" for ch in rhs.strip())
    safe = "_".join(part for part in safe.split("_") if part)
    return safe or "unknown_market"


def _infer_day_key_from_market_name(market_name: str, fallback_ts: float) -> str:
    # Example market_name: "Bitcoin Up or Down - March 12, 4:15AM-4:30AM ET"
    m = re.search(r"-\s*([A-Za-z]+)\s+(\d{1,2}),", market_name)
    if m:
        month = MONTHS.get(m.group(1).lower())
        day = int(m.group(2))
        if month is not None:
            year = datetime.fromtimestamp(fallback_ts, tz=NY_TZ).year
            return f"{year:04d}-{month:02d}-{day:02d}"
    return _day_key(fallback_ts)


def _ensure_day_layout(cfg: MarketConfig, day_key: str) -> Path:
    d = _day_dir(cfg, day_key)
    (d / "csv").mkdir(parents=True, exist_ok=True)
    (d / "charts").mkdir(parents=True, exist_ok=True)
    return d


def _read_csv_market_meta(csv_file: Path) -> tuple[Optional[str], float]:
    """Return (market_name, first_ts_seconds)."""
    import csv as _csv

    first_ts = 0.0
    market_name: Optional[str] = None
    with csv_file.open(newline="", encoding="utf-8") as fh:
        r = _csv.DictReader(fh)
        for row in r:
            if market_name is None:
                val = (row.get("market_name") or "").strip()
                if val:
                    market_name = val
            raw = row.get("timestamp_unix_ms")
            if raw:
                try:
                    first_ts = int(float(raw)) / 1000.0
                    break
                except (TypeError, ValueError):
                    pass
    return market_name, first_ts


def _migrate_legacy_files(cfg: MarketConfig) -> None:
    """Move legacy files from prices/<tf>/csv|charts into daily folders."""
    legacy_csv = cfg.prices_dir / "csv"
    legacy_charts = cfg.prices_dir / "charts"
    moved_stems: dict[str, tuple[str, str]] = {}
    if legacy_csv.exists():
        for f in sorted(legacy_csv.glob("prices_*.csv")):
            market_name, first_ts = _read_csv_market_meta(f)
            if not market_name:
                continue
            day = _infer_day_key_from_market_name(market_name, first_ts or time.time())
            target_name = f"prices_{_market_label_from_market_name(market_name)}.csv"
            target = _ensure_day_layout(cfg, day) / "csv" / target_name
            if target.exists():
                # Keep existing canonical file; drop duplicate legacy copy.
                try:
                    f.unlink(missing_ok=True)
                except Exception:
                    pass
            else:
                f.replace(target)
            moved_stems[f.stem] = (day, Path(target_name).stem)
            # Move matching legacy chart with same stem, if present.
            old_svg = legacy_charts / f"{f.stem}.svg"
            if old_svg.exists():
                svg_target = _ensure_day_layout(cfg, day) / "charts" / f"{Path(target_name).stem}.svg"
                if svg_target.exists():
                    try:
                        old_svg.unlink(missing_ok=True)
                    except Exception:
                        pass
                else:
                    old_svg.replace(svg_target)
    if legacy_charts.exists():
        for f in sorted(legacy_charts.glob("prices_*.svg")):
            # Any leftover charts without matching CSV are likely stale; remove.
            try:
                f.unlink(missing_ok=True)
            except Exception:
                pass


def _normalize_day_files(cfg: MarketConfig) -> int:
    """Re-bucket/rename already migrated day files using market metadata."""
    days_root = cfg.prices_dir / "days"
    if not days_root.exists():
        return 0
    moved = 0
    for day_dir in sorted([p for p in days_root.iterdir() if p.is_dir()]):
        csv_dir = day_dir / "csv"
        charts_dir = day_dir / "charts"
        if not csv_dir.exists():
            continue
        for src_csv in sorted(csv_dir.glob("prices_*.csv")):
            market_name, first_ts = _read_csv_market_meta(src_csv)
            if not market_name:
                continue
            dst_day = _infer_day_key_from_market_name(market_name, first_ts or time.time())
            dst_name = f"prices_{_market_label_from_market_name(market_name)}.csv"
            dst_csv = _ensure_day_layout(cfg, dst_day) / "csv" / dst_name
            if src_csv != dst_csv:
                if dst_csv.exists():
                    # Keep canonical destination if already present.
                    try:
                        src_csv.unlink(missing_ok=True)
                    except Exception:
                        pass
                else:
                    dst_csv.parent.mkdir(parents=True, exist_ok=True)
                    src_csv.replace(dst_csv)
                moved += 1
            # Align matching chart file name/day with CSV stem/day.
            src_svg = charts_dir / f"{src_csv.stem}.svg"
            dst_svg = _ensure_day_layout(cfg, dst_day) / "charts" / f"{Path(dst_name).stem}.svg"
            if src_svg.exists() and src_svg != dst_svg:
                if dst_svg.exists():
                    try:
                        src_svg.unlink(missing_ok=True)
                    except Exception:
                        pass
                else:
                    dst_svg.parent.mkdir(parents=True, exist_ok=True)
                    src_svg.replace(dst_svg)
    return moved


def _csv_span_and_last_ts(csv_file: Path) -> tuple[float, float]:
    """Return (span_seconds, last_ts_seconds) from CSV timestamp_unix_ms."""
    import csv as _csv

    first_ms = None
    last_ms = None
    with csv_file.open(newline="", encoding="utf-8") as fh:
        r = _csv.DictReader(fh)
        for row in r:
            raw = row.get("timestamp_unix_ms")
            if not raw:
                continue
            try:
                ts = int(float(raw))
            except (TypeError, ValueError):
                continue
            if first_ms is None:
                first_ms = ts
            last_ms = ts
    if first_ms is None or last_ms is None:
        return 0.0, 0.0
    return max(0.0, (last_ms - first_ms) / 1000.0), (last_ms / 1000.0)


def _cleanup_incomplete_day_files(cfg: MarketConfig, day_dir: Path, now_ts: float) -> int:
    """Delete stale CSV/SVG whose data span is below required threshold."""
    csv_dir = day_dir / "csv"
    charts_dir = day_dir / "charts"
    removed = 0
    if not csv_dir.exists():
        return removed
    for f in sorted(csv_dir.glob("prices_*.csv")):
        span_s, last_ts_s = _csv_span_and_last_ts(f)
        if span_s >= cfg.min_span_seconds:
            continue
        # Keep currently-active file; prune only stale short files.
        if last_ts_s <= 0 or (now_ts - last_ts_s) < cfg.interval:
            continue
        svg = charts_dir / f"{f.stem}.svg"
        try:
            f.unlink(missing_ok=True)
            svg.unlink(missing_ok=True)
            removed += 1
        except Exception:
            pass
    return removed


def main() -> int:
    args = parse_args()
    setup_logging(args.verbose, quiet=args.quiet, log_file=args.log_file)

    no_reports = args.no_reports or args.report_every_seconds <= 0
    if no_reports:
        logger.info("Reports disabled (--no-reports or --report-every-seconds 0)")

    base = Path(args.base_dir)
    cfgs = [
        MarketConfig("5", 300, "btc-updown-5m-{ts}", base / "5", 270.0),
        MarketConfig("15", 900, "btc-updown-15m-{ts}", base / "15", 870.0),
    ]
    run_bootstrap_reports = args.bootstrap_reports or args.bootstrap_only
    for cfg in cfgs:
        cfg.prices_dir.mkdir(parents=True, exist_ok=True)
        _migrate_legacy_files(cfg)
        normalized = _normalize_day_files(cfg)
        if normalized:
            logger.info("[%sm] normalized %d migrated file(s)", cfg.label, normalized)
        days_root = cfg.prices_dir / "days"
        if days_root.exists():
            for day_dir in sorted([p for p in days_root.iterdir() if p.is_dir()]):
                removed = _cleanup_incomplete_day_files(cfg, day_dir, time.time())
                if removed:
                    logger.info("[%sm] bootstrap removed %d incomplete file(s) in %s", cfg.label, removed, day_dir.name)
                if run_bootstrap_reports and not no_reports:
                    run_report(day_dir, verbose_report=args.verbose)

    if args.bootstrap_only:
        logger.info("Bootstrap-only completed. Exiting.")
        return 0

    now = time.time()
    day_keys: dict[str, str] = {cfg.label: _day_key(now) for cfg in cfgs}
    active_day_dirs: dict[str, Path] = {
        cfg.label: _ensure_day_layout(cfg, day_keys[cfg.label]) for cfg in cfgs
    }

    loggers = {
        cfg.label: PriceLogger(logs_dir=str(active_day_dirs[cfg.label])) for cfg in cfgs
    }
    current_condition: dict[str, Optional[str]] = {cfg.label: None for cfg in cfgs}
    current_market: dict[str, Optional[MarketInfo]] = {cfg.label: None for cfg in cfgs}
    last_market_probe_ts: dict[str, float] = {cfg.label: 0.0 for cfg in cfgs}
    snapshot_count: dict[str, int] = {cfg.label: 0 for cfg in cfgs}
    last_report_snapshot_count: dict[str, int] = {cfg.label: -1 for cfg in cfgs}
    last_report_ts = 0.0
    last_heartbeat_ts = 0.0
    session = requests.Session()

    logger.info(
        "Starting standalone collector (poll=%.2fs, reports=%s, heartbeat=%.0fs, pid=%d)",
        args.poll_seconds,
        "off" if no_reports else f"every {args.report_every_seconds:.0f}s",
        args.heartbeat_seconds,
        os.getpid(),
    )
    if not args.log_file:
        logger.warning(
            "Long H24 runs: use --quiet --log-file logs/book_collector.log "
            "and run outside Cursor terminal to avoid IDE memory bloat."
        )
    try:
        while True:
            now = time.time()
            for cfg in cfgs:
                lbl = cfg.label

                info = current_market[lbl]
                must_probe = (
                    info is None
                    or now >= float(info.end_ts)
                    or (now - last_market_probe_ts[lbl]) >= args.market_probe_seconds
                )
                if must_probe:
                    probed = find_current_market(cfg, session)
                    last_market_probe_ts[lbl] = now
                    if probed is not None:
                        info = probed
                        current_market[lbl] = probed
                    elif info is None:
                        logger.debug("[%sm] no market found", cfg.label)
                        continue
                if info is None:
                    continue

                if current_condition[lbl] != info.condition_id:
                    market_day = _infer_day_key_from_market_name(info.question, now)
                    if market_day != day_keys[lbl]:
                        # Switch logger to the market's own day bucket (ET market date).
                        loggers[lbl].close_market()
                        day_keys[lbl] = market_day
                        active_day_dirs[lbl] = _ensure_day_layout(cfg, market_day)
                        loggers[lbl] = PriceLogger(logs_dir=str(active_day_dirs[lbl]))
                        snapshot_count[lbl] = 0
                        last_report_snapshot_count[lbl] = -1

                    current_condition[lbl] = info.condition_id
                    logger.info(
                        "[%sm] market switched: %s | left=%.0fs",
                        cfg.label,
                        info.question[:80],
                        max(0.0, info.end_ts - now),
                    )
                    loggers[lbl].start_market(info.condition_id, info.question)

                prices = collect_prices(info, session)
                if not prices:
                    continue
                loggers[lbl].log_snapshot(
                    condition_id=info.condition_id,
                    market_name=info.question,
                    prices=prices,
                    time_left_seconds=max(0.0, info.end_ts - now),
                )
                snapshot_count[lbl] += 1

            if (
                not no_reports
                and now - last_report_ts >= args.report_every_seconds
            ):
                for cfg in cfgs:
                    lbl = cfg.label
                    removed = _cleanup_incomplete_day_files(cfg, active_day_dirs[lbl], now)
                    if removed:
                        logger.info("[%sm] removed %d incomplete market file(s)", cfg.label, removed)
                    if snapshot_count[lbl] != last_report_snapshot_count[lbl]:
                        run_report(active_day_dirs[lbl], verbose_report=args.verbose)
                        last_report_snapshot_count[lbl] = snapshot_count[lbl]
                last_report_ts = now

            if args.heartbeat_seconds > 0 and now - last_heartbeat_ts >= args.heartbeat_seconds:
                parts = []
                for cfg in cfgs:
                    lbl = cfg.label
                    info = current_market[lbl]
                    q = (info.question[:50] + "…") if info and len(info.question) > 50 else (info.question if info else "—")
                    parts.append(f"{lbl}:snap={snapshot_count[lbl]} mkt={q}")
                logger.info("heartbeat pid=%d | %s", os.getpid(), " | ".join(parts))
                last_heartbeat_ts = now

            time.sleep(max(0.2, args.poll_seconds))
    except KeyboardInterrupt:
        logger.info("Collector stopped by user")
    finally:
        for lg in loggers.values():
            lg.close_market()
        if not no_reports:
            for cfg in cfgs:
                run_report(active_day_dirs[cfg.label], verbose_report=args.verbose)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
