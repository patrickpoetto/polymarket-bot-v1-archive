#!/usr/bin/env python3
"""
Fetch historical final outcomes for BTC Up/Down Polymarket markets.

Why this version:
- Slug-only scan (btc-updown-5m-{ts}) misses periods where naming changed.
- Here we crawl Gamma markets by offset and filter robustly by timeframe.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import re
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

GAMMA_MARKETS_URL = "https://gamma-api.polymarket.com/markets"
CSV_FIELDS = [
    "timestamp_unix",
    "timestamp_iso",
    "condition_id",
    "slug",
    "question",
    "closed",
    "active",
    "final_outcome_ud",
    "price_up",
    "price_down",
]


def _request_json(url: str, retries: int = 4, timeout: int = 30) -> Any:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
            "Accept": "application/json",
        },
    )
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            last_exc = exc
            if attempt < retries:
                time.sleep(min(8, 1.7 ** attempt))
    raise RuntimeError(f"Request failed after {retries} attempts: {url}\n{last_exc}")


def _parse_list_field(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []
        try:
            p = json.loads(s)
            return p if isinstance(p, list) else []
        except Exception:
            return []
    return []


def _window_minutes_in_question(question: str, minutes: int) -> bool:
    """
    Detect patterns like:
    - 7:40PM-7:45PM ET
    - 11:55PM-12:00AM ET
    """
    q = question.strip()
    m = re.search(
        r"(\d{1,2}):(\d{2})(AM|PM)\s*-\s*(\d{1,2}):(\d{2})(AM|PM)",
        q,
        re.IGNORECASE,
    )
    if not m:
        return False

    h1, mn1, ampm1, h2, mn2, ampm2 = m.groups()
    h1 = int(h1) % 12 + (12 if ampm1.upper() == "PM" else 0)
    h2 = int(h2) % 12 + (12 if ampm2.upper() == "PM" else 0)
    t1 = h1 * 60 + int(mn1)
    t2 = h2 * 60 + int(mn2)
    diff = (t2 - t1) % (24 * 60)
    return diff == minutes


def _is_btc_updown_market(market: dict[str, Any], timeframe_minutes: int) -> bool:
    q = str(market.get("question", "")).strip()
    slug = str(market.get("slug", "")).strip().lower()
    ql = q.lower()

    if not ql.startswith("bitcoin up or down -"):
        return False

    if f"btc-updown-{timeframe_minutes}m-" in slug:
        return True

    # Fallback for alternate naming eras.
    return _window_minutes_in_question(q, timeframe_minutes)


def _final_outcome_ud(market: dict[str, Any]) -> tuple[str, float | None, float | None]:
    outcomes = _parse_list_field(market.get("outcomes"))
    prices = _parse_list_field(market.get("outcomePrices"))
    if len(outcomes) < 2 or len(prices) < 2:
        return "?", None, None
    try:
        p0 = float(prices[0])
        p1 = float(prices[1])
    except Exception:
        return "?", None, None

    if abs(p0 - p1) < 1e-12:
        return "?", p0, p1

    winner_idx = 0 if p0 > p1 else 1
    winner = str(outcomes[winner_idx]).strip().lower()
    if winner.startswith("up"):
        return "U", p0, p1
    if winner.startswith("down"):
        return "D", p0, p1
    return ("U" if winner_idx == 0 else "D"), p0, p1


def _market_timestamp(market: dict[str, Any]) -> int | None:
    end_date = str(market.get("endDate", "")).strip()
    if not end_date:
        return None
    try:
        dt = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
        return int(dt.timestamp())
    except Exception:
        return None


def fetch_btc_updown_history_by_offset(
    start_offset: int,
    max_offset: int,
    limit: int,
    only_closed: bool,
    sleep_ms: int,
    timeframe_minutes: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()

    for offset in range(start_offset, max_offset + 1, limit):
        url = f"{GAMMA_MARKETS_URL}?{urllib.parse.urlencode({'limit': str(limit), 'offset': str(offset)})}"
        payload = _request_json(url)
        if not isinstance(payload, list) or not payload:
            print(f"Stop at offset={offset}: empty page.")
            break

        hits_on_page = 0
        for market in payload:
            if not isinstance(market, dict):
                continue
            if not _is_btc_updown_market(market, timeframe_minutes):
                continue
            if only_closed and not bool(market.get("closed", False)):
                continue

            cid = str(market.get("conditionId", "")).strip()
            slug = str(market.get("slug", "")).strip()
            dedup_key = cid or slug
            if not dedup_key or dedup_key in seen:
                continue
            seen.add(dedup_key)
            hits_on_page += 1

            ts = _market_timestamp(market)
            outcome_ud, p_up, p_down = _final_outcome_ud(market)
            rows.append(
                {
                    "timestamp_unix": ts if ts is not None else "",
                    "timestamp_iso": datetime.fromtimestamp(ts, tz=timezone.utc).isoformat() if ts else "",
                    "condition_id": cid,
                    "slug": slug,
                    "question": str(market.get("question", "")).strip(),
                    "closed": bool(market.get("closed", False)),
                    "active": bool(market.get("active", False)),
                    "final_outcome_ud": outcome_ud,
                    "price_up": "" if p_up is None else f"{p_up:.12f}",
                    "price_down": "" if p_down is None else f"{p_down:.12f}",
                }
            )

        if offset % (limit * 10) == 0:
            print(f"offset={offset:>7} | found_total={len(rows)} | page_hits={hits_on_page}")

        if sleep_ms > 0:
            time.sleep(sleep_ms / 1000.0)

    rows.sort(key=lambda r: (r["timestamp_unix"] == "", r["timestamp_unix"]))
    return rows


def save_csv(rows: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            if not isinstance(row, dict):
                continue
            normalized = {k: row.get(k, "") for k in CSV_FIELDS}
            # Normalize numeric timestamp when possible.
            ts_raw = str(normalized.get("timestamp_unix", "")).strip()
            if ts_raw:
                try:
                    normalized["timestamp_unix"] = str(int(float(ts_raw)))
                except Exception:
                    normalized["timestamp_unix"] = ""
            rows.append(normalized)
    return rows


def _dedup_sort_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[str, dict[str, Any]] = {}
    for row in rows:
        cid = str(row.get("condition_id", "")).strip()
        slug = str(row.get("slug", "")).strip()
        key = cid or slug
        if not key:
            continue
        if key not in by_key:
            by_key[key] = row
            continue
        # Keep row with valid timestamp / newest timestamp if both exist.
        old = by_key[key]
        old_ts = str(old.get("timestamp_unix", "")).strip()
        new_ts = str(row.get("timestamp_unix", "")).strip()
        if not old_ts and new_ts:
            by_key[key] = row
            continue
        if old_ts and new_ts:
            try:
                if int(new_ts) > int(old_ts):
                    by_key[key] = row
            except Exception:
                pass

    merged = list(by_key.values())

    def sort_key(r: dict[str, Any]) -> tuple[int, int]:
        ts_raw = str(r.get("timestamp_unix", "")).strip()
        if not ts_raw:
            return (1, 0)
        try:
            return (0, int(ts_raw))
        except Exception:
            return (1, 0)

    merged.sort(key=sort_key)
    return merged


def merge_csv_glob(pattern: str) -> list[dict[str, Any]]:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    all_rows: list[dict[str, Any]] = []
    for p in files:
        all_rows.extend(_read_csv_rows(p))
    return _dedup_sort_rows(all_rows)


def _print_summary(rows: list[dict[str, Any]], out_path: Path, timeframe_minutes: int) -> None:
    u = sum(1 for r in rows if r["final_outcome_ud"] == "U")
    d = sum(1 for r in rows if r["final_outcome_ud"] == "D")
    unk = sum(1 for r in rows if r["final_outcome_ud"] not in {"U", "D"})
    print(f"Saved: {out_path}")
    print(f"Total BTC {timeframe_minutes}m markets: {len(rows)}")
    print(f"U: {u} | D: {d} | Unknown: {unk}")
    if rows:
        print(f"Range: {rows[0]['timestamp_iso']} -> {rows[-1]['timestamp_iso']}")


def _default_out_for_timeframe(timeframe_minutes: int) -> str:
    return f"archived_data/btc_{timeframe_minutes}m_outcomes_history.csv"


def _default_merge_glob_for_timeframe(timeframe_minutes: int) -> str:
    return f"archived_data/btc_{timeframe_minutes}m_outcomes_history*.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch BTC Up/Down final outcomes history by timeframe.")
    parser.add_argument(
        "--timeframe-minutes",
        type=int,
        default=5,
        help="Market timeframe in minutes (e.g. 5, 15).",
    )
    parser.add_argument(
        "--mode",
        choices=["scan", "merge", "scan-merge"],
        default="scan",
        help="scan: crawl API, merge: merge existing CSVs, scan-merge: crawl then merge.",
    )
    parser.add_argument("--start-offset", type=int, default=0, help="Starting Gamma offset.")
    parser.add_argument("--max-offset", type=int, default=350000, help="Maximum Gamma offset to crawl.")
    parser.add_argument("--limit", type=int, default=500, help="Gamma page size.")
    parser.add_argument("--sleep-ms", type=int, default=10, help="Sleep between pages (ms).")
    parser.add_argument(
        "--include-open",
        action="store_true",
        help="Include non-closed markets too (outcome may be '?').",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output CSV path.",
    )
    parser.add_argument(
        "--merge-glob",
        type=str,
        default=None,
        help="Glob of CSV files to merge in merge/scan-merge mode.",
    )
    parser.add_argument(
        "--allow-unknown",
        action="store_true",
        help="Do not fail when Unknown outcomes are present.",
    )
    args = parser.parse_args()

    if args.timeframe_minutes <= 0:
        raise SystemExit("timeframe-minutes must be > 0")

    out_arg = args.out if args.out else _default_out_for_timeframe(args.timeframe_minutes)
    merge_glob = (
        args.merge_glob if args.merge_glob else _default_merge_glob_for_timeframe(args.timeframe_minutes)
    )
    out_path = Path(out_arg)

    if args.mode in {"scan", "scan-merge"}:
        print(f"Fetching BTC {args.timeframe_minutes}m outcome history via offset crawl...")
        scan_rows = fetch_btc_updown_history_by_offset(
            start_offset=max(0, args.start_offset),
            max_offset=max(0, args.max_offset),
            limit=max(1, args.limit),
            only_closed=not args.include_open,
            sleep_ms=max(0, args.sleep_ms),
            timeframe_minutes=args.timeframe_minutes,
        )
        save_csv(scan_rows, out_path)
        _print_summary(scan_rows, out_path, args.timeframe_minutes)

    if args.mode in {"merge", "scan-merge"}:
        print(f"Merging CSV files with pattern: {merge_glob}")
        merged = merge_csv_glob(merge_glob)
        save_csv(merged, out_path)
        _print_summary(merged, out_path, args.timeframe_minutes)

    # Hard safety check: unknown outcomes must stay at 0 unless explicitly allowed.
    final_rows = _read_csv_rows(out_path)
    unknown = sum(1 for r in final_rows if r.get("final_outcome_ud") not in {"U", "D"})
    if unknown > 0 and not args.allow_unknown:
        raise SystemExit(
            f"❌ Safety check failed: Unknown outcomes = {unknown} (must be 0). "
            f"Use --allow-unknown only if you explicitly want to bypass."
        )


if __name__ == "__main__":
    main()
