#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from bot4_chainlink.chainlink_client import ChainlinkDataStreamsClient
from bot4_chainlink.config import load_config


@dataclass(frozen=True)
class LabelRow:
    timestamp_end: int
    condition_id: str
    final_ud: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Verify merged outcomes against Chainlink prices.")
    p.add_argument("--limit", type=int, default=0, help="Process first N rows after sorting by timestamp (0 = all).")
    p.add_argument("--pre-gap-only", action="store_true", help="Use only rows before historical big gap boundary.")
    p.add_argument("--big-gap-start", type=int, default=1769470200)
    p.add_argument(
        "--allow-mismatch",
        action="store_true",
        help="Do not fail process when mismatches exist (default behavior is strict fail).",
    )
    p.add_argument(
        "--max-mismatch-print",
        type=int,
        default=20,
        help="How many mismatches to print in strict mode.",
    )
    return p.parse_args()


def load_labels(path: Path, pre_gap_only: bool, big_gap_start: int) -> list[LabelRow]:
    out: list[LabelRow] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                t_end = int(float((row.get("timestamp_unix") or "").strip()))
            except Exception:
                continue
            if pre_gap_only and t_end >= big_gap_start:
                continue
            ud = (row.get("final_outcome_ud") or "").strip().upper()
            if ud not in {"U", "D"}:
                continue
            cond = (row.get("condition_id") or "").strip()
            out.append(LabelRow(timestamp_end=t_end, condition_id=cond, final_ud=ud))
    out.sort(key=lambda x: x.timestamp_end)
    return out


def chunked(items: list[LabelRow], n: int) -> Iterable[list[LabelRow]]:
    if n <= 0:
        yield items
        return
    for i in range(0, len(items), n):
        yield items[i : i + n]


def main() -> int:
    args = parse_args()
    cfg = load_config()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    labels = load_labels(cfg.labels_csv, pre_gap_only=args.pre_gap_only, big_gap_start=args.big_gap_start)
    if args.limit > 0:
        labels = labels[: args.limit]
    if not labels:
        print("No labels found for selected range.")
        return 1

    print(f"Loaded labels: {len(labels)}")
    client = ChainlinkDataStreamsClient(
        api_base_url=cfg.api_base_url,
        api_key=cfg.api_key,
        api_secret=cfg.api_secret,
        feed_id=cfg.feed_id,
    )

    report_rows = []
    for i, row in enumerate(labels, start=1):
        t_start = row.timestamp_end - cfg.market_window_seconds
        p_start = client.get_price_at_or_before(t_start)
        p_end = client.get_price_at_or_before(row.timestamp_end)
        derived_ud = "U" if p_end.price >= p_start.price else "D"
        match = derived_ud == row.final_ud
        report_rows.append(
            {
                "idx": i,
                "condition_id": row.condition_id,
                "timestamp_start": t_start,
                "timestamp_end": row.timestamp_end,
                "chainlink_ts_start": p_start.timestamp_unix,
                "chainlink_price_start": f"{p_start.price:.10f}",
                "chainlink_ts_end": p_end.timestamp_unix,
                "chainlink_price_end": f"{p_end.price:.10f}",
                "derived_outcome_ud": derived_ud,
                "merged_outcome_ud": row.final_ud,
                "is_match": "1" if match else "0",
            }
        )
        if i % 100 == 0:
            print(f"Processed {i}/{len(labels)}")

    out_csv = cfg.out_dir / "alignment_report.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = list(report_rows[0].keys())
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(report_rows)

    total = len(report_rows)
    matched = sum(1 for r in report_rows if r["is_match"] == "1")
    mismatches = total - matched
    summary = {
        "total_rows": total,
        "matched_rows": matched,
        "mismatch_rows": mismatches,
        "match_rate_pct": (100.0 * matched / total) if total else 0.0,
        "out_csv": str(out_csv),
    }
    out_json = cfg.out_dir / "alignment_summary.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Match rate: {summary['match_rate_pct']:.4f}% ({matched}/{total})")
    print(f"Report: {out_csv}")
    print(f"Summary: {out_json}")
    if mismatches > 0 and not args.allow_mismatch:
        print("\nSTRICT MODE FAIL: mismatches detected.")
        to_show = max(0, args.max_mismatch_print)
        shown = 0
        for r in report_rows:
            if r["is_match"] == "1":
                continue
            print(
                "  mismatch | "
                f"condition_id={r['condition_id']} | "
                f"start={r['timestamp_start']} end={r['timestamp_end']} | "
                f"derived={r['derived_outcome_ud']} merged={r['merged_outcome_ud']} | "
                f"p_start={r['chainlink_price_start']} p_end={r['chainlink_price_end']}"
            )
            shown += 1
            if shown >= to_show:
                break
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
