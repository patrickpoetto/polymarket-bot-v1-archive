"""
Session Reconciler

Fetches real trade activity from Polymarket Data API for the current bot session
and produces a human-readable summary for comparison with session log estimates.
"""
from __future__ import annotations

import time
import requests
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_ACTIVITY_URL = "https://data-api.polymarket.com/activity"


def fetch_activities_since(
    wallet_address: str,
    since_ts: int,
    max_retries: int = 3,
) -> List[Dict]:
    """
    Fetch all TRADE/REDEEM activities for the wallet since since_ts (Unix seconds).

    Uses the Data API `start` parameter to limit results, and also filters
    client-side in case the API ignores the parameter.
    Returns a list sorted descending by timestamp (most recent first).
    """
    all_activities: List[Dict] = []
    seen_ids: set = set()
    limit = 500
    offset = 0

    while True:
        params: Dict = {
            "user": wallet_address.lower(),
            "limit": limit,
            "offset": offset,
            "sortBy": "TIMESTAMP",
            "sortDirection": "DESC",
            "start": since_ts,
        }

        batch: Optional[List[Dict]] = None
        for attempt in range(1, max_retries + 1):
            try:
                resp = requests.get(_ACTIVITY_URL, params=params, timeout=30)
                resp.raise_for_status()
                batch = resp.json()
                break
            except Exception as exc:
                if attempt == max_retries:
                    raise RuntimeError(
                        f"API fetch failed after {max_retries} attempts: {exc}"
                    ) from exc
                time.sleep(min(8.0, 1.5 ** attempt))

        if not batch or not isinstance(batch, list):
            break

        any_new = False
        for row in batch:
            try:
                ts = int(row.get("timestamp", 0))
            except (TypeError, ValueError):
                ts = 0

            # Client-side filter: skip anything older than session start
            if ts < since_ts:
                continue

            # Dedup by id or fingerprint
            rid = row.get("id") or row.get("activityId")
            if rid is not None:
                key = ("id", str(rid))
            else:
                key = (
                    "fp",
                    str(ts),
                    str(row.get("type", "")),
                    str(row.get("side", "")),
                    str(row.get("conditionId", "")),
                    str(row.get("outcome", "")),
                    str(row.get("size", "")),
                    str(row.get("transactionHash", "")),
                )
            if key in seen_ids:
                continue
            seen_ids.add(key)
            all_activities.append(row)
            any_new = True

        if len(batch) < limit or not any_new:
            break

        offset += limit

    return all_activities


def _parse_activity(a: Dict) -> Optional[Dict]:
    """Parse a raw activity dict into a normalized record. Returns None on failure."""
    a_type = a.get("type", "").upper()
    try:
        ts = int(a.get("timestamp", 0))
    except (TypeError, ValueError):
        ts = 0

    market = (a.get("title") or a.get("conditionId") or "Unknown")[:70]
    outcome = a.get("outcome", "")

    if a_type == "TRADE":
        side = a.get("side", "").upper()
        try:
            size = float(a.get("size", 0))
            price = float(a.get("price", 0))
            usdc = float(a.get("usdcSize", size * price))
        except (TypeError, ValueError):
            return None
        return {
            "ts": ts,
            "type": "TRADE",
            "side": side,
            "market": market,
            "outcome": outcome,
            "size": size,
            "price": price,
            "usdc": usdc,
        }

    if a_type == "REDEEM":
        try:
            usdc = float(a.get("usdcSize", 0))
            size = float(a.get("size", 0))
        except (TypeError, ValueError):
            return None
        return {
            "ts": ts,
            "type": "REDEEM",
            "side": "REDEEM",
            "market": market,
            "outcome": outcome,
            "size": size,
            "price": (usdc / size) if size > 0 else 0.0,
            "usdc": usdc,
        }

    return None


def _dt(ts: int) -> str:
    try:
        return datetime.fromtimestamp(ts).strftime("%H:%M:%S")
    except Exception:
        return "??:??:??"


def build_text_report(
    activities: List[Dict],
    session_start_ts: int,
    session_log_pnl: float,
) -> str:
    """
    Build a human-readable text report of all session trades,
    grouped by (market, outcome), with a final comparison vs session log P&L.
    """
    records = [r for a in activities if (r := _parse_activity(a)) is not None]
    records.sort(key=lambda t: t["ts"])

    # Group by (market, outcome) preserving insertion order
    seen_keys: List[Tuple[str, str]] = []
    groups: Dict[Tuple[str, str], Dict[str, list]] = {}
    for t in records:
        key = (t["market"], t["outcome"])
        if key not in groups:
            seen_keys.append(key)
            groups[key] = {"buys": [], "sells": [], "redeems": []}
        if t["side"] == "BUY":
            groups[key]["buys"].append(t)
        elif t["side"] == "SELL":
            groups[key]["sells"].append(t)
        elif t["side"] == "REDEEM":
            groups[key]["redeems"].append(t)

    buy_n = sum(1 for t in records if t["side"] == "BUY")
    sell_n = sum(1 for t in records if t["side"] == "SELL")
    redeem_n = sum(1 for t in records if t["side"] == "REDEEM")

    SEP = "═" * 62
    SEP2 = "─" * 62

    lines: List[str] = [
        "",
        SEP,
        "📋  REAL TRADE HISTORY  (Polymarket Data API)",
        SEP,
        f"   Session start : {datetime.fromtimestamp(session_start_ts).strftime('%Y-%m-%d %H:%M:%S')}",
        f"   Fetched at    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"   Activities    : {len(records)}  (BUY: {buy_n}, SELL: {sell_n}, REDEEM: {redeem_n})",
        "",
    ]

    total_spent = 0.0
    total_received = 0.0
    position_count = 0

    for key in seen_keys:
        market, outcome = key
        g = groups[key]
        buys = g["buys"]
        sells = g["sells"]
        redeems = g["redeems"]

        if not buys:
            # Orphan sell/redeem without a corresponding buy — still record it
            lines.append(f"  {SEP2}")
            lines.append(f"  ⚠️  {market}")
            lines.append(f"     Outcome: {outcome}  [no matching BUY in session]")
            for t in sells:
                lines.append(f"     SELL   {_dt(t['ts'])}  {t['size']:.4f} sh @ {t['price']:.3f}  +${t['usdc']:.2f}")
                total_received += t["usdc"]
            for t in redeems:
                lines.append(f"     REDEEM {_dt(t['ts'])}  {t['size']:.4f} sh @ {t['price']:.3f}  +${t['usdc']:.2f}")
                total_received += t["usdc"]
            lines.append("")
            continue

        position_count += 1
        spent = sum(t["usdc"] for t in buys)
        received = sum(t["usdc"] for t in sells) + sum(t["usdc"] for t in redeems)
        pnl = received - spent
        pnl_pct = (pnl / spent * 100) if spent > 0 else 0.0
        total_spent += spent
        total_received += received

        sign = "+" if pnl >= 0 else ""
        lines.append(f"  {SEP2}")
        lines.append(f"  📌  {market}")
        lines.append(f"      Outcome: {outcome}")
        for t in buys:
            lines.append(f"      BUY    {_dt(t['ts'])}  {t['size']:.4f} sh @ {t['price']:.3f}  -${t['usdc']:.2f}")
        for t in sells:
            lines.append(f"      SELL   {_dt(t['ts'])}  {t['size']:.4f} sh @ {t['price']:.3f}  +${t['usdc']:.2f}")
        for t in redeems:
            lines.append(f"      REDEEM {_dt(t['ts'])}  {t['size']:.4f} sh @ {t['price']:.3f}  +${t['usdc']:.2f}")
        lines.append(f"      → P&L:  {sign}${pnl:.2f}  ({sign}{pnl_pct:.1f}%)")
        lines.append("")

    net_pnl = total_received - total_spent
    net_sign = "+" if net_pnl >= 0 else ""
    log_sign = "+" if session_log_pnl >= 0 else ""
    discrepancy = net_pnl - session_log_pnl
    disc_sign = "+" if discrepancy >= 0 else ""

    lines += [
        SEP2,
        f"  💰  Total spent       : ${total_spent:.2f}",
        f"  💵  Total received    : ${total_received:.2f}",
        f"  📊  Net P&L (real)    : {net_sign}${net_pnl:.2f}",
        f"  📝  Session log P&L   : {log_sign}${session_log_pnl:.2f}",
        f"  🔍  Discrepancy       : {disc_sign}${discrepancy:.2f}",
        f"  📈  Positions traded  : {position_count}",
        SEP,
        "",
    ]

    return "\n".join(lines)
