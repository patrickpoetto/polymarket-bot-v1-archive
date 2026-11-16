"""Diagnostica: lista (e opzionalmente cancella) tutti gli ordini aperti CLOB
per il wallet configurato. Utile quando Polymarket dice "sum of matched orders"
ma la UI non mostra ordini aperti.

Uso:
    python scripts/inspect_open_orders.py             # solo lista
    python scripts/inspect_open_orders.py --cancel    # cancella tutti
"""
from __future__ import annotations

import sys
import os
import argparse
import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from py_clob_client_v2 import ClobClient, BalanceAllowanceParams, AssetType, TradeParams
from shared.config import Config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("inspect_orders")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cancel", action="store_true", help="cancella tutti gli ordini aperti")
    args = parser.parse_args()

    Config.validate()

    pk = Config.PRIVATE_KEY
    if not pk.startswith("0x"):
        pk = "0x" + pk

    tmp = ClobClient(host="https://clob.polymarket.com", key=pk, chain_id=Config.CHAIN_ID)
    creds = tmp.create_or_derive_api_key()
    client = ClobClient(
        host="https://clob.polymarket.com",
        key=pk,
        chain_id=Config.CHAIN_ID,
        creds=creds,
        signature_type=2,
        funder=Config.WALLET_ADDRESS,
    )
    log.info("Wallet (signer/funder): %s", Config.WALLET_ADDRESS)

    try:
        orders = client.get_open_orders() or []
    except Exception as exc:
        log.error("get_open_orders failed: %s", exc)
        return 2

    log.info("Trovati %d ordini aperti via CLOB API", len(orders))
    for i, o in enumerate(orders, 1):
        oid = o.get("id") or o.get("orderID") or "?"
        tok = o.get("asset_id") or o.get("token_id") or "?"
        side = o.get("side", "?")
        price = o.get("price", "?")
        size = o.get("original_size") or o.get("size") or "?"
        size_matched = o.get("size_matched", "?")
        status = o.get("status", "?")
        market = o.get("market", "?")
        log.info(
            "  [%d] id=%s side=%s @%s size=%s matched=%s status=%s market=%s token=%s",
            i, oid[:16] if isinstance(oid, str) else oid,
            side, price, size, size_matched, status, market, tok[:16] if isinstance(tok, str) else tok,
        )

    try:
        ba = client.get_balance_allowance(BalanceAllowanceParams(asset_type=AssetType.COLLATERAL))
        log.info("CLOB balance/allowance (COLLATERAL): %s", ba)
    except Exception as exc:
        log.warning("get_balance_allowance failed: %s", exc)

    try:
        trades = client.get_trades(TradeParams(maker_address=Config.WALLET_ADDRESS)) or []
        log.info("Trovati %d trade recenti del wallet", len(trades))
        non_confirmed = [t for t in trades if str(t.get("status", "")).upper() not in {"CONFIRMED", "FAILED"}]
        if non_confirmed:
            log.warning("⚠️  %d trade NON in stato terminale (potenziali blocchi):", len(non_confirmed))
            from collections import Counter
            status_counts = Counter(str(t.get("status","")).upper() for t in non_confirmed)
            for st, n in status_counts.items():
                log.warning("    %s = %d", st, n)
            for t in non_confirmed:
                price = float(t.get("price", 0) or 0)
                size = float(t.get("size", 0) or 0)
                log.warning("    id=%s status=%s side=%s @%s size=%s notional=$%.4f ts=%s",
                            str(t.get("id",""))[:16], t.get("status"), t.get("side"),
                            t.get("price"), t.get("size"), price * size,
                            t.get("last_update", t.get("match_time", "?")))
            total = sum(float(t.get("price",0) or 0) * float(t.get("size",0) or 0)
                        for t in non_confirmed if str(t.get("side","")).upper() == "BUY")
            log.warning("    TOTAL BUY notional non-terminale: $%.4f", total)
        for i, t in enumerate(trades[:20], 1):
            tid = t.get("id") or t.get("trade_id") or "?"
            status = t.get("status", "?")
            side = t.get("side", "?")
            price = t.get("price", "?")
            size = t.get("size", "?")
            mt = t.get("match_time") or t.get("last_update", "?")
            log.info("  [trade %d] id=%s status=%s side=%s @%s size=%s ts=%s",
                     i, str(tid)[:16], status, side, price, size, mt)
    except Exception as exc:
        log.warning("get_trades failed: %s", exc)

    if not orders and not args.cancel:
        log.info("Nessun ordine aperto LIVE. Se 'sum of matched orders' persiste, "
                 "controlla i trade qui sopra — uno status MATCHED/MINED non ancora "
                 "CONFIRMED tiene bloccato il collateral.")
        return 0

    if args.cancel:
        try:
            res = client.cancel_all()
            log.info("cancel_all → %s", res)
        except Exception as exc:
            log.error("cancel_all failed: %s", exc)
            return 3
    else:
        log.info("Per cancellarli rilancia con: python scripts/inspect_open_orders.py --cancel")

    return 0


if __name__ == "__main__":
    sys.exit(main())
