"""
Polymarket History Fetcher - COMPLETO VIA API
Fetcha TUTTE le operazioni usando Data API
"""

import json
import csv
import time
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import sys
from pathlib import Path
import re
sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.config import Config
import requests


def _activity_key(row: Dict) -> Tuple:
    """
    Chiave stabile per dedup cross-window.
    Usa id/activityId se disponibile, altrimenti una fingerprint ampia.
    """
    rid = row.get("id") or row.get("activityId")
    if rid is not None:
        return ("id", str(rid))

    return (
        "fp",
        str(row.get("timestamp", "")),
        str(row.get("type", "")),
        str(row.get("side", "")),
        str(row.get("conditionId", "")),
        str(row.get("outcome", "")),
        str(row.get("size", "")),
        str(row.get("usdcSize", "")),
        str(row.get("price", "")),
        str(row.get("transactionHash", "")),
    )


def fetch_all_activities(wallet_address: str) -> List[Dict]:
    """
    Fetcha TUTTE le attività (trades + redeems) da Data API
    """
    base_url = f"https://data-api.polymarket.com/activity"
    
    print(f"📥 Fetchando TUTTA l'attività (trades + redeems) da Data API...")
    print(f"   Wallet: {wallet_address}")
    
    all_trades = []
    limit = 1000
    # Data API currently rejects offset > 3000 for /activity on historical pull.
    # We bypass this using descending time windows via `end=<oldest_ts-1>`.
    max_offset = 3000
    max_retries = 4
    termination_reason = "unknown"
    stopped_on_error = False
    seen_keys = set()
    current_end_ts: Optional[int] = None
    window_idx = 0

    while True:
        window_idx += 1
        offset = 0
        window_oldest_ts: Optional[int] = None
        window_added = 0
        print(f"   🪟 Finestra #{window_idx} | end_ts={current_end_ts if current_end_ts is not None else 'latest'}")

        while True:
            if offset > max_offset:
                termination_reason = f"offset_cap_{max_offset}_window_{window_idx}"
                break

            params = {
                'user': wallet_address.lower(),
                'limit': limit,
                'offset': offset,
                'sortBy': 'TIMESTAMP',
                'sortDirection': 'DESC',
            }
            if current_end_ts is not None:
                params['end'] = current_end_ts

            batch = None
            last_error = None
            discovered_max_offset = None

            for attempt in range(1, max_retries + 1):
                try:
                    response = requests.get(base_url, params=params, timeout=30)
                    if response.status_code == 429:
                        raise requests.HTTPError("429 Too Many Requests")
                    if response.status_code == 400:
                        msg = response.text or ""
                        m = re.search(r"max historical activity offset of (\d+) exceeded", msg)
                        if m:
                            discovered_max_offset = int(m.group(1))
                    response.raise_for_status()
                    batch = response.json()
                    break
                except Exception as e:
                    last_error = e
                    if discovered_max_offset is not None:
                        max_offset = discovered_max_offset
                        termination_reason = f"offset_cap_{max_offset}_window_{window_idx}"
                        print(f"   ⚠️ Offset cap rilevato da API: {max_offset}")
                        batch = []
                        break
                    wait_s = min(8, 1.5 ** attempt)
                    print(f"   ⚠️ Errore offset {offset} (tentativo {attempt}/{max_retries}): {e}")
                    if attempt < max_retries:
                        time.sleep(wait_s)

            if batch is None:
                stopped_on_error = True
                termination_reason = f"error_at_offset_{offset}_window_{window_idx}"
                print(f"   ❌ Stop fetch: impossibile leggere offset {offset} dopo {max_retries} tentativi")
                if last_error:
                    print(f"      Ultimo errore: {last_error}")
                break

            if not isinstance(batch, list):
                stopped_on_error = True
                termination_reason = f"unexpected_payload_at_offset_{offset}_window_{window_idx}"
                print(f"   ❌ Payload inatteso a offset {offset}: {str(batch)[:200]}")
                break

            if len(batch) == 0:
                termination_reason = f"empty_page_at_offset_{offset}_window_{window_idx}"
                break

            added = 0
            for row in batch:
                key = _activity_key(row)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                all_trades.append(row)
                added += 1

                ts_raw = row.get("timestamp")
                try:
                    ts = int(ts_raw)
                except (TypeError, ValueError):
                    ts = None
                if ts is not None:
                    window_oldest_ts = ts if window_oldest_ts is None else min(window_oldest_ts, ts)

            window_added += added
            print(f"   Offset {offset}: +{len(batch)} lette, +{added} nuove (totale: {len(all_trades)})")

            if len(batch) < limit:
                termination_reason = f"short_page_at_offset_{offset}_window_{window_idx}"
                break

            offset += limit

        if stopped_on_error:
            break

        # Se una finestra non aggiunge più nulla, non c'è altro da scaricare.
        if window_added == 0:
            if current_end_ts is None:
                termination_reason = "no_data"
            else:
                termination_reason = f"no_new_data_window_{window_idx}"
            break

        if window_oldest_ts is None:
            termination_reason = f"no_timestamp_window_{window_idx}"
            break

        # Passa alla finestra precedente nel tempo.
        next_end_ts = window_oldest_ts - 1
        if current_end_ts is not None and next_end_ts >= current_end_ts:
            termination_reason = f"stalled_window_cursor_{window_idx}"
            print("   ⚠️ Cursor temporale fermo: stop difensivo.")
            break
        current_end_ts = next_end_ts
    
    if all_trades:
        print(f"   ✅ Totale: {len(all_trades)} operazioni!")
        print(f"   ℹ️  Terminazione fetch: {termination_reason}")
        if not stopped_on_error and "offset_cap_" in termination_reason:
            print("   ℹ️  Cap offset aggirato con finestra temporale successiva.")
    else:
        print(f"   ⚠️  Nessuna operazione trovata")
    
    return all_trades


def parse_activity(activity: Dict) -> Dict:
    """
    Converte activity (trade o redeem) da Data API al formato unificato
    """
    # Type: TRADE or REDEEM
    activity_type = activity.get('type', 'UNKNOWN')
    
    # Timestamp
    timestamp = int(activity.get('timestamp', 0))
    
    # Transaction hash
    tx_hash = activity.get('transactionHash', '')
    
    if activity_type == 'TRADE':
        # Trade: ha side (BUY/SELL), price, size
        side = activity.get('side', '').upper()
        size = float(activity.get('size', 0))
        price = float(activity.get('price', 0))
        usdc_amount = float(activity.get('usdcSize', size * price))
        
        # Market info potrebbe non esserci, uso conditionId come fallback
        market_name = activity.get('title', activity.get('conditionId', 'Unknown'))
        outcome = activity.get('outcome', '')
        
        return {
            'marketName': market_name,
            'action': side.capitalize(),
            'usdcAmount': round(usdc_amount, 6),
            'tokenAmount': round(size, 6),
            'tokenName': outcome,
            'timestamp': timestamp,
            'datetime': datetime.fromtimestamp(timestamp) if timestamp else None,
            'hash': tx_hash
        }
    
    elif activity_type == 'REDEEM':
        # Redeem: ha usdcSize (amount riscosso)
        usdcSize = float(activity.get('usdcSize', 0))
        size = float(activity.get('size', 0))
        
        market_name = activity.get('title', activity.get('conditionId', 'Unknown'))
        outcome = activity.get('outcome', '')
        
        return {
            'marketName': market_name,
            'action': 'Redeem',
            'usdcAmount': round(usdcSize, 6),
            'tokenAmount': round(size, 6),
            'tokenName': outcome,
            'timestamp': timestamp,
            'datetime': datetime.fromtimestamp(timestamp) if timestamp else None,
            'hash': tx_hash
        }
    
    else:
        # Tipo sconosciuto, log basico
        return {
            'marketName': activity.get('conditionId', 'Unknown'),
            'action': activity_type,
            'usdcAmount': float(activity.get('usdcSize', 0)),
            'tokenAmount': float(activity.get('size', 0)),
            'tokenName': '',
            'timestamp': timestamp,
            'datetime': datetime.fromtimestamp(timestamp) if timestamp else None,
            'hash': tx_hash
        }


def fetch_all_history() -> List[Dict]:
    """
    Fetcha TUTTA la storia via Data API
    """
    print("=" * 70)
    print("🔍 POLYMARKET HISTORY FETCHER - DATA API")
    print("=" * 70)
    
    wallet_address = Config.WALLET_ADDRESS
    
    # Fetch all activities (trades + redeems)
    activities_raw = fetch_all_activities(wallet_address)
    
    if not activities_raw:
        return []
    
    # Parse
    operations = [parse_activity(a) for a in activities_raw]
    
    # Ordina per timestamp (più recente prima)
    operations.sort(key=lambda x: x.get('timestamp', 0), reverse=True)
    
    print(f"\n📊 RIEPILOGO:")
    print(f"   Operazioni totali: {len(operations)}")
    print("=" * 70)
    
    return operations


def save_to_json(operations: List[Dict], filename: str = 'polymarket_history_api.json'):
    """Salva in JSON"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(operations, f, indent=2, default=str)
    print(f"💾 Salvato: {filename}")


def save_to_csv(operations: List[Dict], filename: str = 'polymarket_history_api.csv'):
    """Salva in CSV (formato compatibile con export Polymarket)"""
    if not operations:
        return
    
    fieldnames = ['marketName', 'action', 'usdcAmount', 'tokenAmount', 'tokenName', 'timestamp', 'hash']
    
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        
        for op in operations:
            row = {k: op.get(k, '') for k in fieldnames}
            writer.writerow(row)
    
    print(f"💾 Salvato: {filename}")


def calculate_stats(operations: List[Dict]):
    """Calcola statistiche"""
    if not operations:
        return
    
    print(f"\n📈 STATISTICHE:")
    print("=" * 70)
    
    buys = [op for op in operations if op.get('action') == 'Buy']
    sells = [op for op in operations if op.get('action') == 'Sell']
    redeems = [op for op in operations if op.get('action') == 'Redeem']
    
    total_spent = sum(op.get('usdcAmount', 0) for op in buys)
    total_from_sells = sum(op.get('usdcAmount', 0) for op in sells)
    total_from_redeems = sum(op.get('usdcAmount', 0) for op in redeems)
    
    print(f"Buy:    {len(buys):4d} operazioni | ${total_spent:,.2f} spesi")
    print(f"Sell:   {len(sells):4d} operazioni | ${total_from_sells:,.2f} ricevuti")
    print(f"Redeem: {len(redeems):4d} operazioni | ${total_from_redeems:,.2f} ricevuti")
    
    # P/L totale (buy vs sell+redeem)
    net_pnl = (total_from_sells + total_from_redeems) - total_spent
    print(f"\nP/L Totale: ${net_pnl:+,.2f}")
    
    if total_spent > 0:
        roi = (net_pnl / total_spent) * 100
        print(f"ROI:           {roi:+.2f}%")
    
    # Mercati più tradati
    market_counts = {}
    for op in operations:
        market = op.get('marketName', 'Unknown')
        market_counts[market] = market_counts.get(market, 0) + 1
    
    top_markets = sorted(market_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    
    print(f"\nTop 5 mercati più operati:")
    for market, count in top_markets:
        # Truncate long market names
        display_name = market[:50] + "..." if len(market) > 50 else market
        print(f"  {count:3d}x | {display_name}")
    
    # Win rate dai redeem (redeem > 0 = win)
    wins = [r for r in redeems if r.get('usdcAmount', 0) > 0]
    losses = [r for r in redeems if r.get('usdcAmount', 0) == 0]
    
    if redeems:
        win_rate = (len(wins) / len(redeems)) * 100
        print(f"\nWin Rate: {win_rate:.1f}% ({len(wins)} wins, {len(losses)} losses)")
    
    print("=" * 70)


def main():
    """Main entry point"""
    try:
        # Validate config
        Config.validate()
        
        # Fetch all history
        operations = fetch_all_history()
        
        if not operations:
            print("\n⚠️  Nessuna operazione trovata!")
            return
        
        # Save
        save_to_json(operations)
        save_to_csv(operations)
        
        # Stats
        calculate_stats(operations)
        
        print("\n✅ Completato!")
        print("\nFile generati:")
        print("  - polymarket_history_api.json (formato completo)")
        print("  - polymarket_history_api.csv (formato Polymarket)")
        
    except Exception as e:
        print(f"\n❌ Errore: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
