#!/usr/bin/env python3
"""
Helper script per ottenere i token_id dei mercati BTC 5min attivi
Utile per configurare bot2_copytrade_ws.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.market_finder import find_active_5min_market
from shared.config import Config

def main():
    print("=" * 60)
    print("🔍 Ricerca mercati BTC 5min attivi...")
    print("=" * 60)
    print()
    
    market = find_active_5min_market()
    
    if not market:
        print("❌ Nessun mercato attivo trovato")
        print("⏰ Riprova tra qualche minuto")
        return
    
    print(f"✅ Mercato trovato: {market['question']}")
    print(f"⏰ Tempo rimasto: {int(market['minutes_left'])} minuti")
    print()
    
    print("📊 TOKEN IDs (per bot2_copytrade_ws.py):")
    print("-" * 60)
    print()
    print("monitored_tokens = [")
    for outcome, token_id in zip(market['outcomes'], market['token_ids']):
        print(f'    "{token_id}",  # {outcome}')
    print("]")
    print()
    
    print("📋 MARKET INFO:")
    print("-" * 60)
    print(f"Condition ID: {market['condition_id']}")
    print()
    
    print("💡 COME USARLO:")
    print("-" * 60)
    print("1. Copia i token_id sopra")
    print("2. Incollali in bot2_copytrade_ws.py nella sezione monitored_tokens")
    print("3. Avvia il bot WebSocket per copy-trading real-time")
    print()

if __name__ == "__main__":
    main()
