#!/usr/bin/env python3
"""Analizza performance wallet Polymarket"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import requests
from datetime import datetime
from collections import defaultdict

def fetch_all_trades(wallet: str):
    """Fetch tutti i trade"""
    url = "https://data-api.polymarket.com/trades"
    all_trades = []
    offset = 0
    limit = 1000
    
    while True:
        params = {'user': wallet.lower(), 'limit': limit, 'offset': offset}
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code != 200:
            break
        data = resp.json()
        if not data:
            break
        all_trades.extend(data)
        if len(data) < limit:
            break
        offset += limit
    
    return all_trades

def analyze(wallet: str):
    """Analisi completa performance"""
    print(f"\n📊 WALLET: {wallet[:10]}...{wallet[-6:]}\n")
    
    # Fetch all
    print("⏳ Downloading tutti i trade...")
    trades = fetch_all_trades(wallet)
    if not trades:
        print("❌ Nessun trade trovato\n")
        return
    
    print(f"✅ {len(trades)} trade totali\n")
    
    # Split analysis
    recent_500 = trades[:min(500, len(trades))]
    
    # === TOTAL STATS ===
    print("=" * 60)
    print("📈 PERFORMANCE TOTALI")
    print("=" * 60)
    
    buys = [t for t in trades if t.get('side') == 'BUY']
    sells = [t for t in trades if t.get('side') == 'SELL']
    
    total_buy_value = sum(float(t.get('price', 0)) * float(t.get('size', 0)) for t in buys)
    total_sell_value = sum(float(t.get('price', 0)) * float(t.get('size', 0)) for t in sells)
    
    print(f"Total trade: {len(trades)}")
    print(f"BUY: {len(buys)} | SELL: {len(sells)}")
    print(f"Buy volume: ${total_buy_value:,.2f}")
    print(f"Sell volume: ${total_sell_value:,.2f}")
    
    # Price analysis
    prices = [float(t.get('price', 0)) for t in trades]
    sizes = [float(t.get('price', 0)) * float(t.get('size', 0)) for t in trades]
    
    print(f"\nSize: min ${min(sizes):.2f} | avg ${sum(sizes)/len(sizes):.2f} | max ${max(sizes):.2f}")
    print(f"Price: min ${min(prices):.3f} | avg ${sum(prices)/len(prices):.3f} | max ${max(prices):.3f}")
    
    # === RECENT 500 ===
    print("\n" + "=" * 60)
    print("🔥 ULTIMI 500 TRADE")
    print("=" * 60)
    
    buys_500 = [t for t in recent_500 if t.get('side') == 'BUY']
    sells_500 = [t for t in recent_500 if t.get('side') == 'SELL']
    
    buy_val_500 = sum(float(t.get('price', 0)) * float(t.get('size', 0)) for t in buys_500)
    sell_val_500 = sum(float(t.get('price', 0)) * float(t.get('size', 0)) for t in sells_500)
    
    print(f"BUY: {len(buys_500)} (${buy_val_500:,.2f})")
    print(f"SELL: {len(sells_500)} (${sell_val_500:,.2f})")
    
    sizes_500 = [float(t.get('price', 0)) * float(t.get('size', 0)) for t in recent_500]
    prices_500 = [float(t.get('price', 0)) for t in buys_500]
    
    print(f"Size avg: ${sum(sizes_500)/len(sizes_500):.2f}")
    print(f"Price avg (BUY): ${sum(prices_500)/len(prices_500):.3f}" if prices_500 else "No BUY")
    
    # Time range
    if len(recent_500) >= 2:
        try:
            t1 = datetime.fromtimestamp(recent_500[0].get('timestamp', 0))
            t2 = datetime.fromtimestamp(recent_500[-1].get('timestamp', 0))
            hours = (t1 - t2).total_seconds() / 3600
            print(f"Period: {hours:.1f}h | Freq: {len(recent_500)/hours:.1f} trade/h")
        except:
            pass
    
    # Last trade
    try:
        last_ts = trades[0].get('timestamp', 0)
        last_dt = datetime.fromtimestamp(last_ts)
        mins_ago = (datetime.now() - last_dt).total_seconds() / 60
        print(f"Last trade: {mins_ago:.0f} min ago")
    except:
        pass
    
    # === PATTERN ===
    print("\n" + "=" * 60)
    print("🎯 PATTERN (ultimi 500)")
    print("=" * 60)
    
    # Price distribution
    low = sum(1 for p in prices_500 if p < 0.5)
    high = sum(1 for p in prices_500 if p >= 0.5)
    print(f"Price <0.5 (DOWN): {low}/{len(prices_500)} ({low/len(prices_500)*100:.0f}%)")
    print(f"Price ≥0.5 (UP):   {high}/{len(prices_500)} ({high/len(prices_500)*100:.0f}%)")
    
    # Size clusters
    small = sum(1 for s in sizes_500 if s < 20)
    medium = sum(1 for s in sizes_500 if 20 <= s < 50)
    large = sum(1 for s in sizes_500 if s >= 50)
    print(f"\nSize <$20: {small} | $20-50: {medium} | >$50: {large}")
    
    # Detect doubles (same timestamp)
    ts_count = defaultdict(int)
    for t in recent_500:
        ts_count[t.get('timestamp')] += 1
    doubles = sum(1 for c in ts_count.values() if c >= 2)
    print(f"Trade doppi/multipli: {doubles} eventi")
    
    # === VERDICT ===
    print("\n" + "=" * 60)
    print("✅ VERDICT")
    print("=" * 60)
    
    avg_size = sum(sizes_500) / len(sizes_500)
    
    if len(buys_500) < 10:
        print("⚠️  SKIP - Troppo pochi trade")
    elif len(sells_500) > len(buys_500) * 0.5:
        print("⚠️  CAUTELA - Molti SELL (non hold-to-expiry)")
    elif avg_size > 200:
        print("⚠️  WHALE - Size molto alte")
        print(f"    Usa copy_ratio ≤ 0.2")
    elif avg_size > 100:
        print("✅ INTERESSANTE - Size grandi")
        print(f"    Usa copy_ratio: 0.3-0.5")
    elif avg_size > 30:
        print("✅ OTTIMO - Size medie")
        print(f"    Usa copy_ratio: 0.5-1.0")
    else:
        print("✅ BUONO - Size piccole")
        print(f"    Usa copy_ratio: 1.0")
    
    # Config suggestion
    print("\n" + "=" * 60)
    print("⚙️  CONFIG BOT2")
    print("=" * 60)
    print(f"""
TargetWallet(
    address="{wallet}",
    name="Trader_{wallet[:8]}",
    copy_ratio={0.3 if avg_size > 50 else 0.5 if avg_size > 30 else 1.0},
    enabled=True
)
""")
    
    print(f"capital_per_trade = {min(avg_size * 0.3, 20):.1f}\n")

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_wallet.py <WALLET_ADDRESS>")
        return
    
    analyze(sys.argv[1])

if __name__ == "__main__":
    main()
