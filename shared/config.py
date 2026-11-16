"""
Configuration module for Polymarket Bot
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Main configuration class"""
    
    # Polymarket API Configuration (opzionale)
    POLYMARKET_API_KEY = os.getenv('POLYMARKET_API_KEY')
    POLYMARKET_SECRET = os.getenv('POLYMARKET_SECRET')
    POLYMARKET_PASSPHRASE = os.getenv('POLYMARKET_PASSPHRASE')
    
    # Wallet Configuration
    PRIVATE_KEY = os.getenv('PRIVATE_KEY')
    WALLET_ADDRESS = os.getenv('WALLET_ADDRESS')
    
    # Chain Configuration
    CHAIN_ID = int(os.getenv('CHAIN_ID', '137'))
    RPC_URL = os.getenv('RPC_URL', 'https://polygon-rpc.com')
    
    # Builder/Relayer Configuration (for gasless redeem)
    # Note: POLYMARKET_API_KEY might be the same as BUILDER_API_KEY
    BUILDER_API_KEY = os.getenv('BUILDER_API_KEY', os.getenv('POLYMARKET_API_KEY', ''))
    BUILDER_SECRET = os.getenv('BUILDER_SECRET', os.getenv('POLYMARKET_SECRET', ''))
    BUILDER_PASSPHRASE = os.getenv('BUILDER_PASSPHRASE', os.getenv('POLYMARKET_PASSPHRASE', ''))
    RELAYER_URL = os.getenv('RELAYER_URL', 'https://relayer-v2.polymarket.com/')
    
    # Bot1 BTC Configuration
    BOT1_ENTRY_CAPITAL = float(os.getenv('BOT1_ENTRY_CAPITAL', '100.0'))
    BOT1_ENTRY_PRICE = float(os.getenv('BOT1_ENTRY_PRICE', '0.76'))
    BOT1_STOP_LOSS = float(os.getenv('BOT1_STOP_LOSS', '0.45'))
    
    # Take Profit: può essere None per disabilitarlo
    _tp = os.getenv('BOT1_TAKE_PROFIT', '0.93')
    BOT1_TAKE_PROFIT = None if _tp.lower() in ['none', 'null', ''] else float(_tp)
    
    # Trading Strategy Configuration
    BOT1_ONE_TRADE_PER_MARKET = os.getenv('BOT1_ONE_TRADE_PER_MARKET', 'true').lower() == 'true'
    BOT1_MARKET_START_DELAY = int(os.getenv('BOT1_MARKET_START_DELAY', '90'))  # secondi
    BOT1_CLAIM_INTERVAL = int(os.getenv('BOT1_CLAIM_INTERVAL', '30'))  # minuti
    BOT1_WIN_THRESHOLD = float(os.getenv('BOT1_WIN_THRESHOLD', '0.92'))  # prezzo per considerare probabile win
    BOT1_WS_URL = os.getenv('BOT1_WS_URL', 'wss://ws-subscriptions-clob.polymarket.com/ws/market')
    BOT1_WS_STALE_SECONDS = float(os.getenv('BOT1_WS_STALE_SECONDS', '3.0'))
    BOT1_PRICE_LOG_ENABLED = os.getenv('BOT1_PRICE_LOG_ENABLED', 'true').lower() == 'true'
    
    # === TELEGRAM + CIRCUIT BREAKER ===
    BOT1_TELEGRAM_ENABLED = os.getenv('BOT1_TELEGRAM_ENABLED', 'true').lower() == 'true'
    BOT1_TELEGRAM_TOKEN = os.getenv('BOT1_TELEGRAM_TOKEN', os.getenv('TELEGRAM_TOKEN', ''))
    BOT1_TELEGRAM_CHAT_ID = os.getenv('BOT1_TELEGRAM_CHAT_ID', os.getenv('TELEGRAM_CHAT_ID', ''))
    BOT1_CB_ALERT_DAILY_LOSS = float(os.getenv('BOT1_CB_ALERT_DAILY_LOSS', '150'))
    BOT1_CB_ALERT_DRAWDOWN = float(os.getenv('BOT1_CB_ALERT_DRAWDOWN', '263'))
    BOT1_CB_ALERT_STREAK = int(os.getenv('BOT1_CB_ALERT_STREAK', '36'))
    BOT1_CB_KILL_DAILY_LOSS = float(os.getenv('BOT1_CB_KILL_DAILY_LOSS', '200'))
    BOT1_CB_KILL_DRAWDOWN = float(os.getenv('BOT1_CB_KILL_DRAWDOWN', '350'))
    BOT1_CB_KILL_STREAK = int(os.getenv('BOT1_CB_KILL_STREAK', '48'))

    # === MAKER ORDER STRATEGY ===
    BOT1_POST_ONLY = os.getenv('BOT1_POST_ONLY', 'true').lower() == 'true'
    BOT1_ORDER_POLL_SECONDS = float(os.getenv('BOT1_ORDER_POLL_SECONDS', '1.0'))
    BOT1_ENTRY_TIMEOUT_SECONDS = float(os.getenv('BOT1_ENTRY_TIMEOUT_SECONDS', '45'))
    BOT1_ENTRY_TAKER_FALLBACK = os.getenv('BOT1_ENTRY_TAKER_FALLBACK', 'false').lower() == 'true'
    BOT1_SL_MAKER_OFFSET = float(os.getenv('BOT1_SL_MAKER_OFFSET', '0.01'))
    BOT1_SL_MAKER_TIMEOUT = float(os.getenv('BOT1_SL_MAKER_TIMEOUT', '8'))
    # Min seconds between CLOB `update_balance_allowance` calls (any asset).
    # Too low → Cloudflare 1015 / 429 on clob.polymarket.com (balance-driven bot).
    BOT1_BALANCE_UPDATE_MIN_INTERVAL = float(os.getenv('BOT1_BALANCE_UPDATE_MIN_INTERVAL', '3.0'))
    # After a rate-limit response, skip updates for this many seconds (backoff).
    BOT1_BALANCE_COOLDOWN_ON_429 = float(os.getenv('BOT1_BALANCE_COOLDOWN_ON_429', '90.0'))

    # === RECOVERY TRADE STRATEGY ===
    # Strategia per recuperare dopo stop loss entrando sull'outcome opposto
    BOT1_RECOVERY_ENABLED = os.getenv('BOT1_RECOVERY_ENABLED', 'true').lower() == 'true'
    BOT1_RECOVERY_MULTIPLIER = float(os.getenv('BOT1_RECOVERY_MULTIPLIER', '2.0'))  # capitale x2
    BOT1_RECOVERY_TIME_WINDOW = int(os.getenv('BOT1_RECOVERY_TIME_WINDOW', '150'))  # ultimi 150s
    BOT1_RECOVERY_MAX_ATTEMPTS = int(os.getenv('BOT1_RECOVERY_MAX_ATTEMPTS', '1'))  # max 1 recovery
    BOT1_RECOVERY_ENTRY_PRICE = float(os.getenv('BOT1_RECOVERY_ENTRY_PRICE', '0.85'))  # entry recovery
    BOT1_RECOVERY_STOP_LOSS = float(os.getenv('BOT1_RECOVERY_STOP_LOSS', '0.56'))  # SL recovery

    # Bot3 BTC Configuration (threshold strategy, latency-focused)
    BOT3_ENTRY_CAPITAL = float(os.getenv('BOT3_ENTRY_CAPITAL', '15'))
    BOT3_ENTRY_PRICE = float(os.getenv('BOT3_ENTRY_PRICE', '0.35'))
    BOT3_TAKE_PROFIT = float(os.getenv('BOT3_TAKE_PROFIT', '0.49'))
    BOT3_STOP_LOSS = float(os.getenv('BOT3_STOP_LOSS', '0.30'))
    BOT3_CHECK_INTERVAL = float(os.getenv('BOT3_CHECK_INTERVAL', '0.25'))
    BOT3_LOG_EVERY_SECONDS = float(os.getenv('BOT3_LOG_EVERY_SECONDS', '10'))
    BOT3_MARKET_START_DELAY = int(os.getenv('BOT3_MARKET_START_DELAY', '0'))
    BOT3_ONE_TRADE_PER_MARKET = os.getenv('BOT3_ONE_TRADE_PER_MARKET', 'true').lower() == 'true'
    BOT3_REQUIRE_ENTRY_CROSSING = os.getenv('BOT3_REQUIRE_ENTRY_CROSSING', 'true').lower() == 'true'
    BOT3_ENTRY_CROSSING_TICKS = int(os.getenv('BOT3_ENTRY_CROSSING_TICKS', '2'))  # consecutive ticks below threshold required
    BOT3_PRICE_LOG_ENABLED = os.getenv('BOT3_PRICE_LOG_ENABLED', 'true').lower() == 'true'
    BOT3_TRADING_ENABLED = os.getenv('BOT3_TRADING_ENABLED', 'true').lower() == 'true'
    BOT3_BUY_BUFFER = float(os.getenv('BOT3_BUY_BUFFER', '1.00'))
    BOT3_ORDER_WAIT_SECONDS = float(os.getenv('BOT3_ORDER_WAIT_SECONDS', '0.8'))
    BOT3_EXIT_ORDER_WAIT_SECONDS = float(os.getenv('BOT3_EXIT_ORDER_WAIT_SECONDS', '0.3'))  # più basso: exit veloci
    BOT3_ORDER_TYPE = os.getenv('BOT3_ORDER_TYPE', 'FAK').upper()
    BOT3_ORDER_TYPE_ENTRY = os.getenv('BOT3_ORDER_TYPE_ENTRY', 'GTC').upper()
    BOT3_ORDER_TYPE_EXIT = os.getenv('BOT3_ORDER_TYPE_EXIT', 'FAK').upper()
    BOT3_USE_WEBSOCKET = os.getenv('BOT3_USE_WEBSOCKET', 'true').lower() == 'true'
    BOT3_WS_URL = os.getenv('BOT3_WS_URL', 'wss://ws-subscriptions-clob.polymarket.com/ws/market')
    BOT3_WS_STALE_SECONDS = float(os.getenv('BOT3_WS_STALE_SECONDS', '3.0'))
    BOT3_EXIT_MAX_RETRIES = int(os.getenv('BOT3_EXIT_MAX_RETRIES', '6'))
    BOT3_EXIT_RETRY_INTERVAL = float(os.getenv('BOT3_EXIT_RETRY_INTERVAL', '0.30'))
    BOT3_EXIT_REMAINING_TOLERANCE = float(os.getenv('BOT3_EXIT_REMAINING_TOLERANCE', '0.0001'))
    BOT3_POST_ENTRY_GRACE_SECONDS = float(os.getenv('BOT3_POST_ENTRY_GRACE_SECONDS', '2.0'))
    BOT3_POST_ENTRY_SL_TICKS = int(os.getenv('BOT3_POST_ENTRY_SL_TICKS', '8'))  # ticks before SL is active; TP always active
    BOT3_EXIT_BALANCE_RETRY_SECONDS = float(os.getenv('BOT3_EXIT_BALANCE_RETRY_SECONDS', '1.2'))
    BOT3_EXIT_BALANCE_MAX_WAIT_SECONDS = float(os.getenv('BOT3_EXIT_BALANCE_MAX_WAIT_SECONDS', '20.0'))
    BOT3_EXIT_REQUIRE_SPENDABLE_BALANCE = os.getenv('BOT3_EXIT_REQUIRE_SPENDABLE_BALANCE', 'true').lower() == 'true'
    
    # Market Configuration
    BTC_5MIN_MARKET_ID = os.getenv('BTC_5MIN_MARKET_ID')
    TOKEN_ID = os.getenv('TOKEN_ID')
    
    @classmethod
    def validate(cls):
        """Validate that all required configuration is present"""
        required_fields = [
            'PRIVATE_KEY',
            'WALLET_ADDRESS',
            'BTC_5MIN_MARKET_ID',
            'TOKEN_ID'
        ]
        
        missing = []
        for field in required_fields:
            if not getattr(cls, field):
                missing.append(field)
        
        if missing:
            raise ValueError(f"❌ Mancano questi campi nel file .env: {', '.join(missing)}")
        
        return True


if __name__ == '__main__':
    # Test configuration
    print("=" * 60)
    print("🔧 VERIFICA CONFIGURAZIONE")
    print("=" * 60)
    try:
        Config.validate()
        print("\n✅ Configurazione valida!\n")
        print(f"📍 Wallet: {Config.WALLET_ADDRESS}")
        print(f"🎯 Market ID: {Config.BTC_5MIN_MARKET_ID}")
        print(f"🪙 Token ID: {Config.TOKEN_ID}")
        print(f"\n💰 Capitale: ${Config.BOT1_ENTRY_CAPITAL}")
        print(f"📈 Entry: {Config.BOT1_ENTRY_PRICE}")
        print(f"🛑 Stop Loss: {Config.BOT1_STOP_LOSS}")
        print(f"🎯 Take Profit: {Config.BOT1_TAKE_PROFIT}")
        print("\n" + "=" * 60)
        print("✅ Tutto pronto! Puoi avviare: python bot1_btc.py")
        print("=" * 60)
    except ValueError as e:
        print(f"\n❌ {e}\n")
        print("📝 Compila il file .env con le tue credenziali:")
        print("   1. PRIVATE_KEY (da MetaMask)")
        print("   2. WALLET_ADDRESS (da MetaMask)")
        print("   3. BTC_5MIN_MARKET_ID (usa: python find_markets.py)")
        print("   4. TOKEN_ID (usa: python find_markets.py)")
        print("=" * 60)
