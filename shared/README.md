# Shared Modules

Moduli condivisi utilizzati da tutti i bot.

## Moduli Disponibili

### config.py
Gestisce la configurazione dell'applicazione caricando i parametri dal file `.env`:
- `PRIVATE_KEY`: Chiave privata MetaMask per firmare transazioni
- `WALLET_ADDRESS`: Indirizzo proxy wallet Polymarket
- `CHAIN_ID`: ID della chain (137 = Polygon)
- Parametri di trading configurabili

**Uso:**
```python
from shared.config import Config

Config.validate()  # Controlla che le variabili siano settate
print(Config.WALLET_ADDRESS)
```

### market_finder.py
Trova mercati BTC attivi con finestra temporale di 5 minuti.

**Uso:**
```python
from shared.market_finder import find_active_5min_market

market_data = find_active_5min_market()
if market_data:
    condition_id = market_data['condition_id']
    token_ids = market_data['token_ids']  # [Up token, Down token]
    outcomes = market_data['outcomes']    # ['Up', 'Down']
```

### market_result.py
Verifica il risultato di un mercato (quale outcome ha vinto).

**Uso:**
```python
from shared.market_result import check_market_winner

winner = check_market_winner(condition_id, token_id)
# Ritorna 'YES', 'NO', o None se non risolto
```

### trade_logger.py
Sistema di logging dedicato per operazioni di trading.
Crea un file separato per ogni sessione con timestamp.

**Uso:**
```python
from shared.trade_logger import TradeLogger

logger = TradeLogger(logs_dir="logs/bot1_btc")
logger.log_entry("Up", 0.76, 10.5)
logger.log_exit("Up", 0.93, profit=1.78, win=True)
```

### redeem_relayer.py
Sistema di redeem gasless tramite relayer di Polymarket.
Preferito quando disponibile perché non richiede MATIC per gas.

**Uso:**
```python
from shared.redeem_relayer import RelayerPositionRedeemer

redeemer = RelayerPositionRedeemer(
    private_key=Config.PRIVATE_KEY,
    wallet_address=Config.WALLET_ADDRESS
)
success = redeemer.redeem_positions([condition_id])
```

### redeem_positions.py
Sistema di redeem diretto on-chain.
Fallback quando il relayer non è disponibile (richiede MATIC per gas).

**Uso:**
```python
from shared.redeem_positions import PositionRedeemer

redeemer = PositionRedeemer(
    private_key=Config.PRIVATE_KEY,
    wallet_address=Config.WALLET_ADDRESS
)
success = redeemer.redeem_positions([condition_id])
```

## Import nei Bot

I bot nella root importano questi moduli con:

```python
from shared.config import Config
from shared.market_finder import find_active_5min_market
from shared.trade_logger import TradeLogger
# etc...
```

## Import tra Moduli Shared

I moduli in `shared/` che importano altri moduli shared usano import relativi:

```python
# Dentro shared/redeem_positions.py
from .config import Config
```
