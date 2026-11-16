# Utility Scripts

Script standalone per operazioni manuali, testing e configurazione bot.

**IMPORTANTE**: Questi script vanno eseguiti dalla **root del progetto**, non dalla cartella `utils/`:

```bash
# ✅ Corretto (dalla root)
python utils/approve_allowance.py

# ❌ Sbagliato
cd utils && python approve_allowance.py
```

## 🤖 Scripts per Bot Configuration

### get_active_tokens.py
**Trova token_id dei mercati BTC 5min attivi** per configurare bot2_copytrade_ws.py

```bash
python utils/get_active_tokens.py
```

Output:
```python
monitored_tokens = [
    "0xabc123...",  # Up
    "0xdef456...",  # Down
]
```

Copia questo output e incollalo in `bot2_copytrade_ws.py`.

### analyze_wallet.py
**Analizza performance di un wallet** per decidere se copiarlo.
Mostra trade recenti, size media, frequenza, e raccomandazioni.

```bash
python utils/analyze_wallet.py <WALLET_ADDRESS> [limit]
```

Esempio:
```bash
python utils/analyze_wallet.py 0x1234567890abcdef 100
```

Output include:
- 📊 Stats base (BUY/SELL ratio)
- 💰 Size analysis (min/max/media)
- 📊 Price range
- 🎯 Mercati più tradati
- ⏱️ Frequenza trading
- 💡 Raccomandazione (copiare o no)

## 🛠️ Scripts per Operations

### approve_allowance.py
Approva l'allowance dei token per il contratto Exchange di Polymarket.
**Da eseguire una volta** prima di iniziare a tradare.

```bash
python utils/approve_allowance.py
```

Questo script:
- Approva il trasferimento di USDC al contratto Exchange
- Necessario per poter piazzare ordini
- Va eseguito solo 1 volta (o quando si resetta l'allowance)

### manual_claim.py
Reclama manualmente le posizioni vincenti.
Alternativa al claim automatico dei bot.

```bash
python utils/manual_claim.py
```

Questo script:
- Cerca tutte le posizioni redimibili del wallet
- Le reclama in batch usando il relayer (gasless)
- Utile se il bot non ha fatto auto-claim o per test

### poly_parser.py
Fetcha e analizza lo storico completo delle operazioni dal Data API di Polymarket.
Salva i dati in formato CSV e JSON in `archived_data/`.

```bash
python utils/poly_parser.py
```

Questo script:
- Scarica TUTTE le attività del wallet (trade + redeems)
- Calcola statistiche (P&L, win rate, etc.)
- Salva i risultati in `archived_data/polymarket_history_api.*`
- Utile per analisi storiche e performance review

## Configurazione

Tutti gli script usano le configurazioni da `.env` nel root del progetto:

```env
PRIVATE_KEY=0x...
WALLET_ADDRESS=0x...
CHAIN_ID=137
```

## Note Tecniche

Gli script in `utils/` includono automaticamente il path al progetto:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.config import Config
```

Questo permette di importare i moduli da `shared/` anche quando gli script
sono in una sottocartella.
