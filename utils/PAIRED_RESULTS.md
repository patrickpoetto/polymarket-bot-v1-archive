# Paired MM + OSL v2 — risultati (30 giorni, BTC 5m)

## Strategia v2 (implementata)

### Idea
Quando `bid_UP + bid_DN ≤ 0.97`, il libro offre un **arb strutturale**: comprare entrambe le gambe **ai bid** costa meno di $1; a scadenza una paga $1 → margine ≈ `1 − bid_sum` + rebate maker.

### Regole
1. **Trigger**: `bid_sum ≤ 0.97` (e `≤ max_pair_cost` 0.99), cooldown 3s, max 40 trigger/mercato, max 50 share abbinate.
2. **Acquisizione coppia**: BUY immediato UP@bid + DN@bid (maker rebate), se OSL non blocca nessuna gamba.
3. **OSL**: niente BUY sulla gamba con mid più basso di ≥ 1¢; se non si può completare la coppia, non si compra il loser.
4. **Niente rebalance continuo** in modalità `acquire_pair_at_bid` — si tiene la coppia fino a scadenza.
5. **Orphan** (solo modalità limit maker): sellback solo gamba loser; winner tenuto se `hold_winner_orphan`.
6. **Chiusura**: liquida eccesso **loser**; tiene eccesso **winner** fino a $1.

### Avvio
```bash
source venv/bin/activate
PYTHONPATH=. python -m bot6_pairs.run          # paper/live
PYTHONPATH=. python utils/mm_simulate_paired.py --max-days 30
```

---

## Risultati 30d (8038 mercati) — config v2

| Metrica | Valore |
|---------|--------|
| **PnL netto totale** | **−$24,80** |
| Media / mercato | −$0,0031 |
| Mediana / mercato | $0 |
| PnL puro | −$100,40 |
| Rebate | +$75,59 |
| Mercati PnL > 0 | 1 190 (14,8%) |
| Giorni positivi | ~misto (media giornaliera −$0,83) |

### Coppie (obiettivo principale)
| Metrica | Valore |
|---------|--------|
| **Coppie aperte** | **389** (~0,05/mercato) |
| **Costo medio coppia** | **$0,946** → margine teorico **~5,4¢**/share + rebate |
| Share abbinate a scadenza | 4 390 (~0,55/mercato) |
| Orphan sellback | 160 |
| Rebalance | 0 (disabilitati in v2) |

### Confronto versioni
| Versione | PnL 30d | Coppie | Note |
|----------|---------|--------|------|
| v1 (ask trigger, orphan 2s) | −$26 | 0 | quasi solo micro-cicli |
| v2 sbagliata (rebalance ON) | −$10 500 | 0 | churn OSL/rebalance |
| **v2 acquire @ bid (0.97)** | **−$25** | **389** | arb strutturale funziona |

---

## Parametri `.env` consigliati

```env
BOT6_PAIRS_BID_SUM_TRIGGER=0.97
BOT6_PAIRS_MAX_PAIR_COST=0.99
BOT6_PAIRS_ACQUIRE_AT_BID=true
BOT6_PAIRS_OSL_GAP=0.01
BOT6_PAIRS_ORPHAN_TIMEOUT=8
BOT6_PAIRS_MAX_TRIGGERS=40
BOT6_PAIRS_TRIGGER_COOLDOWN=3
```

---

## Prossimi passi
- Abbassare trigger a 0.96 per meno trade ma margine più alto per coppia.
- Paper run 24h e confronto fill reali vs sim.
- Ridurre `final_unpaired` (~1,2 sh) con OSL più stretto o solo trigger quando mid quasi simmetrici.
