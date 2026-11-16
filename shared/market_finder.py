"""
Market Finder - Trova mercati BTC 5min attivi scrapando Polymarket
"""
import requests
from bs4 import BeautifulSoup
import json
import re
import logging
from typing import Optional, Dict
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


def scrape_market_from_url(url: str) -> Optional[Dict]:
    """Scrapa i dati da una pagina mercato specifica"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Trova il tag __NEXT_DATA__
        script = soup.find('script', {'id': '__NEXT_DATA__', 'type': 'application/json'})
        
        if not script:
            logger.error("__NEXT_DATA__ non trovato nella pagina")
            return None
        
        data = json.loads(script.string)
        
        # Polymarket usa dehydratedState (React Query)
        page_props = data.get('props', {}).get('pageProps', {})
        
        market = None
        
        # Cerca in dehydratedState
        dehydrated = page_props.get('dehydratedState', {})
        queries = dehydrated.get('queries', [])
        
        for query in queries:
            state_data = query.get('state', {}).get('data', {})
            
            # Cerca markets array
            if 'markets' in state_data and state_data['markets']:
                market = state_data['markets'][0]
                break
            
            # O direttamente come market
            if 'conditionId' in state_data:
                market = state_data
                break
        
        if not market:
            logger.error("Market data non trovato")
            return None
        
        # Estrai i dati necessari
        condition_id = market.get('conditionId')
        clob_token_ids = market.get('clobTokenIds', '[]')
        outcomes = market.get('outcomes', '[]')
        
        if isinstance(clob_token_ids, str):
            clob_token_ids = json.loads(clob_token_ids)
        if isinstance(outcomes, str):
            outcomes = json.loads(outcomes)
        
        # Calcola tempo rimanente
        end_date_str = market.get('endDate')
        minutes_left = 0
        if end_date_str:
            try:
                end_date = datetime.fromisoformat(end_date_str.replace('Z', '+00:00'))
                now = datetime.now(timezone.utc)
                minutes_left = (end_date - now).total_seconds() / 60
            except:
                pass
        
        return {
            'condition_id': condition_id,
            'token_ids': clob_token_ids,
            'outcomes': outcomes,
            'question': market.get('question'),
            'minutes_left': minutes_left
        }
        
    except Exception as e:
        logger.error(f"Errore scraping URL: {e}")
        return None


def find_btc_5min_links_on_homepage() -> list:
    """Trova link ai mercati BTC 5min sulla homepage"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        }
        
        # Prova diverse pagine
        urls_to_try = [
            'https://polymarket.com',
            'https://polymarket.com/browse',
        ]
        
        found_links = []
        
        for base_url in urls_to_try:
            try:
                response = requests.get(base_url, headers=headers, timeout=15)
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Cerca tutti i link che contengono pattern BTC 5min
                links = soup.find_all('a', href=True)
                
                for link in links:
                    href = link.get('href', '')
                    
                    # Pattern: /event/btc-updown-5m-TIMESTAMP o /event/bitcoin-up-or-down-5-min...
                    if '/event/' in href and ('btc' in href.lower() or 'bitcoin' in href.lower()):
                        if '5m' in href or '5-m' in href or '5min' in href:
                            full_url = href if href.startswith('http') else f'https://polymarket.com{href}'
                            if full_url not in found_links:
                                found_links.append(full_url)
                                logger.info(f"Trovato link: {full_url}")
                
            except:
                continue
        
        return found_links
        
    except Exception as e:
        logger.error(f"Errore ricerca homepage: {e}")
        return []


def find_active_5min_market(last_timestamp: int = None) -> Optional[Dict]:
    """
    Trova un mercato BTC 5min attivo
    
    Args:
        last_timestamp: timestamp del mercato precedente (opzionale)
    """
    logger.info("🔍 Cercando mercati BTC 5min attivi...")
    
    # STRATEGIA 1: Predici il prossimo timestamp (+300 secondi)
    if last_timestamp:
        next_timestamp = last_timestamp + 300
        predicted_url = f"https://polymarket.com/event/btc-updown-5m-{next_timestamp}"
        
        logger.info(f"🔮 Provo mercato predetto: {next_timestamp}")
        market_data = scrape_market_from_url(predicted_url)
        
        if market_data and market_data.get('minutes_left', 0) >= 0.4:
            logger.info(f"✅ Mercato predetto trovato: {market_data['question']}")
            logger.info(f"⏰ Tempo rimasto: {market_data['minutes_left']:.1f} minuti")
            market_data['timestamp'] = next_timestamp
            return market_data
        elif market_data:
            logger.warning(f"⚠️  Mercato predetto trovato ma troppo poco tempo ({market_data.get('minutes_left', 0):.1f} min)")
    
    # STRATEGIA 2: Scraping homepage (fallback)
    links = find_btc_5min_links_on_homepage()
    
    if links:
        logger.info(f"Trovati {len(links)} link dalla homepage, verifico...")
        
        # Prende SOLO il primo link
        link = links[0]
        market_data = scrape_market_from_url(link)
        
        if market_data and market_data.get('minutes_left', 0) >= 0.4:
            logger.info(f"✅ Mercato attivo trovato: {market_data['question']}")
            logger.info(f"⏰ Tempo rimasto: {market_data['minutes_left']:.1f} minuti")
            
            # Estrai timestamp dall'URL per future predizioni
            import re
            match = re.search(r'btc-updown-5m-(\d+)', link)
            if match:
                market_data['timestamp'] = int(match.group(1))
            
            return market_data
        elif market_data:
            logger.warning(f"⚠️  Mercato trovato ma troppo poco tempo ({market_data.get('minutes_left', 0):.1f} min), aspetto il prossimo...")
    
    logger.warning("❌ Nessun mercato 5min attivo trovato")
    return None


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    result = find_active_5min_market()
    
    if result:
        print("\n" + "=" * 70)
        print("✅ MERCATO TROVATO")
        print("=" * 70)
        print(f"Question: {result['question']}")
        print(f"Tempo rimasto: {int(result['minutes_left'])} minuti")
        print(f"\nCondition ID: {result['condition_id']}")
        print(f"\nToken IDs:")
        for outcome, token_id in zip(result['outcomes'], result['token_ids']):
            print(f"  {outcome}: {token_id}")
        print("=" * 70)
    else:
        print("\n❌ Nessun mercato attivo trovato")
        print("\nSE VEDI UN MERCATO ATTIVO SU POLYMARKET:")
        print("Copia l'URL completo e dimmi quale è, farò scraping diretto di quella pagina!")
