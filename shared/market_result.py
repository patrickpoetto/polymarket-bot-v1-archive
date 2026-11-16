"""
Ottieni il risultato di un mercato Polymarket chiuso
"""
import requests
import logging
from typing import Optional

logger = logging.getLogger(__name__)

CLOB_API_URL = "https://clob.polymarket.com"

def get_market_result(condition_id: str) -> Optional[str]:
    """
    Ottieni il risultato di un mercato chiuso
    
    Args:
        condition_id: ID del mercato
        
    Returns:
        'Yes', 'No', o None se non ancora risolto
    """
    try:
        # Query market info
        url = f"{CLOB_API_URL}/markets/{condition_id}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        market_data = response.json()
        
        # Check if market is resolved
        if 'outcomes' in market_data:
            outcomes = market_data['outcomes']
            
            # Cerca l'outcome vincente (quello con outcome = "Yes" dopo risoluzione)
            for outcome_info in outcomes:
                if outcome_info.get('outcome') == 'Yes':
                    return outcome_info.get('title', 'Unknown')
        
        # Fallback: controlla se c'è un winner field
        if 'winner' in market_data:
            return market_data['winner']
            
        logger.warning(f"Market {condition_id[:10]}... non ancora risolto")
        return None
        
    except Exception as e:
        logger.error(f"Errore recupero risultato market: {e}")
        return None

def check_market_winner(condition_id: str, our_outcome: str) -> Optional[bool]:
    """
    Controlla se abbiamo vinto il mercato
    
    Args:
        condition_id: ID del mercato
        our_outcome: 'Up' o 'Down'
        
    Returns:
        True se vinto, False se perso, None se non ancora risolto
    """
    result = get_market_result(condition_id)
    
    if result is None:
        return None
    
    # Match result con il nostro outcome
    result_lower = result.lower()
    our_outcome_lower = our_outcome.lower()
    
    return result_lower == our_outcome_lower
