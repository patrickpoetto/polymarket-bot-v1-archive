#!/usr/bin/env python3
"""
Script manuale per reclamare tutte le posizioni chiuse su Polymarket
"""
import logging
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.redeem_positions import PositionRedeemer
from shared.config import Config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('manual_claim.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    """Esegui claim manuale di tutte le posizioni"""
    try:
        logger.info("=" * 60)
        logger.info("💰 CLAIM MANUALE POSIZIONI CHIUSE")
        logger.info("=" * 60)
        
        # Initialize redeemer
        logger.info("\n🔐 Inizializzazione redeemer...")
        redeemer = PositionRedeemer()
        
        # Get market ID from config (o inserisci manualmente)
        logger.info("\n📊 Controllo market corrente...")
        market_id = Config.BTC_5MIN_MARKET_ID
        logger.info(f"   Market ID: {market_id}")
        
        # Check if current market is resolved
        is_resolved = redeemer.check_payout_status(market_id)
        logger.info(f"   Stato: {'✅ Risolto' if is_resolved else '⏳ Non ancora risolto'}")
        
        if is_resolved:
            logger.info("\n💰 Tento redeem del market corrente...")
            success = redeemer.redeem_position(market_id)
            
            if success:
                logger.info("✅ Claim completato con successo!")
            else:
                logger.warning("⚠️  Claim non riuscito")
        else:
            logger.info("\n⏳ Market non ancora risolto")
            logger.info("💡 Puoi aggiungere altri market IDs manualmente nel codice")
        
        # Esempio: puoi aggiungere più market IDs qui
        # other_markets = [
        #     "0x...",  # Market ID 1
        #     "0x...",  # Market ID 2
        # ]
        # if other_markets:
        #     logger.info(f"\n🔄 Tento redeem di {len(other_markets)} mercati aggiuntivi...")
        #     redeemer.redeem_all_positions(other_markets)
        
        logger.info("\n" + "=" * 60)
        logger.info("✅ Script completato")
        logger.info("=" * 60)
        
    except KeyboardInterrupt:
        logger.info("\n⚠️  Script interrotto dall'utente")
    except Exception as e:
        logger.error(f"\n❌ Errore: {e}", exc_info=True)


if __name__ == '__main__':
    main()
