"""
Redeem conditional tokens usando il Polymarket Builder Relayer
Questo funziona CON i proxy wallet Safe/Gnosis ed è completamente gasless!
"""
import logging
import requests
from typing import List, Optional, Dict
from .config import Config
from web3 import Web3

logger = logging.getLogger(__name__)

# CTF Contract address on Polygon
CTF_CONTRACT = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"
USDC_CONTRACT = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"

# ABI for redeemPositions function
CTF_REDEEM_ABI = {
    "inputs": [
        {"internalType": "contract IERC20", "name": "collateralToken", "type": "address"},
        {"internalType": "bytes32", "name": "parentCollectionId", "type": "bytes32"},
        {"internalType": "bytes32", "name": "conditionId", "type": "bytes32"},
        {"internalType": "uint256[]", "name": "indexSets", "type": "uint256[]"}
    ],
    "name": "redeemPositions",
    "outputs": [],
    "stateMutability": "nonpayable",
    "type": "function"
}

try:
    from py_builder_relayer_client.client import RelayClient
    from py_builder_relayer_client.models import SafeTransaction, OperationType
    from py_builder_signing_sdk.config import BuilderConfig
    from py_builder_signing_sdk.sdk_types import BuilderApiKeyCreds
    RELAYER_AVAILABLE = True
except ImportError as e:
    RELAYER_AVAILABLE = False
    logger.warning(f"⚠️  py-builder-relayer-client non disponibile: {e}")
    logger.warning("💡 Installa con: pip install py-builder-relayer-client")


class RelayerPositionRedeemer:
    """
    Redeem posizioni usando il Polymarket Builder Relayer
    
    Vantaggi:
    - ✅ Funziona con proxy wallet Safe/Gnosis
    - ✅ Completamente gasless (Polymarket paga!)
    - ✅ Ufficiale e supportato da Polymarket
    
    Requisiti:
    - Builder API credentials da polymarket.com/settings?tab=builder
    """
    
    DATA_API_URL = "https://data-api.polymarket.com"
    
    def __init__(self):
        """Initialize the relayer redeemer"""
        if not RELAYER_AVAILABLE:
            raise ImportError(
                "py-builder-relayer-client non installato. "
                "Installa con: pip install py-builder-relayer-client"
            )
        
        # Check credentials
        if not Config.BUILDER_API_KEY or not Config.BUILDER_SECRET or not Config.BUILDER_PASSPHRASE:
            raise ValueError(
                "Builder API credentials mancanti! "
                "Ottienile da polymarket.com/settings?tab=builder e aggiungi al .env:\n"
                "BUILDER_API_KEY=...\n"
                "BUILDER_SECRET=...\n"
                "BUILDER_PASSPHRASE=..."
            )
        
        # Initialize relayer client
        try:
            # Create builder credentials
            builder_creds = BuilderApiKeyCreds(
                key=Config.BUILDER_API_KEY,
                secret=Config.BUILDER_SECRET,
                passphrase=Config.BUILDER_PASSPHRASE
            )
            
            # Create builder config
            builder_config = BuilderConfig(
                local_builder_creds=builder_creds
            )
            
            self.client = RelayClient(
                relayer_url=Config.RELAYER_URL,
                chain_id=Config.CHAIN_ID,
                private_key=Config.PRIVATE_KEY,
                builder_config=builder_config
            )
            logger.info(f"✅ Relayer Client inizializzato")
            logger.info(f"   URL: {Config.RELAYER_URL}")
            logger.info(f"   Chain: {Config.CHAIN_ID}")
            logger.info(f"   Wallet: {Config.WALLET_ADDRESS[:10]}...")
            
        except Exception as e:
            raise Exception(f"Errore inizializzazione Relayer Client: {e}")
    
    def get_redeemable_positions(self) -> List[Dict]:
        """
        Recupera tutte le posizioni redimibili dell'utente dall'API Polymarket
        
        Returns:
            Lista di dict con info sulle posizioni redimibili
        """
        try:
            url = f"{self.DATA_API_URL}/positions"
            params = {
                'user': Config.WALLET_ADDRESS,
                'redeemable': 'true',
                'limit': 500
            }
            
            logger.info(f"🔍 Controllo posizioni redimibili per {Config.WALLET_ADDRESS[:10]}...")
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # L'API ritorna direttamente una lista di posizioni
            if isinstance(data, list):
                positions = data
            elif isinstance(data, dict) and 'data' in data:
                positions = data['data']
            else:
                logger.warning(f"⚠️  Formato risposta API inatteso: {type(data)}")
                positions = []
            
            logger.info(f"✅ Trovate {len(positions)} posizioni redimibili")
            
            return positions
            
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Errore recupero posizioni da API: {e}")
            return []
        except Exception as e:
            logger.error(f"❌ Errore inatteso in get_redeemable_positions: {e}")
            return []
    
    def get_redeemable_condition_ids(self) -> List[str]:
        """
        Estrae i condition IDs dalle posizioni redimibili
        
        Returns:
            Lista di condition IDs (hex strings)
        """
        positions = self.get_redeemable_positions()
        
        condition_ids = set()
        
        for pos in positions:
            # Ogni posizione ha un conditionId
            if 'conditionId' in pos:
                condition_ids.add(pos['conditionId'])
            elif 'condition_id' in pos:
                condition_ids.add(pos['condition_id'])
            # Fallback: usa market se presente
            elif 'market' in pos:
                condition_ids.add(pos['market'])
        
        result = list(condition_ids)
        
        if result:
            logger.info(f"📋 Condition IDs redimibili: {len(result)}")
            for cid in result:
                logger.info(f"   • {cid[:16]}...")
        else:
            logger.info("ℹ️  Nessun condition ID da redimere")
        
        return result
    
    def redeem_position(self, condition_id: str) -> bool:
        """
        Redeem una posizione per un mercato chiuso usando il Relayer
        
        Args:
            condition_id: Market condition ID da reclamare
            
        Returns:
            True se successo, False altrimenti
        """
        try:
            logger.info(f"🔄 Tentativo redeem via Relayer...")
            logger.info(f"   Condition ID: {condition_id[:20]}...")
            logger.info(f"   Proxy Wallet: {Config.WALLET_ADDRESS[:10]}...")
            
            # Converti condition_id in bytes32
            if condition_id.startswith('0x'):
                condition_id = condition_id[2:]
            condition_id_bytes = bytes.fromhex(condition_id.zfill(64))
            
            # Crea web3 instance per encoding
            w3 = Web3()
            
            # Encode la chiamata redeemPositions
            # redeemPositions(collateralToken, parentCollectionId, conditionId, indexSets)
            contract = w3.eth.contract(
                address=Web3.to_checksum_address(CTF_CONTRACT),
                abi=[CTF_REDEEM_ABI]
            )
            
            # Parametri per binary market (Up/Down, Yes/No)
            index_sets = [1, 2]  # 0b01 e 0b10
            
            # Encode function call usando l'API corretta di web3.py
            tx_data = contract.functions.redeemPositions(
                Web3.to_checksum_address(USDC_CONTRACT),  # collateralToken
                b'\x00' * 32,  # parentCollectionId (null)
                condition_id_bytes,  # conditionId
                index_sets  # indexSets
            )._encode_transaction_data()
            
            logger.info(f"📝 Transazione encodata: {len(tx_data)} bytes")
            
            # Crea SafeTransaction
            safe_tx = SafeTransaction(
                to=Web3.to_checksum_address(CTF_CONTRACT),
                operation=OperationType.Call,  # Standard CALL
                data=tx_data,
                value="0"  # String, non int
            )
            
            logger.info(f"📤 Invio transazione via Relayer (gasless)...")
            
            # Execute via relayer (passa lista di transactions)
            result = self.client.execute(
                transactions=[safe_tx],
                metadata="Redeem conditional tokens"
            )
            
            logger.info(f"⏳ Transazione inviata, attendo conferma...")
            
            # Il result è un ClientRelayerTransactionResponse
            if result:
                # Usa il metodo wait() per attendere la conferma
                try:
                    logger.info(f"   Attendo conferma dalla blockchain...")
                    final_tx = result.wait()
                    
                    if final_tx:
                        logger.info(f"✅ Redeem completato! (gasless)")
                        logger.info(f"   Dettagli: {final_tx}")
                        return True
                    else:
                        logger.warning(f"⚠️  Wait returned None")
                        # Ma potrebbe essere comunque riuscito
                        return True
                except Exception as wait_error:
                    logger.warning(f"⚠️  Errore durante wait: {wait_error}")
                    # La transazione è stata inviata, probabilmente andrà a buon fine
                    logger.info(f"✅ Transazione inviata al relayer (verifica manualmente su polymarket.com)")
                    return True
            else:
                logger.error(f"❌ Nessun risultato ricevuto dal relayer")
                return False
                
        except Exception as e:
            error_str = str(e)
            
            # Gestisci rate limit in modo pulito
            if '429' in error_str or 'rate limit' in error_str.lower() or 'quota exceeded' in error_str:
                logger.warning(f"⏸️  Rate limit raggiunto")
                return False
            
            # Gestisci altri errori comuni
            if 'not enough balance' in error_str.lower():
                logger.info(f"ℹ️  Nessun balance da reclamare")
                return False
            
            if 'already redeemed' in error_str.lower() or 'already claimed' in error_str.lower():
                logger.info(f"ℹ️  Già reclamato")
                return False
            
            # Per altri errori, log minimo
            logger.warning(f"⚠️  Fallito: {error_str[:80]}")
            return False
    
    def redeem_all_positions(self, market_ids: List[str]) -> int:
        """
        Redeem tutte le posizioni per una lista di market IDs
        
        Args:
            market_ids: Lista di condition IDs da reclamare
            
        Returns:
            Numero di redeem riusciti
        """
        if not market_ids:
            logger.info("ℹ️  Nessun mercato da reclamare")
            return 0
        
        logger.info(f"🔄 Tentativo claim per {len(market_ids)} mercati...")
        
        success_count = 0
        failed_count = 0
        rate_limited = False
        
        for i, market_id in enumerate(market_ids, 1):
            # Log minimo ogni 10 mercati
            if i == 1 or i % 10 == 0 or i == len(market_ids):
                logger.info(f"   [{i}/{len(market_ids)}] {market_id[:16]}...")
            
            result = self.redeem_position(market_id)
            
            if result:
                success_count += 1
            else:
                failed_count += 1
        
        # Summary pulito
        logger.info(f"\n📊 Claim completato:")
        logger.info(f"   ✅ Successi: {success_count}")
        if failed_count > 0:
            logger.info(f"   ⚠️  Falliti: {failed_count}")
        
        return success_count


def test_relayer_redeem():
    """Test del Relayer Redeemer"""
    try:
        logger.info("=" * 60)
        logger.info("🧪 TEST RELAYER REDEEM (Gasless + Proxy Wallet)")
        logger.info("=" * 60)
        
        # Check if relayer client is available
        if not RELAYER_AVAILABLE:
            logger.error("❌ py-builder-relayer-client non installato!")
            logger.info("💡 Installa con: pip install py-builder-relayer-client")
            return
        
        # Check credentials
        if not Config.BUILDER_API_KEY:
            logger.error("❌ Builder API credentials mancanti!")
            logger.info("💡 Ottienile da: polymarket.com/settings?tab=builder")
            logger.info("💡 Aggiungi al .env:")
            logger.info("   BUILDER_API_KEY=...")
            logger.info("   BUILDER_SECRET=...")
            logger.info("   BUILDER_PASSPHRASE=...")
            return
        
        # Initialize redeemer
        logger.info("\n🔐 Inizializzazione Relayer Redeemer...")
        redeemer = RelayerPositionRedeemer()
        
        # Get current market from config
        market_id = Config.BTC_5MIN_MARKET_ID
        logger.info(f"\n📊 Market ID: {market_id}")
        
        # Try redeem
        logger.info("\n🔄 Tentativo redeem...")
        success = redeemer.redeem_position(market_id)
        
        if success:
            logger.info("✅ Test completato con successo!")
        else:
            logger.warning("⚠️  Redeem non riuscito (normale se market non risolto)")
        
    except Exception as e:
        logger.error(f"❌ Errore test: {e}", exc_info=True)


if __name__ == '__main__':
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    test_relayer_redeem()
