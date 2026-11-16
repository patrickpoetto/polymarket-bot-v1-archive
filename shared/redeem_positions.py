"""
Redeem conditional tokens for closed Polymarket positions
"""
import logging
from typing import List, Optional
from web3 import Web3
from eth_account import Account
from .config import Config

logger = logging.getLogger(__name__)

# Polygon Mainnet addresses
CTF_CONTRACT = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"
USDC_CONTRACT = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"  # USDC on Polygon

# CTF Contract ABI (only the functions we need)
CTF_ABI = [
    {
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
    },
    {
        "inputs": [
            {"internalType": "address", "name": "owner", "type": "address"},
            {"internalType": "bytes32", "name": "conditionId", "type": "bytes32"}
        ],
        "name": "payoutDenominator",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [
            {"internalType": "address", "name": "owner", "type": "address"},
            {"internalType": "uint256", "name": "id", "type": "uint256"}
        ],
        "name": "balanceOf",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function"
    }
]


class PositionRedeemer:
    """Handle redeeming of conditional tokens for closed markets"""
    
    def __init__(self, private_key: str = None, wallet_address: str = None):
        """Initialize the redeemer with web3 connection"""
        # Use config if not provided
        self.private_key = private_key or Config.PRIVATE_KEY
        self.wallet_address = wallet_address or Config.WALLET_ADDRESS
        
        # Ensure private key has 0x prefix
        if not self.private_key.startswith('0x'):
            self.private_key = '0x' + self.private_key
        
        # Connect to Polygon via public RPC
        rpc_urls = [
            "https://polygon-rpc.com",
            "https://rpc-mainnet.matic.network",
            "https://polygon-mainnet.public.blastapi.io"
        ]
        
        self.w3 = None
        for rpc_url in rpc_urls:
            try:
                w3 = Web3(Web3.HTTPProvider(rpc_url))
                if w3.is_connected():
                    self.w3 = w3
                    logger.info(f"✅ Connesso a Polygon via {rpc_url}")
                    break
            except Exception as e:
                logger.warning(f"Errore connessione a {rpc_url}: {e}")
                continue
        
        if not self.w3:
            raise Exception("❌ Impossibile connettersi a Polygon")
        
        # Setup account
        self.account = Account.from_key(self.private_key)
        self.eoa_address = self.account.address
        logger.info(f"🔐 EOA Account: {self.eoa_address}")
        
        # IMPORTANTE: Per Polymarket PROXY WALLETS
        # - Trading usa CLOB API (off-chain) → funziona con EOA + proxy
        # - Redeem è ON-CHAIN → serve gestione Safe multisig
        if self.account.address.lower() != self.wallet_address.lower():
            logger.warning(f"⚠️  Setup Polymarket Proxy Wallet rilevato:")
            logger.warning(f"   EOA (firma): {self.eoa_address}")
            logger.warning(f"   Proxy (fondi): {self.wallet_address}")
            logger.warning(f"⚠️  Redeem on-chain richiede transazione Safe multisig")
            logger.warning(f"⚠️  Tentativo con proxy wallet, ma potrebbe fallire")
        
        # Use proxy wallet address for checks (where the funds are)
        self.actual_wallet = self.wallet_address
        logger.info(f"💼 Wallet per redeem: {self.actual_wallet}")
        
        # Setup CTF contract
        self.ctf_contract = self.w3.eth.contract(
            address=Web3.to_checksum_address(CTF_CONTRACT),
            abi=CTF_ABI
        )
        
        logger.info(f"📝 CTF Contract: {CTF_CONTRACT}")
    
    def check_balance(self, token_id: str) -> float:
        """Check balance of a specific conditional token"""
        try:
            # Convert token_id to int if it's a string
            token_id_int = int(token_id) if isinstance(token_id, str) else token_id
            
            balance = self.ctf_contract.functions.balanceOf(
                Web3.to_checksum_address(self.actual_wallet),
                token_id_int
            ).call()
            
            # Convert from wei (18 decimals for CTF tokens)
            balance_float = balance / 1e18
            return balance_float
        except Exception as e:
            logger.error(f"Errore check balance per token {token_id}: {e}")
            return 0.0
    
    def check_payout_status(self, condition_id: str) -> bool:
        """
        Check if payouts have been reported for a condition
        Returns True if the market is resolved (payout > 0)
        """
        try:
            # Convert condition_id to bytes32
            if isinstance(condition_id, str):
                if condition_id.startswith('0x'):
                    condition_id = condition_id[2:]
                condition_id_bytes = bytes.fromhex(condition_id.zfill(64))
            else:
                condition_id_bytes = condition_id
            
            payout = self.ctf_contract.functions.payoutDenominator(
                Web3.to_checksum_address(self.actual_wallet),
                condition_id_bytes
            ).call()
            
            return payout > 0
        except Exception as e:
            logger.error(f"Errore check payout status: {e}")
            return False
    
    def redeem_position(self, condition_id: str, index_sets: List[int] = None) -> bool:
        """
        Redeem positions for a closed market
        
        NOTA: Questo metodo funziona SOLO se:
        1. I token sono nell'account EOA (non proxy wallet)
        2. Oppure il proxy wallet accetta transazioni dirette dall'EOA
        
        Per proxy wallet Safe/Gnosis di Polymarket, potrebbe essere necessario:
        - Usare l'interfaccia web di Polymarket
        - Implementare Safe SDK per transazioni multisig
        
        Args:
            condition_id: The condition ID of the market (market_id in Polymarket)
            index_sets: List of index sets to redeem [1, 2] for binary markets (Yes/No)
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Default to binary market (redeem both outcomes)
            if index_sets is None:
                index_sets = [1, 2]  # 0b01 and 0b10 for binary markets
            
            # Convert condition_id to bytes32
            if isinstance(condition_id, str):
                if condition_id.startswith('0x'):
                    condition_id = condition_id[2:]
                condition_id_bytes = bytes.fromhex(condition_id.zfill(64))
            else:
                condition_id_bytes = condition_id
            
            logger.info(f"🔄 Tentativo di redeem per condition: {condition_id[:20]}...")
            logger.info(f"   Index sets: {index_sets}")
            logger.info(f"   Wallet: {self.actual_wallet[:10]}...")
            
            # Check if market is resolved
            if not self.check_payout_status(condition_id):
                logger.warning("⚠️  Market non ancora risolto o payout non disponibile")
                return False
            
            # Check if we have any balance to redeem
            logger.info("🔍 Controllo balance dei token...")
            # For binary markets, check both outcomes (we need token IDs though)
            # This is a limitation - we'd need the actual token IDs
            
            # PROBLEMA: Con proxy wallet Safe, non possiamo firmare per conto del proxy
            # Questo funzionerà solo se:
            # 1. I token sono nell'EOA
            # 2. Il proxy wallet è configurato per accettare transazioni dirette
            
            if self.eoa_address.lower() != self.actual_wallet.lower():
                logger.warning("⚠️  Tentativo con proxy wallet - probabilmente fallirà")
                logger.warning("⚠️  Safe/Gnosis richiede transazione multisig via execTransaction")
            
            # Build transaction (firmata dall'EOA)
            txn = self.ctf_contract.functions.redeemPositions(
                Web3.to_checksum_address(USDC_CONTRACT),  # collateralToken (USDC)
                b'\x00' * 32,  # parentCollectionId (null for Polymarket)
                condition_id_bytes,  # conditionId
                index_sets  # indexSets
            ).build_transaction({
                'from': self.eoa_address,  # L'EOA firma
                'nonce': self.w3.eth.get_transaction_count(self.eoa_address),
                'gas': 300000,  # Gas limit
                'gasPrice': self.w3.eth.gas_price,
                'chainId': 137  # Polygon mainnet
            })
            
            # Sign transaction con EOA
            signed_txn = self.w3.eth.account.sign_transaction(txn, self.private_key)
            
            # Send transaction
            logger.info("📤 Invio transazione redeem...")
            tx_hash = self.w3.eth.send_raw_transaction(signed_txn.raw_transaction)
            
            logger.info(f"⏳ Transazione inviata: {tx_hash.hex()}")
            logger.info("   Attendo conferma...")
            
            # Wait for receipt (with timeout)
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
            
            if receipt['status'] == 1:
                logger.info(f"✅ Redeem completato! Gas usato: {receipt['gasUsed']}")
                logger.info(f"   TX: https://polygonscan.com/tx/{tx_hash.hex()}")
                return True
            else:
                logger.error(f"❌ Transazione fallita!")
                return False
                
        except Exception as e:
            logger.error(f"❌ Errore durante redeem: {e}")
            logger.error(f"   Questo è normale con proxy wallet Safe/Gnosis")
            logger.error(f"   💡 Usa l'interfaccia web di Polymarket per il claim manuale")
            return False
    
    def redeem_all_positions(self, market_ids: List[str]) -> int:
        """
        Redeem all positions for a list of market IDs
        
        Args:
            market_ids: List of market condition IDs to redeem
            
        Returns:
            Number of successful redeems
        """
        successful = 0
        
        logger.info(f"🔄 Tentativo di redeem per {len(market_ids)} mercati...")
        
        for i, market_id in enumerate(market_ids, 1):
            logger.info(f"\n[{i}/{len(market_ids)}] Market: {market_id[:20]}...")
            
            if self.redeem_position(market_id):
                successful += 1
            else:
                logger.warning(f"⚠️  Redeem fallito per questo mercato")
        
        logger.info(f"\n✅ Completato: {successful}/{len(market_ids)} redeems riusciti")
        return successful


def test_redeem():
    """Test function to check if redeem works"""
    try:
        logger.info("=" * 60)
        logger.info("🧪 TEST REDEEM POSITIONS")
        logger.info("=" * 60)
        
        # Initialize redeemer
        redeemer = PositionRedeemer()
        
        # Get current market from config (for testing)
        market_id = Config.BTC_5MIN_MARKET_ID
        
        logger.info(f"\n📊 Market ID: {market_id}")
        
        # Check payout status
        is_resolved = redeemer.check_payout_status(market_id)
        logger.info(f"   Resolved: {is_resolved}")
        
        if is_resolved:
            logger.info("\n🔄 Market risolto, tento redeem...")
            success = redeemer.redeem_position(market_id)
            
            if success:
                logger.info("✅ Test completato con successo!")
            else:
                logger.warning("⚠️  Redeem non riuscito")
        else:
            logger.info("⏳ Market non ancora risolto, niente da riscattare")
        
    except Exception as e:
        logger.error(f"❌ Errore test: {e}", exc_info=True)


if __name__ == '__main__':
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    test_redeem()
