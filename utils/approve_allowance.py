"""
Script per approvare l'allowance USDC al contratto CLOB di Polymarket
"""
from py_clob_client.client import ClobClient
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.config import Config

def approve_allowance():
    """Approva il contratto CLOB a spendere USDC"""
    
    # Validate config
    Config.validate()
    
    # Initialize client
    private_key = Config.PRIVATE_KEY
    if not private_key.startswith('0x'):
        private_key = '0x' + private_key
    
    client = ClobClient(
        host="https://clob.polymarket.com",
        key=private_key,
        chain_id=Config.CHAIN_ID
    )
    
    # Genera/deriva API credentials
    print("🔐 Generazione credenziali API...")
    client.set_api_creds(client.create_or_derive_api_creds())
    
    print(f"✅ Client autenticato: {Config.WALLET_ADDRESS[:10]}...")
    
    # Approve allowance per USDC (importo alto per non doverlo rifare)
    # 1000 USDC dovrebbe essere sufficiente per molti trade
    amount_to_approve = 1000
    
    print(f"\n💰 Approvazione allowance di ${amount_to_approve} USDC al CLOB...")
    print("⏳ Questo richiederà una transazione on-chain...")
    print("⚠️  Potresti dover confermare la transazione su MetaMask\n")
    
    try:
        # Prima controlla lo stato attuale
        print(f"\n💵 Controllo saldo e allowance attuali...")
        try:
            balances = client.get_balance_allowance()
            print(f"Balance: ${balances.get('balance', 'N/A')}")
            print(f"Allowance: ${balances.get('allowance', 'N/A')}")
        except Exception as e:
            print(f"⚠️  Non riesco a leggere il saldo: {e}")
        
        print(f"\n🔓 Approvazione allowance tramite transazione on-chain...")
        
        # Il metodo corretto è approve_allowance
        result = client.approve_allowance(amount_to_approve)
        
        print(f"✅ Allowance approvata con successo!")
        print(f"📝 Risultato: {result}")
        
        # Check new balances
        print(f"\n💵 Nuovo saldo e allowance:")
        balances = client.get_balance_allowance()
        print(f"Balance: ${balances.get('balance', 'N/A')}")
        print(f"Allowance: ${balances.get('allowance', 'N/A')}")
        
    except Exception as e:
        print(f"❌ Errore durante l'approvazione: {e}")
        print(f"\n📋 Tipo errore: {type(e).__name__}")
        
        print("\n💡 SOLUZIONE ALTERNATIVA:")
        print("1. Vai su https://polymarket.com/")
        print("2. Fai un piccolo trade manuale (anche $1)")
        print("3. Questo approverà automaticamente l'allowance")
        print("4. Poi riavvia il bot")
        
        import traceback
        print(f"\n🔍 Dettagli errore:")
        traceback.print_exc()

if __name__ == '__main__':
    approve_allowance()
