"""One-shot: approve pUSD and CTF to spend by V2 Exchange contracts via Polymarket Relayer."""
import logging
import sys
from web3 import Web3

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

from shared.config import Config
from py_builder_relayer_client.client import RelayClient
from py_builder_relayer_client.models import SafeTransaction, OperationType
from py_builder_signing_sdk.config import BuilderConfig
from py_builder_signing_sdk.sdk_types import BuilderApiKeyCreds

# Polymarket V2 contracts (Polygon)
PUSD = Web3.to_checksum_address("0xC011a7E12a19f7B1f670d46F03B03f3342E82DFB")
CTF = Web3.to_checksum_address("0x4D97DCd97eC945f40cF65F87097ACe5EA0476045")
EXCHANGE_V2 = Web3.to_checksum_address("0xE111180000d2663C0091e4f400237545B87B996B")
NEG_RISK_V2 = Web3.to_checksum_address("0xe2222d279d744050d28e00520010520000310F59")
NEG_RISK_ADAPTER = Web3.to_checksum_address("0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296")

MAX_UINT256 = 2**256 - 1

ERC20_APPROVE_ABI = [{
    "name": "approve", "type": "function",
    "inputs": [{"name": "spender", "type": "address"}, {"name": "amount", "type": "uint256"}],
    "outputs": [{"type": "bool"}],
}]

ERC1155_SETAPPROVALFORALL_ABI = [{
    "name": "setApprovalForAll", "type": "function",
    "inputs": [{"name": "operator", "type": "address"}, {"name": "approved", "type": "bool"}],
    "outputs": [],
}]


def encode_erc20_approve(spender: str) -> str:
    w3 = Web3()
    contract = w3.eth.contract(address=PUSD, abi=ERC20_APPROVE_ABI)
    return contract.functions.approve(spender, MAX_UINT256)._encode_transaction_data()


def encode_erc1155_setapproval(operator: str) -> str:
    w3 = Web3()
    contract = w3.eth.contract(address=CTF, abi=ERC1155_SETAPPROVALFORALL_ABI)
    return contract.functions.setApprovalForAll(operator, True)._encode_transaction_data()


def main():
    creds = BuilderApiKeyCreds(
        key=Config.BUILDER_API_KEY,
        secret=Config.BUILDER_SECRET,
        passphrase=Config.BUILDER_PASSPHRASE,
    )
    builder_cfg = BuilderConfig(local_builder_creds=creds)
    client = RelayClient(
        relayer_url=Config.RELAYER_URL,
        chain_id=Config.CHAIN_ID,
        private_key=Config.PRIVATE_KEY,
        builder_config=builder_cfg,
    )
    logger.info("RelayClient ready (proxy=%s)", Config.WALLET_ADDRESS)

    # Build batch of approval transactions
    txs = [
        # pUSD → CTF Exchange V2
        SafeTransaction(to=PUSD, operation=OperationType.Call,
                        data=encode_erc20_approve(EXCHANGE_V2), value="0"),
        # pUSD → Neg Risk Exchange V2
        SafeTransaction(to=PUSD, operation=OperationType.Call,
                        data=encode_erc20_approve(NEG_RISK_V2), value="0"),
        # pUSD → Neg Risk Adapter
        SafeTransaction(to=PUSD, operation=OperationType.Call,
                        data=encode_erc20_approve(NEG_RISK_ADAPTER), value="0"),
        # CTF → CTF Exchange V2 (so SELLs of conditional shares work)
        SafeTransaction(to=CTF, operation=OperationType.Call,
                        data=encode_erc1155_setapproval(EXCHANGE_V2), value="0"),
        # CTF → Neg Risk Adapter
        SafeTransaction(to=CTF, operation=OperationType.Call,
                        data=encode_erc1155_setapproval(NEG_RISK_ADAPTER), value="0"),
    ]
    logger.info("Submitting batch of %d approval tx via Relayer (gasless)...", len(txs))

    try:
        result = client.execute(transactions=txs, metadata="V2 approvals: pUSD + CTF")
        logger.info("Submitted. Waiting for on-chain confirmation...")
        final = result.wait()
        logger.info("DONE: %s", final)
    except Exception as e:
        logger.error("Relayer execute failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
