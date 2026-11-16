"""On-chain allowance check for Polymarket V2 (pUSD-based)."""
import requests

PROXY = "0x149dEe5D2B1386b7af2BB72C9b380547B16c8d63"

PUSD = "0xC011a7E12a19f7B1f670d46F03B03f3342E82DFB"  # Polymarket USD (V2 collateral)
CTF = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"

SPENDERS = {
    "CTF Exchange V2":      "0xE111180000d2663C0091e4f400237545B87B996B",
    "Neg Risk Exchange V2": "0xe2222d279d744050d28e00520010520000310F59",
    "Neg Risk Adapter":     "0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296",
    "CollateralOnramp":     "0x93070a847efEf7F70739046A929D47a521F5B8ee",
}

RPC = "https://polygon-rpc.com"

def pad(addr: str) -> str:
    return addr.lower().replace("0x", "").rjust(64, "0")

def call(to: str, data: str) -> int:
    r = requests.post(RPC, json={
        "jsonrpc": "2.0", "id": 1, "method": "eth_call",
        "params": [{"to": to, "data": data}, "latest"],
    }, timeout=10).json()
    return int(r.get("result", "0x0"), 16)

def erc20_allowance(token: str, spender: str) -> int:
    return call(token, "0xdd62ed3e" + pad(PROXY) + pad(spender))

def erc20_balance(token: str) -> int:
    return call(token, "0x70a08231" + pad(PROXY))

def ctf_approved(spender: str) -> bool:
    return call(CTF, "0xe985e9c5" + pad(PROXY) + pad(spender)) == 1

print(f"Proxy: {PROXY}\n")
print(f"pUSD balance:  ${erc20_balance(PUSD)/1e6:,.2f}\n")
print(f"{'Spender':<22}{'pUSD allow':>15}   CTF approved")
print("-" * 56)
for name, sp in SPENDERS.items():
    a = erc20_allowance(PUSD, sp)
    c = ctf_approved(sp)
    a_str = "UNLIMITED" if a > 10**30 else f"${a/1e6:,.2f}"
    print(f"{name:<22}{a_str:>15}   {'YES' if c else 'NO'}")
