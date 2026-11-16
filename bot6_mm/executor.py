"""Order execution for Bot6: GTC limit orders + status polling.

Live mode uses py_clob_client.  Paper mode simulates fills from current mid.
"""
from __future__ import annotations

import logging
import time
import uuid
from abc import ABC, abstractmethod
from decimal import ROUND_DOWN, Decimal
from typing import Optional

import requests

from bot6_mm.config import Bot6Config
from bot2_btc.models import Prices, Side
from bot6_mm.models import ActiveOrder, OrderSide

logger = logging.getLogger(__name__)

CLOB_API = "https://clob.polymarket.com"


# ── Quantization ──────────────────────────────────────────────────────────────

def quantize(order_side: str, price: float, shares: float) -> tuple[float, float]:
    p = Decimal(str(price)).quantize(Decimal("0.0001"), rounding=ROUND_DOWN)
    s = Decimal(str(shares))
    if order_side.upper() == "BUY":
        notional = (s * p).quantize(Decimal("0.01"), rounding=ROUND_DOWN)
        s = (notional / p).quantize(Decimal("0.0001"), rounding=ROUND_DOWN) if p > 0 else Decimal("0")
    else:
        s = s.quantize(Decimal("0.0001"), rounding=ROUND_DOWN)
    return float(p), float(s)


# ── Abstract interface ────────────────────────────────────────────────────────

class BaseExecutor(ABC):
    @abstractmethod
    def place_gtc_buy(self, token_id: str, price: float, shares: float,
                      side: Side) -> Optional[ActiveOrder]:
        """Place a GTC BUY limit order. Returns ActiveOrder or None on failure."""
        ...

    @abstractmethod
    def place_gtc_sell(self, token_id: str, price: float, shares: float,
                       side: Side) -> Optional[ActiveOrder]:
        """Place a GTC SELL limit order. Returns ActiveOrder or None on failure."""
        ...

    @abstractmethod
    def place_fak_sell(self, token_id: str, price_floor: float, shares: float,
                       side: Side) -> Optional[float]:
        """Place a FAK SELL at floor price. Returns filled shares or None."""
        ...

    @abstractmethod
    def check_order(self, order: ActiveOrder) -> tuple[float, bool]:
        """Return (size_filled, is_done). is_done=True if MATCHED/FILLED/CANCELLED."""
        ...

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        ...

    @abstractmethod
    def get_balance_usd(self) -> float:
        ...

    def get_spendable_shares(self, token_id: str) -> Optional[float]:
        return None


# ── Live executor ─────────────────────────────────────────────────────────────

class LiveExecutor(BaseExecutor):
    def __init__(self, cfg: Bot6Config):
        from py_clob_client.client import ClobClient
        from py_clob_client.clob_types import ApiCreds
        self._cfg = cfg
        self._client = ClobClient(
            host=cfg.clob_host,
            chain_id=cfg.chain_id,
            key=cfg.private_key,
            signature_type=2,
            funder=cfg.wallet_address,
        )
        self._client.set_api_creds(self._client.create_or_derive_api_creds())

    def place_gtc_buy(self, token_id: str, price: float, shares: float,
                      side: Side) -> Optional[ActiveOrder]:
        return self._place(token_id, price, shares, side, OrderSide.BUY, "GTC")

    def place_gtc_sell(self, token_id: str, price: float, shares: float,
                       side: Side) -> Optional[ActiveOrder]:
        return self._place(token_id, price, shares, side, OrderSide.SELL, "GTC")

    def place_fak_sell(self, token_id: str, price_floor: float, shares: float,
                       side: Side) -> Optional[float]:
        order = self._place(token_id, price_floor, shares, side, OrderSide.SELL, "FAK")
        if order is None:
            return None
        # FAK: check fill immediately
        filled, _ = self.check_order(order)
        return filled

    def _place(self, token_id: str, price: float, shares: float,
               side: Side, order_side: OrderSide, order_type: str) -> Optional[ActiveOrder]:
        from py_clob_client.clob_types import OrderArgs, OrderType
        from py_clob_client.constants import BUY, SELL
        p, s = quantize(order_side.value, price, shares)
        if s <= 0:
            logger.warning("bot6 %s %s: quantized size=0 (price=%.4f shares=%.4f)", order_side.value, side.value, price, shares)
            return None
        try:
            ot = OrderType.GTC if order_type == "GTC" else OrderType.FAK
            direction = BUY if order_side == OrderSide.BUY else SELL
            args = OrderArgs(token_id=token_id, price=p, size=s, side=direction)
            signed = self._client.create_order(args)
            resp = self._client.post_order(signed, ot)
            order_id = (resp or {}).get("orderID") or (resp or {}).get("order_id")
            if not order_id:
                logger.warning("bot6 place %s %s: no orderID in response %s", order_side.value, side.value, resp)
                return None
            logger.info("bot6 PLACE %s %s %.4f×%.2f → id=%s", order_side.value, side.value, p, s, order_id[:12])
            return ActiveOrder(order_id=order_id, side=side, order_side=order_side, price=p, size=s)
        except Exception as exc:
            logger.error("bot6 place %s %s error: %s", order_side.value, side.value, exc)
            return None

    def check_order(self, order: ActiveOrder) -> tuple[float, bool]:
        try:
            data = self._client.get_order(order.order_id)
            if not data:
                return order.size_filled, False
            status = (data.get("status") or "").upper()
            filled = float(data.get("sizeMatched") or data.get("sizeFilled") or 0)
            done = status in ("MATCHED", "FILLED", "CANCELLED", "CANCELLED_WITH_PARTIAL_FILL")
            return filled, done
        except Exception as exc:
            logger.debug("bot6 check_order %s: %s", order.order_id[:12], exc)
            return order.size_filled, False

    def cancel_order(self, order_id: str) -> bool:
        try:
            self._client.cancel(order_id)
            logger.info("bot6 CANCEL %s", order_id[:12])
            return True
        except Exception as exc:
            logger.warning("bot6 cancel %s: %s", order_id[:12], exc)
            return False

    def get_balance_usd(self) -> float:
        try:
            from py_clob_client.clob_types import BalanceAllowanceParams
            resp = self._client.get_balance_allowance(
                BalanceAllowanceParams(asset_type="COLLATERAL", signature_type=2)
            )
            return float(resp.get("balance", 0)) / 1_000_000.0
        except Exception as exc:
            logger.warning("bot6 get_balance_usd: %s", exc)
            return 0.0

    def get_spendable_shares(self, token_id: str) -> Optional[float]:
        try:
            from py_clob_client.clob_types import BalanceAllowanceParams
            resp = self._client.get_balance_allowance(
                BalanceAllowanceParams(asset_type="CONDITIONAL", token_id=token_id, signature_type=2)
            )
            raw = float(resp.get("balance", 0))
            return raw / 1_000_000.0 if raw > 1_000 else raw
        except Exception:
            return None


# ── Paper executor ────────────────────────────────────────────────────────────

class PaperExecutor(BaseExecutor):
    """Simulates GTC fills: fill BUY when ask ≤ order price, SELL when bid ≥ price."""

    def __init__(self, cfg: Bot6Config):
        self._cfg = cfg
        self._balance = cfg.paper_initial_balance
        self._orders: dict[str, dict] = {}

    def place_gtc_buy(self, token_id: str, price: float, shares: float,
                      side: Side) -> Optional[ActiveOrder]:
        p, s = quantize("BUY", price, shares)
        oid = str(uuid.uuid4())
        self._orders[oid] = {
            "token_id": token_id, "price": p, "size": s,
            "order_side": OrderSide.BUY, "side": side,
            "filled": 0.0, "cancelled": False,
        }
        return ActiveOrder(order_id=oid, side=side, order_side=OrderSide.BUY, price=p, size=s)

    def place_gtc_sell(self, token_id: str, price: float, shares: float,
                       side: Side) -> Optional[ActiveOrder]:
        p, s = quantize("SELL", price, shares)
        oid = str(uuid.uuid4())
        self._orders[oid] = {
            "token_id": token_id, "price": p, "size": s,
            "order_side": OrderSide.SELL, "side": side,
            "filled": 0.0, "cancelled": False,
        }
        return ActiveOrder(order_id=oid, side=side, order_side=OrderSide.SELL, price=p, size=s)

    def place_fak_sell(self, token_id: str, price_floor: float, shares: float,
                       side: Side) -> Optional[float]:
        # Simulate: always fill at price_floor
        logger.info("bot6 PAPER FAK SELL %s %.4f×%.2f", side.value, price_floor, shares)
        return shares

    def simulate_fills(self, token_id: str, current_bid: float, current_ask: float) -> None:
        """Called each tick to check if any paper orders would fill."""
        for oid, o in list(self._orders.items()):
            if o.get("cancelled") or o["filled"] >= o["size"]:
                continue
            if o["token_id"] != token_id:
                continue
            if o["order_side"] == OrderSide.BUY and current_ask <= o["price"]:
                o["filled"] = o["size"]
            elif o["order_side"] == OrderSide.SELL and current_bid >= o["price"]:
                o["filled"] = o["size"]

    def check_order(self, order: ActiveOrder) -> tuple[float, bool]:
        o = self._orders.get(order.order_id)
        if not o:
            return order.size_filled, True
        done = o.get("cancelled", False) or o["filled"] >= o["size"]
        return o["filled"], done

    def cancel_order(self, order_id: str) -> bool:
        if order_id in self._orders:
            self._orders[order_id]["cancelled"] = True
        return True

    def get_balance_usd(self) -> float:
        return self._balance

    def get_spendable_shares(self, token_id: str) -> Optional[float]:
        return None
