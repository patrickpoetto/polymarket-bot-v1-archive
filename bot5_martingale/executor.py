"""Order execution layer – live and paper trading adapters.

Both adapters implement the same interface so the bot orchestrator is
agnostic to the execution mode.
"""
from __future__ import annotations

import logging
import random
import time
import uuid
from abc import ABC, abstractmethod
from decimal import ROUND_DOWN, Decimal
from typing import Optional

from bot5_martingale.config import Bot5Config
from bot5_martingale.models import MarketInfo, OrderRequest, OrderResult, OrderStatus, Side

logger = logging.getLogger(__name__)


# ======================================================================
# Abstract interface
# ======================================================================

class BaseExecutor(ABC):
    """Contract that both live and paper executors satisfy."""

    @abstractmethod
    def place_order(self, req: OrderRequest) -> OrderResult:
        ...

    @abstractmethod
    def check_order(self, order_id: str) -> Optional[OrderResult]:
        ...

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        ...

    @abstractmethod
    def get_best_prices(self, market: MarketInfo) -> dict[str, dict]:
        """Return {"Up": {"bid": ..., "ask": ...}, "Down": {...}}."""
        ...

    @abstractmethod
    def get_balance_usd(self) -> float:
        ...


# ======================================================================
# Quantization helpers (from bot3 pattern)
# ======================================================================

def quantize_order(side: str, price: float, size: float) -> tuple[float, float]:
    """Quantize price (4 dp) and size (2 dp notional for BUY, 4 dp for SELL)."""
    p = Decimal(str(price)).quantize(Decimal("0.0001"), rounding=ROUND_DOWN)
    s = Decimal(str(size))
    if side.upper() == "BUY":
        notional = (s * p).quantize(Decimal("0.01"), rounding=ROUND_DOWN)
        s = (notional / p).quantize(Decimal("0.0001"), rounding=ROUND_DOWN) if p > 0 else Decimal("0")
    else:
        s = s.quantize(Decimal("0.0001"), rounding=ROUND_DOWN)
    return float(p), float(s)


# ======================================================================
# Live executor
# ======================================================================

class LiveExecutor(BaseExecutor):
    """Submits real orders to Polymarket CLOB."""

    def __init__(self, cfg: Bot5Config):
        self.cfg = cfg
        self._client = self._build_client(cfg)
        self._order_cache: dict[str, dict] = {}

    @staticmethod
    def _build_client(cfg: Bot5Config):
        from py_clob_client.client import ClobClient
        from py_clob_client.clob_types import ApiCreds

        pk = cfg.private_key if cfg.private_key.startswith("0x") else f"0x{cfg.private_key}"
        temp = ClobClient(host=cfg.clob_host, key=pk, chain_id=cfg.chain_id)
        creds = temp.create_or_derive_api_creds()
        return ClobClient(
            host=cfg.clob_host,
            key=pk,
            chain_id=cfg.chain_id,
            creds=creds,
            signature_type=2,
            funder=cfg.wallet_address,
        )

    # ------------------------------------------------------------------

    def place_order(self, req: OrderRequest) -> OrderResult:
        from py_clob_client.clob_types import OrderArgs, OrderType

        ORDER_TYPE_MAP = {"GTC": OrderType.GTC, "FOK": OrderType.FOK, "FAK": OrderType.FAK, "GTD": OrderType.GTD}
        order_type = ORDER_TYPE_MAP.get(self.cfg.order_type_entry, OrderType.GTC)

        q_price, q_size = quantize_order("BUY", req.limit_price, req.size_usd)
        if q_size <= 0:
            return OrderResult(
                request=req, status=OrderStatus.REJECTED,
                error="size zero after quantization",
            )

        for attempt in range(1, self.cfg.order_max_retries + 1):
            try:
                args = OrderArgs(
                    token_id=req.token_id,
                    side="BUY",
                    size=q_size,
                    price=q_price,
                )
                signed = self._client.create_order(args)
                raw = self._client.post_order(signed, order_type)
                break
            except Exception as exc:
                logger.warning("Order attempt %d/%d failed: %s", attempt, self.cfg.order_max_retries, exc)
                if attempt == self.cfg.order_max_retries:
                    return OrderResult(
                        request=req, status=OrderStatus.REJECTED,
                        error=str(exc),
                    )
                time.sleep(min(2 ** attempt, 8))
                raw = {}

        if not isinstance(raw, dict):
            raw = {}

        status_str = str(raw.get("status", "")).lower()
        order_id = raw.get("orderID") or raw.get("order_id")

        if status_str == "matched":
            filled_shares = self._extract_shares(raw, q_size)
            return OrderResult(
                request=req, status=OrderStatus.FILLED,
                order_id=order_id,
                filled_size=q_size * q_price,
                filled_price=q_price,
                filled_shares=filled_shares,
                raw=raw,
            )

        if status_str == "live" and order_id:
            return self._wait_and_verify(req, order_id, q_price, q_size, raw)

        return OrderResult(
            request=req, status=OrderStatus.PLACED,
            order_id=order_id, raw=raw,
        )

    def _wait_and_verify(
        self, req: OrderRequest, order_id: str,
        q_price: float, q_size: float, initial_raw: dict,
    ) -> OrderResult:
        deadline = time.time() + self.cfg.order_timeout_seconds
        while time.time() < deadline:
            time.sleep(self.cfg.order_poll_interval)
            try:
                status_resp = self._client.get_order(order_id)
            except Exception:
                continue
            if not isinstance(status_resp, dict):
                continue
            s = str(status_resp.get("status", "")).lower()
            if s == "matched":
                filled = self._extract_shares(status_resp, q_size)
                return OrderResult(
                    request=req, status=OrderStatus.FILLED,
                    order_id=order_id,
                    filled_size=q_size * q_price,
                    filled_price=q_price,
                    filled_shares=filled,
                    raw=status_resp,
                )

        # Timeout: cancel and re-check
        try:
            self._client.cancel(order_id)
        except Exception:
            pass
        time.sleep(0.5)
        try:
            post = self._client.get_order(order_id)
            if isinstance(post, dict) and str(post.get("status", "")).lower() == "matched":
                filled = self._extract_shares(post, q_size)
                return OrderResult(
                    request=req, status=OrderStatus.FILLED,
                    order_id=order_id,
                    filled_size=q_size * q_price,
                    filled_price=q_price,
                    filled_shares=filled,
                    raw=post,
                )
        except Exception:
            pass

        return OrderResult(
            request=req, status=OrderStatus.EXPIRED,
            order_id=order_id, raw=initial_raw,
            error="order timed out and was cancelled",
        )

    def _extract_shares(self, payload: dict, fallback: float) -> float:
        for key in ("takingAmount", "taking_amount", "size", "filled_size"):
            val = payload.get(key)
            if val is not None:
                try:
                    v = float(val)
                    if v > 0:
                        return v
                except (TypeError, ValueError):
                    continue
        return fallback

    def check_order(self, order_id: str) -> Optional[OrderResult]:
        try:
            data = self._client.get_order(order_id)
            if not isinstance(data, dict):
                return None
            s = str(data.get("status", "")).lower()
            status_map = {
                "matched": OrderStatus.FILLED,
                "live": OrderStatus.PLACED,
                "cancelled": OrderStatus.CANCELLED,
            }
            return OrderResult(
                request=OrderRequest(
                    market=MarketInfo("", "", [], [], 0),
                    side=Side.UP, size_usd=0, limit_price=0,
                ),
                status=status_map.get(s, OrderStatus.PENDING),
                order_id=order_id,
                raw=data,
            )
        except Exception:
            return None

    def cancel_order(self, order_id: str) -> bool:
        try:
            self._client.cancel(order_id)
            return True
        except Exception:
            return False

    def get_best_prices(self, market: MarketInfo) -> dict[str, dict]:
        from py_clob_client.clob_types import BookParams

        token_map = market.token_id_by_side
        params = [BookParams(token_id=tid) for tid in token_map.values()]
        prices: dict[str, dict] = {}
        try:
            books = self._client.get_order_books(params)
        except Exception as exc:
            logger.warning("get_order_books failed: %s", exc)
            return {s.value: {"bid": None, "ask": None} for s in Side}

        side_by_token = {v: k for k, v in token_map.items()}
        for book in books:
            aid = str(getattr(book, "asset_id", "") or "")
            side = side_by_token.get(aid)
            if side is None:
                continue
            best_ask = None
            best_bid = None
            if hasattr(book, "asks") and book.asks:
                try:
                    best_ask = float(book.asks[-1].price)
                except (TypeError, ValueError, IndexError):
                    pass
            if hasattr(book, "bids") and book.bids:
                try:
                    best_bid = float(book.bids[0].price)
                except (TypeError, ValueError, IndexError):
                    pass
            prices[side.value] = {"bid": best_bid, "ask": best_ask}

        for s in Side:
            prices.setdefault(s.value, {"bid": None, "ask": None})
        return prices

    def get_balance_usd(self) -> float:
        try:
            from py_clob_client.clob_types import BalanceAllowanceParams
            data = self._client.get_balance_allowance(
                BalanceAllowanceParams(asset_type="USDC", signature_type=2)
            )
            raw = data.get("balance") if isinstance(data, dict) else None
            if raw is None:
                return 0.0
            return int(str(raw)) / 1_000_000.0
        except Exception as exc:
            logger.warning("Balance check failed: %s", exc)
            return 0.0


# ======================================================================
# Paper executor – realistic fill simulation
# ======================================================================

def _fetch_clob_book(token_id: str, clob_host: str = "https://clob.polymarket.com") -> dict:
    """Fetch raw orderbook from CLOB REST API (public, no auth).

    Returns {"asks": [{"price": str, "size": str}, ...], "bids": [...]}
    """
    import requests as _req
    resp = _req.get(
        f"{clob_host}/book",
        params={"token_id": token_id},
        timeout=5,
    )
    resp.raise_for_status()
    return resp.json()


def _walk_asks(asks: list[dict], limit_price: float, shares_needed: float) -> tuple[float, float]:
    """Walk ask levels up to limit_price, return (total_shares_available, avg_fill_price).

    Computes VWAP of the consumed levels.
    """
    total_shares = 0.0
    total_cost = 0.0
    remaining = shares_needed
    for level in sorted(asks, key=lambda x: float(x["price"])):
        p = float(level["price"])
        s = float(level["size"])
        if p > limit_price:
            break
        take = min(s, remaining)
        total_shares += take
        total_cost += take * p
        remaining -= take
        if remaining <= 0:
            break
    if total_shares <= 0:
        return 0.0, limit_price
    return total_shares, total_cost / total_shares


class PaperExecutor(BaseExecutor):
    """Simulates order fills using the live CLOB public REST orderbook.

    Fill model:
        1. Fetch real asks for the target token via CLOB REST (no auth needed).
        2. Walk ask levels up to limit_price to get available shares & VWAP.
        3. Apply fill_pct (random between min_pct and 1.0) bounded by available.
        4. Track virtual balance and positions.

    This gives fills as faithful to reality as possible without placing real orders.
    """

    def __init__(self, cfg: Bot5Config, initial_balance: float = 10_000.0):
        self.cfg = cfg
        self._balance = initial_balance
        self._orders: dict[str, OrderResult] = {}
        self._positions: dict[str, float] = {}  # condition_id -> shares
        # Still try live client for get_best_prices / inference (optional)
        self._live_client: Optional[LiveExecutor] = None
        if cfg.private_key and cfg.wallet_address:
            try:
                self._live_client = LiveExecutor(cfg)
                logger.info("Paper executor attached live client for 99c inference")
            except Exception as exc:
                logger.warning("Paper executor could not init live client: %s", exc)

    def place_order(self, req: OrderRequest) -> OrderResult:
        order_id = f"paper-{uuid.uuid4().hex[:12]}"

        available_vol, avg_price = self._simulate_fill(req)

        latency = self.cfg.paper_fill_latency_ms / 1000.0
        time.sleep(latency)

        if available_vol <= 0 or avg_price > req.limit_price:
            return OrderResult(
                request=req, status=OrderStatus.REJECTED,
                order_id=order_id,
                error="simulated: no volume at limit price",
            )

        # Partial fill simulation
        fill_pct = max(
            self.cfg.paper_partial_fill_min_pct,
            random.uniform(self.cfg.paper_partial_fill_min_pct, 1.0),
        )
        if random.random() > self.cfg.paper_fill_probability_base:
            fill_pct *= 0.5  # rare adverse scenario

        shares_requested = req.size_usd / avg_price
        shares_filled = min(shares_requested * fill_pct, available_vol)
        cost = shares_filled * avg_price

        if cost > self._balance:
            shares_filled = self._balance / avg_price
            cost = shares_filled * avg_price

        self._balance -= cost
        cid = req.market.condition_id
        self._positions[cid] = self._positions.get(cid, 0.0) + shares_filled

        status = (
            OrderStatus.FILLED
            if abs(shares_filled - shares_requested) / max(shares_requested, 1e-9) < 0.02
            else OrderStatus.PARTIALLY_FILLED
        )

        result = OrderResult(
            request=req, status=status,
            order_id=order_id,
            filled_size=cost,
            filled_price=avg_price,
            filled_shares=shares_filled,
            fee=0.0,
        )
        self._orders[order_id] = result
        return result

    def _simulate_fill(self, req: OrderRequest) -> tuple[float, float]:
        """Return (available_shares_at_limit, vwap_fill_price).

        Primary: walk real CLOB asks via py_clob_client (if available).
        Secondary: public REST endpoint.
        Fallback: conservative deterministic estimate.
        """
        token_id = req.market.token_id_by_side.get(req.side)
        shares_needed = req.size_usd / req.limit_price

        # 1) Preferred path: py_clob_client (same stack as live)
        if token_id and self._live_client is not None:
            try:
                from py_clob_client.clob_types import BookParams

                books = self._live_client._client.get_order_books([BookParams(token_id=token_id)])
                if books:
                    book = books[0]
                    asks = []
                    for lvl in getattr(book, "asks", []) or []:
                        try:
                            asks.append({"price": float(lvl.price), "size": float(lvl.size)})
                        except (TypeError, ValueError):
                            continue
                    if asks:
                        available, vwap = _walk_asks(asks, req.limit_price, shares_needed)
                        if available > 0:
                            logger.debug(
                                "Paper fill via py_clob_client: %.2f shares @ %.4f vwap (need %.2f)",
                                available, vwap, shares_needed,
                            )
                            return available, vwap
            except Exception as exc:
                logger.debug("py_clob_client book fetch failed (%s), trying REST fallback", exc)

        # 2) Secondary path: public REST orderbook (no auth)
        if token_id:
            try:
                book = _fetch_clob_book(token_id, self.cfg.clob_host)
                asks = book.get("asks", [])
                if asks:
                    available, vwap = _walk_asks(asks, req.limit_price, shares_needed)
                    if available > 0:
                        logger.debug(
                            "Paper fill from real book: %.2f shares @ %.4f vwap (need %.2f)",
                            available, vwap, shares_needed,
                        )
                        return available, vwap
                # book exists but nothing at limit price
                logger.debug("No asks at or below %.4f for token %s", req.limit_price, token_id[:12])
                return 0.0, req.limit_price
            except Exception as exc:
                logger.debug("CLOB book fetch failed (%s), using fallback", exc)

        # Fallback: assume 0.51 ask with moderate depth (2x order size)
        fallback_price = min(req.limit_price, self.cfg.max_entry_price)
        fallback_vol = shares_needed * 2.0
        return fallback_vol, fallback_price

    def check_order(self, order_id: str) -> Optional[OrderResult]:
        return self._orders.get(order_id)

    def cancel_order(self, order_id: str) -> bool:
        if order_id in self._orders:
            self._orders[order_id] = OrderResult(
                request=self._orders[order_id].request,
                status=OrderStatus.CANCELLED,
                order_id=order_id,
            )
            return True
        return False

    def get_best_prices(self, market: MarketInfo) -> dict[str, dict]:
        # Try live client first (uses ws cache if available)
        if self._live_client:
            try:
                return self._live_client.get_best_prices(market)
            except Exception:
                pass
        # Fallback: hit CLOB REST directly
        result: dict[str, dict] = {}
        for side, token_id in market.token_id_by_side.items():
            try:
                book = _fetch_clob_book(token_id, self.cfg.clob_host)
                asks = book.get("asks", [])
                bids = book.get("bids", [])
                best_ask = min((float(a["price"]) for a in asks), default=None)
                best_bid = max((float(b["price"]) for b in bids), default=None)
                result[side.value] = {"ask": best_ask, "bid": best_bid}
            except Exception:
                result[side.value] = {"ask": 0.51, "bid": 0.50}
        return result if result else {s.value: {"bid": 0.50, "ask": 0.51} for s in Side}

    def get_balance_usd(self) -> float:
        return self._balance

    def credit_win(self, condition_id: str) -> float:
        """Settle a winning position: shares pay $1 each."""
        shares = self._positions.pop(condition_id, 0.0)
        self._balance += shares
        return shares

    def debit_loss(self, condition_id: str) -> float:
        """Settle a losing position: shares are worthless."""
        shares = self._positions.pop(condition_id, 0.0)
        return shares


# ======================================================================
# Factory
# ======================================================================

def create_executor(cfg: Bot5Config) -> BaseExecutor:
    from bot5_martingale.models import BotMode
    if cfg.mode is BotMode.LIVE:
        logger.info("Executor: LIVE mode")
        return LiveExecutor(cfg)
    logger.info("Executor: PAPER mode")
    return PaperExecutor(cfg)
