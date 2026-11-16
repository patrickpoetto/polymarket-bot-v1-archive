"""Order execution layer – live (py_clob_client) and paper adapters.

Both implement ``BaseExecutor`` so ``Bot`` is mode-agnostic.

Key design:
- ``buy(token_id, shares, limit_price)`` / ``sell(...)`` issue orders and
  return immediately with an ``OrderResult``.
- GTC orders are polled until filled or timed out.
- Paper mode uses the current order book (REST) to determine fill price
  and simulates realistic partial fills.
"""
from __future__ import annotations

import logging
import time
import uuid
from abc import ABC, abstractmethod
from decimal import ROUND_DOWN, Decimal
from typing import Optional

import requests

from bot2_btc.config import Bot2Config
from bot2_btc.models import MarketInfo, OrderRequest, OrderResult, OrderStatus, Side

logger = logging.getLogger(__name__)

CLOB_API = "https://clob.polymarket.com"


# ── Quantization ─────────────────────────────────────────────────────────────

def quantize(side: str, price: float, shares: float) -> tuple[float, float]:
    """Return (price, shares) quantized per Polymarket CLOB rules."""
    p = Decimal(str(price)).quantize(Decimal("0.0001"), rounding=ROUND_DOWN)
    s = Decimal(str(shares))
    if side.upper() == "BUY":
        notional = (s * p).quantize(Decimal("0.01"), rounding=ROUND_DOWN)
        s = (notional / p).quantize(Decimal("0.0001"), rounding=ROUND_DOWN) if p > 0 else Decimal("0")
    else:
        s = s.quantize(Decimal("0.0001"), rounding=ROUND_DOWN)
    return float(p), float(s)


# ── Abstract interface ────────────────────────────────────────────────────────

class BaseExecutor(ABC):
    @abstractmethod
    def buy(self, req: OrderRequest) -> OrderResult:
        """Place a BUY limit order."""
        ...

    @abstractmethod
    def sell(self, req: OrderRequest) -> OrderResult:
        """Place a SELL limit order."""
        ...

    @abstractmethod
    def check_order(self, order_id: str) -> Optional[OrderResult]:
        """Check current status of an order."""
        ...

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        ...

    @abstractmethod
    def get_balance_usd(self) -> float:
        ...

    def get_spendable_shares(self, token_id: str) -> Optional[float]:
        """Return actual on-chain spendable balance for *token_id*, or ``None``.

        Default implementation returns ``None`` (paper mode).  ``LiveExecutor``
        overrides this with a real CLOB balance-allowance query so the bot can
        reconcile its internal tracking against the exchange's ground truth.
        """
        return None

    def get_open_positions_value(self) -> tuple[float, int]:
        """(mark-to-market USD, count of positions). Paper: no API positions."""
        return 0.0, 0


# ── Live executor ─────────────────────────────────────────────────────────────

class LiveExecutor(BaseExecutor):
    """Real order submission via py_clob_client."""

    def __init__(self, cfg: Bot2Config):
        self.cfg = cfg
        self._client = self._build_client(cfg)

    @staticmethod
    def _build_client(cfg: Bot2Config):
        from py_clob_client.client import ClobClient
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

    def buy(self, req: OrderRequest) -> OrderResult:
        return self._submit("BUY", req)

    def sell(self, req: OrderRequest) -> OrderResult:
        return self._submit("SELL", req)

    def warm_up_allowance(self, token_id: str) -> None:
        """Call after a buy fill to set CTF allowance early (non-blocking).

        update_balance_allowance triggers an on-chain tx via Polymarket's relayer.
        It needs ~10-30s to confirm on-chain. Calling right after the pre-pos fill
        ensures it's confirmed well before EXIT LOW fires.

        For ERC-1155 CONDITIONAL tokens the approval is a boolean (setApprovalForAll),
        not an amount — get_balance_allowance does NOT return an 'allowance' field.
        Success = no exception from update_balance_allowance.
        """
        import threading

        def _do():
            from py_clob_client.clob_types import BalanceAllowanceParams
            params = BalanceAllowanceParams(
                asset_type="CONDITIONAL",
                token_id=token_id,
                signature_type=2,
            )
            for attempt in range(1, 4):
                try:
                    self._client.update_balance_allowance(params)
                    logger.info("warm_up_allowance OK token=%s… (attempt %d)", token_id[:12], attempt)
                    return
                except Exception as exc:
                    logger.warning("warm_up_allowance attempt %d/3 token=%s: %s", attempt, token_id[:12], exc)
                    if attempt < 3:
                        time.sleep(2)
            logger.error("warm_up_allowance failed after 3 attempts for token=%s", token_id[:12])

        threading.Thread(target=_do, daemon=True, name="warm-allowance").start()

    def _submit(self, direction: str, req: OrderRequest) -> OrderResult:
        from py_clob_client.clob_types import OrderArgs, OrderType

        ORDER_TYPE_MAP = {
            "GTC": OrderType.GTC, "FOK": OrderType.FOK,
            "FAK": OrderType.FAK, "GTD": OrderType.GTD,
        }
        cfg_ot = self.cfg.order_type_sell if direction == "SELL" else self.cfg.order_type
        ot = ORDER_TYPE_MAP.get(cfg_ot, OrderType.GTC)
        q_price, q_shares = quantize(direction, req.limit_price, req.shares)

        if q_shares <= 0:
            return OrderResult(request=req, status=OrderStatus.REJECTED, error="zero shares after quantization")

        token_id = req.market.token_id_for(req.side)
        if not token_id:
            return OrderResult(request=req, status=OrderStatus.REJECTED, error=f"no token_id for {req.side}")

        raw: dict = {}
        sell_started_at = time.time() if direction == "SELL" else 0.0
        _clob_balance: float = 0.0
        attempt = 0
        while True:
            attempt += 1
            try:
                # For SELL: refresh Polymarket's internal balance/allowance cache before
                # every attempt. Pre-positioned tokens bought before market open can take
                # up to ~10-15 minutes for the CLOB to fully index them as sellable.
                # Calling update+get each attempt keeps the cache warm and gives us
                # diagnostic visibility into what the CLOB currently sees.
                if direction == "SELL":
                    from py_clob_client.clob_types import BalanceAllowanceParams
                    _bap = BalanceAllowanceParams(
                        asset_type="CONDITIONAL",
                        token_id=token_id,
                        signature_type=2,
                    )
                    try:
                        self._client.update_balance_allowance(_bap)
                    except Exception as _exc:
                        logger.warning("[%s] update_balance_allowance: %s", req.order_label, _exc)
                    _clob_balance: float = 0.0
                    try:
                        _bal_data = self._client.get_balance_allowance(_bap)
                        _raw_bal = _bal_data.get("balance", "0") if isinstance(_bal_data, dict) else "0"
                        _clob_balance = int(str(_raw_bal)) / 1_000_000.0
                        logger.info(
                            "[%s] CLOB balance=%.4f (need %.4f)",
                            req.order_label, _clob_balance, q_shares,
                        )
                    except Exception as _exc:
                        logger.warning("[%s] get_balance_allowance: %s", req.order_label, _exc)
                args = OrderArgs(token_id=token_id, side=direction, size=q_shares, price=q_price)
                signed = self._client.create_order(args)
                raw = self._client.post_order(signed, ot)
                break
            except Exception as exc:
                err = str(exc).lower()
                is_balance_err = "not enough balance" in err or "allowance" in err
                logger.warning("[%s] attempt %d: %s", req.order_label, attempt, exc)

                if direction == "SELL" and is_balance_err:
                    elapsed = time.time() - sell_started_at
                    if elapsed < self.cfg.sell_balance_max_wait_seconds:
                        logger.warning(
                            "[%s] balance/allowance not ready (%.1fs elapsed, CLOB balance=%.4f) – retrying in %.1fs",
                            req.order_label, elapsed, _clob_balance, self.cfg.sell_balance_retry_seconds,
                        )
                        time.sleep(self.cfg.sell_balance_retry_seconds)
                        continue

                if attempt >= self.cfg.order_max_retries:
                    return OrderResult(request=req, status=OrderStatus.REJECTED, error=str(exc))
                time.sleep(min(2 ** attempt, 8))

        if not isinstance(raw, dict):
            raw = {}

        status_str = str(raw.get("status", "")).lower()
        order_id = raw.get("orderID") or raw.get("order_id")

        if status_str == "matched":
            shares_filled = self._extract_shares(raw, q_shares, direction)
            fill_price = self._extract_fill_price(raw, q_price, shares_filled, direction)
            return OrderResult(
                request=req, status=OrderStatus.FILLED,
                order_id=order_id,
                filled_shares=shares_filled,
                filled_price=fill_price,
                filled_notional=shares_filled * fill_price,
                raw=raw,
            )

        if status_str == "live" and order_id:
            # FAK/FOK orders should never be "live"; only GTC can rest on book.
            if cfg_ot in ("FAK", "FOK"):
                logger.warning("[%s] FAK/FOK order unexpectedly live – treating as no-fill", req.order_label)
                return OrderResult(request=req, status=OrderStatus.REJECTED, error="FAK/FOK order live (no fill)", raw=raw)
            return self._wait_for_fill(req, order_id, q_price, q_shares, direction)

        return OrderResult(request=req, status=OrderStatus.PLACED, order_id=order_id, raw=raw)

    def _wait_for_fill(
        self, req: OrderRequest, order_id: str, q_price: float, q_shares: float,
        direction: str = "BUY",
    ) -> OrderResult:
        deadline = time.time() + self.cfg.order_timeout_seconds
        while time.time() < deadline:
            time.sleep(self.cfg.order_poll_interval)
            try:
                data = self._client.get_order(order_id)
            except Exception:
                continue
            if not isinstance(data, dict):
                continue
            s = str(data.get("status", "")).lower()
            if s == "matched":
                filled = self._extract_shares(data, q_shares, direction)
                fill_price = self._extract_fill_price(data, q_price, filled, direction)
                return OrderResult(
                    request=req, status=OrderStatus.FILLED,
                    order_id=order_id,
                    filled_shares=filled, filled_price=fill_price,
                    filled_notional=filled * fill_price, raw=data,
                )
        # Timeout: cancel
        try:
            self._client.cancel(order_id)
        except Exception:
            pass
        return OrderResult(
            request=req, status=OrderStatus.EXPIRED,
            order_id=order_id,
            error="order timed out",
        )

    def _extract_shares(self, payload: dict, fallback: float, direction: str = "BUY") -> float:
        # Polymarket CLOB response semantics differ by direction:
        #   BUY:  makingAmount=USDC paid,   takingAmount=tokens received  → use takingAmount
        #   SELL: makingAmount=tokens given, takingAmount=USDC received   → use makingAmount
        if direction == "SELL":
            keys = ("makingAmount", "making_amount", "size", "filled_size")
        else:
            keys = ("takingAmount", "taking_amount", "size", "filled_size")
        for key in keys:
            v = payload.get(key)
            if v is not None:
                try:
                    fv = float(v)
                    if fv > 0:
                        return fv
                except (TypeError, ValueError):
                    continue
        return fallback

    def _extract_fill_price(
        self, payload: dict, fallback: float, shares_filled: float, direction: str = "BUY"
    ) -> float:
        """Compute the actual execution price from the CLOB order response.

        For SELL: takingAmount = USDC received, makingAmount = tokens sold
                  fill_price = takingAmount / shares_filled
        For BUY:  makingAmount = USDC paid, takingAmount = tokens received
                  fill_price = makingAmount / shares_filled

        Falls back to the submitted limit price if fields are absent or inconsistent.
        """
        if shares_filled <= 0:
            return fallback
        try:
            if direction == "SELL":
                usdc = float(payload.get("takingAmount") or payload.get("taking_amount") or 0)
            else:
                usdc = float(payload.get("makingAmount") or payload.get("making_amount") or 0)
            if usdc > 0:
                computed = usdc / shares_filled
                # Sanity-check: must be a valid Polymarket price (0.001 – 0.999)
                if 0.001 <= computed <= 0.999:
                    return round(computed, 4)
        except (TypeError, ValueError, ZeroDivisionError):
            pass
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
                    side=Side.UP, shares=0, limit_price=0,
                ),
                status=status_map.get(s, OrderStatus.PENDING),
                order_id=order_id, raw=data,
            )
        except Exception:
            return None

    def cancel_order(self, order_id: str) -> bool:
        try:
            self._client.cancel(order_id)
            return True
        except Exception:
            return False

    def get_balance_usd(self) -> float:
        from py_clob_client.clob_types import BalanceAllowanceParams
        try:
            bap = BalanceAllowanceParams(asset_type="COLLATERAL", signature_type=2)
            data = self._client.get_balance_allowance(bap)
            raw = int(str(data.get("balance", "0"))) if isinstance(data, dict) else 0
            return raw / 1_000_000.0
        except Exception:
            return 0.0

    def get_open_positions_value(self) -> tuple[float, int]:
        """Return (total_current_value, n_positions) from open conditional token positions."""
        try:
            resp = requests.get(
                "https://data-api.polymarket.com/positions",
                params={"user": self.cfg.wallet_address},
                timeout=8,
            )
            if resp.status_code != 200:
                return 0.0, 0
            positions = resp.json()
            if not isinstance(positions, list):
                return 0.0, 0
            total = sum(float(p.get("currentValue", 0)) for p in positions)
            return total, len(positions)
        except Exception:
            return 0.0, 0

    def get_spendable_shares(self, token_id: str) -> Optional[float]:
        """Query the CLOB for the actual conditional-token balance of *token_id*.

        Uses the same ``update_balance_allowance`` + ``get_balance_allowance``
        pattern as bot3.  Balances are returned in 1 × 10⁻⁶ units by the API
        so we divide by 1_000_000.  Returns ``None`` on any error.
        """
        from py_clob_client.clob_types import BalanceAllowanceParams
        try:
            self._client.update_balance_allowance(
                BalanceAllowanceParams(
                    asset_type="CONDITIONAL",
                    token_id=token_id,
                    signature_type=2,
                )
            )
        except Exception:
            pass
        try:
            data = self._client.get_balance_allowance(
                BalanceAllowanceParams(
                    asset_type="CONDITIONAL",
                    token_id=token_id,
                    signature_type=2,
                )
            )
            raw = data.get("balance") if isinstance(data, dict) else None
            if raw is None:
                return None
            return int(str(raw)) / 1_000_000.0
        except Exception:
            return None


# ── Paper executor ────────────────────────────────────────────────────────────

class PaperExecutor(BaseExecutor):
    """Simulated execution using live order book data for realistic fills."""

    def __init__(self, cfg: Bot2Config):
        self.cfg = cfg
        self._balance = cfg.paper_initial_balance
        self._orders: dict[str, dict] = {}       # order_id → state
        self._session = requests.Session()
        logger.info(
            "PaperExecutor started | balance=$%.2f", self._balance
        )

    # ------------------------------------------------------------------

    def buy(self, req: OrderRequest) -> OrderResult:
        return self._simulate("BUY", req)

    def sell(self, req: OrderRequest) -> OrderResult:
        return self._simulate("SELL", req)

    def _simulate(self, direction: str, req: OrderRequest) -> OrderResult:
        q_price, q_shares = quantize(direction, req.limit_price, req.shares)
        if q_shares <= 0:
            return OrderResult(request=req, status=OrderStatus.REJECTED, error="zero shares")

        order_id = f"paper-{uuid.uuid4().hex[:12]}"
        fill_price, fill_shares = self._determine_fill(direction, req, q_price, q_shares)

        if fill_shares <= 0:
            # Store as GTC open order
            self._orders[order_id] = {
                "direction": direction, "req": req,
                "q_price": q_price, "q_shares": q_shares,
                "filled": 0.0, "status": "live",
            }
            logger.debug("[paper] %s %s %.0f@%.4f → placed (no fill)", direction, req.side.value, q_shares, q_price)
            return OrderResult(request=req, status=OrderStatus.PLACED, order_id=order_id)

        notional = fill_shares * fill_price
        if direction == "BUY":
            self._balance -= notional
        else:
            self._balance += notional

        logger.debug(
            "[paper] %s %s %.0f@%.4f → filled %.0f@%.4f | balance=$%.2f",
            direction, req.side.value, q_shares, q_price,
            fill_shares, fill_price, self._balance,
        )
        return OrderResult(
            request=req, status=OrderStatus.FILLED,
            order_id=order_id,
            filled_shares=fill_shares,
            filled_price=fill_price,
            filled_notional=notional,
        )

    def _determine_fill(
        self, direction: str, req: OrderRequest, q_price: float, q_shares: float
    ) -> tuple[float, float]:
        """Determine fill price/shares based on live order book.

        For BUY orders the fill is always the full requested size (at q_price)
        when asks exist at or below q_price – we do NOT cap by book depth because
        in paper mode we can't realistically model liquidity consumption and the
        depth figures from the REST book are often incomplete.  For SELL orders we
        are more conservative and require bid depth to cover the size.
        """
        token_id = req.market.token_id_for(req.side)
        book = self._fetch_book(token_id) if token_id else {}
        bids = book.get("bids", [])
        asks = book.get("asks", [])

        if direction == "BUY":
            # Collect all ask levels at or below our limit, sorted best-first
            eligible = sorted(
                (float(a["price"]), float(a.get("size", 0)))
                for a in asks
                if a.get("price") and float(a["price"]) <= q_price
            )
            if not eligible:
                # No ask ≤ limit in book.  Assume fill at limit (pre-market
                # books are often thin but orders do get matched at open).
                return q_price, q_shares

            # Walk the book to fill as much as possible
            fill_price = eligible[0][0]       # best ask price (price improvement)
            depth_avail = sum(s for _, s in eligible)

            if depth_avail >= q_shares:
                # Enough depth – full fill at best ask
                return fill_price, q_shares
            elif depth_avail >= q_shares * 0.1:
                # Partial depth – fill what's available; rest at limit price
                # (weighted average price)
                weighted = fill_price * depth_avail + q_price * (q_shares - depth_avail)
                avg_price = weighted / q_shares
                return round(avg_price, 4), q_shares
            else:
                # Negligible depth (< 10% of order) – fill at limit price
                return q_price, q_shares

        else:  # SELL
            # We SELL at limit q_price: fill against bids ≥ q_price
            available = sorted(
                [
                    (float(b["price"]), float(b.get("size", 0)))
                    for b in bids
                    if b.get("price") and float(b["price"]) >= q_price
                ],
                reverse=True,
            )
            if not available:
                # No bid ≥ limit – order stays live (GTC)
                return q_price, 0.0
            fill_price = available[0][0]
            fill_avail = sum(s for _, s in available)
            fill_shares = min(q_shares, fill_avail)
            return fill_price, fill_shares

    def _fetch_book(self, token_id: str) -> dict:
        try:
            resp = self._session.get(
                f"{CLOB_API}/book",
                params={"token_id": token_id},
                timeout=5,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            logger.debug("PaperExecutor book fetch failed for %s: %s", token_id[:12], exc)
            return {}

    def check_order(self, order_id: str) -> Optional[OrderResult]:
        state = self._orders.get(order_id)
        if not state:
            return None
        # Re-evaluate fill for pending GTC orders
        req: OrderRequest = state["req"]
        direction: str = state["direction"]
        q_price: float = state["q_price"]
        q_shares: float = state["q_shares"]
        fill_price, fill_shares = self._determine_fill(direction, req, q_price, q_shares)
        if fill_shares > 0:
            state["status"] = "matched"
            state["filled"] = fill_shares
            notional = fill_shares * fill_price
            if direction == "BUY":
                self._balance -= notional
            else:
                self._balance += notional
            return OrderResult(
                request=req, status=OrderStatus.FILLED,
                order_id=order_id,
                filled_shares=fill_shares,
                filled_price=fill_price,
                filled_notional=notional,
            )
        return OrderResult(request=req, status=OrderStatus.PLACED, order_id=order_id)

    def cancel_order(self, order_id: str) -> bool:
        if order_id in self._orders:
            self._orders[order_id]["status"] = "cancelled"
            return True
        return False

    def get_balance_usd(self) -> float:
        return self._balance


# ── Factory ──────────────────────────────────────────────────────────────────

def create_executor(cfg: Bot2Config) -> BaseExecutor:
    if cfg.paper_mode:
        return PaperExecutor(cfg)
    try:
        ex = LiveExecutor(cfg)
        logger.info("LiveExecutor initialised")
        return ex
    except Exception as exc:
        logger.error("LiveExecutor init failed: %s – falling back to paper mode", exc)
        return PaperExecutor(cfg)
