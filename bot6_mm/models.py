"""Data models for Bot6 Market Maker."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

# Re-export Side so bot6 code can import it from here if needed
from bot2_btc.models import MarketInfo, Side  # noqa: F401


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class ActiveOrder:
    order_id: str
    side: Side
    order_side: OrderSide    # BUY or SELL
    price: float
    size: float
    placed_at: float = field(default_factory=time.time)
    size_filled: float = 0.0


@dataclass
class SideInventory:
    """Tracks inventory and average cost for one outcome side."""
    side: Side
    shares: float = 0.0
    avg_cost: float = 0.0
    realized_pnl: float = 0.0
    total_buys: int = 0
    total_sells: int = 0
    total_buy_shares: float = 0.0
    total_sell_shares: float = 0.0

    def on_buy_fill(self, price: float, shares: float) -> None:
        total = self.avg_cost * self.shares + price * shares
        self.shares += shares
        self.avg_cost = total / self.shares if self.shares > 0 else 0.0
        self.total_buys += 1
        self.total_buy_shares += shares

    def on_sell_fill(self, price: float, shares: float) -> None:
        self.realized_pnl += (price - self.avg_cost) * shares
        self.shares = max(0.0, self.shares - shares)
        self.total_sells += 1
        self.total_sell_shares += shares


@dataclass
class MarketSession:
    """State for a single market's MM session."""
    market_info: MarketInfo
    market_index: int
    started_at: float = field(default_factory=time.time)

    up_inv: SideInventory = field(default_factory=lambda: SideInventory(Side.UP))
    dn_inv: SideInventory = field(default_factory=lambda: SideInventory(Side.DOWN))

    # (Side, OrderSide) → ActiveOrder
    active_orders: dict = field(default_factory=dict)

    last_requote_time: float = 0.0
    last_up_mid: Optional[float] = None
    last_dn_mid: Optional[float] = None

    quoting_stopped: bool = False
    session_closed: bool = False

    # Winner-hold: after mid stays ≥ mm_stop_selling_above for confirm seconds,
    # stop placing SELLs on that side (ride to $1). Cleared when mid < reset_below.
    up_strong_since: Optional[float] = None   # first wall-clock ts in current ≥threshold streak
    dn_strong_since: Optional[float] = None
    up_winner_hold: bool = False
    dn_winner_hold: bool = False
    up_buy_cooldown_until: float = 0.0
    dn_buy_cooldown_until: float = 0.0
    last_endgame_derisk_at: float = 0.0
    last_inv_mgmt_trim_at: float = 0.0
    # EMA of implied P(Up) from mids: up_mid / (up_mid + dn_mid), not raw up_mid alone.
    ema_p_up: Optional[float] = None
    up_target_inventory: float = 0.0
    dn_target_inventory: float = 0.0

    @property
    def total_realized_pnl(self) -> float:
        return self.up_inv.realized_pnl + self.dn_inv.realized_pnl

    def order_key(self, side: Side, order_side: OrderSide) -> tuple:
        return (side, order_side)
