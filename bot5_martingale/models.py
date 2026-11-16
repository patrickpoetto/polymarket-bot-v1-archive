"""Shared data models used across all bot5 modules."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Side(str, Enum):
    UP = "Up"
    DOWN = "Down"

    @property
    def opposite(self) -> "Side":
        return Side.DOWN if self is Side.UP else Side.UP


class BotMode(str, Enum):
    LIVE = "live"
    PAPER = "paper"


class MartingaleMode(str, Enum):
    SINGLE = "single"
    DOUBLE = "double"


class OrderStatus(str, Enum):
    PENDING = "pending"
    PLACED = "placed"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class MarketPhase(str, Enum):
    UNKNOWN = "unknown"
    PRE_OPEN = "pre_open"
    OPEN = "open"
    LAST_MINUTE = "last_minute"
    CLOSED = "closed"
    RESOLVED = "resolved"


class TradeResult(str, Enum):
    WIN = "win"
    LOSS = "loss"
    PENDING = "pending"
    SKIPPED = "skipped"


# ---------------------------------------------------------------------------
# Market
# ---------------------------------------------------------------------------

@dataclass
class MarketInfo:
    condition_id: str
    question: str
    outcomes: list[str]
    token_ids: list[str]
    end_timestamp: int
    event_slug: str = ""
    timestamp_tag: int = 0

    @property
    def token_id_by_side(self) -> dict[Side, str]:
        mapping: dict[Side, str] = {}
        for outcome, tid in zip(self.outcomes, self.token_ids):
            side = Side.UP if outcome.lower() in ("up", "yes") else Side.DOWN
            mapping[side] = tid
        return mapping

    @property
    def seconds_left(self) -> float:
        return max(0.0, self.end_timestamp - time.time())


# ---------------------------------------------------------------------------
# Orders
# ---------------------------------------------------------------------------

@dataclass
class OrderRequest:
    market: MarketInfo
    side: Side
    size_usd: float
    limit_price: float
    tag: str = ""

    @property
    def token_id(self) -> str:
        return self.market.token_id_by_side[self.side]


@dataclass
class OrderResult:
    request: OrderRequest
    status: OrderStatus
    order_id: Optional[str] = None
    filled_size: float = 0.0
    filled_price: float = 0.0
    filled_shares: float = 0.0
    fee: float = 0.0
    ts: float = field(default_factory=time.time)
    raw: Optional[dict] = None
    error: Optional[str] = None

    @property
    def is_filled(self) -> bool:
        return self.status in (OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED)


# ---------------------------------------------------------------------------
# Trade record (for logging / state)
# ---------------------------------------------------------------------------

@dataclass
class TradeRecord:
    market_question: str
    condition_id: str
    side: Side
    progression_level: int
    target_size_usd: float
    filled_size_usd: float
    filled_price: float
    filled_shares: float
    fee: float
    result: TradeResult = TradeResult.PENDING
    pnl: float = 0.0
    ts_entry: float = 0.0
    ts_resolved: float = 0.0
    inferred: bool = False


# ---------------------------------------------------------------------------
# Progression snapshot
# ---------------------------------------------------------------------------

@dataclass
class ProgressionState:
    level: int = 0
    locked: bool = False
    locked_side: Optional[Side] = None
    consecutive_losses: int = 0
    book_idx: int = 0  # for double-martingale: 0 or 1
