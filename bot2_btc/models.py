"""Data models for Bot2 BTC."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ── Enums ───────────────────────────────────────────────────────────────────

class Side(str, Enum):
    UP = "Up"
    DOWN = "Down"

    @classmethod
    def from_str(cls, s: str) -> "Side":
        return cls.UP if s.strip().lower() in ("up", "yes") else cls.DOWN

    def opposite(self) -> "Side":
        return Side.DOWN if self is Side.UP else Side.UP


class OrderStatus(str, Enum):
    PENDING = "pending"
    PLACED = "placed"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class SessionPhase(str, Enum):
    """Lifecycle per mercato 15m dopo il pre-pos."""

    BOTH_OPEN = "both_open"   # straddle attivo; qui andrà la strategia intramarket
    DONE = "done"             # niente più azioni fino a chiusura contabile


class BotPhase(str, Enum):
    STARTING = "starting"
    AWAITING_NEXT_MARKET = "awaiting_next_market"
    PRE_POSITIONING = "pre_positioning"
    POSITIONED = "positioned"     # ordini piazzati/riempiti, in attesa di start
    MANAGING = "managing"         # mercato live
    IDLE = "idle"


# ── Market ──────────────────────────────────────────────────────────────────

@dataclass
class MarketInfo:
    condition_id: str
    question: str
    outcomes: list[str]           # e.g. ["Up", "Down"]
    token_ids: list[str]          # same order as outcomes
    end_timestamp: int
    event_slug: str = ""
    timestamp_tag: int = 0        # unix ts extracted from slug (= start_ts)

    @property
    def start_timestamp(self) -> int:
        return self.timestamp_tag

    @property
    def token_id_by_side(self) -> dict[str, str]:
        """Map outcome string (e.g. 'Up') → token_id."""
        return {
            outcome: tid
            for outcome, tid in zip(self.outcomes, self.token_ids)
        }

    def token_id_for(self, side: Side) -> Optional[str]:
        return self.token_id_by_side.get(side.value)

    def __str__(self) -> str:
        return f"{self.question[:60]} (ends {self.end_timestamp})"


# ── Orders ───────────────────────────────────────────────────────────────────

@dataclass
class OrderRequest:
    market: MarketInfo
    side: Side
    shares: float          # number of shares
    limit_price: float     # per-share limit
    order_label: str = ""  # human-readable label for logging

    @property
    def notional_usd(self) -> float:
        return self.shares * self.limit_price


@dataclass
class OrderResult:
    request: OrderRequest
    status: OrderStatus
    order_id: Optional[str] = None
    filled_shares: float = 0.0
    filled_price: float = 0.0
    filled_notional: float = 0.0
    error: Optional[str] = None
    raw: Optional[dict] = None

    @property
    def ok(self) -> bool:
        return self.status in (OrderStatus.FILLED, OrderStatus.PARTIAL)


# ── Session ─────────────────────────────────────────────────────────────────

@dataclass
class Position:
    """Tracks shares held for one side."""
    side: Side
    shares: float = 0.0
    avg_entry_price: float = 0.0
    order_id: Optional[str] = None

    @property
    def cost_basis(self) -> float:
        return self.shares * self.avg_entry_price


@dataclass
class MarketSession:
    """Stato di un mercato 15m: solo straddle + PnL realizzato (strategia = bot.py)."""
    market: MarketInfo
    phase: SessionPhase = SessionPhase.BOTH_OPEN

    up_position: Position = field(default_factory=lambda: Position(Side.UP))
    down_position: Position = field(default_factory=lambda: Position(Side.DOWN))

    realised_pnl: float = 0.0
    market_index: int = 0

    # Intramarket strategy (snapshot at open + risk budget)
    strat_ref_up: float = 0.0
    strat_ref_down: float = 0.0
    strat_turnover_shares: float = 0.0

    def position_for(self, side: Side) -> Position:
        return self.up_position if side is Side.UP else self.down_position

    @property
    def total_cost(self) -> float:
        return self.up_position.cost_basis + self.down_position.cost_basis


# ── Prices ───────────────────────────────────────────────────────────────────

@dataclass
class Prices:
    """Best bid/ask snapshot for one token."""
    bid: Optional[float] = None
    ask: Optional[float] = None
    stale: bool = False

    @property
    def mid(self) -> Optional[float]:
        if self.bid is not None and self.ask is not None:
            return (self.bid + self.ask) / 2
        return self.bid or self.ask
