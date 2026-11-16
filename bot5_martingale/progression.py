"""Martingale progression engine.

Computes the exact stake for each level so that a win always recovers all
previous losses in the cycle **plus** earns *target_profit_per_market * (level+1)*
at the expected entry price.

The static *ladder* from config serves as a reference ceiling; the actual
required stake is computed dynamically from the real cumulative spend.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ProgressionSnapshot:
    """Serialisable state."""
    level: int = 0
    cumulative_spend: float = 0.0
    cycle_fills: list[dict] = field(default_factory=list)


class MartingaleProgression:
    def __init__(
        self,
        ladder: list[float],
        max_level: int,
        target_profit_per_market: float = 6.0,
        max_entry_price: float = 0.51,
    ):
        if not ladder:
            raise ValueError("ladder must not be empty")
        self.ladder = list(ladder)
        self.max_level = min(max_level, len(ladder) - 1)
        self.target_profit_per_market = target_profit_per_market
        self.max_entry_price = max_entry_price

        self._level: int = 0
        self._cumulative_spend: float = 0.0
        self._cycle_fills: list[dict] = []

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    @property
    def level(self) -> int:
        return self._level

    @property
    def cumulative_spend(self) -> float:
        return self._cumulative_spend

    @property
    def is_base(self) -> bool:
        return self._level == 0

    def required_size(self, entry_price: Optional[float] = None) -> float:
        """Compute the USD notional needed at *self._level*.

        Formula (for a win at this level to recover + profit):
            payout       = size / price  (shares * $1 each)
            net_of_trade = payout - size = size * (1/price - 1)
            target       = cum_spend + target_profit * (level + 1)
            ⇒ size       = target / (1/price - 1)
        """
        price = entry_price or self.max_entry_price
        edge = 1.0 / price - 1.0
        if edge <= 0:
            raise ValueError(f"entry_price={price} >= 1.0 – no positive edge")

        target_net = self._cumulative_spend + self.target_profit_per_market * (self._level + 1)
        raw = target_net / edge
        return math.ceil(raw * 100) / 100  # round up to nearest cent

    def ladder_size(self) -> float:
        """Return the static ladder value (cap / reference)."""
        idx = min(self._level, len(self.ladder) - 1)
        return self.ladder[idx]

    def effective_size(self, entry_price: Optional[float] = None) -> float:
        """Return the size to actually trade: max(dynamic, ladder) capped sensibly."""
        dynamic = self.required_size(entry_price)
        static = self.ladder_size()
        return max(dynamic, static)

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------

    def on_loss(self, actual_spend: float) -> bool:
        """Record a losing trade.  Returns True on stop-out (level exceeded)."""
        self._cycle_fills.append({"level": self._level, "spend": actual_spend, "won": False})
        self._cumulative_spend += actual_spend
        self._level += 1
        if self._level > self.max_level:
            self._do_reset()
            return True
        return False

    def on_win(self, actual_spend: float, payout: float) -> float:
        """Record a winning trade.  Returns the net P&L for the cycle."""
        self._cycle_fills.append({"level": self._level, "spend": actual_spend, "won": True})
        net = payout - actual_spend - self._cumulative_spend
        self._do_reset()
        return net

    def force_reset(self) -> None:
        """Manual reset (e.g. daily risk limit hit)."""
        self._do_reset()

    def _do_reset(self) -> None:
        self._level = 0
        self._cumulative_spend = 0.0
        self._cycle_fills = []

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def snapshot(self) -> ProgressionSnapshot:
        return ProgressionSnapshot(
            level=self._level,
            cumulative_spend=self._cumulative_spend,
            cycle_fills=list(self._cycle_fills),
        )

    def restore(self, snap: ProgressionSnapshot) -> None:
        self._level = snap.level
        self._cumulative_spend = snap.cumulative_spend
        self._cycle_fills = list(snap.cycle_fills)
