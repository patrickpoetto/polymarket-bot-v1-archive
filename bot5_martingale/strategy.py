"""Lag-2 strategy with lock-after-N-losses policy.

Pure strategy logic – no I/O, no side-effects.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from bot5_martingale.models import Side


@dataclass
class StrategyState:
    """Serialisable snapshot of strategy state."""
    outcomes: list[str] = field(default_factory=list)
    consecutive_losses: int = 0
    locked: bool = False
    locked_side: Optional[str] = None


class Lag2LockStrategy:
    """Predict outcome[t] = outcome[t-lag], with optional lock on losing streak.

    Lock modes:
        - "same": after *lock_after_n* consecutive losses, keep predicting
          the **same** side that lost until we win.
        - "anti": after *lock_after_n* consecutive losses, flip to the
          **opposite** side until we win.
    """

    def __init__(self, lag: int = 2, lock_after_n: int = 1, lock_mode: str = "same"):
        if lag < 1:
            raise ValueError(f"lag must be >= 1, got {lag}")
        if lock_after_n < 1:
            raise ValueError(f"lock_after_n must be >= 1, got {lock_after_n}")
        if lock_mode not in ("same", "anti"):
            raise ValueError(f"lock_mode must be 'same' or 'anti', got {lock_mode!r}")

        self.lag = lag
        self.lock_after_n = lock_after_n
        self.lock_mode = lock_mode

        self._outcomes: list[Side] = []
        self._consecutive_losses: int = 0
        self._locked: bool = False
        self._locked_side: Optional[Side] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_outcome(self, outcome: Side) -> None:
        """Append a resolved market outcome to history."""
        self._outcomes.append(outcome)

    def predict(self, market_index: int) -> Optional[Side]:
        """Return the predicted side for *market_index*, or None if not
        enough history for lag-based prediction and not locked."""
        if self._locked and self._locked_side is not None:
            return self._locked_side

        ref = market_index - self.lag
        if ref < 0 or ref >= len(self._outcomes):
            return None
        return self._outcomes[ref]

    def predict_lag_only(self, market_index: int) -> Optional[Side]:
        """Pure lag-based prediction, ignoring any lock state.

        Used for pre-placed orders whose side is determined at placement
        time, before the lock state may change.
        """
        ref = market_index - self.lag
        if ref < 0 or ref >= len(self._outcomes):
            return None
        return self._outcomes[ref]

    def report_result(self, predicted_side: Side, won: bool) -> None:
        """Feed back the trade result to update lock/streak state."""
        if won:
            self._locked = False
            self._locked_side = None
            self._consecutive_losses = 0
        else:
            self._consecutive_losses += 1
            if self._consecutive_losses >= self.lock_after_n:
                self._locked = True
                if self.lock_mode == "same":
                    self._locked_side = predicted_side
                else:
                    self._locked_side = predicted_side.opposite

    # ------------------------------------------------------------------
    # Warmup helpers
    # ------------------------------------------------------------------

    @property
    def ready(self) -> bool:
        """True once enough outcomes have been recorded to start trading."""
        return len(self._outcomes) >= self.lag

    @property
    def next_tradeable_index(self) -> int:
        """Index of the next market that can be traded (= first index
        for which predict() would return a non-None value, ignoring lock)."""
        return len(self._outcomes) + self.lag if not self.ready else len(self._outcomes)

    @property
    def market_count(self) -> int:
        return len(self._outcomes)

    @property
    def is_locked(self) -> bool:
        return self._locked

    @property
    def locked_side(self) -> Optional[Side]:
        return self._locked_side

    @property
    def consecutive_losses(self) -> int:
        return self._consecutive_losses

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def snapshot(self) -> StrategyState:
        return StrategyState(
            outcomes=[s.value for s in self._outcomes],
            consecutive_losses=self._consecutive_losses,
            locked=self._locked,
            locked_side=self._locked_side.value if self._locked_side else None,
        )

    def restore(self, state: StrategyState) -> None:
        self._outcomes = [Side(s) for s in state.outcomes]
        self._consecutive_losses = state.consecutive_losses
        self._locked = state.locked
        self._locked_side = Side(state.locked_side) if state.locked_side else None
