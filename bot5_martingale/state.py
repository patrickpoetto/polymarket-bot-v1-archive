"""Persistent bot state – JSON-backed, crash-recoverable.

The state file is rewritten atomically after every meaningful transition
so the bot can resume from its last known-good state after a restart.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from bot5_martingale.models import Side, TradeRecord, TradeResult
from bot5_martingale.progression import ProgressionSnapshot
from bot5_martingale.strategy import StrategyState

logger = logging.getLogger(__name__)


@dataclass
class DailyStats:
    date: str = ""
    trades: int = 0
    wins: int = 0
    losses: int = 0
    stopouts: int = 0
    pnl: float = 0.0


@dataclass
class BotStateData:
    """Full serialisable bot state."""
    # Strategy
    strategy: StrategyState = field(default_factory=StrategyState)
    # Progression
    progression: ProgressionSnapshot = field(default_factory=ProgressionSnapshot)
    # Market tracking
    last_resolved_start_ts: int = 0
    market_sequential_index: int = 0
    # Trade log
    trades: list[dict] = field(default_factory=list)
    # Cumulative
    total_pnl: float = 0.0
    total_trades: int = 0
    total_wins: int = 0
    total_losses: int = 0
    total_stopouts: int = 0
    # Daily
    daily: DailyStats = field(default_factory=DailyStats)
    # Pending orders (condition_id -> serialised OrderResult-like dict)
    pending_orders: dict = field(default_factory=dict)


class StateManager:
    """Load / save bot state to a JSON file with atomic writes."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data = BotStateData()

    def load(self) -> BotStateData:
        if not self.path.exists():
            logger.info("No prior state file; starting fresh")
            self.data = BotStateData()
            return self.data
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            self.data = self._deserialise(raw)
            logger.info(
                "State loaded: idx=%d, pnl=%.2f, trades=%d",
                self.data.market_sequential_index,
                self.data.total_pnl,
                self.data.total_trades,
            )
        except Exception as exc:
            logger.error("Failed to load state from %s: %s – starting fresh", self.path, exc)
            self.data = BotStateData()
        return self.data

    def save(self) -> None:
        try:
            payload = json.dumps(self._serialise(self.data), indent=2, default=str)
            fd, tmp = tempfile.mkstemp(dir=str(self.path.parent), suffix=".tmp")
            try:
                os.write(fd, payload.encode("utf-8"))
                os.fsync(fd)
            finally:
                os.close(fd)
            os.replace(tmp, str(self.path))
        except Exception as exc:
            logger.error("State save failed: %s", exc)

    # ------------------------------------------------------------------
    # Convenience mutators
    # ------------------------------------------------------------------

    def record_trade(self, rec: TradeRecord) -> None:
        self.data.trades.append({
            "market": rec.market_question,
            "condition_id": rec.condition_id,
            "side": rec.side.value,
            "level": rec.progression_level,
            "target_usd": rec.target_size_usd,
            "filled_usd": rec.filled_size_usd,
            "price": rec.filled_price,
            "shares": rec.filled_shares,
            "fee": rec.fee,
            "result": rec.result.value,
            "pnl": rec.pnl,
            "ts_entry": rec.ts_entry,
            "ts_resolved": rec.ts_resolved,
            "inferred": rec.inferred,
        })
        # Keep last 500 in state file to limit size
        if len(self.data.trades) > 500:
            self.data.trades = self.data.trades[-500:]

        self.data.total_trades += 1
        if rec.result is TradeResult.WIN:
            self.data.total_wins += 1
            self.data.daily.wins += 1
        elif rec.result is TradeResult.LOSS:
            self.data.total_losses += 1
            self.data.daily.losses += 1
        self.data.total_pnl += rec.pnl
        self.data.daily.pnl += rec.pnl
        self.data.daily.trades += 1

    def record_stopout(self) -> None:
        self.data.total_stopouts += 1
        self.data.daily.stopouts += 1

    def reset_daily(self, date_str: str) -> None:
        self.data.daily = DailyStats(date=date_str)

    def reset_all(self) -> None:
        """Reset all persisted state and write it immediately."""
        self.data = BotStateData()
        self.save()

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    @staticmethod
    def _serialise(data: BotStateData) -> dict:
        return {
            "strategy": asdict(data.strategy),
            "progression": asdict(data.progression),
            "last_resolved_start_ts": data.last_resolved_start_ts,
            "market_sequential_index": data.market_sequential_index,
            "trades": data.trades,
            "total_pnl": data.total_pnl,
            "total_trades": data.total_trades,
            "total_wins": data.total_wins,
            "total_losses": data.total_losses,
            "total_stopouts": data.total_stopouts,
            "daily": asdict(data.daily),
            "pending_orders": data.pending_orders,
        }

    @staticmethod
    def _deserialise(raw: dict) -> BotStateData:
        strat = raw.get("strategy", {})
        prog = raw.get("progression", {})
        daily = raw.get("daily", {})
        return BotStateData(
            strategy=StrategyState(**{k: strat[k] for k in StrategyState.__dataclass_fields__ if k in strat}),
            progression=ProgressionSnapshot(
                level=prog.get("level", 0),
                cumulative_spend=prog.get("cumulative_spend", 0.0),
                cycle_fills=prog.get("cycle_fills", []),
            ),
            last_resolved_start_ts=raw.get("last_resolved_start_ts", 0),
            market_sequential_index=raw.get("market_sequential_index", 0),
            trades=raw.get("trades", []),
            total_pnl=raw.get("total_pnl", 0.0),
            total_trades=raw.get("total_trades", 0),
            total_wins=raw.get("total_wins", 0),
            total_losses=raw.get("total_losses", 0),
            total_stopouts=raw.get("total_stopouts", 0),
            daily=DailyStats(**{k: daily[k] for k in DailyStats.__dataclass_fields__ if k in daily}),
            pending_orders=raw.get("pending_orders", {}),
        )
