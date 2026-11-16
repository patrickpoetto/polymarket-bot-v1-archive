"""Env config for Bot6 Paired MM (matched UP+DN, OSL)."""
from __future__ import annotations

from dataclasses import dataclass

from dotenv import load_dotenv

from bot6_mm.config import Bot6Config, _bool, _float, _int, _str

load_dotenv()


@dataclass(frozen=True)
class Bot6PairsConfig:
    """Paired-MM v2: bid_sum trigger, quote-at-bid, long orphan window."""

    bid_sum_trigger: float = 0.97
    quote_at_bid_on_trigger: bool = True
    aggressive_half_spread: float = 0.0
    half_spread: float = 0.01

    max_pair_cost: float = 0.99
    max_ask_sum: float = 1.02
    order_size: float = 10.0
    max_pairs: float = 50.0
    max_triggers_per_market: int = 40
    trigger_cooldown_s: float = 3.0
    order_timeout_s: float = 15.0

    osl_gap: float = 0.01
    opposite_side_lock: bool = True
    orphan_timeout_s: float = 8.0
    orphan_sell_back: bool = True
    hold_winner_orphan: bool = True
    acquire_pair_at_bid: bool = True

    trade_window_seconds: float = 280.0
    stop_quoting_seconds: float = 30.0
    exit_sell_floor: float = 0.01
    requote_interval: float = 0.5
    log_file: str = "logs/bot6_pairs/bot6_pairs.log"

    @classmethod
    def from_env(cls) -> "Bot6PairsConfig":
        return cls(
            bid_sum_trigger=_float("BOT6_PAIRS_BID_SUM_TRIGGER", 0.98),
            quote_at_bid_on_trigger=_bool("BOT6_PAIRS_QUOTE_AT_BID", True),
            aggressive_half_spread=_float("BOT6_PAIRS_AGGRESSIVE_HALF_SPREAD", 0.0),
            half_spread=_float("BOT6_PAIRS_HALF_SPREAD", 0.01),
            max_pair_cost=_float("BOT6_PAIRS_MAX_PAIR_COST", 0.99),
            max_ask_sum=_float("BOT6_PAIRS_MAX_ASK_SUM", 1.02),
            order_size=_float("BOT6_PAIRS_ORDER_SIZE", 10.0),
            max_pairs=_float("BOT6_PAIRS_MAX_PAIRS", 50.0),
            max_triggers_per_market=_int("BOT6_PAIRS_MAX_TRIGGERS", 40),
            trigger_cooldown_s=_float("BOT6_PAIRS_TRIGGER_COOLDOWN", 3.0),
            order_timeout_s=_float("BOT6_PAIRS_ORDER_TIMEOUT", 15.0),
            osl_gap=_float("BOT6_PAIRS_OSL_GAP", 0.01),
            opposite_side_lock=_bool("BOT6_PAIRS_OSL_ENABLED", True),
            orphan_timeout_s=_float("BOT6_PAIRS_ORPHAN_TIMEOUT", 8.0),
            orphan_sell_back=_bool("BOT6_PAIRS_ORPHAN_SELL_BACK", True),
            hold_winner_orphan=_bool("BOT6_PAIRS_HOLD_WINNER_ORPHAN", True),
            acquire_pair_at_bid=_bool("BOT6_PAIRS_ACQUIRE_AT_BID", True),
            trade_window_seconds=_float("BOT6_PAIRS_TRADE_WINDOW", 280.0),
            stop_quoting_seconds=_float("BOT6_PAIRS_STOP_QUOTING", 30.0),
            exit_sell_floor=_float("BOT6_PAIRS_EXIT_SELL_FLOOR", 0.01),
            requote_interval=_float("BOT6_PAIRS_REQUOTE_INTERVAL", 0.5),
            log_file=_str("BOT6_PAIRS_LOG_FILE", "logs/bot6_pairs/bot6_pairs.log"),
        )

    def base(self) -> Bot6Config:
        return Bot6Config.from_env()
