"""Centralised, env-driven configuration for bot5."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from bot5_martingale.models import BotMode, MartingaleMode

load_dotenv()

_TRUE = frozenset({"1", "true", "yes", "on"})


def _bool(key: str, default: bool = False) -> bool:
    v = os.getenv(key, "").strip().lower()
    if not v:
        return default
    return v in _TRUE


def _float(key: str, default: float) -> float:
    v = os.getenv(key)
    return float(v) if v else default


def _int(key: str, default: int) -> int:
    v = os.getenv(key)
    return int(v) if v else default


def _str(key: str, default: str = "") -> str:
    return os.getenv(key, default).strip()


def _float_list(key: str, default: str) -> list[float]:
    raw = os.getenv(key, default).strip()
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


@dataclass(frozen=True)
class Bot5Config:
    # -- mode --
    mode: BotMode = BotMode.PAPER
    martingale_mode: MartingaleMode = MartingaleMode.SINGLE

    # -- wallet / exchange --
    private_key: str = ""
    wallet_address: str = ""
    chain_id: int = 137
    clob_host: str = "https://clob.polymarket.com"

    # -- strategy --
    lag: int = 2
    lock_after_n_losses: int = 1
    lock_mode: str = "same"  # "same" or "anti"

    # -- market --
    market_interval_seconds: int = 900
    market_url_pattern: str = "btc-updown-15m-{ts}"
    market_search_slug: str = "Will the price of Bitcoin"

    # -- progression --
    ladder: list[float] = field(default_factory=lambda: [6, 18, 42, 90, 186, 378, 762, 1530])
    max_level: int = 7  # 0-indexed; len(ladder)-1

    # -- pricing / execution --
    max_entry_price: float = 0.51
    target_profit_per_win: float = 6.0
    inference_window_seconds: int = 60
    inference_price_threshold: float = 0.99
    early_signal_enabled: bool = True
    early_signal_window_seconds: int = 30
    early_signal_price_threshold: float = 0.90
    early_signal_confirm_seconds: int = 3
    topup_price_threshold: float = 0.52
    order_timeout_seconds: float = 15.0
    order_poll_interval: float = 1.0
    order_max_retries: int = 5
    order_type_entry: str = "GTC"

    # -- risk --
    max_stopouts_per_day: int = 3
    max_daily_loss_usd: float = 5000.0
    cooldown_after_stopout_seconds: float = 0.0

    # -- timing --
    tick_interval: float = 1.0
    warmup_markets: int = 2
    pre_place_seconds_before_end: int = 120

    # -- paper sim --
    paper_fill_probability_base: float = 0.95
    paper_fill_latency_ms: float = 200.0
    paper_partial_fill_min_pct: float = 0.80

    # -- logging --
    log_dir: str = "logs/bot5_martingale"
    log_level: str = "INFO"

    # -- telegram --
    telegram_enabled: bool = False
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    telegram_commands_enabled: bool = False

    # -- state persistence --
    state_file: str = "bot5_martingale/state.json"
    state_reset_on_stop: bool = False

    # -- websocket --
    ws_enabled: bool = True
    ws_url: str = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
    ws_stale_seconds: float = 3.0

    @classmethod
    def from_env(cls) -> "Bot5Config":
        ladder = _float_list("BOT5_LADDER", "6,18,42,90,186,378,762,1530")
        max_level = _int("BOT5_MAX_LEVEL", len(ladder) - 1)
        return cls(
            mode=BotMode(_str("BOT5_MODE", "paper")),
            martingale_mode=MartingaleMode(_str("BOT5_MARTINGALE_MODE", "single")),
            private_key=_str("PRIVATE_KEY"),
            wallet_address=_str("WALLET_ADDRESS"),
            chain_id=_int("CHAIN_ID", 137),
            clob_host=_str("BOT5_CLOB_HOST", "https://clob.polymarket.com"),
            lag=_int("BOT5_LAG", 2),
            lock_after_n_losses=_int("BOT5_LOCK_AFTER_N_LOSSES", 1),
            lock_mode=_str("BOT5_LOCK_MODE", "same"),
            market_interval_seconds=_int("BOT5_MARKET_INTERVAL_SECONDS", 900),
            market_url_pattern=_str("BOT5_MARKET_URL_PATTERN", "btc-updown-15m-{ts}"),
            market_search_slug=_str("BOT5_MARKET_SEARCH_SLUG", "Will the price of Bitcoin"),
            ladder=ladder,
            max_level=max_level,
            max_entry_price=_float("BOT5_MAX_ENTRY_PRICE", 0.51),
            target_profit_per_win=_float("BOT5_TARGET_PROFIT_PER_WIN", 6.0),
            inference_window_seconds=_int("BOT5_INFERENCE_WINDOW_SECONDS", 60),
            inference_price_threshold=_float("BOT5_INFERENCE_PRICE_THRESHOLD", 0.99),
            early_signal_enabled=_bool("BOT5_EARLY_SIGNAL_ENABLED", True),
            early_signal_window_seconds=_int("BOT5_EARLY_SIGNAL_WINDOW_SECONDS", 30),
            early_signal_price_threshold=_float("BOT5_EARLY_SIGNAL_PRICE_THRESHOLD", 0.90),
            early_signal_confirm_seconds=_int("BOT5_EARLY_SIGNAL_CONFIRM_SECONDS", 3),
            topup_price_threshold=_float("BOT5_TOPUP_PRICE_THRESHOLD", 0.52),
            order_timeout_seconds=_float("BOT5_ORDER_TIMEOUT_SECONDS", 15.0),
            order_poll_interval=_float("BOT5_ORDER_POLL_INTERVAL", 1.0),
            order_max_retries=_int("BOT5_ORDER_MAX_RETRIES", 5),
            order_type_entry=_str("BOT5_ORDER_TYPE_ENTRY", "GTC"),
            max_stopouts_per_day=_int("BOT5_MAX_STOPOUTS_PER_DAY", 3),
            max_daily_loss_usd=_float("BOT5_MAX_DAILY_LOSS_USD", 5000.0),
            cooldown_after_stopout_seconds=_float("BOT5_COOLDOWN_AFTER_STOPOUT_SECONDS", 0.0),
            tick_interval=_float("BOT5_TICK_INTERVAL", 1.0),
            warmup_markets=_int("BOT5_WARMUP_MARKETS", 2),
            pre_place_seconds_before_end=_int("BOT5_PRE_PLACE_SECONDS_BEFORE_END", 120),
            paper_fill_probability_base=_float("BOT5_PAPER_FILL_PROB", 0.95),
            paper_fill_latency_ms=_float("BOT5_PAPER_FILL_LATENCY_MS", 200.0),
            paper_partial_fill_min_pct=_float("BOT5_PAPER_PARTIAL_FILL_MIN_PCT", 0.80),
            log_dir=_str("BOT5_LOG_DIR", "logs/bot5_martingale"),
            log_level=_str("BOT5_LOG_LEVEL", "INFO"),
            telegram_enabled=_bool("BOT5_TELEGRAM_ENABLED"),
            telegram_bot_token=_str("BOT5_TELEGRAM_BOT_TOKEN"),
            telegram_chat_id=_str("BOT5_TELEGRAM_CHAT_ID"),
            telegram_commands_enabled=_bool("BOT5_TELEGRAM_COMMANDS_ENABLED"),
            state_file=_str("BOT5_STATE_FILE", "bot5_martingale/state.json"),
            state_reset_on_stop=_bool("BOT5_STATE_RESET_ON_STOP", False),
            ws_enabled=_bool("BOT5_WS_ENABLED", True),
            ws_url=_str("BOT5_WS_URL", "wss://ws-subscriptions-clob.polymarket.com/ws/market"),
            ws_stale_seconds=_float("BOT5_WS_STALE_SECONDS", 3.0),
        )

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.mode is BotMode.LIVE:
            if not self.private_key:
                errors.append("PRIVATE_KEY required in live mode")
            if not self.wallet_address:
                errors.append("WALLET_ADDRESS required in live mode")
        if not self.ladder:
            errors.append("Ladder must contain at least one level")
        if self.max_level < 0 or self.max_level >= len(self.ladder):
            errors.append(f"max_level={self.max_level} out of range for ladder len={len(self.ladder)}")
        if self.max_entry_price <= 0 or self.max_entry_price >= 1.0:
            errors.append(f"max_entry_price={self.max_entry_price} must be in (0, 1)")
        if self.telegram_enabled and not self.telegram_bot_token:
            errors.append("TELEGRAM_BOT_TOKEN required when telegram enabled")
        return errors
