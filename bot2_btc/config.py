"""Env-driven configuration for Bot2 BTC."""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

_TRUE = frozenset({"1", "true", "yes", "on"})


def _bool(key: str, default: bool = False) -> bool:
    v = os.getenv(key, "").strip().lower()
    return (v in _TRUE) if v else default


def _float(key: str, default: float) -> float:
    v = os.getenv(key, "").strip()
    return float(v) if v else default


def _int(key: str, default: int) -> int:
    v = os.getenv(key, "").strip()
    return int(v) if v else default


def _str(key: str, default: str = "") -> str:
    return os.getenv(key, default).strip()


@dataclass(frozen=True)
class Bot2Config:
    # ── Wallet / Exchange ───────────────────────────────────────────────
    private_key: str = ""
    wallet_address: str = ""
    chain_id: int = 137
    clob_host: str = "https://clob.polymarket.com"

    # ── Mode ────────────────────────────────────────────────────────────
    paper_mode: bool = True
    paper_initial_balance: float = 10_000.0

    # ── Market ──────────────────────────────────────────────────────────
    market_interval_seconds: int = 900
    market_url_pattern: str = "btc-updown-15m-{ts}"

    # ── Entry (straddle) ────────────────────────────────────────────────
    entry_shares: float = 100.0        # shares to buy on EACH side
    max_entry_price: float = 0.51      # per-share limit for initial buy

    # ── Order execution ─────────────────────────────────────────────────
    order_type: str = "GTC"            # BUY (pre-pos; future intramarket buys)
    order_type_sell: str = "FAK"       # SELL (flatten / future trims)
    # FAK limit floor when simulating market sells (abort, future exits)
    sell_price_floor: float = 0.01
    order_max_retries: int = 3
    order_timeout_seconds: float = 12.0
    order_poll_interval: float = 0.5
    # If a sell fails with "not enough balance/allowance", retry every N seconds
    # for up to M seconds (on-chain tx confirmation delay).
    sell_balance_retry_seconds: float = 2.0
    sell_balance_max_wait_seconds: float = 300.0

    # ── Price feed ──────────────────────────────────────────────────────
    ws_url: str = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
    ws_enabled: bool = True
    ws_stale_seconds: float = 3.0
    rest_poll_interval: float = 0.25   # fallback REST polling interval

    # ── Main loop ───────────────────────────────────────────────────────
    tick_interval: float = 0.1         # main loop tick (seconds)

    # ── Risk ────────────────────────────────────────────────────────────
    max_daily_loss_usd: float = 1_000.0

    # ── Telegram ────────────────────────────────────────────────────────
    telegram_enabled: bool = False
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    telegram_commands_enabled: bool = True

    # ── Logging ─────────────────────────────────────────────────────────
    log_dir: str = "logs/bot2_btc"
    log_level: str = "INFO"
    log_per_run_file: bool = True

    # ── State ───────────────────────────────────────────────────────────
    state_file: str = "bot2_btc/state.json"
    state_reset_on_start: bool = True

    # ── Pre-position timing ─────────────────────────────────────────────
    pre_position_seconds_before_start: int = 120  # start placing entry orders this far ahead

    # ── Pre-position price buffer ────────────────────────────────────────
    # Added to the best ask when setting the limit for pre-pos BUY orders.
    # A small positive value (e.g. 0.02) ensures the order crosses the spread
    # and matches immediately rather than resting live on the book.
    # Total limit = min(ask + buffer, max_entry_price + buffer).
    pre_pos_price_buffer: float = 0.02

    # ── Pre-position safety abort ────────────────────────────────────────
    # If the straddle is still incomplete this many seconds before market open,
    # sell any partial fill and skip the market entirely.
    prepos_abort_seconds: float = 1.0

    # ── Pre-position retry guard ─────────────────────────────────────────
    # _complete_prepos_with_retry will NOT attempt a retry if fewer than
    # this many seconds remain before market open (don't enter at last second).
    prepos_min_seconds_before_start: int = 20
    # When the ask is above cap, wait this many seconds before re-checking.
    prepos_cap_wait_seconds: int = 10
    # When stuck above cap, log a status update at most every N seconds.
    prepos_cap_log_interval: int = 60

    # ── Intramarket inventory strategy (hybrid strong/weak) ───────────────
    # Env: BOT2_INV_* — 0 on numeric caps often means “no limit” (see bot logic).
    inv_enabled: bool = True
    # First inv_phase_split of the window = “early” (sell strong bias).
    inv_phase_split: float = 0.5
    # In early phase, sell the high-bid leg only if its bid ≥ this.
    inv_early_strong_min_bid: float = 0.62
    # If |up_bid − down_bid| ≥ this, treat as directional (even in early window):
    # switch to late-style behaviour (sell weak).
    inv_directional_spread: float = 0.45
    # Fraction of current leg to sell per trim (0–1).
    inv_trim_fraction: float = 0.35
    # Max shares per single sell/buy; 0 = uncapped (only trim fraction / deficit).
    inv_max_shares_per_trade: float = 0.0
    # Cumulative intramarket share volume (buys+sells); 0 = uncapped.
    inv_max_turnover_shares: float = 0.0
    # Min seconds between orders on the same leg.
    inv_cooldown_seconds: float = 5.0
    # Sell limit ≈ bid − slippage (floored at sell_price_floor).
    inv_sell_slippage: float = 0.02
    # Min USDC notional to attempt a sell (Polymarket ~$1 minimum).
    inv_min_sell_notional_usd: float = 1.0
    # FAK sell retries per decision.
    inv_sell_max_retries: int = 2
    # Re-entry: |up_bid−down_bid| must be below this (spread tightened).
    inv_reentry_spread_max: float = 0.28
    # Re-entry: both bids must be ≥ this (liquidity / recovery on both legs).
    inv_reentry_both_min_bid: float = 0.12
    # Re-buy up to this fraction of the deficit vs strat_ref_*.
    inv_reentry_fraction: float = 0.55
    # Cap re-buy size per order; 0 = uncapped (still limited by deficit).
    inv_max_rebuy_shares_per_trade: float = 0.0
    # Throttled MTM log interval when strategy enabled (seconds).
    inv_mtm_log_interval: float = 60.0
    # Endgame safety: in strongly directional tape, force-sell weak leg
    # in the last N seconds even if cooldown/turnover budget would block it.
    inv_force_flatten_seconds: float = 180.0

    @classmethod
    def from_env(cls) -> "Bot2Config":
        return cls(
            private_key=_str("PRIVATE_KEY"),
            wallet_address=_str("WALLET_ADDRESS"),
            chain_id=_int("CHAIN_ID", 137),
            clob_host=_str("BOT2_CLOB_HOST", "https://clob.polymarket.com"),
            paper_mode=_bool("BOT2_PAPER_MODE", True),
            paper_initial_balance=_float("BOT2_PAPER_INITIAL_BALANCE", 10_000.0),
            market_interval_seconds=_int("BOT2_MARKET_INTERVAL_SECONDS", 900),
            market_url_pattern=_str("BOT2_MARKET_URL_PATTERN", "btc-updown-15m-{ts}"),
            entry_shares=_float("BOT2_ENTRY_SHARES", 100.0),
            max_entry_price=_float("BOT2_MAX_ENTRY_PRICE", 0.51),
            order_type=_str("BOT2_ORDER_TYPE", "GTC"),
            order_type_sell=_str("BOT2_ORDER_TYPE_SELL", "FAK"),
            sell_price_floor=_float("BOT2_SELL_PRICE_FLOOR", 0.01),
            order_max_retries=_int("BOT2_ORDER_MAX_RETRIES", 3),
            order_timeout_seconds=_float("BOT2_ORDER_TIMEOUT_SECONDS", 12.0),
            order_poll_interval=_float("BOT2_ORDER_POLL_INTERVAL", 0.5),
            sell_balance_retry_seconds=_float("BOT2_SELL_BALANCE_RETRY_SECONDS", 2.0),
            sell_balance_max_wait_seconds=_float("BOT2_SELL_BALANCE_MAX_WAIT_SECONDS", 300.0),
            prepos_cap_wait_seconds=_int("BOT2_PREPOS_CAP_WAIT_SECONDS", 10),
            prepos_cap_log_interval=_int("BOT2_PREPOS_CAP_LOG_INTERVAL", 60),
            ws_url=_str("BOT2_WS_URL", "wss://ws-subscriptions-clob.polymarket.com/ws/market"),
            ws_enabled=_bool("BOT2_WS_ENABLED", True),
            ws_stale_seconds=_float("BOT2_WS_STALE_SECONDS", 3.0),
            rest_poll_interval=_float("BOT2_REST_POLL_INTERVAL", 0.25),
            tick_interval=_float("BOT2_TICK_INTERVAL", 0.1),
            max_daily_loss_usd=_float("BOT2_MAX_DAILY_LOSS_USD", 1_000.0),
            telegram_enabled=_bool("BOT2_TELEGRAM_ENABLED"),
            telegram_bot_token=_str("BOT2_TELEGRAM_BOT_TOKEN"),
            telegram_chat_id=_str("BOT2_TELEGRAM_CHAT_ID"),
            telegram_commands_enabled=_bool("BOT2_TELEGRAM_COMMANDS_ENABLED", True),
            log_dir=_str("BOT2_LOG_DIR", "logs/bot2_btc"),
            log_level=_str("BOT2_LOG_LEVEL", "INFO"),
            log_per_run_file=_bool("BOT2_LOG_PER_RUN_FILE", True),
            state_file=_str("BOT2_STATE_FILE", "bot2_btc/state.json"),
            state_reset_on_start=_bool("BOT2_STATE_RESET_ON_START", True),
            pre_position_seconds_before_start=_int("BOT2_PRE_POSITION_SECONDS", 120),
            pre_pos_price_buffer=_float("BOT2_PRE_POS_PRICE_BUFFER", 0.02),
            prepos_min_seconds_before_start=_int("BOT2_PREPOS_MIN_SECONDS_BEFORE_START", 20),
            prepos_abort_seconds=_float("BOT2_PREPOS_ABORT_SECONDS", 1.0),
            inv_enabled=_bool("BOT2_INV_ENABLED", True),
            inv_phase_split=_float("BOT2_INV_PHASE_SPLIT", 0.5),
            inv_early_strong_min_bid=_float("BOT2_INV_EARLY_STRONG_MIN_BID", 0.62),
            inv_directional_spread=_float("BOT2_INV_DIRECTIONAL_SPREAD", 0.45),
            inv_trim_fraction=_float("BOT2_INV_TRIM_FRACTION", 0.35),
            inv_max_shares_per_trade=_float("BOT2_INV_MAX_SHARES_PER_TRADE", 0.0),
            inv_max_turnover_shares=_float("BOT2_INV_MAX_TURNOVER_SHARES", 0.0),
            inv_cooldown_seconds=_float("BOT2_INV_COOLDOWN_SECONDS", 5.0),
            inv_sell_slippage=_float("BOT2_INV_SELL_SLIPPAGE", 0.02),
            inv_min_sell_notional_usd=_float("BOT2_INV_MIN_SELL_NOTIONAL_USD", 1.0),
            inv_sell_max_retries=_int("BOT2_INV_SELL_MAX_RETRIES", 2),
            inv_reentry_spread_max=_float("BOT2_INV_REENTRY_SPREAD_MAX", 0.28),
            inv_reentry_both_min_bid=_float("BOT2_INV_REENTRY_BOTH_MIN_BID", 0.12),
            inv_reentry_fraction=_float("BOT2_INV_REENTRY_FRACTION", 0.55),
            inv_max_rebuy_shares_per_trade=_float("BOT2_INV_MAX_REBUY_SHARES_PER_TRADE", 0.0),
            inv_mtm_log_interval=_float("BOT2_INV_MTM_LOG_INTERVAL", 60.0),
            inv_force_flatten_seconds=_float("BOT2_INV_FORCE_FLATTEN_SECONDS", 180.0),
        )

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.paper_mode:
            if not self.private_key:
                errors.append("PRIVATE_KEY is required in live mode")
            if not self.wallet_address:
                errors.append("WALLET_ADDRESS is required in live mode")
        if self.entry_shares <= 0:
            errors.append("BOT2_ENTRY_SHARES must be > 0")
        if not (0 < self.max_entry_price < 1):
            errors.append("BOT2_MAX_ENTRY_PRICE must be in (0, 1)")
        if self.telegram_enabled and not self.telegram_bot_token:
            errors.append("BOT2_TELEGRAM_BOT_TOKEN required when Telegram is enabled")
        if not (0.0 < self.inv_phase_split <= 1.0):
            errors.append("BOT2_INV_PHASE_SPLIT must be in (0, 1]")
        if not (0.0 < self.inv_trim_fraction <= 1.0):
            errors.append("BOT2_INV_TRIM_FRACTION must be in (0, 1]")
        if not (0.0 <= self.inv_reentry_fraction <= 1.0):
            errors.append("BOT2_INV_REENTRY_FRACTION must be in [0, 1]")
        return errors
