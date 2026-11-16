"""Env-driven configuration for Bot6 Market Maker."""
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


def _wallet_private_key() -> str:
    return _str("POLYMARKET_PRIVATE_KEY") or _str("PRIVATE_KEY")


def _wallet_address() -> str:
    return _str("POLYMARKET_WALLET_ADDRESS") or _str("WALLET_ADDRESS")


def _bot6_telegram_token() -> str:
    return _str("BOT6_TELEGRAM_BOT_TOKEN") or _str("TELEGRAM_TOKEN")


def _bot6_telegram_chat_id() -> str:
    return _str("BOT6_TELEGRAM_CHAT_ID") or _str("TELEGRAM_CHAT_ID")


@dataclass(frozen=True)
class Bot6Config:
    # ── Wallet / Exchange ───────────────────────────────────────────────
    private_key: str = ""
    wallet_address: str = ""
    chain_id: int = 137
    clob_host: str = "https://clob.polymarket.com"

    # ── Mode ────────────────────────────────────────────────────────────
    paper_mode: bool = True
    paper_initial_balance: float = 1_000.0

    # ── Market ──────────────────────────────────────────────────────────
    market_interval_seconds: int = 300
    market_url_pattern: str = "btc-updown-5m-{ts}"
    # If True, after discovery use find_next(current) (skips one 5m window — avoid after roll)
    mm_prefer_next_window: bool = False

    # ── Market Making: quoting ──────────────────────────────────────────
    # We place one GTC buy and one GTC sell on each side (UP & DOWN).
    # Buy at (mid - half_spread), sell at (mid + half_spread).
    mm_half_spread: float = 0.01         # 1c each side → 2c round-trip
    mm_order_size: float = 10.0          # shares per limit order
    mm_max_inventory: float = 50.0       # max net long shares per side
    # Dynamic target allocation (asymmetric inventory management)
    mm_target_total_inventory: float = 60.0
    mm_target_deadband: float = 5.0
    mm_target_ema_alpha: float = 0.20
    mm_max_skew_normal: float = 0.70
    mm_max_skew_endgame: float = 0.85
    # When winner-hold is ON for one side, cap the *opponent* target (fewer BUYs on loser).
    mm_hold_opponent_max_target: float = 12.0
    # Skewed SELL sizing: unload overweight / disfavored side faster than mm_order_size clips.
    mm_overweight_sell_mult: float = 2.25   # inv > target+deadband → min(inv, cap, order_size*mult)
    mm_disfavored_mid_sell_mult: float = 1.75  # weak mid vs other side → boost sell mult (max with overweight)
    mm_skew_sell_cap: float = 35.0          # max shares per GTC sell when skewing
    mm_endgame_sell_urgency_mult: float = 1.3  # extra mult in last mm_endgame_seconds if skew applies
    mm_endgame_derisk_trim_mult: float = 2.0   # FAK derisk trim ≥ order_size * this (capped by skew cap)
    # Endgame loser BUY guard: if side mid is behind by this gap, stop BUYs on that side.
    mm_endgame_loser_stop_buy_gap: float = 0.08
    # Inventory-aware price skew (continuous, not hard stops):
    # positive imbalance (inv > target) -> BUY less aggressive and SELL more aggressive.
    mm_inventory_skew_step: float = 0.003
    mm_inventory_skew_max: float = 0.03
    # In endgame, if a side is clearly loser, quote only SELL (no BUY) on that side.
    mm_endgame_loser_only_unwind: bool = True
    # If opposite mid is ahead by this gap, switch this side to unwind pricing.
    mm_loser_unwind_gap: float = 0.10
    # In unwind mode, quote SELL near current market instead of avg+spread.
    mm_loser_unwind_sell_edge: float = 0.003
    mm_requote_threshold: float = 0.01   # reprice when mid moves by this much
    mm_requote_interval: float = 1.0     # minimum seconds between requotes
    mm_min_price: float = 0.03           # don't quote below this (too illiquid)
    mm_max_price: float = 0.97           # don't quote above this
    # Before endgame, if a side is weak (< threshold), cap its inventory lower.
    mm_weak_side_threshold: float = 0.40
    mm_weak_side_max_inventory: float = 20.0
    mm_weak_side_relative_gap: float = 0.10

    # Continuous mid-based inventory mgmt (no market skip — matches sim --inv-mgmt)
    mm_inv_mgmt_enabled: bool = True
    mm_inv_neutral_gap: float = 0.02
    mm_inv_stop_buy_gap: float = 0.02
    mm_inv_cap_slope: float = 80.0
    mm_inv_taker_trim_gap: float = 0.99   # ≥0.99 = off; else FAK-trim disfavored inv at bid
    mm_inv_neutral_max_cap: float = 1.0   # cap per side when |gap| < neutral (limits 50/50 buildup)
    mm_inv_trim_interval: float = 2.0

    # ── Exit / risk ─────────────────────────────────────────────────────
    stop_quoting_seconds: float = 30.0   # cancel all GTC orders N secs before end (~10% of 5m)
    exit_sell_floor: float = 0.01        # FAK sell floor for loser at end

    # Regime filter (adverse selection / winner retention)
    # Pre-endgame: optional global stop-buy threshold (0 disables).
    mm_stop_buying_below: float = 0.0
    # Endgame: in the last N seconds, stop BUYs when side mid is below this.
    mm_endgame_seconds: float = 40.0
    mm_endgame_stop_buy_below: float = 0.20
    # Endgame de-risk: if divergence is clear, trim loser inventory down to this cap.
    mm_endgame_loser_max_inventory: float = 20.0
    mm_endgame_divergence_threshold: float = 0.25
    mm_endgame_derisk_min_mid: float = 0.25
    mm_endgame_derisk_interval: float = 2.0
    # Winner: stop new SELLs only after mid stays ≥ this for confirm_seconds (not a single tick).
    mm_stop_selling_above: float = 0.70
    mm_winner_hold_confirm_seconds: float = 3.0  # seconds ≥ threshold before holding winner
    # After hold is on, keep it until mid falls below this (hysteresis vs brief 0.70 spikes).
    mm_winner_hold_reset_below: float = 0.60
    mm_winner_hold_buy_cooldown_seconds: float = 15.0

    # ── Order polling ───────────────────────────────────────────────────
    order_poll_interval: float = 0.5     # seconds between order status checks
    order_cancel_timeout: float = 30.0   # cancel GTC if unfilled for this long

    # ── Price feed ──────────────────────────────────────────────────────
    ws_url: str = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
    price_stale_seconds: float = 3.0
    rest_poll_interval: float = 0.5      # fallback REST poll interval

    # ── Main loop ───────────────────────────────────────────────────────
    tick_interval: float = 0.25

    # ── Telegram ────────────────────────────────────────────────────────
    telegram_token: str = ""
    telegram_chat_id: str = ""
    telegram_enabled: bool = True

    # ── Logging / state ─────────────────────────────────────────────────
    log_level: str = "INFO"
    log_file: str = "logs/bot6_mm/bot6.log"
    state_file: str = "bot6_mm/state.json"

    @classmethod
    def from_env(cls) -> "Bot6Config":
        return cls(
            private_key=_wallet_private_key(),
            wallet_address=_wallet_address(),
            chain_id=(
                _int("POLYMARKET_CHAIN_ID", 137)
                if _str("POLYMARKET_CHAIN_ID")
                else _int("CHAIN_ID", 137)
            ),
            clob_host=_str("POLYMARKET_CLOB_HOST", "https://clob.polymarket.com"),

            paper_mode=_bool("BOT6_PAPER_MODE", True),
            paper_initial_balance=_float("BOT6_PAPER_INITIAL_BALANCE", 1_000.0),

            market_interval_seconds=_int("BOT6_MARKET_INTERVAL_SECONDS", 300),
            market_url_pattern=_str("BOT6_MARKET_URL_PATTERN", "btc-updown-5m-{ts}"),
            mm_prefer_next_window=_bool("BOT6_MM_PREFER_NEXT_WINDOW", False),

            mm_half_spread=_float("BOT6_MM_HALF_SPREAD", 0.01),
            mm_order_size=_float("BOT6_MM_ORDER_SIZE", 10.0),
            mm_max_inventory=_float("BOT6_MM_MAX_INVENTORY", 50.0),
            mm_target_total_inventory=_float("BOT6_MM_TARGET_TOTAL_INVENTORY", 60.0),
            mm_target_deadband=_float("BOT6_MM_TARGET_DEADBAND", 5.0),
            mm_target_ema_alpha=_float("BOT6_MM_TARGET_EMA_ALPHA", 0.20),
            mm_max_skew_normal=_float("BOT6_MM_MAX_SKEW_NORMAL", 0.70),
            mm_max_skew_endgame=_float("BOT6_MM_MAX_SKEW_ENDGAME", 0.85),
            mm_hold_opponent_max_target=_float("BOT6_MM_HOLD_OPPONENT_MAX_TARGET", 12.0),
            mm_overweight_sell_mult=_float("BOT6_MM_OVERWEIGHT_SELL_MULT", 2.25),
            mm_disfavored_mid_sell_mult=_float("BOT6_MM_DISFAVORED_MID_SELL_MULT", 1.75),
            mm_skew_sell_cap=_float("BOT6_MM_SKEW_SELL_CAP", 35.0),
            mm_endgame_sell_urgency_mult=_float("BOT6_MM_ENDGAME_SELL_URGENCY_MULT", 1.3),
            mm_endgame_derisk_trim_mult=_float("BOT6_MM_ENDGAME_DERISK_TRIM_MULT", 2.0),
            mm_endgame_loser_stop_buy_gap=_float("BOT6_MM_ENDGAME_LOSER_STOP_BUY_GAP", 0.08),
            mm_inventory_skew_step=_float("BOT6_MM_INVENTORY_SKEW_STEP", 0.003),
            mm_inventory_skew_max=_float("BOT6_MM_INVENTORY_SKEW_MAX", 0.03),
            mm_endgame_loser_only_unwind=_bool("BOT6_MM_ENDGAME_LOSER_ONLY_UNWIND", True),
            mm_loser_unwind_gap=_float("BOT6_MM_LOSER_UNWIND_GAP", 0.10),
            mm_loser_unwind_sell_edge=_float("BOT6_MM_LOSER_UNWIND_SELL_EDGE", 0.003),
            mm_requote_threshold=_float("BOT6_MM_REQUOTE_THRESHOLD", 0.01),
            mm_requote_interval=_float("BOT6_MM_REQUOTE_INTERVAL", 1.0),
            mm_min_price=_float("BOT6_MM_MIN_PRICE", 0.03),
            mm_max_price=_float("BOT6_MM_MAX_PRICE", 0.97),
            mm_weak_side_threshold=_float("BOT6_MM_WEAK_SIDE_THRESHOLD", 0.40),
            mm_weak_side_max_inventory=_float("BOT6_MM_WEAK_SIDE_MAX_INVENTORY", 20.0),
            mm_weak_side_relative_gap=_float("BOT6_MM_WEAK_SIDE_RELATIVE_GAP", 0.10),

            mm_inv_mgmt_enabled=_bool("BOT6_MM_INV_MGMT_ENABLED", True),
            mm_inv_neutral_gap=_float("BOT6_MM_INV_NEUTRAL_GAP", 0.02),
            mm_inv_stop_buy_gap=_float("BOT6_MM_INV_STOP_BUY_GAP", 0.02),
            mm_inv_cap_slope=_float("BOT6_MM_INV_CAP_SLOPE", 80.0),
            mm_inv_taker_trim_gap=_float("BOT6_MM_INV_TAKER_TRIM_GAP", 0.99),
            mm_inv_neutral_max_cap=_float("BOT6_MM_INV_NEUTRAL_MAX_CAP", 1.0),
            mm_inv_trim_interval=_float("BOT6_MM_INV_TRIM_INTERVAL", 2.0),

            stop_quoting_seconds=_float("BOT6_STOP_QUOTING_SECONDS", 30.0),
            exit_sell_floor=_float("BOT6_EXIT_SELL_FLOOR", 0.01),

            mm_stop_buying_below=_float("BOT6_MM_STOP_BUYING_BELOW", 0.0),
            mm_endgame_seconds=_float("BOT6_MM_ENDGAME_SECONDS", 40.0),
            mm_endgame_stop_buy_below=_float("BOT6_MM_ENDGAME_STOP_BUY_BELOW", 0.20),
            mm_endgame_loser_max_inventory=_float("BOT6_MM_ENDGAME_LOSER_MAX_INVENTORY", 20.0),
            mm_endgame_divergence_threshold=_float("BOT6_MM_ENDGAME_DIVERGENCE_THRESHOLD", 0.25),
            mm_endgame_derisk_min_mid=_float("BOT6_MM_ENDGAME_DERISK_MIN_MID", 0.25),
            mm_endgame_derisk_interval=_float("BOT6_MM_ENDGAME_DERISK_INTERVAL", 2.0),
            mm_stop_selling_above=_float("BOT6_MM_STOP_SELLING_ABOVE", 0.70),
            mm_winner_hold_confirm_seconds=_float("BOT6_MM_WINNER_HOLD_CONFIRM_SECONDS", 3.0),
            mm_winner_hold_reset_below=_float("BOT6_MM_WINNER_HOLD_RESET_BELOW", 0.60),
            mm_winner_hold_buy_cooldown_seconds=_float("BOT6_MM_WINNER_HOLD_BUY_COOLDOWN_SECONDS", 15.0),

            order_poll_interval=_float("BOT6_ORDER_POLL_INTERVAL", 0.5),
            order_cancel_timeout=_float("BOT6_ORDER_CANCEL_TIMEOUT", 30.0),

            ws_url=_str("BOT6_WS_URL", "wss://ws-subscriptions-clob.polymarket.com/ws/market"),
            price_stale_seconds=_float("BOT6_PRICE_STALE_SECONDS", 3.0),
            rest_poll_interval=_float("BOT6_REST_POLL_INTERVAL", 0.5),

            tick_interval=_float("BOT6_TICK_INTERVAL", 0.25),

            telegram_token=_bot6_telegram_token(),
            telegram_chat_id=_bot6_telegram_chat_id(),
            telegram_enabled=_bool("BOT6_TELEGRAM_ENABLED", True),

            log_level=_str("BOT6_LOG_LEVEL", "INFO"),
            log_file=_str("BOT6_LOG_FILE", "logs/bot6_mm/bot6.log"),
            state_file=_str("BOT6_STATE_FILE", "bot6_mm/state.json"),
        )
