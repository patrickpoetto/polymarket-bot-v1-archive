"""Telegram notifier for Bot2 BTC."""
from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Callable, Optional

import requests

from bot2_btc.config import Bot2Config

logger = logging.getLogger(__name__)

TELEGRAM_API = "https://api.telegram.org"


class TelegramNotifier:
    """Non-blocking Telegram sender with a background worker thread."""

    def __init__(self, cfg: Bot2Config):
        self.enabled = cfg.telegram_enabled and bool(cfg.telegram_bot_token)
        self._token = cfg.telegram_bot_token
        self._chat_id = cfg.telegram_chat_id
        self._commands_enabled = cfg.telegram_commands_enabled
        self._command_handlers: dict[str, Callable[[], str]] = {}
        self._queue: queue.Queue[str] = queue.Queue(maxsize=200)
        self._stop = threading.Event()
        self._last_update_id: int = 0
        self._thread: Optional[threading.Thread] = None
        self._poll_thread: Optional[threading.Thread] = None

        if self.enabled:
            self._thread = threading.Thread(target=self._sender_loop, daemon=True, name="bot2-tg-send")
            self._thread.start()
            if self._commands_enabled:
                self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True, name="bot2-tg-poll")
                self._poll_thread.start()

    # ── Public API ───────────────────────────────────────────────────────────

    def send(self, text: str) -> None:
        if not self.enabled:
            return
        try:
            self._queue.put_nowait(text)
        except queue.Full:
            logger.warning("Telegram queue full – dropping message")

    def register_command(self, command: str, handler: Callable[[], str]) -> None:
        self._command_handlers[command.lstrip("/")] = handler

    def shutdown(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        if self._poll_thread and self._poll_thread.is_alive():
            self._poll_thread.join(timeout=5)

    # ── Formatting helpers ───────────────────────────────────────────────────

    @staticmethod
    def _side_label(side: str) -> str:
        s = side.strip().lower()
        return "⬆️ UP" if s == "up" else "⬇️ DOWN"

    @staticmethod
    def _market_header(market_idx: int, end_ts: int) -> str:
        """Return e.g. '[#24] 20 mar – 6:00-6:15 ET'"""
        from datetime import datetime, timezone, timedelta
        ET = timezone(timedelta(hours=-5))
        end_dt = datetime.fromtimestamp(end_ts, tz=ET)
        start_dt = datetime.fromtimestamp(end_ts - 900, tz=ET)
        months = ["gen", "feb", "mar", "apr", "mag", "giu",
                  "lug", "ago", "set", "ott", "nov", "dic"]
        day = start_dt.day
        month = months[start_dt.month - 1]
        def _t(dt: datetime) -> str:
            h = dt.hour % 12 or 12
            return f"{h}:{dt.strftime('%M')}"
        return f"[#{market_idx}] {day} {month} – {_t(start_dt)}-{_t(end_dt)} ET"

    # ── Bot-level events ─────────────────────────────────────────────────────

    def notify_start(self, mode: str, balance_usd: float = 0.0,
                     positions_value: float = 0.0, n_positions: int = 0) -> None:
        portfolio = balance_usd + positions_value
        pos_str = f"\n📂 Open positions: <b>${positions_value:.2f}</b> ({n_positions})" if n_positions > 0 else ""
        self.send(
            f"<b>Bot2 BTC started</b> – mode: <code>{mode.upper()}</code>\n"
            f"💵 Cash: <b>${balance_usd:.2f}</b>{pos_str}\n"
            f"📊 Portfolio: <b>${portfolio:.2f}</b>"
        )

    def notify_stop(self, reason: str = "manual",
                    cash_usd: float = 0.0, positions_value: float = 0.0, n_positions: int = 0) -> None:
        portfolio = cash_usd + positions_value
        pos_str = f"\n📂 Open positions: <b>${positions_value:.2f}</b> ({n_positions})" if n_positions > 0 else ""
        bal_str = f"\n💵 Cash: <b>${cash_usd:.2f}</b>{pos_str}\n📊 Portfolio: <b>${portfolio:.2f}</b>" if cash_usd > 0 else ""
        self.send(f"<b>Bot2 BTC stopped</b> – {reason}{bal_str}")

    def notify_error(self, msg: str) -> None:
        self.send(f"⚠️ <b>ERROR</b>: {msg[:200]}")

    def notify_daily_summary(self, date: str, markets: int, pnl: float, cum_pnl: float) -> None:
        dot = "🟢" if pnl >= 0 else "🔴"
        self.send(
            f"{dot} <b>Daily summary – {date}</b>\n"
            f"Markets: {markets} | Day PnL: {pnl:+.2f} | Total: {cum_pnl:+.2f}"
        )

    # ── Market-level events ──────────────────────────────────────────────────

    def notify_pre_position(self, market_idx: int, end_ts: int,
                            up_price: float, down_price: float, shares: float) -> None:
        hdr = self._market_header(market_idx, end_ts)
        self.send(
            f"🎯 {hdr}\n"
            f"<b>PRE-POSITION</b>\n"
            f"⬆️ UP   {shares:.0f} sh @ {up_price:.4f}\n"
            f"⬇️ DOWN {shares:.0f} sh @ {down_price:.4f}\n"
            f"Cost: ~${shares * (up_price + down_price):.2f}"
        )

    def notify_position_filled(self, side: str, shares: float, price: float,
                               market_idx: int, end_ts: int) -> None:
        hdr = self._market_header(market_idx, end_ts)
        self.send(
            f"✅ {hdr}\n"
            f"<b>FILLED</b> {self._side_label(side)} – {shares:.2f} sh @ {price:.4f}"
        )

    def notify_exit_low(self, side: str, shares: float, price: float,
                        market_idx: int, end_ts: int) -> None:
        hdr = self._market_header(market_idx, end_ts)
        kept = "DOWN" if side.lower() == "up" else "UP"
        self.send(
            f"📉 {hdr}\n"
            f"<b>EXIT LOW</b> {self._side_label(side)}\n"
            f"Sold {shares:.2f} sh @ {price:.4f} | Kept: {kept}"
        )

    def notify_rebuy(self, side: str, shares: float, price: float,
                     cycle: int, market_idx: int, end_ts: int) -> None:
        hdr = self._market_header(market_idx, end_ts)
        self.send(
            f"🔄 {hdr}\n"
            f"<b>REBUY #{cycle}</b> {self._side_label(side)}\n"
            f"{shares:.2f} sh @ {price:.4f}"
        )

    def notify_rebuy_tp(self, side: str, shares: float, price: float,
                        pnl: float, market_idx: int, end_ts: int) -> None:
        hdr = self._market_header(market_idx, end_ts)
        self.send(
            f"💰 {hdr}\n"
            f"<b>REBUY TP</b> {self._side_label(side)}\n"
            f"Sold {shares:.2f} sh @ {price:.4f} | PnL: {pnl:+.2f}"
        )

    def notify_reversal(self, buy_side: str, buy_price: float,
                        sell_side: str, sell_price: float,
                        market_idx: int, end_ts: int) -> None:
        hdr = self._market_header(market_idx, end_ts)
        self.send(
            f"🔀 {hdr}\n"
            f"<b>REVERSAL</b>\n"
            f"BUY  {self._side_label(buy_side)} @ {buy_price:.4f}\n"
            f"SELL {self._side_label(sell_side)} @ {sell_price:.4f}"
        )

    def notify_session_close(self, market_idx: int, end_ts: int,
                             pnl: float, cum_pnl: float,
                             cash_usd: float = 0.0, positions_value: float = 0.0,
                             n_positions: int = 0) -> None:
        dot = "🟢" if pnl >= 0 else "🔴"
        hdr = self._market_header(market_idx, end_ts)
        portfolio = cash_usd + positions_value
        pos_str = f" | 📂 ${positions_value:.2f} ({n_positions})" if n_positions > 0 else ""
        bal_str = (f"\n💵 Cash: ${cash_usd:.2f}{pos_str} | 📊 Portfolio: ${portfolio:.2f}"
                   if cash_usd > 0 else "")
        self.send(
            f"{dot} {hdr}\n"
            f"<b>SESSION CLOSED</b> | PnL: {pnl:+.2f} | Cum: {cum_pnl:+.2f}{bal_str}"
        )

    # ── Background workers ───────────────────────────────────────────────────

    def _sender_loop(self) -> None:
        while not self._stop.is_set():
            try:
                text = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue
            self._send_raw(text)

    def _send_raw(self, text: str) -> None:
        url = f"{TELEGRAM_API}/bot{self._token}/sendMessage"
        try:
            resp = requests.post(
                url,
                json={"chat_id": self._chat_id, "text": text, "parse_mode": "HTML"},
                timeout=10,
            )
            if resp.status_code != 200:
                logger.debug("Telegram send failed: %s", resp.text[:200])
        except Exception as exc:
            logger.debug("Telegram send error: %s", exc)

    def _poll_loop(self) -> None:
        while not self._stop.is_set():
            try:
                url = f"{TELEGRAM_API}/bot{self._token}/getUpdates"
                resp = requests.get(
                    url,
                    params={"offset": self._last_update_id + 1, "timeout": 15},
                    timeout=20,
                )
                if resp.status_code != 200:
                    time.sleep(5)
                    continue
                for update in resp.json().get("result", []):
                    self._last_update_id = update["update_id"]
                    msg = update.get("message", {})
                    text = msg.get("text", "").strip()
                    chat = str(msg.get("chat", {}).get("id", ""))
                    if chat != str(self._chat_id):
                        continue
                    if not text:
                        continue
                    # Accept both "/stop" and plain "Stop" (case-insensitive).
                    cmd = text.lstrip("/").split()[0].lower()
                    handler = self._command_handlers.get(cmd)
                    if handler:
                        try:
                            self._send_raw(handler())
                        except Exception as exc:
                            self._send_raw(f"Command error: {exc}")
                    elif text.startswith("/"):
                        self._send_raw(f"Unknown command: /{cmd}")
            except Exception as exc:
                logger.debug("Telegram poll error: %s", exc)
                time.sleep(5)


class NullNotifier(TelegramNotifier):
    """No-op notifier when Telegram is disabled."""

    def __init__(self) -> None:  # type: ignore[override]
        self.enabled = False
        self._commands_enabled = False
        self._command_handlers = {}

    def send(self, text: str) -> None:
        pass

    def register_command(self, command: str, handler: Callable[[], str]) -> None:
        pass

    def shutdown(self) -> None:
        pass


def create_notifier(cfg: Bot2Config) -> TelegramNotifier:
    if cfg.telegram_enabled and cfg.telegram_bot_token and cfg.telegram_chat_id:
        return TelegramNotifier(cfg)
    return NullNotifier()
