"""Telegram notifier – async-safe, non-blocking.

Sends operational messages to a Telegram chat.  Optionally listens for
simple commands (/status, /pnl, /streak, /stop_bot, /start_bot) when
command handling is enabled.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Any, Callable, Optional

import requests

from bot5_martingale.config import Bot5Config

logger = logging.getLogger(__name__)

TELEGRAM_API = "https://api.telegram.org"


class TelegramNotifier:
    """Fire-and-forget Telegram sender with background thread."""

    def __init__(self, cfg: Bot5Config):
        self.enabled = cfg.telegram_enabled and bool(cfg.telegram_bot_token)
        self.token = cfg.telegram_bot_token
        self.chat_id = cfg.telegram_chat_id
        self.commands_enabled = cfg.telegram_commands_enabled
        self._command_handlers: dict[str, Callable[[], str]] = {}
        self._send_queue: queue.Queue[str] = queue.Queue(maxsize=200)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._poll_thread: Optional[threading.Thread] = None
        self._last_update_id: int = 0

        if self.enabled:
            self._thread = threading.Thread(target=self._sender_loop, daemon=True, name="tg-sender")
            self._thread.start()
            if self.commands_enabled:
                self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True, name="tg-poller")
                self._poll_thread.start()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def send(self, text: str, parse_mode: str = "HTML") -> None:
        if not self.enabled:
            return
        try:
            self._send_queue.put_nowait(text)
        except queue.Full:
            logger.warning("Telegram send queue full, dropping message")

    def register_command(self, command: str, handler: Callable[[], str]) -> None:
        self._command_handlers[command.lstrip("/")] = handler

    def shutdown(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        if self._poll_thread and self._poll_thread.is_alive():
            self._poll_thread.join(timeout=5)

    # ------------------------------------------------------------------
    # Formatted messages
    # ------------------------------------------------------------------

    @staticmethod
    def _side_decorated(side: str) -> str:
        s = side.strip().lower()
        if s == "up":
            return "🟢⬆️ UP"
        if s == "down":
            return "🔴⬇️ DOWN"
        return side.upper()

    def notify_start(self, mode: str) -> None:
        self.send(f"<b>Bot5 started</b> – mode: <code>{mode}</code>")

    def notify_stop(self, reason: str = "manual") -> None:
        self.send(f"<b>Bot5 stopped</b> – {reason}")

    def notify_order(
        self, side: str, size: float, price: float,
        market: str, level: int, idx: int, early: bool = False,
    ) -> None:
        side_label = self._side_decorated(side)
        prefix = "⚡ EARLY " if early else ""
        self.send(
            f"[#{idx}] <b>{prefix}ORDER</b> {side_label} ${size:.2f} @ {price:.4f}\n"
            f"Market: <code>{market[:60]}</code>\n"
            f"Level: {level}"
        )

    def notify_fill(self, side: str, shares: float, price: float, order_id: str, idx: int) -> None:
        side_label = self._side_decorated(side)
        self.send(
            f"✅ [#{idx}] <b>FILL</b> {side_label} {shares:.2f} shares @ {price:.4f}\n"
            f"Order: <code>{order_id or 'n/a'}</code>"
        )

    def notify_result(
        self, won: bool, pnl: float, level: int,
        cum_pnl: float, total_trades: int, idx: int, side: str,
    ) -> None:
        dot = "🟢" if won else "🔴"
        label = "WIN" if won else "LOSS"
        side_label = self._side_decorated(side)
        self.send(
            f"{dot} [#{idx}] <b>{label}</b> {side_label} | Level: {level} | PnL: {pnl:+.2f}\n"
            f"Cum PnL: {cum_pnl:+.2f} | Trades: {total_trades}"
        )

    def notify_stopout(self, level: int, cum_pnl: float) -> None:
        self.send(f"<b>STOP-OUT</b> at level {level} | Cum PnL: {cum_pnl:+.2f}")

    def notify_error(self, msg: str) -> None:
        self.send(f"<b>ERROR</b>: {msg}")

    def notify_daily_summary(
        self, date: str, trades: int, wins: int,
        losses: int, pnl: float, cum_pnl: float,
    ) -> None:
        self.send(
            f"<b>Daily summary – {date}</b>\n"
            f"Trades: {trades} | W: {wins} | L: {losses}\n"
            f"Day PnL: {pnl:+.2f} | Total PnL: {cum_pnl:+.2f}"
        )

    # ------------------------------------------------------------------
    # Background workers
    # ------------------------------------------------------------------

    def _sender_loop(self) -> None:
        while not self._stop.is_set():
            try:
                text = self._send_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            self._send_raw(text)

    def _send_raw(self, text: str) -> None:
        url = f"{TELEGRAM_API}/bot{self.token}/sendMessage"
        try:
            resp = requests.post(
                url,
                json={"chat_id": self.chat_id, "text": text, "parse_mode": "HTML"},
                timeout=10,
            )
            if resp.status_code != 200:
                logger.debug("Telegram send failed: %s", resp.text[:200])
        except Exception as exc:
            logger.debug("Telegram send error: %s", exc)

    def _poll_loop(self) -> None:
        """Long-poll for incoming commands."""
        while not self._stop.is_set():
            try:
                url = f"{TELEGRAM_API}/bot{self.token}/getUpdates"
                resp = requests.get(
                    url,
                    params={"offset": self._last_update_id + 1, "timeout": 15},
                    timeout=20,
                )
                if resp.status_code != 200:
                    time.sleep(5)
                    continue
                data = resp.json()
                for update in data.get("result", []):
                    self._last_update_id = update["update_id"]
                    msg = update.get("message", {})
                    text = msg.get("text", "").strip()
                    chat = str(msg.get("chat", {}).get("id", ""))
                    if chat != str(self.chat_id):
                        continue
                    if text.startswith("/"):
                        cmd = text.lstrip("/").split()[0].lower()
                        handler = self._command_handlers.get(cmd)
                        if handler:
                            try:
                                reply = handler()
                                self._send_raw(reply)
                            except Exception as exc:
                                self._send_raw(f"Command error: {exc}")
                        else:
                            self._send_raw(f"Unknown command: /{cmd}")
            except Exception as exc:
                logger.debug("Telegram poll error: %s", exc)
                time.sleep(5)


class NullNotifier(TelegramNotifier):
    """No-op notifier for when Telegram is disabled."""

    def __init__(self) -> None:
        self.enabled = False
        self.commands_enabled = False
        self._command_handlers = {}

    def send(self, text: str, parse_mode: str = "HTML") -> None:
        pass

    def register_command(self, command: str, handler: Callable[[], str]) -> None:
        pass

    def shutdown(self) -> None:
        pass


def create_notifier(cfg: Bot5Config) -> TelegramNotifier:
    if cfg.telegram_enabled and cfg.telegram_bot_token and cfg.telegram_chat_id:
        return TelegramNotifier(cfg)
    return NullNotifier()
