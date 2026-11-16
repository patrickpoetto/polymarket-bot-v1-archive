"""Telegram notifications for Bot6 MM."""
from __future__ import annotations

import logging
from typing import Optional

import requests

from bot6_mm.config import Bot6Config
from bot6_mm.models import MarketSession  # noqa: F401 – used in type hints

logger = logging.getLogger(__name__)


class Notifier:
    def __init__(self, cfg: Bot6Config):
        self._cfg = cfg
        self._enabled = cfg.telegram_enabled and bool(cfg.telegram_token) and bool(cfg.telegram_chat_id)

    def _send(self, text: str) -> None:
        if not self._enabled:
            return
        try:
            requests.post(
                f"https://api.telegram.org/bot{self._cfg.telegram_token}/sendMessage",
                json={"chat_id": self._cfg.telegram_chat_id, "text": text, "parse_mode": "HTML"},
                timeout=5,
            )
        except Exception as exc:
            logger.warning("Telegram send failed: %s", exc)

    def notify_start(self, balance: float, paper: bool) -> None:
        mode = "📄 PAPER" if paper else "🟢 LIVE"
        self._send(
            f"🤖 Bot6 MM avviato {mode}\n"
            f"Wallet: <b>${balance:.2f}</b>\n"
            f"Spread: ±{self._cfg.mm_half_spread*100:.1f}c │ Size: {self._cfg.mm_order_size:.0f}sh │ MaxInv: {self._cfg.mm_max_inventory:.0f}sh"
        )

    def notify_stop(self, balance: float, cum_pnl: float) -> None:
        self._send(
            f"🛑 Bot6 MM fermato\n"
            f"PnL cumulato: <b>${cum_pnl:+.2f}</b>\n"
            f"Wallet: <b>${balance:.2f}</b>"
        )

    def notify_session_start(self, session: MarketSession) -> None:
        mkt = session.market_info
        self._send(
            f"📊 [#{session.market_index}] MM start\n"
            f"{mkt.question}"
        )

    def notify_session_close(
        self,
        session: MarketSession,
        session_pnl: float,
        cum_pnl: float,
        balance: float,
    ) -> None:
        inv = session
        emoji = "✅" if session_pnl >= 0 else "🔴"
        self._send(
            f"{emoji} [#{session.market_index}] MM close\n"
            f"Realized round-trips: <b>${session.total_realized_pnl:+.4f}</b>\n"
            f"Session PnL (est.):   <b>${session_pnl:+.4f}</b>\n"
            f"Cum PnL:              <b>${cum_pnl:+.4f}</b>\n"
            f"Wallet: ${balance:.2f}\n"
            f"Fills: UP {session.up_inv.total_buys}B/{session.up_inv.total_sells}S │ "
            f"DN {session.dn_inv.total_buys}B/{session.dn_inv.total_sells}S"
        )
