"""Telegram notifications + circuit breaker for Bot1 BTC.

Sends per-trade reports comparing real execution vs backtest targets,
tracks rolling stats (P&L, drawdown, win rate, streaks), and auto-kills
the bot when thresholds are exceeded.
"""
from __future__ import annotations

import logging
import time
import threading
from dataclasses import dataclass, field
from typing import Optional

import requests

logger = logging.getLogger(__name__)


# ── Telegram transport ────────────────────────────────────────────────────────

class _TelegramSender:
    """Thin wrapper around the Telegram Bot API (sync, fire-and-forget)."""

    def __init__(self, token: str, chat_id: str, enabled: bool = True):
        self._token = token
        self._chat_id = chat_id
        self._enabled = enabled and bool(token) and bool(chat_id)

    @property
    def enabled(self) -> bool:
        return self._enabled

    def send(self, text: str) -> None:
        if not self._enabled:
            return
        try:
            requests.post(
                f"https://api.telegram.org/bot{self._token}/sendMessage",
                json={
                    "chat_id": self._chat_id,
                    "text": text,
                    "parse_mode": "HTML",
                },
                timeout=5,
            )
        except Exception as exc:
            logger.warning("Telegram send failed: %s", exc)


# ── Circuit breaker ───────────────────────────────────────────────────────────

@dataclass
class CircuitBreakerConfig:
    # Alert thresholds (yellow — Telegram warning, bot continues)
    alert_daily_loss: float = 150.0      # max daily loss $
    alert_drawdown: float = 263.0        # max cumulative drawdown $
    alert_streak: int = 36               # max consecutive losses
    alert_win_rate_floor: float = 0.25   # min win rate over rolling window

    # Kill thresholds (red — Telegram alert + bot stops)
    kill_daily_loss: float = 200.0
    kill_drawdown: float = 350.0
    kill_streak: int = 48
    kill_win_rate_floor: float = 0.20

    win_rate_window: int = 100           # rolling window for win rate check


@dataclass
class _TradeRecord:
    timestamp: float
    outcome: str
    entry_price: float
    exit_price: float
    reason: str       # tp, sl, market_resolved
    pnl: float
    shares: float
    is_win: bool


class CircuitBreaker:
    """Tracks live performance and compares against backtest expectations."""

    def __init__(self, cfg: CircuitBreakerConfig):
        self.cfg = cfg
        self._trades: list[_TradeRecord] = []
        self._daily_pnl: float = 0.0
        self._cum_pnl: float = 0.0
        self._peak_pnl: float = 0.0
        self._drawdown: float = 0.0
        self._current_streak: int = 0  # consecutive losses (resets on win)
        self._worst_streak: int = 0
        self._day_start: float = time.time()
        self._killed: bool = False
        self._kill_reason: str = ""
        self._n_wins: int = 0
        self._n_losses: int = 0
        self._n_entries: int = 0  # markets where we entered
        self._n_markets: int = 0  # total markets seen

    @property
    def killed(self) -> bool:
        return self._killed

    @property
    def kill_reason(self) -> str:
        return self._kill_reason

    def record_market(self) -> None:
        """Called once per market (entered or not) for stats."""
        self._n_markets += 1

    def record_trade(self, entry_price: float, exit_price: float,
                     reason: str, pnl: float, shares: float,
                     outcome: str = "") -> list[str]:
        """Record a completed trade. Returns list of alert/kill messages."""
        is_win = pnl > 0

        rec = _TradeRecord(
            timestamp=time.time(), outcome=outcome,
            entry_price=entry_price, exit_price=exit_price,
            reason=reason, pnl=pnl, shares=shares, is_win=is_win,
        )
        self._trades.append(rec)
        self._n_entries += 1

        # Reset daily if new day
        now = time.time()
        if now - self._day_start > 86400:
            self._daily_pnl = 0.0
            self._day_start = now

        self._daily_pnl += pnl
        self._cum_pnl += pnl
        if self._cum_pnl > self._peak_pnl:
            self._peak_pnl = self._cum_pnl
        self._drawdown = self._peak_pnl - self._cum_pnl

        if is_win:
            self._n_wins += 1
            self._current_streak = 0
        else:
            self._n_losses += 1
            self._current_streak += 1
            if self._current_streak > self._worst_streak:
                self._worst_streak = self._current_streak

        return self._check_thresholds()

    def _rolling_win_rate(self) -> Optional[float]:
        n = self.cfg.win_rate_window
        if len(self._trades) < n:
            return None
        recent = self._trades[-n:]
        wins = sum(1 for t in recent if t.is_win)
        return wins / n

    def _check_thresholds(self) -> list[str]:
        alerts: list[str] = []
        cfg = self.cfg

        daily_loss = -self._daily_pnl if self._daily_pnl < 0 else 0

        # Kill checks (highest priority)
        if daily_loss >= cfg.kill_daily_loss:
            self._killed = True
            self._kill_reason = f"Daily loss ${daily_loss:.2f} >= kill ${cfg.kill_daily_loss:.0f}"
            alerts.append(f"🔴 KILL: {self._kill_reason}")

        if self._drawdown >= cfg.kill_drawdown:
            self._killed = True
            self._kill_reason = f"Drawdown ${self._drawdown:.2f} >= kill ${cfg.kill_drawdown:.0f}"
            alerts.append(f"🔴 KILL: {self._kill_reason}")

        if self._current_streak >= cfg.kill_streak:
            self._killed = True
            self._kill_reason = f"Losing streak {self._current_streak} >= kill {cfg.kill_streak}"
            alerts.append(f"🔴 KILL: {self._kill_reason}")

        wr = self._rolling_win_rate()
        if wr is not None and wr <= cfg.kill_win_rate_floor:
            self._killed = True
            self._kill_reason = f"Win rate {wr:.1%} <= kill {cfg.kill_win_rate_floor:.0%} (last {cfg.win_rate_window})"
            alerts.append(f"🔴 KILL: {self._kill_reason}")

        if self._killed:
            return alerts

        # Alert checks (bot continues)
        if daily_loss >= cfg.alert_daily_loss:
            alerts.append(f"🟡 ALERT: Daily loss ${daily_loss:.2f} >= ${cfg.alert_daily_loss:.0f}")

        if self._drawdown >= cfg.alert_drawdown:
            alerts.append(f"🟡 ALERT: Drawdown ${self._drawdown:.2f} >= ${cfg.alert_drawdown:.0f}")

        if self._current_streak >= cfg.alert_streak:
            alerts.append(f"🟡 ALERT: Losing streak {self._current_streak} >= {cfg.alert_streak}")

        if wr is not None and wr <= cfg.alert_win_rate_floor:
            alerts.append(f"🟡 ALERT: Win rate {wr:.1%} <= {cfg.alert_win_rate_floor:.0%} (last {cfg.win_rate_window})")

        return alerts

    # ── Summary stats ─────────────────────────────────────────────────────

    @property
    def summary(self) -> dict:
        n = self._n_wins + self._n_losses
        wr = self._n_wins / n if n > 0 else 0
        avg_pnl = self._cum_pnl / n if n > 0 else 0
        return {
            "trades": n,
            "markets": self._n_markets,
            "wins": self._n_wins,
            "losses": self._n_losses,
            "win_rate": wr,
            "cum_pnl": self._cum_pnl,
            "daily_pnl": self._daily_pnl,
            "drawdown": self._drawdown,
            "peak_pnl": self._peak_pnl,
            "worst_streak": self._worst_streak,
            "current_streak": self._current_streak,
            "avg_pnl": avg_pnl,
        }


# ── Bot1 Notifier (combines Telegram + circuit breaker) ──────────────────────

class Bot1Notifier:
    """All-in-one notifier for Bot1: Telegram reports + circuit breaker."""

    def __init__(
        self,
        token: str,
        chat_id: str,
        enabled: bool = True,
        cb_config: Optional[CircuitBreakerConfig] = None,
        target_entry: float = 0.60,
        target_tp: float = 0.97,
        target_sl: float = 0.45,
    ):
        self._tg = _TelegramSender(token, chat_id, enabled)
        self.cb = CircuitBreaker(cb_config or CircuitBreakerConfig())
        self._target_entry = target_entry
        self._target_tp = target_tp
        self._target_sl = target_sl
        self._session_start = time.time()
        self._last_hourly = time.time()

    @property
    def killed(self) -> bool:
        return self.cb.killed

    @property
    def kill_reason(self) -> str:
        return self.cb.kill_reason

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def notify_start(self, capital: float, entry: float, tp: float, sl: float) -> None:
        self._target_entry = entry
        self._target_tp = tp
        self._target_sl = sl
        self._tg.send(
            f"🟢 <b>Bot1 BTC avviato</b>\n"
            f"Capitale: <b>${capital:.2f}</b>/trade\n"
            f"Entry: {entry} │ TP: {tp} │ SL: {sl}\n"
            f"Circuit breaker: kill a -${self.cb.cfg.kill_daily_loss:.0f}/day "
            f"o -${self.cb.cfg.kill_drawdown:.0f} DD"
        )

    def notify_stop(self, reason: str = "manual") -> None:
        s = self.cb.summary
        self._tg.send(
            f"🛑 <b>Bot1 BTC fermato</b> ({reason})\n"
            f"PnL: <b>${s['cum_pnl']:+.2f}</b> │ "
            f"Trades: {s['trades']} ({s['win_rate']:.0%} win)\n"
            f"DD: ${s['drawdown']:.2f} │ Streak: {s['worst_streak']}"
        )

    # ── Per-trade notifications ────────────────────────────────────────────

    def notify_entry(self, outcome: str, price: float, capital: float,
                     shares: float, is_recovery: bool = False) -> None:
        diff = price - self._target_entry
        tag = "🔄 RECOVERY" if is_recovery else "📥 ENTRY"
        arrow = "🔺" if diff > 0.005 else ("🔻" if diff < -0.005 else "✅")
        self._tg.send(
            f"{tag} <b>{outcome}</b> @ {price:.3f}\n"
            f"Target: {self._target_entry:.2f} │ Diff: {arrow} {diff:+.3f}\n"
            f"${capital:.2f} → {shares:.2f} shares"
        )

    def notify_exit(self, outcome: str, entry_price: float, exit_price: float,
                    reason: str, pnl: float, pnl_pct: float, shares: float) -> None:
        # Record in circuit breaker
        alerts = self.cb.record_trade(
            entry_price=entry_price, exit_price=exit_price,
            reason=reason, pnl=pnl, shares=shares, outcome=outcome,
        )

        s = self.cb.summary

        if reason == "take_profit":
            emoji = "🎯"
            diff = exit_price - self._target_tp
            target_label = f"Target TP: {self._target_tp:.2f}"
        elif reason == "stop_loss":
            emoji = "🛑"
            diff = exit_price - self._target_sl
            target_label = f"Target SL: {self._target_sl:.2f}"
        else:
            emoji = "📊"
            diff = 0
            target_label = "Market resolved"

        arrow = "🔺" if diff > 0.005 else ("🔻" if diff < -0.005 else "✅")

        self._tg.send(
            f"{emoji} <b>EXIT {outcome}</b> — {reason}\n"
            f"Entry: {entry_price:.3f} → Exit: {exit_price:.3f}\n"
            f"{target_label} │ Diff: {arrow} {diff:+.3f}\n"
            f"PnL: <b>${pnl:+.2f}</b> ({pnl_pct:+.1f}%)\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"Sessione: ${s['cum_pnl']:+.2f} │ WR: {s['win_rate']:.0%} │ "
            f"T: {s['trades']} │ DD: ${s['drawdown']:.2f}"
        )

        # Send alerts/kills
        for msg in alerts:
            self._tg.send(msg)

    # ── Periodic summary (call from main loop) ────────────────────────────

    def maybe_send_hourly(self) -> None:
        """Send hourly summary if an hour has passed."""
        now = time.time()
        if now - self._last_hourly < 3600:
            return
        self._last_hourly = now
        s = self.cb.summary
        if s["trades"] == 0:
            return

        elapsed_h = (now - self._session_start) / 3600
        self._tg.send(
            f"📊 <b>Report orario</b> ({elapsed_h:.1f}h)\n"
            f"Trades: {s['trades']} │ WR: {s['win_rate']:.0%}\n"
            f"PnL: <b>${s['cum_pnl']:+.2f}</b> (oggi: ${s['daily_pnl']:+.2f})\n"
            f"DD: ${s['drawdown']:.2f} │ Peak: ${s['peak_pnl']:.2f}\n"
            f"Streak attuale: {s['current_streak']} │ Peggiore: {s['worst_streak']}"
        )

    def record_market(self) -> None:
        """Track market count even if no entry."""
        self.cb.record_market()
