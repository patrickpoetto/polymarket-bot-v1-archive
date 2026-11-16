"""
Trade Logger - Sistema di logging dedicato per operazioni di trading
Crea file separati per ogni sessione con solo le info essenziali
"""
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

class TradeLogger:
    """Logger dedicato per operazioni di trading"""
    
    def __init__(self, logs_dir="logs/bot1_btc"):
        """
        Inizializza il trade logger
        
        Args:
            logs_dir: Directory dove salvare i log
        """
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(exist_ok=True)

        # Capture session start time for later reconciliation
        _now = datetime.now()
        self.session_start_ts: int = int(_now.timestamp())

        # Nome file basato su timestamp sessione
        session_time = _now.strftime("%Y-%m-%d_%H-%M-%S")
        self.log_file = self.logs_dir / f"session_{session_time}.log"
        self.history_file: Path = self.logs_dir / f"session_{session_time}_history.txt"
        
        # Setup logger dedicato
        self.logger = logging.getLogger('TradeLogger')
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False  # Non propagare al root logger
        
        # Rimuovi handlers esistenti
        self.logger.handlers = []
        
        # File handler - solo per file
        file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # Formato pulito senza timestamp (aggiungeremo noi)
        formatter = logging.Formatter('%(message)s')
        file_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        
        # Scrivi header
        self._write_header()
        
        # Stats sessione
        self.session_trades = 0
        self.session_wins = 0
        self.session_losses = 0
        self.session_pnl = 0.0
    
    def _write_header(self):
        """Scrivi header del file"""
        self.logger.info("=" * 80)
        self.logger.info(f"📊 TRADING SESSION - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info("=" * 80)
        self.logger.info("")
    
    def _get_timestamp(self):
        """Ottieni timestamp formattato"""
        return datetime.now().strftime("%H:%M:%S")
    
    def log_market_start(self, market_name):
        """Log inizio nuovo mercato"""
        self.logger.info("-" * 80)
        self.logger.info(f"🔔 NUOVO MERCATO - {self._get_timestamp()}")
        self.logger.info(f"   {market_name}")
        self.logger.info("")
    
    def log_entry(self, outcome, entry_price, capital, shares, is_recovery=False):
        """Log entrata in posizione"""
        entry_type = "🔄 RECOVERY ENTRY" if is_recovery else "📈 ENTRY"
        self.logger.info(f"{entry_type} - {self._get_timestamp()}")
        self.logger.info(f"   Outcome: {outcome}")
        self.logger.info(f"   Entry Price: ${entry_price:.3f}")
        self.logger.info(f"   Capital: ${capital:.2f}")
        self.logger.info(f"   Shares: {shares:.2f}")
        if is_recovery:
            self.logger.info(f"   Type: Recovery Trade")
        self.logger.info("")
    
    def log_exit(self, outcome, entry_price, exit_price, reason, shares, pnl, pnl_pct, did_win=None):
        """
        Log uscita da posizione
        
        Args:
            outcome: 'Up' o 'Down'
            entry_price: Prezzo di entrata
            exit_price: Prezzo di uscita
            reason: 'take_profit', 'stop_loss', 'market_expiry', 'market_resolved'
            shares: Numero di shares
            pnl: Profit/Loss in $
            pnl_pct: Profit/Loss in %
        """
        self.session_trades += 1
        
        # Per market_resolved, usa did_win se disponibile
        if reason == 'market_resolved':
            if did_win is True:
                self.session_wins += 1
                self.session_pnl += pnl
                status = "✅ WIN"
            elif did_win == "probable":
                # Win probabile (basato su prezzo alto)
                self.session_wins += 1
                self.session_pnl += pnl
                status = "✅ WIN (probable)"
            elif did_win is False:
                self.session_losses += 1
                self.session_pnl += pnl
                status = "❌ LOSS"
            else:
                # Non sappiamo ancora
                status = "🎲 PENDING"
        else:
            won = pnl > 0
            if won:
                self.session_wins += 1
                status = "✅ WIN"
            else:
                self.session_losses += 1
                status = "❌ LOSS"
            self.session_pnl += pnl
        
        # Emoji per reason
        reason_emoji = {
            'take_profit': '🎯',
            'stop_loss': '🛑',
            'market_resolved': '🎲',
            'stop_loss_failed': '⚠️',
            'take_profit_failed': '⚠️'
        }
        emoji = reason_emoji.get(reason, '🔚')
        
        # Reason leggibile
        reason_text = {
            'take_profit': 'Take Profit',
            'stop_loss': 'Stop Loss',
            'market_resolved': 'Market Resolved',
            'stop_loss_failed': 'Stop Loss (sell failed)',
            'take_profit_failed': 'Take Profit (sell failed)'
        }
        reason_str = reason_text.get(reason, reason)
        
        self.logger.info(f"{emoji} EXIT - {self._get_timestamp()} - {status}")
        self.logger.info(f"   Outcome: {outcome}")
        self.logger.info(f"   Reason: {reason_str}")
        
        # Per market_resolved, mostra risultato se noto
        if reason == 'market_resolved':
            if did_win is True:
                self.logger.info(f"   Entry: ${entry_price:.3f} → Resolved: ${exit_price:.2f} ({outcome} WON)")
            elif did_win == "probable":
                self.logger.info(f"   Entry: ${entry_price:.3f} → Last price: ${exit_price:.2f} (Probable WIN)")
            elif did_win is False:
                self.logger.info(f"   Entry: ${entry_price:.3f} → Resolved: ${exit_price:.2f} ({outcome} LOST)")
            else:
                # Pending, mostra ultimo prezzo se disponibile
                if exit_price > 0:
                    self.logger.info(f"   Entry: ${entry_price:.3f} → Last price: ${exit_price:.2f} (Below threshold)")
                else:
                    self.logger.info(f"   Entry: ${entry_price:.3f} → Price unavailable")
            
            self.logger.info(f"   Shares: {shares:.2f}")
            
            if did_win is not None:
                self.logger.info(f"   P/L: ${pnl:+.2f} ({pnl_pct:+.1f}%)")
            else:
                if exit_price > 0:
                    self.logger.info(f"   Current P/L: ${pnl:+.2f} ({pnl_pct:+.1f}%)")
                    self.logger.info(f"   Note: Awaiting final resolution")
                else:
                    self.logger.info(f"   P/L: Pending resolution")
        else:
            self.logger.info(f"   Entry: ${entry_price:.3f} → Exit: ${exit_price:.3f}")
            self.logger.info(f"   Shares: {shares:.2f}")
            self.logger.info(f"   P/L: ${pnl:+.2f} ({pnl_pct:+.1f}%)")

        # Riga cumulativa dopo ogni trade chiuso
        wr = (self.session_wins / self.session_trades * 100) if self.session_trades else 0
        cum_status = "📈" if self.session_pnl >= 0 else "📉"
        self.logger.info(
            f"   {'─' * 44}"
        )
        self.logger.info(
            f"   {cum_status} Cumulativo: ${self.session_pnl:+.2f} | "
            f"{self.session_wins}W {self.session_losses}L | WR: {wr:.0f}%"
        )
        self.logger.info("")
    
    def log_no_trade(self, reason):
        """Log quando non si trade in un mercato"""
        self.logger.info(f"⏸️  NO TRADE - {self._get_timestamp()}")
        self.logger.info(f"   Reason: {reason}")
        self.logger.info("")
    
    def log_session_summary(self):
        """Log summary finale della sessione"""
        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info(f"📊 SESSION SUMMARY - {self._get_timestamp()}")
        self.logger.info("=" * 80)
        self.logger.info(f"   Total Trades: {self.session_trades}")
        self.logger.info(f"   Wins: {self.session_wins}")
        self.logger.info(f"   Losses: {self.session_losses}")
        
        if self.session_trades > 0:
            win_rate = (self.session_wins / self.session_trades) * 100
            self.logger.info(f"   Win Rate: {win_rate:.1f}%")
        
        status = "✅ PROFIT" if self.session_pnl > 0 else "❌ LOSS"
        self.logger.info(f"   Total P/L: ${self.session_pnl:+.2f} {status}")
        self.logger.info("=" * 80)
    
    def log_real_history_summary(self, text_summary: str, history_file: Optional[Path] = None) -> None:
        """
        Append the real trade history summary (from Polymarket API) to the
        session log, and save a copy to a dedicated history text file.
        """
        for line in text_summary.splitlines():
            self.logger.info(line)

        hfile = history_file or self.history_file
        try:
            hfile.write_text(text_summary, encoding="utf-8")
        except Exception as exc:
            self.logger.info(f"   ⚠️ Impossibile salvare history file: {exc}")

    def log_error(self, error_msg):
        """Log errori importanti"""
        self.logger.info(f"❌ ERROR - {self._get_timestamp()}")
        self.logger.info(f"   {error_msg}")
        self.logger.info("")
    
    def log_claim(self, num_markets, success):
        """Log claim operazioni"""
        status = "✅" if success else "❌"
        self.logger.info(f"{status} CLAIM - {self._get_timestamp()}")
        self.logger.info(f"   Markets: {num_markets}")
        self.logger.info("")

# Istanza globale (opzionale)
_trade_logger = None

def get_trade_logger():
    """Ottieni o crea istanza del trade logger"""
    global _trade_logger
    if _trade_logger is None:
        _trade_logger = TradeLogger()
    return _trade_logger
