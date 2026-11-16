"""
Price Logger - Registrazione prezzi Up/Down per mercato in CSV.
"""
import csv
from datetime import datetime, timezone
from pathlib import Path


class PriceLogger:
    """Logger dedicato agli snapshot prezzi di mercato."""

    def __init__(self, logs_dir="logs/bot1_btc/prices"):
        self.logs_dir = Path(logs_dir)
        self.csv_dir = self.logs_dir / "csv"
        self.csv_dir.mkdir(parents=True, exist_ok=True)
        self.current_condition_id = None
        self.current_market_file = None
        self._fh = None
        self._writer = None

    @staticmethod
    def _compact_market_label(market_name: str) -> str:
        """Build compact filename label: March_12__4_15AM_4_30AM_ET."""
        rhs = market_name
        if " - " in market_name:
            rhs = market_name.split(" - ", 1)[1]
        safe = "".join(ch if ch.isalnum() else "_" for ch in rhs.strip())
        safe = "_".join(part for part in safe.split("_") if part)
        return safe or "unknown_market"

    def start_market(self, condition_id: str, market_name: str):
        """Apre un nuovo CSV per il mercato corrente."""
        if not condition_id:
            return
        if self.current_condition_id == condition_id and self._writer is not None:
            return

        self.close_market()

        market_label = self._compact_market_label(market_name)
        filename = f"prices_{market_label}.csv"
        self.current_market_file = self.csv_dir / filename
        file_exists = self.current_market_file.exists()
        self._fh = open(self.current_market_file, "a", newline="", encoding="utf-8")
        self._writer = csv.writer(self._fh)
        if not file_exists:
            self._writer.writerow(
                [
                    "timestamp_iso",
                    "timestamp_unix_ms",
                    "condition_id",
                    "market_name",
                    "outcome",
                    "token_id",
                    "ask",
                    "bid",
                    "time_left_seconds",
                ]
            )
        self._fh.flush()
        self.current_condition_id = condition_id

    def log_snapshot(self, condition_id: str, market_name: str, prices: dict, time_left_seconds: float):
        """Registra uno snapshot dei prezzi per tutti gli outcome disponibili."""
        if not condition_id or not prices:
            return

        if self.current_condition_id != condition_id or self._writer is None:
            self.start_market(condition_id, market_name)
        if self._writer is None:
            return

        now = datetime.now(timezone.utc)
        now_iso = now.isoformat()
        now_ms = int(now.timestamp() * 1000)
        tl = round(time_left_seconds, 3) if time_left_seconds is not None else ""

        for outcome, payload in prices.items():
            token_id = payload.get("token_id", "")
            ask = payload.get("ask") if payload.get("ask") is not None else payload.get("price")
            bid = payload.get("bid")
            self._writer.writerow(
                [
                    now_iso,
                    now_ms,
                    condition_id,
                    market_name,
                    outcome,
                    token_id,
                    "" if ask is None else f"{ask:.6f}",
                    "" if bid is None else f"{bid:.6f}",
                    tl,
                ]
            )
        self._fh.flush()

    def close_market(self):
        """Chiude il file corrente."""
        if self._fh:
            self._fh.close()
        self._fh = None
        self._writer = None
        self.current_condition_id = None
