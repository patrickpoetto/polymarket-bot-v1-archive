#!/usr/bin/env bash
# Lightweight watchdog for collect_market_book_data.py (nohup runs).
# Usage:
#   ./scripts/watch_collector.sh              # log every 60s
#   ./scripts/watch_collector.sh --restart    # relaunch collector if dead
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p logs
INTERVAL="${WATCH_INTERVAL:-60}"
LOG="${WATCH_LOG:-logs/watch_collector.log}"
RESTART=0
if [[ "${1:-}" == "--restart" ]]; then
  RESTART=1
fi

while true; do
  ts="$(date '+%Y-%m-%d %H:%M:%S')"
  if pid=$(pgrep -f "utils/collect_market_book_data.py" 2>/dev/null | head -1); then
    day="$(ls -1t prices/5/days 2>/dev/null | head -1 || echo "?")"
    csv="$(find "prices/5/days/$day/csv" -name 'prices_*.csv' -type f 2>/dev/null | head -1 || true)"
    if [[ -n "$csv" && -f "$csv" ]]; then
      sz="$(stat -f%z "$csv" 2>/dev/null || echo 0)"
      echo "$ts OK pid=$pid day=$day sz=$sz $(basename "$csv")" >>"$LOG"
    else
      echo "$ts OK pid=$pid day=$day (no csv yet)" >>"$LOG"
    fi
  else
    echo "$ts DEAD collector not running" >>"$LOG"
    if [[ "$RESTART" -eq 1 ]]; then
      nohup ./venv/bin/python utils/collect_market_book_data.py \
        --quiet --no-reports --log-file logs/book_collector.log \
        >> logs/book_collector.nohup 2>&1 &
      echo "$ts RESTARTED pid=$!" >>"$LOG"
    fi
  fi
  sleep "$INTERVAL"
done
