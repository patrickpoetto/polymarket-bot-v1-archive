# Polymarket Bot — v1 Archive

**Status: archival / exploratory.** This repository is a snapshot of an early,
broad round of experimentation across **six-plus distinct trading strategies**
on Polymarket, built while figuring out which approaches were worth pursuing
further. It is not a single polished product — it's a record of breadth-first
exploration: trying many ideas, in parallel, to see what stuck.

It predates and directly informed a later, far more refined project
(`poly_bot` / `polymarket-btc5m`), which took the single most promising
strategy from this codebase (BTC up/down 5-minute prediction) and rebuilt it
with much tighter execution, testing, and risk controls. This repo is kept
public-ish (private, for portfolio review) to show that progression honestly:
where the ideas started, and how much exploration preceded the eventual
focus.

## What's in here

Six-plus strategies, each in its own module, sharing common infrastructure:

- **`bot1_btc.py`, `bot3_btc.py`** — Autonomous and threshold-based BTC
  up/down 5-minute prediction bots (the strategy that was later extracted
  and refined into its own dedicated project).
- **`bot2_btc/`** — Copy-trading bot: follows and mirrors a target wallet's
  positions via REST and WebSocket feeds.
- **`bot4_chainlink/`** — Chainlink price-feed verification, used to
  cross-check Polymarket market resolution against an independent oracle.
- **`bot5_martingale/`** — Martingale-style staking/progression strategy on
  BTC up/down markets.
- **`bot6_mm/`** — Market-making experiments (quoting both sides of the
  book, inventory/skew management).
- **`bot6_pairs/`** — Pairs-trading experiments across correlated markets.
- **`shared/`** — Common modules used across bots: configuration loading,
  market discovery, market-result verification, trade logging, and
  position redemption (gasless via relayer, and direct on-chain).
- **`utils/`** — Standalone analysis/backtesting scripts (allowance
  approval, wallet analysis, order-book data collection, market-making
  simulation sweeps, etc.).

## Tech stack

- [`py-clob-client`](https://github.com/Polymarket/py-clob-client) — Polymarket CLOB order placement/signing
- `web3.py` — on-chain interaction (allowances, redemptions, signing)
- `aiohttp` / `websockets` — async REST and real-time market data feeds
- `numpy` / `numba` — vectorized/JIT-compiled backtesting and simulation
- `python-dotenv` — configuration via `.env`
- `beautifulsoup4` — lightweight HTML parsing for market discovery

## Honest caveats

This is archival code from an experimentation phase, not a maintained
product:

- Strategies vary widely in maturity — some are closer to working bots,
  others are closer to spikes/prototypes.
- Historical price data, logs, and local virtual environments were
  intentionally excluded from this repository (see `.gitignore`) — only
  source code and documentation were kept.
- Configuration is via a `.env` file (see `.env.example` for the variable
  names expected); no real credentials are included or were ever committed.
- Expect inconsistent error handling, duplicated logic across bots, and
  rough edges typical of fast iteration across many parallel ideas.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # then fill in your own credentials
```

Each bot is run directly, e.g. `python bot1_btc.py`, `python bot3_btc.py`,
or from within its own subdirectory for the others.
