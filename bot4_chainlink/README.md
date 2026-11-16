# Bot4 Chainlink Price Pipeline

This folder is intentionally isolated from other bots.

## Purpose

Build a reproducible, auditable pipeline to:

1. Download Chainlink BTC/USD Data Streams historical data (via SDK).
2. Reconstruct market outcomes from Chainlink prices.
3. Verify one-to-one alignment with `archived_data/btc_5m_outcomes_history_merged.csv`.
4. Produce a clean dataset for model/backtest work.

## Design Rules

- Do not modify shared bot logic in `shared/` for this workflow.
- Use only `BOT4_*` environment variables (namespaced config).
- Keep all generated artifacts under `bot4_chainlink/data/`.
- Keep scripts deterministic and idempotent (safe to re-run).
- Always output a mismatch report when verifying alignment.

## Files

- `config.py`: namespaced env config (`BOT4_*` only).
- `chainlink_client.py`: Chainlink SDK wrapper.
- `verify_chainlink_alignment.py`: end-to-end verification CLI.
- `data/`: local artifacts (ignored in git if needed).

## Quick Start

1. Set `BOT4_*` variables in your `.env` (see comments in `config.py`).
2. Run:

```bash
python bot4_chainlink/verify_chainlink_alignment.py --limit 200
```

Strict zero-mismatch mode is default: if even one mismatch is found, the command exits with non-zero code.
If you need exploratory runs, use:

```bash
python bot4_chainlink/verify_chainlink_alignment.py --allow-mismatch
```

3. Check outputs in `bot4_chainlink/data/`:
   - `chainlink_prices.csv`
   - `alignment_report.csv`
   - `alignment_summary.json`

## Notes

- This scaffold keeps strict separation from existing bots.
- The first implementation is intentionally conservative: clear logs, explicit assumptions, and auditable outputs.
