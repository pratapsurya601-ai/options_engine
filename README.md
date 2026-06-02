# Options Engine

NIFTY options analytics + recommender. Two layers:

- **v1 ingestor** — pulls NSE option chain (5-min) and FII/DII activity (EOD) into Postgres.
- **v2 engine** — Black-Scholes pricing, lognormal win-probabilities, FII/DII smart-money scoring, and a strategy recommender that takes your directional view + target win-rate and ranks trades.

**Honest caveat:** the engine does NOT predict prices from learned patterns. It models option payoffs and probabilities under standard assumptions (Black-Scholes / lognormal), then ranks strategies that satisfy your stated win-rate target. Learned-pattern modules (IV crush, OI buildup, gamma pin) still need >=2 weeks of clean snapshots before they're meaningful.

## Quick start — engine only (no DB needed)

```bash
pip install -r requirements.txt

# Bullish view, target 65% win-rate, Rs 50k budget
python -m engine.cli --view bullish --conviction 0.6 --win 0.65 --capital 50000

# Bearish with manual smart-money override
python -m engine.cli --view bearish --win 0.6 --sm-score -0.4

# Neutral / range-bound (iron condors, strangles)
python -m engine.cli --view neutral --conviction 0.4 --win 0.7
```

Add `--source db` to read the chain from your ingestor's Postgres instead of fetching live, and `--smart-money` to pull the latest FII/DII positioning from the DB.

## What the engine outputs

For each candidate strategy (long calls/puts, debit & credit vertical spreads, straddles, strangles, iron condors):
- cost / max profit / max loss / breakevens (sized to your capital)
- **P(profit at expiry)** under lognormal model with drift = your view x conviction x IV
- **Expected P&L** (numerically integrated payoff x pdf)
- **Smart-money alignment** — your view checked against latest FII/DII positioning; conviction boosted if aligned, warned if opposed

## Engine module map

```
engine/
  pricing.py       Black-Scholes + Greeks, IV solver
  probability.py   P(above/below/between/touch) under lognormal, view->drift mapping
  smart_money.py   FII/DII row -> bias score in [-1, +1]
  strategies.py    Leg/Strategy payoff models, builders for 10 common structures
  chain.py         Chain data class; loaders from live NSE or DB
  recommender.py   Top-N ranked TradeIdea given (view, conviction, target win, capital)
  cli.py           python -m engine.cli ...
```

Run engine math tests (no network, no DB): `python scripts/test_engine.py`

---

## v1 ingestor

Persistent data accumulation. Backs the `--source db` path on the engine.

**Original v1 framing:** This is data accumulation only. No analytics, no signals, no UI. That's intentional — every downstream module needs >=2 weeks of clean snapshots before it's useful.

## What's in v1

- NSE option chain ingestor (NIFTY + BANKNIFTY, 5-min cadence)
- FII/DII activity ingestor (cash + derivatives, EOD)
- Postgres schema with trade journal table ready for v2
- Ingestion run log for observability
- Railway-ready deployment config

## Setup

```bash
# 1. Clone repo, install deps
pip install -r requirements.txt

# 2. Create Aiven Postgres database (or use existing YieldIQ instance with separate schema)
export DATABASE_URL='postgresql://user:pass@host:port/dbname?sslmode=require'

# 3. Run migration
psql $DATABASE_URL -f db/001_initial_schema.sql

# 4. Smoke test (read-only)
python scripts/smoke_test.py

# 5. First real ingestion
python -m ingestor.runner option_chain
```

## Operation modes

```bash
# One-shot (for cron)
python -m ingestor.runner option_chain
python -m ingestor.runner fii_dii

# Continuous poll (for local dev or single-service deploy)
python -m ingestor.runner loop
```

## Deploy on Railway

```bash
railway login
railway link <project>
railway up
# Set DATABASE_URL in Railway dashboard env vars
```

Cron schedule lives in `railway.toml`.

## Monitoring

Check ingestion health:
```sql
SELECT job_name, status, COUNT(*) AS runs,
       MAX(started_at) AS last_run,
       SUM(rows_written) AS total_rows
FROM ingestion_runs
WHERE started_at > NOW() - INTERVAL '7 days'
GROUP BY job_name, status
ORDER BY job_name, status;
```

Check option chain freshness:
```sql
SELECT symbol, MAX(snapshot_ts) AS last_snapshot,
       NOW() - MAX(snapshot_ts) AS staleness
FROM option_chain_snapshots
GROUP BY symbol;
```

## Known issues / things to watch

- **NSE blocks** — if you see repeated 401/403, the user-agent may be flagged. Rotate UA in `config/settings.py`.
- **Holidays** — `is_market_open()` doesn't check NSE holiday calendar. Add `nse_holidays` table in v2 if needed.
- **FII/DII field drift** — NSE has changed field names twice in the last 2 years. `fii_dii_fetcher.normalize_cash_row()` defends against this; if it silently returns no data, check raw payload in `fii_dii_activity.raw_payload`.
- **Lot size hardcoded** — Nifty lot is 75 (as of 2025). If changed, update in `positions` table default and any new ingestion logic.

## Next: Weekend 2

- Streamlit journal UI (entry/exit logging with mandatory thesis)
- Position monitor with theta/IV crush alerts
- Telegram bot for alerts

Do NOT skip Weekend 2 to build analytics. The journal is where actual edge comes from.
