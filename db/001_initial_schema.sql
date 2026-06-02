-- Options Engine v1 — Initial schema
-- Run on your Aiven Postgres instance
-- psql $DATABASE_URL -f 001_initial_schema.sql

BEGIN;

-- =========================================================
-- Raw option chain snapshots (5-min granularity during market hours)
-- =========================================================
CREATE TABLE IF NOT EXISTS option_chain_snapshots (
  id              BIGSERIAL PRIMARY KEY,
  symbol          TEXT NOT NULL,
  spot            NUMERIC(12,2) NOT NULL,
  expiry          DATE NOT NULL,
  strike          INTEGER NOT NULL,
  option_type     CHAR(2) NOT NULL CHECK (option_type IN ('CE','PE')),
  ltp             NUMERIC(12,2),
  bid             NUMERIC(12,2),
  ask             NUMERIC(12,2),
  volume          BIGINT,
  oi              BIGINT,
  oi_change       BIGINT,
  iv              NUMERIC(8,2),
  snapshot_ts     TIMESTAMPTZ NOT NULL,
  CONSTRAINT uq_snapshot UNIQUE (symbol, expiry, strike, option_type, snapshot_ts)
);

CREATE INDEX IF NOT EXISTS idx_chain_recent
  ON option_chain_snapshots (symbol, expiry, snapshot_ts DESC);

CREATE INDEX IF NOT EXISTS idx_chain_strike_lookup
  ON option_chain_snapshots (symbol, expiry, strike, option_type, snapshot_ts DESC);

-- =========================================================
-- Daily EOD aggregates (computed nightly from snapshots)
-- =========================================================
CREATE TABLE IF NOT EXISTS option_daily_stats (
  symbol          TEXT NOT NULL,
  expiry          DATE NOT NULL,
  strike          INTEGER NOT NULL,
  option_type     CHAR(2) NOT NULL,
  trade_date      DATE NOT NULL,
  open_ltp        NUMERIC(12,2),
  high_ltp        NUMERIC(12,2),
  low_ltp         NUMERIC(12,2),
  close_ltp       NUMERIC(12,2),
  open_iv         NUMERIC(8,2),
  close_iv        NUMERIC(8,2),
  high_iv         NUMERIC(8,2),
  low_iv          NUMERIC(8,2),
  total_volume    BIGINT,
  close_oi        BIGINT,
  oi_change_daily BIGINT,
  PRIMARY KEY (symbol, expiry, strike, option_type, trade_date)
);

-- =========================================================
-- FII / DII activity (daily, sourced EOD ~6pm IST)
-- =========================================================
CREATE TABLE IF NOT EXISTS fii_dii_activity (
  trade_date                         DATE PRIMARY KEY,
  fii_cash_buy                       NUMERIC(14,2),
  fii_cash_sell                      NUMERIC(14,2),
  fii_cash_net                       NUMERIC(14,2),
  dii_cash_buy                       NUMERIC(14,2),
  dii_cash_sell                      NUMERIC(14,2),
  dii_cash_net                       NUMERIC(14,2),
  fii_index_fut_long_contracts       BIGINT,
  fii_index_fut_short_contracts      BIGINT,
  fii_index_fut_net_contracts        BIGINT,
  fii_index_fut_long_short_ratio     NUMERIC(8,4),
  fii_index_call_long                BIGINT,
  fii_index_call_short               BIGINT,
  fii_index_put_long                 BIGINT,
  fii_index_put_short                BIGINT,
  fii_stock_fut_long                 BIGINT,
  fii_stock_fut_short                BIGINT,
  raw_payload                        JSONB,
  ingested_at                        TIMESTAMPTZ DEFAULT NOW()
);

-- =========================================================
-- Positions (trade journal — the keystone table)
-- =========================================================
CREATE TABLE IF NOT EXISTS positions (
  id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  symbol              TEXT NOT NULL,
  expiry              DATE NOT NULL,
  strike              INTEGER NOT NULL,
  option_type         CHAR(2) NOT NULL,
  action              CHAR(4) NOT NULL CHECK (action IN ('BUY','SELL')),
  lots                INTEGER NOT NULL CHECK (lots > 0),
  lot_size            INTEGER NOT NULL DEFAULT 75,
  entry_price         NUMERIC(12,2) NOT NULL,
  entry_ts            TIMESTAMPTZ NOT NULL,
  entry_spot          NUMERIC(12,2),
  entry_iv            NUMERIC(8,2),
  thesis              TEXT NOT NULL CHECK (length(thesis) >= 50),
  setup_tag           TEXT NOT NULL,
  planned_stop        NUMERIC(12,2),
  planned_target      NUMERIC(12,2),
  exit_price          NUMERIC(12,2),
  exit_ts             TIMESTAMPTZ,
  exit_reason         TEXT,
  pnl                 NUMERIC(14,2),
  status              TEXT NOT NULL DEFAULT 'open' CHECK (status IN ('open','closed')),
  created_at          TIMESTAMPTZ DEFAULT NOW(),
  updated_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_positions_open ON positions (status) WHERE status = 'open';
CREATE INDEX IF NOT EXISTS idx_positions_setup ON positions (setup_tag, exit_ts);

-- =========================================================
-- Ingestion run log (debugging + monitoring)
-- =========================================================
CREATE TABLE IF NOT EXISTS ingestion_runs (
  id              BIGSERIAL PRIMARY KEY,
  job_name        TEXT NOT NULL,
  started_at      TIMESTAMPTZ NOT NULL,
  finished_at     TIMESTAMPTZ,
  rows_written    INTEGER,
  status          TEXT,
  error_message   TEXT
);

COMMIT;
