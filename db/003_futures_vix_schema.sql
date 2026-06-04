-- Options Engine v2 - Futures + VIX schema additions
-- Run on Aiven Postgres:
--   psql $DATABASE_URL -f 003_futures_vix_schema.sql

BEGIN;

-- =========================================================
-- NIFTY futures snapshots (5-min granularity during market hours)
-- One row per (symbol, expiry, snapshot_ts) — typically current + next
-- month per symbol.
-- =========================================================
CREATE TABLE IF NOT EXISTS futures_snapshots (
  id              BIGSERIAL PRIMARY KEY,
  symbol          TEXT NOT NULL,
  snapshot_ts     TIMESTAMPTZ NOT NULL,
  expiry          DATE NOT NULL,
  ltp             NUMERIC(12,2) NOT NULL,
  bid             NUMERIC(12,2),
  ask             NUMERIC(12,2),
  volume          BIGINT,
  oi              BIGINT,
  oi_change       BIGINT,
  spot            NUMERIC(12,2),               -- spot at the same snapshot for basis math
  basis           NUMERIC(12,2),               -- ltp - spot, signed
  basis_pct       NUMERIC(8,4),                -- basis / spot * 100
  CONSTRAINT uq_futures_snapshot UNIQUE (symbol, expiry, snapshot_ts)
);

CREATE INDEX IF NOT EXISTS idx_futures_recent
  ON futures_snapshots (symbol, snapshot_ts DESC);

CREATE INDEX IF NOT EXISTS idx_futures_expiry_ts
  ON futures_snapshots (symbol, expiry, snapshot_ts DESC);

-- =========================================================
-- India VIX snapshots (5-min granularity)
-- =========================================================
CREATE TABLE IF NOT EXISTS vix_snapshots (
  snapshot_ts     TIMESTAMPTZ PRIMARY KEY,
  vix             NUMERIC(8,2) NOT NULL,
  regime          TEXT GENERATED ALWAYS AS (
    CASE
      WHEN vix < 12 THEN 'LOW'
      WHEN vix < 18 THEN 'NORMAL'
      WHEN vix < 25 THEN 'ELEVATED'
      ELSE 'HIGH'
    END
  ) STORED
);

CREATE INDEX IF NOT EXISTS idx_vix_recent
  ON vix_snapshots (snapshot_ts DESC);

-- =========================================================
-- fii_dii_activity already exists from 001_initial_schema.sql.
-- No changes here.
-- =========================================================

COMMIT;
