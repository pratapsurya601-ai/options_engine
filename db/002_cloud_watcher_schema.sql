-- Options Engine v2 - Cloud Watcher schema additions
-- Run on Aiven Postgres after 001_initial_schema.sql:
--   psql $DATABASE_URL -f 002_cloud_watcher_schema.sql

BEGIN;

-- =========================================================
-- Signals log: every rule evaluation that produced a fire.
-- Independent of whether a position was actually opened
-- (e.g. signal fired but cooldown active).
-- =========================================================
CREATE TABLE IF NOT EXISTS signals (
  id              BIGSERIAL PRIMARY KEY,
  rule_name       TEXT NOT NULL,
  symbol          TEXT NOT NULL,
  ts              TIMESTAMPTZ NOT NULL,
  spot            NUMERIC(12,2),
  action          TEXT CHECK (action IN ('BUY_CE','BUY_PE','SELL_CE','SELL_PE','ALERT')),
  strike          INTEGER,
  expiry          DATE,
  premium         NUMERIC(12,2),
  target_premium  NUMERIC(12,2),
  stop_premium    NUMERIC(12,2),
  trigger_context JSONB,
  outcome         TEXT,  -- 'opened_position', 'skipped_cooldown', 'skipped_already_open', 'alert_only'
  created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_signals_rule_ts
  ON signals (rule_name, ts DESC);

CREATE INDEX IF NOT EXISTS idx_signals_recent
  ON signals (ts DESC);

-- =========================================================
-- Rule cooldown tracker: when did each rule last fire?
-- Cloud watcher uses this in lieu of in-memory cooldown state.
-- One row per (rule_name, symbol). Upsert on every fire.
-- =========================================================
CREATE TABLE IF NOT EXISTS rule_cooldowns (
  rule_name       TEXT NOT NULL,
  symbol          TEXT NOT NULL,
  last_fired_at   TIMESTAMPTZ NOT NULL,
  cooldown_min    INTEGER NOT NULL,
  PRIMARY KEY (rule_name, symbol)
);

-- =========================================================
-- Trailing stop / position state extensions to positions table.
-- These columns are needed by the cloud watcher's exit logic.
-- =========================================================
ALTER TABLE positions
  ADD COLUMN IF NOT EXISTS high_water_mark NUMERIC(12,2),
  ADD COLUMN IF NOT EXISTS rule_name TEXT,
  ADD COLUMN IF NOT EXISTS trail_activation_pts NUMERIC(12,2),
  ADD COLUMN IF NOT EXISTS trail_distance_pts NUMERIC(12,2),
  ADD COLUMN IF NOT EXISTS hold_until_ts TIMESTAMPTZ,
  ADD COLUMN IF NOT EXISTS source TEXT NOT NULL DEFAULT 'paper'
    CHECK (source IN ('paper','real'));

CREATE INDEX IF NOT EXISTS idx_positions_open_rule
  ON positions (status, rule_name) WHERE status = 'open';

-- =========================================================
-- Cloud watcher run log (debugging + monitoring).
-- =========================================================
CREATE TABLE IF NOT EXISTS watcher_runs (
  id              BIGSERIAL PRIMARY KEY,
  rule_name       TEXT NOT NULL,
  symbol          TEXT NOT NULL,
  started_at      TIMESTAMPTZ NOT NULL,
  finished_at     TIMESTAMPTZ,
  bars_loaded     INTEGER,
  signals_fired   INTEGER DEFAULT 0,
  positions_opened INTEGER DEFAULT 0,
  positions_closed INTEGER DEFAULT 0,
  trail_updated   INTEGER DEFAULT 0,
  status          TEXT,
  error_message   TEXT
);

CREATE INDEX IF NOT EXISTS idx_watcher_runs_recent
  ON watcher_runs (rule_name, started_at DESC);

-- Reasonable thesis default for cloud paper trades so the existing
-- (length >= 50) CHECK constraint is satisfied.
COMMENT ON COLUMN positions.thesis IS
  'Required >= 50 chars. Cloud watcher fills with rule_name + trigger context summary.';

COMMIT;
