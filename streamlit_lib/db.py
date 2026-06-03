"""Aiven Postgres connection + cached queries for the Streamlit dashboard.

All reads are cached for 5 minutes (`@st.cache_data(ttl=300)`) to stay
well within Streamlit Cloud free tier limits (1 GB RAM, 1 CPU).

DATABASE_URL is loaded from (in order):
  1. ``st.secrets['DATABASE_URL']``  (Streamlit Cloud)
  2. ``os.environ['DATABASE_URL']``  (local dev)

The connection itself is held in ``@st.cache_resource`` so it is reused across
reruns.
"""
from __future__ import annotations

import os
from typing import Optional

import pandas as pd
import psycopg
import streamlit as st
from psycopg.rows import dict_row


_NUMERIC_CHAIN_COLS = (
    "spot", "ltp", "bid", "ask", "iv",
    "oi", "oi_change", "volume", "strike",
)


def _coerce_chain_numerics(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce Postgres NUMERIC/BIGINT columns from Decimal/object to float.

    psycopg returns NUMERIC as decimal.Decimal which lands in object-dtype
    pandas columns. Under Python 3.14 / new pandas, comparisons like
    `series > 0`, unary `-series`, and Plotly's serialization all break on
    object columns. Coercing everything to numeric here means every Streamlit
    page downstream gets clean float series without per-page guards."""
    if df is None or len(df) == 0:
        return df
    for col in _NUMERIC_CHAIN_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _get_database_url() -> str:
    """Resolve the DATABASE_URL from Streamlit secrets or env."""
    # st.secrets raises if no secrets file exists; guard with try/except
    try:
        if "DATABASE_URL" in st.secrets:
            return st.secrets["DATABASE_URL"]
    except Exception:
        pass
    return os.environ.get("DATABASE_URL", "")


@st.cache_resource
def get_connection():
    """Open and cache a single psycopg connection.

    NOTE: connection has the DEFAULT (tuple) row_factory so that
    ``pd.read_sql`` produces correct DataFrames (it iterates the cursor
    expecting sequences; dict_row would give back the column names as
    values). Code paths that need dict-style access should construct their
    own cursor via ``_dict_cursor(conn)``.
    """
    url = _get_database_url()
    if not url:
        raise RuntimeError(
            "DATABASE_URL not configured. Set it in .streamlit/secrets.toml or env."
        )
    return psycopg.connect(url, autocommit=True)


def _dict_cursor(conn):
    """Return a cursor that yields dicts (use for manual fetchone/fetchall paths)."""
    return conn.cursor(row_factory=dict_row)


@st.cache_data(ttl=300)
def coverage_stats() -> dict:
    """Total snapshots, days covered, oldest/newest. Empty defaults on no data."""
    conn = get_connection()
    with _dict_cursor(conn) as cur:
        cur.execute(
            """
            SELECT
                COUNT(*)                                                          AS total_rows,
                COUNT(DISTINCT snapshot_ts)                                       AS total_snapshots,
                COUNT(DISTINCT DATE(snapshot_ts AT TIME ZONE 'Asia/Kolkata'))     AS days_covered,
                MIN(snapshot_ts)                                                  AS oldest,
                MAX(snapshot_ts)                                                  AS newest
            FROM option_chain_snapshots
            WHERE symbol = 'NIFTY'
            """
        )
        row = cur.fetchone() or {}
    return {
        "total_rows": int(row.get("total_rows") or 0),
        "total_snapshots": int(row.get("total_snapshots") or 0),
        "days_covered": int(row.get("days_covered") or 0),
        "oldest": row.get("oldest"),
        "newest": row.get("newest"),
    }


@st.cache_data(ttl=300)
def latest_snapshot_summary() -> Optional[dict]:
    """Spot, expiry, n_strikes at the most recent snapshot. None if no data."""
    conn = get_connection()
    with _dict_cursor(conn) as cur:
        cur.execute(
            """
            SELECT snapshot_ts, spot, expiry, COUNT(*) AS n_strikes
            FROM option_chain_snapshots
            WHERE symbol = 'NIFTY'
              AND snapshot_ts = (
                  SELECT MAX(snapshot_ts) FROM option_chain_snapshots
                  WHERE symbol = 'NIFTY'
              )
            GROUP BY snapshot_ts, spot, expiry
            ORDER BY expiry
            LIMIT 1
            """
        )
        row = cur.fetchone()
    if row is None:
        return None
    # Coerce Decimal -> float at the boundary
    out = dict(row)
    if out.get("spot") is not None:
        try:
            out["spot"] = float(out["spot"])
        except (TypeError, ValueError):
            out["spot"] = None
    return out


@st.cache_data(ttl=300)
def snapshots_per_day() -> pd.DataFrame:
    """Count distinct snapshot timestamps per IST day. Returns date, n_snapshots."""
    conn = get_connection()
    return pd.read_sql(
        """
        SELECT
            DATE(snapshot_ts AT TIME ZONE 'Asia/Kolkata') AS date,
            COUNT(DISTINCT snapshot_ts)                    AS n_snapshots
        FROM option_chain_snapshots
        WHERE symbol = 'NIFTY'
        GROUP BY date
        ORDER BY date
        """,
        conn,
    )


@st.cache_data(ttl=300)
def expiries_available() -> pd.DataFrame:
    """All distinct expiries with snapshot counts."""
    conn = get_connection()
    return pd.read_sql(
        """
        SELECT expiry, COUNT(DISTINCT snapshot_ts) AS n_snapshots
        FROM option_chain_snapshots
        WHERE symbol = 'NIFTY'
        GROUP BY expiry
        ORDER BY expiry
        """,
        conn,
    )


@st.cache_data(ttl=300)
def latest_chain() -> pd.DataFrame:
    """Full chain at the most recent snapshot (all expiries)."""
    conn = get_connection()
    df = pd.read_sql(
        """
        SELECT *
        FROM option_chain_snapshots
        WHERE symbol = 'NIFTY'
          AND snapshot_ts = (
              SELECT MAX(snapshot_ts) FROM option_chain_snapshots
              WHERE symbol = 'NIFTY'
          )
        ORDER BY expiry, strike, option_type
        """,
        conn,
    )
    return _coerce_chain_numerics(df)


@st.cache_data(ttl=300)
def chain_at_timestamp(ts) -> pd.DataFrame:
    """Full chain at a specific snapshot timestamp."""
    conn = get_connection()
    df = pd.read_sql(
        """
        SELECT * FROM option_chain_snapshots
        WHERE symbol = 'NIFTY' AND snapshot_ts = %s
        ORDER BY expiry, strike, option_type
        """,
        conn,
        params=(ts,),
    )
    return _coerce_chain_numerics(df)


@st.cache_data(ttl=300)
def recent_snapshot_timestamps(n: int = 10) -> list:
    """Most recent N distinct snapshot timestamps (newest first)."""
    conn = get_connection()
    with _dict_cursor(conn) as cur:
        cur.execute(
            """
            SELECT DISTINCT snapshot_ts
            FROM option_chain_snapshots
            WHERE symbol = 'NIFTY'
            ORDER BY snapshot_ts DESC
            LIMIT %s
            """,
            (n,),
        )
        return [r["snapshot_ts"] for r in cur.fetchall()]


# ============================================================================
# Cloud watcher tables — signals, paper positions, watcher runs
# Cached at 60s (vs 300s for chain data) so signals show up quickly.
# ============================================================================

@st.cache_data(ttl=60)
def recent_signals(limit: int = 200) -> pd.DataFrame:
    """Most recent rule fires from signals table. Newest first."""
    conn = get_connection()
    return pd.read_sql(
        """
        SELECT id, rule_name, symbol, ts, spot, action, strike, expiry,
               premium, target_premium, stop_premium, outcome,
               trigger_context, created_at
        FROM signals
        ORDER BY ts DESC
        LIMIT %s
        """,
        conn,
        params=(limit,),
    )


@st.cache_data(ttl=60)
def signal_counts_by_rule(days: int = 30) -> pd.DataFrame:
    """Signal fire counts grouped by rule over the last N days."""
    conn = get_connection()
    return pd.read_sql(
        """
        SELECT rule_name,
               COUNT(*)                                    AS total_fires,
               COUNT(*) FILTER (WHERE outcome = 'opened_position')   AS opened,
               COUNT(*) FILTER (WHERE outcome = 'skipped_cooldown')  AS cooldown_skipped,
               COUNT(*) FILTER (WHERE outcome = 'alert_only')        AS alerts,
               MAX(ts)                                     AS last_fired_at
        FROM signals
        WHERE ts > NOW() - (%s || ' days')::interval
        GROUP BY rule_name
        ORDER BY total_fires DESC
        """,
        conn,
        params=(str(days),),
    )


@st.cache_data(ttl=60)
def open_paper_positions() -> pd.DataFrame:
    """All currently open paper positions."""
    conn = get_connection()
    return pd.read_sql(
        """
        SELECT id, rule_name, symbol, expiry, strike, option_type, action,
               lots, lot_size, entry_price, entry_ts, entry_spot,
               planned_stop, planned_target, high_water_mark, setup_tag
        FROM positions
        WHERE status = 'open'
        ORDER BY entry_ts DESC
        """,
        conn,
    )


@st.cache_data(ttl=60)
def closed_paper_positions(limit: int = 200) -> pd.DataFrame:
    """Most recently closed paper positions with realized PnL."""
    conn = get_connection()
    return pd.read_sql(
        """
        SELECT id, rule_name, symbol, expiry, strike, option_type, action,
               lots, lot_size, entry_price, entry_ts, exit_price, exit_ts,
               exit_reason, pnl, setup_tag
        FROM positions
        WHERE status = 'closed'
        ORDER BY exit_ts DESC NULLS LAST
        LIMIT %s
        """,
        conn,
        params=(limit,),
    )


@st.cache_data(ttl=60)
def position_pnl_summary() -> pd.DataFrame:
    """Realized PnL aggregates per rule (closed positions only)."""
    conn = get_connection()
    return pd.read_sql(
        """
        SELECT rule_name,
               COUNT(*)                                  AS n_trades,
               COUNT(*) FILTER (WHERE pnl > 0)           AS wins,
               COUNT(*) FILTER (WHERE pnl < 0)           AS losses,
               COALESCE(SUM(pnl), 0)                     AS total_pnl,
               COALESCE(AVG(pnl), 0)                     AS avg_pnl,
               COALESCE(AVG(pnl) FILTER (WHERE pnl > 0), 0)  AS avg_win,
               COALESCE(AVG(pnl) FILTER (WHERE pnl < 0), 0)  AS avg_loss,
               MIN(exit_ts)                              AS first_exit,
               MAX(exit_ts)                              AS last_exit
        FROM positions
        WHERE status = 'closed'
        GROUP BY rule_name
        ORDER BY total_pnl DESC
        """,
        conn,
    )


@st.cache_data(ttl=60)
def recent_watcher_runs(limit: int = 50) -> pd.DataFrame:
    """Recent cloud watcher invocations for debugging."""
    conn = get_connection()
    return pd.read_sql(
        """
        SELECT id, rule_name, symbol, started_at, finished_at,
               bars_loaded, signals_fired, positions_opened, positions_closed,
               trail_updated, status, error_message
        FROM watcher_runs
        ORDER BY started_at DESC
        LIMIT %s
        """,
        conn,
        params=(limit,),
    )


@st.cache_data(ttl=300)
def historical_snapshots(symbol: str = "NIFTY", expiry=None, days: int = 30) -> pd.DataFrame:
    """Time-series chain rows for trend analysis (last `days` calendar days)."""
    conn = get_connection()
    if expiry is None:
        query = """
            SELECT snapshot_ts, spot, expiry, strike, option_type,
                   ltp, oi, volume, iv
            FROM option_chain_snapshots
            WHERE symbol = %s
              AND snapshot_ts > NOW() - (%s || ' days')::interval
            ORDER BY snapshot_ts, strike, option_type
        """
        return _coerce_chain_numerics(pd.read_sql(query, conn, params=(symbol, str(days))))
    query = """
        SELECT snapshot_ts, spot, expiry, strike, option_type,
               ltp, oi, volume, iv
        FROM option_chain_snapshots
        WHERE symbol = %s
          AND snapshot_ts > NOW() - (%s || ' days')::interval
          AND expiry = %s
        ORDER BY snapshot_ts, strike, option_type
    """
    return _coerce_chain_numerics(pd.read_sql(query, conn, params=(symbol, str(days), expiry)))
