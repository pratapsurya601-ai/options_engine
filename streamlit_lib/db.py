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
    """Open and cache a single psycopg connection."""
    url = _get_database_url()
    if not url:
        raise RuntimeError(
            "DATABASE_URL not configured. Set it in .streamlit/secrets.toml or env."
        )
    return psycopg.connect(url, row_factory=dict_row, autocommit=True)


@st.cache_data(ttl=300)
def coverage_stats() -> dict:
    """Total snapshots, days covered, oldest/newest. Empty defaults on no data."""
    conn = get_connection()
    with conn.cursor() as cur:
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
    with conn.cursor() as cur:
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
        return cur.fetchone()


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
    return pd.read_sql(
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


@st.cache_data(ttl=300)
def chain_at_timestamp(ts) -> pd.DataFrame:
    """Full chain at a specific snapshot timestamp."""
    conn = get_connection()
    return pd.read_sql(
        """
        SELECT * FROM option_chain_snapshots
        WHERE symbol = 'NIFTY' AND snapshot_ts = %s
        ORDER BY expiry, strike, option_type
        """,
        conn,
        params=(ts,),
    )


@st.cache_data(ttl=300)
def recent_snapshot_timestamps(n: int = 10) -> list:
    """Most recent N distinct snapshot timestamps (newest first)."""
    conn = get_connection()
    with conn.cursor() as cur:
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
        return pd.read_sql(query, conn, params=(symbol, str(days)))
    query = """
        SELECT snapshot_ts, spot, expiry, strike, option_type,
               ltp, oi, volume, iv
        FROM option_chain_snapshots
        WHERE symbol = %s
          AND snapshot_ts > NOW() - (%s || ' days')::interval
          AND expiry = %s
        ORDER BY snapshot_ts, strike, option_type
    """
    return pd.read_sql(query, conn, params=(symbol, str(days), expiry))
