"""Tests for streamlit_lib.db with mocked psycopg.connect.

These cover the shape contract (return types, empty-table handling) without
needing an actual Aiven connection.
"""
from __future__ import annotations

import datetime as dt
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_streamlit_caches():
    """Clear st.cache_data / st.cache_resource between tests so each test
    runs against a fresh mocked connection."""
    import streamlit as st
    try:
        st.cache_data.clear()
    except Exception:
        pass
    try:
        st.cache_resource.clear()
    except Exception:
        pass
    yield
    try:
        st.cache_data.clear()
    except Exception:
        pass
    try:
        st.cache_resource.clear()
    except Exception:
        pass


def _make_conn(fetchone_value=None, fetchall_value=None):
    """Build a mock psycopg connection whose cursor returns canned values."""
    cur = MagicMock()
    cur.__enter__.return_value = cur
    cur.__exit__.return_value = False
    cur.fetchone.return_value = fetchone_value
    cur.fetchall.return_value = fetchall_value or []
    cur.execute.return_value = None
    conn = MagicMock()
    conn.cursor.return_value = cur
    return conn


# -----------------------------------------------------------------------------
# coverage_stats
# -----------------------------------------------------------------------------


def test_coverage_stats_shape(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgres://fake")
    conn = _make_conn(fetchone_value={
        "total_rows": 1234,
        "total_snapshots": 73,
        "days_covered": 4,
        "oldest": dt.datetime(2026, 5, 28, 9, 15),
        "newest": dt.datetime(2026, 6, 2, 15, 25),
    })
    with patch("psycopg.connect", return_value=conn):
        from streamlit_lib import db
        out = db.coverage_stats()

    assert set(out.keys()) >= {"total_rows", "total_snapshots", "days_covered",
                               "oldest", "newest"}
    assert out["total_rows"] == 1234
    assert out["total_snapshots"] == 73
    assert out["days_covered"] == 4
    assert out["newest"].year == 2026


def test_coverage_stats_empty_table(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgres://fake")
    # Empty table: counts come back as 0/None
    conn = _make_conn(fetchone_value={
        "total_rows": 0,
        "total_snapshots": 0,
        "days_covered": 0,
        "oldest": None,
        "newest": None,
    })
    with patch("psycopg.connect", return_value=conn):
        from streamlit_lib import db
        out = db.coverage_stats()

    assert out["total_rows"] == 0
    assert out["total_snapshots"] == 0
    assert out["days_covered"] == 0
    assert out["oldest"] is None
    assert out["newest"] is None


def test_coverage_stats_no_row(monkeypatch):
    """psycopg returns None for fetchone on truly empty results."""
    monkeypatch.setenv("DATABASE_URL", "postgres://fake")
    conn = _make_conn(fetchone_value=None)
    with patch("psycopg.connect", return_value=conn):
        from streamlit_lib import db
        out = db.coverage_stats()

    assert out["total_rows"] == 0
    assert out["total_snapshots"] == 0


# -----------------------------------------------------------------------------
# latest_snapshot_summary
# -----------------------------------------------------------------------------


def test_latest_snapshot_summary_returns_dict(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgres://fake")
    conn = _make_conn(fetchone_value={
        "snapshot_ts": dt.datetime(2026, 6, 2, 15, 25),
        "spot": 24750.0,
        "expiry": dt.date(2026, 6, 5),
        "n_strikes": 80,
    })
    with patch("psycopg.connect", return_value=conn):
        from streamlit_lib import db
        out = db.latest_snapshot_summary()

    assert out is not None
    assert out["spot"] == 24750.0
    assert out["n_strikes"] == 80


def test_latest_snapshot_summary_empty(monkeypatch):
    """Empty table -> None, NOT a crash."""
    monkeypatch.setenv("DATABASE_URL", "postgres://fake")
    conn = _make_conn(fetchone_value=None)
    with patch("psycopg.connect", return_value=conn):
        from streamlit_lib import db
        out = db.latest_snapshot_summary()
    assert out is None


# -----------------------------------------------------------------------------
# snapshots_per_day (uses pd.read_sql)
# -----------------------------------------------------------------------------


def test_snapshots_per_day_returns_dataframe(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgres://fake")
    fake_df = pd.DataFrame({
        "date": [dt.date(2026, 5, 28), dt.date(2026, 5, 29)],
        "n_snapshots": [17, 18],
    })
    conn = _make_conn()
    with patch("psycopg.connect", return_value=conn), \
         patch("pandas.read_sql", return_value=fake_df) as mock_rs:
        from streamlit_lib import db
        out = db.snapshots_per_day()

    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == ["date", "n_snapshots"]
    assert len(out) == 2
    assert mock_rs.called


def test_snapshots_per_day_empty(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgres://fake")
    empty_df = pd.DataFrame({"date": [], "n_snapshots": []})
    conn = _make_conn()
    with patch("psycopg.connect", return_value=conn), \
         patch("pandas.read_sql", return_value=empty_df):
        from streamlit_lib import db
        out = db.snapshots_per_day()

    assert isinstance(out, pd.DataFrame)
    assert out.empty


# -----------------------------------------------------------------------------
# DATABASE_URL resolution
# -----------------------------------------------------------------------------


def test_get_database_url_falls_back_to_env(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgres://from-env")
    from streamlit_lib import db
    assert db._get_database_url() == "postgres://from-env"


def test_get_database_url_missing_raises(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    from streamlit_lib import db
    # No secrets file in test env, no env var -> empty string
    url = db._get_database_url()
    assert url == ""
    # And get_connection should refuse
    with pytest.raises(RuntimeError):
        db.get_connection()
