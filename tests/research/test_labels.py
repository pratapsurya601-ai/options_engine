"""Tests for engine.research.labels."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from engine.research.labels import (
    label_timestamps,
    _BarIndex,
    _classify,
    DEFAULT_UP,
    DEFAULT_DOWN,
)

IST = timezone(timedelta(hours=5, minutes=30))


def _synthetic_bars(prices: list[float], start: datetime) -> pd.DataFrame:
    """Build a 1-min cadence DataFrame with the given close prices."""
    rows = []
    ts = start
    for p in prices:
        rows.append({
            "ts": pd.Timestamp(ts),
            "open": p, "high": p, "low": p, "close": p, "volume": 0,
        })
        ts = ts + timedelta(minutes=1)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# _classify
# ---------------------------------------------------------------------------

def test_classify_thresholds():
    assert _classify(0.003, DEFAULT_UP, DEFAULT_DOWN) == "UP"
    assert _classify(-0.003, DEFAULT_UP, DEFAULT_DOWN) == "DOWN"
    assert _classify(0.0, DEFAULT_UP, DEFAULT_DOWN) == "NEUTRAL"
    assert _classify(0.0025, DEFAULT_UP, DEFAULT_DOWN) == "UP"     # inclusive
    assert _classify(-0.0025, DEFAULT_UP, DEFAULT_DOWN) == "DOWN"  # inclusive
    assert _classify(0.001, DEFAULT_UP, DEFAULT_DOWN) == "NEUTRAL"
    # NaN passthrough
    out = _classify(float("nan"), DEFAULT_UP, DEFAULT_DOWN)
    assert isinstance(out, float) and np.isnan(out)


# ---------------------------------------------------------------------------
# Regression returns
# ---------------------------------------------------------------------------

def test_ret_15m_known_prices():
    start = datetime(2026, 3, 2, 9, 15, tzinfo=IST)
    # 60 bars, price increases linearly by 1.0 per bar: 100, 101, ... 159
    prices = [100.0 + i for i in range(60)]
    bars = _synthetic_bars(prices, start)
    t = datetime(2026, 3, 2, 9, 30, tzinfo=IST)  # bar at index 15, close=115
    df = label_timestamps([t], symbol="X", horizons_minutes=(15,),
                          include_eod=False, bars_df=bars, tolerance_minutes=0)
    # t+15m -> bar index 30, close=130
    expected = (130.0 - 115.0) / 115.0
    assert df.loc[0, "ret_15m"] == pytest.approx(expected, rel=1e-12)
    assert df.loc[0, "cls_15m"] == "UP"


def test_cls_neutral_and_down():
    start = datetime(2026, 3, 2, 9, 15, tzinfo=IST)
    # Tiny moves -> NEUTRAL
    prices = [1000.0] * 30
    bars = _synthetic_bars(prices, start)
    t = datetime(2026, 3, 2, 9, 30, tzinfo=IST)
    df = label_timestamps([t], horizons_minutes=(15,), include_eod=False,
                          bars_df=bars)
    assert df.loc[0, "ret_15m"] == pytest.approx(0.0)
    assert df.loc[0, "cls_15m"] == "NEUTRAL"

    # Sharp drop -> DOWN
    prices2 = [1000.0 - i * 0.5 for i in range(60)]
    bars2 = _synthetic_bars(prices2, start)
    df2 = label_timestamps([t], horizons_minutes=(15,), include_eod=False,
                           bars_df=bars2, tolerance_minutes=0)
    # t at bar 15 close=992.5; t+15 at bar 30 close=985.0
    expected = (985.0 - 992.5) / 992.5
    assert df2.loc[0, "ret_15m"] == pytest.approx(expected, rel=1e-12)
    assert df2.loc[0, "cls_15m"] == "DOWN"


# ---------------------------------------------------------------------------
# NaN when past last bar
# ---------------------------------------------------------------------------

def test_nan_when_horizon_past_last_bar():
    start = datetime(2026, 3, 2, 9, 15, tzinfo=IST)
    prices = [100.0 + i for i in range(20)]  # last bar at 09:34
    bars = _synthetic_bars(prices, start)
    t = datetime(2026, 3, 2, 9, 30, tzinfo=IST)  # bar at index 15
    # t+15m would be 09:45 — beyond last bar 09:34 -> NaN
    df = label_timestamps([t], horizons_minutes=(15,), include_eod=False,
                          bars_df=bars)
    assert np.isnan(df.loc[0, "ret_15m"])
    assert (df.loc[0, "cls_15m"] is np.nan) or (
        isinstance(df.loc[0, "cls_15m"], float) and np.isnan(df.loc[0, "cls_15m"])
    )


# ---------------------------------------------------------------------------
# Overnight session-cross
# ---------------------------------------------------------------------------

def test_nan_when_session_cross_disallowed():
    # Build two trading days of bars, full session 09:15..15:30 each
    rows = []
    for d in (datetime(2026, 3, 2, tzinfo=IST), datetime(2026, 3, 3, tzinfo=IST)):
        t0 = d.replace(hour=9, minute=15)
        for i in range(376):  # 09:15..15:30 inclusive = 376 minutes
            ts = t0 + timedelta(minutes=i)
            rows.append({"ts": pd.Timestamp(ts),
                         "open": 100.0, "high": 100.0, "low": 100.0,
                         "close": 100.0 + i * 0.01, "volume": 0})
    bars = pd.DataFrame(rows)
    # t=15:25 on day 1; t+60m would be 16:25 day 1 — past session
    t = datetime(2026, 3, 2, 15, 25, tzinfo=IST)
    df = label_timestamps([t], horizons_minutes=(60,), include_eod=True,
                          allow_overnight=False, bars_df=bars)
    assert np.isnan(df.loc[0, "ret_60m"])
    # EOD should be valid (close 15:30 minus close 15:25)
    assert not np.isnan(df.loc[0, "ret_eod"])

    # With allow_overnight=True there is no bar at 16:25 within tolerance,
    # because the next day's bars start at 09:15. So still NaN — but for a
    # DIFFERENT reason (no nearby bar), which is correct behaviour.
    df2 = label_timestamps([t], horizons_minutes=(60,), include_eod=False,
                           allow_overnight=True, bars_df=bars)
    assert np.isnan(df2.loc[0, "ret_60m"])


# ---------------------------------------------------------------------------
# Tolerance window
# ---------------------------------------------------------------------------

def test_tolerance_finds_nearest_bar():
    start = datetime(2026, 3, 2, 9, 15, tzinfo=IST)
    prices = [100.0 + i for i in range(60)]
    bars = _synthetic_bars(prices, start)
    # Ask for 09:30:20 — should match the 09:30 bar (close=115) within 2-min tol
    t = datetime(2026, 3, 2, 9, 30, 20, tzinfo=IST)
    df = label_timestamps([t], horizons_minutes=(15,), include_eod=False,
                          tolerance_minutes=2, bars_df=bars)
    # t+15m -> 09:45:20, nearest bar 09:45 (close=130, 20s away)
    expected = (130.0 - 115.0) / 115.0
    assert df.loc[0, "ret_15m"] == pytest.approx(expected, rel=1e-12)

    # With tolerance 0 it should refuse the off-grid timestamp
    df2 = label_timestamps([t], horizons_minutes=(15,), include_eod=False,
                           tolerance_minutes=0, bars_df=bars)
    assert np.isnan(df2.loc[0, "ret_15m"])


# ---------------------------------------------------------------------------
# EOD edge cases
# ---------------------------------------------------------------------------

def test_eod_at_session_end_is_nan():
    # Full session of bars on one day
    t0 = datetime(2026, 3, 2, 9, 15, tzinfo=IST)
    rows = []
    for i in range(376):
        ts = t0 + timedelta(minutes=i)
        rows.append({"ts": pd.Timestamp(ts),
                     "open": 100.0, "high": 100.0, "low": 100.0,
                     "close": 100.0 + i * 0.1, "volume": 0})
    bars = pd.DataFrame(rows)
    last_t = datetime(2026, 3, 2, 15, 30, tzinfo=IST)
    df = label_timestamps([last_t], horizons_minutes=(15,), include_eod=True,
                          bars_df=bars)
    assert np.isnan(df.loc[0, "ret_eod"])


def test_eod_normal_case():
    t0 = datetime(2026, 3, 2, 9, 15, tzinfo=IST)
    rows = []
    for i in range(376):
        ts = t0 + timedelta(minutes=i)
        rows.append({"ts": pd.Timestamp(ts),
                     "open": 100.0, "high": 100.0, "low": 100.0,
                     "close": 100.0 + i * 0.1, "volume": 0})
    bars = pd.DataFrame(rows)
    t = datetime(2026, 3, 2, 14, 0, tzinfo=IST)
    df = label_timestamps([t], horizons_minutes=(15,), include_eod=True,
                          bars_df=bars)
    # t at minute 285 -> close 128.5; EOD at minute 375 -> close 137.5
    expected = (137.5 - 128.5) / 128.5
    assert df.loc[0, "ret_eod"] == pytest.approx(expected, rel=1e-12)


# ---------------------------------------------------------------------------
# Schema & empties
# ---------------------------------------------------------------------------

def test_empty_input():
    bars = _synthetic_bars([100.0, 101.0, 102.0],
                           datetime(2026, 3, 2, 9, 15, tzinfo=IST))
    df = label_timestamps([], bars_df=bars)
    assert df.empty
    assert "ret_15m" in df.columns and "cls_eod" in df.columns


def test_timestamp_outside_cache_all_nan():
    start = datetime(2026, 3, 2, 9, 15, tzinfo=IST)
    bars = _synthetic_bars([100.0] * 30, start)
    t = datetime(2025, 1, 1, 10, 0, tzinfo=IST)  # before cache
    df = label_timestamps([t], horizons_minutes=(15, 30), include_eod=True,
                          bars_df=bars)
    assert np.isnan(df.loc[0, "ret_15m"])
    assert np.isnan(df.loc[0, "ret_30m"])
    assert np.isnan(df.loc[0, "ret_eod"])


def test_bar_index_same_day_guard():
    """Tolerance lookup must not jump across a date boundary."""
    rows = []
    # Day 1: only a single bar at 15:29
    rows.append({"ts": pd.Timestamp(datetime(2026, 3, 2, 15, 29, tzinfo=IST)),
                 "open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0,
                 "volume": 0})
    # Day 2: bar at 09:15
    rows.append({"ts": pd.Timestamp(datetime(2026, 3, 3, 9, 15, tzinfo=IST)),
                 "open": 200.0, "high": 200.0, "low": 200.0, "close": 200.0,
                 "volume": 0})
    bars = pd.DataFrame(rows)
    idx = _BarIndex.from_df(bars)
    # Query 15:31 on day 1 — nearest in raw ns might be 09:15 next day if we
    # didn't guard. With same-day guard, the 15:29 bar is the only candidate
    # and is within 2-min tolerance.
    target = pd.Timestamp(datetime(2026, 3, 2, 15, 31, tzinfo=IST))
    pos = idx.find_nearest(int(target.value), 2 * 60 * 1_000_000_000)
    assert pos == 0
    # Query far from any same-day bar -> None
    far = pd.Timestamp(datetime(2026, 3, 2, 10, 0, tzinfo=IST))
    assert idx.find_nearest(int(far.value), 2 * 60 * 1_000_000_000) is None
