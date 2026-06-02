"""Tests for engine.research.flow_features.

Run:
    C:\\ProgramData\\miniconda3\\python.exe -m pytest tests/research/test_flow_features.py -v
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from engine.research import flow_features as ff


IST = timezone(timedelta(hours=5, minutes=30))


def _make_chain(
    snapshot_ts: datetime,
    spot: float = 25000.0,
    strike_step: int = 50,
    n_each_side: int = 5,
    expiry: str = "2026-06-25",
    symbol: str = "NIFTY",
    ce_oi_at: dict[float, float] | None = None,
    pe_oi_at: dict[float, float] | None = None,
    ce_vol_at: dict[float, float] | None = None,
    pe_vol_at: dict[float, float] | None = None,
    iv_base: float = 0.15,
) -> pd.DataFrame:
    """Synthetic 10-strike chain (5 ITM + 5 OTM) for both CE and PE."""
    atm = round(spot / strike_step) * strike_step
    strikes = [atm + i * strike_step for i in range(-n_each_side, n_each_side + 1)]
    rows = []
    for k in strikes:
        # Default OI rises near ATM, falls away
        dist = abs(k - atm) / strike_step
        ce_oi = (ce_oi_at or {}).get(k, max(1000.0, 5000.0 - dist * 600.0))
        pe_oi = (pe_oi_at or {}).get(k, max(1000.0, 5000.0 - dist * 600.0))
        ce_vol = (ce_vol_at or {}).get(k, ce_oi * 0.1)
        pe_vol = (pe_vol_at or {}).get(k, pe_oi * 0.1)
        # Make IV smile-ish so iv_skew_25d has something to chew on
        iv = iv_base + 0.001 * dist
        rows.append({
            "snapshot_ts": snapshot_ts,
            "symbol": symbol, "spot": spot, "expiry": expiry,
            "strike": float(k), "option_type": "CE",
            "ltp": 50.0, "bid": 49.0, "ask": 51.0,
            "iv": iv, "oi": ce_oi, "volume": ce_vol,
        })
        rows.append({
            "snapshot_ts": snapshot_ts,
            "symbol": symbol, "spot": spot, "expiry": expiry,
            "strike": float(k), "option_type": "PE",
            "ltp": 50.0, "bid": 49.0, "ask": 51.0,
            "iv": iv, "oi": pe_oi, "volume": pe_vol,
        })
    return pd.DataFrame(rows)


@pytest.fixture
def snap_basic():
    return _make_chain(datetime(2026, 6, 2, 10, 0, 0, tzinfo=IST))


def _val(feats: pd.DataFrame, name: str) -> float:
    sub = feats[feats["feature_name"] == name]
    assert len(sub) == 1, f"expected single row for {name}, got {len(sub)}"
    return float(sub["feature_value"].iloc[0])


def test_pcr_oi_computes_correctly(snap_basic):
    feats = ff.compute_features_for_snapshot(snap_basic)
    ce_total = snap_basic.loc[snap_basic["option_type"] == "CE", "oi"].sum()
    pe_total = snap_basic.loc[snap_basic["option_type"] == "PE", "oi"].sum()
    assert _val(feats, "pcr_oi") == pytest.approx(pe_total / ce_total)
    assert _val(feats, "total_call_oi") == pytest.approx(ce_total)
    assert _val(feats, "total_put_oi") == pytest.approx(pe_total)


def test_max_pain_on_crafted_example():
    """Craft a chain where all OI sits on one strike. Max pain == that strike."""
    ts = datetime(2026, 6, 2, 10, 0, 0, tzinfo=IST)
    df = _make_chain(
        ts, spot=25000.0,
        ce_oi_at={k: 10.0 for k in [24800, 24850, 24900, 24950, 25000, 25050, 25100, 25150, 25200]},
        pe_oi_at={k: 10.0 for k in [24800, 24850, 24900, 24950, 25000, 25050, 25100, 25150, 25200]},
    )
    # Now overload one strike heavily
    df.loc[(df["strike"] == 24900) & (df["option_type"] == "CE"), "oi"] = 1_000_000
    df.loc[(df["strike"] == 24900) & (df["option_type"] == "PE"), "oi"] = 1_000_000
    assert ff.max_pain(df) == pytest.approx(24900.0)


def test_walls_return_max_oi_strikes(snap_basic):
    snap_basic.loc[
        (snap_basic["strike"] == 25100.0) & (snap_basic["option_type"] == "CE"),
        "oi",
    ] = 99_999_999.0
    snap_basic.loc[
        (snap_basic["strike"] == 24800.0) & (snap_basic["option_type"] == "PE"),
        "oi",
    ] = 99_999_999.0
    feats = ff.compute_features_for_snapshot(snap_basic)
    assert _val(feats, "call_wall") == pytest.approx(25100.0)
    assert _val(feats, "put_wall") == pytest.approx(24800.0)


def test_oi_delta_total_call_is_difference():
    t0 = datetime(2026, 6, 2, 10, 0, 0, tzinfo=IST)
    t1 = t0 + timedelta(minutes=5)
    prev = _make_chain(t0)
    cur = _make_chain(t1)
    # bump current CE OI uniformly by +500 per strike
    cur.loc[cur["option_type"] == "CE", "oi"] += 500.0
    n_ce_strikes = (cur["option_type"] == "CE").sum()
    feats = ff.compute_features_for_snapshot(cur, prev_snapshot_df=prev)
    assert _val(feats, "oi_delta_total_call") == pytest.approx(500.0 * n_ce_strikes)
    # PE unchanged
    assert _val(feats, "oi_delta_total_put") == pytest.approx(0.0)


def test_writing_and_unwinding_velocities_partition_changes():
    t0 = datetime(2026, 6, 2, 10, 0, 0, tzinfo=IST)
    t1 = t0 + timedelta(minutes=1)  # 1 minute -> velocity == raw delta
    prev = _make_chain(t0)
    cur = _make_chain(t1)
    # One CE strike +1000 (writing), another CE strike -400 (unwinding)
    cur.loc[(cur["strike"] == 25000.0) & (cur["option_type"] == "CE"), "oi"] += 1000.0
    cur.loc[(cur["strike"] == 25100.0) & (cur["option_type"] == "CE"), "oi"] -= 400.0
    # PE: +700 at one strike, -200 at another
    cur.loc[(cur["strike"] == 24900.0) & (cur["option_type"] == "PE"), "oi"] += 700.0
    cur.loc[(cur["strike"] == 24800.0) & (cur["option_type"] == "PE"), "oi"] -= 200.0
    feats = ff.compute_features_for_snapshot(cur, prev_snapshot_df=prev)
    assert _val(feats, "call_writing_velocity") == pytest.approx(1000.0)
    assert _val(feats, "call_unwinding_velocity") == pytest.approx(400.0)
    assert _val(feats, "put_writing_velocity") == pytest.approx(700.0)
    assert _val(feats, "put_unwinding_velocity") == pytest.approx(200.0)


def test_missing_prev_snapshot_returns_only_pit_features(snap_basic):
    feats = ff.compute_features_for_snapshot(snap_basic, prev_snapshot_df=None)
    names = set(feats["feature_name"])
    # Should not raise, and should not include inter-snapshot features:
    for inter in (
        "oi_delta_total_call", "oi_delta_total_put",
        "call_writing_velocity", "put_writing_velocity",
        "max_pain_migration",
    ):
        assert inter not in names
    # And should include core PIT features
    for pit in ("pcr_oi", "max_pain", "call_wall", "put_wall", "atm_iv"):
        assert pit in names
    # No NaN errors -- pcr_oi must be a finite positive number
    assert np.isfinite(_val(feats, "pcr_oi"))


def test_oi_acceleration_requires_two_prev_snapshots():
    t0 = datetime(2026, 6, 2, 10, 0, 0, tzinfo=IST)
    t1 = t0 + timedelta(minutes=5)
    t2 = t0 + timedelta(minutes=10)
    s0 = _make_chain(t0)
    s1 = _make_chain(t1)
    s2 = _make_chain(t2)
    # Make CE total go up by +1000 step1, then +3000 step2 -> accel = +2000
    s1.loc[s1["option_type"] == "CE", "oi"] += 1000.0 / (s1["option_type"] == "CE").sum()
    s2.loc[s2["option_type"] == "CE", "oi"] += 4000.0 / (s2["option_type"] == "CE").sum()
    feats = ff.compute_features_for_snapshot(
        s2, prev_snapshot_df=s1, prev_prev_snapshot_df=s0,
    )
    # d1 - d0 = 3000 - 1000 = 2000 (approx within float)
    assert _val(feats, "oi_acceleration_call") == pytest.approx(2000.0, rel=1e-6)


def test_output_schema(snap_basic):
    feats = ff.compute_features_for_snapshot(snap_basic)
    assert list(feats.columns) == ["timestamp", "symbol", "expiry",
                                   "feature_name", "feature_value"]
    assert (feats["symbol"] == "NIFTY").all()
    assert (feats["expiry"] == "2026-06-25").all()
    # All feature_values are floats (or NaN), never None
    assert feats["feature_value"].dtype.kind == "f"


def test_malformed_snapshot_returns_empty():
    bad = pd.DataFrame({"foo": [1, 2, 3]})
    with pytest.warns(UserWarning):
        out = ff.compute_features_for_snapshot(bad)
    assert out.empty
    assert list(out.columns) == ["timestamp", "symbol", "expiry",
                                 "feature_name", "feature_value"]
