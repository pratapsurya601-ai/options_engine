"""Tests for engine.research.conditional_edge."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from engine.research.conditional_edge import (
    analyze_conditional,
    rank_regime_specific,
    analyze_dataset_conditional,
    join_features_labels_regimes,
    _load_regime_timeline_for_join,
    OVERALL_LABEL,
)
from engine.research.regimes import REGIMES


IST = timezone(timedelta(hours=5, minutes=30))


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def _make_synth_regime_split(
    n_per_regime: int,
    signal_regime: str,
    noise_regime: str,
    seed: int = 0,
) -> pd.DataFrame:
    """Half samples in `signal_regime` (strong x->y), half in `noise_regime`
    (y independent of x). Includes a timestamp column for completeness."""
    rng = _rng(seed)
    x_sig = rng.standard_normal(n_per_regime)
    y_sig = 0.6 * x_sig + 0.1 * rng.standard_normal(n_per_regime)

    x_noise = rng.standard_normal(n_per_regime)
    y_noise = rng.standard_normal(n_per_regime)  # independent of x

    feature = np.concatenate([x_sig, x_noise])
    label = np.concatenate([y_sig, y_noise])
    regimes = [signal_regime] * n_per_regime + [noise_regime] * n_per_regime
    base = datetime(2026, 1, 1, 9, 15, tzinfo=IST)
    timestamps = [base + timedelta(minutes=i) for i in range(len(feature))]

    return pd.DataFrame({
        "timestamp": timestamps,
        "regime": regimes,
        "feat": feature,
        "ret_60m": label,
    })


# ---------------------------------------------------------------------------
# Test 1: regime-specific signal -- strong in TREND, noise in MEAN_REVERSION
# ---------------------------------------------------------------------------


def test_signal_only_in_one_regime_flagged_specific():
    df = _make_synth_regime_split(
        n_per_regime=400,
        signal_regime="TREND",
        noise_regime="MEAN_REVERSION",
        seed=42,
    )
    results = analyze_conditional(df, feature="feat", horizon="ret_60m",
                                  min_samples_per_regime=30)
    # 5 rows: OVERALL + 4 regimes
    assert len(results) == 5
    by_regime = {r.regime: r for r in results}

    assert OVERALL_LABEL in by_regime
    for r in REGIMES:
        assert r in by_regime

    trend = by_regime["TREND"]
    mr = by_regime["MEAN_REVERSION"]

    assert trend.n == 400
    assert mr.n == 400
    assert trend.ic > 0.3, f"expected strong TREND IC, got {trend.ic}"
    assert abs(mr.ic) < 0.2, f"expected weak MEAN_REVERSION IC, got {mr.ic}"

    # TREND should be flagged regime-specific (delta IC big enough)
    assert trend.is_regime_specific is True
    # Empty regimes (PANIC, LOW_VOL) have n=0 -> not specific
    assert by_regime["PANIC"].n == 0
    assert by_regime["PANIC"].is_regime_specific is False
    assert by_regime["LOW_VOL"].n == 0


# ---------------------------------------------------------------------------
# Test 2: signal works equally in both regimes -> not regime-specific
# ---------------------------------------------------------------------------


def test_uniform_signal_not_regime_specific():
    rng = _rng(7)
    n = 400
    x = rng.standard_normal(2 * n)
    y = 0.5 * x + 0.1 * rng.standard_normal(2 * n)  # same model in both regimes
    regimes = ["TREND"] * n + ["MEAN_REVERSION"] * n
    base = datetime(2026, 1, 1, 9, 15, tzinfo=IST)
    timestamps = [base + timedelta(minutes=i) for i in range(2 * n)]
    df = pd.DataFrame({
        "timestamp": timestamps, "regime": regimes,
        "feat": x, "ret_60m": y,
    })

    results = analyze_conditional(df, feature="feat", horizon="ret_60m",
                                  min_samples_per_regime=30)
    by_regime = {r.regime: r for r in results}

    # delta IC should be small for both populated regimes
    trend = by_regime["TREND"]
    mr = by_regime["MEAN_REVERSION"]
    assert abs(trend.delta_ic_vs_overall) < 0.1
    assert abs(mr.delta_ic_vs_overall) < 0.1
    assert trend.is_regime_specific is False
    assert mr.is_regime_specific is False


# ---------------------------------------------------------------------------
# Test 3: insufficient samples in a regime -> INSUFFICIENT, not specific
# ---------------------------------------------------------------------------


def test_insufficient_samples_not_regime_specific():
    rng = _rng(3)
    # 5 PANIC, 400 MEAN_REVERSION
    n_panic = 5
    n_mr = 400
    x = np.concatenate([rng.standard_normal(n_panic),
                        rng.standard_normal(n_mr)])
    # Make PANIC look like strong signal but tiny n
    y_panic = 0.9 * x[:n_panic] + 0.05 * rng.standard_normal(n_panic)
    y_mr = 0.05 * rng.standard_normal(n_mr)  # noise
    y = np.concatenate([y_panic, y_mr])
    regimes = ["PANIC"] * n_panic + ["MEAN_REVERSION"] * n_mr
    base = datetime(2026, 1, 1, 9, 15, tzinfo=IST)
    timestamps = [base + timedelta(minutes=i) for i in range(len(x))]
    df = pd.DataFrame({
        "timestamp": timestamps, "regime": regimes,
        "feat": x, "ret_60m": y,
    })

    results = analyze_conditional(df, feature="feat", horizon="ret_60m",
                                  min_samples_per_regime=30)
    by_regime = {r.regime: r for r in results}

    panic = by_regime["PANIC"]
    assert panic.n == n_panic
    assert panic.confidence_band == "INSUFFICIENT"
    # Even if delta IC looks huge, sample size guard rejects it
    assert panic.is_regime_specific is False


# ---------------------------------------------------------------------------
# Test 4: intraday timestamps inherit the day's regime
# ---------------------------------------------------------------------------


def test_intraday_inherits_daily_regime(monkeypatch, tmp_path):
    # Build a fake labeled wide frame: 3 intraday timestamps on a single day,
    # plus a fake daily regime timeline assigning TREND to that day.
    day = datetime(2026, 6, 2, tzinfo=IST).date()
    intraday = [
        datetime(2026, 6, 2, 9, 30, tzinfo=IST),
        datetime(2026, 6, 2, 11, 0, tzinfo=IST),
        datetime(2026, 6, 2, 14, 0, tzinfo=IST),
    ]
    labeled = pd.DataFrame({
        "timestamp": intraday,
        "symbol": ["NIFTY"] * 3,
        "feat": [1.0, 2.0, 3.0],
        "ret_60m": [0.001, 0.002, -0.001],
    })

    fake_regimes = pd.DataFrame({"date": [day], "regime": ["TREND"]})

    monkeypatch.setattr(
        "engine.research.conditional_edge._load_labeled_wide",
        lambda symbol: labeled.copy(),
    )
    monkeypatch.setattr(
        "engine.research.conditional_edge._load_regime_timeline_for_join",
        lambda symbol: fake_regimes.copy(),
    )

    out = join_features_labels_regimes("NIFTY")
    assert len(out) == 3
    assert (out["regime"] == "TREND").all()
    assert list(out["feat"]) == [1.0, 2.0, 3.0]


# ---------------------------------------------------------------------------
# Test 5: missing regime drops the row (does NOT fabricate)
# ---------------------------------------------------------------------------


def test_missing_regime_drops_row(monkeypatch):
    day_known = datetime(2026, 6, 2, tzinfo=IST).date()
    rows = [
        datetime(2026, 6, 2, 10, 0, tzinfo=IST),    # known regime
        datetime(2026, 6, 3, 10, 0, tzinfo=IST),    # NO regime -- drop
        datetime(2026, 6, 2, 14, 0, tzinfo=IST),    # known regime
    ]
    labeled = pd.DataFrame({
        "timestamp": rows,
        "symbol": ["NIFTY"] * 3,
        "feat": [1.0, 9.9, 2.0],
        "ret_60m": [0.001, 0.999, 0.002],
    })
    fake_regimes = pd.DataFrame({"date": [day_known], "regime": ["PANIC"]})

    monkeypatch.setattr(
        "engine.research.conditional_edge._load_labeled_wide",
        lambda symbol: labeled.copy(),
    )
    monkeypatch.setattr(
        "engine.research.conditional_edge._load_regime_timeline_for_join",
        lambda symbol: fake_regimes.copy(),
    )

    out = join_features_labels_regimes("NIFTY")
    assert len(out) == 2
    assert (out["regime"] == "PANIC").all()
    # The row at 2026-06-03 with feat=9.9 must be gone
    assert 9.9 not in list(out["feat"])
    # And the surviving ones are exactly the known-regime ones
    assert sorted(out["feat"].tolist()) == [1.0, 2.0]


# ---------------------------------------------------------------------------
# Bonus: rank_regime_specific sorts by |delta_ic|
# ---------------------------------------------------------------------------


def test_rank_regime_specific_sorts_by_abs_delta():
    df = pd.DataFrame({
        "feature": ["a", "b", "c", "d"],
        "horizon": ["ret_60m"] * 4,
        "regime": ["TREND", "TREND", "PANIC", "TREND"],
        "n": [100, 100, 100, 100],
        "ic": [0.1, 0.2, -0.3, 0.05],
        "pearson": [0.0] * 4,
        "mutual_info": [0.0] * 4,
        "decile_spread": [0.0] * 4,
        "composite_score": [0.0] * 4,
        "delta_ic_vs_overall": [0.07, -0.15, 0.25, 0.02],
        "is_regime_specific": [True, True, True, False],
        "confidence_band": ["LOW"] * 4,
    })
    ranked = rank_regime_specific(df, horizon="ret_60m", top_k=10)
    # 'd' filtered out (is_regime_specific=False); rest sorted by abs delta desc
    assert ranked["feature"].tolist() == ["c", "b", "a"]
