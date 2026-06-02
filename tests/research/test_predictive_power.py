"""Tests for engine.research.predictive_power."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from engine.research.predictive_power import (
    analyze_feature,
    analyze_dataset,
    rank_features,
    FeaturePredictiveScore,
    _composite_score,
    _confidence_band,
    DEFAULT_HORIZONS,
)


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# Test 1: strong synthetic signal -> high IC + high composite score
# ---------------------------------------------------------------------------


def test_strong_signal_high_composite():
    rng = _rng(0)
    n = 1500
    x = rng.standard_normal(n)
    y = 0.5 * x + 0.1 * rng.standard_normal(n)  # high SNR
    score = analyze_feature(pd.Series(x), pd.Series(y),
                            feature_name="x", horizon="ret_60m")
    assert score.n == n
    assert abs(score.ic) > 0.5
    assert abs(score.pearson) > 0.5
    assert score.composite_score > 0.4
    assert score.is_significant_5pct is True
    # Big n + significant + |IC| > 0.05 should land HIGH
    assert score.confidence_band == "HIGH"


# ---------------------------------------------------------------------------
# Test 2: pure noise -> low composite, not significant
# ---------------------------------------------------------------------------


def test_noise_low_composite_not_significant():
    rng = _rng(42)
    n = 500
    x = rng.standard_normal(n)
    # Scale y to realistic intraday-return magnitudes so the decile-spread
    # cap (20 bps) is meaningful. With y ~ N(0, 1) the random spread would
    # dwarf any honest signal-vs-noise check.
    y = 0.001 * rng.standard_normal(n)
    score = analyze_feature(pd.Series(x), pd.Series(y),
                            feature_name="x", horizon="ret_60m")
    assert score.composite_score < 0.15
    # With pure noise we'd be unlucky to get p<0.05 -- run is deterministic
    assert score.is_significant_5pct is False


# ---------------------------------------------------------------------------
# Test 3: constant feature -> NaN metrics + warning
# ---------------------------------------------------------------------------


def test_constant_feature_warns_and_nans():
    n = 100
    x = pd.Series(np.ones(n))
    y = pd.Series(_rng(1).standard_normal(n))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        score = analyze_feature(x, y, feature_name="const", horizon="ret_60m")
    assert any("constant" in str(w.message).lower() for w in caught)
    assert np.isnan(score.pearson)
    assert np.isnan(score.ic)
    assert np.isnan(score.composite_score)
    assert score.confidence_band == "INSUFFICIENT"


# ---------------------------------------------------------------------------
# Test 4: monotone feature -> positive decile spread
# ---------------------------------------------------------------------------


def test_monotone_decile_spread_positive():
    n = 1000
    x = np.linspace(-1, 1, n)
    y = x + 0.05 * _rng(7).standard_normal(n)
    score = analyze_feature(pd.Series(x), pd.Series(y),
                            feature_name="lin", horizon="ret_60m")
    assert score.decile_spread > 0
    assert score.decile_means[-1] > score.decile_means[0]


# ---------------------------------------------------------------------------
# Test 5: AUC ~0.9+ when feature perfectly separates UP/DOWN
# ---------------------------------------------------------------------------


def test_auc_high_when_feature_separates_classes():
    n = 400
    rng = _rng(11)
    cls = np.where(rng.uniform(size=n) > 0.5, "UP", "DOWN")
    # Feature shifted up for UP, down for DOWN
    x = np.where(cls == "UP", 1.0, -1.0) + 0.2 * rng.standard_normal(n)
    y = np.where(cls == "UP", 0.01, -0.01) + 0.001 * rng.standard_normal(n)
    score = analyze_feature(
        feature_values=pd.Series(x),
        label_values=pd.Series(y),
        cls_labels=pd.Series(cls),
        feature_name="sep", horizon="ret_60m",
    )
    assert score.auc_up_vs_down is not None
    assert score.auc_up_vs_down > 0.9
    assert score.anova_f is not None
    assert score.anova_f > 100  # well-separated groups


# ---------------------------------------------------------------------------
# Test 6: very small n -> INSUFFICIENT band
# ---------------------------------------------------------------------------


def test_small_n_insufficient_band():
    n = 10
    rng = _rng(2)
    x = pd.Series(rng.standard_normal(n))
    y = pd.Series(rng.standard_normal(n))
    score = analyze_feature(x, y, feature_name="tiny", horizon="ret_60m",
                            min_samples=30)
    assert score.confidence_band == "INSUFFICIENT"


# ---------------------------------------------------------------------------
# Test 7: NaN handling -- report correct nan_pct and n
# ---------------------------------------------------------------------------


def test_nan_handling_reports_correct_pct():
    n = 400
    rng = _rng(3)
    x = rng.standard_normal(n)
    y = 0.3 * x + 0.5 * rng.standard_normal(n)
    # Inject 50% NaN into feature
    mask = rng.uniform(size=n) < 0.5
    xs = pd.Series(x.copy())
    xs[mask] = np.nan
    score = analyze_feature(xs, pd.Series(y),
                            feature_name="half_nan", horizon="ret_60m")
    assert 0.4 < score.nan_pct < 0.6
    assert score.n == int((~mask).sum())
    assert np.isfinite(score.composite_score)


# ---------------------------------------------------------------------------
# Test 8: analyze_dataset processes multiple features correctly
# ---------------------------------------------------------------------------


def test_analyze_dataset_multi_feature():
    n = 600
    rng = _rng(5)
    x_good = rng.standard_normal(n)
    x_noise = rng.standard_normal(n)
    y_60 = 0.4 * x_good + 0.3 * rng.standard_normal(n)
    y_15 = 0.05 * x_good + 0.3 * rng.standard_normal(n)
    cls_60 = np.where(y_60 > 0.0025, "UP",
              np.where(y_60 < -0.0025, "DOWN", "NEUTRAL"))
    cls_15 = np.where(y_15 > 0.0025, "UP",
              np.where(y_15 < -0.0025, "DOWN", "NEUTRAL"))
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-05-01", periods=n, freq="min"),
        "x_good": x_good,
        "x_noise": x_noise,
        "ret_15m": y_15, "cls_15m": cls_15,
        "ret_60m": y_60, "cls_60m": cls_60,
    })
    scores = analyze_dataset(df, horizons=("ret_15m", "ret_60m"))
    # Should be 2 features x 2 horizons = 4 rows
    assert len(scores) == 4
    assert set(scores["feature"]) == {"x_good", "x_noise"}
    assert set(scores["horizon"]) == {"ret_15m", "ret_60m"}
    # x_good @ 60m should beat x_noise @ 60m on composite
    good_60 = scores[(scores["feature"] == "x_good")
                     & (scores["horizon"] == "ret_60m")
                     ]["composite_score"].iloc[0]
    noise_60 = scores[(scores["feature"] == "x_noise")
                      & (scores["horizon"] == "ret_60m")
                      ]["composite_score"].iloc[0]
    assert good_60 > noise_60

    # rank_features sanity
    top = rank_features(scores, horizon="ret_60m", top_k=2)
    assert top.iloc[0]["feature"] == "x_good"


# ---------------------------------------------------------------------------
# Direct unit tests for helpers
# ---------------------------------------------------------------------------


def test_composite_score_bounds():
    # Max-out everything
    val = _composite_score(ic=1.0, pearson=1.0, mi=1.0, decile_spread=0.01)
    assert 0.99 <= val <= 1.0 + 1e-9
    # All NaN -> 0
    val = _composite_score(ic=float("nan"), pearson=float("nan"),
                           mi=float("nan"), decile_spread=float("nan"))
    assert val == 0.0


def test_confidence_band_thresholds():
    assert _confidence_band(n=10, pearson_p=0.001, ic=0.5,
                            min_samples=30) == "INSUFFICIENT"
    assert _confidence_band(n=2000, pearson_p=0.001, ic=0.10,
                            min_samples=30) == "HIGH"
    assert _confidence_band(n=500, pearson_p=0.10, ic=0.10,
                            min_samples=30) == "MEDIUM"
    assert _confidence_band(n=500, pearson_p=0.50, ic=0.10,
                            min_samples=30) == "LOW"
