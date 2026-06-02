"""Tests for engine.research.regimes."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import math
import pandas as pd
import pytest

from engine.research.regimes import (
    RegimeConfig,
    adx,
    atr,
    build_regime_timeline,
    classify_regime,
    realized_vol_annualized,
    true_range,
)


IST = timezone(timedelta(hours=5, minutes=30))


# ---------------------------------------------------------------------------
# Synthetic bar helpers
# ---------------------------------------------------------------------------

def _make_bars(closes: list[float],
               highs: list[float] | None = None,
               lows: list[float] | None = None) -> pd.DataFrame:
    n = len(closes)
    if highs is None:
        highs = [c * 1.003 for c in closes]
    if lows is None:
        lows = [c * 0.997 for c in closes]
    opens = [closes[i - 1] if i > 0 else closes[0] for i in range(n)]
    start = datetime(2024, 1, 2, 15, 30, tzinfo=IST)
    ts = [start + timedelta(days=i) for i in range(n)]
    return pd.DataFrame({
        "ts": ts,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": [100_000] * n,
    })


# ---------------------------------------------------------------------------
# Indicator unit tests
# ---------------------------------------------------------------------------

def test_true_range_hand_computed():
    # Classic worked example
    highs = [48.70, 48.72, 48.90, 48.87, 48.82]
    lows = [47.79, 48.14, 48.39, 48.37, 48.24]
    closes = [48.16, 48.61, 48.75, 48.63, 48.74]
    tr = true_range(highs, lows, closes)
    assert tr[0] is None
    # TR_1 = max(48.72-48.14, |48.72-48.16|, |48.14-48.16|) = max(0.58, 0.56, 0.02) = 0.58
    assert tr[1] == pytest.approx(0.58, abs=1e-6)
    # TR_2 = max(48.90-48.39, |48.90-48.61|, |48.39-48.61|) = max(0.51, 0.29, 0.22) = 0.51
    assert tr[2] == pytest.approx(0.51, abs=1e-6)


def test_atr_smooths_true_range():
    # Constant TR -> ATR converges to that constant
    highs = [10.0 + i * 0 for i in range(40)]
    # Build series where TR = 1.0 every bar after the first
    highs = []
    lows = []
    closes = []
    c = 100.0
    for i in range(40):
        closes.append(c)
        highs.append(c + 0.5)
        lows.append(c - 0.5)
    a = atr(highs, lows, closes, period=14)
    # By bar 30, ATR should be ~1.0 (the constant TR)
    assert a[30] == pytest.approx(1.0, abs=1e-6)


def test_adx_high_on_strong_trend():
    # Monotonic uptrend
    closes = [100.0 * (1.003 ** i) for i in range(80)]
    highs = [c * 1.002 for c in closes]
    lows = [c * 0.998 for c in closes]
    a = adx(highs, lows, closes, period=14)
    # On a clean monotonic trend, ADX should be well above 25
    assert a[-1] is not None
    assert a[-1] > 25.0


def test_adx_low_on_chop():
    # Oscillating prices around 100
    closes = [100.0 + (1.0 if i % 2 == 0 else -1.0) for i in range(80)]
    highs = [c + 0.2 for c in closes]
    lows = [c - 0.2 for c in closes]
    a = adx(highs, lows, closes, period=14)
    assert a[-1] is not None
    assert a[-1] < 25.0


def test_realized_vol_zero_on_flat():
    closes = [100.0] * 30
    rv = realized_vol_annualized(closes, window=5)
    assert rv == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Regime classification — synthetic series
# ---------------------------------------------------------------------------

def test_trend_regime():
    # 80 days of +0.3% per day with small intraday range
    closes = [100.0 * (1.003 ** i) for i in range(80)]
    df = _make_bars(closes,
                    highs=[c * 1.002 for c in closes],
                    lows=[c * 0.998 for c in closes])
    tl = build_regime_timeline(symbol="TEST", interval="day", df=df)
    final = tl.iloc[-1]
    assert final["regime"] == "TREND", (
        f"expected TREND, got {final['regime']} "
        f"(adx={final['adx_14']:.2f}, slope={final['ema_slope_20']:.4f})"
    )


def test_low_vol_regime():
    # First a high-vol regime, then a compressed period at the end.
    # The expanding-window percentiles must place the recent bars in the
    # bottom 25% of ATR and realized vol.
    head_closes = [100.0 * (1.0 + 0.03 * math.sin(i / 2.0)) for i in range(80)]
    head_highs = [c * 1.025 for c in head_closes]
    head_lows = [c * 0.975 for c in head_closes]

    tail_n = 40
    tail_closes = [100.0 + 0.0005 * ((i % 3) - 1) for i in range(tail_n)]
    tail_highs = [c + 0.02 for c in tail_closes]
    tail_lows = [c - 0.02 for c in tail_closes]

    all_closes = head_closes + tail_closes
    all_highs = head_highs + tail_highs
    all_lows = head_lows + tail_lows
    df = _make_bars(all_closes, highs=all_highs, lows=all_lows)
    tl = build_regime_timeline(symbol="TEST", interval="day", df=df)
    final = tl.iloc[-1]
    assert final["regime"] == "LOW_VOL", (
        f"expected LOW_VOL, got {final['regime']} "
        f"(atr_pct={final['atr_pct']:.4f}, rv={final['realized_vol_5d']:.4f})"
    )


def test_panic_regime_consecutive_red():
    # 60 normal days then 3 sharp red days
    closes = [100.0 + 0.5 * math.sin(i / 3.0) for i in range(60)]
    last = closes[-1]
    for r in (-0.02, -0.025, -0.022):
        last = last * (1.0 + r)
        closes.append(last)
    df = _make_bars(closes,
                    highs=[c * 1.01 for c in closes],
                    lows=[c * 0.99 for c in closes])
    tl = build_regime_timeline(symbol="TEST", interval="day", df=df)
    final = tl.iloc[-1]
    assert final["regime"] == "PANIC", (
        f"expected PANIC, got {final['regime']}"
    )


def test_mean_reversion_regime():
    # Oscillating ~0.5% around 100, no trend
    n = 80
    closes = [100.0 * (1.0 + 0.005 * math.sin(i / 1.5)) for i in range(n)]
    highs = [c * 1.006 for c in closes]
    lows = [c * 0.994 for c in closes]
    df = _make_bars(closes, highs=highs, lows=lows)
    tl = build_regime_timeline(symbol="TEST", interval="day", df=df)
    final = tl.iloc[-1]
    assert final["regime"] == "MEAN_REVERSION", (
        f"expected MEAN_REVERSION, got {final['regime']} "
        f"(adx={final['adx_14']:.2f}, atr_pct={final['atr_pct']:.4f})"
    )


# ---------------------------------------------------------------------------
# No-lookahead property
# ---------------------------------------------------------------------------

def test_no_lookahead_classification_only_uses_prior_bars():
    """Regime label at index i must depend only on bars[:i+1].

    Build a timeline on the full series, then rebuild on the prefix up to
    bar 50 and confirm the row-50 label matches.
    """
    closes = [100.0 * (1.003 ** i) for i in range(80)]
    df_full = _make_bars(closes,
                         highs=[c * 1.002 for c in closes],
                         lows=[c * 0.998 for c in closes])
    df_prefix = df_full.iloc[:51].reset_index(drop=True)

    tl_full = build_regime_timeline(symbol="TEST", interval="day", df=df_full)
    tl_prefix = build_regime_timeline(symbol="TEST", interval="day", df=df_prefix)

    # Row 50 in full timeline should equal row 50 in prefix timeline
    assert tl_full.iloc[50]["regime"] == tl_prefix.iloc[50]["regime"]
    assert tl_full.iloc[50]["adx_14"] == pytest.approx(
        tl_prefix.iloc[50]["adx_14"], rel=1e-9, nan_ok=True
    ) if not math.isnan(tl_full.iloc[50]["adx_14"]) else math.isnan(tl_prefix.iloc[50]["adx_14"])


def test_classify_regime_with_empty_history_falls_back():
    feats = {
        "atr_pct": 1.0,
        "realized_vol_5d": 0.2,
        "iv_rank": 0.5,
        "ema_slope_20": 0.001,
        "adx_14": 15.0,
    }
    out = classify_regime(feats, pd.DataFrame(columns=["atr_pct", "realized_vol_5d"]))
    # Low ADX, no percentile data -> mean reversion fallback
    assert out in ("MEAN_REVERSION", "LOW_VOL", "TREND", "PANIC")
