"""Tests for engine.research.position_sizing."""

from __future__ import annotations

import math

import pytest

from engine.research.position_sizing import (
    SizingResult,
    TradeSetup,
    all_methods,
    conviction_scaled,
    fixed_fractional,
    kelly_capped,
    recommend,
    vol_adjusted,
)


# Standard test setup: NIFTY 23500 CE @ Rs 240, 25% stop, Rs 5L capital.
def make_setup(**overrides) -> TradeSetup:
    base = dict(
        capital=500_000,
        entry_premium=240.0,
        stop_loss_pct=0.25,
        lot_size=75,
        win_rate=0.71,
        avg_win_r=2.38,
        avg_loss_r=1.0,
        predictive_score=0.55,
    )
    base.update(overrides)
    return TradeSetup(**base)


# ---------------------------------------------------------------------------
# fixed_fractional
# ---------------------------------------------------------------------------


def test_fixed_fractional_known_lot_count():
    # capital=500k, premium=240, stop=25%, lot=75
    # max_loss/lot = 240 * 0.25 * 75 = 4500
    # 1% of 500000 = 5000  -> floor(5000/4500) = 1 lot
    setup = make_setup()
    res = fixed_fractional(setup, risk_per_trade_pct=0.01)
    assert isinstance(res, SizingResult)
    assert res.lots == 1
    assert math.isclose(res.capital_at_risk, 4500.0, rel_tol=1e-9)
    assert res.capital_at_risk_pct == pytest.approx(0.009, abs=1e-4)


def test_fixed_fractional_scales_with_risk_pct():
    setup = make_setup()
    r1 = fixed_fractional(setup, risk_per_trade_pct=0.01)
    r5 = fixed_fractional(setup, risk_per_trade_pct=0.05)
    assert r5.lots > r1.lots


def test_fixed_fractional_rejects_bad_premium():
    with pytest.raises(ValueError):
        fixed_fractional(make_setup(entry_premium=0))


# ---------------------------------------------------------------------------
# kelly_capped
# ---------------------------------------------------------------------------


def test_kelly_positive_edge_returns_positive_lots():
    setup = make_setup(win_rate=0.7, avg_win_r=2.0)
    res = kelly_capped(setup)
    assert res.lots > 0


def test_kelly_negative_edge_returns_zero_lots():
    # p=0.4, b=1 -> kelly = (0.4 - 0.6)/1 = -0.2 -> 0 lots
    setup = make_setup(win_rate=0.4, avg_win_r=1.0)
    res = kelly_capped(setup)
    assert res.lots == 0
    assert any("skip" in w.lower() or "insufficient" in w.lower()
               for w in res.warnings)


def test_kelly_never_exceeds_max_kelly_pct():
    # Extremely favorable inputs would push raw Kelly very high; ensure cap.
    setup = make_setup(
        capital=10_000_000,
        entry_premium=240.0,
        stop_loss_pct=0.25,
        win_rate=0.95,
        avg_win_r=10.0,
    )
    res = kelly_capped(setup, fractional_kelly=0.25, max_kelly_pct=0.05)
    # Capital at risk pct should be <= max_kelly_pct (allow tiny floor slack)
    assert res.capital_at_risk_pct <= 0.05 + 1e-9


def test_kelly_falls_back_when_win_rate_missing():
    setup = make_setup(win_rate=None, avg_win_r=None)
    res = kelly_capped(setup)
    assert "fallback" in res.method
    assert any("fallback" in w.lower() for w in res.warnings)


# ---------------------------------------------------------------------------
# vol_adjusted
# ---------------------------------------------------------------------------


def test_vol_adjusted_basic():
    setup = make_setup()
    # vol/lot = 240 * 0.30 * 75 = 5400
    # budget  = 500000 * 0.005  = 2500 -> 0 lots
    res = vol_adjusted(setup)
    assert res.lots == 0
    # Looser target -> more lots
    res2 = vol_adjusted(setup, target_portfolio_vol_pct=0.02)
    assert res2.lots > res.lots


# ---------------------------------------------------------------------------
# conviction_scaled
# ---------------------------------------------------------------------------


def test_conviction_scaled_scales_with_predictive_score():
    low = make_setup(capital=5_000_000, predictive_score=0.1)
    high = make_setup(capital=5_000_000, predictive_score=0.9)
    r_low = conviction_scaled(low)
    r_high = conviction_scaled(high)
    assert r_high.lots >= r_low.lots
    # Multiplier should clip at 2.0x for high score
    base_high = fixed_fractional(high)
    assert r_high.lots <= base_high.lots * 2


def test_conviction_scaled_min_multiplier():
    # score=0 -> multiplier clipped to 0.5
    setup = make_setup(capital=5_000_000, predictive_score=0.0)
    base = fixed_fractional(setup)
    res = conviction_scaled(setup)
    assert res.lots == int(math.floor(base.lots * 0.5))


def test_conviction_scaled_kelly_base():
    setup = make_setup(capital=5_000_000)
    res = conviction_scaled(setup, base_method="kelly_capped")
    assert res.lots >= 0


# ---------------------------------------------------------------------------
# recommend
# ---------------------------------------------------------------------------


def test_recommend_ordering_by_aggression():
    setup = make_setup(capital=5_000_000)
    cons = recommend(setup, "conservative")
    bal = recommend(setup, "balanced")
    agg = recommend(setup, "aggressive")
    assert cons.lots <= bal.lots <= agg.lots


def test_recommend_invalid_aggression():
    with pytest.raises(ValueError):
        recommend(make_setup(), "yolo")


# ---------------------------------------------------------------------------
# Guardrails / warnings
# ---------------------------------------------------------------------------


def test_zero_lots_warning():
    # Tiny capital -> 0 lots from fixed_fractional
    setup = make_setup(capital=1000)
    res = fixed_fractional(setup)
    assert res.lots == 0
    assert any("skip" in w.lower() for w in res.warnings)


def test_oversized_risk_warning():
    # 20% risk per trade should trip the >5% warning when lots > 0
    setup = make_setup(capital=10_000_000)
    res = fixed_fractional(setup, risk_per_trade_pct=0.20)
    assert res.lots > 0
    assert any(">5%" in w for w in res.warnings)


def test_large_position_warning():
    setup = make_setup(capital=50_000_000)
    res = fixed_fractional(setup, risk_per_trade_pct=0.01)
    assert res.lots > 10
    assert any("Large position" in w for w in res.warnings)


def test_margin_cap_enforced():
    # If raw_lots demanded would exceed capital/(premium*lot_size), clamp.
    # capital=500k, premium=240, lot=75 -> margin cap = floor(500000/18000)=27
    setup = make_setup(capital=500_000)
    res = fixed_fractional(setup, risk_per_trade_pct=1.0)
    margin_cap = 500_000 // (240 * 75)
    assert res.lots <= margin_cap


def test_all_methods_returns_four_results():
    results = all_methods(make_setup())
    assert set(results.keys()) == {
        "fixed_fractional",
        "kelly_capped",
        "vol_adjusted",
        "conviction_scaled",
    }
    for r in results.values():
        assert isinstance(r, SizingResult)
        assert r.lots >= 0
