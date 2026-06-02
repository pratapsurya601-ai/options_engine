"""
Probability calculations under the standard lognormal model used by BS.

These are RISK-NEUTRAL probabilities by default (drift = r-q). For "real-world"
probabilities pass drift = expected_return (typically the user's directional view
encoded as an annualized drift). Honest framing matters here — risk-neutral P
is a pricing artifact, not a forecast.
"""
from __future__ import annotations

import math


SQRT_2 = math.sqrt(2.0)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / SQRT_2))


def prob_above(
    spot: float, level: float, t_years: float, iv: float, drift: float = 0.0
) -> float:
    """P(S_T > level) under lognormal with given drift and vol."""
    if t_years <= 0 or iv <= 0:
        return 1.0 if spot > level else 0.0
    sigma_t = iv * math.sqrt(t_years)
    d = (math.log(spot / level) + (drift - 0.5 * iv * iv) * t_years) / sigma_t
    return _norm_cdf(d)


def prob_below(spot, level, t_years, iv, drift=0.0) -> float:
    return 1.0 - prob_above(spot, level, t_years, iv, drift)


def prob_between(
    spot: float, low: float, high: float, t_years: float, iv: float, drift: float = 0.0
) -> float:
    return prob_above(spot, low, t_years, iv, drift) - prob_above(spot, high, t_years, iv, drift)


def prob_touch_above(spot, level, t_years, iv, drift=0.0) -> float:
    """
    P(max S_t >= level for t in [0, T]) — barrier touch from below.
    Reflection principle approximation (drift-adjusted).
    """
    if spot >= level:
        return 1.0
    if t_years <= 0 or iv <= 0:
        return 0.0
    # Standard formula: P(touch) ≈ 2 * P(S_T >= level)  for zero drift.
    # With drift, use the reflection result:
    sigma_t = iv * math.sqrt(t_years)
    mu = (drift - 0.5 * iv * iv)
    a = math.log(level / spot)
    p1 = _norm_cdf((-a + mu * t_years) / sigma_t)
    # Reflection adjustment
    p2 = math.exp(2 * mu * a / (iv * iv)) * _norm_cdf((-a - mu * t_years) / sigma_t)
    return min(1.0, max(0.0, p1 + p2))


def prob_touch_below(spot, level, t_years, iv, drift=0.0) -> float:
    if spot <= level:
        return 1.0
    if t_years <= 0 or iv <= 0:
        return 0.0
    sigma_t = iv * math.sqrt(t_years)
    mu = (drift - 0.5 * iv * iv)
    a = math.log(spot / level)
    p1 = _norm_cdf((-a - mu * t_years) / sigma_t)
    p2 = math.exp(-2 * mu * a / (iv * iv)) * _norm_cdf((-a + mu * t_years) / sigma_t)
    return min(1.0, max(0.0, p1 + p2))


def expected_move(spot: float, t_years: float, iv: float) -> float:
    """1-sigma expected move (absolute). Useful for sizing strike selection."""
    return spot * iv * math.sqrt(t_years)


def view_to_drift(view: str, conviction: float, iv: float) -> float:
    """
    Translate a user's directional view into an annualized drift used as the
    'real-world' assumption for win-probability estimates.

    view: 'bullish' | 'bearish' | 'neutral'
    conviction: 0.0 to 1.0 (how strongly)

    Convention: drift scaled by IV so high-vol regimes get proportionally
    larger expected moves. A fully-convinced bull on 20% IV gets +20% drift.
    Conservative — adjust to taste.
    """
    sign = {"bullish": 1.0, "bearish": -1.0, "neutral": 0.0}.get(view.lower())
    if sign is None:
        raise ValueError(f"view must be bullish/bearish/neutral, got {view!r}")
    return sign * conviction * iv
