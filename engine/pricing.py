"""
Black-Scholes pricing + Greeks for European options on a non-dividend-paying
underlying. NIFTY is a price index, so q=0 is the right default; for stock
options pass q=dividend_yield.

All inputs/outputs in decimal form (iv=0.18 means 18%, T in years).
"""
from __future__ import annotations

import math
from dataclasses import dataclass


SQRT_2PI = math.sqrt(2.0 * math.pi)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / SQRT_2PI


@dataclass(frozen=True)
class Greeks:
    price: float
    delta: float
    gamma: float
    vega: float       # per 1.00 (i.e. 100 vol points) change — divide by 100 for per-vol-point
    theta_year: float # per year
    theta_day: float  # per calendar day (theta_year / 365)
    rho: float


def black_scholes(
    spot: float,
    strike: float,
    t_years: float,
    iv: float,
    r: float = 0.065,
    q: float = 0.0,
    option_type: str = "CE",
) -> Greeks:
    """Closed-form BS for European option. option_type in {'CE','PE'}."""
    if t_years <= 0 or iv <= 0:
        # Expired / degenerate — return intrinsic only
        if option_type == "CE":
            intrinsic = max(spot - strike, 0.0)
            delta = 1.0 if spot > strike else 0.0
        else:
            intrinsic = max(strike - spot, 0.0)
            delta = -1.0 if spot < strike else 0.0
        return Greeks(intrinsic, delta, 0.0, 0.0, 0.0, 0.0, 0.0)

    sqrt_t = math.sqrt(t_years)
    d1 = (math.log(spot / strike) + (r - q + 0.5 * iv * iv) * t_years) / (iv * sqrt_t)
    d2 = d1 - iv * sqrt_t

    disc_r = math.exp(-r * t_years)
    disc_q = math.exp(-q * t_years)
    pdf_d1 = _norm_pdf(d1)

    if option_type == "CE":
        price = spot * disc_q * _norm_cdf(d1) - strike * disc_r * _norm_cdf(d2)
        delta = disc_q * _norm_cdf(d1)
        rho = strike * t_years * disc_r * _norm_cdf(d2) / 100.0
        theta_year = (
            -(spot * disc_q * pdf_d1 * iv) / (2.0 * sqrt_t)
            - r * strike * disc_r * _norm_cdf(d2)
            + q * spot * disc_q * _norm_cdf(d1)
        )
    elif option_type == "PE":
        price = strike * disc_r * _norm_cdf(-d2) - spot * disc_q * _norm_cdf(-d1)
        delta = -disc_q * _norm_cdf(-d1)
        rho = -strike * t_years * disc_r * _norm_cdf(-d2) / 100.0
        theta_year = (
            -(spot * disc_q * pdf_d1 * iv) / (2.0 * sqrt_t)
            + r * strike * disc_r * _norm_cdf(-d2)
            - q * spot * disc_q * _norm_cdf(-d1)
        )
    else:
        raise ValueError(f"option_type must be CE or PE, got {option_type!r}")

    gamma = disc_q * pdf_d1 / (spot * iv * sqrt_t)
    vega = spot * disc_q * pdf_d1 * sqrt_t / 100.0  # per 1 vol-point

    return Greeks(
        price=price,
        delta=delta,
        gamma=gamma,
        vega=vega,
        theta_year=theta_year,
        theta_day=theta_year / 365.0,
        rho=rho,
    )


def implied_vol(
    market_price: float,
    spot: float,
    strike: float,
    t_years: float,
    r: float = 0.065,
    q: float = 0.0,
    option_type: str = "CE",
    tol: float = 1e-5,
    max_iter: int = 100,
) -> float | None:
    """Back out IV via bisection. Returns None if no solution in [0.001, 5.0]."""
    if t_years <= 0 or market_price <= 0:
        return None

    # Intrinsic floor
    if option_type == "CE":
        intrinsic = max(spot * math.exp(-q * t_years) - strike * math.exp(-r * t_years), 0.0)
    else:
        intrinsic = max(strike * math.exp(-r * t_years) - spot * math.exp(-q * t_years), 0.0)
    if market_price < intrinsic - 1e-6:
        return None

    lo, hi = 0.001, 5.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        price = black_scholes(spot, strike, t_years, mid, r, q, option_type).price
        if abs(price - market_price) < tol:
            return mid
        if price > market_price:
            hi = mid
        else:
            lo = mid
    return mid


def project_price(
    spot_now: float,
    strike: float,
    t_now_years: float,
    iv_now: float,
    spot_target: float,
    t_target_years: float,
    iv_target: float | None = None,
    r: float = 0.065,
    q: float = 0.0,
    option_type: str = "CE",
) -> float:
    """
    What would this option be worth at (spot_target, t_target) assuming IV holds
    (or moves to iv_target)? This is NOT a forecast — it's a what-if under BS.
    """
    iv_use = iv_target if iv_target is not None else iv_now
    return black_scholes(spot_target, strike, t_target_years, iv_use, r, q, option_type).price
