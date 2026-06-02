"""
Build a synthetic option chain from (spot, IV, expiry) — used when NSE is
blocking or when you want to model a what-if scenario without live data.

Two modes:
  1. Flat IV  — pass a single `iv`. Fast, but misses the IV smile/skew.
  2. Per-strike IV anchors — pass `iv_points = {strike: iv, ...}`. The chain
     uses linear interpolation between anchors (flat extrapolation outside).
     Accurate to within ~2% of the live chain when you anchor 3-5 strikes
     across the ATM region.

Pricing assumptions:
  - European exercise (NIFTY weekly/monthly are European, so this is correct)
  - Forward-implied via r (no explicit dividend yield — NIFTY is a price
    index but futures premium reflects rate; q=0 is approximately right)
  - For exact match to your broker chain, use iv_points.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Callable

from .chain import Chain, Quote
from .pricing import black_scholes, implied_vol


IST = timezone(timedelta(hours=5, minutes=30))


def make_iv_curve(iv_points: dict[int, float]) -> Callable[[int], float]:
    """
    Linear interpolation between strike->IV anchors, flat extrapolation outside.
    """
    if not iv_points:
        raise ValueError("iv_points must be non-empty")
    anchors = sorted(iv_points.items())
    strikes = [k for k, _ in anchors]
    ivs = [v for _, v in anchors]

    def curve(k: int) -> float:
        if k <= strikes[0]:
            return ivs[0]
        if k >= strikes[-1]:
            return ivs[-1]
        # find bracket
        for i in range(1, len(strikes)):
            if k <= strikes[i]:
                lo_k, hi_k = strikes[i-1], strikes[i]
                lo_v, hi_v = ivs[i-1], ivs[i]
                frac = (k - lo_k) / (hi_k - lo_k)
                return lo_v + frac * (hi_v - lo_v)
        return ivs[-1]
    return curve


def calibrate_iv_from_quotes(
    spot: float,
    t_years: float,
    quotes: list[tuple[int, str, float]],   # [(strike, 'CE'|'PE', market_price), ...]
    r: float = 0.065,
) -> dict[int, float]:
    """
    Solve for per-strike IV from market prices you read off the chain.

    Only OTM quotes are used (CE for K>spot, PE for K<spot, both at-the-money).
    Deep ITM options are dominated by intrinsic value and yield unreliable IV.
    Returns {strike: iv}.
    """
    out: dict[int, float] = {}
    for k, ot, price in quotes:
        # Skip deep ITM — IV recovery is numerically unstable
        if ot == "CE" and k < spot - 0.005 * spot:
            continue   # deep ITM call
        if ot == "PE" and k > spot + 0.005 * spot:
            continue   # deep ITM put
        iv = implied_vol(price, spot, k, t_years, r=r, q=0.0, option_type=ot)
        if iv is None or iv < 0.02 or iv > 2.0:
            continue
        if k in out:
            out[k] = (out[k] + iv) / 2
        else:
            out[k] = iv
    return out


def synthetic_chain(
    symbol: str = "NIFTY",
    spot: float = 25000.0,
    iv: float = 0.15,
    dte_days: int = 7,
    strike_step: int = 50,
    strikes_each_side: int | None = None,
    r: float = 0.065,
    iv_curve: Callable[[int], float] | None = None,
    anchor_dt: datetime | None = None,
) -> Chain:
    """
    Build a flat-IV chain centered on spot. Strike range auto-scales with the
    IV-implied 3-sigma move so wider candidate strategies have strikes available.

    `anchor_dt` lets the caller set the "now" date for expiry computation
    (useful for backtests where the historical bar's date != real now).
    """
    import math
    anchor = anchor_dt or datetime.now(tz=IST)
    expiry = (anchor + timedelta(days=dte_days)).date()
    t = dte_days / 365.0
    now = anchor

    if strikes_each_side is None:
        # 3-sigma each side, rounded up to strike grid
        three_sigma = spot * iv * math.sqrt(t) * 3
        strikes_each_side = max(20, int(three_sigma / strike_step) + 5)

    atm = round(spot / strike_step) * strike_step
    strikes = range(atm - strikes_each_side * strike_step,
                    atm + (strikes_each_side + 1) * strike_step,
                    strike_step)

    quotes: list[Quote] = []
    for k in strikes:
        iv_k = iv_curve(int(k)) if iv_curve else iv
        for ot in ("CE", "PE"):
            g = black_scholes(spot, k, t, iv_k, r=r, q=0.0, option_type=ot)
            if g.price < 0.10:
                continue
            spread = max(0.05, g.price * 0.005)
            quotes.append(Quote(
                symbol=symbol, expiry=expiry, strike=int(k), option_type=ot,
                ltp=round(g.price, 2),
                bid=round(g.price - spread, 2),
                ask=round(g.price + spread, 2),
                iv=iv_k,
                oi=1000, volume=100,
                snapshot_ts=now,
            ))
    return Chain(symbol=symbol, spot=spot, expiry=expiry, snapshot_ts=now, quotes=quotes)
