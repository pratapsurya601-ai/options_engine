"""Position sizing framework for NIFTY options trades.

Translates research outputs (expected edge, win rate, volatility, predictive
score) into actionable lot counts. All methods are honest, capped, and
defensive: they refuse to recommend trades with negative edge, they cap
Kelly fractions to avoid over-betting, and they never exceed margin
sanity (lots * premium * lot_size <= capital).

Four sizing methods are exposed:

1. ``fixed_fractional`` - constant risk per trade (default 1% of capital).
2. ``kelly_capped``     - Kelly formula, fractional (default 1/4 Kelly),
                          hard-capped at 5% of capital.
3. ``vol_adjusted``     - sized so expected trade volatility matches a
                          target portfolio vol contribution.
4. ``conviction_scaled`` - takes a base method (fixed_fractional or
                          kelly_capped) and scales lots by a [0,1]
                          predictive score.

Plus aggregators ``all_methods`` and ``recommend(aggression=...)``.

Conventions
-----------
* All money values in Rs (no glyph in output).
* ``lot_size`` defaults to 75 (current NIFTY lot size).
* All returned ``lots`` are integers >= 0.

Example
-------
>>> setup = TradeSetup(
...     capital=500_000,
...     entry_premium=240.0,
...     stop_loss_pct=0.25,
...     win_rate=0.71,
...     avg_win_r=2.38,
...     predictive_score=0.65,
... )
>>> res = fixed_fractional(setup)
>>> res.lots >= 0
True
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import floor
from typing import Optional


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class TradeSetup:
    """Inputs describing a candidate options trade.

    Parameters
    ----------
    capital
        Available capital, Rs.
    entry_premium
        Per-unit option premium at entry, Rs.
    stop_loss_pct
        Stop-loss as a fraction of premium, e.g. ``0.25`` for a 25%
        premium stop.
    lot_size
        Contract lot size. NIFTY = 75.
    win_rate
        Historical win rate in [0, 1]. Required for Kelly.
    avg_win_r
        Average winner expressed in R-multiples (R = 1 unit of risk).
        e.g. ``2.38`` means winners average 2.38x the loss size.
    avg_loss_r
        Average loser in R-multiples; defaults to 1.0.
    expected_edge_pct
        Optional expected gain per trade (currently informational).
    predictive_score
        Confidence / quality score in [0, 1] from the predictive_power
        module. Used by ``conviction_scaled``.
    """

    capital: float
    entry_premium: float
    stop_loss_pct: float
    lot_size: int = 75
    win_rate: Optional[float] = None
    avg_win_r: Optional[float] = None
    avg_loss_r: float = 1.0
    expected_edge_pct: Optional[float] = None
    predictive_score: Optional[float] = None


@dataclass
class SizingResult:
    """Output of a sizing method."""

    method: str
    lots: int
    capital_at_risk: float
    capital_at_risk_pct: float
    rationale: str
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate(setup: TradeSetup) -> None:
    if setup.entry_premium <= 0:
        raise ValueError("entry_premium must be > 0")
    if setup.capital <= 0:
        raise ValueError("capital must be > 0")
    if setup.stop_loss_pct <= 0 or setup.stop_loss_pct > 1:
        raise ValueError("stop_loss_pct must be in (0, 1]")
    if setup.lot_size <= 0:
        raise ValueError("lot_size must be > 0")


def _max_loss_per_lot(setup: TradeSetup) -> float:
    """Rs lost per lot if stop hits."""
    return setup.entry_premium * setup.stop_loss_pct * setup.lot_size


def _margin_cap(setup: TradeSetup) -> int:
    """Upper bound on lots from notional sanity: total premium <= capital.

    For long options the premium is the cash outlay; for short options
    SPAN margin would replace this, but as a conservative sanity check
    we treat premium*lot_size as the per-lot capital lock-up.
    """
    per_lot = setup.entry_premium * setup.lot_size
    if per_lot <= 0:
        return 0
    return int(floor(setup.capital / per_lot))


def _apply_guardrails(
    lots: int,
    capital_at_risk_pct: float,
    setup: TradeSetup,
    warnings: list[str],
) -> int:
    """Cap lots at margin sanity and append common warnings."""
    margin_cap = _margin_cap(setup)
    if lots > margin_cap:
        warnings.append(
            f"Lots clamped from {lots} to margin cap {margin_cap}"
        )
        lots = margin_cap

    if lots == 0:
        warnings.append("Negative or insufficient edge - skip trade")
    if capital_at_risk_pct > 0.05:
        warnings.append("Risking >5% on single trade - high")
    if lots > 10:
        warnings.append("Large position; verify intentional")

    return max(lots, 0)


def _build_result(
    method: str,
    lots: int,
    setup: TradeSetup,
    rationale: str,
    warnings: Optional[list[str]] = None,
) -> SizingResult:
    warnings = list(warnings or [])
    loss_per_lot = _max_loss_per_lot(setup)
    cap_at_risk = lots * loss_per_lot
    cap_at_risk_pct = cap_at_risk / setup.capital if setup.capital else 0.0

    # Compute pct before guardrails so the >5% warning uses pre-clamp value
    # if margin cap kicks in further down. But to keep warnings consistent
    # with FINAL lots, recompute after clamping.
    lots = _apply_guardrails(lots, cap_at_risk_pct, setup, warnings)
    cap_at_risk = lots * loss_per_lot
    cap_at_risk_pct = cap_at_risk / setup.capital if setup.capital else 0.0

    return SizingResult(
        method=method,
        lots=lots,
        capital_at_risk=round(cap_at_risk, 2),
        capital_at_risk_pct=round(cap_at_risk_pct, 4),
        rationale=rationale,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Sizing methods
# ---------------------------------------------------------------------------


def fixed_fractional(
    setup: TradeSetup,
    risk_per_trade_pct: float = 0.01,
) -> SizingResult:
    """Risk a fixed fraction of capital per trade.

    Example
    -------
    NIFTY 23500 CE @ Rs 240, 25% stop, Rs 5L capital, 1% risk:
        max_loss_per_lot = 240 * 0.25 * 75 = Rs 4,500
        budget           = 500000 * 0.01    = Rs 5,000
        lots             = floor(5000/4500) = 1
    """
    _validate(setup)
    if not 0 < risk_per_trade_pct <= 1:
        raise ValueError("risk_per_trade_pct must be in (0, 1]")

    loss_per_lot = _max_loss_per_lot(setup)
    risk_budget = setup.capital * risk_per_trade_pct
    raw_lots = int(floor(risk_budget / loss_per_lot)) if loss_per_lot > 0 else 0

    rationale = (
        f"Fixed fractional: risking {risk_per_trade_pct*100:.2f}% "
        f"(Rs {risk_budget:,.0f}) of Rs {setup.capital:,.0f}; "
        f"max loss/lot = Rs {loss_per_lot:,.0f} -> {raw_lots} lots."
    )
    return _build_result("fixed_fractional", raw_lots, setup, rationale)


def kelly_capped(
    setup: TradeSetup,
    fractional_kelly: float = 0.25,
    max_kelly_pct: float = 0.05,
) -> SizingResult:
    """Kelly fraction, fractionally scaled AND hard-capped.

    Falls back to ``fixed_fractional`` (with a warning) if ``win_rate``
    is not supplied. Returns 0 lots if the edge is negative.

    The Kelly fraction is computed as::

        b = avg_win_r / avg_loss_r
        kelly = (b * p - q) / b      where q = 1 - p

    Then::

        used = min(kelly * fractional_kelly, max_kelly_pct)

    NEVER full Kelly.
    """
    _validate(setup)
    if not 0 < fractional_kelly <= 1:
        raise ValueError("fractional_kelly must be in (0, 1]")
    if not 0 < max_kelly_pct <= 1:
        raise ValueError("max_kelly_pct must be in (0, 1]")

    warnings: list[str] = []

    if setup.win_rate is None or setup.avg_win_r is None:
        warnings.append(
            "Win rate required for Kelly; using fixed_fractional fallback"
        )
        fallback = fixed_fractional(setup)
        return SizingResult(
            method="kelly_capped(fallback=fixed_fractional)",
            lots=fallback.lots,
            capital_at_risk=fallback.capital_at_risk,
            capital_at_risk_pct=fallback.capital_at_risk_pct,
            rationale=fallback.rationale,
            warnings=warnings + fallback.warnings,
        )

    p = float(setup.win_rate)
    if not 0 <= p <= 1:
        raise ValueError("win_rate must be in [0, 1]")
    q = 1.0 - p
    b = setup.avg_win_r / setup.avg_loss_r

    if b <= 0:
        raise ValueError("avg_win_r / avg_loss_r must be > 0")

    kelly = (b * p - q) / b
    capped = min(kelly * fractional_kelly, max_kelly_pct)

    if capped <= 0:
        rationale = (
            f"Kelly: p={p:.3f}, b={b:.3f} -> raw Kelly={kelly:.4f} "
            "(non-positive); recommend 0 lots."
        )
        return _build_result(
            "kelly_capped", 0, setup, rationale, warnings=warnings
        )

    loss_per_lot = _max_loss_per_lot(setup)
    risk_budget = setup.capital * capped
    raw_lots = int(floor(risk_budget / loss_per_lot)) if loss_per_lot > 0 else 0

    rationale = (
        f"Kelly: p={p:.3f}, b={b:.3f}, raw Kelly={kelly:.4f}; "
        f"fractional({fractional_kelly})*Kelly={kelly*fractional_kelly:.4f}; "
        f"capped at {max_kelly_pct:.4f} -> using {capped:.4f} "
        f"(Rs {risk_budget:,.0f}); {raw_lots} lots."
    )
    return _build_result(
        "kelly_capped", raw_lots, setup, rationale, warnings=warnings
    )


def vol_adjusted(
    setup: TradeSetup,
    target_portfolio_vol_pct: float = 0.005,
    trade_volatility_estimate: float = 0.30,
) -> SizingResult:
    """Size so the trade's expected volatility matches a target.

    ``trade_volatility_estimate`` is the expected stdev of premium over
    the hold period as a fraction of premium (e.g. 0.30 = 30%).

    expected_trade_vol_per_lot = entry_premium * trade_vol_est * lot_size
    lots = floor(capital * target_pct / expected_trade_vol_per_lot)
    """
    _validate(setup)
    if trade_volatility_estimate <= 0:
        raise ValueError("trade_volatility_estimate must be > 0")
    if not 0 < target_portfolio_vol_pct <= 1:
        raise ValueError("target_portfolio_vol_pct must be in (0, 1]")

    vol_per_lot = (
        setup.entry_premium * trade_volatility_estimate * setup.lot_size
    )
    vol_budget = setup.capital * target_portfolio_vol_pct
    raw_lots = int(floor(vol_budget / vol_per_lot)) if vol_per_lot > 0 else 0

    rationale = (
        f"Vol-adjusted: trade vol/lot = Rs {vol_per_lot:,.0f} "
        f"(premium {setup.entry_premium} * "
        f"{trade_volatility_estimate*100:.0f}% * {setup.lot_size}); "
        f"target vol budget = Rs {vol_budget:,.0f} "
        f"({target_portfolio_vol_pct*100:.2f}% of capital) "
        f"-> {raw_lots} lots."
    )
    return _build_result("vol_adjusted", raw_lots, setup, rationale)


def conviction_scaled(
    setup: TradeSetup,
    base_method: str = "fixed_fractional",
    confidence_range: tuple[float, float] = (0.5, 2.0),
) -> SizingResult:
    """Scale a base sizing method by a [0,1] predictive score.

    ``confidence_multiplier = clip(predictive_score * 4, lo, hi)``

    A score of 0.25 -> 1.0x (neutral); 0.5 -> 2.0x (max);
    0.0 -> 0.5x (min). Capped by ``confidence_range``.

    If ``predictive_score`` is missing, multiplier = 1.0 with a warning.
    """
    _validate(setup)
    lo, hi = confidence_range
    if lo <= 0 or hi <= lo:
        raise ValueError("confidence_range must satisfy 0 < lo < hi")

    if base_method == "fixed_fractional":
        base = fixed_fractional(setup)
    elif base_method == "kelly_capped":
        base = kelly_capped(setup)
    else:
        raise ValueError(
            f"base_method must be 'fixed_fractional' or 'kelly_capped', "
            f"got {base_method!r}"
        )

    warnings = list(base.warnings)

    if setup.predictive_score is None:
        warnings.append(
            "predictive_score missing; conviction multiplier = 1.0"
        )
        mult = 1.0
        score_str = "n/a"
    else:
        s = float(setup.predictive_score)
        if not 0 <= s <= 1:
            raise ValueError("predictive_score must be in [0, 1]")
        mult = max(lo, min(hi, s * 4.0))
        score_str = f"{s:.3f}"

    raw_lots = int(floor(base.lots * mult))

    rationale = (
        f"Conviction-scaled on {base_method}: base={base.lots} lots, "
        f"predictive_score={score_str}, multiplier={mult:.2f} "
        f"(clipped to [{lo}, {hi}]) -> {raw_lots} lots."
    )
    return _build_result(
        "conviction_scaled", raw_lots, setup, rationale, warnings=warnings
    )


# ---------------------------------------------------------------------------
# Aggregators
# ---------------------------------------------------------------------------


def all_methods(setup: TradeSetup) -> dict[str, SizingResult]:
    """Run every sizing method. Useful for dashboards / comparison tables."""
    return {
        "fixed_fractional": fixed_fractional(setup),
        "kelly_capped": kelly_capped(setup),
        "vol_adjusted": vol_adjusted(setup),
        "conviction_scaled": conviction_scaled(setup),
    }


def recommend(
    setup: TradeSetup,
    aggression: str = "conservative",
) -> SizingResult:
    """Pick a single sizing recommendation based on aggression preference.

    * ``conservative`` -> the method giving the FEWEST lots (most defensive).
    * ``balanced``     -> ``kelly_capped`` directly.
    * ``aggressive``   -> the method giving the MOST lots (still capped
                          by margin and >5% warnings).
    """
    aggression = aggression.lower().strip()
    results = all_methods(setup)

    if aggression == "balanced":
        chosen = results["kelly_capped"]
        label = "balanced(kelly_capped)"
    elif aggression == "conservative":
        chosen = min(results.values(), key=lambda r: r.lots)
        label = f"conservative(min:{chosen.method})"
    elif aggression == "aggressive":
        chosen = max(results.values(), key=lambda r: r.lots)
        label = f"aggressive(max:{chosen.method})"
    else:
        raise ValueError(
            "aggression must be 'conservative', 'balanced', or 'aggressive'"
        )

    return SizingResult(
        method=label,
        lots=chosen.lots,
        capital_at_risk=chosen.capital_at_risk,
        capital_at_risk_pct=chosen.capital_at_risk_pct,
        rationale=chosen.rationale,
        warnings=list(chosen.warnings),
    )


# ---------------------------------------------------------------------------
# Smoke test (htf_naked)
# ---------------------------------------------------------------------------


def _smoke() -> None:  # pragma: no cover - manual invocation
    setup = TradeSetup(
        capital=500_000,
        entry_premium=240.0,
        stop_loss_pct=0.25,
        lot_size=75,
        win_rate=0.71,
        avg_win_r=2.38,
        avg_loss_r=1.0,
        predictive_score=0.55,
    )
    print(f"htf_naked smoke: capital=Rs {setup.capital:,.0f}, "
          f"premium=Rs {setup.entry_premium}, stop={setup.stop_loss_pct*100:.0f}%, "
          f"p={setup.win_rate}, R/R={setup.avg_win_r}")
    print("-" * 70)
    for name, res in all_methods(setup).items():
        print(f"[{name}]")
        print(f"  lots         : {res.lots}")
        print(f"  cap_at_risk  : Rs {res.capital_at_risk:,.0f} "
              f"({res.capital_at_risk_pct*100:.2f}%)")
        print(f"  rationale    : {res.rationale}")
        if res.warnings:
            for w in res.warnings:
                print(f"  ! {w}")
        print()
    for agg in ("conservative", "balanced", "aggressive"):
        r = recommend(setup, agg)
        print(f"recommend({agg:>12}) -> {r.lots} lots "
              f"(method={r.method}, risk={r.capital_at_risk_pct*100:.2f}%)")


if __name__ == "__main__":  # pragma: no cover
    _smoke()
