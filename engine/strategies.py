"""
Option strategy payoff models. Each strategy is a list of legs; payoff at
expiry is the sum of leg payoffs. We also compute the cost (debit) or credit,
max profit, max loss, and breakeven(s).

Strategies covered (most common Indian retail):
  - long_call, long_put
  - bull_call_spread, bear_put_spread        (debit spreads)
  - bull_put_spread, bear_call_spread        (credit spreads)
  - long_straddle, long_strangle              (volatility long)
  - short_straddle, short_strangle            (volatility short — undefined risk)
  - iron_condor, iron_butterfly               (defined-risk credit)

Each builder returns a Strategy with legs sized to 1 lot. Caller scales lots.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .pricing import black_scholes


@dataclass
class Leg:
    option_type: str    # 'CE' or 'PE'
    strike: int
    action: str         # 'BUY' or 'SELL'
    premium: float      # per-share premium (positive number)
    iv: float           # iv used (for sanity)

    def signed_premium(self) -> float:
        return -self.premium if self.action == "BUY" else self.premium

    def payoff_at(self, spot: float) -> float:
        intrinsic = max(spot - self.strike, 0.0) if self.option_type == "CE" \
                    else max(self.strike - spot, 0.0)
        if self.action == "BUY":
            return intrinsic - self.premium
        else:
            return self.premium - intrinsic


@dataclass
class Strategy:
    name: str
    legs: list[Leg]
    view: str           # 'bullish' | 'bearish' | 'neutral' | 'vol_long' | 'vol_short'
    lot_size: int = 75
    notes: list[str] = field(default_factory=list)

    def net_debit(self) -> float:
        """Positive = debit paid. Negative = credit received."""
        return -sum(leg.signed_premium() for leg in self.legs)

    def cost_per_lot(self) -> float:
        return self.net_debit() * self.lot_size

    def payoff_at(self, spot: float) -> float:
        return sum(leg.payoff_at(spot) for leg in self.legs) * self.lot_size

    def breakevens(self, spot_min: float, spot_max: float, step: float = 1.0) -> list[float]:
        """Numeric BE search across [spot_min, spot_max]."""
        bes: list[float] = []
        prev_s = spot_min
        prev_p = self.payoff_at(prev_s)
        s = spot_min + step
        while s <= spot_max:
            p = self.payoff_at(s)
            if (prev_p == 0) or (prev_p * p < 0):
                # linear interp
                if p == prev_p:
                    bes.append(s)
                else:
                    be = prev_s - prev_p * (s - prev_s) / (p - prev_p)
                    bes.append(round(be, 2))
            prev_s, prev_p = s, p
            s += step
        # dedupe close values
        out: list[float] = []
        for b in bes:
            if not out or abs(b - out[-1]) > 1.0:
                out.append(b)
        return out

    def max_profit_loss(self, spot_min: float, spot_max: float, step: float = 1.0) -> tuple[float, float]:
        ps = []
        s = spot_min
        while s <= spot_max:
            ps.append(self.payoff_at(s))
            s += step
        return max(ps), min(ps)


# --- builders ----------------------------------------------------------------

PricerFn = Callable[[int, str], tuple[float, float]]
"""Pricer: given (strike, option_type) returns (premium, iv)."""


def long_call(strike: int, pricer: PricerFn, lot_size: int = 75) -> Strategy:
    p, iv = pricer(strike, "CE")
    return Strategy("long_call", [Leg("CE", strike, "BUY", p, iv)], view="bullish", lot_size=lot_size)


def long_put(strike: int, pricer: PricerFn, lot_size: int = 75) -> Strategy:
    p, iv = pricer(strike, "PE")
    return Strategy("long_put", [Leg("PE", strike, "BUY", p, iv)], view="bearish", lot_size=lot_size)


def bull_call_spread(lo: int, hi: int, pricer: PricerFn, lot_size: int = 75) -> Strategy:
    p1, iv1 = pricer(lo, "CE")
    p2, iv2 = pricer(hi, "CE")
    return Strategy(
        f"bull_call_spread {lo}/{hi}",
        [Leg("CE", lo, "BUY", p1, iv1), Leg("CE", hi, "SELL", p2, iv2)],
        view="bullish", lot_size=lot_size,
    )


def bear_put_spread(hi: int, lo: int, pricer: PricerFn, lot_size: int = 75) -> Strategy:
    p1, iv1 = pricer(hi, "PE")
    p2, iv2 = pricer(lo, "PE")
    return Strategy(
        f"bear_put_spread {hi}/{lo}",
        [Leg("PE", hi, "BUY", p1, iv1), Leg("PE", lo, "SELL", p2, iv2)],
        view="bearish", lot_size=lot_size,
    )


def bull_put_spread(hi: int, lo: int, pricer: PricerFn, lot_size: int = 75) -> Strategy:
    """Credit spread: sell higher-strike put, buy lower-strike put. Bullish/neutral."""
    p1, iv1 = pricer(hi, "PE")
    p2, iv2 = pricer(lo, "PE")
    return Strategy(
        f"bull_put_spread {hi}/{lo}",
        [Leg("PE", hi, "SELL", p1, iv1), Leg("PE", lo, "BUY", p2, iv2)],
        view="bullish", lot_size=lot_size,
        notes=["credit spread"],
    )


def bear_call_spread(lo: int, hi: int, pricer: PricerFn, lot_size: int = 75) -> Strategy:
    p1, iv1 = pricer(lo, "CE")
    p2, iv2 = pricer(hi, "CE")
    return Strategy(
        f"bear_call_spread {lo}/{hi}",
        [Leg("CE", lo, "SELL", p1, iv1), Leg("CE", hi, "BUY", p2, iv2)],
        view="bearish", lot_size=lot_size,
        notes=["credit spread"],
    )


def long_straddle(strike: int, pricer: PricerFn, lot_size: int = 75) -> Strategy:
    pc, ivc = pricer(strike, "CE")
    pp, ivp = pricer(strike, "PE")
    return Strategy(
        f"long_straddle {strike}",
        [Leg("CE", strike, "BUY", pc, ivc), Leg("PE", strike, "BUY", pp, ivp)],
        view="vol_long", lot_size=lot_size,
    )


def long_strangle(call_k: int, put_k: int, pricer: PricerFn, lot_size: int = 75) -> Strategy:
    pc, ivc = pricer(call_k, "CE")
    pp, ivp = pricer(put_k, "PE")
    return Strategy(
        f"long_strangle {put_k}/{call_k}",
        [Leg("CE", call_k, "BUY", pc, ivc), Leg("PE", put_k, "BUY", pp, ivp)],
        view="vol_long", lot_size=lot_size,
    )


def short_call(strike: int, pricer: PricerFn, lot_size: int = 75) -> Strategy:
    p, iv = pricer(strike, "CE")
    return Strategy(
        f"short_call {strike}", [Leg("CE", strike, "SELL", p, iv)],
        view="bearish", lot_size=lot_size,
        notes=["NAKED SHORT — undefined upside risk; ~Rs 1.5-2L margin per lot"],
    )


def short_put(strike: int, pricer: PricerFn, lot_size: int = 75) -> Strategy:
    p, iv = pricer(strike, "PE")
    return Strategy(
        f"short_put {strike}", [Leg("PE", strike, "SELL", p, iv)],
        view="bullish", lot_size=lot_size,
        notes=["NAKED SHORT — large downside risk; ~Rs 1.5-2L margin per lot"],
    )


def short_straddle(strike: int, pricer: PricerFn, lot_size: int = 75) -> Strategy:
    pc, ivc = pricer(strike, "CE")
    pp, ivp = pricer(strike, "PE")
    return Strategy(
        f"short_straddle {strike}",
        [Leg("CE", strike, "SELL", pc, ivc), Leg("PE", strike, "SELL", pp, ivp)],
        view="vol_short", lot_size=lot_size,
        notes=["NAKED SHORT — undefined risk both sides; ~Rs 2-3L margin per lot"],
    )


def short_strangle(call_k: int, put_k: int, pricer: PricerFn, lot_size: int = 75) -> Strategy:
    pc, ivc = pricer(call_k, "CE")
    pp, ivp = pricer(put_k, "PE")
    return Strategy(
        f"short_strangle {put_k}/{call_k}",
        [Leg("CE", call_k, "SELL", pc, ivc), Leg("PE", put_k, "SELL", pp, ivp)],
        view="vol_short", lot_size=lot_size,
        notes=["NAKED SHORT — undefined risk both sides; ~Rs 1.5-2.5L margin per lot"],
    )


def iron_condor(put_lo: int, put_hi: int, call_lo: int, call_hi: int,
                pricer: PricerFn, lot_size: int = 75) -> Strategy:
    """Sell put_hi + buy put_lo (bull put), sell call_lo + buy call_hi (bear call)."""
    pp_sell, iv1 = pricer(put_hi, "PE")
    pp_buy, iv2 = pricer(put_lo, "PE")
    pc_sell, iv3 = pricer(call_lo, "CE")
    pc_buy, iv4 = pricer(call_hi, "CE")
    return Strategy(
        f"iron_condor {put_lo}/{put_hi}/{call_lo}/{call_hi}",
        [
            Leg("PE", put_hi, "SELL", pp_sell, iv1),
            Leg("PE", put_lo, "BUY", pp_buy, iv2),
            Leg("CE", call_lo, "SELL", pc_sell, iv3),
            Leg("CE", call_hi, "BUY", pc_buy, iv4),
        ],
        view="neutral", lot_size=lot_size,
        notes=["defined-risk credit"],
    )
