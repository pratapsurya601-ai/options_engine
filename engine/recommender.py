"""
Strategy recommender. Given:
  - user's directional view (bullish/bearish/neutral)
  - conviction (0..1)
  - target_win_pct (e.g. 0.70 — only return trades whose modelled win-prob >= this)
  - max_capital_inr
  - smart_money_score (-1..+1) — optional, biases conviction
  - chain (Chain object), days/T-to-expiry

Generates candidate strategies, prices each leg from the chain (uses LTP, falls
back to mid if available), computes:
  - cost / max profit / max loss
  - breakevens
  - probability of profit (lognormal at expiry, with drift = view * conviction * IV)
  - expected payoff
  - risk-reward, return-on-capital
Then ranks by a composite score weighted toward user's chosen target win-rate.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

from .chain import Chain, Quote
from .probability import prob_above, prob_below, prob_between, view_to_drift, expected_move
from . import strategies as strat


@dataclass
class TradeIdea:
    strategy: strat.Strategy
    lots: int
    cost_inr: float           # debit (positive) or credit (negative)
    max_profit_inr: float
    max_loss_inr: float
    breakevens: list[float]
    prob_profit: float        # 0..1
    expected_payoff_inr: float
    return_on_risk: float     # max_profit / max_loss (capped)
    score: float              # composite ranking score
    rationale: list[str] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "name": self.strategy.name,
            "view": self.strategy.view,
            "lots": self.lots,
            "cost_inr": round(self.cost_inr, 2),
            "max_profit_inr": round(self.max_profit_inr, 2),
            "max_loss_inr": round(self.max_loss_inr, 2),
            "breakevens": self.breakevens,
            "prob_profit": round(self.prob_profit, 4),
            "expected_payoff_inr": round(self.expected_payoff_inr, 2),
            "return_on_risk": round(self.return_on_risk, 3),
            "score": round(self.score, 4),
            "rationale": self.rationale,
            "legs": [
                {
                    "type": l.option_type, "strike": l.strike,
                    "action": l.action, "premium": round(l.premium, 2),
                    "iv": round(l.iv, 4) if l.iv else None,
                }
                for l in self.strategy.legs
            ],
        }


def _pricer_from_chain(chain: Chain):
    by_strike = chain.by_strike()
    def pricer(strike: int, option_type: str) -> tuple[float, float]:
        q = by_strike.get(strike, {}).get(option_type)
        if q is None or not q.ltp:
            raise KeyError(f"No quote for {strike}{option_type}")
        # mid if available, else LTP
        if q.bid and q.ask and q.ask > q.bid > 0:
            premium = 0.5 * (q.bid + q.ask)
        else:
            premium = q.ltp
        iv = q.iv if q.iv is not None else 0.20
        return premium, iv
    return pricer


def _atm_iv(chain: Chain) -> float:
    atm = chain.atm_strike()
    by_strike = chain.by_strike()
    ivs = []
    for k in (atm - 50, atm, atm + 50):
        for ot in ("CE", "PE"):
            q = by_strike.get(k, {}).get(ot)
            if q and q.iv:
                ivs.append(q.iv)
    if not ivs:
        return 0.18
    return sum(ivs) / len(ivs)


def _expected_payoff(stg: strat.Strategy, spot: float, t: float, iv: float,
                     drift: float, spot_min: float, spot_max: float, step: float = 10.0) -> float:
    """Numerically integrate payoff(S_T) * lognormal_pdf(S_T) over [spot_min, spot_max]."""
    import math
    if iv <= 0 or t <= 0:
        return stg.payoff_at(spot)
    sigma_t = iv * math.sqrt(t)
    mu = math.log(spot) + (drift - 0.5 * iv * iv) * t
    total = 0.0
    s = spot_min
    prev_cdf = 0.0
    levels: list[tuple[float, float]] = []
    while s <= spot_max:
        z = (math.log(s) - mu) / sigma_t
        cdf = 0.5 * (1.0 + math.erf(z / math.sqrt(2)))
        levels.append((s, cdf))
        s += step
    for i in range(1, len(levels)):
        s_mid = 0.5 * (levels[i-1][0] + levels[i][0])
        weight = levels[i][1] - levels[i-1][1]
        total += stg.payoff_at(s_mid) * weight
    return total


def _prob_profit_for(stg: strat.Strategy, bes: list[float], spot: float,
                     t: float, iv: float, drift: float) -> float:
    """
    Estimate P(profit at expiry). For monotone payoff (long call/put, spreads),
    profit region is one of: (-inf, BE], [BE, +inf), [BE_low, BE_hi], or its
    complement. Determine which side is profitable by sampling.
    """
    if not bes:
        # No breakeven — either always profit or always loss
        return 1.0 if stg.payoff_at(spot) > 0 else 0.0

    # Sample payoff well outside each BE to classify
    if len(bes) == 1:
        be = bes[0]
        below = stg.payoff_at(max(be - 100, 1.0))
        above = stg.payoff_at(be + 100)
        if above > 0 and below <= 0:
            return prob_above(spot, be, t, iv, drift)
        if below > 0 and above <= 0:
            return prob_below(spot, be, t, iv, drift)
        # both sides same sign — pathological
        return 1.0 if above > 0 else 0.0

    # Two breakevens — profit either between them or outside
    lo, hi = sorted(bes[:2])
    mid_payoff = stg.payoff_at(0.5 * (lo + hi))
    if mid_payoff > 0:
        return prob_between(spot, lo, hi, t, iv, drift)
    # Outside region
    return 1.0 - prob_between(spot, lo, hi, t, iv, drift)


def _max_lots_for_capital(stg: strat.Strategy, max_cap: float) -> int:
    """Risk-budget sizing.  For debit trades use debit; for credit use max_loss."""
    if max_cap <= 0:
        return 0
    cost = stg.cost_per_lot()
    if cost > 0:
        return max(int(max_cap // cost), 0)
    # credit — need to bound max_loss
    # search [-2000, +2000] of spot? caller pre-knows; do a broad scan.
    # We'll compute later — for now allow up to 5 lots on credit; recommender refines.
    return 5


def recommend(
    chain: Chain,
    t_years: float,
    view: str,
    conviction: float,
    target_win_pct: float = 0.6,
    max_capital_inr: float = 50_000.0,
    smart_money_score: float = 0.0,
    lot_size: int = 75,
    strike_step: int = 50,
    top_n: int = 5,
    event_risk=None,         # engine.events.EventRisk | None
    iv_rank_info=None,       # engine.iv_rank.IVRankResult | None
    include_rejected: bool = False,  # also return ideas below target_win
    allow_naked_shorts: bool = False,
) -> list[TradeIdea] | tuple[list[TradeIdea], list[TradeIdea]]:
    """Return up to top_n ranked trade ideas matching the user's criteria."""
    spot = chain.spot
    atm = chain.atm_strike(strike_step)
    iv = _atm_iv(chain)
    pricer = _pricer_from_chain(chain)

    # Adjust effective conviction with smart-money bias if directions align
    eff_conviction = conviction
    sm_aligned = False
    if view == "bullish" and smart_money_score > 0:
        eff_conviction = min(1.0, conviction + 0.3 * smart_money_score)
        sm_aligned = True
    elif view == "bearish" and smart_money_score < 0:
        eff_conviction = min(1.0, conviction + 0.3 * abs(smart_money_score))
        sm_aligned = True
    elif view == "neutral":
        # Neutral — smart-money fights us if it's strong directional
        eff_conviction = max(0.0, conviction - 0.5 * abs(smart_money_score))

    drift = view_to_drift(view if view in ("bullish", "bearish", "neutral") else "neutral",
                          eff_conviction, iv)
    one_sigma = expected_move(spot, t_years, iv)

    # Build candidate strikes — scaled by IV-implied 1-sigma move, not fixed steps.
    # This is the difference between a toy candidate set and one that actually
    # produces high-prob iron condors when IV is high or DTE is long.
    def _round_step(x: float) -> int:
        return int(round(x / strike_step) * strike_step)

    # Strikes at 0.5, 1.0, 1.5, 2.0 sigma each side, rounded to grid.
    sig = one_sigma
    otm_p5 = _round_step(spot + 0.5 * sig)
    otm_1s = _round_step(spot + 1.0 * sig)
    otm_1p5 = _round_step(spot + 1.5 * sig)
    otm_2s = _round_step(spot + 2.0 * sig)
    itm_p5 = _round_step(spot - 0.5 * sig)
    itm_1s = _round_step(spot - 1.0 * sig)
    itm_1p5 = _round_step(spot - 1.5 * sig)
    itm_2s = _round_step(spot - 2.0 * sig)

    # Adjacent-strike legs for tight directional plays
    one = strike_step
    otm1 = atm + one
    otm2 = atm + 2 * one
    itm1 = atm - one
    itm2 = atm - 2 * one

    candidates: list[strat.Strategy] = []

    def safe(fn, *args):
        try:
            candidates.append(fn(*args, pricer, lot_size))
        except KeyError:
            pass  # missing strike

    if view in ("bullish",):
        safe(strat.long_call, atm)
        safe(strat.long_call, otm_p5)
        # Debit spreads — width tied to expected move
        safe(strat.bull_call_spread, atm, otm_1s)
        safe(strat.bull_call_spread, otm_p5, otm_1p5)
        # Credit spreads at various OTM-ness
        safe(strat.bull_put_spread, itm_p5, itm_1p5)
        safe(strat.bull_put_spread, itm_1s, itm_2s)
        safe(strat.bull_put_spread, atm, itm_1s)
    if view in ("bearish",):
        safe(strat.long_put, atm)
        safe(strat.long_put, itm_p5)
        safe(strat.bear_put_spread, atm, itm_1s)
        safe(strat.bear_put_spread, itm_p5, itm_1p5)
        safe(strat.bear_call_spread, otm_p5, otm_1p5)
        safe(strat.bear_call_spread, otm_1s, otm_2s)
        safe(strat.bear_call_spread, atm, otm_1s)
    if view in ("neutral",):
        # Iron condors at multiple widths — wider = higher P(win), smaller credit.
        # Short legs at ~1σ each side give ~68% mathematical win-rate on a flat-drift assumption.
        safe(strat.iron_condor, itm_2s, itm_1s, otm_1s, otm_2s)        # ~1σ shorts
        safe(strat.iron_condor, itm_1p5, itm_p5, otm_p5, otm_1p5)      # ~0.5σ shorts (tighter, more credit)
        # Wider wings — higher P(win), worse R/R
        safe(strat.iron_condor, _round_step(spot - 2.5*sig), itm_1s,
             otm_1s, _round_step(spot + 2.5*sig))
        # Bigger profit zone iron condor (~1.5σ shorts) — for low-conviction neutral
        safe(strat.iron_condor, _round_step(spot - 2.5*sig), itm_1p5,
             otm_1p5, _round_step(spot + 2.5*sig))
    if view in ("vol_long",):
        safe(strat.long_straddle, atm)
        safe(strat.long_strangle, otm_p5, itm_p5)
        safe(strat.long_strangle, otm_1s, itm_1s)

    spot_min = spot - 5 * one_sigma
    spot_max = spot + 5 * one_sigma

    # Event + IV-rank gating
    iv_rank_val = iv_rank_info.value if iv_rank_info else None
    block_debits = False
    prefer_credits = False
    gate_reasons: list[str] = []
    if event_risk and event_risk.spans_event:
        if not event_risk.is_safe_for_debit(iv_rank_val):
            block_debits = True
            prefer_credits = True
            gate_reasons.append("event-in-window + IV not cheap -> debit trades blocked")
    if iv_rank_info and iv_rank_info.bucket in ("rich", "extreme"):
        prefer_credits = True
        gate_reasons.append(f"IV bucket = {iv_rank_info.bucket} -> credit structures preferred")
    if iv_rank_info and iv_rank_info.bucket == "cheap":
        gate_reasons.append("IV bucket = cheap -> debit structures favored")

    ideas: list[TradeIdea] = []
    for stg in candidates:
        try:
            bes = stg.breakevens(spot_min, spot_max, step=1.0)
            max_p, max_l = stg.max_profit_loss(spot_min, spot_max, step=10.0)
            cost = stg.cost_per_lot()
        except Exception:
            continue

        is_debit = cost > 0
        is_credit = cost < 0

        if block_debits and is_debit:
            continue

        # Per-lot risk for sizing
        risk_per_lot = max(cost, -max_l) if cost > 0 else abs(max_l)
        if risk_per_lot <= 0:
            continue
        lots = int(max_capital_inr // risk_per_lot)
        if lots < 1:
            continue

        # Win prob and expected payoff per lot
        pop = _prob_profit_for(stg, bes, spot, t_years, iv, drift)
        ep = _expected_payoff(stg, spot, t_years, iv, drift, spot_min, spot_max, step=10.0) * stg.lot_size

        below_target = pop < target_win_pct
        if below_target and not include_rejected:
            continue

        ror = (max_p / abs(max_l)) if max_l < 0 else 0.0
        # Composite score: weight win-prob, expected payoff scaled by capital, and R/R
        ep_norm = ep / max(risk_per_lot, 1.0)
        score = 0.5 * pop + 0.35 * ep_norm + 0.15 * min(ror, 5.0) / 5.0

        # Vol-regime tilt: penalize debit trades when vol is rich, reward credits;
        # opposite when vol is cheap. Modest tilt so user prefs still dominate.
        if prefer_credits:
            if is_credit:
                score *= 1.15
            elif is_debit:
                score *= 0.80
        if iv_rank_info and iv_rank_info.bucket == "cheap":
            if is_debit:
                score *= 1.10
            elif is_credit:
                score *= 0.90

        rationale = [
            f"ATM IV ~ {iv*100:.1f}%, 1-sigma move ~ +/-{one_sigma:.0f} pts over {t_years*365:.1f}d",
            f"P(profit @ expiry) ~ {pop*100:.1f}%, drift used = {drift*100:+.1f}%/yr",
            f"Risk per lot ~ Rs {risk_per_lot:,.0f}; sized to {lots} lot(s) within Rs {max_capital_inr:,.0f} budget",
        ]
        if sm_aligned:
            rationale.append(f"FII/DII positioning ({smart_money_score:+.2f}) aligns with view — conviction boosted to {eff_conviction:.2f}")
        elif abs(smart_money_score) > 0.3 and view != "neutral":
            opposing = (smart_money_score > 0 and view == "bearish") or (smart_money_score < 0 and view == "bullish")
            if opposing:
                rationale.append(f"WARNING: smart money ({smart_money_score:+.2f}) opposes your view")
        if iv_rank_info:
            rationale.extend(iv_rank_info.to_rationale())
        if event_risk:
            rationale.extend(event_risk.to_rationale())
        for g in gate_reasons:
            rationale.append(f"gate: {g}")

        idea = TradeIdea(
            strategy=stg, lots=lots,
            cost_inr=cost * lots,
            max_profit_inr=max_p * lots,
            max_loss_inr=max_l * lots,
            breakevens=bes,
            prob_profit=pop,
            expected_payoff_inr=ep * lots,
            return_on_risk=ror,
            score=score,
            rationale=rationale,
        )
        # Tag rejected ones
        if below_target:
            idea.rationale.insert(0, f"REJECTED (P(win)={pop*100:.1f}% < target {target_win_pct*100:.0f}%)")
        ideas.append(idea)

    qualified = [i for i in ideas if not any(r.startswith("REJECTED") for r in i.rationale)]
    rejected = [i for i in ideas if any(r.startswith("REJECTED") for r in i.rationale)]
    qualified.sort(key=lambda i: i.score, reverse=True)
    rejected.sort(key=lambda i: i.prob_profit, reverse=True)

    if include_rejected:
        return qualified[:top_n], rejected[:top_n]
    return qualified[:top_n]
