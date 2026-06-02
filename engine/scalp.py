"""
Naked-long option scalping mode.

Goal: buy a single CE or PE, target X rupees of premium gain within Y hours/days.

Math:
  - For a long option, premium ~ delta * spot_move - theta_day * days_held (first order)
  - To gain `target_points` on premium, spot must move ~ target_points / delta
  - P(hitting that spot level within hold_days) is a BARRIER TOUCH probability,
    not an expiry probability. The lognormal touch formula already in
    engine.probability handles this.

What we report per candidate:
  - Premium (cost per lot)
  - Delta, theta/day, gamma
  - Required spot move for target
  - P(touch) within hold window
  - Theta bleed if target NOT hit in hold window
  - Risk/reward = target_gain / theta_bleed (rough — true max loss is full premium)
  - Score: P(touch) * target - (1 - P(touch)) * theta_bleed

Honest framing:
  - 50-65% touch probability on a tight target is typical when IV is fair
  - The edge here is NOT prediction; it's matching the right strike (delta) to
    your target and time horizon. Buying far OTM "lottery tickets" for 20-point
    targets is mathematically inferior to buying 0.40-0.50 delta options.
  - Without a real directional edge, naked-long scalping is approximately
    zero-EV after slippage. The engine shows the math; it doesn't manufacture edge.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .chain import Chain, Quote
from .pricing import black_scholes
from .probability import prob_touch_above, prob_touch_below, view_to_drift


@dataclass
class ScalpIdea:
    option_type: str          # CE | PE
    strike: int
    premium: float            # per-share LTP/mid
    delta: float
    theta_day: float          # per-share theta per day (negative for long)
    gamma: float
    required_spot_move: float # absolute pts
    spot_target: float
    prob_touch: float         # 0..1 — touch within hold window
    cost_per_lot: float
    target_gain_per_lot: float
    theta_bleed_per_lot: float  # if held full window without hitting target
    max_loss_per_lot: float   # full premium
    risk_reward: float        # target_gain / theta_bleed
    expected_pnl: float       # P*target - (1-P)*theta_bleed, per lot
    score: float
    rationale: list[str] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "type": self.option_type, "strike": self.strike,
            "premium": round(self.premium, 2),
            "delta": round(self.delta, 3),
            "theta_day": round(self.theta_day, 3),
            "gamma": round(self.gamma, 5),
            "required_spot_move": round(self.required_spot_move, 1),
            "spot_target": round(self.spot_target, 1),
            "prob_touch": round(self.prob_touch, 4),
            "cost_per_lot": round(self.cost_per_lot, 0),
            "target_gain_per_lot": round(self.target_gain_per_lot, 0),
            "theta_bleed_per_lot": round(self.theta_bleed_per_lot, 0),
            "max_loss_per_lot": round(self.max_loss_per_lot, 0),
            "risk_reward": round(self.risk_reward, 2),
            "expected_pnl_per_lot": round(self.expected_pnl, 0),
            "score": round(self.score, 4),
            "rationale": self.rationale,
        }


def scalp_recommend(
    chain: Chain,
    t_years_to_expiry: float,
    view: str,
    conviction: float,
    target_points: float = 15.0,
    hold_days: float = 1.0,
    max_premium_per_lot: float = 20_000.0,
    capital_inr: float = 50_000.0,
    lot_size: int = 75,
    min_prob_touch: float = 0.40,
    top_n: int = 5,
    r: float = 0.065,
) -> list[ScalpIdea]:
    """
    Scan single-leg long CE/PE candidates that can plausibly capture
    target_points of premium within hold_days.
    """
    spot = chain.spot
    by_strike = chain.by_strike()

    # IV per strike from chain
    candidates = []
    for strike, leg_map in by_strike.items():
        for ot in ("CE", "PE"):
            q = leg_map.get(ot)
            if q is None or not q.ltp:
                continue
            if q.iv is None or q.iv <= 0:
                continue
            # Filter by view — bullish wants CE, bearish wants PE, vol_long takes both
            if view == "bullish" and ot != "CE":
                continue
            if view == "bearish" and ot != "PE":
                continue
            candidates.append((strike, ot, q))

    # Drift used for touch probability (real-world view, not risk-neutral)
    drift_iv = _median_iv(by_strike, spot)
    drift = view_to_drift(view if view in ("bullish", "bearish") else "neutral",
                          conviction, drift_iv)
    hold_t = hold_days / 365.0

    ideas: list[ScalpIdea] = []
    for strike, ot, q in candidates:
        premium = q.ltp
        if premium * lot_size > max_premium_per_lot:
            continue
        if premium * lot_size > capital_inr:
            continue

        g = black_scholes(spot, strike, t_years_to_expiry, q.iv, r=r, q=0.0, option_type=ot)
        if abs(g.delta) < 0.05:
            continue   # too far OTM — delta too small for target to be reachable cheaply

        # Required spot move to gain `target_points` on premium (first-order)
        required_move = target_points / abs(g.delta)
        if required_move <= 0 or required_move > 5 * spot * q.iv * (hold_t ** 0.5):
            # Would need > 5 sigma — astronomical
            continue

        # P(touch) of spot target within hold window
        if ot == "CE":
            spot_target = spot + required_move
            p_touch = prob_touch_above(spot, spot_target, hold_t, q.iv, drift)
        else:
            spot_target = spot - required_move
            p_touch = prob_touch_below(spot, spot_target, hold_t, q.iv, drift)

        if p_touch < min_prob_touch:
            continue

        cost = premium * lot_size
        gain = target_points * lot_size
        bleed = abs(g.theta_day) * hold_days * lot_size
        max_loss = cost   # worst case: full premium gone

        rr = gain / max(bleed, 1.0)
        ep = p_touch * gain - (1 - p_touch) * bleed
        # Score: weight touch prob highest, then EV, then capital efficiency
        score = 0.5 * p_touch + 0.3 * (ep / max(cost, 1.0)) + 0.2 * min(rr, 5.0) / 5.0

        rationale = [
            f"{ot} delta={g.delta:.2f}, theta/day=Rs {g.theta_day:.2f}, gamma={g.gamma:.4f}",
            f"To gain {target_points} pts on premium, spot must move {required_move:.0f} pts ({'up' if ot=='CE' else 'down'}) to {spot_target:.0f}",
            f"P(touch) within {hold_days:.1f}d = {p_touch*100:.1f}% (IV={q.iv*100:.1f}%, drift={drift*100:+.1f}%/yr)",
            f"If target NOT hit and held {hold_days:.1f}d: theta bleed Rs {bleed:,.0f}/lot",
            f"Worst case (full decay): -Rs {max_loss:,.0f}/lot — full premium",
        ]
        if abs(g.delta) < 0.20:
            rationale.append("WARNING: far-OTM lottery — required move is large relative to IV; touch math fragile here")
        if hold_days < 0.5:
            rationale.append("Intraday: theta bleed dominates if no move within first 2-3 hours; cut quickly")

        ideas.append(ScalpIdea(
            option_type=ot, strike=strike, premium=premium,
            delta=g.delta, theta_day=g.theta_day, gamma=g.gamma,
            required_spot_move=required_move, spot_target=spot_target,
            prob_touch=p_touch,
            cost_per_lot=cost, target_gain_per_lot=gain,
            theta_bleed_per_lot=bleed, max_loss_per_lot=max_loss,
            risk_reward=rr, expected_pnl=ep, score=score,
            rationale=rationale,
        ))

    ideas.sort(key=lambda i: i.score, reverse=True)
    return ideas[:top_n]


def _median_iv(by_strike: dict, spot: float) -> float:
    ivs = []
    for k, legs in by_strike.items():
        if abs(k - spot) / spot > 0.02:
            continue
        for q in legs.values():
            if q.iv and q.iv > 0:
                ivs.append(q.iv)
    if not ivs:
        return 0.15
    ivs.sort()
    return ivs[len(ivs) // 2]
