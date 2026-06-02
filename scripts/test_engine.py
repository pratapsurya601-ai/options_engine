"""Smoke tests for engine math — no network, no DB."""
import math
import sys
from datetime import date, datetime, timedelta, timezone

sys.path.insert(0, ".")

from engine.pricing import black_scholes, implied_vol
from engine.probability import prob_above, prob_below, prob_touch_above, expected_move
from engine.strategies import long_call, bull_call_spread, iron_condor
from engine.chain import Chain, Quote
from engine.recommender import recommend
from engine.smart_money import score_latest


IST = timezone(timedelta(hours=5, minutes=30))


def test_bs_put_call_parity():
    """C - P = S*e^-qT - K*e^-rT"""
    s, k, t, iv, r = 25000.0, 25000.0, 7/365, 0.15, 0.065
    c = black_scholes(s, k, t, iv, r, 0.0, "CE").price
    p = black_scholes(s, k, t, iv, r, 0.0, "PE").price
    lhs = c - p
    rhs = s - k * math.exp(-r * t)
    assert abs(lhs - rhs) < 0.5, f"put-call parity broken: {lhs} vs {rhs}"
    print(f"PASS put-call parity: C-P={lhs:.3f}  S-K*e^-rT={rhs:.3f}")


def test_iv_roundtrip():
    """Price -> IV -> Price should match."""
    s, k, t, iv_true, r = 25000.0, 25100.0, 14/365, 0.18, 0.065
    price = black_scholes(s, k, t, iv_true, r, 0.0, "CE").price
    iv_back = implied_vol(price, s, k, t, r, 0.0, "CE")
    assert iv_back is not None
    assert abs(iv_back - iv_true) < 1e-3, f"IV roundtrip: {iv_back} vs {iv_true}"
    print(f"PASS IV roundtrip: input={iv_true:.4f} recovered={iv_back:.4f}")


def test_probabilities():
    s, t, iv = 25000.0, 7/365, 0.15
    em = expected_move(s, t, iv)
    p_up = prob_above(s, s + em, t, iv, drift=0.0)
    p_dn = prob_below(s, s - em, t, iv, drift=0.0)
    # Risk-neutral, 1-sigma move ~ 16% each tail (lognormal slight skew)
    assert 0.12 < p_up < 0.22, f"p_up={p_up}"
    assert 0.12 < p_dn < 0.22, f"p_dn={p_dn}"
    print(f"PASS probabilities: 1sigma={em:.0f}  P(>+1sigma)={p_up:.3f}  P(<-1sigma)={p_dn:.3f}")


def test_strategies():
    """Build a fake chain and run recommender end-to-end."""
    spot = 25000.0
    iv = 0.15
    t = 7 / 365
    expiry = (datetime.now(tz=IST) + timedelta(days=7)).date()
    quotes = []
    for k in range(24000, 26050, 50):
        for ot in ("CE", "PE"):
            price = black_scholes(spot, k, t, iv, 0.065, 0.0, ot).price
            if price < 0.5:
                continue
            quotes.append(Quote(
                symbol="NIFTY", expiry=expiry, strike=k, option_type=ot,
                ltp=price, bid=price*0.995, ask=price*1.005, iv=iv,
                oi=1000, volume=500,
                snapshot_ts=datetime.now(tz=IST),
            ))
    chain = Chain("NIFTY", spot, expiry, datetime.now(tz=IST), quotes)
    ideas = recommend(
        chain=chain, t_years=t,
        view="bullish", conviction=0.6,
        target_win_pct=0.3, max_capital_inr=50_000,
        smart_money_score=0.4,
    )
    assert ideas, "no ideas generated"
    print(f"PASS recommender: {len(ideas)} ideas")
    for i, idea in enumerate(ideas[:3], 1):
        s = idea.summary()
        print(f"  [{i}] {s['name']}  P(win)={s['prob_profit']*100:.1f}%  E[P&L]=Rs{s['expected_payoff_inr']:.0f}  cost=Rs{s['cost_inr']:.0f}")


def test_smart_money():
    row = {
        "fii_cash_net": 2500.0,
        "dii_cash_net": -800.0,
        "fii_index_fut_long_contracts": 80000,
        "fii_index_fut_short_contracts": 40000,
        "fii_index_call_long": 50000,
        "fii_index_call_short": 30000,
        "fii_index_put_long": 20000,
        "fii_index_put_short": 25000,
    }
    sig = score_latest(row)
    assert sig.score > 0.3, f"expected bullish, got {sig.score}"
    print(f"PASS smart money: score={sig.score:+.3f}  label={sig.label}")


def test_events():
    from engine.events import assess, events_between, SCHEDULED_EVENTS
    # Should have some events loaded
    assert SCHEDULED_EVENTS, "no events loaded"
    # Window spanning a year should hit multiple
    today = date(2026, 1, 15)
    end = date(2026, 12, 31)
    es = events_between(today, end)
    assert len(es) >= 10, f"expected many 2026 events, got {len(es)}"
    # assess() with event-in-window + rich IV should block debits
    er = assess(today, date(2026, 2, 7), iv_rank=0.80)
    assert er.spans_event, "should detect Budget+RBI in window"
    assert not er.is_safe_for_debit(0.80), "should block debit at rich IV"
    assert er.is_safe_for_debit(0.25), "should allow debit at cheap IV"
    print(f"PASS events: {len(es)} events in 2026; gating logic correct")


def test_iv_rank_static_fallback():
    """No DB: function should return degraded result with static bucket."""
    from engine.iv_rank import iv_rank_atm
    # Pass a dummy conn that will fail on cursor() — simulates no DB
    class DeadConn:
        def cursor(self): raise RuntimeError("no db")
    result = iv_rank_atm(DeadConn(), "NIFTY", current_iv=0.14, dte_days=7)
    assert result.confidence == "insufficient"
    assert result.bucket in ("cheap", "normal", "rich", "extreme")
    assert result.value is None
    print(f"PASS iv_rank static fallback: bucket={result.bucket} for IV=14%")


def test_recommender_with_gates():
    """Recommender should respect event+IV gates."""
    from engine.events import EventRisk, Event
    from engine.iv_rank import IVRankResult
    spot = 25000.0; iv = 0.15; t = 7/365
    expiry = (datetime.now(tz=IST) + timedelta(days=7)).date()
    quotes = []
    for k in range(24000, 26050, 50):
        for ot in ("CE", "PE"):
            price = black_scholes(spot, k, t, iv, 0.065, 0.0, ot).price
            if price < 0.5: continue
            quotes.append(Quote("NIFTY", expiry, k, ot, price, price*0.995, price*1.005,
                                iv, 1000, 500, datetime.now(tz=IST)))
    chain = Chain("NIFTY", spot, expiry, datetime.now(tz=IST), quotes)

    # Construct a rich-IV + event-spanning scenario
    er = EventRisk(
        spans_event=True,
        events=[Event("RBI MPC", expiry - timedelta(days=2), "high", "expand_before_crush_after")],
        pre_event_now=[],
        advice="blocked",
    )
    rich_iv = IVRankResult(value=0.85, percentile=0.9, current_iv=iv,
                           window_days=60, sample_count=45, confidence="medium",
                           bucket="extreme", advice="sell premium")

    from engine.recommender import recommend
    ideas = recommend(
        chain=chain, t_years=t, view="bullish", conviction=0.6,
        target_win_pct=0.3, max_capital_inr=50_000, smart_money_score=0.0,
        event_risk=er, iv_rank_info=rich_iv,
    )
    # Under gate: debits should be blocked, only credit structures remain
    for idea in ideas:
        cost = idea.cost_inr
        assert cost <= 0, f"debit trade leaked through gate: {idea.strategy.name} cost={cost}"
    print(f"PASS recommender gates: {len(ideas)} ideas, all credits (debits blocked by event+rich-IV)")


if __name__ == "__main__":
    test_bs_put_call_parity()
    test_iv_roundtrip()
    test_probabilities()
    test_strategies()
    test_smart_money()
    test_events()
    test_iv_rank_static_fallback()
    test_recommender_with_gates()
    print("\nAll tests passed.")
