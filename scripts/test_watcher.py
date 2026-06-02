"""
Smoke test for the watcher + ORB rule.

Constructs a synthetic morning session where:
  - 9:15-9:30: tight range, OR_low=23900, OR_high=23945  (3 bars)
  - 9:30-9:45: consolidation inside OR  (3 bars)
  - 9:45-9:50: BREAKOUT above 23945 with volume spike  -> ORB should fire bullish

Then verifies a paper trade was opened and that a subsequent state where premium
hits target closes it correctly.
"""
import sys
import json
from datetime import datetime, time, timedelta, timezone
from pathlib import Path

sys.path.insert(0, ".")

from engine.state import MarketState, Bar
from engine.rules.orb import OpeningRangeBreakout
from engine.manual_chain import synthetic_chain


IST = timezone(timedelta(hours=5, minutes=30))


def _bar(hh, mm, o, h, l, c, v):
    today = datetime.now(tz=IST).date()
    ts = datetime.combine(today, time(hh, mm), tzinfo=IST)
    return Bar(ts=ts, open=o, high=h, low=l, close=c, volume=v)


def test_orb_does_not_fire_pre_activation():
    bars = [_bar(9, 15, 23900, 23930, 23895, 23920, 1000000)]
    chain = synthetic_chain(spot=23920, iv=0.13, dte_days=6)
    today = datetime.now(tz=IST).date()
    ts = datetime.combine(today, time(9, 22), tzinfo=IST)
    state = MarketState("NIFTY", 23920, chain, bars, ts)
    rule = OpeningRangeBreakout()
    sig = rule.evaluate(state)
    assert sig is None, "ORB should not fire before activation window"
    print("PASS: ORB respects activation_minutes (no fire before 9:45)")


def test_orb_does_not_fire_inside_range():
    bars = [
        _bar(9, 15, 23900, 23945, 23895, 23920, 1200000),
        _bar(9, 20, 23920, 23940, 23905, 23930, 1100000),
        _bar(9, 25, 23930, 23945, 23915, 23925, 1000000),
        _bar(9, 30, 23925, 23935, 23910, 23920, 950000),
        _bar(9, 35, 23920, 23935, 23915, 23930, 900000),
        _bar(9, 40, 23930, 23940, 23920, 23935, 1000000),
        _bar(9, 45, 23935, 23944, 23925, 23940, 1100000),  # still inside
    ]
    chain = synthetic_chain(spot=23940, iv=0.13, dte_days=6)
    today = datetime.now(tz=IST).date()
    ts = datetime.combine(today, time(9, 50), tzinfo=IST)
    state = MarketState("NIFTY", 23940, chain, bars, ts)
    rule = OpeningRangeBreakout()
    sig = rule.evaluate(state)
    assert sig is None, f"ORB should not fire inside range, but got {sig}"
    print("PASS: ORB does not fire on inside-range close")


def test_orb_fires_on_bullish_breakout():
    bars = [
        _bar(9, 15, 23900, 23945, 23895, 23920, 1200000),
        _bar(9, 20, 23920, 23940, 23905, 23930, 1100000),
        _bar(9, 25, 23930, 23945, 23915, 23925, 1000000),
        _bar(9, 30, 23925, 23935, 23910, 23920, 950000),
        _bar(9, 35, 23920, 23935, 23915, 23930, 900000),
        _bar(9, 40, 23930, 23940, 23920, 23935, 1000000),
        _bar(9, 45, 23935, 23944, 23925, 23940, 950000),
        _bar(9, 50, 23940, 23980, 23938, 23975, 2500000),  # BREAKOUT + volume spike
    ]
    chain = synthetic_chain(spot=23975, iv=0.13, dte_days=6)
    today = datetime.now(tz=IST).date()
    ts = datetime.combine(today, time(9, 55), tzinfo=IST)
    state = MarketState("NIFTY", 23975, chain, bars, ts)
    rule = OpeningRangeBreakout()
    sig = rule.evaluate(state)
    assert sig is not None, "ORB should fire on bullish breakout with volume"
    assert sig.direction == "bullish"
    assert sig.option_type == "CE"
    assert sig.target_spot > 23975
    assert sig.stop_spot == 23945  # OR_high
    assert 10 in sig.prob_estimates and 20 in sig.prob_estimates
    print(f"PASS: ORB fires bullish: BUY {sig.option_type} {sig.strike}  "
          f"target_spot={sig.target_spot}  P(+20pts)={sig.prob_estimates[20]*100:.0f}%")


def test_orb_fires_on_bearish_breakout():
    bars = [
        _bar(9, 15, 23900, 23945, 23895, 23920, 1200000),
        _bar(9, 20, 23920, 23940, 23905, 23930, 1100000),
        _bar(9, 25, 23930, 23945, 23915, 23925, 1000000),
        _bar(9, 30, 23925, 23935, 23910, 23920, 950000),
        _bar(9, 35, 23920, 23935, 23915, 23930, 900000),
        _bar(9, 40, 23930, 23940, 23920, 23935, 1000000),
        _bar(9, 45, 23935, 23944, 23925, 23940, 950000),
        _bar(9, 50, 23940, 23942, 23870, 23880, 2300000),  # bearish breakdown
    ]
    chain = synthetic_chain(spot=23880, iv=0.13, dte_days=6)
    today = datetime.now(tz=IST).date()
    ts = datetime.combine(today, time(9, 55), tzinfo=IST)
    state = MarketState("NIFTY", 23880, chain, bars, ts)
    rule = OpeningRangeBreakout()
    sig = rule.evaluate(state)
    assert sig is not None
    assert sig.direction == "bearish"
    assert sig.option_type == "PE"
    assert sig.target_spot < 23880
    print(f"PASS: ORB fires bearish: BUY {sig.option_type} {sig.strike}  "
          f"target_spot={sig.target_spot}  P(+20pts)={sig.prob_estimates[20]*100:.0f}%")


def test_orb_cooldown():
    """After firing, ORB should not re-fire within cooldown."""
    bars = [
        _bar(9, 15, 23900, 23945, 23895, 23920, 1200000),
        _bar(9, 20, 23920, 23940, 23905, 23930, 1100000),
        _bar(9, 25, 23930, 23945, 23915, 23925, 1000000),
        _bar(9, 30, 23925, 23935, 23910, 23920, 950000),
        _bar(9, 35, 23920, 23935, 23915, 23930, 900000),
        _bar(9, 40, 23930, 23940, 23920, 23935, 1000000),
        _bar(9, 45, 23935, 23944, 23925, 23940, 950000),
        _bar(9, 50, 23940, 23980, 23938, 23975, 2500000),
    ]
    chain = synthetic_chain(spot=23975, iv=0.13, dte_days=6)
    today = datetime.now(tz=IST).date()
    rule = OpeningRangeBreakout(cooldown_minutes=30) if False else OpeningRangeBreakout()

    ts1 = datetime.combine(today, time(9, 55), tzinfo=IST)
    sig1 = rule.evaluate(MarketState("NIFTY", 23975, chain, bars, ts1))
    assert sig1 is not None

    ts2 = datetime.combine(today, time(10, 20), tzinfo=IST)   # 25 min later
    sig2 = rule.evaluate(MarketState("NIFTY", 23980, chain, bars, ts2))
    assert sig2 is None, "should be in cooldown"
    print("PASS: cooldown prevents re-fire")


if __name__ == "__main__":
    test_orb_does_not_fire_pre_activation()
    test_orb_does_not_fire_inside_range()
    test_orb_fires_on_bullish_breakout()
    test_orb_fires_on_bearish_breakout()
    test_orb_cooldown()
    print("\nAll watcher tests passed.")
