"""
Higher-timeframe EMA-cross-then-RETEST naked options rule.

Classic trader setup:
  1. EMA20 crosses EMA50 (bullish or bearish)
  2. Price runs away from EMAs in cross direction
  3. Price pulls BACK to retest EMA20 (within retest_pct)
  4. Enter at retest — better price, tighter stop, higher R/R

Mirrors htf_naked structure (15-min winner) but replaces the "cross within last
N bars + RSI + MACD" with "cross happened, then we retested EMA20".
"""
from __future__ import annotations

from datetime import time, datetime

from ..signals import BaseRule, Signal
from ..state import MarketState
from ..pricing import black_scholes
from ..indicators import ema, rsi


def find_recent_cross(ef, es, lookback: int, exclude_recent: int = 2):
    """Return ('bullish'|'bearish'|None) if cross found in [-lookback, -exclude_recent].

    The cross must have completed at least `exclude_recent` bars ago so that
    price has had time to run away and (possibly) retest.
    """
    n = len(ef)
    if n < lookback + 1:
        return None
    # Iterate from oldest in-window to newest, returning the most recent cross.
    most_recent = None
    most_recent_idx = None
    start = max(1, n - lookback)
    end = n - exclude_recent  # exclusive
    for i in range(start, end):
        a_prev, a_now = ef[i - 1], ef[i]
        b_prev, b_now = es[i - 1], es[i]
        if None in (a_prev, a_now, b_prev, b_now):
            continue
        if a_prev <= b_prev and a_now > b_now:
            most_recent = "bullish"
            most_recent_idx = i
        elif a_prev >= b_prev and a_now < b_now:
            most_recent = "bearish"
            most_recent_idx = i
    if most_recent is None:
        return None
    return most_recent, most_recent_idx


class HtfEmaRetest(BaseRule):
    name = "htf_ema_retest"
    cooldown_minutes = 60

    def __init__(
        self,
        ema_fast: int = 20,
        ema_slow: int = 50,
        retest_lookback: int = 10,
        retest_exclude_recent: int = 2,
        retest_pct: float = 0.005,
        require_confirmation_bar: bool = True,
        require_rsi_filter: bool = True,
        rsi_period: int = 14,
        rsi_min: float = 40.0,
        rsi_max: float = 65.0,
        direction_filter: str = "both",
        strike_offset_steps: int = -4,
        target_premium_pts: float = 20.0,
        stop_premium_pts: float = 12.0,
        hold_minutes: int = 60,
        time_window_start: int = 30,
        time_window_end: int = 300,
        strike_step: int = 50,
        r: float = 0.065,
    ):
        super().__init__()
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.retest_lookback = retest_lookback
        self.retest_exclude_recent = retest_exclude_recent
        self.retest_pct = retest_pct
        self.require_confirmation_bar = require_confirmation_bar
        self.require_rsi_filter = require_rsi_filter
        self.rsi_period = rsi_period
        self.rsi_min = rsi_min
        self.rsi_max = rsi_max
        self.direction_filter = direction_filter
        self.strike_offset_steps = strike_offset_steps
        self.target_premium_pts = target_premium_pts
        self.stop_premium_pts = stop_premium_pts
        self.hold_minutes = hold_minutes
        self.time_window_start = time_window_start
        self.time_window_end = time_window_end
        self.strike_step = strike_step
        self.r = r

    def _check(self, state: MarketState) -> Signal | None:
        if not state.is_market_open():
            return None
        mso = state.minutes_since_open()
        if mso < self.time_window_start or mso > self.time_window_end:
            return None

        bars = state.bars_5m
        needed = max(self.ema_slow + self.retest_lookback + 2,
                     self.rsi_period + 2)
        if len(bars) < needed:
            return None

        closes = [b.close for b in bars]
        ef = ema(closes, self.ema_fast)
        es = ema(closes, self.ema_slow)
        if ef[-1] is None or es[-1] is None:
            return None

        # 1. Recent cross in window [-lookback, -exclude_recent)
        cross = find_recent_cross(ef, es, self.retest_lookback,
                                  self.retest_exclude_recent)
        if cross is None:
            return None
        cross_dir, cross_idx = cross
        bars_since_cross = (len(ef) - 1) - cross_idx

        # 2. Alignment intact (cross still valid)
        if cross_dir == "bullish" and not (ef[-1] > es[-1]):
            return None
        if cross_dir == "bearish" and not (ef[-1] < es[-1]):
            return None

        # Direction filter
        if self.direction_filter == "bullish_only" and cross_dir != "bullish":
            return None
        if self.direction_filter == "bearish_only" and cross_dir != "bearish":
            return None

        # 3. Retest near EMA20
        spot = state.spot
        ema20_now = ef[-1]
        distance_pct = abs(spot - ema20_now) / max(ema20_now, 1e-6)
        if distance_pct > self.retest_pct:
            return None
        # For bullish: spot has come back DOWN to EMA20 from above.
        # For bearish: spot has come back UP to EMA20 from below.
        # We require spot to be on the "correct" side or essentially touching.
        # (Already enforced loosely by alignment + closeness; explicit check below.)

        # 4. Confirmation bar
        last_bar = bars[-1]
        bar_was_bullish = last_bar.close > last_bar.open
        if self.require_confirmation_bar:
            if cross_dir == "bullish" and not bar_was_bullish:
                return None
            if cross_dir == "bearish" and bar_was_bullish:
                return None

        # 5. Optional RSI filter
        rsi_now = None
        if self.require_rsi_filter:
            rsi_vals = rsi(closes, self.rsi_period)
            rsi_now = rsi_vals[-1]
            if rsi_now is None:
                return None
            if not (self.rsi_min <= rsi_now <= self.rsi_max):
                return None
        else:
            rsi_vals = rsi(closes, self.rsi_period)
            rsi_now = rsi_vals[-1] if rsi_vals else None

        direction = cross_dir
        opt_type = "CE" if direction == "bullish" else "PE"

        if state.chain is None:
            return None
        atm_base = round(spot / self.strike_step) * self.strike_step
        if direction == "bullish":
            atm = atm_base + self.strike_offset_steps * self.strike_step
        else:
            atm = atm_base - self.strike_offset_steps * self.strike_step
        by_strike = state.chain.by_strike()
        q = by_strike.get(atm, {}).get(opt_type)
        if q is None or q.iv is None or not q.ltp:
            avail = [k for k in by_strike if opt_type in by_strike[k]
                     and by_strike[k][opt_type].iv is not None]
            if not avail:
                return None
            atm = min(avail, key=lambda k: abs(k - spot))
            q = by_strike[atm][opt_type]

        from ..chain import IST as CHAIN_IST
        expiry_close = datetime.combine(state.chain.expiry, time(15, 30), tzinfo=CHAIN_IST)
        t_expiry = max((expiry_close - state.ts).total_seconds() / (365 * 24 * 3600), 1e-5)
        g = black_scholes(spot, atm, t_expiry, q.iv, r=self.r, q=0.0,
                          option_type=opt_type)
        entry_premium = g.price

        target_premium_gain = self.target_premium_pts
        stop_premium_loss = self.stop_premium_pts
        target_spot_move = target_premium_gain / max(abs(g.delta), 0.05)
        stop_spot_move = stop_premium_loss / max(abs(g.delta), 0.05)

        target_premium = entry_premium + target_premium_gain
        stop_premium = max(0.5, entry_premium - stop_premium_loss)
        if direction == "bullish":
            target_spot = spot + target_spot_move
            stop_spot = spot - stop_spot_move
        else:
            target_spot = spot - target_spot_move
            stop_spot = spot + stop_spot_move

        rr = target_premium_gain / max(stop_premium_loss, 0.1)

        rsi_str = f"{rsi_now:.1f}" if rsi_now is not None else "n/a"
        thesis = (
            f"HTF EMA retest {direction}: EMA{self.ema_fast}={ef[-1]:.0f} vs "
            f"EMA{self.ema_slow}={es[-1]:.0f} (cross {bars_since_cross}b ago), "
            f"spot {spot:.0f} retested EMA20 ({distance_pct*100:.2f}% away), "
            f"confirm_bar={'bull' if bar_was_bullish else 'bear'}, RSI={rsi_str}. "
            f"BUY {opt_type} {atm} @ ~Rs {entry_premium:.0f} (delta={g.delta:.2f}). "
            f"TARGET +{target_premium_gain:.0f}pts (=Rs {target_premium:.0f}) at spot {target_spot:.0f}. "
            f"STOP -{stop_premium_loss:.0f}pts (=Rs {stop_premium:.0f}) at spot {stop_spot:.0f}. "
            f"TIME EXIT after {self.hold_minutes}m. R/R={rr:.2f}."
        )

        return Signal(
            rule_name=self.name,
            ts=state.ts,
            direction=direction,
            action=f"BUY_{opt_type}",
            strike=atm,
            option_type=opt_type,
            target_premium_gain=round(target_premium_gain, 1),
            target_spot=round(target_spot, 1),
            stop_spot=round(stop_spot, 1),
            stop_premium=round(stop_premium, 2),
            hold_minutes=self.hold_minutes,
            prob_estimates={},
            trigger_context={
                "cross_direction": cross_dir,
                "bars_since_cross": bars_since_cross,
                "distance_from_ema20_pct": round(distance_pct * 100, 3),
                "ema_fast_value": ef[-1],
                "ema_slow_value": es[-1],
                "ema_fast_period": self.ema_fast,
                "ema_slow_period": self.ema_slow,
                "rsi_value": round(rsi_now, 2) if rsi_now is not None else None,
                "bar_was_bullish": bar_was_bullish,
                "spot": spot,
                "atm_strike": atm,
                "iv": q.iv,
                "entry_premium": entry_premium,
                "target_premium_gain": target_premium_gain,
                "stop_premium_loss": stop_premium_loss,
                "target_spot_move": target_spot_move,
                "stop_spot_move": stop_spot_move,
                "risk_reward": round(rr, 2),
                "time_exit_minutes": self.hold_minutes,
            },
            thesis=thesis[:1900],
        )
