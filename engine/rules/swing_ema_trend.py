"""
Swing EMA Trend rule — designed to ride slow grinding moves.

Differences vs simple_ema_cross:
  - Wider stops (default ~25 pts premium)
  - Longer hold (default 4 hours, or session close)
  - Higher target (default 35-50 pts premium)
  - Cooldown 60 min (vs 10 min)
  - Two entry modes:
      A) Fresh EMA20/EMA50 cross (like scalp)
      B) PULLBACK to EMA20 in already-established trend

For mode B: if EMA20 has been below EMA50 for 5+ bars, and price rallies
back to within 0.3% of EMA20 from above, fire BUY PE. (Mirror for bullish.)

This rule will fire 1-3 times per session max but holds winners much longer.
Designed for choppy-but-trending days (50+ pt grinds over 2-4 hours).
"""
from __future__ import annotations

from datetime import time

from ..signals import BaseRule, Signal
from ..state import MarketState
from ..pricing import black_scholes
from ..probability import prob_touch_above, prob_touch_below
from ..indicators import ema, crossed_above, crossed_below


class SwingEmaTrend(BaseRule):
    name = "swing_ema_trend"
    cooldown_minutes = 60

    def __init__(
        self,
        ema_fast: int = 20,
        ema_slow: int = 50,
        # Cross entry mode
        cross_recency_bars: int = 1,
        # Pullback entry mode
        enable_pullback: bool = True,
        trend_stability_bars: int = 5,
        pullback_pct: float = 0.003,    # within 0.3% of EMA20
        # Exits — wider than scalp
        target_premium_pts: float = 35.0,
        stop_premium_pts: float = 25.0,
        hold_minutes: int = 240,         # 4 hours
        # Trail stop (default on for swing)
        trail_activation_pts: float = 12.0,   # after +12 pts gain, trail kicks in
        trail_distance_pts: float = 8.0,      # stop trails 8 pts below high-water-mark
        # Time window
        time_window_start: int = 30,
        time_window_end: int = 330,
        strike_step: int = 50,
        strike_offset_steps: int = 0,
        r: float = 0.065,
    ):
        super().__init__()
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.cross_recency_bars = cross_recency_bars
        self.enable_pullback = enable_pullback
        self.trend_stability_bars = trend_stability_bars
        self.pullback_pct = pullback_pct
        self.target_premium_pts = target_premium_pts
        self.stop_premium_pts = stop_premium_pts
        self.hold_minutes = hold_minutes
        self.trail_activation_pts = trail_activation_pts
        self.trail_distance_pts = trail_distance_pts
        self.time_window_start = time_window_start
        self.time_window_end = time_window_end
        self.strike_step = strike_step
        self.strike_offset_steps = strike_offset_steps
        self.r = r

    def _check_pullback(self, state, ef, es) -> str | None:
        """Detect pullback to EMA20 in established trend. Returns 'bullish'/'bearish'/None."""
        n = self.trend_stability_bars
        if len(ef) < n or len(es) < n:
            return None

        recent_signs = []
        for i in range(-n, 0):
            if ef[i] is None or es[i] is None:
                return None
            recent_signs.append(ef[i] > es[i])

        if all(recent_signs):
            trend = "bullish"
        elif not any(recent_signs):
            trend = "bearish"
        else:
            return None    # not stable

        spot = state.spot
        ema20_now = ef[-1]
        if ema20_now is None or ema20_now <= 0:
            return None
        distance_pct = abs(spot - ema20_now) / ema20_now
        if distance_pct > self.pullback_pct:
            return None    # too far from EMA20

        # Pullback entry direction must match trend
        if trend == "bearish" and spot >= ema20_now:
            return "bearish"   # spot rallied UP to EMA20 in bearish trend — fade
        if trend == "bullish" and spot <= ema20_now:
            return "bullish"   # spot dropped TO EMA20 in bullish trend — buy
        return None

    def _check(self, state: MarketState) -> Signal | None:
        if not state.is_market_open():
            return None
        mso = state.minutes_since_open()
        if mso < self.time_window_start or mso > self.time_window_end:
            return None

        bars = state.bars_5m
        if len(bars) < self.ema_slow + 2:
            return None

        closes = [b.close for b in bars]
        ef = ema(closes, self.ema_fast)
        es = ema(closes, self.ema_slow)
        if ef[-1] is None or es[-1] is None:
            return None

        # Entry mode A: fresh cross
        entry_mode = None
        direction = None
        if crossed_above(ef, es, self.cross_recency_bars):
            direction = "bullish"
            entry_mode = "cross"
        elif crossed_below(ef, es, self.cross_recency_bars):
            direction = "bearish"
            entry_mode = "cross"

        # Entry mode B: pullback in established trend
        if direction is None and self.enable_pullback:
            pullback = self._check_pullback(state, ef, es)
            if pullback:
                direction = pullback
                entry_mode = "pullback"

        if direction is None:
            return None

        opt_type = "CE" if direction == "bullish" else "PE"
        spot = state.spot

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

        from datetime import datetime
        from ..chain import IST as CHAIN_IST
        expiry_close = datetime.combine(state.chain.expiry, time(15, 30), tzinfo=CHAIN_IST)
        t_expiry = max((expiry_close - state.ts).total_seconds() / (365 * 24 * 3600), 1e-5)
        g = black_scholes(spot, atm, t_expiry, q.iv, r=self.r, q=0.0, option_type=opt_type)
        entry_premium = g.price

        target_premium = entry_premium + self.target_premium_pts
        stop_premium = max(0.5, entry_premium - self.stop_premium_pts)

        # Convert to spot for display
        spot_pts_for_target = self.target_premium_pts / max(abs(g.delta), 0.05)
        spot_pts_for_stop = self.stop_premium_pts / max(abs(g.delta), 0.05)
        if direction == "bullish":
            target_spot = spot + spot_pts_for_target
            stop_spot = spot - spot_pts_for_stop
        else:
            target_spot = spot - spot_pts_for_target
            stop_spot = spot + spot_pts_for_stop

        thesis = (
            f"Swing EMA trend {direction} [{entry_mode} entry]: "
            f"EMA20={ef[-1]:.0f}, EMA50={es[-1]:.0f}. "
            f"BUY {opt_type} {atm} @ ~Rs {entry_premium:.0f} (delta={g.delta:.2f}). "
            f"TARGET: +{self.target_premium_pts:.0f}pts (=Rs {target_premium:.0f}) at spot {target_spot:.0f}. "
            f"STOP: -{self.stop_premium_pts:.0f}pts (=Rs {stop_premium:.0f}) at spot {stop_spot:.0f}. "
            f"TRAIL: after +{self.trail_activation_pts}pts, stop trails {self.trail_distance_pts}pts below peak. "
            f"TIME EXIT: {self.hold_minutes}m. "
            f"Premise: ride established trend or fresh cross with patience."
        )

        return Signal(
            rule_name=self.name,
            ts=state.ts,
            direction=direction,
            action=f"BUY_{opt_type}",
            strike=atm,
            option_type=opt_type,
            target_premium_gain=self.target_premium_pts,
            target_spot=round(target_spot, 1),
            stop_spot=round(stop_spot, 1),
            stop_premium=round(stop_premium, 2),
            hold_minutes=self.hold_minutes,
            prob_estimates={},
            trigger_context={
                "entry_mode": entry_mode,
                "ema_fast": ef[-1], "ema_slow": es[-1],
                "spot": spot, "atm_strike": atm, "iv": q.iv,
                "entry_premium": entry_premium,
                "trail_activation_pts": self.trail_activation_pts,
                "trail_distance_pts": self.trail_distance_pts,
            },
            thesis=thesis[:1900],
        )
