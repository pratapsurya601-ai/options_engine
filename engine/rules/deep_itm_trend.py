"""
Deep-ITM Trend rule — designed to ride trends without theta destroying you.

Same EMA20/EMA50 cross signal as simple_ema_cross BUT buys deep-ITM options
(delta ~0.75-0.85) instead of ATM. Behaves much more like futures:
  - Theta drag ~30% of ATM
  - Adverse 30-50pt spot moves don't catastrophically dent premium
  - Captures bigger % of trending moves
  - Costs 2-3x more capital per lot

Additional entry mode: continuation on established trend.
If EMA gap has been same-sign for `continuation_bars` (default 30 = 60 min on 2m),
and spot pulls back within `continuation_pullback_pct` of EMA20, allow re-entry.
This catches trend-day moves that started before your cross window.

Exits:
  - Premium drops `stop_premium_pct` from entry (default 25% = -25%)
  - Premium gains `target_premium_pct` (default 40%)
  - Time exit at `hold_minutes` (default 240 = 4 hours)
  - End of session (15:15)
"""
from __future__ import annotations

from datetime import time

from ..signals import BaseRule, Signal
from ..state import MarketState
from ..pricing import black_scholes
from ..indicators import ema, crossed_above, crossed_below


class DeepItmTrend(BaseRule):
    name = "deep_itm_trend"
    cooldown_minutes = 30

    def __init__(
        self,
        ema_fast: int = 20,
        ema_slow: int = 50,
        cross_recency_bars: int = 1,
        # Continuation entry knobs
        enable_continuation: bool = True,
        continuation_bars: int = 30,             # EMAs same-sign for this many bars
        continuation_pullback_pct: float = 0.005, # spot within 0.5% of EMA20
        # Deep-ITM strike selection
        itm_offset_steps: int = 4,    # 4 × 50 = 200 pts ITM (delta ~0.80)
        # Exit knobs
        target_premium_pct: float = 0.40,         # +40% premium = book
        stop_premium_pct: float = 0.25,           # -25% premium = stop
        hold_minutes: int = 240,                  # 4 hours
        eod_close_minutes: int = 360,             # 15:15
        # Time window
        time_window_start: int = 30,              # 9:45
        time_window_end: int = 270,               # 13:45 (no fires after lunch peaks)
        strike_step: int = 50,
        r: float = 0.065,
    ):
        super().__init__()
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.cross_recency_bars = cross_recency_bars
        self.enable_continuation = enable_continuation
        self.continuation_bars = continuation_bars
        self.continuation_pullback_pct = continuation_pullback_pct
        self.itm_offset_steps = itm_offset_steps
        self.target_premium_pct = target_premium_pct
        self.stop_premium_pct = stop_premium_pct
        self.hold_minutes = hold_minutes
        self.eod_close_minutes = eod_close_minutes
        self.time_window_start = time_window_start
        self.time_window_end = time_window_end
        self.strike_step = strike_step
        self.r = r

    def _check_continuation(self, ef, es, spot) -> str | None:
        """Detect pullback-in-trend opportunity (no fresh cross required)."""
        n = self.continuation_bars
        if len(ef) < n or len(es) < n:
            return None
        signs = []
        for i in range(-n, 0):
            if ef[i] is None or es[i] is None:
                return None
            signs.append(ef[i] > es[i])
        if all(signs):
            trend = "bullish"
        elif not any(signs):
            trend = "bearish"
        else:
            return None
        ema20_now = ef[-1]
        if ema20_now is None or ema20_now <= 0:
            return None
        dist_pct = abs(spot - ema20_now) / ema20_now
        if dist_pct > self.continuation_pullback_pct:
            return None
        return trend

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

        direction = None
        entry_mode = None
        if crossed_above(ef, es, self.cross_recency_bars):
            direction, entry_mode = "bullish", "cross"
        elif crossed_below(ef, es, self.cross_recency_bars):
            direction, entry_mode = "bearish", "cross"
        elif self.enable_continuation:
            t = self._check_continuation(ef, es, state.spot)
            if t:
                direction, entry_mode = t, "continuation"
        if direction is None:
            return None

        spot = state.spot
        opt_type = "CE" if direction == "bullish" else "PE"

        # Deep ITM strike selection
        atm_base = round(spot / self.strike_step) * self.strike_step
        if direction == "bullish":
            # Bullish CE -> ITM is strike BELOW spot
            strike = atm_base - self.itm_offset_steps * self.strike_step
        else:
            # Bearish PE -> ITM is strike ABOVE spot
            strike = atm_base + self.itm_offset_steps * self.strike_step

        if state.chain is None:
            return None
        by_strike = state.chain.by_strike()
        q = by_strike.get(strike, {}).get(opt_type)
        if q is None or q.iv is None or not q.ltp:
            avail = [k for k in by_strike if opt_type in by_strike[k]
                     and by_strike[k][opt_type].iv is not None]
            if not avail:
                return None
            # Pick deepest available ITM still respecting offset roughly
            if direction == "bullish":
                itm_avail = [k for k in avail if k < spot]
            else:
                itm_avail = [k for k in avail if k > spot]
            if itm_avail:
                strike = min(itm_avail, key=lambda k: abs(k - (atm_base - self.itm_offset_steps * self.strike_step)
                              if direction == "bullish"
                              else atm_base + self.itm_offset_steps * self.strike_step))
            else:
                strike = min(avail, key=lambda k: abs(k - spot))
            q = by_strike[strike][opt_type]

        from datetime import datetime
        from ..chain import IST as CHAIN_IST
        expiry_close = datetime.combine(state.chain.expiry, time(15, 30), tzinfo=CHAIN_IST)
        t_expiry = max((expiry_close - state.ts).total_seconds() / (365 * 24 * 3600), 1e-5)
        g = black_scholes(spot, strike, t_expiry, q.iv, r=self.r, q=0.0, option_type=opt_type)
        entry_premium = g.price

        target_premium = entry_premium * (1 + self.target_premium_pct)
        stop_premium = max(0.5, entry_premium * (1 - self.stop_premium_pct))
        target_premium_gain = entry_premium * self.target_premium_pct

        thesis = (
            f"Deep-ITM trend {direction} [{entry_mode} entry, delta={g.delta:.2f}]: "
            f"EMA20={ef[-1]:.0f}, EMA50={es[-1]:.0f}. "
            f"BUY {opt_type} {strike} @ ~Rs {entry_premium:.0f}. "
            f"TARGET +{self.target_premium_pct*100:.0f}% (=Rs {target_premium:.0f}). "
            f"STOP -{self.stop_premium_pct*100:.0f}% (=Rs {stop_premium:.0f}). "
            f"Hold <= {self.hold_minutes}m, EOD close at 15:15. "
            f"Premise: deep ITM behaves like futures — low theta drag, "
            f"survives chop while waiting for the real trend move."
        )

        return Signal(
            rule_name=self.name,
            ts=state.ts,
            direction=direction,
            action=f"BUY_{opt_type}",
            strike=strike,
            option_type=opt_type,
            target_premium_gain=target_premium_gain,
            target_spot=None,
            stop_spot=None,
            stop_premium=round(stop_premium, 2),
            hold_minutes=self.hold_minutes,
            prob_estimates={},
            trigger_context={
                "entry_mode": entry_mode,
                "ema_fast": ef[-1], "ema_slow": es[-1],
                "spot": spot, "strike": strike, "iv": q.iv,
                "delta": g.delta,
                "entry_premium": entry_premium,
                "target_premium_pct": self.target_premium_pct,
                "stop_premium_pct": self.stop_premium_pct,
                "eod_close_minutes": self.eod_close_minutes,
            },
            thesis=thesis[:1900],
        )
