"""
NIFTY Intraday Option Buyer — a complete, opinionated framework rule.

Timeframe : 5-minute bars.
Window    : 9:30 AM (mso >= 15) to 2:30 PM (mso <= 315). No fresh entries after.

Market bias:
  Bullish  (allows BUY_CE):  close > VWAP, EMA20 > EMA50,
                             last 3 highs strictly increasing AND last 3 lows
                             strictly increasing.
  Bearish  (allows BUY_PE):  close < VWAP, EMA20 < EMA50,
                             last 3 highs strictly decreasing AND last 3 lows
                             strictly decreasing.

Trigger:
  CE: price breaks above first 30-min high   AND  activity_z >= 0.3.
  PE: price breaks below first 30-min low    AND  activity_z >= 0.3.

Strike: ATM by default; configurable to 1-strike-ITM via itm_steps.
  ITM CE strike = ATM_base - itm_steps * 50    (CE ITM is below spot)
  ITM PE strike = ATM_base + itm_steps * 50    (PE ITM is above spot)

Risk : stop_premium = entry * (1 - stop_premium_pct)  (default 20%).
Target: target_premium = entry + target_rr * (entry - stop_premium)  (default 2R).

Trail (managed by paper-trade/monitor engine):
  trail_activation_pts = entry * stop_premium_pct      (= 1R gain)
  trail_distance_pts   = entry * stop_premium_pct       (peak - distance = entry => BREAKEVEN stop)
"""
from __future__ import annotations

from datetime import datetime, time

from ..signals import BaseRule, Signal
from ..state import MarketState
from ..pricing import black_scholes
from ..indicators import ema
from ..chain import IST as CHAIN_IST


class NiftyIntradayBuyer(BaseRule):
    name = "nifty_intraday_buyer"
    cooldown_minutes = 60   # max 2-3 fires per session

    def __init__(
        self,
        ema_fast: int = 20,
        ema_slow: int = 50,
        or_minutes: int = 30,
        trend_bars: int = 3,
        min_activity_z: float = 0.3,
        time_window_start: int = 15,    # 9:30
        time_window_end: int = 315,     # 14:30
        direction_filter: str = "both", # 'both' | 'bullish_only' | 'bearish_only'
        itm_steps: int = 0,             # 0 = ATM, 1 = 1-strike ITM
        strike_step: int = 50,
        stop_premium_pct: float = 0.20, # 20% of entry = stop loss
        target_rr: float = 2.0,         # target = 2R
        hold_minutes: int = 120,
        r: float = 0.065,
    ):
        super().__init__()
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.or_minutes = or_minutes
        self.trend_bars = trend_bars
        self.min_activity_z = min_activity_z
        self.time_window_start = time_window_start
        self.time_window_end = time_window_end
        self.direction_filter = direction_filter
        self.itm_steps = itm_steps
        self.strike_step = strike_step
        self.stop_premium_pct = stop_premium_pct
        self.target_rr = target_rr
        self.hold_minutes = hold_minutes
        self.r = r

    # ---- helpers ----
    def _is_hh_hl(self, bars) -> bool:
        """Last `trend_bars` bars: strictly increasing highs AND strictly increasing lows."""
        if len(bars) < self.trend_bars:
            return False
        last = bars[-self.trend_bars:]
        for i in range(1, len(last)):
            if not (last[i].high > last[i-1].high and last[i].low > last[i-1].low):
                return False
        return True

    def _is_lh_ll(self, bars) -> bool:
        if len(bars) < self.trend_bars:
            return False
        last = bars[-self.trend_bars:]
        for i in range(1, len(last)):
            if not (last[i].high < last[i-1].high and last[i].low < last[i-1].low):
                return False
        return True

    def _check(self, state: MarketState) -> Signal | None:
        if not state.is_market_open():
            return None
        mso = state.minutes_since_open()
        if mso < self.time_window_start or mso > self.time_window_end:
            return None

        bars = state.bars_5m
        if len(bars) < max(self.ema_slow + 2, self.trend_bars + 1):
            return None

        last_bar = bars[-1]
        close = last_bar.close

        # VWAP
        vwap = state.vwap()
        if vwap is None:
            return None

        # EMAs
        closes = [b.close for b in bars]
        ef = ema(closes, self.ema_fast)
        es = ema(closes, self.ema_slow)
        if ef[-1] is None or es[-1] is None:
            return None

        # Opening range
        orng = state.opening_range(self.or_minutes)
        if orng is None:
            return None
        or_low, or_high = orng

        # Don't trade until opening range is complete
        if mso < self.or_minutes:
            return None

        # Activity / volume
        activity_z = state.activity_z(14)
        if activity_z is None:
            activity_z = 0.0
        if activity_z < self.min_activity_z:
            return None

        bullish_bias = (
            close > vwap
            and ef[-1] > es[-1]
            and self._is_hh_hl(bars)
        )
        bearish_bias = (
            close < vwap
            and ef[-1] < es[-1]
            and self._is_lh_ll(bars)
        )

        # Direction filter
        if self.direction_filter == "bullish_only":
            bearish_bias = False
        elif self.direction_filter == "bearish_only":
            bullish_bias = False

        # Entry trigger: break of OR + volume already checked
        bullish_trigger = bullish_bias and close > or_high
        bearish_trigger = bearish_bias and close < or_low

        if not (bullish_trigger or bearish_trigger):
            return None

        direction = "bullish" if bullish_trigger else "bearish"
        opt_type = "CE" if bullish_trigger else "PE"
        spot = state.spot

        if state.chain is None:
            return None

        atm_base = round(spot / self.strike_step) * self.strike_step
        if direction == "bullish":
            # ITM CE is BELOW spot
            atm = atm_base - self.itm_steps * self.strike_step
        else:
            # ITM PE is ABOVE spot
            atm = atm_base + self.itm_steps * self.strike_step

        by_strike = state.chain.by_strike()
        q = by_strike.get(atm, {}).get(opt_type)
        if q is None or q.iv is None or not q.ltp:
            avail = [k for k in by_strike if opt_type in by_strike[k]
                     and by_strike[k][opt_type].iv is not None]
            if not avail:
                return None
            atm = min(avail, key=lambda k: abs(k - spot))
            q = by_strike[atm][opt_type]

        # Entry premium via BS
        expiry_close = datetime.combine(state.chain.expiry, time(15, 30), tzinfo=CHAIN_IST)
        t_expiry = max((expiry_close - state.ts).total_seconds() / (365 * 24 * 3600), 1e-5)
        g = black_scholes(spot, atm, t_expiry, q.iv, r=self.r, q=0.0, option_type=opt_type)
        entry_premium = g.price
        if entry_premium <= 0.5:
            return None

        # Stop / target on PREMIUM
        risk_pts = entry_premium * self.stop_premium_pct       # 1R in premium points
        stop_premium = max(0.5, entry_premium - risk_pts)
        target_pts = self.target_rr * risk_pts
        target_premium = entry_premium + target_pts

        # Translate to spot levels via delta for display
        delta_abs = max(abs(g.delta), 0.05)
        stop_spot_move = risk_pts / delta_abs
        target_spot_move = target_pts / delta_abs
        if direction == "bullish":
            target_spot = spot + target_spot_move
            stop_spot = spot - stop_spot_move
        else:
            target_spot = spot - target_spot_move
            stop_spot = spot + stop_spot_move

        # Trail: when premium gains 1R, stop trails 1R below peak => exactly breakeven at +1R
        trail_activation_pts = risk_pts
        trail_distance_pts = risk_pts

        thesis = (
            f"NIFTY Intraday Buyer {direction} [activity_z={activity_z:+.2f}, itm={self.itm_steps}]: "
            f"close={close:.0f} {'>' if direction == 'bullish' else '<'} VWAP={vwap:.0f}, "
            f"EMA{self.ema_fast}={ef[-1]:.0f} {'>' if direction == 'bullish' else '<'} "
            f"EMA{self.ema_slow}={es[-1]:.0f}, OR=({or_low:.0f}-{or_high:.0f}). "
            f"BUY {opt_type} {atm} @ ~Rs {entry_premium:.0f} (delta={g.delta:.2f}). "
            f"STOP: -{risk_pts:.0f}pts (=Rs {stop_premium:.0f}, {self.stop_premium_pct*100:.0f}% loss). "
            f"TARGET: +{target_pts:.0f}pts (=Rs {target_premium:.0f}, {self.target_rr:.1f}R). "
            f"TRAIL: on +1R gain, stop -> entry (breakeven). "
            f"TIME EXIT: {self.hold_minutes} min."
        )

        return Signal(
            rule_name=self.name,
            ts=state.ts,
            direction=direction,
            action=f"BUY_{opt_type}",
            strike=atm,
            option_type=opt_type,
            target_premium_gain=round(target_pts, 2),
            target_spot=round(target_spot, 1),
            stop_spot=round(stop_spot, 1),
            stop_premium=round(stop_premium, 2),
            hold_minutes=self.hold_minutes,
            prob_estimates={},
            trigger_context={
                "vwap": round(vwap, 2),
                "ema_fast": round(ef[-1], 2),
                "ema_slow": round(es[-1], 2),
                "or_high": round(or_high, 2),
                "or_low": round(or_low, 2),
                "bias_direction": direction,
                "volume_z": round(activity_z, 2),
                "spot": spot,
                "atm_strike": atm,
                "iv": q.iv,
                "entry_premium": round(entry_premium, 2),
                "stop_premium": round(stop_premium, 2),
                "target_premium": round(target_premium, 2),
                "risk_pts": round(risk_pts, 2),
                "target_pts": round(target_pts, 2),
                "itm_steps": self.itm_steps,
                "expiry": state.chain.expiry.isoformat(),
                "trail_activation_pts": round(trail_activation_pts, 2),
                "trail_distance_pts": round(trail_distance_pts, 2),
                "time_exit_minutes": self.hold_minutes,
            },
            thesis=thesis[:1900],
        )
