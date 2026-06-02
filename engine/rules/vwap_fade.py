"""
VWAP Mean-Reversion Fade rule.

Premise (regime-appropriate for mean-reverting NIFTY):
  Intraday price tends to mean-revert toward VWAP. When price extends > X%
  away from VWAP, the next significant move is often back toward it.

Entry conditions (all must hold):
  1. Within trade time window (default 9:45-13:30)
  2. Spot is > extension_threshold above/below VWAP (configurable, default 0.4%)
  3. Last 1-2 bars show counter-trend reversal:
       - For PE fade (price too high): last bar close < last bar open (red),
         AND high made earlier in bar (close in lower half of bar range)
       - For CE fade (price too low): last bar close > last bar open (green),
         AND low made earlier in bar (close in upper half of bar range)

Action:
  - Price above VWAP -> BUY PE (fade rally)
  - Price below VWAP -> BUY CE (fade dip)

Target = VWAP (full revert)
Stop = current spot ± 0.3% (further extension away from VWAP)
Hold = 60 minutes
"""
from __future__ import annotations

from datetime import time, timedelta

from ..signals import BaseRule, Signal
from ..state import MarketState
from ..pricing import black_scholes
from ..probability import prob_touch_above, prob_touch_below


class VwapFade(BaseRule):
    name = "vwap_fade"
    cooldown_minutes = 60   # allow 2 fires per session

    def __init__(
        self,
        extension_threshold_pct: float = 0.004,   # 0.4% from VWAP to qualify
        stop_extension_pct: float = 0.003,        # 0.3% further = stop
        time_window_start: int = 30,
        time_window_end: int = 255,               # 13:30
        hold_minutes: int = 60,
        require_reversal_bar: bool = True,
        strike_step: int = 50,
        r: float = 0.065,
    ):
        super().__init__()
        self.extension_threshold_pct = extension_threshold_pct
        self.stop_extension_pct = stop_extension_pct
        self.time_window_start = time_window_start
        self.time_window_end = time_window_end
        self.hold_minutes = hold_minutes
        self.require_reversal_bar = require_reversal_bar
        self.strike_step = strike_step
        self.r = r

    def _check(self, state: MarketState) -> Signal | None:
        if not state.is_market_open():
            return None
        mso = state.minutes_since_open()
        if mso < self.time_window_start or mso > self.time_window_end:
            return None

        vwap = state.vwap(since_open=True)
        if vwap is None or vwap <= 0:
            return None
        spot = state.spot
        extension = (spot - vwap) / vwap

        if abs(extension) < self.extension_threshold_pct:
            return None

        today = state.ts.date()
        today_bars = [b for b in state.bars_5m if b.ts.date() == today]
        if len(today_bars) < 2:
            return None
        last_bar = today_bars[-1]

        if extension > 0:
            # Above VWAP -> fade with PE
            if self.require_reversal_bar:
                # Want: bar opened, made high, came back down — close in lower 60% of range
                bar_range = max(last_bar.high - last_bar.low, 0.01)
                pos_in_bar = (last_bar.close - last_bar.low) / bar_range
                if last_bar.close >= last_bar.open or pos_in_bar > 0.6:
                    return None
            direction = "bearish"
            opt_type = "PE"
        else:
            if self.require_reversal_bar:
                bar_range = max(last_bar.high - last_bar.low, 0.01)
                pos_in_bar = (last_bar.close - last_bar.low) / bar_range
                if last_bar.close <= last_bar.open or pos_in_bar < 0.4:
                    return None
            direction = "bullish"
            opt_type = "CE"

        target_spot = vwap
        target_pts = abs(target_spot - spot)

        if direction == "bearish":
            stop_spot = spot * (1 + self.stop_extension_pct)
        else:
            stop_spot = spot * (1 - self.stop_extension_pct)

        if state.chain is None:
            return None
        atm = round(spot / self.strike_step) * self.strike_step
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
        target_premium_gain = abs(g.delta) * target_pts
        stop_premium = black_scholes(
            stop_spot, atm, t_expiry, q.iv, r=self.r, q=0.0, option_type=opt_type
        ).price
        hold_t = self.hold_minutes / (60 * 24 * 365.0)
        if direction == "bullish":
            p_touch = prob_touch_above(spot, target_spot, hold_t, q.iv, drift=0.0)
        else:
            p_touch = prob_touch_below(spot, target_spot, hold_t, q.iv, drift=0.0)

        thesis = (
            f"VWAP fade {direction}: VWAP={vwap:.0f}, spot={spot:.0f} "
            f"({extension*100:+.2f}% from VWAP); "
            f"last bar: O={last_bar.open:.0f} H={last_bar.high:.0f} "
            f"L={last_bar.low:.0f} C={last_bar.close:.0f} (reversal: "
            f"{'yes' if self.require_reversal_bar else 'not required'}); "
            f"target=VWAP {target_spot:.0f} ({target_pts:.0f}pts spot / "
            f"{target_premium_gain:.0f}pts premium); stop={stop_spot:.0f}; "
            f"hold <= {self.hold_minutes}m. Premise: intraday VWAP magnet."
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
            prob_estimates={int(target_pts): round(p_touch, 4)},
            trigger_context={
                "vwap": vwap, "spot": spot, "extension_pct": extension,
                "last_bar": {"o": last_bar.open, "h": last_bar.high,
                             "l": last_bar.low, "c": last_bar.close},
                "atm_strike": atm, "iv": q.iv,
            },
            thesis=thesis[:1900],
        )
