"""
Max-Pain Magnet rule — expiry-day fade.

Premise:
  On expiry day (Tuesday for NIFTY weekly post-SEBI norms), as OI from
  losing-side options gets unwound, spot tends to gravitate toward the
  strike where option writers are maximally protected — "max pain."
  This effect is strongest in the last 2-3 hours of the session.

Entry conditions:
  1. Today IS expiry day (state.chain.expiry == today)
  2. After 13:00 IST (later in session = stronger pin)
  3. Spot is significantly displaced from max pain (>0.3% by default)
  4. Spot has begun moving back toward max pain (last bar shows reversal)

Action:
  - Spot above max pain -> BUY PE (expecting drift down)
  - Spot below max pain -> BUY CE (expecting drift up)

Target = max pain strike (the pin)
Stop = spot ± 0.25% (extension further from max pain)
Hold = 90 minutes (or to expiry close, whichever sooner)

HONEST CAVEATS:
  - Expiry days only (~4 per month) -> small total sample
  - Requires real OI -> live-only, no backtest (until we add OI history)
  - Max-pain pin theory is empirically supported on NIFTY but weakens
    during news events (RBI/Fed/Budget falling on expiry)
"""
from __future__ import annotations

from datetime import time

from ..signals import BaseRule, Signal
from ..state import MarketState
from ..pricing import black_scholes
from ..probability import prob_touch_above, prob_touch_below
from ..oi_zones import compute_zones


class MaxPainMagnet(BaseRule):
    name = "max_pain_magnet"
    cooldown_minutes = 60

    def __init__(
        self,
        min_displacement_pct: float = 0.003,    # 0.3% from max pain
        max_displacement_pct: float = 0.015,    # don't try if too far (news day)
        stop_extension_pct: float = 0.0025,
        time_window_start: int = 225,           # 13:00
        time_window_end: int = 360,             # 15:15
        hold_minutes: int = 90,
        strike_step: int = 50,
        r: float = 0.065,
        require_reversal: bool = True,
    ):
        super().__init__()
        self.min_displacement_pct = min_displacement_pct
        self.max_displacement_pct = max_displacement_pct
        self.stop_extension_pct = stop_extension_pct
        self.time_window_start = time_window_start
        self.time_window_end = time_window_end
        self.hold_minutes = hold_minutes
        self.strike_step = strike_step
        self.r = r
        self.require_reversal = require_reversal

    def _check(self, state: MarketState) -> Signal | None:
        if not state.is_market_open() or state.chain is None:
            return None

        # Must be expiry day
        if state.chain.expiry != state.ts.date():
            return None

        mso = state.minutes_since_open()
        if mso < self.time_window_start or mso > self.time_window_end:
            return None

        try:
            zones = compute_zones(state.chain)
        except Exception:
            return None
        if zones.max_pain is None:
            return None

        spot = state.spot
        displacement = (spot - zones.max_pain) / zones.max_pain
        if abs(displacement) < self.min_displacement_pct:
            return None
        if abs(displacement) > self.max_displacement_pct:
            return None

        today_bars = [b for b in state.bars_5m if b.ts.date() == state.ts.date()]
        if len(today_bars) < 2:
            return None
        last_bar = today_bars[-1]

        if displacement > 0:
            # Spot above max pain -> expect drift down -> BUY PE
            if self.require_reversal:
                if last_bar.close >= last_bar.open:
                    return None    # not a reversal bar
            direction = "bearish"
            opt_type = "PE"
        else:
            if self.require_reversal:
                if last_bar.close <= last_bar.open:
                    return None
            direction = "bullish"
            opt_type = "CE"

        target_spot = float(zones.max_pain)
        target_pts = abs(target_spot - spot)
        if direction == "bearish":
            stop_spot = spot * (1 + self.stop_extension_pct)
        else:
            stop_spot = spot * (1 - self.stop_extension_pct)

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
            f"Max-pain magnet {direction}: expiry={state.chain.expiry}, "
            f"max_pain={zones.max_pain}, spot={spot:.0f} ({displacement*100:+.2f}%). "
            f"Last bar reversal: O={last_bar.open:.0f} C={last_bar.close:.0f}. "
            f"Target=max_pain {target_spot:.0f} ({target_pts:.0f}pts); "
            f"stop={stop_spot:.0f}; hold<={self.hold_minutes}m. "
            f"Premise: expiry-day OI pin. Most reliable in last 2h, weakens on news days."
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
                "max_pain": zones.max_pain, "displacement_pct": displacement,
                "expiry": str(state.chain.expiry),
                "spot": spot, "atm_strike": atm, "iv": q.iv,
                "put_wall": zones.put_wall, "call_wall": zones.call_wall,
            },
            thesis=thesis[:1900],
        )
