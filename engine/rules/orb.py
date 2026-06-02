"""
Opening Range Breakout (ORB) rule.

Setup:
  1. First 15 min of session (9:15-9:30 IST) establishes the opening range:
     OR_low = min low of those bars, OR_high = max high.
  2. After 9:45 IST (give 15-min buffer), monitor:
       - Upside break: spot closes > OR_high on a 5-min bar with volume_z >= 0.5
       - Downside break: spot closes < OR_low on a 5-min bar with volume_z >= 0.5
  3. Volume confirmation reduces fakeouts but doesn't eliminate them.

Action:
  - Bull break: BUY ATM CE
  - Bear break: BUY ATM PE
  - Target: OR_width pts of UNDERLYING move (NOT premium)
  - Premium target derived via delta: ~ delta * OR_width
  - Stop: spot back inside [OR_low, OR_high] (false breakout)
  - Hold: 90 min max — ORB plays exhaust by lunch typically

Premise: "First 15-min range often acts as a magnet/launch pad. Breakouts on
volume tend to continue for ~30-60 min before mean reversion."
This is YOUR hypothesis. The engine logs every fire + outcome so after 20-30
trades you'll have a forward-tracked hit rate for current market regime.
"""
from __future__ import annotations

from datetime import time
from ..signals import BaseRule, Signal
from ..state import MarketState
from ..pricing import black_scholes
from ..probability import prob_touch_above, prob_touch_below


class OpeningRangeBreakout(BaseRule):
    name = "orb"
    cooldown_minutes = 240  # one fire per session

    def __init__(
        self,
        or_minutes: int = 15,
        activation_minutes: int = 30,   # don't fire before 9:45
        min_volume_z: float = 0.5,
        hold_minutes: int = 90,
        target_multiple: float = 1.0,   # target = OR_width * this
        strike_step: int = 50,
        r: float = 0.065,
        # Tuning knobs (defaults preserve original behavior)
        stop_mode: str = "break_level",     # 'break_level' | 'or_midpoint' | 'or_far_side'
        entry_buffer_frac: float = 0.0,     # close must exceed break by this * OR_width
        time_window_end: int = 360,         # minutes-after-open after which no fires (360 = 15:15)
        time_window_start: int = 30,        # same as activation_minutes by default
        direction_filter: str = "none",     # 'none' | 'or_drive' | 'gap'
    ):
        super().__init__()
        self.or_minutes = or_minutes
        self.activation_minutes = activation_minutes
        self.min_volume_z = min_volume_z
        self.hold_minutes = hold_minutes
        self.target_multiple = target_multiple
        self.strike_step = strike_step
        self.r = r
        self.stop_mode = stop_mode
        self.entry_buffer_frac = entry_buffer_frac
        self.time_window_start = max(time_window_start, activation_minutes)
        self.time_window_end = time_window_end
        self.direction_filter = direction_filter

    def _check(self, state: MarketState) -> Signal | None:
        if not state.is_market_open():
            return None
        mso = state.minutes_since_open()
        if mso < self.time_window_start or mso > self.time_window_end:
            return None

        or_band = state.opening_range(minutes=self.or_minutes)
        if or_band is None:
            return None
        or_low, or_high = or_band
        or_width = or_high - or_low
        if or_width <= 0:
            return None
        buffer = self.entry_buffer_frac * or_width

        last_bar = state.last_bar()
        if last_bar is None:
            return None

        # Direction filter: only accept breakouts aligned with chosen bias
        allowed_directions = {"bullish", "bearish"}
        if self.direction_filter == "or_drive":
            # Read the OR's own bullish/bearish tilt: close of last OR bar vs first
            session_date = state.ts.astimezone(state.ts.tzinfo).date()
            from datetime import time as dtime, datetime as dt, timedelta as td
            open_dt = dt.combine(session_date, dtime(9, 15), tzinfo=state.ts.tzinfo)
            cutoff = open_dt + td(minutes=self.or_minutes)
            or_bars = [b for b in state.bars_5m if open_dt <= b.ts < cutoff]
            if len(or_bars) >= 2:
                drive_bullish = or_bars[-1].close > or_bars[0].open
                allowed_directions = {"bullish"} if drive_bullish else {"bearish"}
        elif self.direction_filter == "gap":
            # Need yesterday's close; approximate as first bar's open vs the OR low/high.
            # If first bar's open is above OR midpoint, treat as gap-up.
            if state.bars_5m:
                first_open = state.bars_5m[0].open
                or_mid = (or_low + or_high) / 2
                allowed_directions = {"bullish"} if first_open >= or_mid else {"bearish"}

        vol_z = state.volume_z(lookback=12)
        if vol_z is None:
            # Short history fallback: compare last bar vol to mean of available prior bars
            if len(state.bars_5m) >= 3:
                prior = state.bars_5m[:-1]
                avg = sum(b.volume for b in prior) / len(prior)
                ratio = last_bar.volume / max(avg, 1)
                # Map ratio 1.5x -> z 1.0, 2x -> z 2.0 (rough)
                vol_z = (ratio - 1.0) * 2.0
            else:
                vol_z = 0.0
        if vol_z < self.min_volume_z:
            return None

        # Determine break direction (with optional entry buffer)
        if last_bar.close > or_high + buffer and "bullish" in allowed_directions:
            direction = "bullish"
            opt_type = "CE"
            target_spot = state.spot + or_width * self.target_multiple
            if self.stop_mode == "or_midpoint":
                stop_spot = (or_low + or_high) / 2
            elif self.stop_mode == "or_far_side":
                stop_spot = or_low
            else:
                stop_spot = or_high
        elif last_bar.close < or_low - buffer and "bearish" in allowed_directions:
            direction = "bearish"
            opt_type = "PE"
            target_spot = state.spot - or_width * self.target_multiple
            if self.stop_mode == "or_midpoint":
                stop_spot = (or_low + or_high) / 2
            elif self.stop_mode == "or_far_side":
                stop_spot = or_high
            else:
                stop_spot = or_low
        else:
            return None

        if state.chain is None:
            return None

        # Pick ATM strike
        atm = round(state.spot / self.strike_step) * self.strike_step
        by_strike = state.chain.by_strike()
        q = by_strike.get(atm, {}).get(opt_type)
        if q is None or not q.iv or not q.ltp:
            # fall back to nearest with data
            available = [k for k in by_strike if opt_type in by_strike[k]
                         and by_strike[k][opt_type].iv]
            if not available:
                return None
            atm = min(available, key=lambda k: abs(k - state.spot))
            q = by_strike[atm][opt_type]

        iv = q.iv
        # Time to expiry — use chain's expiry from snapshot
        from datetime import datetime, timedelta
        from ..chain import IST as CHAIN_IST
        expiry_close = datetime.combine(
            state.chain.expiry, time(15, 30), tzinfo=CHAIN_IST
        )
        t_expiry = max((expiry_close - state.ts).total_seconds() / (365 * 24 * 3600), 1e-5)

        g = black_scholes(state.spot, atm, t_expiry, iv, r=self.r, q=0.0, option_type=opt_type)

        # Required spot move for premium target points
        # We'll report touch probs at several premium gain levels
        hold_t = self.hold_minutes / (60 * 24 * 365.0)
        prob_estimates = {}
        for tgt_pts in (10, 20, 30, 50):
            required_move = tgt_pts / max(abs(g.delta), 0.05)
            if direction == "bullish":
                s_tgt = state.spot + required_move
                p = prob_touch_above(state.spot, s_tgt, hold_t, iv, drift=0.0)
            else:
                s_tgt = state.spot - required_move
                p = prob_touch_below(state.spot, s_tgt, hold_t, iv, drift=0.0)
            prob_estimates[tgt_pts] = round(p, 4)

        # Premium target = delta * OR_width (first-order)
        target_premium_gain = abs(g.delta) * or_width * self.target_multiple
        # Stop premium: spot returning to break level
        # Approximate premium at that spot (no IV change)
        stop_premium = black_scholes(
            stop_spot, atm, t_expiry, iv, r=self.r, q=0.0, option_type=opt_type
        ).price

        thesis = (
            f"ORB {direction}: OR=[{or_low:.0f},{or_high:.0f}] width={or_width:.0f}; "
            f"break on 5m bar close={last_bar.close:.0f} vol_z={vol_z:+.2f}; "
            f"buying ATM {opt_type} {atm} delta={g.delta:.2f}; "
            f"target={target_spot:.0f} ({target_premium_gain:.0f} pts premium); "
            f"stop=spot back through {stop_spot:.0f}; hold<={self.hold_minutes}m. "
            f"Premise: range-break-with-volume tends to continue ~30-60m. "
            f"Engine reports option math; trader owns the premise."
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
            prob_estimates=prob_estimates,
            trigger_context={
                "or_low": or_low, "or_high": or_high, "or_width": or_width,
                "spot": state.spot, "last_close": last_bar.close,
                "volume_z": vol_z, "delta": g.delta, "iv": iv,
                "atm_premium": q.ltp,
            },
            thesis=thesis[:1900],  # positions.thesis has length>=50 check
        )
