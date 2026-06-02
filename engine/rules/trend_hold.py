"""
Trend-Hold rule — designed to ride strong trending days like today.

Entry conditions:
  - EMA20 vs EMA50 has been stably bearish (or bullish) for last N bars
  - Gap |EMA20 - EMA50| > min_gap_pts (filters sideways markets)
  - Within time window (default 9:45 to 14:00)
  - No active position from this rule (enforced via long cooldown)

Action:
  - Bearish trend → BUY ATM PE
  - Bullish trend → BUY ATM CE

Exits (the entire design difference from scalp/swing):
  1. EMA flip — when EMA20 crosses back over EMA50 against your direction
     This is the PRIMARY exit. Monitor checks it every poll.
  2. Catastrophic stop — premium drops to `catastrophic_stop_pct` of entry
     (default 50% = kill ticket if trend completely fails)
  3. End-of-day — auto-close 10 min before session close
  4. No fixed target — let the trend run

Trade-offs vs scalp/swing:
  - Captures big trending moves (like today's 280-pt NIFTY drop)
  - Whipsaws hard on choppy days (every fake-out EMA flip = exit + loss)
  - Wider stops = bigger max loss per trade
  - One position per direction held all day
"""
from __future__ import annotations

from datetime import time, timedelta

from ..signals import BaseRule, Signal
from ..state import MarketState
from ..pricing import black_scholes
from ..indicators import ema


class TrendHold(BaseRule):
    name = "trend_hold"
    # Long cooldown to discourage re-entry on minor chop after exit.
    cooldown_minutes = 90

    def __init__(
        self,
        ema_fast: int = 20,
        ema_slow: int = 50,
        trend_stability_bars: int = 5,
        min_gap_pts: float = 5.0,             # require EMA20-EMA50 separation
        time_window_start: int = 30,           # 9:45
        time_window_end: int = 285,            # 14:00 (3 hours before close)
        # Catastrophic stop only — main exit is EMA flip handled by monitor
        catastrophic_stop_pct: float = 0.50,   # exit if premium drops to 50% of entry
        hold_minutes: int = 360,               # 6 hours (effectively all day)
        # End-of-day forced close (in minutes since open; 360 = 15:15)
        eod_close_minutes: int = 360,
        strike_step: int = 50,
        strike_offset_steps: int = 0,
        r: float = 0.065,
    ):
        super().__init__()
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.trend_stability_bars = trend_stability_bars
        self.min_gap_pts = min_gap_pts
        self.time_window_start = time_window_start
        self.time_window_end = time_window_end
        self.catastrophic_stop_pct = catastrophic_stop_pct
        self.hold_minutes = hold_minutes
        self.eod_close_minutes = eod_close_minutes
        self.strike_step = strike_step
        self.strike_offset_steps = strike_offset_steps
        self.r = r

    def _check(self, state: MarketState) -> Signal | None:
        if not state.is_market_open():
            return None
        mso = state.minutes_since_open()
        if mso < self.time_window_start or mso > self.time_window_end:
            return None

        bars = state.bars_5m
        n = self.trend_stability_bars
        if len(bars) < self.ema_slow + n:
            return None

        closes = [b.close for b in bars]
        ef = ema(closes, self.ema_fast)
        es = ema(closes, self.ema_slow)
        if ef[-1] is None or es[-1] is None:
            return None

        # Check trend stability over last N bars
        recent_signs = []
        for i in range(-n, 0):
            if ef[i] is None or es[i] is None:
                return None
            recent_signs.append(ef[i] > es[i])

        if all(recent_signs):
            direction = "bullish"
        elif not any(recent_signs):
            direction = "bearish"
        else:
            return None

        # Filter sideways: require meaningful EMA gap
        gap = abs(ef[-1] - es[-1])
        if gap < self.min_gap_pts:
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

        # Catastrophic stop only — primary exit is via EMA-flip detected by monitor
        catastrophic_stop = entry_premium * self.catastrophic_stop_pct
        # Set target very high so it effectively never triggers (EMA-flip exit handles wins)
        far_target = entry_premium * 3.0

        thesis = (
            f"Trend-hold {direction}: EMA20={ef[-1]:.0f}, EMA50={es[-1]:.0f}, "
            f"gap={gap:+.1f} (stable {n}+ bars). "
            f"BUY {opt_type} {atm} @ ~Rs {entry_premium:.0f} (delta={g.delta:.2f}). "
            f"PRIMARY EXIT: EMA20 crosses back over EMA50 against direction. "
            f"CATASTROPHIC STOP: Rs {catastrophic_stop:.0f} (50% of entry). "
            f"EOD: auto-close at 15:15. "
            f"Premise: ride the trend until it ends. Whipsaws on choppy days."
        )

        return Signal(
            rule_name=self.name,
            ts=state.ts,
            direction=direction,
            action=f"BUY_{opt_type}",
            strike=atm,
            option_type=opt_type,
            target_premium_gain=entry_premium * 2.0,    # far target
            target_spot=None,
            stop_spot=None,
            stop_premium=round(catastrophic_stop, 2),
            hold_minutes=self.hold_minutes,
            prob_estimates={},
            trigger_context={
                "exit_on_ema_flip": True,
                "ema_fast_period": self.ema_fast,
                "ema_slow_period": self.ema_slow,
                "entry_direction": direction,
                "eod_close_minutes": self.eod_close_minutes,
                "ema_fast_at_entry": ef[-1],
                "ema_slow_at_entry": es[-1],
                "spot": spot, "atm_strike": atm, "iv": q.iv,
                "entry_premium": entry_premium,
            },
            thesis=thesis[:1900],
        )
