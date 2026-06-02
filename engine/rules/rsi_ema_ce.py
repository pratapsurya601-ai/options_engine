"""
RSI + EMA naked CE scalp — bullish only.

Fires BUY CE when:
  - EMA20 > EMA50 (uptrend established)
  - EMA20 crossed above EMA50 within last `cross_recency_bars`
  - RSI(14) is in a "go zone" (not extreme overbought, not deeply oversold)
  - Within the intraday time window

Exit:
  - Target: entry_premium + target_premium_pts
  - Stop:   entry_premium - stop_premium_pts (via stop_spot derived from delta)
  - Timeout: hold_minutes
"""
from __future__ import annotations

from datetime import time, datetime

from ..signals import BaseRule, Signal
from ..state import MarketState
from ..pricing import black_scholes
from ..indicators import ema, rsi, crossed_above


class RsiEmaCE(BaseRule):
    name = "rsi_ema_ce"
    cooldown_minutes = 15

    def __init__(
        self,
        ema_fast: int = 20,
        ema_slow: int = 50,
        cross_recency_bars: int = 5,
        rsi_period: int = 14,
        rsi_min: float = 40.0,
        rsi_max: float = 70.0,
        strike_offset_steps: int = 0,
        target_premium_pts: float = 10.0,
        stop_premium_pts: float = 8.0,
        hold_minutes: int = 15,
        time_window_start: int = 30,    # 9:45
        time_window_end: int = 270,     # 13:45
        strike_step: int = 50,
        r: float = 0.065,
    ):
        super().__init__()
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.cross_recency_bars = cross_recency_bars
        self.rsi_period = rsi_period
        self.rsi_min = rsi_min
        self.rsi_max = rsi_max
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
        # Need enough bars for EMA slow seed plus a couple for cross detection,
        # and for RSI warmup.
        needed = max(self.ema_slow + 2, self.rsi_period + 2)
        if len(bars) < needed:
            return None

        closes = [b.close for b in bars]
        ef = ema(closes, self.ema_fast)
        es = ema(closes, self.ema_slow)
        if ef[-1] is None or es[-1] is None:
            return None

        # Trend up
        if not (ef[-1] > es[-1]):
            return None

        # Recent bullish cross
        if not crossed_above(ef, es, self.cross_recency_bars):
            return None

        rsi_vals = rsi(closes, self.rsi_period)
        rsi_now = rsi_vals[-1]
        if rsi_now is None:
            return None
        if not (self.rsi_min <= rsi_now <= self.rsi_max):
            return None

        spot = state.spot
        if state.chain is None:
            return None

        atm_base = round(spot / self.strike_step) * self.strike_step
        atm = atm_base + self.strike_offset_steps * self.strike_step
        by_strike = state.chain.by_strike()
        q = by_strike.get(atm, {}).get("CE")
        if q is None or q.iv is None or not q.ltp:
            avail = [k for k in by_strike if "CE" in by_strike[k]
                     and by_strike[k]["CE"].iv is not None]
            if not avail:
                return None
            atm = min(avail, key=lambda k: abs(k - spot))
            q = by_strike[atm]["CE"]

        from ..chain import IST as CHAIN_IST
        expiry_close = datetime.combine(state.chain.expiry, time(15, 30), tzinfo=CHAIN_IST)
        t_expiry = max((expiry_close - state.ts).total_seconds() / (365 * 24 * 3600), 1e-5)
        g = black_scholes(spot, atm, t_expiry, q.iv, r=self.r, q=0.0, option_type="CE")
        entry_premium = g.price

        target_premium_gain = self.target_premium_pts
        stop_premium_loss = self.stop_premium_pts
        # Translate premium moves to spot moves via delta
        target_spot_move = target_premium_gain / max(abs(g.delta), 0.05)
        stop_spot_move = stop_premium_loss / max(abs(g.delta), 0.05)

        target_premium = entry_premium + target_premium_gain
        stop_premium = max(0.5, entry_premium - stop_premium_loss)
        target_spot = spot + target_spot_move
        stop_spot = spot - stop_spot_move

        rr = target_premium_gain / max(stop_premium_loss, 0.1)

        thesis = (
            f"RSI+EMA CE scalp: EMA{self.ema_fast}={ef[-1]:.0f} > EMA{self.ema_slow}={es[-1]:.0f} "
            f"(recent bullish cross), RSI({self.rsi_period})={rsi_now:.1f} in "
            f"[{self.rsi_min:.0f},{self.rsi_max:.0f}]. "
            f"BUY CE {atm} @ ~Rs {entry_premium:.0f} (delta={g.delta:.2f}). "
            f"TARGET +{target_premium_gain:.0f}pts (=Rs {target_premium:.0f}) at spot {target_spot:.0f}. "
            f"STOP -{stop_premium_loss:.0f}pts (=Rs {stop_premium:.0f}) at spot {stop_spot:.0f}. "
            f"TIME EXIT after {self.hold_minutes}m. R/R={rr:.2f}."
        )

        return Signal(
            rule_name=self.name,
            ts=state.ts,
            direction="bullish",
            action="BUY_CE",
            strike=atm,
            option_type="CE",
            target_premium_gain=round(target_premium_gain, 1),
            target_spot=round(target_spot, 1),
            stop_spot=round(stop_spot, 1),
            stop_premium=round(stop_premium, 2),
            hold_minutes=self.hold_minutes,
            prob_estimates={},
            trigger_context={
                "ema_fast": ef[-1],
                "ema_slow": es[-1],
                "ema_fast_period": self.ema_fast,
                "ema_slow_period": self.ema_slow,
                "rsi": round(rsi_now, 2),
                "rsi_period": self.rsi_period,
                "rsi_min": self.rsi_min,
                "rsi_max": self.rsi_max,
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
