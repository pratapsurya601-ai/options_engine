"""
Higher-timeframe naked options rule — designed for 15-min / 30-min bars.

Fires BUY CE/PE when ALL of:
  1. EMA20 crossed above (bullish) or below (bearish) EMA50 within last
     `cross_recency_bars` bars on the higher timeframe.
  2. RSI(14) is between rsi_min..rsi_max (momentum confirmed, not exhausted).
  3. MACD line is above signal (bullish) or below signal (bearish), in the
     direction of the trade.
  4. Within intraday time window.

Less noise than a 2-min scalper: fewer signals per day, larger target/stop.
"""
from __future__ import annotations

from datetime import time, datetime

from ..signals import BaseRule, Signal
from ..state import MarketState
from ..pricing import black_scholes
from ..indicators import ema, rsi, macd, crossed_above, crossed_below


class HtfNaked(BaseRule):
    name = "htf_naked"
    cooldown_minutes = 60

    def __init__(
        self,
        direction_filter: str = "both",
        ema_fast: int = 20,
        ema_slow: int = 50,
        cross_recency_bars: int = 3,
        rsi_period: int = 14,
        rsi_min: float = 40.0,
        rsi_max: float = 60.0,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        strike_offset_steps: int = 0,
        target_premium_pts: float = 20.0,
        stop_premium_pts: float = 12.0,
        hold_minutes: int = 60,
        time_window_start: int = 30,
        time_window_end: int = 300,
        strike_step: int = 50,
        r: float = 0.065,
    ):
        super().__init__()
        self.direction_filter = direction_filter
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.cross_recency_bars = cross_recency_bars
        self.rsi_period = rsi_period
        self.rsi_min = rsi_min
        self.rsi_max = rsi_max
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
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
        needed = max(self.ema_slow + 2, self.rsi_period + 2,
                     self.macd_slow + self.macd_signal + 2)
        if len(bars) < needed:
            return None

        closes = [b.close for b in bars]
        ef = ema(closes, self.ema_fast)
        es = ema(closes, self.ema_slow)
        if ef[-1] is None or es[-1] is None:
            return None

        bullish = crossed_above(ef, es, self.cross_recency_bars)
        bearish = crossed_below(ef, es, self.cross_recency_bars)
        if not (bullish or bearish):
            return None

        if self.direction_filter == "bullish_only" and not bullish:
            return None
        if self.direction_filter == "bearish_only" and not bearish:
            return None

        # RSI confirmation
        rsi_vals = rsi(closes, self.rsi_period)
        rsi_now = rsi_vals[-1]
        if rsi_now is None:
            return None
        if not (self.rsi_min <= rsi_now <= self.rsi_max):
            return None

        # MACD direction confirmation
        macd_line, signal_line, _ = macd(
            closes, self.macd_fast, self.macd_slow, self.macd_signal
        )
        if macd_line[-1] is None or signal_line[-1] is None:
            return None
        if bullish and not (macd_line[-1] > signal_line[-1]):
            return None
        if bearish and not (macd_line[-1] < signal_line[-1]):
            return None

        direction = "bullish" if bullish else "bearish"
        opt_type = "CE" if bullish else "PE"
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

        thesis = (
            f"HTF naked {direction}: EMA{self.ema_fast}={ef[-1]:.0f} vs "
            f"EMA{self.ema_slow}={es[-1]:.0f} (recent {direction} cross), "
            f"RSI({self.rsi_period})={rsi_now:.1f} in [{self.rsi_min:.0f},{self.rsi_max:.0f}], "
            f"MACD={macd_line[-1]:.2f} vs sig={signal_line[-1]:.2f}. "
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
                "ema_fast": ef[-1],
                "ema_slow": es[-1],
                "ema_fast_period": self.ema_fast,
                "ema_slow_period": self.ema_slow,
                "rsi": round(rsi_now, 2),
                "rsi_period": self.rsi_period,
                "macd": round(macd_line[-1], 3),
                "macd_signal": round(signal_line[-1], 3),
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
