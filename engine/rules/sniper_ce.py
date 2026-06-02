"""
Sniper CE — ultra-selective naked CE rule (bullish only).

Fires BUY CE only when ALL 9 high-conviction filters align:
  1. Trend stable: EMA20 > EMA50 for last `trend_stable_bars` bars
  2. RSI(14) in pullback zone [rsi_min, rsi_max]
  3. MACD turning up: hist crossed >0 within `macd_cross_bars` OR line just crossed signal
  4. close > VWAP
  5. PSAR below close
  6. activity_z >= min_activity_z
  7. Within intraday window [time_window_start, time_window_end]
  8. Current bar is green (close > open)
  9. (Optional) Spot within `pullback_pct` of EMA20

Cooldown 90 min (max ~3-4 fires/session). Goal: 5-10 fires/month, 65-70% win rate.
"""
from __future__ import annotations

from datetime import time, datetime

from ..signals import BaseRule, Signal
from ..state import MarketState
from ..pricing import black_scholes
from ..indicators import ema, rsi, macd, parabolic_sar, crossed_above


class SniperCE(BaseRule):
    name = "sniper_ce"
    cooldown_minutes = 90

    def __init__(
        self,
        trend_stable_bars: int = 10,
        ema_fast: int = 20,
        ema_slow: int = 50,
        rsi_period: int = 14,
        rsi_min: float = 40.0,
        rsi_max: float = 60.0,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        macd_cross_bars: int = 3,
        min_activity_z: float = 0.5,
        require_pullback_to_ema20: bool = True,
        pullback_pct: float = 0.005,
        time_window_start: int = 45,
        time_window_end: int = 225,
        strike_offset_steps: int = 0,
        target_premium_pts: float = 15.0,
        stop_premium_pts: float = 8.0,
        hold_minutes: int = 30,
        eod_close_minutes: int = 360,
        strike_step: int = 50,
        activity_lookback: int = 12,
        r: float = 0.065,
    ):
        super().__init__()
        self.trend_stable_bars = trend_stable_bars
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.rsi_period = rsi_period
        self.rsi_min = rsi_min
        self.rsi_max = rsi_max
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.macd_cross_bars = macd_cross_bars
        self.min_activity_z = min_activity_z
        self.require_pullback_to_ema20 = require_pullback_to_ema20
        self.pullback_pct = pullback_pct
        self.time_window_start = time_window_start
        self.time_window_end = time_window_end
        self.strike_offset_steps = strike_offset_steps
        self.target_premium_pts = target_premium_pts
        self.stop_premium_pts = stop_premium_pts
        self.hold_minutes = hold_minutes
        self.eod_close_minutes = eod_close_minutes
        self.strike_step = strike_step
        self.activity_lookback = activity_lookback
        self.r = r

    def _check(self, state: MarketState) -> Signal | None:
        if not state.is_market_open():
            return None
        mso = state.minutes_since_open()
        # Filter 7: time window
        if mso < self.time_window_start or mso > self.time_window_end:
            return None

        bars = state.bars_5m
        need = max(
            self.ema_slow + self.trend_stable_bars + 2,
            self.macd_slow + self.macd_signal + 5,
            self.rsi_period + 2,
        )
        if len(bars) < need:
            return None

        closes = [b.close for b in bars]
        highs = [b.high for b in bars]
        lows = [b.low for b in bars]

        ef = ema(closes, self.ema_fast)
        es = ema(closes, self.ema_slow)
        if ef[-1] is None or es[-1] is None:
            return None

        # Filter 1: trend stable — EMA fast > EMA slow for last N bars
        stable_ok = True
        for k in range(1, self.trend_stable_bars + 1):
            a = ef[-k]
            b = es[-k]
            if a is None or b is None or not (a > b):
                stable_ok = False
                break
        if not stable_ok:
            return None

        # Filter 2: RSI pullback zone
        rsi_vals = rsi(closes, self.rsi_period)
        rsi_now = rsi_vals[-1]
        if rsi_now is None:
            return None
        if not (self.rsi_min <= rsi_now <= self.rsi_max):
            return None

        # Filter 3: MACD turning up
        macd_line, signal_line, hist = macd(
            closes, self.macd_fast, self.macd_slow, self.macd_signal
        )
        macd_turn_up = False
        # Histogram crossed above 0 within last macd_cross_bars
        n = len(hist)
        for i in range(max(1, n - self.macd_cross_bars), n):
            h_prev = hist[i - 1]
            h_now = hist[i]
            if h_prev is None or h_now is None:
                continue
            if h_prev <= 0 and h_now > 0:
                macd_turn_up = True
                break
        # OR MACD line crossed above signal recently
        if not macd_turn_up:
            if crossed_above(macd_line, signal_line, self.macd_cross_bars):
                macd_turn_up = True
        if not macd_turn_up:
            return None

        close_now = closes[-1]

        # Filter 4: close > VWAP
        vwap_val = state.vwap()
        if vwap_val is None or not (close_now > vwap_val):
            return None

        # Filter 5: PSAR below close
        sar = parabolic_sar(highs, lows)
        if not sar or sar[-1] is None or not (sar[-1] < close_now):
            return None

        # Filter 6: activity confirmation
        activity_z = state.activity_z(self.activity_lookback)
        if activity_z is None or activity_z < self.min_activity_z:
            return None

        # Filter 8: current bar green
        last_bar = bars[-1]
        if not (last_bar.close > last_bar.open):
            return None

        # Filter 9 (optional): pullback to EMA20
        if self.require_pullback_to_ema20:
            ema_fast_val = ef[-1]
            if abs(close_now - ema_fast_val) / max(ema_fast_val, 1.0) > self.pullback_pct:
                return None

        # All filters passed — build signal
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
        delta_safe = max(abs(g.delta), 0.05)
        target_spot_move = target_premium_gain / delta_safe
        stop_spot_move = stop_premium_loss / delta_safe
        target_premium = entry_premium + target_premium_gain
        stop_premium = max(0.5, entry_premium - stop_premium_loss)
        target_spot = spot + target_spot_move
        stop_spot = spot - stop_spot_move
        rr = target_premium_gain / max(stop_premium_loss, 0.1)

        thesis = (
            f"Sniper CE [9/9 filters aligned]: trend stable {self.trend_stable_bars}b, "
            f"RSI={rsi_now:.1f} in [{self.rsi_min:.0f},{self.rsi_max:.0f}], MACD turning up, "
            f"close {close_now:.0f} > VWAP {vwap_val:.0f}, PSAR {sar[-1]:.0f} < close, "
            f"activity_z={activity_z:+.2f}, green bar. "
            f"BUY CE {atm} @ ~Rs {entry_premium:.0f} (delta={g.delta:.2f}). "
            f"TARGET +{target_premium_gain:.0f}pts (Rs {target_premium:.0f}) at spot {target_spot:.0f}. "
            f"STOP -{stop_premium_loss:.0f}pts (Rs {stop_premium:.0f}) at spot {stop_spot:.0f}. "
            f"TIME EXIT {self.hold_minutes}m. R/R={rr:.2f}."
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
                "filter_1_trend_stable": True,
                "filter_2_rsi_in_zone": True,
                "filter_3_macd_turning_up": True,
                "filter_4_above_vwap": True,
                "filter_5_psar_below": True,
                "filter_6_activity_ok": True,
                "filter_7_time_window": True,
                "filter_8_green_bar": True,
                "filter_9_pullback_to_ema20": self.require_pullback_to_ema20,
                "trend_stable_bars": self.trend_stable_bars,
                "ema_fast": ef[-1],
                "ema_slow": es[-1],
                "ema_fast_period": self.ema_fast,
                "ema_slow_period": self.ema_slow,
                "rsi": round(rsi_now, 2),
                "rsi_period": self.rsi_period,
                "rsi_min": self.rsi_min,
                "rsi_max": self.rsi_max,
                "macd": macd_line[-1],
                "macd_signal": signal_line[-1],
                "macd_hist": hist[-1],
                "vwap": vwap_val,
                "psar": sar[-1],
                "activity_z": round(activity_z, 2),
                "bar_open": last_bar.open,
                "bar_close": last_bar.close,
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
                "eod_close_minutes": self.eod_close_minutes,
            },
            thesis=thesis[:1900],
        )
