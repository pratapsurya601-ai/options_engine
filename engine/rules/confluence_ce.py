"""
Multi-indicator confluence rule — naked CE (bullish only).

Fires BUY CE when at least `min_signals` out of 6 bullish indicators agree:
  1. EMA alignment: EMA(fast) > EMA(slow)
  2. MACD bullish: MACD line > signal line
  3. EMA recent cross: EMA(fast) crossed above EMA(slow) within `cross_recency_bars`
  4. Price above VWAP
  5. PSAR below current close
  6. activity_z >= activity_z_threshold

Entry: BS premium at ATM + strike_offset_steps * 50.
Exit: target +`target_premium_pts`, stop -`stop_premium_pts`, timeout `hold_minutes`.
"""
from __future__ import annotations

from datetime import time, datetime

from ..signals import BaseRule, Signal
from ..state import MarketState
from ..pricing import black_scholes
from ..indicators import ema, macd, parabolic_sar, crossed_above


class ConfluenceCE(BaseRule):
    name = "confluence_ce"
    cooldown_minutes = 15

    def __init__(
        self,
        min_signals: int = 4,
        ema_fast: int = 20,
        ema_slow: int = 50,
        cross_recency_bars: int = 5,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        activity_z_threshold: float = 0.3,
        strike_offset_steps: int = 0,
        target_premium_pts: float = 10.0,
        stop_premium_pts: float = 8.0,
        hold_minutes: int = 20,
        time_window_start: int = 30,
        time_window_end: int = 270,
        strike_step: int = 50,
        activity_lookback: int = 12,
        r: float = 0.065,
    ):
        super().__init__()
        self.min_signals = min_signals
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.cross_recency_bars = cross_recency_bars
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.activity_z_threshold = activity_z_threshold
        self.strike_offset_steps = strike_offset_steps
        self.target_premium_pts = target_premium_pts
        self.stop_premium_pts = stop_premium_pts
        self.hold_minutes = hold_minutes
        self.time_window_start = time_window_start
        self.time_window_end = time_window_end
        self.strike_step = strike_step
        self.activity_lookback = activity_lookback
        self.r = r

    def _check(self, state: MarketState) -> Signal | None:
        if not state.is_market_open():
            return None
        mso = state.minutes_since_open()
        if mso < self.time_window_start or mso > self.time_window_end:
            return None

        bars = state.bars_5m
        # Need enough for MACD: slow + signal + buffer
        need = max(self.ema_slow, self.macd_slow + self.macd_signal) + 5
        if len(bars) < need:
            return None

        closes = [b.close for b in bars]
        highs = [b.high for b in bars]
        lows = [b.low for b in bars]

        ef = ema(closes, self.ema_fast)
        es = ema(closes, self.ema_slow)
        if ef[-1] is None or es[-1] is None:
            return None

        macd_line, signal_line, _hist = macd(closes, self.macd_fast,
                                             self.macd_slow, self.macd_signal)
        sar = parabolic_sar(highs, lows)
        vwap_val = state.vwap()
        activity_z = state.activity_z(self.activity_lookback)
        if activity_z is None:
            activity_z = 0.0

        close_now = closes[-1]
        signals_fired: list[str] = []

        # 1. EMA alignment
        if ef[-1] > es[-1]:
            signals_fired.append("ema_alignment")
        # 2. MACD bullish
        if (macd_line[-1] is not None and signal_line[-1] is not None
                and macd_line[-1] > signal_line[-1]):
            signals_fired.append("macd_bullish")
        # 3. EMA recent cross above
        if crossed_above(ef, es, self.cross_recency_bars):
            signals_fired.append("ema_recent_cross")
        # 4. Price above VWAP
        if vwap_val is not None and close_now > vwap_val:
            signals_fired.append("price_above_vwap")
        # 5. PSAR below price
        if sar and sar[-1] is not None and sar[-1] < close_now:
            signals_fired.append("psar_below_price")
        # 6. Activity strong
        if activity_z >= self.activity_z_threshold:
            signals_fired.append("activity_strong")

        if len(signals_fired) < self.min_signals:
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
        expiry_close = datetime.combine(state.chain.expiry, time(15, 30),
                                        tzinfo=CHAIN_IST)
        t_expiry = max((expiry_close - state.ts).total_seconds()
                       / (365 * 24 * 3600), 1e-5)
        g = black_scholes(spot, atm, t_expiry, q.iv, r=self.r, q=0.0,
                          option_type="CE")
        entry_premium = g.price

        target_premium_gain = self.target_premium_pts
        stop_premium_loss = self.stop_premium_pts
        target_premium = entry_premium + target_premium_gain
        stop_premium = max(0.5, entry_premium - stop_premium_loss)
        delta_safe = max(abs(g.delta), 0.05)
        target_spot_move = target_premium_gain / delta_safe
        stop_spot_move = stop_premium_loss / delta_safe
        target_spot = spot + target_spot_move
        stop_spot = spot - stop_spot_move
        rr = target_premium_gain / max(stop_premium_loss, 0.1)

        thesis = (
            f"Confluence CE [{len(signals_fired)}/6 signals]: "
            f"{', '.join(signals_fired)}. "
            f"EMA{self.ema_fast}={ef[-1]:.0f}, EMA{self.ema_slow}={es[-1]:.0f}. "
            f"BUY CE {atm} @ ~Rs {entry_premium:.0f} (delta={g.delta:.2f}). "
            f"TARGET: +{target_premium_gain:.0f}pts (Rs {target_premium:.0f}) at spot {target_spot:.0f}. "
            f"STOP: -{stop_premium_loss:.0f}pts (Rs {stop_premium:.0f}) at spot {stop_spot:.0f}. "
            f"TIME EXIT: {self.hold_minutes}m. R/R={rr:.2f}."
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
                "signals_fired": signals_fired,
                "signal_count": len(signals_fired),
                "min_signals": self.min_signals,
                "activity_z": round(activity_z, 2),
                "ema_fast": ef[-1],
                "ema_slow": es[-1],
                "macd": macd_line[-1],
                "macd_signal": signal_line[-1],
                "psar": sar[-1] if sar else None,
                "vwap": vwap_val,
                "spot": spot,
                "atm_strike": atm,
                "iv": q.iv,
                "entry_premium": entry_premium,
                "target_premium_gain": target_premium_gain,
                "stop_premium_loss": stop_premium_loss,
                "risk_reward": round(rr, 2),
                "time_exit_minutes": self.hold_minutes,
            },
            thesis=thesis[:1900],
        )
