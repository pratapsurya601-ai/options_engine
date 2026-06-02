"""
MACD + Bollinger Bands naked CE rule.

Bullish-only scalp. Fires BUY CE when ALL of:
  1. MACD line crossed above signal line within `cross_recency_bars`
  2. Price is near or below lower Bollinger Band — measured by
     bb_pos = (close - lower) / (upper - lower), require bb_pos <= bb_pct
  3. Within time window
  4. (Optional) close > EMA(ema_for_trend) when require_trend_alignment=True

Entry premium via Black-Scholes. Fixed-pts target/stop. Time-based final exit.
"""
from __future__ import annotations

from datetime import time, datetime

from ..signals import BaseRule, Signal
from ..state import MarketState
from ..pricing import black_scholes
from ..indicators import ema, macd, bollinger, crossed_above


class MacdBbCE(BaseRule):
    name = "macd_bb_ce"
    cooldown_minutes = 15

    def __init__(
        self,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        cross_recency_bars: int = 3,
        bb_period: int = 20,
        bb_std: float = 2.0,
        bb_pct: float = 0.3,
        require_trend_alignment: bool = False,
        ema_for_trend: int = 50,
        strike_offset_steps: int = 0,
        strike_step: int = 50,
        target_premium_pts: float = 10.0,
        stop_premium_pts: float = 8.0,
        hold_minutes: int = 20,
        time_window_start: int = 30,    # 9:45
        time_window_end: int = 270,     # 13:45
        r: float = 0.065,
    ):
        super().__init__()
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.cross_recency_bars = cross_recency_bars
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.bb_pct = bb_pct
        self.require_trend_alignment = require_trend_alignment
        self.ema_for_trend = ema_for_trend
        self.strike_offset_steps = strike_offset_steps
        self.strike_step = strike_step
        self.target_premium_pts = target_premium_pts
        self.stop_premium_pts = stop_premium_pts
        self.hold_minutes = hold_minutes
        self.time_window_start = time_window_start
        self.time_window_end = time_window_end
        self.r = r

    def _check(self, state: MarketState) -> Signal | None:
        if not state.is_market_open():
            return None
        mso = state.minutes_since_open()
        if mso < self.time_window_start or mso > self.time_window_end:
            return None

        bars = state.bars_5m
        min_bars = max(self.macd_slow + self.macd_signal + 2,
                       self.bb_period + 2,
                       (self.ema_for_trend + 2) if self.require_trend_alignment else 0)
        if len(bars) < min_bars:
            return None

        closes = [b.close for b in bars]

        macd_line, signal_line, _hist = macd(
            closes, self.macd_fast, self.macd_slow, self.macd_signal
        )
        if macd_line[-1] is None or signal_line[-1] is None:
            return None
        if not crossed_above(macd_line, signal_line, self.cross_recency_bars):
            return None

        mid, up, lo = bollinger(closes, self.bb_period, self.bb_std)
        if up[-1] is None or lo[-1] is None:
            return None
        band_width = up[-1] - lo[-1]
        if band_width <= 0:
            return None
        close = closes[-1]
        bb_pos = (close - lo[-1]) / band_width
        if bb_pos > self.bb_pct:
            return None

        if self.require_trend_alignment:
            e_trend = ema(closes, self.ema_for_trend)
            if e_trend[-1] is None or close <= e_trend[-1]:
                return None

        if state.chain is None:
            return None
        spot = state.spot
        atm_base = round(spot / self.strike_step) * self.strike_step
        # CE: positive offset = OTM higher strike; negative = ITM lower strike
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
        target_premium = entry_premium + target_premium_gain
        stop_premium = max(0.5, entry_premium - stop_premium_loss)
        # Spot levels derived from delta for the spot-based stop in backtester
        delta = max(abs(g.delta), 0.05)
        target_spot_move = target_premium_gain / delta
        stop_spot_move = stop_premium_loss / delta
        target_spot = spot + target_spot_move
        stop_spot = spot - stop_spot_move

        rr = target_premium_gain / max(stop_premium_loss, 0.1)

        thesis = (
            f"MACD+BB CE: MACD crossed up (line={macd_line[-1]:.2f}, sig={signal_line[-1]:.2f}); "
            f"close={close:.0f} near lower BB (pos={bb_pos:.2f}, lo={lo[-1]:.0f}, up={up[-1]:.0f}). "
            f"BUY CE {atm} @ ~Rs {entry_premium:.0f} (delta={g.delta:.2f}). "
            f"TARGET +{target_premium_gain:.0f}pts (=Rs {target_premium:.0f}, spot {target_spot:.0f}). "
            f"STOP -{stop_premium_loss:.0f}pts (=Rs {stop_premium:.0f}, spot {stop_spot:.0f}). "
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
                "macd_line": macd_line[-1],
                "macd_signal": signal_line[-1],
                "bb_pos": round(bb_pos, 3),
                "bb_lower": lo[-1],
                "bb_upper": up[-1],
                "bb_middle": mid[-1],
                "spot": spot,
                "atm_strike": atm,
                "iv": q.iv,
                "entry_premium": entry_premium,
                "target_premium_gain": target_premium_gain,
                "stop_premium_loss": stop_premium_loss,
                "risk_reward": round(rr, 2),
                "time_exit_minutes": self.hold_minutes,
                "require_trend_alignment": self.require_trend_alignment,
            },
            thesis=thesis[:1900],
        )
