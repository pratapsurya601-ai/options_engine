"""
VWAP-reclaim + volume confirmation naked CE.

Bullish-only intraday rule:
  - Price has been BELOW VWAP for the last `min_below_bars` bars
  - Current bar's close just crossed ABOVE VWAP (prev close <= VWAP, now close > VWAP)
  - Activity confirmation: state.activity_z(lookback=12) >= min_activity_z
  - Optional MACD bullish alignment: macd_line >= signal_line on current bar
  - Within configured time window (default 9:45 - 13:45 IST)

Action: BUY CE at ATM ± strike_offset_steps × 50 with fixed-pt premium target / stop
and a time-based hold cap.
"""
from __future__ import annotations

from datetime import datetime, time, timedelta

from ..signals import BaseRule, Signal
from ..state import MarketState, IST
from ..pricing import black_scholes
from ..indicators import macd


def _vwap_of(bars) -> float | None:
    """Plain VWAP over an arbitrary bar slice; falls back to typical-price avg
    when volumes are all zero (NIFTY index case)."""
    if not bars:
        return None
    typical = [(b.high + b.low + b.close) / 3 for b in bars]
    total_vol = sum(b.volume for b in bars)
    if total_vol == 0:
        return sum(typical) / len(typical)
    return sum(tp * b.volume for tp, b in zip(typical, bars)) / total_vol


class VwapReclaimCE(BaseRule):
    name = "vwap_reclaim_ce"
    cooldown_minutes = 20

    def __init__(
        self,
        min_below_bars: int = 10,
        min_activity_z: float = 0.5,
        require_macd_bullish: bool = True,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        strike_offset_steps: int = 0,
        target_premium_pts: float = 10.0,
        stop_premium_pts: float = 8.0,
        hold_minutes: int = 20,
        time_window_start: int = 30,    # 9:45
        time_window_end: int = 270,     # 13:45
        r: float = 0.065,
        strike_step: int = 50,
    ):
        super().__init__()
        self.min_below_bars = min_below_bars
        self.min_activity_z = min_activity_z
        self.require_macd_bullish = require_macd_bullish
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.strike_offset_steps = strike_offset_steps
        self.target_premium_pts = target_premium_pts
        self.stop_premium_pts = stop_premium_pts
        self.hold_minutes = hold_minutes
        self.time_window_start = time_window_start
        self.time_window_end = time_window_end
        self.r = r
        self.strike_step = strike_step

    def _check(self, state: MarketState) -> Signal | None:
        if not state.is_market_open():
            return None
        mso = state.minutes_since_open()
        if mso < self.time_window_start or mso > self.time_window_end:
            return None

        bars = state.bars_5m
        # Need enough bars for MACD warmup AND a meaningful "below" history
        min_bars = max(self.min_below_bars + 2,
                       self.macd_slow + self.macd_signal + 2)
        if len(bars) < min_bars:
            return None

        # Restrict to today's session bars for VWAP context
        session_date = state.ts.astimezone(IST).date()
        open_dt = datetime.combine(session_date, time(9, 15), tzinfo=IST)
        today_bars = [b for b in bars if b.ts >= open_dt]
        if len(today_bars) < self.min_below_bars + 1:
            return None

        # Current and previous VWAP (cumulative within today)
        vwap_now = _vwap_of(today_bars)
        vwap_prev = _vwap_of(today_bars[:-1])
        if vwap_now is None or vwap_prev is None:
            return None

        cur = today_bars[-1]
        prev = today_bars[-2]

        # Cross detection: previous close <= prev VWAP, current close > current VWAP
        if not (prev.close <= vwap_prev and cur.close > vwap_now):
            return None

        # Require the prior `min_below_bars` bars (excluding current) were all
        # at/below VWAP-at-that-time. Cheap proxy: use the prev VWAP as ref for
        # the look-back window (cumulative VWAP is stable enough intraday).
        lookback = today_bars[-1 - self.min_below_bars:-1]
        if len(lookback) < self.min_below_bars:
            return None
        # Compute VWAP at each prior bar and require close <= vwap_then
        for i in range(len(today_bars) - 1 - self.min_below_bars,
                       len(today_bars) - 1):
            vwap_i = _vwap_of(today_bars[:i + 1])
            if vwap_i is None or today_bars[i].close > vwap_i:
                return None

        # Activity confirmation
        activity_z = state.activity_z(12)
        if activity_z is None:
            return None
        if activity_z < self.min_activity_z:
            return None

        # Optional MACD bullish alignment
        macd_line_val = None
        signal_line_val = None
        if self.require_macd_bullish:
            closes = [b.close for b in bars]
            macd_line, signal_line, _ = macd(
                closes, self.macd_fast, self.macd_slow, self.macd_signal
            )
            macd_line_val = macd_line[-1]
            signal_line_val = signal_line[-1]
            if macd_line_val is None or signal_line_val is None:
                return None
            if macd_line_val < signal_line_val:
                return None

        # ---- Build CE order ----
        if state.chain is None:
            return None
        spot = state.spot
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

        expiry_close = datetime.combine(state.chain.expiry, time(15, 30), tzinfo=IST)
        t_expiry = max((expiry_close - state.ts).total_seconds() / (365 * 24 * 3600), 1e-5)
        g = black_scholes(spot, atm, t_expiry, q.iv, r=self.r, q=0.0, option_type="CE")
        entry_premium = g.price

        target_premium_gain = self.target_premium_pts
        stop_premium_loss = self.stop_premium_pts
        target_premium = entry_premium + target_premium_gain
        stop_premium = max(0.5, entry_premium - stop_premium_loss)

        delta = max(abs(g.delta), 0.05)
        target_spot = spot + target_premium_gain / delta
        stop_spot = spot - stop_premium_loss / delta

        rr = target_premium_gain / max(stop_premium_loss, 0.1)

        thesis = (
            f"VWAP reclaim CE: spot {spot:.0f} crossed above VWAP {vwap_now:.0f} "
            f"after {self.min_below_bars}+ bars below (prev close {prev.close:.0f} "
            f"<= prev VWAP {vwap_prev:.0f}). activity_z={activity_z:+.2f}"
            + (f", MACD {macd_line_val:.2f} >= signal {signal_line_val:.2f}"
               if self.require_macd_bullish else ", MACD filter off")
            + f".  BUY CE {atm} @ ~Rs {entry_premium:.0f} (delta={g.delta:.2f}). "
            f"TARGET +{target_premium_gain:.0f}pts (=Rs {target_premium:.0f}, spot {target_spot:.0f}).  "
            f"STOP -{stop_premium_loss:.0f}pts (=Rs {stop_premium:.0f}, spot {stop_spot:.0f}).  "
            f"TIME EXIT: {self.hold_minutes} min. R/R={rr:.2f}."
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
                "vwap_now": round(vwap_now, 2),
                "vwap_prev": round(vwap_prev, 2),
                "prev_close": prev.close,
                "cur_close": cur.close,
                "min_below_bars": self.min_below_bars,
                "activity_z": round(activity_z, 2),
                "macd_line": macd_line_val,
                "signal_line": signal_line_val,
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
