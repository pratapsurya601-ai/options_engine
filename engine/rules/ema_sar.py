"""
EMA crossover + PSAR confirmation rule. No OI, no MACD.

Trigger:
  Bullish (BUY CE):
    - EMA20 > EMA50 (uptrend alignment)
    - EMA20 crossed above EMA50 within last N bars (cross recency)
    - PSAR is below price (confirms uptrend per parabolic-SAR)

  Bearish (BUY PE):
    - EMA20 < EMA50 (downtrend alignment)
    - EMA20 crossed below EMA50 within last N bars
    - PSAR is above price

Action: BUY ATM CE (bullish) or PE (bearish).
Stop: PSAR-based trailing OR 1.5x ATR fallback.
Target: 2x ATR (5-min ATR over 14 bars).
Hold: 60 minutes default.

HONEST: This is a pure continuation rule. EMA crossover lags the move by ~10-15
bars typically. PSAR flips can lead OR lag. Both are derivative of price.
Expect similar regime sensitivity as ORB and MACD/EMA — works in trending
markets, fails in mean-reverting ones.

Backtestable: yes (no OI dependency).
"""
from __future__ import annotations

from datetime import time

from ..signals import BaseRule, Signal
from ..state import MarketState
from ..pricing import black_scholes
from ..probability import prob_touch_above, prob_touch_below
from ..indicators import ema, parabolic_sar, crossed_above, crossed_below


class EmaSarRule(BaseRule):
    name = "ema_sar"
    cooldown_minutes = 30

    def __init__(
        self,
        ema_fast: int = 20,
        ema_slow: int = 50,
        cross_recency_bars: int = 3,
        time_window_start: int = 30,    # 9:45
        time_window_end: int = 240,     # 13:15
        hold_minutes: int = 60,
        target_atr_multiple: float = 2.0,
        stop_atr_multiple: float = 1.5,
        strike_step: int = 50,
        r: float = 0.065,
        require_psar_confirm: bool = True,
    ):
        super().__init__()
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.cross_recency_bars = cross_recency_bars
        self.time_window_start = time_window_start
        self.time_window_end = time_window_end
        self.hold_minutes = hold_minutes
        self.target_atr_multiple = target_atr_multiple
        self.stop_atr_multiple = stop_atr_multiple
        self.strike_step = strike_step
        self.r = r
        self.require_psar_confirm = require_psar_confirm

    def _check(self, state: MarketState) -> Signal | None:
        if not state.is_market_open():
            return None
        mso = state.minutes_since_open()
        if mso < self.time_window_start or mso > self.time_window_end:
            return None

        bars = state.bars_5m
        if len(bars) < self.ema_slow + 2:
            return None

        closes = [b.close for b in bars]
        highs = [b.high for b in bars]
        lows = [b.low for b in bars]

        ef = ema(closes, self.ema_fast)
        es = ema(closes, self.ema_slow)
        sar = parabolic_sar(highs, lows) if self.require_psar_confirm else [None] * len(bars)

        if ef[-1] is None or es[-1] is None:
            return None

        # Detect bullish: EMA aligned + recent cross + PSAR below price
        bullish = (
            ef[-1] > es[-1] and
            crossed_above(ef, es, self.cross_recency_bars)
        )
        bearish = (
            ef[-1] < es[-1] and
            crossed_below(ef, es, self.cross_recency_bars)
        )

        if self.require_psar_confirm and sar[-1] is not None:
            if bullish and sar[-1] >= closes[-1]:
                bullish = False
            if bearish and sar[-1] <= closes[-1]:
                bearish = False

        if not (bullish or bearish):
            return None

        direction = "bullish" if bullish else "bearish"
        opt_type = "CE" if bullish else "PE"
        spot = state.spot

        # ATR-based target/stop
        atr = sum(b.high - b.low for b in bars[-14:]) / min(14, len(bars))
        target_pts = self.target_atr_multiple * atr

        if direction == "bullish":
            target_spot = spot + target_pts
            stop_spot = (sar[-1] if sar[-1] is not None and sar[-1] < spot
                         else spot - self.stop_atr_multiple * atr)
        else:
            target_spot = spot - target_pts
            stop_spot = (sar[-1] if sar[-1] is not None and sar[-1] > spot
                         else spot + self.stop_atr_multiple * atr)

        if state.chain is None:
            return None
        atm = round(spot / self.strike_step) * self.strike_step
        by_strike = state.chain.by_strike()
        q = by_strike.get(atm, {}).get(opt_type)
        if q is None or q.iv is None or not q.ltp:
            available = [k for k in by_strike
                         if opt_type in by_strike[k]
                         and by_strike[k][opt_type].iv is not None]
            if not available:
                return None
            atm = min(available, key=lambda k: abs(k - spot))
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

        psar_str = f"{sar[-1]:.0f}" if sar[-1] is not None else "n/a"
        thesis = (
            f"EMA+SAR {direction}: EMA20={ef[-1]:.0f}, EMA50={es[-1]:.0f}, "
            f"PSAR={psar_str}; "
            f"ATR={atr:.0f}, target={target_spot:.0f} ({target_pts:.0f}pts), "
            f"stop={stop_spot:.0f}, hold <= {self.hold_minutes}m. "
            f"Premise: trend continuation after EMA cross + PSAR confirm. "
            f"Highly correlated indicators; expect regime-sensitivity."
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
                "spot": spot, "atm_strike": atm, "iv": q.iv,
                "ema20": ef[-1], "ema50": es[-1],
                "psar": sar[-1] if sar[-1] is not None else None,
                "atr_14": atr,
            },
            thesis=thesis[:1900],
        )
