"""
OI-Direction + (PSAR or MACD/EMA) Trigger rule.

Strategy:
  Step 1 — Direction from OI:
    Compute OI zones (put wall, call wall, max pain). Determine bias:
      bullish if spot in lower-third of [put_wall, call_wall] AND spot < max_pain
      bearish if spot in upper-third of [put_wall, call_wall] AND spot > max_pain
    Mixed signals (e.g. spot near put wall but above max pain) -> no trade.

  Step 2 — Wait for trigger (EITHER):
    A) PSAR flip in the bias direction (just crossed price)
    B) MACD/EMA crossover (MACD crosses signal in direction, EMA20/50 aligned)

  Fire on the first trigger that fires after direction is established.

Honest framing:
  - OI direction is the independent edge. Indicator triggers are timing.
  - PSAR flips lag the move by ~2 bars typically; MACD crosses by 3-5 bars.
  - Both triggers can give a "second chance" entry — more fires than requiring both.
  - Requires LIVE OI data; cannot be backtested with Kite historical_data.

Action:
  - Bullish: BUY ATM CE
  - Bearish: BUY ATM PE

Exit (handled by watcher monitor):
  - Target: spot moves toward call_wall (bullish) or put_wall (bearish)
  - Stop: PSAR flips against position OR initial stop ~1.5× ATR(14)
  - Time-out: hold_minutes elapsed
"""
from __future__ import annotations

from datetime import time, timedelta
from typing import Literal

from ..signals import BaseRule, Signal
from ..state import MarketState
from ..pricing import black_scholes
from ..probability import prob_touch_above, prob_touch_below
from ..indicators import ema, macd, parabolic_sar, crossed_above, crossed_below
from ..oi_zones import compute_zones


class OIDirectionRule(BaseRule):
    name = "oi_direction"
    cooldown_minutes = 30   # allow 2-3 fires per session

    def __init__(
        self,
        ema_fast: int = 20,
        ema_slow: int = 50,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        cross_recency_bars: int = 3,
        time_window_start: int = 30,    # 9:45
        time_window_end: int = 240,     # 13:15
        hold_minutes: int = 60,
        target_atr_multiple: float = 2.0,
        strike_step: int = 50,
        r: float = 0.065,
        require_both_triggers: bool = False,
    ):
        super().__init__()
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.cross_recency_bars = cross_recency_bars
        self.time_window_start = time_window_start
        self.time_window_end = time_window_end
        self.hold_minutes = hold_minutes
        self.target_atr_multiple = target_atr_multiple
        self.strike_step = strike_step
        self.r = r
        self.require_both_triggers = require_both_triggers

    def _direction_from_oi(self, zones, spot: float) -> Literal["bullish", "bearish", None]:
        """Determine directional bias from OI zones + max pain."""
        if zones is None or zones.position_in_zone is None or zones.max_pain is None:
            return None
        pos = zones.position_in_zone
        max_pain_pull_bullish = spot < zones.max_pain
        max_pain_pull_bearish = spot > zones.max_pain

        # Lower-third + max pain above = bullish (bounce off put wall + magnet up)
        if pos <= 0.33 and max_pain_pull_bullish:
            return "bullish"
        # Upper-third + max pain below = bearish (rejection at call wall + magnet down)
        if pos >= 0.67 and max_pain_pull_bearish:
            return "bearish"
        return None

    def _check_triggers(self, direction: str, closes, highs, lows):
        """Return dict of which triggers fired this bar."""
        ef = ema(closes, self.ema_fast)
        es = ema(closes, self.ema_slow)
        macd_line, signal_line, _ = macd(
            closes, self.macd_fast, self.macd_slow, self.macd_signal,
        )
        sar = parabolic_sar(highs, lows)

        triggers = {"macd_ema": False, "psar": False}

        if ef[-1] is None or es[-1] is None or macd_line[-1] is None or signal_line[-1] is None:
            return triggers, ef, es, macd_line, signal_line, sar

        if direction == "bullish":
            ema_aligned = ef[-1] > es[-1]
            macd_aligned = macd_line[-1] > signal_line[-1]
            macd_crossed = crossed_above(macd_line, signal_line, self.cross_recency_bars)
            triggers["macd_ema"] = ema_aligned and macd_aligned and macd_crossed
            # PSAR flip: was above price recently, now below
            if len(sar) >= 2 and sar[-1] is not None and sar[-2] is not None:
                triggers["psar"] = sar[-1] < closes[-1] and sar[-2] >= closes[-2]
        else:
            ema_aligned = ef[-1] < es[-1]
            macd_aligned = macd_line[-1] < signal_line[-1]
            macd_crossed = crossed_below(macd_line, signal_line, self.cross_recency_bars)
            triggers["macd_ema"] = ema_aligned and macd_aligned and macd_crossed
            if len(sar) >= 2 and sar[-1] is not None and sar[-2] is not None:
                triggers["psar"] = sar[-1] > closes[-1] and sar[-2] <= closes[-2]

        return triggers, ef, es, macd_line, signal_line, sar

    def _check(self, state: MarketState) -> Signal | None:
        if not state.is_market_open():
            return None
        mso = state.minutes_since_open()
        if mso < self.time_window_start or mso > self.time_window_end:
            return None

        bars = state.bars_5m
        min_bars = max(self.ema_slow, self.macd_slow + self.macd_signal) + 2
        if len(bars) < min_bars:
            return None

        if state.chain is None:
            return None

        # ---- Step 1: direction from OI ----
        try:
            zones = compute_zones(state.chain)
        except Exception:
            return None

        direction = self._direction_from_oi(zones, state.spot)
        if direction is None:
            return None

        # ---- Step 2: trigger ----
        closes = [b.close for b in bars]
        highs = [b.high for b in bars]
        lows = [b.low for b in bars]

        triggers, ef, es, macd_line, signal_line, sar = self._check_triggers(
            direction, closes, highs, lows
        )

        fired = (
            (triggers["macd_ema"] and triggers["psar"])
            if self.require_both_triggers
            else (triggers["macd_ema"] or triggers["psar"])
        )
        if not fired:
            return None

        # ---- Build signal ----
        spot = state.spot
        opt_type = "CE" if direction == "bullish" else "PE"
        atm = round(spot / self.strike_step) * self.strike_step

        # Target from OI zones — toward the opposite wall
        if direction == "bullish":
            target_spot = min(zones.call_wall or (spot + 100), spot + 0.6 * zones.zone_width)
        else:
            target_spot = max(zones.put_wall or (spot - 100), spot - 0.6 * zones.zone_width)
        target_pts = abs(target_spot - spot)

        # Stop: PSAR (trailing) or 1.5x ATR fallback
        atr = sum(b.high - b.low for b in bars[-14:]) / min(14, len(bars))
        if direction == "bullish":
            stop_spot = sar[-1] if sar[-1] is not None and sar[-1] < spot else spot - 1.5 * atr
        else:
            stop_spot = sar[-1] if sar[-1] is not None and sar[-1] > spot else spot + 1.5 * atr

        # Premium at ATM
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

        trig_names = [k for k, v in triggers.items() if v]
        thesis = (
            f"OI-direction {direction}: spot={spot:.0f} in zone pos={zones.position_in_zone:.2f} "
            f"[PW={zones.put_wall}, CW={zones.call_wall}], max_pain={zones.max_pain} "
            f"({'above' if spot > zones.max_pain else 'below'} spot). "
            f"Triggers fired: {', '.join(trig_names)} "
            f"(EMA20={ef[-1]:.0f}, EMA50={es[-1]:.0f}, MACD-sig={(macd_line[-1] - signal_line[-1]):+.2f}, "
            f"PSAR={(f'{sar[-1]:.0f}' if sar[-1] is not None else 'n/a')}). "
            f"Target spot={target_spot:.0f} ({target_pts:.0f}pts spot / {target_premium_gain:.0f}pts premium); "
            f"stop spot={stop_spot:.0f}; hold <= {self.hold_minutes}m. "
            f"Premise: OI direction is independent edge; indicators are timing. "
            f"Validate by tracking fires and outcomes."
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
                "direction_source": "oi_zones",
                "zones": {
                    "put_wall": zones.put_wall, "call_wall": zones.call_wall,
                    "max_pain": zones.max_pain,
                    "position_in_zone": zones.position_in_zone,
                },
                "triggers": trig_names,
                "ema20": ef[-1], "ema50": es[-1],
                "macd": macd_line[-1], "signal": signal_line[-1],
                "psar": sar[-1] if sar[-1] is not None else None,
                "spot": spot, "atm_strike": atm, "iv": q.iv,
            },
            thesis=thesis[:1900],
        )
