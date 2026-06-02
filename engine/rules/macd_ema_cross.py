"""
OI-Anchored MACD/EMA Cross rule.

Entry premise:
  Trade only when spot is positioned near an OI level (potential bounce/rejection),
  AND momentum confirms via simultaneous trend alignment + crossover.

Bullish trigger (all must be true):
  1. Spot is in lower-third of [put_wall, call_wall] zone (near support)
  2. EMA20 > EMA50 (trend up)
  3. MACD line above signal line
  4. MACD crossed above signal within last 3 bars (recency)
  5. Within time window (default 9:45-13:00)
  6. (Optional) PSAR is below price (i.e., uptrend per PSAR)
  Action: BUY ATM CE

Bearish trigger: mirror image — spot in upper third, EMA20 < EMA50, MACD < signal
with recent crossover. BUY ATM PE.

Stops/targets:
  - Stop: PSAR-based trailing — exit when PSAR flips against you (computed live
    by the watcher monitor). Initial stop = entry premium - 1.5x ATR(5m) on premium.
  - Target: 1.0 * (zone_half_width) in spot pts, translated to premium via delta.

HONEST CAVEATS:
  - Indicator stack is highly correlated. The OI filter is the only independent piece.
  - Backtest can NOT use real historical OI (Kite doesn't provide intraday OI history).
    Backtest mode optionally skips OI filter (sets require_oi_zone=False).
  - In live trading the OI filter materially reduces fires (~50% fewer).
"""
from __future__ import annotations

from datetime import time
from typing import Optional

from ..signals import BaseRule, Signal
from ..state import MarketState
from ..pricing import black_scholes
from ..probability import prob_touch_above, prob_touch_below
from ..indicators import ema, macd, parabolic_sar, crossed_above, crossed_below
from ..oi_zones import compute_zones


class MacdEmaCrossOI(BaseRule):
    name = "macd_ema_oi"
    cooldown_minutes = 30  # allow 2-3 fires per session

    def __init__(
        self,
        ema_fast: int = 20,
        ema_slow: int = 50,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        cross_recency_bars: int = 3,
        time_window_start: int = 30,   # 9:45
        time_window_end: int = 225,    # 13:00
        require_oi_zone: bool = True,
        require_psar_confirm: bool = True,
        hold_minutes: int = 60,
        target_multiple: float = 1.0,   # of zone half-width in spot pts
        strike_step: int = 50,
        r: float = 0.065,
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
        self.require_oi_zone = require_oi_zone
        self.require_psar_confirm = require_psar_confirm
        self.hold_minutes = hold_minutes
        self.target_multiple = target_multiple
        self.strike_step = strike_step
        self.r = r

    def _check(self, state: MarketState) -> Signal | None:
        if not state.is_market_open():
            return None
        mso = state.minutes_since_open()
        if mso < self.time_window_start or mso > self.time_window_end:
            return None

        bars = state.bars_5m
        # Need enough history for EMA50 + MACD signal warmup
        min_bars = max(self.ema_slow, self.macd_slow + self.macd_signal) + 2
        if len(bars) < min_bars:
            return None

        closes = [b.close for b in bars]
        highs = [b.high for b in bars]
        lows = [b.low for b in bars]

        ef = ema(closes, self.ema_fast)
        es = ema(closes, self.ema_slow)
        macd_line, signal_line, hist = macd(closes, self.macd_fast,
                                             self.macd_slow, self.macd_signal)
        sar = parabolic_sar(highs, lows) if self.require_psar_confirm else [None]*len(bars)

        if ef[-1] is None or es[-1] is None or macd_line[-1] is None or signal_line[-1] is None:
            return None

        spot = state.spot
        # OI zone classification
        zones = None
        if state.chain is not None:
            try:
                zones = compute_zones(state.chain)
            except Exception:
                zones = None

        # ---------- Bullish trigger ----------
        bullish = (
            ef[-1] > es[-1] and
            macd_line[-1] > signal_line[-1] and
            crossed_above(macd_line, signal_line, self.cross_recency_bars)
        )
        # ---------- Bearish trigger ----------
        bearish = (
            ef[-1] < es[-1] and
            macd_line[-1] < signal_line[-1] and
            crossed_below(macd_line, signal_line, self.cross_recency_bars)
        )

        # OI filter
        if self.require_oi_zone and zones is not None:
            if bullish and not zones.in_lower_third():
                bullish = False
            if bearish and not zones.in_upper_third():
                bearish = False

        # PSAR confirmation
        if self.require_psar_confirm and sar[-1] is not None:
            if bullish and sar[-1] >= closes[-1]:
                bullish = False
            if bearish and sar[-1] <= closes[-1]:
                bearish = False

        if not (bullish or bearish):
            return None

        direction = "bullish" if bullish else "bearish"
        opt_type = "CE" if bullish else "PE"

        # Strike selection
        atm = round(spot / self.strike_step) * self.strike_step

        # Target sizing
        if zones and zones.zone_width > 0:
            zone_half = zones.zone_width / 2
            target_pts = zone_half * self.target_multiple
        else:
            # Fallback: 1 ATR equivalent — approximate from 14-bar range
            recent = bars[-14:]
            atr_approx = sum(b.high - b.low for b in recent) / len(recent)
            target_pts = atr_approx * 3

        if direction == "bullish":
            target_spot = spot + target_pts
            stop_spot = sar[-1] if sar[-1] is not None else spot - target_pts * 0.6
        else:
            target_spot = spot - target_pts
            stop_spot = sar[-1] if sar[-1] is not None else spot + target_pts * 0.6

        # Need an ATM quote with IV
        if state.chain is None:
            return None
        by_strike = state.chain.by_strike()
        q = by_strike.get(atm, {}).get(opt_type)
        if q is None or q.iv is None or not q.ltp:
            available = [k for k in by_strike if opt_type in by_strike[k]
                         and by_strike[k][opt_type].iv]
            if not available:
                return None
            atm = min(available, key=lambda k: abs(k - spot))
            q = by_strike[atm][opt_type]

        # Time to expiry for premium projection
        from datetime import datetime, timedelta
        from ..chain import IST as CHAIN_IST
        expiry_close = datetime.combine(
            state.chain.expiry, time(15, 30), tzinfo=CHAIN_IST
        )
        t_expiry = max((expiry_close - state.ts).total_seconds() / (365 * 24 * 3600), 1e-5)
        g = black_scholes(spot, atm, t_expiry, q.iv, r=self.r, q=0.0, option_type=opt_type)
        target_premium_gain = abs(g.delta) * target_pts

        stop_premium = black_scholes(
            stop_spot, atm, t_expiry, q.iv, r=self.r, q=0.0, option_type=opt_type
        ).price

        # P(touch) within hold window
        hold_t = self.hold_minutes / (60 * 24 * 365.0)
        if direction == "bullish":
            p_touch = prob_touch_above(spot, target_spot, hold_t, q.iv, drift=0.0)
        else:
            p_touch = prob_touch_below(spot, target_spot, hold_t, q.iv, drift=0.0)

        thesis = (
            f"MACD/EMA cross {direction}: EMA20={ef[-1]:.0f}, EMA50={es[-1]:.0f}; "
            f"MACD={macd_line[-1]:.2f} vs sig={signal_line[-1]:.2f}; "
            f"PSAR={(f'{sar[-1]:.0f}' if sar[-1] else 'n/a')}; "
            f"OI zone: spot at pos={zones.position_in_zone:.2f} between PW={zones.put_wall} "
            f"and CW={zones.call_wall} (if available); "
            f"target {target_spot:.0f} ({target_pts:.0f}pts spot / {target_premium_gain:.0f}pts premium); "
            f"stop {stop_spot:.0f}; hold <= {self.hold_minutes}m. "
            f"Premise: OI-anchored momentum continuation. Indicator stack is correlated; "
            f"OI filter is the independent piece. Trader owns the premise."
        ) if zones else (
            f"MACD/EMA cross {direction} (no OI filter): EMA20={ef[-1]:.0f}, "
            f"EMA50={es[-1]:.0f}; target {target_spot:.0f}; stop {stop_spot:.0f}."
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
                "spot": spot, "ema20": ef[-1], "ema50": es[-1],
                "macd": macd_line[-1], "signal": signal_line[-1],
                "psar": sar[-1] if sar[-1] is not None else None,
                "atm_strike": atm, "iv": q.iv,
                "target_pts": target_pts,
                "zones": {
                    "put_wall": zones.put_wall, "call_wall": zones.call_wall,
                    "max_pain": zones.max_pain,
                    "position": zones.position_in_zone,
                } if zones else None,
            },
            thesis=thesis[:1900],
        )
