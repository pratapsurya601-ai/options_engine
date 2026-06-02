"""
Fast-timeframe EMA cross + retest rule — tuned for 2-min / 5-min bars.

Concept (same as HtfEmaRetest but for faster TFs):
  1. EMA20 crossed EMA50 within the last `retest_lookback` bars
     (excluding the very last 2 bars so a meaningful retest can occur).
  2. Alignment still intact (EMA20 still above EMA50 for bullish, etc.).
  3. Price pulled back to EMA20 (within retest_pct).
  4. Current bar confirms direction (close back in trend direction).

Tighter retest, more lookback bars, smaller target/stop than HTF version.
"""
from __future__ import annotations

from datetime import time, datetime

from ..signals import BaseRule, Signal
from ..state import MarketState
from ..pricing import black_scholes
from ..indicators import ema, crossed_above, crossed_below


class FastEmaRetest(BaseRule):
    name = "fast_ema_retest"
    cooldown_minutes = 30  # faster TF -> allow more fires per session

    def __init__(
        self,
        direction_filter: str = "both",
        ema_fast: int = 20,
        ema_slow: int = 50,
        retest_lookback: int = 20,     # more bars on faster TF
        retest_pct: float = 0.003,     # tighter retest on faster TF (0.3%)
        strike_offset_steps: int = 0,  # ATM default
        target_premium_pts: float = 10.0,
        stop_premium_pts: float = 8.0,
        hold_minutes: int = 20,
        time_window_start: int = 30,
        time_window_end: int = 270,
        strike_step: int = 50,
        r: float = 0.065,
    ):
        super().__init__()
        self.direction_filter = direction_filter
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.retest_lookback = retest_lookback
        self.retest_pct = retest_pct
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
        needed = self.ema_slow + self.retest_lookback + 5
        if len(bars) < needed:
            return None

        closes = [b.close for b in bars]
        ef = ema(closes, self.ema_fast)
        es = ema(closes, self.ema_slow)
        if ef[-1] is None or es[-1] is None:
            return None

        # --- 1. Detect cross within [-retest_lookback, -2] (exclude last 2 bars) ---
        bullish_cross = False
        bearish_cross = False
        for k in range(2, self.retest_lookback + 1):
            # crossed_above checks if cross happened on bar at index -k
            # We slice up to that point
            sub_ef = ef[: len(ef) - (k - 1)]
            sub_es = es[: len(es) - (k - 1)]
            if len(sub_ef) < 2 or sub_ef[-1] is None or sub_ef[-2] is None:
                continue
            if sub_es[-1] is None or sub_es[-2] is None:
                continue
            if sub_ef[-2] <= sub_es[-2] and sub_ef[-1] > sub_es[-1]:
                bullish_cross = True
                break
            if sub_ef[-2] >= sub_es[-2] and sub_ef[-1] < sub_es[-1]:
                bearish_cross = True
                break

        if not (bullish_cross or bearish_cross):
            return None

        # --- 2. Alignment still intact ---
        if bullish_cross and not (ef[-1] > es[-1]):
            return None
        if bearish_cross and not (ef[-1] < es[-1]):
            return None

        if self.direction_filter == "bullish_only" and not bullish_cross:
            return None
        if self.direction_filter == "bearish_only" and not bearish_cross:
            return None

        # --- 3. Price pulled back to EMA20 (within retest_pct) ---
        # Check that in last few bars, price touched (came within retest_pct of) EMA20
        retest_touched = False
        for j in range(2, 8):  # look at last few bars (excluding current)
            if j >= len(bars):
                break
            b = bars[-j]
            ef_j = ef[-j]
            if ef_j is None:
                continue
            dist = abs(b.low - ef_j) if bullish_cross else abs(b.high - ef_j)
            # Tighter: just check distance ratio
            min_dist_pct = min(
                abs(b.high - ef_j) / ef_j,
                abs(b.low - ef_j) / ef_j,
                abs(b.close - ef_j) / ef_j,
            )
            if min_dist_pct <= self.retest_pct:
                retest_touched = True
                break

        if not retest_touched:
            return None

        # --- 4. Current bar confirms direction ---
        last_bar = bars[-1]
        if bullish_cross and not (last_bar.close > last_bar.open):
            return None
        if bearish_cross and not (last_bar.close < last_bar.open):
            return None
        # Also confirm current bar close is on the trend side of EMA20
        if bullish_cross and not (last_bar.close > ef[-1]):
            return None
        if bearish_cross and not (last_bar.close < ef[-1]):
            return None

        direction = "bullish" if bullish_cross else "bearish"
        opt_type = "CE" if bullish_cross else "PE"
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
            f"Fast EMA retest {direction}: EMA{self.ema_fast}={ef[-1]:.0f} vs "
            f"EMA{self.ema_slow}={es[-1]:.0f}, recent {direction} cross + retest. "
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
                "retest_lookback": self.retest_lookback,
                "retest_pct": self.retest_pct,
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
