"""
Simple EMA Crossover scalp — no PSAR, no MACD, no OI filter.

Just one thing:
  - EMA20 crosses above EMA50 -> BUY CE (ATM)
  - EMA20 crosses below EMA50 -> BUY PE (ATM)

Exit:
  - Target: entry_premium + target_premium_pts (small, e.g. 15 pts)
  - Stop:   entry_premium - stop_premium_pts (tight, e.g. 10 pts)
  - Timeout: hold_minutes elapsed (default 30)

That's the entire rule.
"""
from __future__ import annotations

from datetime import time

from ..signals import BaseRule, Signal
from ..state import MarketState
from ..pricing import black_scholes
from ..indicators import ema, crossed_above, crossed_below


class SimpleEmaCross(BaseRule):
    name = "simple_ema_cross"
    cooldown_minutes = 10   # let it fire multiple times per session

    def __init__(
        self,
        ema_fast: int = 20,
        ema_slow: int = 50,
        cross_recency_bars: int = 1,
        # Exit mode: 'fixed' = use the fixed-pts numbers below;
        #           'atr'  = scale target/stop with recent volatility
        exit_mode: str = "atr",
        # Fixed mode (used when exit_mode='fixed')
        target_premium_pts: float = 15.0,
        stop_premium_pts: float = 10.0,
        # ATR mode (used when exit_mode='atr')
        atr_period: int = 14,
        target_atr_mult: float = 2.0,    # spot target = entry_spot ± 2 × ATR
        stop_atr_mult: float = 1.2,      # spot stop   = entry_spot ∓ 1.2 × ATR
        time_window_start: int = 30,     # 9:45
        time_window_end: int = 330,      # 14:45
        # Direction filter: "both", "bullish_only" (BUY CE only), "bearish_only" (BUY PE only)
        direction_filter: str = "both",
        # Time-based exit — close position if neither target nor stop hit
        hold_minutes: int = 30,
        strike_step: int = 50,
        strike_offset_steps: int = 0,
        # Trailing stop (BIG impact in choppy regimes):
        # Once premium gains `trail_activation_pts`, stop trails `trail_distance_pts`
        # below the running peak premium (high-water-mark).
        trail_activation_pts: float = 0.0,    # 0 = trailing disabled
        trail_distance_pts: float = 0.0,       # how far to trail below peak (was trail_lock_pts)
        # Volume / activity gating
        min_activity_z: float = -99.0,   # -99 = disabled; e.g. 0.5 requires above-avg bar
        activity_lookback: int = 12,
        # Volume / activity-based target/stop scaling
        # target *= (1 + activity_scale * max(0, activity_z))
        # stop   *= (1 + activity_scale * 0.5 * max(0, activity_z))
        activity_scale: float = 0.0,     # 0 = no scaling; 0.2 = 20% target widen per 1 z-score
        r: float = 0.065,
    ):
        super().__init__()
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.cross_recency_bars = cross_recency_bars
        self.exit_mode = exit_mode
        self.target_premium_pts = target_premium_pts
        self.stop_premium_pts = stop_premium_pts
        self.atr_period = atr_period
        self.target_atr_mult = target_atr_mult
        self.stop_atr_mult = stop_atr_mult
        self.time_window_start = time_window_start
        self.time_window_end = time_window_end
        self.direction_filter = direction_filter
        self.hold_minutes = hold_minutes
        self.strike_step = strike_step
        self.strike_offset_steps = strike_offset_steps
        self.trail_activation_pts = trail_activation_pts
        self.trail_distance_pts = trail_distance_pts
        self.min_activity_z = min_activity_z
        self.activity_lookback = activity_lookback
        self.activity_scale = activity_scale
        self.r = r

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
        ef = ema(closes, self.ema_fast)
        es = ema(closes, self.ema_slow)
        if ef[-1] is None or es[-1] is None:
            return None

        bullish = crossed_above(ef, es, self.cross_recency_bars)
        bearish = crossed_below(ef, es, self.cross_recency_bars)

        if not (bullish or bearish):
            return None

        # Direction filter
        if self.direction_filter == "bullish_only" and not bullish:
            return None
        if self.direction_filter == "bearish_only" and not bearish:
            return None

        # --- Volume / activity confirmation (uses volume if available, else range) ---
        activity_z = state.activity_z(self.activity_lookback)
        if activity_z is None:
            activity_z = 0.0
        if activity_z < self.min_activity_z:
            return None    # bar wasn't active enough — skip the cross

        direction = "bullish" if bullish else "bearish"
        opt_type = "CE" if bullish else "PE"
        spot = state.spot

        if state.chain is None:
            return None
        atm_base = round(spot / self.strike_step) * self.strike_step
        # OTM offset: bullish CE goes higher, bearish PE goes lower
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

        from datetime import datetime
        from ..chain import IST as CHAIN_IST
        expiry_close = datetime.combine(state.chain.expiry, time(15, 30), tzinfo=CHAIN_IST)
        t_expiry = max((expiry_close - state.ts).total_seconds() / (365 * 24 * 3600), 1e-5)
        entry_premium = black_scholes(
            spot, atm, t_expiry, q.iv, r=self.r, q=0.0, option_type=opt_type
        ).price

        g = black_scholes(spot, atm, t_expiry, q.iv, r=self.r, q=0.0, option_type=opt_type)

        # --- compute target/stop based on chosen mode ---
        if self.exit_mode == "atr":
            # ATR(period) over recent bars on spot
            recent = bars[-self.atr_period:] if len(bars) >= self.atr_period else bars
            atr = sum(b.high - b.low for b in recent) / max(len(recent), 1)
            target_spot_move = self.target_atr_mult * atr
            stop_spot_move = self.stop_atr_mult * atr
            target_premium_gain = abs(g.delta) * target_spot_move
            stop_premium_loss = abs(g.delta) * stop_spot_move
        else:
            # Fixed-premium-pts mode
            target_premium_gain = self.target_premium_pts
            stop_premium_loss = self.stop_premium_pts
            target_spot_move = target_premium_gain / max(abs(g.delta), 0.05)
            stop_spot_move = stop_premium_loss / max(abs(g.delta), 0.05)

        # --- Volume / activity scaling: hot bars get wider target + slightly wider stop ---
        if self.activity_scale > 0 and activity_z > 0:
            target_multiplier = 1.0 + self.activity_scale * activity_z
            stop_multiplier = 1.0 + 0.5 * self.activity_scale * activity_z
            target_spot_move *= target_multiplier
            stop_spot_move *= stop_multiplier
            target_premium_gain *= target_multiplier
            stop_premium_loss *= stop_multiplier

        target_premium = entry_premium + target_premium_gain
        stop_premium = max(0.5, entry_premium - stop_premium_loss)
        if direction == "bullish":
            target_spot = spot + target_spot_move
            stop_spot = spot - stop_spot_move
        else:
            target_spot = spot - target_spot_move
            stop_spot = spot + stop_spot_move

        # Compute R/R as informational
        rr = target_premium_gain / max(stop_premium_loss, 0.1)

        thesis = (
            f"Simple EMA cross {direction} [{self.exit_mode} exits, activity_z={activity_z:+.2f}]: "
            f"EMA{self.ema_fast}={ef[-1]:.0f}, EMA{self.ema_slow}={es[-1]:.0f}. "
            f"BUY {opt_type} {atm} @ ~Rs {entry_premium:.0f} (delta={g.delta:.2f}).  "
            f"TARGET: +{target_premium_gain:.0f}pts premium (=Rs {target_premium:.0f}) "
            f"when spot reaches {target_spot:.0f}.  "
            f"STOP-LOSS: -{stop_premium_loss:.0f}pts premium (=Rs {stop_premium:.0f}) "
            f"if spot hits {stop_spot:.0f}.  "
            f"TIME EXIT: close at market after {self.hold_minutes} min if neither target/stop hit.  "
            f"R/R={rr:.2f}."
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
                "exit_mode": self.exit_mode,
                "activity_z": round(activity_z, 2),
                "ema_fast": ef[-1], "ema_slow": es[-1],
                "spot": spot, "atm_strike": atm, "iv": q.iv,
                "entry_premium": entry_premium,
                "target_premium_gain": target_premium_gain,
                "stop_premium_loss": stop_premium_loss,
                "target_spot_move": target_spot_move,
                "stop_spot_move": stop_spot_move,
                "risk_reward": round(rr, 2),
                "time_exit_minutes": self.hold_minutes,
                "trail_activation_pts": self.trail_activation_pts,
                "trail_distance_pts": getattr(self, "trail_distance_pts", 0.0),
            },
            thesis=thesis[:1900],
        )
