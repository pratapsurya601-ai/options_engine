"""
Panic Bounce CE — BNF-inspired NIFTY options rule.

Takashi Kotegawa (BNF) bought stocks that had dropped excessively below their
moving average and bounced back to the MA. Adapted for NIFTY index options:

Setup (ALL must hold):
  1. Spot is >= min_drop_pct (default 2.5%) below the 20-day SMA.
  2. Last `drop_lookback_days` daily closes have ALL been below 20-day SMA
     (sustained panic, not a one-day blip).
  3. 14-period RSI on the BAR series < rsi_max (default 35) — oversold.
  4. Latest 5-min bar is a bullish reversal bar:
       - close > open  (green)
       - close in upper half of (high-low) range
       - body >= 30% of total range (not a doji)
  5. activity_z >= min_activity_z (default 0.3).
  6. mso in [time_window_start, time_window_end] (default 9:45-12:00).

Action: BUY ATM CE (configurable offset).
Target: spot back to the 20-day SMA  =>  target_premium = entry + delta*(sma_20 - spot)
Stop  : 25% premium loss.
Hold  : 750 min (~current session + next morning).
Trail : after +1R gain, stop moves to entry (breakeven).

Cooldown: 1440 min => at most one fire per day.
"""
from __future__ import annotations

from datetime import datetime, time, timedelta

from ..signals import BaseRule, Signal
from ..state import MarketState
from ..pricing import black_scholes
from ..indicators import rsi
from ..chain import IST as CHAIN_IST


class PanicBounceCE(BaseRule):
    name = "panic_bounce_ce"
    cooldown_minutes = 1440   # max one fire per day

    def __init__(
        self,
        min_drop_pct: float = 0.025,
        drop_lookback_days: int = 3,
        rsi_period: int = 14,
        rsi_max: float = 35.0,
        min_activity_z: float = 0.3,
        strike_offset_steps: int = 0,
        stop_premium_pct: float = 0.25,
        hold_minutes: int = 750,
        time_window_start: int = 30,    # 9:45
        time_window_end: int = 165,     # 12:00
        strike_step: int = 50,
        r: float = 0.065,
        retracement_frac: float = 0.4,
        atr_target_mult: float = 2.0,
        atr_period: int = 14,
    ):
        super().__init__()
        self.min_drop_pct = min_drop_pct
        self.drop_lookback_days = drop_lookback_days
        self.rsi_period = rsi_period
        self.rsi_max = rsi_max
        self.min_activity_z = min_activity_z
        self.strike_offset_steps = strike_offset_steps
        self.stop_premium_pct = stop_premium_pct
        self.hold_minutes = hold_minutes
        self.time_window_start = time_window_start
        self.time_window_end = time_window_end
        self.strike_step = strike_step
        self.r = r
        self.retracement_frac = retracement_frac
        self.atr_target_mult = atr_target_mult
        self.atr_period = atr_period

        # Daily SMA cache: (date) -> (sma_20, [last N closes])
        self._daily_cache: dict = {}
        self._daily_bars_cache: list | None = None

    def _load_daily_bars(self):
        if self._daily_bars_cache is not None:
            return self._daily_bars_cache
        try:
            from ..data.cache import load_cache
            from ..state import Bar
            df = load_cache("NIFTY", "day")
            if df.empty:
                self._daily_bars_cache = []
                return self._daily_bars_cache
            df = df.sort_values("ts").reset_index(drop=True)
            self._daily_bars_cache = [
                Bar(
                    ts=row["ts"].to_pydatetime() if hasattr(row["ts"], "to_pydatetime") else row["ts"],
                    open=float(row["open"]), high=float(row["high"]),
                    low=float(row["low"]), close=float(row["close"]),
                    volume=int(row["volume"]),
                )
                for _, row in df.iterrows()
            ]
        except Exception:
            self._daily_bars_cache = []
        return self._daily_bars_cache

    def _get_daily_context(self, ref_date):
        """
        Compute 20-day SMA and the last `drop_lookback_days` daily closes that
        occurred STRICTLY BEFORE ref_date. Cached per ref_date.
        """
        if ref_date in self._daily_cache:
            return self._daily_cache[ref_date]

        daily_bars = self._load_daily_bars()
        if not daily_bars:
            self._daily_cache[ref_date] = None
            return None

        prior = [b for b in daily_bars if b.ts.date() < ref_date]
        if len(prior) < 20:
            self._daily_cache[ref_date] = None
            return None

        last20 = prior[-20:]
        sma_20 = sum(b.close for b in last20) / 20.0
        recent_closes = [b.close for b in prior[-self.drop_lookback_days:]]

        # ATR(atr_period) on prior daily bars (true range avg)
        atr_14 = None
        if len(prior) >= self.atr_period + 1:
            atr_bars = prior[-(self.atr_period + 1):]
            trs = []
            for i in range(1, len(atr_bars)):
                cur = atr_bars[i]
                prev_close = atr_bars[i - 1].close
                tr = max(
                    cur.high - cur.low,
                    abs(cur.high - prev_close),
                    abs(cur.low - prev_close),
                )
                trs.append(tr)
            if trs:
                atr_14 = sum(trs) / len(trs)

        out = (sma_20, recent_closes, atr_14)
        self._daily_cache[ref_date] = out
        return out

    def _check(self, state: MarketState) -> Signal | None:
        if not state.is_market_open():
            return None
        mso = state.minutes_since_open()
        if mso < self.time_window_start or mso > self.time_window_end:
            return None

        bars = state.bars_5m
        if len(bars) < self.rsi_period + 2:
            return None

        spot = state.spot

        # --- Daily SMA context ---
        ctx = self._get_daily_context(state.ts.date())
        if ctx is None:
            return None
        sma_20, recent_closes, atr_14 = ctx
        if sma_20 <= 0:
            return None

        # 1. Drop magnitude
        drop_pct = (sma_20 - spot) / sma_20
        if drop_pct < self.min_drop_pct:
            return None

        # 2. Sustained drop — last N daily closes all below SMA
        if len(recent_closes) < self.drop_lookback_days:
            return None
        if not all(c < sma_20 for c in recent_closes):
            return None

        # 3. RSI oversold on bar series
        closes = [b.close for b in bars]
        rsi_vals = rsi(closes, self.rsi_period)
        rsi_now = rsi_vals[-1]
        if rsi_now is None or rsi_now >= self.rsi_max:
            return None

        # 4. Reversal bar shape
        bar = bars[-1]
        rng = bar.high - bar.low
        if rng <= 0:
            return None
        body = abs(bar.close - bar.open)
        body_pct = body / rng
        upper_close_pct = (bar.close - bar.low) / rng    # 1.0 = closed at high

        is_green = bar.close > bar.open
        upper_half_close = (bar.high - bar.close) < (bar.close - bar.low)
        decent_body = body_pct > 0.30
        if not (is_green and upper_half_close and decent_body):
            return None

        # 5. Volume / activity
        activity_z = state.activity_z(14)
        if activity_z is None:
            activity_z = 0.0
        if activity_z < self.min_activity_z:
            return None

        # --- Build option leg ---
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

        expiry_close = datetime.combine(state.chain.expiry, time(15, 30), tzinfo=CHAIN_IST)
        t_expiry = max((expiry_close - state.ts).total_seconds() / (365 * 24 * 3600), 1e-5)
        g = black_scholes(spot, atm, t_expiry, q.iv, r=self.r, q=0.0, option_type="CE")
        entry_premium = g.price
        if entry_premium <= 0.5:
            return None

        delta_abs = max(abs(g.delta), 0.05)

        # Target: partial retracement toward SMA, capped by ATR multiple.
        sma_20_spot = sma_20
        raw_target_spot = spot + self.retracement_frac * (sma_20 - spot)
        if atr_14 is not None and atr_14 > 0:
            max_spot_move = self.atr_target_mult * atr_14
            atr_capped_spot = spot + max_spot_move
            target_spot = min(raw_target_spot, atr_capped_spot)
        else:
            target_spot = raw_target_spot
        spot_gain_to_target = target_spot - spot
        target_premium_gain = delta_abs * spot_gain_to_target
        target_premium = entry_premium + target_premium_gain

        # Stop: 25% premium loss
        stop_premium = max(0.5, entry_premium * (1.0 - self.stop_premium_pct))
        risk_pts = entry_premium - stop_premium
        stop_spot_move = risk_pts / delta_abs
        stop_spot = spot - stop_spot_move

        # Trail: after +1R premium gain, stop -> entry (BE)
        trail_activation_pts = risk_pts
        trail_distance_pts = risk_pts

        rr = target_premium_gain / max(risk_pts, 0.1)

        thesis = (
            f"Panic Bounce CE [BNF-inspired]: spot={spot:.0f} is {drop_pct*100:.2f}% below "
            f"20d SMA={sma_20:.0f}; last {self.drop_lookback_days} daily closes all below SMA. "
            f"RSI({self.rsi_period})={rsi_now:.1f} < {self.rsi_max:.0f}. Green reversal bar "
            f"(body={body_pct*100:.0f}%, close_upper={upper_close_pct*100:.0f}%), "
            f"activity_z={activity_z:+.2f}. "
            f"BUY CE {atm} @ ~Rs {entry_premium:.0f} (delta={g.delta:.2f}). "
            f"TARGET: spot -> {target_spot:.0f} ({self.retracement_frac*100:.0f}% retrace to SMA {sma_20_spot:.0f}, "
            f"ATR cap {self.atr_target_mult:.1f}x={atr_14 if atr_14 else 0:.0f}) "
            f"(=Rs {target_premium:.0f}, +{target_premium_gain:.0f}pts, R/R={rr:.2f}). "
            f"STOP: -{self.stop_premium_pct*100:.0f}% (=Rs {stop_premium:.0f}, spot {stop_spot:.0f}). "
            f"TRAIL: on +1R, stop -> entry (BE). HOLD: {self.hold_minutes}m."
        )

        return Signal(
            rule_name=self.name,
            ts=state.ts,
            direction="bullish",
            action="BUY_CE",
            strike=atm,
            option_type="CE",
            target_premium_gain=round(target_premium_gain, 2),
            target_spot=round(target_spot, 1),
            stop_spot=round(stop_spot, 1),
            stop_premium=round(stop_premium, 2),
            hold_minutes=self.hold_minutes,
            prob_estimates={},
            trigger_context={
                "sma_20": round(sma_20, 2),
                "sma_20_spot": round(sma_20_spot, 2),
                "atr_14": round(atr_14, 2) if atr_14 is not None else None,
                "retracement_frac": self.retracement_frac,
                "atr_target_mult": self.atr_target_mult,
                "current_spot": round(spot, 2),
                "drop_pct": round(drop_pct, 4),
                "rsi_14": round(rsi_now, 2),
                "bar_shape": {
                    "open": round(bar.open, 2),
                    "high": round(bar.high, 2),
                    "low": round(bar.low, 2),
                    "close": round(bar.close, 2),
                    "body_pct": round(body_pct, 3),
                    "upper_close_pct": round(upper_close_pct, 3),
                },
                "volume_z": round(activity_z, 2),
                "target_spot": round(target_spot, 1),
                "target_premium_gain": round(target_premium_gain, 2),
                "entry_premium": round(entry_premium, 2),
                "stop_premium": round(stop_premium, 2),
                "delta": round(g.delta, 4),
                "iv": q.iv,
                "atm_strike": atm,
                "strike_offset_steps": self.strike_offset_steps,
                "expiry": state.chain.expiry.isoformat(),
                "trail_activation_pts": round(trail_activation_pts, 2),
                "trail_distance_pts": round(trail_distance_pts, 2),
                "risk_reward": round(rr, 2),
                "time_exit_minutes": self.hold_minutes,
            },
            thesis=thesis[:1900],
        )
