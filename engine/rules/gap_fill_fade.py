"""
Gap-Fill Fade rule.

Premise (regime-appropriate for mean-reverting NIFTY):
  Morning gaps statistically tend to FILL toward prior-day close. Most gap
  research on Indian index futures shows ~70-80% fill rate when gap is
  between 0.3% and 1.5% — too small = noise, too large = news-driven.

Entry conditions (all must hold):
  1. Today's open differs from prior day's close by between MIN_GAP and MAX_GAP
  2. After 9:45 (post-activation), the 15-min opening range shows REJECTION
     of the gap direction:
       - Gap UP   -> last bar close is BELOW OR midpoint (sellers pushed back)
       - Gap DOWN -> last bar close is ABOVE OR midpoint (buyers pushed back)
  3. Within trade time window (default 9:45-12:00)

Action:
  - Gap UP   -> BUY PE (expecting fade DOWN to prior close)
  - Gap DOWN -> BUY CE (expecting fade UP to prior close)

Target = state.spot - target_fill_pct * gap (fade 70% of the way by default)
Stop   = state.spot + stop_extension_pct * gap (gap extends 50% further -> stop)
Hold   = 90 minutes

HONEST CAVEATS:
  - Regime-dependent. In trending markets (e.g. budget rally, election trend),
    gaps DON'T fill - they extend. Use IV-rank / VIX to gate live trades.
  - Big-news gaps (>1.5%) are usually trending; we filter those out.
  - Earnings season can change behavior on individual sectors but NIFTY index
    smooths this out.
  - Fire rate: 1-3 per week in current regime (most days have small gaps).
"""
from __future__ import annotations

from datetime import time, timedelta

from ..signals import BaseRule, Signal
from ..state import MarketState
from ..pricing import black_scholes
from ..probability import prob_touch_above, prob_touch_below


class GapFillFade(BaseRule):
    name = "gap_fill_fade"
    cooldown_minutes = 240   # one fire per session max

    def __init__(
        self,
        min_gap_pct: float = 0.003,        # 0.3% minimum gap
        max_gap_pct: float = 0.015,        # 1.5% maximum (avoid news gaps)
        or_minutes: int = 15,
        rejection_required: bool = True,    # require OR rejection of gap
        time_window_start: int = 30,        # 9:45
        time_window_end: int = 165,         # 12:00
        target_fill_pct: float = 0.7,       # fade 70% of gap
        stop_extension_pct: float = 0.5,    # stop if gap extends 50% more
        hold_minutes: int = 90,
        strike_step: int = 50,
        r: float = 0.065,
    ):
        super().__init__()
        self.min_gap_pct = min_gap_pct
        self.max_gap_pct = max_gap_pct
        self.or_minutes = or_minutes
        self.rejection_required = rejection_required
        self.time_window_start = time_window_start
        self.time_window_end = time_window_end
        self.target_fill_pct = target_fill_pct
        self.stop_extension_pct = stop_extension_pct
        self.hold_minutes = hold_minutes
        self.strike_step = strike_step
        self.r = r

    def _check(self, state: MarketState) -> Signal | None:
        if not state.is_market_open():
            return None
        mso = state.minutes_since_open()
        if mso < self.time_window_start or mso > self.time_window_end:
            return None

        today = state.ts.date()
        prev_close = None
        today_bars = []
        for b in state.bars_5m:
            if b.ts.date() < today:
                prev_close = b.close
            elif b.ts.date() == today:
                today_bars.append(b)

        if prev_close is None or len(today_bars) < 3:
            return None

        today_open = today_bars[0].open
        gap = today_open - prev_close
        gap_pct = gap / prev_close

        if abs(gap_pct) < self.min_gap_pct or abs(gap_pct) > self.max_gap_pct:
            return None

        # Opening range — needed for rejection check
        or_band = state.opening_range(minutes=self.or_minutes)
        if or_band is None:
            return None
        or_low, or_high = or_band
        or_mid = (or_low + or_high) / 2

        last_bar = today_bars[-1]

        if gap > 0:
            # Gap up — fade by buying PE
            if self.rejection_required and last_bar.close > or_mid:
                return None
            direction = "bearish"
            opt_type = "PE"
        else:
            if self.rejection_required and last_bar.close < or_mid:
                return None
            direction = "bullish"
            opt_type = "CE"

        spot = state.spot
        target_spot = spot - gap * self.target_fill_pct
        stop_spot = spot + gap * self.stop_extension_pct
        target_pts = abs(target_spot - spot)

        # Strike + premium
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

        thesis = (
            f"Gap-fill fade {direction}: prev_close={prev_close:.0f}, today_open={today_open:.0f}, "
            f"gap={gap:+.0f} ({gap_pct*100:+.2f}%); "
            f"OR=[{or_low:.0f},{or_high:.0f}], last_close={last_bar.close:.0f} "
            f"{'below' if last_bar.close < or_mid else 'above'} OR mid -> "
            f"{'fading gap-up via PE' if direction=='bearish' else 'fading gap-down via CE'}; "
            f"target={target_spot:.0f} ({target_pts:.0f}pts spot / {target_premium_gain:.0f}pts premium); "
            f"stop={stop_spot:.0f}; hold <= {self.hold_minutes}m. "
            f"Premise: mean-reverting regime, gaps statistically fill 70-80%. "
            f"Avoid in trending regimes (election week, budget reaction days)."
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
                "prev_close": prev_close,
                "today_open": today_open,
                "gap": gap,
                "gap_pct": gap_pct,
                "or_low": or_low, "or_high": or_high, "or_mid": or_mid,
                "last_close": last_bar.close,
                "spot": spot, "atm_strike": atm, "iv": q.iv,
            },
            thesis=thesis[:1900],
        )
