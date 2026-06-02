"""
Bull Put Credit Spread — defined-risk premium-selling rule.

Premise:
  Bullish bias on NIFTY -> sell an OTM put spread for net credit. Premium
  decays in our favor. Defined risk via long put wing below the short.

Strike selection:
  1sigma move = spot * ATM_IV * sqrt(DTE/365)
  SHORT PUT  K = round_to_step(spot - sigma_short_mult * 1sigma)
  LONG  PUT  K = SHORT PUT - wing_width_pts

Entry (all must hold):
  - weekday in `weekday_fire` (default Mon-Thu)
  - within [time_window_start, time_window_end] minutes since open
  - DTE in [min_dte_days, max_dte_days]
  - EMA(fast) > EMA(slow) on the 5-min closes (bullish trend)
  - RSI(rsi_period) in [rsi_min, rsi_max]
  - not spanning a scheduled high-impact event (skip_on_event)
  - cooldown 1440 minutes (one fire per session)
  - all 4 strikes have valid quotes (LTP>0, IV>0); nearest-strike fallback
  - net credit >= min_credit_pts

Output: single Signal with both legs in trigger_context["legs"], plus
  sizing/risk metadata. Hold until ~90 min before expiry close.
"""
from __future__ import annotations

import math
from datetime import datetime, time

from ..signals import BaseRule, Signal
from ..state import MarketState
from ..pricing import black_scholes
from ..probability import prob_above
from ..events import assess
from ..indicators import ema, rsi


class BullPutSpread(BaseRule):
    name = "bull_put_spread"
    cooldown_minutes = 1440

    def __init__(
        self,
        weekday_fire: list[int] | None = None,
        time_window_start: int = 105,
        time_window_end: int = 285,
        min_dte_days: int = 1,
        max_dte_days: int = 7,
        ema_fast: int = 20,
        ema_slow: int = 50,
        rsi_period: int = 14,
        rsi_min: float = 45.0,
        rsi_max: float = 70.0,
        sigma_short_mult: float = 1.0,
        wing_width_pts: int = 100,
        strike_step: int = 50,
        min_credit_pts: float = 2.0,
        target_credit_pct_remaining: float = 0.25,
        skip_on_event: bool = True,
        r: float = 0.065,
    ):
        super().__init__()
        self.weekday_fire = weekday_fire if weekday_fire is not None else [0, 1, 2, 3]
        self.time_window_start = time_window_start
        self.time_window_end = time_window_end
        self.min_dte_days = min_dte_days
        self.max_dte_days = max_dte_days
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.rsi_period = rsi_period
        self.rsi_min = rsi_min
        self.rsi_max = rsi_max
        self.sigma_short_mult = sigma_short_mult
        self.wing_width_pts = wing_width_pts
        self.strike_step = strike_step
        self.min_credit_pts = min_credit_pts
        self.target_credit_pct_remaining = target_credit_pct_remaining
        self.skip_on_event = skip_on_event
        self.r = r

    # ------------------------------------------------------------------
    def _round_strike(self, x: float) -> int:
        return int(round(x / self.strike_step) * self.strike_step)

    def _pick_quote(self, by_strike: dict, strike: int, opt_type: str):
        q = by_strike.get(strike, {}).get(opt_type)
        if q is not None and q.iv is not None and q.ltp:
            return strike, q
        avail = [
            k for k in by_strike
            if opt_type in by_strike[k]
            and by_strike[k][opt_type].iv is not None
            and by_strike[k][opt_type].ltp
        ]
        if not avail:
            return None, None
        k2 = min(avail, key=lambda k: abs(k - strike))
        return k2, by_strike[k2][opt_type]

    # ------------------------------------------------------------------
    def _check(self, state: MarketState) -> Signal | None:
        if not state.is_market_open():
            return None

        ist_now = state.ts
        weekday = ist_now.weekday()
        if weekday not in self.weekday_fire:
            return None

        mso = state.minutes_since_open()
        if mso < self.time_window_start or mso > self.time_window_end:
            return None

        if state.chain is None:
            return None

        today = state.ts.date()
        expiry = state.chain.expiry
        dte_days = (expiry - today).days
        if dte_days < self.min_dte_days or dte_days > self.max_dte_days:
            return None

        if self.skip_on_event:
            risk = assess(today, expiry, iv_rank=None)
            if risk.spans_event:
                return None

        # Trend / momentum gates via bar history
        bars = state.bars_5m or []
        closes = [b.close for b in bars]
        if len(closes) < max(self.ema_slow, self.rsi_period) + 2:
            return None
        ef = ema(closes, self.ema_fast)
        es = ema(closes, self.ema_slow)
        if ef[-1] is None or es[-1] is None:
            return None
        if not (ef[-1] > es[-1]):
            return None
        rv = rsi(closes, self.rsi_period)
        if rv[-1] is None:
            return None
        if not (self.rsi_min <= rv[-1] <= self.rsi_max):
            return None

        spot = state.spot
        by_strike = state.chain.by_strike()

        # ATM IV via nearest valid quote around spot
        atm_iv = None
        atm_base = self._round_strike(spot)
        for k_try in [atm_base, atm_base + self.strike_step, atm_base - self.strike_step]:
            for ot in ("PE", "CE"):
                q = by_strike.get(k_try, {}).get(ot)
                if q is not None and q.iv is not None and q.iv > 0:
                    atm_iv = q.iv
                    break
            if atm_iv is not None:
                break
        if atm_iv is None:
            return None

        from ..chain import IST as CHAIN_IST
        expiry_close = datetime.combine(expiry, time(15, 30), tzinfo=CHAIN_IST)
        t_years = max((expiry_close - state.ts).total_seconds() / (365 * 24 * 3600), 1e-5)
        one_sigma = spot * atm_iv * math.sqrt(t_years)

        short_put_target = self._round_strike(spot - self.sigma_short_mult * one_sigma)
        long_put_target = short_put_target - self.wing_width_pts

        short_put_k, short_put_q = self._pick_quote(by_strike, short_put_target, "PE")
        long_put_k, long_put_q = self._pick_quote(by_strike, long_put_target, "PE")

        if short_put_q is None or long_put_q is None:
            return None
        if not (long_put_k < short_put_k):
            return None

        def _prem(q, K):
            ltp = float(q.ltp) if q.ltp else 0.0
            bs = black_scholes(spot, K, t_years, q.iv, r=self.r, q=0.0, option_type="PE").price
            return ltp if ltp > 0 else bs

        prem_short = _prem(short_put_q, short_put_k)
        prem_long = _prem(long_put_q, long_put_k)
        net_credit = prem_short - prem_long
        if net_credit < self.min_credit_pts:
            return None

        actual_wing_width = short_put_k - long_put_k
        lot_size = 75
        max_profit_inr = net_credit * lot_size
        max_loss_inr = (actual_wing_width - net_credit) * lot_size

        breakeven = short_put_k - net_credit
        # Probability of profit = P(spot at expiry > breakeven)
        p_profit = prob_above(spot, breakeven, t_years, atm_iv, drift=0.0)

        minutes_to_close = (expiry_close - state.ts).total_seconds() / 60.0
        hold_minutes = max(int(minutes_to_close - 90), 30)

        target_premium_gain = net_credit * 0.75
        stop_premium = net_credit * -1.0

        legs = [
            {"action": "SELL", "strike": short_put_k, "option_type": "PE",
             "premium": round(prem_short, 2), "iv": round(short_put_q.iv, 4)},
            {"action": "BUY", "strike": long_put_k, "option_type": "PE",
             "premium": round(prem_long, 2), "iv": round(long_put_q.iv, 4)},
        ]

        thesis = (
            f"Bull put spread {long_put_k}/{short_put_k} on NIFTY @ {spot:.0f}, "
            f"ATM IV {atm_iv*100:.1f}%, DTE={dte_days}d. "
            f"EMA{self.ema_fast}>EMA{self.ema_slow} + RSI {rv[-1]:.0f} (bullish). "
            f"Credit Rs {net_credit:.1f}/share (Rs {max_profit_inr:.0f}/lot). "
            f"Breakeven {breakeven:.0f}, P(profit) {p_profit*100:.0f}% lognormal. "
            f"Max loss Rs {max_loss_inr:.0f}/lot. "
            f"BOOK at 25% credit remaining (~Rs {(net_credit*0.25)*lot_size:.0f}); "
            f"KILL if premium doubles. Defined-risk bullish theta play."
        )

        return Signal(
            rule_name=self.name,
            ts=state.ts,
            direction="bullish",
            action="ALERT_BULL_PUT",
            strike=short_put_k,
            option_type="PE",
            target_premium_gain=round(target_premium_gain, 2),
            target_spot=None,
            stop_spot=None,
            stop_premium=round(stop_premium, 2),
            hold_minutes=hold_minutes,
            prob_estimates={"p_profit_lognormal": round(p_profit, 4)},
            trigger_context={
                "kind": "bull_put_spread",
                "legs": legs,
                "net_credit": round(net_credit, 2),
                "max_profit_inr": round(max_profit_inr, 2),
                "max_loss_inr": round(max_loss_inr, 2),
                "target_credit_pct_remaining": self.target_credit_pct_remaining,
                "breakeven": round(breakeven, 2),
                "prob_profit": round(p_profit, 4),
                "spot": spot,
                "iv": atm_iv,
                "dte_days": dte_days,
                "expiry": expiry.isoformat(),
                "1_sigma_move": round(one_sigma, 2),
                "wing_width_pts": actual_wing_width,
                "weekday": weekday,
                "rsi": round(rv[-1], 2),
                "ema_fast": round(ef[-1], 2),
                "ema_slow": round(es[-1], 2),
            },
            thesis=thesis[:1900],
        )
