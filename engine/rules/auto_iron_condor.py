"""
Auto Iron Condor rule — fires defined-risk weekly iron condors near expiry.

Why this rule:
  - Theta decay is steepest in the final 1-3 days to expiry.
  - A 1-sigma-short, fixed-wing condor monetizes time decay while capping risk.
  - Designed to fire AT MOST ONCE per session (cooldown 1440 min) on a
    chosen weekday inside a quiet midday window (11:00-14:00 by default).

Entry conditions (all must hold):
  1. Day-of-week in `weekday_fire` (default Tue=1, Wed=2)
  2. Session time within [time_window_start, time_window_end] minutes since open
  3. Days-to-expiry in [min_dte_days, max_dte_days]
  4. ATM IV available from chain
  5. NOT spanning a scheduled high-impact event (skip_on_event=True)
  6. Cooldown elapsed (1 fire per session)

Strike selection:
  - 1-sigma move = spot * ATM_IV * sqrt(DTE/365)
  - Short PE = round_to_step(spot - sigma_short_mult * 1sigma)
  - Short CE = round_to_step(spot + sigma_short_mult * 1sigma)
  - Wings = shorts +/- wing_width_pts
  - All 4 strikes must have liquid quotes (LTP>0, IV not None); else fall back
    to nearest available strike with a valid quote.

Signal output:
  Single-leg infra in paper.py can't model 4-leg structures, so this rule
  emits an ALERT_IRON_CONDOR signal carrying the full condor in
  `trigger_context` for human execution.
"""
from __future__ import annotations

import math
from datetime import datetime, time, timedelta

from ..signals import BaseRule, Signal
from ..state import MarketState
from ..pricing import black_scholes
from ..probability import prob_between
from ..events import assess


class AutoIronCondor(BaseRule):
    name = "auto_iron_condor"
    cooldown_minutes = 1440   # one fire per session max

    def __init__(
        self,
        weekday_fire: list[int] | None = None,
        time_window_start: int = 105,    # 11:00
        time_window_end: int = 285,      # 14:00
        min_dte_days: int = 1,
        max_dte_days: int = 3,
        sigma_short_mult: float = 1.0,
        wing_width_pts: int = 200,
        strike_step: int = 50,
        target_credit_pct_remaining: float = 0.25,
        min_credit_pts: float = 8.0,
        skip_on_event: bool = True,
        r: float = 0.065,
    ):
        super().__init__()
        self.weekday_fire = weekday_fire if weekday_fire is not None else [1, 2]
        self.time_window_start = time_window_start
        self.time_window_end = time_window_end
        self.min_dte_days = min_dte_days
        self.max_dte_days = max_dte_days
        self.sigma_short_mult = sigma_short_mult
        self.wing_width_pts = wing_width_pts
        self.strike_step = strike_step
        self.target_credit_pct_remaining = target_credit_pct_remaining
        self.min_credit_pts = min_credit_pts
        self.skip_on_event = skip_on_event
        self.r = r

    # ------------------------------------------------------------------
    def _round_strike(self, x: float) -> int:
        return int(round(x / self.strike_step) * self.strike_step)

    def _pick_quote(self, by_strike: dict, strike: int, opt_type: str):
        """Return (strike, quote) preferring `strike`, falling back to nearest
        available strike with valid IV+LTP for the requested option type."""
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

        # Weekday gate
        ist_now = state.ts.astimezone(state.ts.tzinfo) if state.ts.tzinfo else state.ts
        weekday = ist_now.weekday()
        if weekday not in self.weekday_fire:
            return None

        # Time-of-day gate
        mso = state.minutes_since_open()
        if mso < self.time_window_start or mso > self.time_window_end:
            return None

        if state.chain is None:
            return None

        # DTE gate
        today = state.ts.date()
        expiry = state.chain.expiry
        dte_days = (expiry - today).days
        if dte_days < self.min_dte_days or dte_days > self.max_dte_days:
            return None

        # Event gate
        if self.skip_on_event:
            risk = assess(today, expiry, iv_rank=None)
            if risk.spans_event:
                return None

        spot = state.spot
        by_strike = state.chain.by_strike()

        # ATM IV — find quote nearest spot with valid IV (try CE then PE)
        atm_iv = None
        atm_base = self._round_strike(spot)
        for k_try in [atm_base, atm_base + self.strike_step, atm_base - self.strike_step]:
            for ot in ("CE", "PE"):
                q = by_strike.get(k_try, {}).get(ot)
                if q is not None and q.iv is not None and q.iv > 0:
                    atm_iv = q.iv
                    break
            if atm_iv is not None:
                break
        if atm_iv is None:
            return None

        # 1-sigma move using fractional DTE
        from ..chain import IST as CHAIN_IST
        expiry_close = datetime.combine(expiry, time(15, 30), tzinfo=CHAIN_IST)
        t_years = max((expiry_close - state.ts).total_seconds() / (365 * 24 * 3600), 1e-5)
        one_sigma = spot * atm_iv * math.sqrt(t_years)

        short_pe_target = self._round_strike(spot - self.sigma_short_mult * one_sigma)
        short_ce_target = self._round_strike(spot + self.sigma_short_mult * one_sigma)
        wing_pe_target = short_pe_target - self.wing_width_pts
        wing_ce_target = short_ce_target + self.wing_width_pts

        short_pe_k, short_pe_q = self._pick_quote(by_strike, short_pe_target, "PE")
        short_ce_k, short_ce_q = self._pick_quote(by_strike, short_ce_target, "CE")
        wing_pe_k, wing_pe_q = self._pick_quote(by_strike, wing_pe_target, "PE")
        wing_ce_k, wing_ce_q = self._pick_quote(by_strike, wing_ce_target, "CE")

        if None in (short_pe_q, short_ce_q, wing_pe_q, wing_ce_q):
            return None

        # Ensure structural ordering after fallback substitution
        if not (wing_pe_k < short_pe_k < short_ce_k < wing_ce_k):
            return None

        # Premiums — prefer chain LTP, cross-check vs BS as sanity floor
        def _prem(q, K, ot):
            ltp = float(q.ltp) if q.ltp else 0.0
            bs = black_scholes(spot, K, t_years, q.iv, r=self.r, q=0.0, option_type=ot).price
            return ltp if ltp > 0 else bs

        prem_short_pe = _prem(short_pe_q, short_pe_k, "PE")
        prem_short_ce = _prem(short_ce_q, short_ce_k, "CE")
        prem_wing_pe = _prem(wing_pe_q, wing_pe_k, "PE")
        prem_wing_ce = _prem(wing_ce_q, wing_ce_k, "CE")

        net_credit = (prem_short_pe + prem_short_ce) - (prem_wing_pe + prem_wing_ce)
        if net_credit < self.min_credit_pts:
            return None

        actual_wing_width = min(
            short_pe_k - wing_pe_k,
            wing_ce_k - short_ce_k,
        )

        lot_size = 75
        max_profit_inr = net_credit * lot_size
        max_loss_inr = (actual_wing_width - net_credit) * lot_size

        be_low = short_pe_k - net_credit
        be_high = short_ce_k + net_credit
        p_profit = prob_between(spot, be_low, be_high, t_years, atm_iv, drift=0.0)

        # Hold until 90 min before expiry close
        minutes_to_close = (expiry_close - state.ts).total_seconds() / 60.0
        hold_minutes = max(int(minutes_to_close - 90), 30)

        target_premium_gain = net_credit * 0.75   # book at 25% remaining
        stop_premium = net_credit * -1.0           # kill if loses 1x credit

        legs = [
            {"action": "SELL", "strike": short_pe_k, "option_type": "PE",
             "premium": round(prem_short_pe, 2), "iv": round(short_pe_q.iv, 4)},
            {"action": "BUY", "strike": wing_pe_k, "option_type": "PE",
             "premium": round(prem_wing_pe, 2), "iv": round(wing_pe_q.iv, 4)},
            {"action": "SELL", "strike": short_ce_k, "option_type": "CE",
             "premium": round(prem_short_ce, 2), "iv": round(short_ce_q.iv, 4)},
            {"action": "BUY", "strike": wing_ce_k, "option_type": "CE",
             "premium": round(prem_wing_ce, 2), "iv": round(wing_ce_q.iv, 4)},
        ]

        thesis = (
            f"Auto iron condor {wing_pe_k}/{short_pe_k}/{short_ce_k}/{wing_ce_k} "
            f"on NIFTY @ {spot:.0f}, ATM IV {atm_iv*100:.1f}%, DTE={dte_days}d. "
            f"Credit collected ~Rs {net_credit:.1f}/share (Rs {max_profit_inr:.0f}/lot). "
            f"Breakevens {be_low:.0f} - {be_high:.0f}, P(profit) {p_profit*100:.0f}% "
            f"under lognormal. Max loss Rs {max_loss_inr:.0f}/lot. "
            f"BOOK at 25% credit remaining (~Rs {(net_credit*0.25)*lot_size:.0f}); "
            f"KILL if premium doubles against us. Defined-risk theta play."
        )

        return Signal(
            rule_name=self.name,
            ts=state.ts,
            direction="neutral",
            action="ALERT_IRON_CONDOR",
            strike=short_pe_k,                # representative
            option_type="PE",                  # representative
            target_premium_gain=round(target_premium_gain, 2),
            target_spot=None,
            stop_spot=None,
            stop_premium=round(stop_premium, 2),
            hold_minutes=hold_minutes,
            prob_estimates={"p_profit_lognormal": round(p_profit, 4)},
            trigger_context={
                "kind": "iron_condor",
                "legs": legs,
                "net_credit": round(net_credit, 2),
                "max_profit_inr": round(max_profit_inr, 2),
                "max_loss_inr": round(max_loss_inr, 2),
                "target_credit_pct_remaining": self.target_credit_pct_remaining,
                "breakeven_low": round(be_low, 2),
                "breakeven_high": round(be_high, 2),
                "prob_profit": round(p_profit, 4),
                "spot": spot,
                "iv": atm_iv,
                "dte_days": dte_days,
                "expiry": expiry.isoformat(),
                "1_sigma_move": round(one_sigma, 2),
                "wing_width_pts": actual_wing_width,
                "weekday": weekday,
            },
            thesis=thesis[:1900],
        )
