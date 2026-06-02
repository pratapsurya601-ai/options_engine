"""
IV-Regime Gated rule — adaptive premium buyer/seller based on ATM IV level.

Concept:
  - Low IV (cheap premium)  -> BUY ATM CE (with bullish-trend filter)
  - High IV (rich premium)  -> SELL premium via ATM iron fly (short straddle + wings)
  - Mid IV                  -> stand aside

Single rule, three modes, regime-aware.
"""
from __future__ import annotations

from datetime import datetime, time

from ..signals import BaseRule, Signal
from ..state import MarketState
from ..pricing import black_scholes
from ..indicators import ema, rsi


class IVRegime(BaseRule):
    name = "iv_regime"
    cooldown_minutes = 60

    def __init__(
        self,
        low_iv_threshold: float = 0.13,
        high_iv_threshold: float = 0.18,
        ema_fast: int = 20,
        ema_slow: int = 50,
        rsi_period: int = 14,
        rsi_min: float = 40.0,
        rsi_max: float = 65.0,
        low_iv_target_pts: float = 20.0,
        low_iv_stop_pts: float = 12.0,
        hold_minutes_low_iv: int = 45,
        high_iv_threshold_wing_width_pts: int = 100,
        min_credit_pts: float = 4.0,
        target_credit_pct_remaining: float = 0.25,
        hold_minutes_high_iv: int = 240,
        strike_step: int = 50,
        time_window_start: int = 30,
        time_window_end: int = 270,
        r: float = 0.065,
    ):
        super().__init__()
        self.low_iv_threshold = low_iv_threshold
        self.high_iv_threshold = high_iv_threshold
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.rsi_period = rsi_period
        self.rsi_min = rsi_min
        self.rsi_max = rsi_max
        self.low_iv_target_pts = low_iv_target_pts
        self.low_iv_stop_pts = low_iv_stop_pts
        self.hold_minutes_low_iv = hold_minutes_low_iv
        self.high_iv_threshold_wing_width_pts = high_iv_threshold_wing_width_pts
        self.min_credit_pts = min_credit_pts
        self.target_credit_pct_remaining = target_credit_pct_remaining
        self.hold_minutes_high_iv = hold_minutes_high_iv
        self.strike_step = strike_step
        self.time_window_start = time_window_start
        self.time_window_end = time_window_end
        self.r = r

    # ------------------------------------------------------------------
    def _atm_iv(self, by_strike: dict, spot: float):
        atm_base = round(spot / self.strike_step) * self.strike_step
        for k_try in (atm_base, atm_base + self.strike_step, atm_base - self.strike_step):
            for ot in ("CE", "PE"):
                q = by_strike.get(k_try, {}).get(ot)
                if q is not None and q.iv is not None and q.iv > 0:
                    return atm_base, q.iv
        return atm_base, None

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
        mso = state.minutes_since_open()
        if mso < self.time_window_start or mso > self.time_window_end:
            return None
        if state.chain is None:
            return None

        bars = state.bars_5m
        needed = max(self.ema_slow + 2, self.rsi_period + 2)
        if len(bars) < needed:
            return None

        spot = state.spot
        by_strike = state.chain.by_strike()
        atm, atm_iv = self._atm_iv(by_strike, spot)
        if atm_iv is None:
            return None

        # Regime gating
        if self.low_iv_threshold <= atm_iv <= self.high_iv_threshold:
            return None  # mid -> stand aside

        from ..chain import IST as CHAIN_IST
        expiry = state.chain.expiry
        expiry_close = datetime.combine(expiry, time(15, 30), tzinfo=CHAIN_IST)
        t_years = max((expiry_close - state.ts).total_seconds() / (365 * 24 * 3600), 1e-5)

        if atm_iv < self.low_iv_threshold:
            return self._low_iv_buy_ce(state, by_strike, spot, atm, atm_iv, t_years, bars)

        if atm_iv > self.high_iv_threshold:
            return self._high_iv_sell_fly(state, by_strike, spot, atm, atm_iv, t_years)

        return None

    # ------------------------------------------------------------------
    def _low_iv_buy_ce(self, state, by_strike, spot, atm, atm_iv, t_years, bars):
        closes = [b.close for b in bars]
        ef = ema(closes, self.ema_fast)
        es = ema(closes, self.ema_slow)
        if ef[-1] is None or es[-1] is None:
            return None
        if not (ef[-1] > es[-1]):
            return None

        rsi_vals = rsi(closes, self.rsi_period)
        rsi_now = rsi_vals[-1]
        if rsi_now is None:
            return None
        if not (self.rsi_min <= rsi_now <= self.rsi_max):
            return None

        atm_k, q = self._pick_quote(by_strike, atm, "CE")
        if q is None:
            return None
        iv_used = q.iv or atm_iv
        g = black_scholes(spot, atm_k, t_years, iv_used, r=self.r, q=0.0, option_type="CE")
        entry_premium = g.price

        target_gain = self.low_iv_target_pts
        stop_loss_pts = self.low_iv_stop_pts
        delta_floor = max(abs(g.delta), 0.05)
        target_spot_move = target_gain / delta_floor
        stop_spot_move = stop_loss_pts / delta_floor
        target_premium = entry_premium + target_gain
        stop_premium = max(0.5, entry_premium - stop_loss_pts)
        target_spot = spot + target_spot_move
        stop_spot = spot - stop_spot_move

        rr = target_gain / max(stop_loss_pts, 0.1)
        thesis = (
            f"IV-Regime LOW-IV BUY: ATM IV {atm_iv*100:.1f}% < {self.low_iv_threshold*100:.1f}% "
            f"(cheap premium). EMA{self.ema_fast}>{self.ema_slow}, RSI={rsi_now:.1f}. "
            f"BUY CE {atm_k} @ ~Rs {entry_premium:.0f}. "
            f"TARGET +{target_gain:.0f}pts (Rs {target_premium:.0f}) STOP -{stop_loss_pts:.0f}pts. "
            f"Hold {self.hold_minutes_low_iv}m. R/R={rr:.2f}."
        )
        return Signal(
            rule_name=self.name,
            ts=state.ts,
            direction="bullish",
            action="BUY_CE",
            strike=atm_k,
            option_type="CE",
            target_premium_gain=round(target_gain, 1),
            target_spot=round(target_spot, 1),
            stop_spot=round(stop_spot, 1),
            stop_premium=round(stop_premium, 2),
            hold_minutes=self.hold_minutes_low_iv,
            prob_estimates={},
            trigger_context={
                "kind": "low_iv_buy_ce",
                "regime": "low_iv",
                "atm_iv": round(atm_iv, 4),
                "spot": spot,
                "atm_strike": atm_k,
                "entry_premium": round(entry_premium, 2),
                "ema_fast": ef[-1], "ema_slow": es[-1],
                "rsi": round(rsi_now, 2),
                "delta": round(g.delta, 3),
                "risk_reward": round(rr, 2),
            },
            thesis=thesis[:1900],
        )

    # ------------------------------------------------------------------
    def _high_iv_sell_fly(self, state, by_strike, spot, atm, atm_iv, t_years):
        w = self.high_iv_threshold_wing_width_pts
        body_ce_k, body_ce_q = self._pick_quote(by_strike, atm, "CE")
        body_pe_k, body_pe_q = self._pick_quote(by_strike, atm, "PE")
        wing_ce_k, wing_ce_q = self._pick_quote(by_strike, atm + w, "CE")
        wing_pe_k, wing_pe_q = self._pick_quote(by_strike, atm - w, "PE")

        if None in (body_ce_q, body_pe_q, wing_ce_q, wing_pe_q):
            return None
        if not (wing_pe_k < body_pe_k <= body_ce_k < wing_ce_k):
            return None

        def _prem(q, K, ot):
            ltp = float(q.ltp) if q.ltp else 0.0
            bs = black_scholes(spot, K, t_years, q.iv, r=self.r, q=0.0, option_type=ot).price
            return ltp if ltp > 0 else bs

        prem_body_ce = _prem(body_ce_q, body_ce_k, "CE")
        prem_body_pe = _prem(body_pe_q, body_pe_k, "PE")
        prem_wing_ce = _prem(wing_ce_q, wing_ce_k, "CE")
        prem_wing_pe = _prem(wing_pe_q, wing_pe_k, "PE")

        net_credit = (prem_body_ce + prem_body_pe) - (prem_wing_ce + prem_wing_pe)
        if net_credit < self.min_credit_pts:
            return None

        wing_width = min(body_ce_k - wing_pe_k, wing_ce_k - body_ce_k,
                         wing_ce_k - body_pe_k, body_pe_k - wing_pe_k)
        lot_size = 75
        max_profit_inr = net_credit * lot_size
        max_loss_inr = (wing_width - net_credit) * lot_size

        target_premium_gain = net_credit * (1.0 - self.target_credit_pct_remaining)
        stop_premium = -net_credit  # close cost = 2x credit -> lose 1x credit

        legs = [
            {"action": "SELL", "strike": body_ce_k, "option_type": "CE",
             "premium": round(prem_body_ce, 2), "iv": round(body_ce_q.iv, 4)},
            {"action": "SELL", "strike": body_pe_k, "option_type": "PE",
             "premium": round(prem_body_pe, 2), "iv": round(body_pe_q.iv, 4)},
            {"action": "BUY", "strike": wing_ce_k, "option_type": "CE",
             "premium": round(prem_wing_ce, 2), "iv": round(wing_ce_q.iv, 4)},
            {"action": "BUY", "strike": wing_pe_k, "option_type": "PE",
             "premium": round(prem_wing_pe, 2), "iv": round(wing_pe_q.iv, 4)},
        ]

        thesis = (
            f"IV-Regime HIGH-IV SELL: ATM IV {atm_iv*100:.1f}% > {self.high_iv_threshold*100:.1f}% "
            f"(rich premium). Iron fly {wing_pe_k}/{body_pe_k}-{body_ce_k}/{wing_ce_k} "
            f"on NIFTY @ {spot:.0f}. Credit Rs {net_credit:.1f}/share "
            f"(Rs {max_profit_inr:.0f}/lot). Max loss Rs {max_loss_inr:.0f}/lot. "
            f"BOOK at {self.target_credit_pct_remaining*100:.0f}% credit remaining. "
            f"Hold up to {self.hold_minutes_high_iv}m."
        )

        return Signal(
            rule_name=self.name,
            ts=state.ts,
            direction="neutral",
            action="ALERT_IRON_FLY",
            strike=atm,
            option_type="CE",
            target_premium_gain=round(target_premium_gain, 2),
            target_spot=None,
            stop_spot=None,
            stop_premium=round(stop_premium, 2),
            hold_minutes=self.hold_minutes_high_iv,
            prob_estimates={},
            trigger_context={
                "kind": "iron_fly",
                "regime": "high_iv",
                "legs": legs,
                "net_credit": round(net_credit, 2),
                "max_profit_inr": round(max_profit_inr, 2),
                "max_loss_inr": round(max_loss_inr, 2),
                "target_credit_pct_remaining": self.target_credit_pct_remaining,
                "spot": spot,
                "atm_iv": round(atm_iv, 4),
                "atm_strike": atm,
                "wing_width_pts": wing_width,
            },
            thesis=thesis[:1900],
        )
