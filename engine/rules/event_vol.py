"""
Event-aware volatility rules.

Two rules that exploit the systematic IV expand-then-crush cycle around scheduled
high-impact macro events (RBI MPC, US FOMC, Union Budget):

  PreEventVolBuy
    24-36h before the event, ATM IV often UNDER-prices the expected move.
    Fire a long-straddle ALERT to harvest the pre-event vol-expansion.

  PostEventIVCrush
    On the event day, AFTER the announcement, IV crushes 5-10 vol pts within
    minutes. Fire a short-straddle (or iron-fly with wings) ALERT to harvest
    the decay.

Both rules emit a single Signal with a `trigger_context.legs` list describing
the multi-leg structure — the paper-trade infra wires this as a single ALERT
(same pattern as the iron-condor scaffold). The actual leg fills happen
manually or via a downstream router; this rule only decides WHEN and WHAT.
"""
from __future__ import annotations

from datetime import datetime, time, timedelta

from ..signals import BaseRule, Signal
from ..state import MarketState, IST
from ..pricing import black_scholes
from ..events import events_between


LOT_SIZE = 75  # NIFTY lot size


def _atm_strike(spot: float, step: int) -> int:
    return int(round(spot / step) * step)


def _resolve_leg(state: MarketState, strike: int, opt_type: str,
                 r: float = 0.065):
    """Look up a CE/PE quote on the chain, return (strike, iv, premium) or None."""
    if state.chain is None:
        return None
    by_strike = state.chain.by_strike()
    q = by_strike.get(strike, {}).get(opt_type)
    if q is None or q.iv is None:
        # Try nearest strike that has IV for this side
        avail = [k for k in by_strike
                 if opt_type in by_strike[k]
                 and by_strike[k][opt_type].iv is not None]
        if not avail:
            return None
        strike = min(avail, key=lambda k: abs(k - state.spot))
        q = by_strike[strike][opt_type]
    expiry_close = datetime.combine(state.chain.expiry, time(15, 30), tzinfo=IST)
    t_years = max((expiry_close - state.ts).total_seconds() / (365 * 24 * 3600), 1e-5)
    g = black_scholes(state.spot, strike, t_years, q.iv, r=r, q=0.0,
                      option_type=opt_type)
    premium = g.price if g.price > 0 else (q.ltp or g.price)
    return strike, float(q.iv), float(premium), t_years


# ---------------------------------------------------------------------------
# Rule 1: Pre-event vol buy
# ---------------------------------------------------------------------------
class PreEventVolBuy(BaseRule):
    name = "pre_event_vol_buy"
    cooldown_minutes = 720   # 12h — one entry per pre-event window

    def __init__(
        self,
        min_hours_to_event: float = 24.0,
        max_hours_to_event: float = 36.0,
        time_window_start: int = 60,    # 10:00
        time_window_end: int = 315,     # 14:30
        iv_skip_threshold: float = 0.22,
        target_pct_gain: float = 0.40,
        stop_pct_loss: float = 0.30,
        strike_step: int = 50,
        r: float = 0.065,
    ):
        super().__init__()
        self.min_hours_to_event = min_hours_to_event
        self.max_hours_to_event = max_hours_to_event
        self.time_window_start = time_window_start
        self.time_window_end = time_window_end
        self.iv_skip_threshold = iv_skip_threshold
        self.target_pct_gain = target_pct_gain
        self.stop_pct_loss = stop_pct_loss
        self.strike_step = strike_step
        self.r = r

    def _check(self, state: MarketState) -> Signal | None:
        if not state.is_market_open():
            return None
        mso = state.minutes_since_open()
        if mso < self.time_window_start or mso > self.time_window_end:
            return None

        today = state.ts.astimezone(IST).date()
        # Look 2 days ahead — covers the 24-36h window with margin.
        upcoming = events_between(today - timedelta(days=1),
                                   today + timedelta(days=2),
                                   min_impact="high")
        if not upcoming:
            return None

        # Pick the next event strictly in the future whose distance lands in
        # [min_hours, max_hours] window.
        chosen = None
        hours_until = None
        for ev in upcoming:
            ev_dt = datetime.combine(ev.event_date, time(11, 0), tzinfo=IST)
            delta = (ev_dt - state.ts).total_seconds() / 3600.0
            if self.min_hours_to_event <= delta <= self.max_hours_to_event:
                chosen = ev
                hours_until = delta
                break
        if chosen is None:
            return None

        spot = state.spot
        if state.chain is None:
            return None
        atm = _atm_strike(spot, self.strike_step)

        ce = _resolve_leg(state, atm, "CE", r=self.r)
        pe = _resolve_leg(state, atm, "PE", r=self.r)
        if ce is None or pe is None:
            return None
        ce_strike, ce_iv, ce_prem, _t = ce
        pe_strike, pe_iv, pe_prem, _t = pe

        avg_iv = (ce_iv + pe_iv) / 2.0
        if avg_iv >= self.iv_skip_threshold:
            return None  # IV already expanded — don't chase

        total_debit_pts = ce_prem + pe_prem
        total_debit_inr = total_debit_pts * LOT_SIZE

        target_premium_gain = total_debit_pts * self.target_pct_gain
        stop_premium_loss = total_debit_pts * self.stop_pct_loss
        # stop_premium = the absolute remaining premium that triggers stop
        stop_premium = max(0.5, total_debit_pts - stop_premium_loss)
        target_premium_abs = total_debit_pts + target_premium_gain

        # Hold until event start (minutes), capped to a sane positive int
        hold_minutes = max(1, int(round(hours_until * 60.0)))

        breakeven_up = atm + total_debit_pts
        breakeven_down = atm - total_debit_pts

        legs = [
            {"action": "BUY", "strike": ce_strike, "option_type": "CE",
             "premium": round(ce_prem, 2), "iv": round(ce_iv, 4)},
            {"action": "BUY", "strike": pe_strike, "option_type": "PE",
             "premium": round(pe_prem, 2), "iv": round(pe_iv, 4)},
        ]

        thesis = (
            f"PRE-EVENT VOL BUY: {chosen.name} on {chosen.event_date} "
            f"(~{hours_until:.1f}h away). ATM IV={avg_iv*100:.1f}% below skip "
            f"threshold {self.iv_skip_threshold*100:.0f}% — IV likely to "
            f"expand 2-4 vol pts as event approaches. "
            f"BUY ATM straddle {atm} CE+PE, total debit {total_debit_pts:.0f} "
            f"pts (Rs {total_debit_inr:.0f} on {LOT_SIZE} lot). "
            f"Breakevens: {breakeven_down:.0f} / {breakeven_up:.0f}. "
            f"TARGET: +{self.target_pct_gain*100:.0f}% on combined premium "
            f"(=Rs {target_premium_abs*LOT_SIZE:.0f}). "
            f"STOP: -{self.stop_pct_loss*100:.0f}% on combined premium "
            f"(=Rs {stop_premium*LOT_SIZE:.0f}). "
            f"Hold until event start (~{hold_minutes} min)."
        )

        return Signal(
            rule_name=self.name,
            ts=state.ts,
            direction="neutral",
            action="ALERT_LONG_STRADDLE",
            strike=atm,
            option_type="CE",  # representative leg
            target_premium_gain=round(target_premium_gain, 2),
            target_spot=None,
            stop_spot=None,
            stop_premium=round(stop_premium, 2),
            hold_minutes=hold_minutes,
            prob_estimates={},
            trigger_context={
                "kind": "long_straddle",
                "event_name": chosen.name,
                "event_date": chosen.event_date.isoformat(),
                "hours_until_event": round(hours_until, 2),
                "legs": legs,
                "atm_strike": atm,
                "spot": spot,
                "current_iv": round(avg_iv, 4),
                "expected_iv_expansion_pts": "+2 to +4 vol pts",
                "total_debit_pts": round(total_debit_pts, 2),
                "total_debit_inr": round(total_debit_inr, 2),
                "breakeven_up": round(breakeven_up, 2),
                "breakeven_down": round(breakeven_down, 2),
                "target_premium_total": round(target_premium_abs, 2),
                "stop_premium_total": round(stop_premium, 2),
                "lot_size": LOT_SIZE,
            },
            thesis=thesis[:1900],
        )


# ---------------------------------------------------------------------------
# Rule 2: Post-event IV crush
# ---------------------------------------------------------------------------
class PostEventIVCrush(BaseRule):
    name = "post_event_iv_crush"
    cooldown_minutes = 1440   # once per event day

    def __init__(
        self,
        time_window_start: int = 195,   # 12:30
        time_window_end: int = 315,     # 14:30
        target_pct_decay: float = 0.30,
        stop_pct_loss: float = 1.0,
        add_wings: bool = True,
        wing_width_pts: int = 250,
        strike_step: int = 50,
        r: float = 0.065,
    ):
        super().__init__()
        self.time_window_start = time_window_start
        self.time_window_end = time_window_end
        self.target_pct_decay = target_pct_decay
        self.stop_pct_loss = stop_pct_loss
        self.add_wings = add_wings
        self.wing_width_pts = wing_width_pts
        self.strike_step = strike_step
        self.r = r

    def _check(self, state: MarketState) -> Signal | None:
        if not state.is_market_open():
            return None
        mso = state.minutes_since_open()
        if mso < self.time_window_start or mso > self.time_window_end:
            return None

        today = state.ts.astimezone(IST).date()
        # Check whether TODAY is an event day.
        todays_events = events_between(today - timedelta(days=1), today,
                                        min_impact="high")
        todays_events = [e for e in todays_events if e.event_date == today]
        if not todays_events:
            return None

        chosen = todays_events[0]
        # Nominal event time heuristic (best-effort).
        if "Budget" in chosen.name:
            nominal_time = time(11, 0)
        elif "FOMC" in chosen.name:
            nominal_time = time(9, 15)   # impact at next-day open
        else:  # RBI MPC default
            nominal_time = time(11, 45)
        event_dt = datetime.combine(chosen.event_date, nominal_time, tzinfo=IST)
        hours_since_event = (state.ts - event_dt).total_seconds() / 3600.0

        spot = state.spot
        if state.chain is None:
            return None
        atm = _atm_strike(spot, self.strike_step)

        ce = _resolve_leg(state, atm, "CE", r=self.r)
        pe = _resolve_leg(state, atm, "PE", r=self.r)
        if ce is None or pe is None:
            return None
        ce_strike, ce_iv, ce_prem, _t = ce
        pe_strike, pe_iv, pe_prem, _t = pe

        legs = [
            {"action": "SELL", "strike": ce_strike, "option_type": "CE",
             "premium": round(ce_prem, 2), "iv": round(ce_iv, 4)},
            {"action": "SELL", "strike": pe_strike, "option_type": "PE",
             "premium": round(pe_prem, 2), "iv": round(pe_iv, 4)},
        ]
        total_credit_pts = ce_prem + pe_prem

        action = "ALERT_SHORT_STRADDLE"
        kind = "short_straddle"
        if self.add_wings:
            up_wing_strike = atm + self.wing_width_pts
            dn_wing_strike = atm - self.wing_width_pts
            up_wing = _resolve_leg(state, up_wing_strike, "CE", r=self.r)
            dn_wing = _resolve_leg(state, dn_wing_strike, "PE", r=self.r)
            if up_wing is not None and dn_wing is not None:
                uw_strike, uw_iv, uw_prem, _ = up_wing
                dw_strike, dw_iv, dw_prem, _ = dn_wing
                legs.append({"action": "BUY", "strike": uw_strike,
                             "option_type": "CE",
                             "premium": round(uw_prem, 2),
                             "iv": round(uw_iv, 4)})
                legs.append({"action": "BUY", "strike": dw_strike,
                             "option_type": "PE",
                             "premium": round(dw_prem, 2),
                             "iv": round(dw_iv, 4)})
                total_credit_pts = (ce_prem + pe_prem) - (uw_prem + dw_prem)
                action = "ALERT_IRON_FLY"
                kind = "iron_fly"

        total_credit_inr = total_credit_pts * LOT_SIZE

        target_premium_gain = total_credit_pts * self.target_pct_decay
        stop_premium_loss = total_credit_pts * self.stop_pct_loss

        # For a short premium structure, exit when the combined premium decays
        # to (1 - target_pct_decay) × credit. Use stop_premium as the absolute
        # remaining premium that triggers the cut (entry_credit × (1 + stop_pct)).
        stop_premium = max(0.5, total_credit_pts * (1.0 + self.stop_pct_loss))

        breakeven_up = atm + total_credit_pts
        breakeven_down = atm - total_credit_pts

        # Hold until ~90 min before expiry close.
        expiry_close = datetime.combine(state.chain.expiry, time(15, 30),
                                         tzinfo=IST)
        mins_to_expiry = max(1,
                              int((expiry_close - state.ts).total_seconds() / 60))
        hold_minutes = max(1, mins_to_expiry - 90)

        thesis = (
            f"POST-EVENT IV CRUSH: {chosen.name} on {chosen.event_date} "
            f"(~{hours_since_event:.1f}h after nominal event time). "
            f"SELL ATM {kind.replace('_', ' ')} {atm}"
            f"{' with ' + str(self.wing_width_pts) + '-pt wings' if kind == 'iron_fly' else ''}. "
            f"Credit collected: {total_credit_pts:.0f} pts "
            f"(Rs {total_credit_inr:.0f} on {LOT_SIZE} lot). "
            f"Breakevens: {breakeven_down:.0f} / {breakeven_up:.0f}. "
            f"TARGET: capture {self.target_pct_decay*100:.0f}% of credit as "
            f"IV decays (=Rs {target_premium_gain*LOT_SIZE:.0f}). "
            f"STOP: cut if combined premium reaches "
            f"{stop_premium:.0f} pts (={self.stop_pct_loss:.1f}x credit). "
            f"Hold up to {hold_minutes} min (≈expiry − 90m)."
        )

        return Signal(
            rule_name=self.name,
            ts=state.ts,
            direction="neutral",
            action=action,
            strike=atm,
            option_type="CE",
            target_premium_gain=round(target_premium_gain, 2),
            target_spot=None,
            stop_spot=None,
            stop_premium=round(stop_premium, 2),
            hold_minutes=hold_minutes,
            prob_estimates={},
            trigger_context={
                "kind": kind,
                "event_name": chosen.name,
                "event_date": chosen.event_date.isoformat(),
                "hours_since_event": round(hours_since_event, 2),
                "legs": legs,
                "atm_strike": atm,
                "spot": spot,
                "total_credit_pts": round(total_credit_pts, 2),
                "total_credit_inr": round(total_credit_inr, 2),
                "breakeven_up": round(breakeven_up, 2),
                "breakeven_down": round(breakeven_down, 2),
                "wing_width_pts": self.wing_width_pts if kind == "iron_fly" else None,
                "stop_premium_total": round(stop_premium, 2),
                "target_premium_gain": round(target_premium_gain, 2),
                "lot_size": LOT_SIZE,
            },
            thesis=thesis[:1900],
        )
