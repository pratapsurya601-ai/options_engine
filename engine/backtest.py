"""
ORB backtest — replays N days of historical NIFTY 5-min bars through the
opening-range-breakout rule, simulates option entries/exits at Black-Scholes
prices, and reports performance.

HONEST CAVEATS (read before trusting numbers):
  1. Option premiums are synthesized via BS at a constant IV. Real chains have
     IV smile, intraday vol moves, and bid-ask spreads. Backtest premiums are
     systematically smoother than reality.
  2. No slippage modeled. Real fills cost 1-3 pts per leg on naked options.
  3. Stops are simulated at the stop SPOT, with premium computed at that spot.
     Real stop fills can be 2-5 pts worse on fast moves.
  4. Regime risk: 60-day backtest captures current regime only. Performance
     in different volatility/trend regimes is unknowable from this.

Run:
  python -m engine.backtest --rule orb --days 60 --iv 0.13
  python -m engine.backtest --rule orb --days 30 --iv 0.13 --csv out.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, date, time, timedelta, timezone
from pathlib import Path

from .pricing import black_scholes
from .state import MarketState, Bar
from .manual_chain import synthetic_chain
from .rules.orb import OpeningRangeBreakout


IST = timezone(timedelta(hours=5, minutes=30))


@dataclass
class MultiLegBTTrade:
    day: date
    entry_ts: datetime
    rule_name: str
    structure: str
    legs: list[dict]
    entry_spot: float
    entry_net_credit: float
    exit_ts: datetime
    exit_spot: float
    exit_net_credit: float
    exit_reason: str
    pnl_per_lot: float
    bars_held: int


@dataclass
class BTTrade:
    day: date
    entry_ts: datetime
    direction: str
    option_type: str
    strike: int
    entry_spot: float
    entry_premium: float
    target_premium: float
    stop_spot: float
    stop_premium: float
    exit_ts: datetime
    exit_spot: float
    exit_premium: float
    exit_reason: str
    pnl_per_lot: float
    bars_held: int


def _bars_from_raw(raw: list[dict]) -> list[Bar]:
    out = []
    for b in raw:
        ts = b["date"]
        if hasattr(ts, "astimezone"):
            ts = ts.astimezone(IST)
        out.append(Bar(
            ts=ts, open=float(b["open"]), high=float(b["high"]),
            low=float(b["low"]), close=float(b["close"]),
            volume=int(b["volume"]),
        ))
    return out


def _group_by_day(bars: list[Bar]) -> dict[date, list[Bar]]:
    by_day: dict[date, list[Bar]] = defaultdict(list)
    for b in bars:
        by_day[b.ts.date()].append(b)
    # Filter to bars in market hours only
    for d in list(by_day.keys()):
        in_hours = [b for b in by_day[d] if time(9, 15) <= b.ts.time() <= time(15, 30)]
        if len(in_hours) < 10:   # partial day — skip (low threshold to support HTF intervals)
            del by_day[d]
        else:
            by_day[d] = sorted(in_hours, key=lambda b: b.ts)
    return by_day


def _t_years(entry_ts: datetime, dte_days_at_entry: int) -> float:
    """Time to expiry in years for BS. Expiry assumed at 15:30 IST."""
    expiry_date = entry_ts.date() + timedelta(days=dte_days_at_entry)
    expiry_close = datetime.combine(expiry_date, time(15, 30), tzinfo=IST)
    seconds = max((expiry_close - entry_ts).total_seconds(), 60)
    return seconds / (365.0 * 24 * 3600)


def _make_rule(rule_name: str, min_volume_z: float, rule_kwargs: dict):
    if rule_name == "orb":
        return OpeningRangeBreakout(min_volume_z=min_volume_z, **rule_kwargs)
    if rule_name == "macd_ema_oi":
        from .rules.macd_ema_cross import MacdEmaCrossOI
        kwargs = {"require_oi_zone": False, **rule_kwargs}
        return MacdEmaCrossOI(**kwargs)
    if rule_name == "ema_sar":
        from .rules.ema_sar import EmaSarRule
        return EmaSarRule(**rule_kwargs)
    if rule_name == "gap_fill_fade":
        from .rules.gap_fill_fade import GapFillFade
        return GapFillFade(**rule_kwargs)
    if rule_name == "vwap_fade":
        from .rules.vwap_fade import VwapFade
        return VwapFade(**rule_kwargs)
    if rule_name == "max_pain_magnet":
        from .rules.max_pain_magnet import MaxPainMagnet
        return MaxPainMagnet(**rule_kwargs)
    if rule_name == "simple_ema_cross":
        from .rules.simple_ema_cross import SimpleEmaCross
        return SimpleEmaCross(**rule_kwargs)
    if rule_name == "swing_ema_trend":
        from .rules.swing_ema_trend import SwingEmaTrend
        return SwingEmaTrend(**rule_kwargs)
    if rule_name == "trend_hold":
        from .rules.trend_hold import TrendHold
        return TrendHold(**rule_kwargs)
    if rule_name == "auto_iron_condor":
        from .rules.auto_iron_condor import AutoIronCondor
        return AutoIronCondor(**rule_kwargs)
    if rule_name == "bull_put_spread":
        from .rules.bull_put_spread import BullPutSpread
        return BullPutSpread(**rule_kwargs)
    if rule_name == "deep_itm_trend":
        from .rules.deep_itm_trend import DeepItmTrend
        return DeepItmTrend(**rule_kwargs)
    if rule_name == "rsi_ema_ce":
        from .rules.rsi_ema_ce import RsiEmaCE
        return RsiEmaCE(**rule_kwargs)
    if rule_name == "htf_naked":
        from .rules.htf_naked import HtfNaked
        return HtfNaked(**rule_kwargs)
    if rule_name == "htf_ema_retest":
        from .rules.htf_ema_retest import HtfEmaRetest
        return HtfEmaRetest(**rule_kwargs)
    if rule_name == "fast_ema_retest":
        from .rules.fast_ema_retest import FastEmaRetest
        return FastEmaRetest(**rule_kwargs)
    if rule_name == "iv_regime":
        from .rules.iv_regime import IVRegime
        return IVRegime(**rule_kwargs)
    if rule_name == "vwap_reclaim_ce":
        from .rules.vwap_reclaim_ce import VwapReclaimCE
        return VwapReclaimCE(**rule_kwargs)
    if rule_name == "macd_bb_ce":
        from .rules.macd_bb_ce import MacdBbCE
        return MacdBbCE(**rule_kwargs)
    if rule_name == "confluence_ce":
        from .rules.confluence_ce import ConfluenceCE
        return ConfluenceCE(**rule_kwargs)
    if rule_name == "sniper_ce":
        from .rules.sniper_ce import SniperCE
        return SniperCE(**rule_kwargs)
    if rule_name == "nifty_intraday_buyer":
        from .rules.nifty_intraday_buyer import NiftyIntradayBuyer
        return NiftyIntradayBuyer(**rule_kwargs)
    if rule_name == "panic_bounce_ce":
        from .rules.panic_bounce_ce import PanicBounceCE
        return PanicBounceCE(**rule_kwargs)
    raise ValueError(f"Unknown rule: {rule_name}")


def _simulate_multileg_day(day, day_bars, sig, i_entry, entry_bar, dte_days):
    """Walk forward simulating a multi-leg structure (iron condor, straddle, etc.).
    Uses each leg's own IV from signal — preserves the IV smile context."""
    ctx = sig.trigger_context
    legs = ctx.get("legs") or []
    structure = ctx.get("kind", "unknown")
    leg_dte = ctx.get("dte_days", dte_days)

    if not legs:
        return None

    # Entry net credit (sum of signed leg premiums)
    entry_net = 0.0
    for leg in legs:
        sign = +1.0 if leg["action"] == "SELL" else -1.0
        entry_net += sign * float(leg["premium"])

    # Determine target/stop cost-to-close levels
    # Rule provides: target_premium_gain = net_credit * 0.75 (so close at 25% of credit)
    # And: stop_premium = -net_credit (i.e. lose 1x credit, close cost = 2x credit)
    target_close_cost = abs(entry_net) * 0.25     # target close cost
    stop_close_cost = abs(entry_net) * 2.0        # 1x credit loss

    # Hold window
    hold_minutes = sig.hold_minutes or 240
    if len(day_bars) >= 2:
        bar_minutes = max(1, int((day_bars[1].ts - day_bars[0].ts).total_seconds() / 60))
    else:
        bar_minutes = 5
    max_hold_bars = max(1, hold_minutes // bar_minutes)
    end_idx = min(i_entry + max_hold_bars, len(day_bars) - 1)

    exit_reason = "timeout"
    exit_net = None
    exit_bar = day_bars[end_idx]

    for j in range(i_entry + 1, end_idx + 1):
        b = day_bars[j]
        # Compute current cost to close (sum of signed BS prices using each leg's iv)
        t_now = _t_years(b.ts, leg_dte)
        current_net = 0.0
        for leg in legs:
            sign = +1.0 if leg["action"] == "SELL" else -1.0
            iv_leg = float(leg.get("iv") or 0.13)
            prem = black_scholes(spot=b.close, strike=int(leg["strike"]),
                                  t_years=t_now, iv=iv_leg,
                                  option_type=leg["option_type"]).price
            current_net += sign * prem

        # For credit positions (entry_net > 0): profit = entry_net - current_net (we want current_net to drop)
        # For debit positions (entry_net < 0): profit = current_net - entry_net (we want current_net to rise)
        if entry_net > 0:
            # Credit. Target when current_net <= target_close_cost
            if current_net <= target_close_cost:
                exit_net = current_net
                exit_reason = "target"; exit_bar = b; break
            # Stop when current_net >= stop_close_cost
            if current_net >= stop_close_cost:
                exit_net = current_net
                exit_reason = "stop"; exit_bar = b; break
        else:
            # Debit. Profit when current_net rises by target amount
            if current_net - entry_net >= abs(entry_net) * 0.5:    # generic profit target
                exit_net = current_net
                exit_reason = "target"; exit_bar = b; break
            if current_net <= entry_net * 0.5:    # 50% decay = stop on debit
                exit_net = current_net
                exit_reason = "stop"; exit_bar = b; break

    if exit_net is None:
        # Timeout
        t_now = _t_years(exit_bar.ts, leg_dte)
        exit_net = 0.0
        for leg in legs:
            sign = +1.0 if leg["action"] == "SELL" else -1.0
            iv_leg = float(leg.get("iv") or 0.13)
            prem = black_scholes(spot=exit_bar.close, strike=int(leg["strike"]),
                                  t_years=t_now, iv=iv_leg,
                                  option_type=leg["option_type"]).price
            exit_net += sign * prem

    pnl_per_lot = (entry_net - exit_net) * 75 if entry_net > 0 else (exit_net - entry_net) * 75

    return MultiLegBTTrade(
        day=day,
        entry_ts=entry_bar.ts,
        rule_name=sig.rule_name,
        structure=structure,
        legs=legs,
        entry_spot=entry_bar.close,
        entry_net_credit=round(entry_net, 2),
        exit_ts=exit_bar.ts,
        exit_spot=exit_bar.close,
        exit_net_credit=round(exit_net, 2),
        exit_reason=exit_reason,
        pnl_per_lot=round(pnl_per_lot, 2),
        bars_held=day_bars.index(exit_bar) - i_entry,
    )


def _simulate_day(day: date, day_bars: list[Bar], iv: float,
                  dte_days: int = 6,
                  min_volume_z: float = -99.0,
                  rule_kwargs: dict | None = None,
                  rule_name: str = "orb",
                  today_start_idx: int = 0):
    """Walk a day's bars forward; if rule fires, simulate the trade.

    Returns BTTrade (single-leg) or MultiLegBTTrade (multi-leg structure).
    """
    rule_kwargs = rule_kwargs or {}
    rule = _make_rule(rule_name, min_volume_z, rule_kwargs)
    entry: tuple[int, Bar, object] | None = None

    # For rules like auto_iron_condor that depend on realistic weekly DTE,
    # compute DTE = days to next Thursday from each bar's date. Otherwise
    # use the static dte_days passed in.
    needs_weekly_dte = rule_name in ("auto_iron_condor",)

    def _dte_for(bar_date):
        if not needs_weekly_dte:
            return dte_days
        # Next Thursday (weekday 3). If today is Thu, dte=0; Tue=2; Wed=1; Fri=6
        days_to_thu = (3 - bar_date.weekday()) % 7
        if days_to_thu == 0:
            days_to_thu = 7  # roll to next Thursday on Thursday itself
        return days_to_thu

    start_eval = today_start_idx + 6
    for i in range(start_eval, len(day_bars)):
        b = day_bars[i]
        bar_dte = _dte_for(b.ts.date())
        # Build a synthetic chain at current spot so ATM strike lookup works.
        # anchor_dt anchors the chain's expiry to the bar's date (not real "now").
        chain = synthetic_chain(symbol="NIFTY", spot=b.close, iv=iv,
                                dte_days=bar_dte, strikes_each_side=40,
                                anchor_dt=b.ts)
        state = MarketState(
            symbol="NIFTY", spot=b.close, chain=chain,
            bars_5m=day_bars[:i+1], ts=b.ts,
        )
        sig = rule.evaluate(state)
        if sig is not None:
            entry = (i, b, sig)
            break

    if entry is None:
        return None

    i_entry, entry_bar, sig = entry

    # Dispatch to multi-leg simulator if signal carries `legs` in trigger_context
    if sig.trigger_context.get("legs"):
        return _simulate_multileg_day(day, day_bars, sig, i_entry, entry_bar, dte_days)

    strike = sig.strike
    opt_type = sig.option_type

    # Entry premium via BS at entry spot
    t_entry = _t_years(entry_bar.ts, dte_days)
    entry_premium = black_scholes(
        spot=entry_bar.close, strike=strike, t_years=t_entry,
        iv=iv, option_type=opt_type,
    ).price
    target_premium = entry_premium + (sig.target_premium_gain or 0)
    stop_spot = sig.stop_spot
    stop_premium_level = sig.stop_premium
    exit_on_ema_flip = bool(sig.trigger_context.get("exit_on_ema_flip"))
    eod_close_minutes = sig.trigger_context.get("eod_close_minutes")
    ema_fast_n = sig.trigger_context.get("ema_fast_period") or 20
    ema_slow_n = sig.trigger_context.get("ema_slow_period") or 50
    entry_direction = sig.trigger_context.get("entry_direction") or sig.direction

    # Walk forward bars within rule's declared hold window
    hold_minutes = sig.hold_minutes or 90
    if len(day_bars) >= 2:
        bar_minutes = max(1, int((day_bars[1].ts - day_bars[0].ts).total_seconds() / 60))
    else:
        bar_minutes = 5
    max_hold_bars = max(1, hold_minutes // bar_minutes)
    end_idx = min(i_entry + max_hold_bars, len(day_bars) - 1)

    exit_reason = "timeout"
    exit_premium = None
    exit_bar = day_bars[end_idx]

    for j in range(i_entry + 1, end_idx + 1):
        b = day_bars[j]
        t_now = _t_years(b.ts, dte_days)

        # Spot-based stop (when stop_spot is defined)
        if stop_spot is not None:
            if sig.direction == "bullish" and b.low <= stop_spot:
                exit_premium = black_scholes(spot=stop_spot, strike=strike,
                                             t_years=t_now, iv=iv,
                                             option_type=opt_type).price
                exit_reason = "stop"; exit_bar = b; break
            if sig.direction == "bearish" and b.high >= stop_spot:
                exit_premium = black_scholes(spot=stop_spot, strike=strike,
                                             t_years=t_now, iv=iv,
                                             option_type=opt_type).price
                exit_reason = "stop"; exit_bar = b; break

        # Premium-based catastrophic stop (trend_hold uses this)
        if stop_premium_level is not None:
            adverse_spot = b.high if opt_type == "PE" else b.low
            adverse_prem = black_scholes(spot=adverse_spot, strike=strike,
                                         t_years=t_now, iv=iv,
                                         option_type=opt_type).price
            if adverse_prem <= float(stop_premium_level):
                exit_premium = adverse_prem
                exit_reason = "stop"; exit_bar = b; break

        # Target (use intra-bar favourable price)
        check_spot = b.high if opt_type == "CE" else b.low
        check_premium = black_scholes(spot=check_spot, strike=strike,
                                      t_years=t_now, iv=iv,
                                      option_type=opt_type).price
        if check_premium >= target_premium:
            exit_premium = target_premium
            exit_reason = "target"; exit_bar = b; break

        # EMA-flip exit (trend_hold)
        if exit_on_ema_flip:
            from .indicators import ema as _ema
            closes_so_far = [bb.close for bb in day_bars[:j+1]]
            if len(closes_so_far) >= ema_slow_n + 2:
                ef = _ema(closes_so_far, ema_fast_n)
                es = _ema(closes_so_far, ema_slow_n)
                if ef[-1] is not None and es[-1] is not None:
                    flipped = (
                        (entry_direction == "bearish" and ef[-1] > es[-1]) or
                        (entry_direction == "bullish" and ef[-1] < es[-1])
                    )
                    if flipped:
                        exit_premium = black_scholes(spot=b.close, strike=strike,
                                                     t_years=t_now, iv=iv,
                                                     option_type=opt_type).price
                        exit_reason = "ema_flip"; exit_bar = b; break

        # EOD close
        if eod_close_minutes is not None:
            mins_since_open = (b.ts.hour * 60 + b.ts.minute) - (9 * 60 + 15)
            if mins_since_open >= float(eod_close_minutes):
                exit_premium = black_scholes(spot=b.close, strike=strike,
                                             t_years=t_now, iv=iv,
                                             option_type=opt_type).price
                exit_reason = "eod"; exit_bar = b; break

    if exit_premium is None:
        # Timeout — use last bar's close
        t_now = _t_years(exit_bar.ts, dte_days)
        exit_premium = black_scholes(
            spot=exit_bar.close, strike=strike, t_years=t_now,
            iv=iv, option_type=opt_type,
        ).price

    return BTTrade(
        day=day,
        entry_ts=entry_bar.ts,
        direction=sig.direction,
        option_type=opt_type,
        strike=strike,
        entry_spot=entry_bar.close,
        entry_premium=round(entry_premium, 2),
        target_premium=round(target_premium, 2),
        stop_spot=stop_spot,
        stop_premium=round(sig.stop_premium or 0, 2),
        exit_ts=exit_bar.ts,
        exit_spot=exit_bar.close,
        exit_premium=round(exit_premium, 2),
        exit_reason=exit_reason,
        pnl_per_lot=round((exit_premium - entry_premium) * 75, 2),
        bars_held=day_bars.index(exit_bar) - i_entry,
    )


def run_backtest(days: int = 60, iv: float = 0.13,
                 dte_days: int = 6, lot_size: int = 75,
                 min_volume_z: float = -99.0,
                 rule_kwargs: dict | None = None,
                 _bars_cache: list | None = None,
                 rule_name: str = "orb",
                 interval: str = "5minute") -> list[BTTrade]:
    if _bars_cache is None:
        bars = fetch_bars_cached(days, interval=interval)
    else:
        bars = _bars_cache

    by_day = _group_by_day(bars)
    if _bars_cache is None:
        print(f"Got {len(bars)} bars across {len(by_day)} trading days.", file=sys.stderr)

    sorted_days = sorted(by_day.keys())
    trades: list[BTTrade] = []
    for idx, day in enumerate(sorted_days):
        if day == datetime.now(tz=IST).date():
            continue
        # Provide up to 3 prior days as warmup for indicators that need history.
        warmup: list[Bar] = []
        for prev_day in sorted_days[max(0, idx - 3):idx]:
            warmup.extend(by_day[prev_day])
        combined = warmup + by_day[day]
        today_start_idx = len(warmup)
        t = _simulate_day(day, combined, iv=iv, dte_days=dte_days,
                          min_volume_z=min_volume_z, rule_kwargs=rule_kwargs,
                          rule_name=rule_name, today_start_idx=today_start_idx)
        if t is not None:
            trades.append(t)
    return trades


_KITE_INTERVALS = {"minute", "3minute", "5minute", "10minute", "15minute", "30minute", "60minute"}


def _resample_bars(bars: list[Bar], group_minutes: int) -> list[Bar]:
    """Aggregate 1-min bars into `group_minutes`-min bars anchored at 9:15."""
    if not bars or group_minutes <= 1:
        return bars
    out: list[Bar] = []
    by_day: dict = {}
    for b in bars:
        by_day.setdefault(b.ts.date(), []).append(b)
    for day, day_bars in sorted(by_day.items()):
        from datetime import datetime as dt, time as dtime
        anchor = dt.combine(day, dtime(9, 15), tzinfo=IST)
        bucket: dict[int, list[Bar]] = {}
        for b in day_bars:
            minutes_offset = int((b.ts - anchor).total_seconds() // 60)
            if minutes_offset < 0:
                continue
            bucket_idx = minutes_offset // group_minutes
            bucket.setdefault(bucket_idx, []).append(b)
        for idx in sorted(bucket):
            bs = bucket[idx]
            out.append(Bar(
                ts=anchor + timedelta(minutes=idx * group_minutes),
                open=bs[0].open,
                high=max(b.high for b in bs),
                low=min(b.low for b in bs),
                close=bs[-1].close,
                volume=sum(b.volume for b in bs),
            ))
    return out


def _compute_leg_net(bar: Bar, legs: list[dict], dte_days: int) -> float:
    """Net signed premium = sum over legs of (sign * BS price). SELL=+1, BUY=-1."""
    t_now = _t_years(bar.ts, dte_days)
    net = 0.0
    for leg in legs:
        sign = +1.0 if leg["action"] == "SELL" else -1.0
        iv_leg = float(leg.get("iv") or 0.13)
        prem = black_scholes(
            spot=bar.close, strike=int(leg["strike"]),
            t_years=t_now, iv=iv_leg, option_type=leg["option_type"]
        ).price
        net += sign * prem
    return net


def run_multileg_backtest(days: int, iv: float, rule_name: str,
                          rule_kwargs: dict | None = None,
                          interval: str = "5minute",
                          symbol: str = "NIFTY") -> list:
    """
    Cross-day backtest for multi-leg structures (iron condor etc).
    Walks bars chronologically; when rule fires, holds position across days
    until target/stop hit, expiry, or hold timeout.
    """
    rule_kwargs = rule_kwargs or {}
    # Split backtest-only knobs from rule constructor kwargs
    BACKTEST_ONLY_KEYS = {"backtest_target_close_pct", "backtest_stop_credit_mult"}
    rule_only_kwargs = {k: v for k, v in rule_kwargs.items() if k not in BACKTEST_ONLY_KEYS}
    rule = _make_rule(rule_name, -99.0, rule_only_kwargs)

    bars = fetch_bars_cached(days, interval=interval, symbol=symbol)
    bars = sorted(bars, key=lambda b: b.ts)
    if not bars:
        return []

    by_day = _group_by_day(bars)
    sorted_days = sorted(by_day.keys())

    # Build a global bar index with at-least-1-day warmup so rules see prior bars
    bars_for_rule = []
    bar_to_global_idx = {}
    for d in sorted_days:
        for b in by_day[d]:
            bar_to_global_idx[id(b)] = len(bars_for_rule)
            bars_for_rule.append(b)

    if not bars_for_rule:
        return []

    trades: list = []
    position: dict | None = None
    today = datetime.now(tz=IST).date()

    # Allow at least one warmup day before evaluating
    warmup_idx = max(0, len(by_day[sorted_days[0]]))   # bars in first day

    for gi, b in enumerate(bars_for_rule):
        if b.ts.date() == today:
            break
        if gi < warmup_idx:
            continue

        # Compute DTE for this bar (days to next Thursday)
        days_to_thu = (3 - b.ts.date().weekday()) % 7
        if days_to_thu == 0:
            days_to_thu = 7

        if position is None:
            # --- Look for entry ---
            # Apply a realistic NIFTY IV smile for IC pricing: OTM > ATM
            from .manual_chain import make_iv_curve
            atm_strike = round(b.close / 50) * 50
            smile_anchors = {
                atm_strike - 600: iv + 0.045,
                atm_strike - 400: iv + 0.030,
                atm_strike - 200: iv + 0.015,
                atm_strike - 100: iv + 0.007,
                atm_strike: iv,
                atm_strike + 100: iv + 0.005,
                atm_strike + 200: iv + 0.012,
                atm_strike + 400: iv + 0.025,
                atm_strike + 600: iv + 0.038,
            }
            iv_curve = make_iv_curve(smile_anchors)
            chain = synthetic_chain(symbol=symbol, spot=b.close, iv=iv,
                                    dte_days=days_to_thu, strikes_each_side=40,
                                    anchor_dt=b.ts, iv_curve=iv_curve)
            state = MarketState(symbol=symbol, spot=b.close, chain=chain,
                                bars_5m=bars_for_rule[:gi+1], ts=b.ts)
            try:
                sig = rule.evaluate(state)
            except Exception:
                continue
            if sig is None or not sig.trigger_context.get("legs"):
                continue

            legs = sig.trigger_context["legs"]
            entry_dte_at_signal = sig.trigger_context.get("dte_days", days_to_thu)
            entry_net = sum(
                (+1.0 if leg["action"] == "SELL" else -1.0) * float(leg["premium"])
                for leg in legs
            )
            expiry_date = b.ts.date() + timedelta(days=entry_dte_at_signal)
            position = {
                "entry_bar": b, "entry_gi": gi, "sig": sig,
                "legs": legs, "entry_net": entry_net,
                "entry_dte": entry_dte_at_signal,
                "expiry_date": expiry_date,
            }
        else:
            # --- Hold; check exit conditions ---
            legs = position["legs"]
            entry_net = position["entry_net"]
            expiry_date = position["expiry_date"]
            entry_bar = position["entry_bar"]

            # Compute remaining DTE at this bar
            dte_remaining_days = (expiry_date - b.ts.date()).days
            seconds_to_expiry_close = (
                datetime.combine(expiry_date, time(15, 30), tzinfo=IST) - b.ts
            ).total_seconds()
            t_years_now = max(seconds_to_expiry_close / (365 * 24 * 3600), 1e-5)
            current_net = 0.0
            for leg in legs:
                sign = +1.0 if leg["action"] == "SELL" else -1.0
                iv_leg = float(leg.get("iv") or 0.13)
                prem = black_scholes(
                    spot=b.close, strike=int(leg["strike"]),
                    t_years=t_years_now, iv=iv_leg, option_type=leg["option_type"]
                ).price
                current_net += sign * prem

            exit_reason = None
            kwargs = rule_kwargs or {}
            target_pct = kwargs.get("backtest_target_close_pct", 0.25)
            stop_mult = kwargs.get("backtest_stop_credit_mult", 2.0)
            target_close_cost = abs(entry_net) * target_pct
            stop_close_cost = abs(entry_net) * stop_mult

            if entry_net > 0:
                if current_net <= target_close_cost:
                    exit_reason = "target"
                elif stop_mult > 0 and current_net >= stop_close_cost:
                    exit_reason = "stop"
            else:
                if current_net - entry_net >= abs(entry_net) * 0.5:
                    exit_reason = "target"
                elif current_net <= entry_net * 0.5:
                    exit_reason = "stop"

            # Expiry close = forced exit
            if exit_reason is None and seconds_to_expiry_close < 30 * 60:
                exit_reason = "expiry"

            if exit_reason:
                pnl = (entry_net - current_net) * 75 if entry_net > 0 else (current_net - entry_net) * 75
                trades.append(MultiLegBTTrade(
                    day=entry_bar.ts.date(),
                    entry_ts=entry_bar.ts,
                    rule_name=position["sig"].rule_name,
                    structure=position["sig"].trigger_context.get("kind", "unknown"),
                    legs=legs,
                    entry_spot=entry_bar.close,
                    entry_net_credit=round(entry_net, 2),
                    exit_ts=b.ts,
                    exit_spot=b.close,
                    exit_net_credit=round(current_net, 2),
                    exit_reason=exit_reason,
                    pnl_per_lot=round(pnl, 2),
                    bars_held=gi - position["entry_gi"],
                ))
                position = None

    return trades


def fetch_bars_cached(days: int, interval: str = "5minute",
                      symbol: str = "NIFTY") -> list[Bar]:
    """
    Returns bars for the requested range. Pulls 1-min from the persistent disk
    cache and resamples to the requested interval. (Day interval served directly
    from day cache.) Falls back to direct Kite pull only if 1-min cache is empty.
    """
    from .data.cache import get_bars

    if interval == "day":
        return get_bars(symbol, "day", days)

    # Always resample from 1-min cache — covers minute, 2/3/4/5/10/15-minute, etc.
    if interval == "minute":
        bars = get_bars(symbol, "minute", days)
    else:
        if not interval.endswith("minute"):
            raise ValueError(f"Unsupported interval: {interval}")
        try:
            group = int(interval.replace("minute", ""))
        except ValueError:
            raise ValueError(f"Bad interval: {interval}")
        bars_1m = get_bars(symbol, "minute", days)
        bars = _resample_bars(bars_1m, group)
        print(f"  resampled {len(bars_1m)} 1-min bars -> {len(bars)} {group}-min bars",
              file=sys.stderr)

    print(f"Got {len(bars)} bars.", file=sys.stderr)
    return bars


def summarize(trades: list, lot_size: int = 75) -> dict:
    """Works for both BTTrade (single-leg) and MultiLegBTTrade. Computes universal stats."""
    if not trades:
        return {"trades": 0}
    wins = [t for t in trades if t.exit_reason == "target"]
    stops = [t for t in trades if t.exit_reason == "stop"]
    timeouts = [t for t in trades if t.exit_reason == "timeout"]
    pos_pnl = [t for t in trades if t.pnl_per_lot > 0]
    neg_pnl = [t for t in trades if t.pnl_per_lot <= 0]

    total_pnl = sum(t.pnl_per_lot for t in trades)
    cum = 0.0; peak = 0.0; max_dd = 0.0
    for t in trades:
        cum += t.pnl_per_lot
        peak = max(peak, cum)
        max_dd = min(max_dd, cum - peak)

    avg_win = sum(t.pnl_per_lot for t in pos_pnl) / len(pos_pnl) if pos_pnl else 0
    avg_loss = sum(t.pnl_per_lot for t in neg_pnl) / len(neg_pnl) if neg_pnl else 0

    # Direction is only meaningful for single-leg trades
    is_multileg = isinstance(trades[0], MultiLegBTTrade)
    bull = [] if is_multileg else [t for t in trades if t.direction == "bullish"]
    bear = [] if is_multileg else [t for t in trades if t.direction == "bearish"]

    return {
        "trades": len(trades),
        "structure": getattr(trades[0], "structure", "single_leg") if is_multileg else "single_leg",
        "wins_target": len(wins),
        "stops": len(stops),
        "timeouts": len(timeouts),
        "win_rate_target": len(wins) / len(trades),
        "profit_rate_any": len(pos_pnl) / len(trades),
        "avg_win_inr": round(avg_win, 0),
        "avg_loss_inr": round(avg_loss, 0),
        "avg_rr": round(abs(avg_win / avg_loss), 2) if avg_loss else None,
        "total_pnl_inr": round(total_pnl, 0),
        "max_drawdown_inr": round(max_dd, 0),
        "bull_count": len(bull),
        "bear_count": len(bear),
        "best_trade": round(max(t.pnl_per_lot for t in trades), 0),
        "worst_trade": round(min(t.pnl_per_lot for t in trades), 0),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rule", default="orb")
    p.add_argument("--days", type=int, default=60)
    p.add_argument("--iv", type=float, default=0.13,
                   help="constant IV assumption (NIFTY ATM-ish)")
    p.add_argument("--dte", type=int, default=6, help="DTE at each entry")
    p.add_argument("--csv", default=None, help="write per-trade CSV here")
    p.add_argument("--json", action="store_true")
    p.add_argument("--min-vol-z", type=float, default=-99.0,
                   help="min volume z-score (default -99 disables; NIFTY index has no volume). "
                        "Set to 0.5 to match live-trading filter behavior.")
    p.add_argument("--interval", default="5minute",
                   help="Bar interval: 'minute', '3minute', '4minute' (resampled), "
                        "'5minute' (default), '10minute', etc.")
    p.add_argument("--rule-kwargs", default=None,
                   help='JSON dict of rule kwargs (inline). Use --rule-config-file on Windows.')
    p.add_argument("--rule-config-file", default=None,
                   help='Path to a JSON file with rule kwargs. Easier than inline on Windows.')
    args = p.parse_args()

    valid_rules = ("orb", "macd_ema_oi", "ema_sar", "gap_fill_fade", "vwap_fade",
                   "max_pain_magnet", "simple_ema_cross", "swing_ema_trend", "trend_hold",
                   "auto_iron_condor", "deep_itm_trend", "rsi_ema_ce",
                   "vwap_reclaim_ce", "macd_bb_ce", "confluence_ce",
                   "bull_put_spread", "sniper_ce", "htf_naked", "iv_regime",
                   "fast_ema_retest", "htf_ema_retest", "nifty_intraday_buyer",
                   "panic_bounce_ce")
    if args.rule not in valid_rules:
        print(f"Unknown rule '{args.rule}'. Use: {' | '.join(valid_rules)}", file=sys.stderr)
        sys.exit(2)

    rule_kwargs = None
    if args.rule_config_file:
        with open(args.rule_config_file, "r", encoding="utf-8") as f:
            rule_kwargs = json.load(f)
    elif args.rule_kwargs:
        rule_kwargs = json.loads(args.rule_kwargs)

    MULTILEG_RULES = {"auto_iron_condor", "bull_put_spread"}
    if args.rule in MULTILEG_RULES:
        trades = run_multileg_backtest(
            days=args.days, iv=args.iv,
            rule_name=args.rule, rule_kwargs=rule_kwargs,
            interval=args.interval,
        )
    else:
        trades = run_backtest(days=args.days, iv=args.iv, dte_days=args.dte,
                              min_volume_z=args.min_vol_z,
                              rule_name=args.rule, interval=args.interval,
                              rule_kwargs=rule_kwargs)
    stats = summarize(trades)

    if args.json:
        print(json.dumps({
            "params": {"rule": args.rule, "days": args.days, "iv": args.iv, "dte": args.dte},
            "stats": stats,
            "trades": [t.__dict__ for t in trades],
        }, indent=2, default=str))
        return

    # Human-readable
    print(f"\n=== {args.rule.upper()} backtest — last {args.days}d @ IV={args.iv*100:.1f}%, DTE={args.dte}d ===\n")
    if stats["trades"] == 0:
        print("No trades. Either rule never fired or no historical data available.")
        return
    print(f"Total trades:        {stats['trades']}")
    print(f"  bullish:           {stats['bull_count']}")
    print(f"  bearish:           {stats['bear_count']}")
    print()
    print(f"Exit breakdown:")
    print(f"  target hit:        {stats['wins_target']:3}  ({stats['win_rate_target']*100:.1f}%)")
    print(f"  stop hit:          {stats['stops']:3}")
    print(f"  timeout (90m):     {stats['timeouts']:3}")
    print()
    print(f"PnL (per 1 lot @ 75):")
    print(f"  total:             Rs {stats['total_pnl_inr']:>+12,.0f}")
    print(f"  avg win:           Rs {stats['avg_win_inr']:>+12,.0f}")
    print(f"  avg loss:          Rs {stats['avg_loss_inr']:>+12,.0f}")
    print(f"  R/R (|win/loss|):  {stats['avg_rr'] if stats['avg_rr'] else '-':>12}")
    print(f"  best trade:        Rs {stats['best_trade']:>+12,.0f}")
    print(f"  worst trade:       Rs {stats['worst_trade']:>+12,.0f}")
    print(f"  max drawdown:      Rs {stats['max_drawdown_inr']:>+12,.0f}")
    print(f"  profit rate:       {stats['profit_rate_any']*100:.1f}%   (any positive close)")
    print()

    # Honest caveats
    print("CAVEATS: synthetic BS premiums (real chain has smile + spread).")
    print("         No slippage. Real fills typically 1-3 pts worse per leg.")
    print(f"         Constant IV={args.iv*100:.1f}% across the period.")
    print(f"         60-day window — captures current regime only.")
    print()
    print("If win rate > 60% or PnL > 0 by a lot, be suspicious. Real-world numbers")
    print("after costs/slippage are usually 8-15% lower than backtest.")

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(trades[0].__dict__.keys()))
            writer.writeheader()
            for t in trades:
                writer.writerow(t.__dict__)
        print(f"\nPer-trade CSV written to {args.csv}")


if __name__ == "__main__":
    main()
