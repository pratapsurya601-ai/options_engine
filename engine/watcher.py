"""
Rule watcher — polling loop that:
  1. Builds MarketState from a data source (Kite live OR manual snapshot)
  2. Evaluates each registered rule against the state
  3. On Signal: prints, logs to JSONL, opens a paper position
  4. Monitors open paper positions; closes on target/stop/timeout
  5. Loops every `interval_sec` while market is open

Usage:
  # Kite live
  python -m engine.watcher --source kite --rule orb --interval 60

  # Manual one-shot (for testing rule logic with pasted state)
  python -m engine.watcher --source manual --rule orb \\
    --spot 23907 --iv 0.125 --dte 6 \\
    --bars-file bars.json

  bars.json format:
    [{"ts": "2026-05-27T09:15:00+05:30", "open": 23900, "high": 23945,
      "low": 23890, "close": 23920, "volume": 1200000}, ...]
"""
from __future__ import annotations

import argparse
import json
import sys
import time as time_mod
from datetime import datetime, time, timedelta, timezone, date
from pathlib import Path

from .state import MarketState, Bar
from .signals import Signal
from .paper import write_open, write_close, open_paper_positions
from .pricing import black_scholes


IST = timezone(timedelta(hours=5, minutes=30))
SIGNAL_LOG = Path("logs/signals.jsonl")


def _ensure_logs():
    SIGNAL_LOG.parent.mkdir(parents=True, exist_ok=True)


def _log_signal(sig: Signal):
    _ensure_logs()
    with SIGNAL_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(sig.to_dict(), default=str) + "\n")


def _print_signal(sig: Signal):
    print(f"\n[{sig.ts.strftime('%Y-%m-%d %H:%M:%S')}] {sig.rule_name}  TRIGGERED  ({sig.direction})")
    print(f"  Action: {sig.action}  strike={sig.strike}  target_spot={sig.target_spot}  stop_spot={sig.stop_spot}")
    print(f"  Target premium gain: ~{sig.target_premium_gain} pts  Stop premium: Rs {sig.stop_premium}")
    if sig.prob_estimates:
        probs = "  ".join(f"+{k}pts:{v*100:.0f}%" for k, v in sig.prob_estimates.items())
        print(f"  P(touch) within {sig.hold_minutes}m: {probs}")
    print(f"  Context: {sig.trigger_context}")
    print(f"  Thesis: {sig.thesis[:200]}...")


# --- Rule registry ---
def _load_rules(names: list[str], config_files: list[str] | None = None):
    from .rules.orb import OpeningRangeBreakout
    from .rules.macd_ema_cross import MacdEmaCrossOI
    from .rules.oi_direction import OIDirectionRule
    from .rules.ema_sar import EmaSarRule
    from .rules.gap_fill_fade import GapFillFade
    from .rules.vwap_fade import VwapFade
    from .rules.max_pain_magnet import MaxPainMagnet
    from .rules.simple_ema_cross import SimpleEmaCross
    from .rules.swing_ema_trend import SwingEmaTrend
    from .rules.trend_hold import TrendHold
    from .rules.auto_iron_condor import AutoIronCondor
    from .rules.event_vol import PreEventVolBuy, PostEventIVCrush
    from .rules.smart_money_flow import SmartMoneyFlow
    from .rules.htf_naked import HtfNaked
    from .rules.rsi_ema_ce import RsiEmaCE
    from .rules.vwap_reclaim_ce import VwapReclaimCE
    from .rules.macd_bb_ce import MacdBbCE
    from .rules.confluence_ce import ConfluenceCE
    from .rules.sniper_ce import SniperCE
    from .rules.htf_ema_retest import HtfEmaRetest
    from .rules.fast_ema_retest import FastEmaRetest
    from .rules.nifty_intraday_buyer import NiftyIntradayBuyer
    from .rules.panic_bounce_ce import PanicBounceCE
    registry = {
        "orb": OpeningRangeBreakout,
        "macd_ema_oi": MacdEmaCrossOI,
        "oi_direction": OIDirectionRule,
        "ema_sar": EmaSarRule,
        "gap_fill_fade": GapFillFade,
        "vwap_fade": VwapFade,
        "max_pain_magnet": MaxPainMagnet,
        "simple_ema_cross": SimpleEmaCross,
        "swing_ema_trend": SwingEmaTrend,
        "trend_hold": TrendHold,
        "auto_iron_condor": AutoIronCondor,
        "pre_event_vol_buy": PreEventVolBuy,
        "post_event_iv_crush": PostEventIVCrush,
        "smart_money_flow": SmartMoneyFlow,
        "htf_naked": HtfNaked,
        "rsi_ema_ce": RsiEmaCE,
        "vwap_reclaim_ce": VwapReclaimCE,
        "macd_bb_ce": MacdBbCE,
        "confluence_ce": ConfluenceCE,
        "sniper_ce": SniperCE,
        "htf_ema_retest": HtfEmaRetest,
        "fast_ema_retest": FastEmaRetest,
        "nifty_intraday_buyer": NiftyIntradayBuyer,
        "panic_bounce_ce": PanicBounceCE,
    }
    config_files = config_files or [None] * len(names)
    while len(config_files) < len(names):
        config_files.append(None)
    rules = []
    for n, cfg_path in zip(names, config_files):
        if n not in registry:
            raise ValueError(f"Unknown rule: {n}. Available: {list(registry)}")
        kwargs = {}
        if cfg_path:
            with open(cfg_path, "r", encoding="utf-8") as f:
                kwargs = json.load(f)
        rules.append(registry[n](**kwargs))
    return rules


# --- State builders ---
def _build_state_kite(symbol: str, with_oi: bool = False,
                      bar_interval: str = "5minute",
                      warmup_days: int = 3,
                      use_futures_bars: bool = False,
                      expiry_offset: int = 0) -> MarketState:
    from .data.kite_source import (
        spot_ltp, historical_bars, option_chain, populate_ivs,
        NIFTY_SPOT_TOKEN, BANKNIFTY_SPOT_TOKEN, nearest_future_token,
        list_expiries,
    )

    if use_futures_bars:
        token, tradingsymbol, fut_expiry = nearest_future_token(symbol)
    else:
        token = NIFTY_SPOT_TOKEN if symbol == "NIFTY" else BANKNIFTY_SPOT_TOKEN
        tradingsymbol = symbol
        fut_expiry = None

    now = datetime.now(tz=IST)
    from_dt = now - timedelta(days=warmup_days)
    bars_raw = historical_bars(token, bar_interval, from_dt=from_dt, to_dt=now)
    bars = [
        Bar(
            ts=b["date"].astimezone(IST) if hasattr(b["date"], "astimezone") else b["date"],
            open=float(b["open"]), high=float(b["high"]),
            low=float(b["low"]), close=float(b["close"]),
            volume=int(b["volume"]),
        )
        for b in bars_raw
    ]
    # Resolve expiry by offset (0 = current week, 1 = next week, etc.)
    selected_expiry = None
    if expiry_offset > 0:
        expiries = list_expiries(symbol)
        if expiry_offset < len(expiries):
            selected_expiry = expiries[expiry_offset]
            print(f"  Using expiry-offset={expiry_offset}: {selected_expiry} "
                  f"(available: {[str(e) for e in expiries[:5]]})")
        else:
            print(f"  WARN: expiry_offset={expiry_offset} but only {len(expiries)} "
                  f"expiries available; falling back to nearest")

    chain = option_chain(symbol, expiry=selected_expiry, with_oi=with_oi)
    populate_ivs(chain)
    spot = chain.spot
    return MarketState(symbol=symbol, spot=spot, chain=chain, bars_5m=bars,
                       ts=now)


def _build_state_manual(symbol: str, spot: float, iv: float, dte: int,
                        bars_file: str | None, ts_override: datetime | None = None) -> MarketState:
    from .manual_chain import synthetic_chain
    chain = synthetic_chain(symbol=symbol, spot=spot, iv=iv, dte_days=dte)
    bars: list[Bar] = []
    if bars_file:
        data = json.loads(Path(bars_file).read_text())
        for b in data:
            ts = datetime.fromisoformat(b["ts"])
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=IST)
            bars.append(Bar(
                ts=ts, open=float(b["open"]), high=float(b["high"]),
                low=float(b["low"]), close=float(b["close"]),
                volume=int(b["volume"]),
            ))
    state = MarketState(symbol=symbol, spot=spot, chain=chain, bars_5m=bars,
                        ts=ts_override or datetime.now(tz=IST))
    return state


# --- Open-position monitor ---
# High-water-mark cache for trailing stops, keyed by position id.
# Lost on watcher restart (acceptable — trail re-arms on first profitable poll).
_HIGH_WATER: dict[str, float] = {}


def _monitor_open_positions(state: MarketState, in_mem_positions: list[dict]):
    """Check open paper positions for target/stop/timeout exits."""
    if state.chain is None:
        return
    by_strike = state.chain.by_strike()
    now = state.ts
    survivors = []

    db_positions = open_paper_positions()
    # Dedup by position id — same position can appear in both JSONL & in_mem
    seen_ids: set[str] = set()
    in_play: list[dict] = []
    for p in list(db_positions) + [p for p in in_mem_positions if p["status"] == "open"]:
        pid = str(p.get("id") or p.get("position_id") or "")
        if pid and pid in seen_ids:
            continue
        seen_ids.add(pid)
        in_play.append(p)
    if in_play:
        print(f"  [monitor] tracking {len(in_play)} open position(s):")
        for p in in_play:
            print(f"    {p.get('strike')}{p.get('option_type')} "
                  f"entry={p.get('entry_price')} "
                  f"target={p.get('planned_target')} "
                  f"stop={p.get('planned_stop')} "
                  f"id={p.get('id') or p.get('position_id')}")

    for pos in in_play:
        strike = pos["strike"]
        ot = pos["option_type"]
        q = by_strike.get(strike, {}).get(ot)
        if q is None or not q.ltp:
            print(f"    -> {strike}{ot}: NO QUOTE in chain — survivor")
            survivors.append(pos); continue
        current_premium = q.ltp
        entry_ts = (datetime.fromisoformat(pos["entry_ts"])
                    if isinstance(pos["entry_ts"], str) else pos["entry_ts"])
        if entry_ts.tzinfo is None:
            entry_ts = entry_ts.replace(tzinfo=IST)
        held_min = (now - entry_ts).total_seconds() / 60.0

        target = pos.get("planned_target")
        stop = pos.get("planned_stop")
        entry_px = float(pos["entry_price"])

        # Respect the rule's intended hold_minutes; default 90 only when unknown
        max_hold = pos.get("hold_minutes") or pos.get("planned_hold_minutes") or 90

        # --- Trailing-stop adjustment (if rule armed it) ---
        pid_for_trail = str(pos.get("id") or pos.get("position_id") or "")
        trail_act = pos.get("trail_activation_pts") or 0
        trail_dist = pos.get("trail_distance_pts") or 0
        effective_stop = float(stop) if stop is not None else None
        if trail_act > 0 and trail_dist > 0 and pid_for_trail:
            high = _HIGH_WATER.get(pid_for_trail, entry_px)
            if current_premium > high:
                high = current_premium
                _HIGH_WATER[pid_for_trail] = high
            gain = high - entry_px
            if gain >= trail_act:
                trail_stop = high - trail_dist
                if effective_stop is None or trail_stop > effective_stop:
                    effective_stop = trail_stop

        exit_reason = None
        if target is not None and current_premium >= float(target):
            exit_reason = "target"
        elif effective_stop is not None and current_premium <= effective_stop:
            exit_reason = "stop"
        elif held_min >= max_hold:
            exit_reason = "timeout"

        # --- Trend-hold extra exits: EMA flip + EOD ---
        if exit_reason is None and pos.get("exit_on_ema_flip"):
            entry_dir = pos.get("entry_direction")
            fast_n = pos.get("ema_fast_period") or 20
            slow_n = pos.get("ema_slow_period") or 50
            try:
                from .indicators import ema
                closes = [b.close for b in state.bars_5m]
                ef = ema(closes, fast_n); es = ema(closes, slow_n)
                if ef[-1] is not None and es[-1] is not None:
                    flipped = (
                        (entry_dir == "bearish" and ef[-1] > es[-1]) or
                        (entry_dir == "bullish" and ef[-1] < es[-1])
                    )
                    if flipped:
                        exit_reason = "ema_flip"
            except Exception:
                pass
        if exit_reason is None and pos.get("eod_close_minutes"):
            mso = state.minutes_since_open()
            if mso >= float(pos["eod_close_minutes"]):
                exit_reason = "eod"

        trail_note = ""
        if trail_act > 0 and effective_stop is not None and stop is not None and effective_stop > float(stop):
            trail_note = f" [trail-stop={effective_stop:.2f}, hw={_HIGH_WATER.get(pid_for_trail, entry_px):.2f}]"
        print(f"    -> {strike}{ot}: current={current_premium:.2f} "
              f"target={target} stop={stop}{trail_note} held={held_min:.1f}m/{max_hold}m "
              f"-> {exit_reason or 'no-exit'}")

        if exit_reason:
            pid = pos.get("id") or pos.get("position_id") or "in-mem"
            write_close(str(pid), current_premium, now, exit_reason,
                        lots=pos["lots"], lot_size=pos["lot_size"],
                        entry_premium=float(entry_px))
            pnl = (current_premium - float(entry_px)) * pos["lots"] * pos["lot_size"]
            print(f"  [paper CLOSE] {pos['strike']}{pos['option_type']}  reason={exit_reason}  "
                  f"entry={entry_px:.2f} exit={current_premium:.2f}  PnL=Rs {pnl:+,.0f}")
            pos["status"] = "closed"
        else:
            survivors.append(pos)
    in_mem_positions[:] = survivors


# --- Main loop ---
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=["kite", "manual"], required=True)
    p.add_argument("--symbol", default="NIFTY")
    p.add_argument("--rule", action="append", required=True,
                   help="Rule name (repeatable). Available: orb, simple_ema_cross, etc.")
    p.add_argument("--rule-config-file", action="append", default=None,
                   help="JSON config file for the matching --rule (repeatable)")
    p.add_argument("--interval", type=int, default=60, help="Poll interval (sec)")
    p.add_argument("--bar-interval", default="5minute",
                   help="Bar timeframe to fetch: 2minute, 3minute, 5minute, etc.")
    p.add_argument("--warmup-days", type=int, default=3,
                   help="Calendar days of bars to fetch for indicator warmup")
    p.add_argument("--use-futures-bars", action="store_true",
                   help="Use NIFTY/BANKNIFTY monthly futures bars (real volume) "
                        "for indicators. Spot still comes from the index. "
                        "Required for any volume-based filter to work.")
    p.add_argument("--expiry-offset", type=int, default=0,
                   help="Which expiry to trade: 0=current week, 1=next week, "
                        "2=week after, etc. Higher expiry = more premium, slower theta.")
    p.add_argument("--once", action="store_true", help="Single evaluation then exit")
    # manual-mode args
    p.add_argument("--spot", type=float)
    p.add_argument("--iv", type=float)
    p.add_argument("--dte", type=int, default=7)
    p.add_argument("--bars-file", type=str)
    p.add_argument("--ts", type=str, help="(manual) Override 'now' for replay; ISO format")
    args = p.parse_args()

    rules = _load_rules(args.rule, args.rule_config_file)
    in_mem_positions: list[dict] = []
    # Rules that need OI need a heavier kite.quote() call
    oi_rules = {"oi_direction"}
    needs_oi = any(r.name in oi_rules for r in rules)

    print(f"Watcher starting: source={args.source} rules={[r.name for r in rules]} "
          f"interval={args.interval}s  with_oi={needs_oi}")

    while True:
        try:
            if args.source == "kite":
                state = _build_state_kite(args.symbol, with_oi=needs_oi,
                                          bar_interval=args.bar_interval,
                                          warmup_days=args.warmup_days,
                                          use_futures_bars=args.use_futures_bars,
                                          expiry_offset=args.expiry_offset)
            else:
                if args.spot is None or args.iv is None:
                    print("manual mode requires --spot and --iv", file=sys.stderr); sys.exit(2)
                ts_override = datetime.fromisoformat(args.ts).astimezone(IST) if args.ts else None
                state = _build_state_manual(
                    args.symbol, args.spot, args.iv, args.dte,
                    args.bars_file, ts_override,
                )

            # Add quick EMA20/EMA50 readout if we have enough bars
            ema_info = ""
            if len(state.bars_5m) >= 52:
                from .indicators import ema
                closes = [b.close for b in state.bars_5m]
                ef = ema(closes, 20); es = ema(closes, 50)
                if ef[-1] is not None and es[-1] is not None:
                    gap = ef[-1] - es[-1]
                    direction = "bullish" if gap > 0 else "bearish"
                    ema_info = f"  EMA20={ef[-1]:.1f} EMA50={es[-1]:.1f} gap={gap:+.2f} ({direction})"
            print(f"\n[{state.ts.strftime('%H:%M:%S')}] spot={state.spot:.2f}  "
                  f"bars={len(state.bars_5m)}  "
                  f"min_since_open={state.minutes_since_open()}"
                  f"{ema_info}")

            # Monitor existing paper trades first
            _monitor_open_positions(state, in_mem_positions)

            # Evaluate rules
            for rule in rules:
                sig = rule.evaluate(state)
                if sig is None:
                    continue
                _print_signal(sig)
                _log_signal(sig)
                # Open paper position
                if sig.strike and sig.option_type:
                    entry_premium = state.chain.by_strike()[sig.strike][sig.option_type].ltp
                    # Inject chain expiry into signal's trigger_context so paper.py persists it.
                    # This fixes dashboard P&L lookup which previously used the nearest expiry.
                    if state.chain and state.chain.expiry:
                        sig.trigger_context["expiry"] = state.chain.expiry.isoformat()
                    pid = write_open(sig, entry_premium=entry_premium)
                    in_mem_positions.append({
                        "id": pid, "status": "open",
                        "strike": sig.strike, "option_type": sig.option_type,
                        "lots": 1, "lot_size": 75,
                        "entry_price": entry_premium,
                        "entry_ts": sig.ts.isoformat(),
                        "planned_target": entry_premium + (sig.target_premium_gain or 0),
                        "planned_stop": sig.stop_premium,
                        "planned_hold_minutes": sig.hold_minutes,
                        "trail_activation_pts": sig.trigger_context.get("trail_activation_pts"),
                        "trail_distance_pts": sig.trigger_context.get("trail_distance_pts"),
                        "exit_on_ema_flip": sig.trigger_context.get("exit_on_ema_flip", False),
                        "entry_direction": sig.trigger_context.get("entry_direction"),
                        "ema_fast_period": sig.trigger_context.get("ema_fast_period"),
                        "ema_slow_period": sig.trigger_context.get("ema_slow_period"),
                        "eod_close_minutes": sig.trigger_context.get("eod_close_minutes"),
                    })
                    print(f"  [paper OPEN] {sig.strike}{sig.option_type} @ {entry_premium:.2f}  id={pid}")

            if args.once:
                break
            time_mod.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nShutting down.")
            break
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}", file=sys.stderr)
            if args.once:
                raise
            time_mod.sleep(args.interval)


if __name__ == "__main__":
    main()
