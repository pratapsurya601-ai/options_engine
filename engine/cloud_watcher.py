"""
Stateless cloud watcher — designed for GitHub Actions cron invocation.

Every 5 minutes during market hours, this module:
  1. Loads recent bars from Kite (last ~4 hours for indicator warmup)
  2. Loads open paper positions from Aiven Postgres
  3. Loads cooldown state from Aiven
  4. Evaluates the configured rule against the market state
  5. Records any new signal and (optionally) opens a paper position
  6. Manages exits / trailing stops on existing open positions
  7. Writes everything back to Aiven and exits

Stateless = every run starts fresh. No in-memory caches. Aiven is the
single source of truth so the watcher can run from any node, anywhere.

CLI:
  python -m engine.cloud_watcher --rule htf_naked \
      --config configs/htf_naked/02_15min_itm.json \
      --symbol NIFTY

Env vars required:
  KITE_API_KEY
  KITE_ACCESS_TOKEN     (refreshed daily by refresh_kite_token.yml)
  DATABASE_URL          (Aiven Postgres)

Exits 0 always (failures logged + recorded to watcher_runs but don't fail
the workflow — we don't want one bad rule to break others in matrix).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from datetime import datetime, time as dtime, timedelta, timezone
from pathlib import Path
from typing import Any

# Aiven state interface
from db import cloud_state as cs

IST = timezone(timedelta(hours=5, minutes=30))

# ---- Rule registry (extend here as more rules graduate to cloud) ----
def _make_rule(rule_name: str, config: dict):
    """Construct a rule instance by name."""
    if rule_name == "htf_naked":
        from .rules.htf_naked import HtfNaked
        return HtfNaked(**config)
    if rule_name == "nifty_intraday_buyer":
        from .rules.nifty_intraday_buyer import NiftyIntradayBuyer
        return NiftyIntradayBuyer(**config)
    if rule_name == "panic_bounce_ce":
        from .rules.panic_bounce_ce import PanicBounceCE
        return PanicBounceCE(**config)
    raise ValueError(f"Unknown rule for cloud watcher: {rule_name}")


def _market_open(now: datetime) -> bool:
    if now.weekday() >= 5:
        return False
    return dtime(9, 15) <= now.time() <= dtime(15, 30)


def _resample_5min_to(bars: list, interval_min: int) -> list:
    """Resample a list of 5-minute Bar objects to `interval_min` bars.
    Returns a list of Bar with same dataclass shape."""
    from .state import Bar
    if interval_min == 5 or len(bars) == 0:
        return bars
    step = interval_min // 5
    if step < 1:
        return bars
    out = []
    for i in range(0, len(bars) - step + 1, step):
        chunk = bars[i:i + step]
        if not chunk:
            continue
        out.append(Bar(
            ts=chunk[0].ts,
            open=chunk[0].open,
            high=max(b.high for b in chunk),
            low=min(b.low for b in chunk),
            close=chunk[-1].close,
            volume=sum((b.volume or 0) for b in chunk),
        ))
    return out


def _load_recent_bars(symbol: str, interval_min: int,
                       lookback_min: int = 300) -> list:
    """Fetch recent bars from Kite. Returns list of Bar.
    We always request 5-min bars and resample to the target interval to
    keep the Kite call uniform across rules."""
    from .data.kite_source import historical_bars, nearest_future_token
    from .state import Bar

    token, _, _ = nearest_future_token(symbol)
    now = datetime.now(tz=IST)
    # Session start today
    session_start = datetime.combine(now.date(), dtime(9, 15), tzinfo=IST)
    from_dt = max(session_start, now - timedelta(minutes=lookback_min))
    raw = historical_bars(token, interval="5minute", from_dt=from_dt, to_dt=now)
    bars_5m = [
        Bar(ts=r["date"], open=float(r["open"]), high=float(r["high"]),
            low=float(r["low"]), close=float(r["close"]),
            volume=int(r.get("volume", 0) or 0))
        for r in raw
    ]
    return _resample_5min_to(bars_5m, interval_min)


def _build_state(symbol: str, bars: list):
    """Build a MarketState from bars. Mirrors what engine.watcher does."""
    from .state import MarketState
    from .data.kite_source import spot_ltp
    spot = spot_ltp(symbol)
    return MarketState(
        symbol=symbol,
        spot=spot,
        bars=bars,
        chain=None,  # cloud_watcher only loads chain on-demand
    )


def _get_chain(symbol: str, expiry=None):
    """Fetch current option chain for the given symbol (with OI)."""
    from .data.kite_source import option_chain, populate_ivs
    chain = option_chain(symbol, expiry=expiry, with_oi=True)
    populate_ivs(chain)
    return chain


def _current_premium(chain, strike: int, option_type: str) -> float | None:
    """Look up current LTP for a strike+type from a Chain object."""
    for q in chain.quotes:
        if q.strike == strike and q.option_type == option_type:
            return float(q.ltp) if q.ltp else None
    return None


def _process_signal(conn, *, signal, rule_name: str, symbol: str,
                    now: datetime, cooldown_min: int) -> tuple[int, int]:
    """Record the signal + open a paper position if appropriate.
    Returns (signals_fired_delta, positions_opened_delta)."""
    ctx = dict(signal.trigger_context or {})
    strike = ctx.get("strike")
    option_type = ctx.get("option_type")
    target_premium = ctx.get("target_premium")
    stop_premium = ctx.get("stop_premium")
    entry_premium = ctx.get("entry_premium") or ctx.get("premium")
    expiry = ctx.get("expiry")

    outcome = "opened_position"
    sig_id = cs.log_signal(
        conn,
        rule_name=rule_name, symbol=symbol, ts=now,
        spot=ctx.get("spot"), action=signal.action,
        strike=strike, expiry=expiry,
        premium=entry_premium, target_premium=target_premium,
        stop_premium=stop_premium, trigger_context=ctx,
        outcome=outcome,
    )
    if not (strike and option_type and entry_premium):
        # alert-only signal (e.g. iron condor); nothing to track in positions
        return 1, 0

    thesis = (
        f"{rule_name}: {signal.action} @ {entry_premium} | "
        f"target {target_premium} stop {stop_premium} | "
        f"trigger {json.dumps({k: v for k, v in ctx.items() if k not in ('chain',)})[:200]}"
    )
    cs.open_position(
        conn,
        symbol=symbol, expiry=expiry, strike=int(strike),
        option_type=option_type,
        action="BUY" if signal.action.startswith("BUY") else "SELL",
        lots=1, lot_size=75,
        entry_price=float(entry_premium), entry_ts=now,
        entry_spot=ctx.get("spot"), entry_iv=ctx.get("iv"),
        thesis=thesis, setup_tag=f"PAPER_{rule_name}", rule_name=rule_name,
        planned_stop=stop_premium, planned_target=target_premium,
        trail_activation_pts=ctx.get("trail_activation_pts"),
        trail_distance_pts=ctx.get("trail_distance_pts"),
        hold_until_ts=ctx.get("hold_until_ts"),
    )
    cs.record_fire(conn, rule_name, symbol, cooldown_min, ts=now)
    return 1, 1


def _check_exits(conn, *, open_positions, chain, now: datetime,
                 rule_name: str) -> tuple[int, int]:
    """For each open position, fetch current premium and check exit conditions.
    Returns (positions_closed_delta, trail_updated_delta)."""
    closed = 0
    trail_updates = 0
    for p in open_positions:
        cur = _current_premium(chain, p.strike, p.option_type)
        if cur is None:
            continue
        pnl = (cur - p.entry_price) * p.lots * p.lot_size

        exit_reason = None
        exit_price = cur

        # Hard target / stop
        if p.planned_target is not None and cur >= p.planned_target:
            exit_reason = "target_hit"
        elif p.planned_stop is not None and cur <= p.planned_stop:
            exit_reason = "stop_hit"

        # Trailing stop: if cur exceeds high_water_mark + activation,
        # lift stop to (hwm - trail_distance)
        if (exit_reason is None and p.trail_activation_pts is not None
                and p.trail_distance_pts is not None):
            hwm = max(p.high_water_mark or p.entry_price, cur)
            activation_threshold = p.entry_price + p.trail_activation_pts
            if hwm >= activation_threshold:
                new_stop = hwm - p.trail_distance_pts
                if (p.planned_stop is None or new_stop > p.planned_stop):
                    cs.update_high_water_mark(conn, p.id, hwm, new_stop=new_stop)
                    trail_updates += 1
                elif hwm > (p.high_water_mark or p.entry_price):
                    cs.update_high_water_mark(conn, p.id, hwm)
                    trail_updates += 1

        # Hold-until timeout
        if exit_reason is None and p.hold_until_ts is not None:
            hold_ts = p.hold_until_ts
            if hold_ts.tzinfo is None:
                hold_ts = hold_ts.replace(tzinfo=IST)
            if now >= hold_ts:
                exit_reason = "timeout"

        # EOD auto-close (15:25 IST safety margin)
        if exit_reason is None and now.time() >= dtime(15, 25):
            exit_reason = "eod_close"

        if exit_reason is not None:
            cs.close_position(
                conn, p.id, exit_price=exit_price, exit_ts=now,
                exit_reason=exit_reason, pnl=pnl,
            )
            closed += 1
    return closed, trail_updates


def run_once(rule_name: str, config: dict, symbol: str = "NIFTY",
             interval_min: int = 15, force: bool = False) -> dict:
    """Single watcher pass. Returns summary dict."""
    now = datetime.now(tz=IST)
    if not force and not _market_open(now):
        print(f"[{now.strftime('%H:%M:%S')}] market closed, skipping")
        return {"skipped": True}

    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL not set")

    summary = {
        "rule_name": rule_name, "symbol": symbol,
        "started_at": now.isoformat(),
        "bars_loaded": 0, "signals_fired": 0,
        "positions_opened": 0, "positions_closed": 0,
        "trail_updated": 0, "status": "ok", "error": None,
    }

    rule = _make_rule(rule_name, config)
    cooldown_min = getattr(rule, "cooldown_min", 0) or getattr(rule, "cooldown", 0) or 0

    conn = cs.get_conn(db_url)
    run_id = cs.start_watcher_run(conn, rule_name=rule_name, symbol=symbol, ts=now)

    try:
        # 1. Recent bars
        bars = _load_recent_bars(symbol, interval_min, lookback_min=300)
        summary["bars_loaded"] = len(bars)
        if len(bars) < 5:
            print(f"[{now.strftime('%H:%M:%S')}] insufficient bars ({len(bars)})")
            cs.finish_watcher_run(
                conn, run_id, bars_loaded=len(bars), signals_fired=0,
                positions_opened=0, positions_closed=0, trail_updated=0,
                status="ok_insufficient_bars",
            )
            return summary

        # 2. Build state
        state = _build_state(symbol, bars)

        # 3. Chain (needed both for new signal premium AND open-position exits)
        chain = _get_chain(symbol)
        state.chain = chain  # rules need access for premium lookup

        # 4. Existing open positions
        open_positions = cs.get_open_positions(
            conn, rule_name=rule_name, symbol=symbol)

        # 5. Exits first (so we free up cooldown when a position closes)
        closed, trail_updates = _check_exits(
            conn, open_positions=open_positions, chain=chain,
            now=now, rule_name=rule_name)
        summary["positions_closed"] = closed
        summary["trail_updated"] = trail_updates

        # 6. Cooldown check before evaluating
        if cs.is_in_cooldown(conn, rule_name, symbol, cooldown_min, now=now):
            print(f"[{now.strftime('%H:%M:%S')}] in cooldown, skip evaluate")
        else:
            # 7. Evaluate rule
            signal = None
            try:
                signal = rule.evaluate(state)
            except Exception as e:
                print(f"rule.evaluate() raised: {e}", file=sys.stderr)
                traceback.print_exc()

            if signal is not None:
                signals, opened = _process_signal(
                    conn, signal=signal, rule_name=rule_name, symbol=symbol,
                    now=now, cooldown_min=cooldown_min)
                summary["signals_fired"] += signals
                summary["positions_opened"] += opened

        cs.finish_watcher_run(
            conn, run_id,
            bars_loaded=summary["bars_loaded"],
            signals_fired=summary["signals_fired"],
            positions_opened=summary["positions_opened"],
            positions_closed=summary["positions_closed"],
            trail_updated=summary["trail_updated"],
            status="ok",
        )

    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        tb = traceback.format_exc()
        print(f"[{now.strftime('%H:%M:%S')}] watcher FAILED: {err}", file=sys.stderr)
        print(tb, file=sys.stderr)
        summary["status"] = "error"
        summary["error"] = err
        try:
            cs.finish_watcher_run(
                conn, run_id,
                bars_loaded=summary["bars_loaded"],
                signals_fired=summary["signals_fired"],
                positions_opened=summary["positions_opened"],
                positions_closed=summary["positions_closed"],
                trail_updated=summary["trail_updated"],
                status="error", error_message=err,
            )
        except Exception:
            pass
    finally:
        try:
            conn.close()
        except Exception:
            pass

    print(f"[{now.strftime('%H:%M:%S')}] {rule_name}: "
          f"bars={summary['bars_loaded']} fired={summary['signals_fired']} "
          f"opened={summary['positions_opened']} closed={summary['positions_closed']} "
          f"trail={summary['trail_updated']} status={summary['status']}")
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rule", required=True)
    p.add_argument("--config", required=True, help="Path to JSON config file")
    p.add_argument("--symbol", default="NIFTY")
    p.add_argument("--interval-min", type=int, default=15)
    p.add_argument("--force", action="store_true",
                   help="Skip market-hours check (for testing)")
    args = p.parse_args()

    config = json.loads(Path(args.config).read_text())
    interval = config.pop("__interval_min__", args.interval_min)
    summary = run_once(args.rule, config, symbol=args.symbol,
                       interval_min=interval, force=args.force)
    # Always exit 0 — don't fail the workflow on a single rule's error
    print(json.dumps(summary, default=str))
    sys.exit(0)


if __name__ == "__main__":
    main()
