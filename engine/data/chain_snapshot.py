"""
Option chain snapshot daemon.

Polls the live option chain via Kite at regular intervals during market hours
and persists every snapshot to:
  1. Local parquet (data/chain_snapshots/{symbol}/{YYYY-MM-DD}/{HHMMSS}.parquet)
  2. Aiven Postgres (option_chain_snapshots table), if DATABASE_URL is set

After 1-2 months of runtime you'll have REAL historical intraday option premium /
IV / OI data — backtests can use real premiums instead of Black-Scholes synthetic.

Run modes:
  python -m engine.data.chain_snapshot run --interval-min 5
  python -m engine.data.chain_snapshot run --once               # single capture, exit
  python -m engine.data.chain_snapshot run --once --aiven-only  # CI: no parquet
  python -m engine.data.chain_snapshot run --once --no-aiven    # local only
  python -m engine.data.chain_snapshot run --once --force       # skip market-hours check
  python -m engine.data.chain_snapshot status                   # local parquet stats
  python -m engine.data.chain_snapshot aiven-status             # Aiven row counts
  python -m engine.data.chain_snapshot read --date 2026-06-02 --strike 23500

Env vars:
  DATABASE_URL       — Aiven Postgres connection string (optional; enables Aiven write)
  KITE_API_KEY       — required for Kite auth
  KITE_ACCESS_TOKEN  — optional; if set, bypasses ~/.kite_token.json (CI mode)

Honest design: the daemon NEVER modifies existing snapshots — strictly append-only.
Local parquet and Aiven writes are independent; one failure doesn't block the other.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, time as dtime, timedelta, timezone, date
from pathlib import Path

import pandas as pd


IST = timezone(timedelta(hours=5, minutes=30))
SNAPSHOT_ROOT = Path("data/chain_snapshots")


def _market_open() -> bool:
    now = datetime.now(tz=IST)
    if now.weekday() >= 5:
        return False
    t = now.time()
    return dtime(9, 15) <= t <= dtime(15, 30)


def _snapshot_dir(symbol: str, day: date) -> Path:
    return SNAPSHOT_ROOT / symbol / day.isoformat()


def _snapshot_path(symbol: str, ts: datetime) -> Path:
    fn = ts.strftime("%H%M%S") + ".parquet"
    return _snapshot_dir(symbol, ts.date()) / fn


def _load_prev_snapshot(symbol: str, ts: datetime) -> pd.DataFrame | None:
    """Find the most recent prior snapshot for oi_change computation. Same day only."""
    day_dir = _snapshot_dir(symbol, ts.date())
    if not day_dir.exists():
        return None
    files = sorted(day_dir.glob("*.parquet"))
    if not files:
        return None
    # All files BEFORE the current ts. Filenames are HHMMSS sorted ascending.
    cur_name = ts.strftime("%H%M%S")
    prior = [f for f in files if f.stem < cur_name]
    if not prior:
        return None
    try:
        return pd.read_parquet(prior[-1])
    except Exception:
        return None


def _compute_oi_change(df: pd.DataFrame, prev_df: pd.DataFrame | None) -> pd.DataFrame:
    """Add oi_change column = current oi - prev oi per (expiry, strike, option_type).
    NaN where no match in prev_df or prev_df is None. Returns a copy with added column."""
    out = df.copy()
    if prev_df is None or prev_df.empty:
        out["oi_change"] = pd.NA
        return out
    keys = ["expiry", "strike", "option_type"]
    prev_slim = prev_df[keys + ["oi"]].rename(columns={"oi": "_prev_oi"})
    merged = out.merge(prev_slim, on=keys, how="left")
    merged["oi_change"] = merged["oi"] - merged["_prev_oi"]
    merged = merged.drop(columns=["_prev_oi"])
    return merged


def _sanitize_for_db(df: pd.DataFrame) -> pd.DataFrame:
    """Defensive conversion before Postgres insert.
    - BIGINT columns: pd.NA / NaN / out-of-range -> None
    - NUMERIC columns: pd.NA / NaN / inf -> None
    Pandas float64 columns with NaN can leak into psycopg as float('nan'),
    which Postgres rejects on a BIGINT column with 'bigint out of range'.
    Forcing object dtype with explicit None breaks that path."""
    import math
    out = df.copy()
    BIGINT_MAX = 2**63 - 1
    BIGINT_MIN = -(2**63)

    def safe_int(v):
        if v is None:
            return None
        try:
            if pd.isna(v):
                return None
        except (TypeError, ValueError):
            return None
        try:
            iv = int(v)
        except (TypeError, ValueError, OverflowError):
            return None
        if BIGINT_MIN <= iv <= BIGINT_MAX:
            return iv
        return None

    def safe_float(v):
        if v is None:
            return None
        try:
            if pd.isna(v):
                return None
        except (TypeError, ValueError):
            return None
        try:
            fv = float(v)
        except (TypeError, ValueError):
            return None
        if math.isnan(fv) or math.isinf(fv):
            return None
        return fv

    for col in ("oi", "volume", "oi_change"):
        if col in out.columns:
            out[col] = out[col].map(safe_int).astype(object)
    for col in ("ltp", "bid", "ask", "iv", "spot"):
        if col in out.columns:
            out[col] = out[col].map(safe_float).astype(object)
    return out


def _write_to_aiven(df: pd.DataFrame) -> tuple[int | None, str | None]:
    """Write snapshot rows to Aiven Postgres. Returns (rows_written, error).
    Returns (None, None) if DATABASE_URL not set. Returns (None, error) on failure."""
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        return None, None
    try:
        from db.writer import get_conn, write_option_chain
        clean_df = _sanitize_for_db(df)
        with get_conn(database_url) as conn:
            n = write_option_chain(conn, clean_df)
        return n, None
    except Exception as e:
        return None, str(e)


def _capture_futures_and_vix(symbol: str, snapshot_ts: datetime,
                              spot: float) -> tuple[int, int, list[str]]:
    """Fetch + write futures snapshots and India VIX for this 5-min tick.

    Best-effort: failures here NEVER abort the option-chain capture.
    Returns (futures_rows_written, vix_rows_written, errors)."""
    from .kite_source import futures_quotes, india_vix_ltp
    errors: list[str] = []
    futures_written = 0
    vix_written = 0

    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        return 0, 0, []

    # --- Futures ---
    try:
        futs = futures_quotes(symbol)
        if futs:
            rows = []
            for f in futs:
                ltp = f.get("ltp")
                if not ltp or not spot:
                    continue
                basis = ltp - spot
                basis_pct = (basis / spot * 100.0) if spot else None
                rows.append({
                    "symbol": symbol,
                    "snapshot_ts": snapshot_ts,
                    "expiry": f["expiry"],
                    "ltp": ltp,
                    "bid": f.get("bid"),
                    "ask": f.get("ask"),
                    "volume": f.get("volume"),
                    "oi": f.get("oi"),
                    "oi_change": None,
                    "spot": spot,
                    "basis": basis,
                    "basis_pct": basis_pct,
                })
            if rows:
                from db.writer import get_conn, write_futures_snapshots
                fdf = _sanitize_for_db(pd.DataFrame(rows))
                with get_conn(database_url) as conn:
                    futures_written = write_futures_snapshots(conn, fdf)
    except Exception as e:
        errors.append(f"futures: {type(e).__name__}: {e}")

    # --- VIX ---
    try:
        vix = india_vix_ltp()
        if vix is not None:
            from db.writer import get_conn, write_vix_snapshot
            with get_conn(database_url) as conn:
                vix_written = write_vix_snapshot(conn, snapshot_ts, vix)
    except Exception as e:
        errors.append(f"vix: {type(e).__name__}: {e}")

    return futures_written, vix_written, errors


def take_snapshot(
    symbol: str = "NIFTY",
    with_oi: bool = True,
    write_parquet: bool = True,
    write_aiven: bool | None = None,
) -> dict:
    """Pull one chain snapshot.

    write_aiven: True forces Aiven write (errors if DATABASE_URL unset).
                 False skips Aiven entirely.
                 None auto-detects: writes if DATABASE_URL is set.

    Returns dict with: parquet_path (Path|None), aiven_rows_written (int|None),
    error (str|None), n_rows (int).
    """
    from .kite_source import option_chain, populate_ivs
    result = {
        "parquet_path": None,
        "aiven_rows_written": None,
        "error": None,
        "n_rows": 0,
    }
    try:
        chain = option_chain(symbol, with_oi=with_oi)
        populate_ivs(chain)
    except Exception as e:
        msg = f"[{datetime.now(tz=IST).strftime('%H:%M:%S')}] snapshot fail: {e}"
        print(msg, file=sys.stderr)
        result["error"] = str(e)
        return result

    ts = datetime.now(tz=IST)
    rows = []
    for q in chain.quotes:
        rows.append({
            "snapshot_ts": ts,
            "symbol": symbol,
            "spot": chain.spot,
            "expiry": q.expiry,
            "strike": q.strike,
            "option_type": q.option_type,
            "ltp": q.ltp,
            "bid": q.bid,
            "ask": q.ask,
            "iv": q.iv,
            "oi": q.oi,
            "volume": q.volume,
        })
    if not rows:
        return result

    df = pd.DataFrame(rows)
    result["n_rows"] = len(df)

    # Compute oi_change from prior snapshot (parquet-based lookup is cheap)
    prev_df = _load_prev_snapshot(symbol, ts)
    df = _compute_oi_change(df, prev_df)

    # --- Write parquet
    if write_parquet:
        try:
            path = _snapshot_path(symbol, ts)
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(path, index=False)
            result["parquet_path"] = path
        except Exception as e:
            print(f"[{ts.strftime('%H:%M:%S')}] parquet write failed: {e}",
                  file=sys.stderr)
            # Don't abort — Aiven write may still succeed

    # --- Write Aiven
    do_aiven = write_aiven if write_aiven is not None else bool(os.environ.get("DATABASE_URL"))
    if do_aiven:
        n_written, err = _write_to_aiven(df)
        result["aiven_rows_written"] = n_written
        if err:
            # Log but DON'T fail — parquet still saved (unless aiven-only)
            print(f"[{ts.strftime('%H:%M:%S')}] Aiven write FAILED: {err}",
                  file=sys.stderr)
            if not write_parquet:
                # In aiven-only mode, propagate the error
                result["error"] = f"Aiven write failed: {err}"
        else:
            print(f"[{ts.strftime('%H:%M:%S')}] Aiven write: {n_written} rows OK",
                  flush=True)

        # Futures + VIX best-effort (never aborts the snapshot)
        fut_n, vix_n, fv_errs = _capture_futures_and_vix(symbol, ts, chain.spot)
        result["futures_rows_written"] = fut_n
        result["vix_rows_written"] = vix_n
        if fv_errs:
            for e in fv_errs:
                print(f"[{ts.strftime('%H:%M:%S')}] aux: {e}", file=sys.stderr)
        print(f"[{ts.strftime('%H:%M:%S')}] aux write: futures={fut_n} vix={vix_n}",
              flush=True)

    return result


def run_daemon(
    symbol: str = "NIFTY",
    interval_min: int = 5,
    with_oi: bool = True,
    max_minutes: int | None = None,
    aiven_only: bool = False,
    no_aiven: bool = False,
) -> None:
    """Continuously take snapshots every `interval_min` minutes during market hours.

    aiven_only: skip parquet writes (CI/cloud mode)
    no_aiven:   skip Aiven writes (local-only mode)
    """
    write_parquet = not aiven_only
    write_aiven = False if no_aiven else None  # None = auto via DATABASE_URL

    print(f"Chain snapshot daemon started "
          f"(symbol={symbol}, interval={interval_min}min, with_oi={with_oi}, "
          f"parquet={write_parquet}, aiven={'auto' if write_aiven is None else write_aiven})",
          flush=True)
    start = datetime.now(tz=IST)
    while True:
        if max_minutes is not None:
            elapsed = (datetime.now(tz=IST) - start).total_seconds() / 60
            if elapsed >= max_minutes:
                print(f"Reached max_minutes={max_minutes}, exiting.")
                break

        if _market_open():
            try:
                r = take_snapshot(symbol, with_oi=with_oi,
                                  write_parquet=write_parquet,
                                  write_aiven=write_aiven)
                if r["parquet_path"]:
                    size_kb = r["parquet_path"].stat().st_size / 1024
                    ts = datetime.now(tz=IST).strftime("%H:%M:%S")
                    print(f"[{ts}] saved {r['parquet_path'].name} ({size_kb:.0f} KB)",
                          flush=True)
            except KeyboardInterrupt:
                print("Interrupted by user.")
                return
            except Exception as e:
                print(f"snapshot error: {e}", file=sys.stderr, flush=True)
            time.sleep(interval_min * 60)
        else:
            now = datetime.now(tz=IST)
            if now.time() < dtime(9, 15) and now.weekday() < 5:
                next_open = datetime.combine(now.date(), dtime(9, 15), tzinfo=IST)
            else:
                days_ahead = 1
                while True:
                    candidate = (now + timedelta(days=days_ahead))
                    if candidate.weekday() < 5:
                        next_open = datetime.combine(
                            candidate.date(), dtime(9, 15), tzinfo=IST
                        )
                        break
                    days_ahead += 1
            sleep_sec = max((next_open - now).total_seconds(), 60)
            print(f"Market closed; sleeping until {next_open} "
                  f"({sleep_sec/60:.0f} min)", flush=True)
            time.sleep(min(sleep_sec, 3600))


def list_snapshots(symbol: str = "NIFTY") -> list[Path]:
    sym_root = SNAPSHOT_ROOT / symbol
    if not sym_root.exists():
        return []
    return sorted(sym_root.glob("*/*.parquet"))


def read_snapshot(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def read_nearest(symbol: str, target: datetime,
                 max_delta_min: int = 10) -> pd.DataFrame | None:
    sym_day = _snapshot_dir(symbol, target.date())
    if not sym_day.exists():
        return None
    candidates = sorted(sym_day.glob("*.parquet"))
    if not candidates:
        return None
    best = None
    best_delta = float("inf")
    for f in candidates:
        try:
            t = datetime.strptime(f.stem, "%H%M%S").time()
            file_ts = datetime.combine(target.date(), t, tzinfo=IST)
            delta = abs((file_ts - target).total_seconds())
            if delta < best_delta:
                best_delta = delta
                best = f
        except ValueError:
            continue
    if best is None or best_delta > max_delta_min * 60:
        return None
    return pd.read_parquet(best)


# ---- CLI handlers ----

def cli_status(args):
    """Print summary of cached parquet snapshots (local)."""
    if not SNAPSHOT_ROOT.exists():
        print(f"No snapshots yet at {SNAPSHOT_ROOT}")
        return
    for sym_dir in sorted(SNAPSHOT_ROOT.iterdir()):
        if not sym_dir.is_dir():
            continue
        total_files = 0
        total_size = 0
        days = []
        for day_dir in sorted(sym_dir.iterdir()):
            if not day_dir.is_dir():
                continue
            day_files = list(day_dir.glob("*.parquet"))
            if not day_files:
                continue
            day_size = sum(f.stat().st_size for f in day_files)
            total_files += len(day_files)
            total_size += day_size
            days.append((day_dir.name, len(day_files), day_size / 1024))
        print(f"\n{sym_dir.name}: {total_files} snapshots, "
              f"{total_size / (1024*1024):.1f} MB across {len(days)} days")
        for d, n, kb in days[-10:]:
            print(f"  {d}: {n} snapshots ({kb:.0f} KB)")
        if len(days) > 10:
            print(f"  ... and {len(days) - 10} more days")


def cli_aiven_status(args):
    """Query Aiven Postgres for row counts and date coverage."""
    url = os.environ.get("DATABASE_URL")
    if not url:
        print("DATABASE_URL not set", file=sys.stderr)
        sys.exit(1)
    try:
        from db.writer import get_conn
        with get_conn(url) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT symbol,
                           COUNT(*) AS total_rows,
                           COUNT(DISTINCT snapshot_ts) AS n_snapshots,
                           COUNT(DISTINCT DATE(snapshot_ts AT TIME ZONE 'Asia/Kolkata')) AS days,
                           MIN(snapshot_ts) AS oldest,
                           MAX(snapshot_ts) AS newest
                    FROM option_chain_snapshots
                    GROUP BY symbol
                    ORDER BY symbol
                """)
                rows = cur.fetchall()
        if not rows:
            print("Aiven: option_chain_snapshots table exists but is empty.")
            return
        for r in rows:
            print(f"{r['symbol']}: {r['total_rows']:,} rows, "
                  f"{r['n_snapshots']:,} snapshots, {r['days']} days "
                  f"({r['oldest']} → {r['newest']})")
    except Exception as e:
        print(f"Aiven query failed: {e}", file=sys.stderr)
        sys.exit(1)


def cli_run(args):
    if getattr(args, "once", False):
        # Single-shot mode (for CI / GitHub Actions)
        if not args.force and not _market_open():
            print(f"[{datetime.now(tz=IST).strftime('%H:%M:%S')}] "
                  f"Market closed, exiting (use --force to override).",
                  flush=True)
            sys.exit(0)
        if args.aiven_only and not os.environ.get("DATABASE_URL"):
            print("--aiven-only requires DATABASE_URL env var", file=sys.stderr)
            sys.exit(2)
        write_parquet = not args.aiven_only
        write_aiven = False if args.no_aiven else None
        r = take_snapshot(args.symbol, with_oi=not args.no_oi,
                          write_parquet=write_parquet,
                          write_aiven=write_aiven)
        ts = datetime.now(tz=IST).strftime("%H:%M:%S")
        if r["error"] and not r["n_rows"]:
            print(f"[{ts}] snapshot FAILED: {r['error']}", file=sys.stderr)
            sys.exit(1)
        parquet_str = f"parquet={r['parquet_path'].name}" if r["parquet_path"] else "parquet=skip"
        aiven_str = (f"aiven={r['aiven_rows_written']} rows"
                     if r["aiven_rows_written"] is not None else "aiven=skip")
        print(f"[{ts}] snapshot OK: {r['n_rows']} rows, {parquet_str}, {aiven_str}",
              flush=True)
        sys.exit(0)
    else:
        run_daemon(symbol=args.symbol, interval_min=args.interval_min,
                   with_oi=not args.no_oi, max_minutes=args.max_minutes,
                   aiven_only=args.aiven_only, no_aiven=args.no_aiven)


def cli_read(args):
    target = datetime.fromisoformat(args.date + "T" + (args.time or "12:00:00") + "+05:30")
    df = read_nearest(args.symbol, target, max_delta_min=args.max_delta_min)
    if df is None:
        print(f"No snapshot found within {args.max_delta_min} min of {target}")
        return
    if args.strike is not None:
        df = df[df["strike"] == args.strike]
    print(df.to_string(index=False))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run", help="Run daemon (continuous) or --once (single capture)")
    pr.add_argument("--symbol", default="NIFTY")
    pr.add_argument("--interval-min", type=int, default=5)
    pr.add_argument("--no-oi", action="store_true",
                    help="Skip OI (uses kite.ltp instead of kite.quote — cheaper)")
    pr.add_argument("--max-minutes", type=int, default=None,
                    help="Stop after this many minutes (default: run forever)")
    pr.add_argument("--once", action="store_true",
                    help="Take exactly ONE snapshot and exit. Critical for CI.")
    pr.add_argument("--force", action="store_true",
                    help="Skip market-hours check (only with --once)")
    pr.add_argument("--aiven-only", action="store_true",
                    help="Skip parquet writes (CI/cloud mode)")
    pr.add_argument("--no-aiven", action="store_true",
                    help="Skip Aiven writes (local-only mode)")
    pr.set_defaults(func=cli_run)

    ps = sub.add_parser("status", help="Show cached parquet snapshot stats (local)")
    ps.set_defaults(func=cli_status)

    pa = sub.add_parser("aiven-status", help="Query row counts from Aiven Postgres")
    pa.set_defaults(func=cli_aiven_status)

    pd_ = sub.add_parser("read", help="Read snapshot near a timestamp (local)")
    pd_.add_argument("--symbol", default="NIFTY")
    pd_.add_argument("--date", required=True, help="YYYY-MM-DD")
    pd_.add_argument("--time", default=None, help="HH:MM:SS (IST), default 12:00:00")
    pd_.add_argument("--strike", type=int, default=None)
    pd_.add_argument("--max-delta-min", type=int, default=10)
    pd_.set_defaults(func=cli_read)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
