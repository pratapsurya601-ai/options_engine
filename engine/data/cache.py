"""
Permanent historical bar cache.

Pulls NIFTY/BANKNIFTY OHLC from Kite ONCE per (symbol, interval), stores to
parquet on disk, and serves all future backtests from the cache. No more
credit burn for repeated strategy iteration.

Layout:
  data/cache/
    NIFTY_minute.parquet         # all 1-min bars ever pulled
    NIFTY_5minute.parquet
    NIFTY_day.parquet
    BANKNIFTY_minute.parquet
    ...

Incremental pulls: if cache has data through 2026-04-01 and you ask for 90 days
from today, only the missing 2-month window is pulled. Kite's 60-day chunk
limit is handled automatically.

CLI:
  python -m engine.data.cache pull --interval minute --days 120
  python -m engine.data.cache pull --interval 5minute --days 180
  python -m engine.data.cache status
  python -m engine.data.cache clear --interval minute   # nuke one cache
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd


IST = timezone(timedelta(hours=5, minutes=30))
CACHE_DIR = Path("data/cache")

# Kite historical_data chunk limits (days per call). Empirically:
# - minute: ~60 days
# - 3minute: ~100 days
# - 5minute / 10minute / 15minute: ~200 days
# - day: ~2000 days
_CHUNK_DAYS = {
    # Per Kite Connect historical_data limits (slightly conservative)
    "minute": 55,
    "3minute": 95,
    "5minute": 95,
    "10minute": 95,
    "15minute": 180,
    "30minute": 180,
    "60minute": 380,
    "day": 1800,
}


def _cache_path(symbol: str, interval: str) -> Path:
    return CACHE_DIR / f"{symbol}_{interval}.parquet"


def load_cache(symbol: str, interval: str) -> pd.DataFrame:
    path = _cache_path(symbol, interval)
    if not path.exists():
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])
    df = pd.read_parquet(path)
    if not df.empty and df["ts"].dt.tz is None:
        df["ts"] = df["ts"].dt.tz_localize(IST)
    return df


def save_cache(symbol: str, interval: str, df: pd.DataFrame) -> None:
    path = _cache_path(symbol, interval)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def _kite_token_for(symbol: str) -> int:
    from .kite_source import NIFTY_SPOT_TOKEN, BANKNIFTY_SPOT_TOKEN
    if symbol == "NIFTY":
        return NIFTY_SPOT_TOKEN
    if symbol == "BANKNIFTY":
        return BANKNIFTY_SPOT_TOKEN
    raise ValueError(f"Unknown symbol: {symbol}")


def _pull_kite_range(kite, token: int, start: datetime, end: datetime,
                    interval: str) -> list[dict]:
    """Chunked pull respecting Kite's per-call day limit."""
    chunk_days = _CHUNK_DAYS.get(interval, 60)
    out: list[dict] = []
    current = start
    while current < end:
        chunk_end = min(current + timedelta(days=chunk_days), end)
        print(f"  pulling {interval} {current.date()} -> {chunk_end.date()}",
              file=sys.stderr)
        raw = kite.historical_data(token, current, chunk_end, interval)
        for b in raw:
            ts = b["date"]
            if hasattr(ts, "astimezone"):
                ts = ts.astimezone(IST)
            out.append({
                "ts": pd.Timestamp(ts),
                "open": float(b["open"]),
                "high": float(b["high"]),
                "low": float(b["low"]),
                "close": float(b["close"]),
                "volume": int(b["volume"]),
            })
        current = chunk_end + timedelta(seconds=1)
    return out


def pull_and_cache(symbol: str, interval: str, days: int,
                   force_refresh: bool = False) -> pd.DataFrame:
    """
    Ensure the cache covers the last `days` for (symbol, interval). Pulls only
    the missing portion. Returns the full cached DataFrame.

    If the cache already covers the requested range, Kite is NOT touched —
    so backtests run offline once data is cached.
    """
    end = datetime.now(tz=IST)
    requested_start = end - timedelta(days=days + 5)

    existing = pd.DataFrame() if force_refresh else load_cache(symbol, interval)

    pulls: list[tuple[datetime, datetime]] = []
    if existing.empty:
        pulls.append((requested_start, end))
    else:
        cached_start = existing["ts"].min().to_pydatetime()
        cached_end = existing["ts"].max().to_pydatetime()
        if requested_start < cached_start:
            pulls.append((requested_start, cached_start))
        # Only consider extending forward if market session has likely added data
        # since cache. Skip outside-hours auto-refresh to avoid Kite calls when offline.
        now_t = end.time()
        from datetime import time as dtime
        market_likely_running = dtime(9, 15) <= now_t <= dtime(15, 45)
        if market_likely_running and end > cached_end + timedelta(minutes=10):
            pulls.append((cached_end + timedelta(seconds=1), end))

    if not pulls:
        return existing

    # Only authenticate to Kite when we actually need to pull
    from .kite_source import get_kite
    try:
        kite = get_kite()
    except Exception as e:
        print(f"  WARN: Kite unavailable ({e}); returning cached data only "
              f"({len(existing)} rows)", file=sys.stderr)
        return existing

    token = _kite_token_for(symbol)

    new_rows: list[dict] = []
    for ps, pe in pulls:
        new_rows.extend(_pull_kite_range(kite, token, ps, pe, interval))

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        if existing.empty:
            combined = new_df
        else:
            combined = (
                pd.concat([existing, new_df], ignore_index=True)
                  .drop_duplicates(subset=["ts"])
                  .sort_values("ts")
                  .reset_index(drop=True)
            )
    else:
        combined = existing

    save_cache(symbol, interval, combined)
    print(f"  cache saved: {len(combined)} rows total, "
          f"{combined['ts'].min()} -> {combined['ts'].max()}", file=sys.stderr)
    return combined


def get_bars(symbol: str, interval: str, days: int,
             force_refresh: bool = False) -> list:
    """
    Returns a list[Bar] for backtest. Pulls from cache; refreshes only if
    missing range or force_refresh=True.
    """
    from ..state import Bar
    df = pull_and_cache(symbol, interval, days, force_refresh=force_refresh)
    if df.empty:
        return []
    end = datetime.now(tz=IST)
    start = end - timedelta(days=days + 5)
    sliced = df[(df["ts"] >= pd.Timestamp(start)) & (df["ts"] <= pd.Timestamp(end))]
    return [
        Bar(
            ts=r["ts"].to_pydatetime() if hasattr(r["ts"], "to_pydatetime") else r["ts"],
            open=float(r["open"]), high=float(r["high"]),
            low=float(r["low"]), close=float(r["close"]),
            volume=int(r["volume"]),
        )
        for _, r in sliced.iterrows()
    ]


# -------- CLI --------

def _cmd_pull(args):
    df = pull_and_cache(args.symbol, args.interval, args.days,
                        force_refresh=args.refresh)
    print(f"Cached {len(df)} {args.interval} bars for {args.symbol}")
    if not df.empty:
        print(f"Range: {df['ts'].min()} -> {df['ts'].max()}")


def _cmd_status(args):
    if not CACHE_DIR.exists():
        print(f"No cache at {CACHE_DIR}")
        return
    files = sorted(CACHE_DIR.glob("*.parquet"))
    if not files:
        print(f"Cache dir exists but empty: {CACHE_DIR}")
        return
    print(f"{'File':<32} {'Rows':>8} {'First bar':<26} {'Last bar':<26} {'Size':>10}")
    print("-" * 110)
    for f in files:
        df = pd.read_parquet(f)
        size_kb = f.stat().st_size / 1024
        first = str(df["ts"].min()) if not df.empty else "—"
        last = str(df["ts"].max()) if not df.empty else "—"
        print(f"{f.name:<32} {len(df):>8} {first:<26} {last:<26} {size_kb:>8.0f}KB")


def _cmd_clear(args):
    path = _cache_path(args.symbol, args.interval)
    if path.exists():
        path.unlink()
        print(f"Deleted {path}")
    else:
        print(f"No cache file at {path}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("pull", help="Pull bars into cache (incremental)")
    pp.add_argument("--symbol", default="NIFTY")
    pp.add_argument("--interval", required=True,
                    help="minute, 3minute, 5minute, 10minute, 15minute, 30minute, 60minute, day")
    pp.add_argument("--days", type=int, default=120,
                    help="how many days back from today to ensure cached")
    pp.add_argument("--refresh", action="store_true",
                    help="discard existing cache and refetch entire range")
    pp.set_defaults(func=_cmd_pull)

    ss = sub.add_parser("status", help="Show cache contents")
    ss.set_defaults(func=_cmd_status)

    cc = sub.add_parser("clear", help="Delete cache for (symbol, interval)")
    cc.add_argument("--symbol", default="NIFTY")
    cc.add_argument("--interval", required=True)
    cc.set_defaults(func=_cmd_clear)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
