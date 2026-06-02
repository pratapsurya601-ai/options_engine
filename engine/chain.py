"""
Option chain data access — wraps the option_chain_snapshots table OR a live
fetch via the NSE client. Returns a normalized in-memory view that the
recommender consumes.

Two modes:
  1. live   — fetch directly from NSE (no DB needed); use when ingestor hasn't
              been running yet but you want to try the engine today
  2. db     — read latest snapshot per (symbol, expiry, strike, option_type) from
              option_chain_snapshots
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, date, timezone, timedelta
from typing import Optional

import pandas as pd


IST = timezone(timedelta(hours=5, minutes=30))


@dataclass
class Quote:
    symbol: str
    expiry: date
    strike: int
    option_type: str
    ltp: float
    bid: float | None
    ask: float | None
    iv: float | None        # decimal (0.18 = 18%)
    oi: int | None
    volume: int | None
    snapshot_ts: datetime


@dataclass
class Chain:
    symbol: str
    spot: float
    expiry: date
    snapshot_ts: datetime
    quotes: list[Quote]

    def by_strike(self) -> dict[int, dict[str, Quote]]:
        out: dict[int, dict[str, Quote]] = {}
        for q in self.quotes:
            out.setdefault(q.strike, {})[q.option_type] = q
        return out

    def strikes(self) -> list[int]:
        return sorted({q.strike for q in self.quotes})

    def atm_strike(self, step: int = 50) -> int:
        ks = self.strikes()
        if not ks:
            return round(self.spot / step) * step
        return min(ks, key=lambda k: abs(k - self.spot))


def chain_from_live(symbol: str = "NIFTY", user_agent: str | None = None) -> Chain:
    """Fetch live chain via the v1 NSEClient. Uses nearest expiry."""
    from ingestor.nse_fetcher import NSEClient, parse_chain_to_dataframe
    from config.settings import Config

    ua = user_agent or Config(database_url="x").user_agent
    client = NSEClient(user_agent=ua)
    raw = client.fetch_option_chain(symbol)
    ts = datetime.now(tz=IST)
    df = parse_chain_to_dataframe(raw, symbol, ts)

    # Pick nearest expiry
    nearest = df["expiry"].min()
    df = df[df["expiry"] == nearest].copy()

    spot = float(df["spot"].iloc[0])
    quotes = [
        Quote(
            symbol=r["symbol"], expiry=r["expiry"], strike=int(r["strike"]),
            option_type=r["option_type"], ltp=float(r["ltp"] or 0),
            bid=_nz(r["bid"]), ask=_nz(r["ask"]),
            iv=(float(r["iv"]) / 100.0) if r["iv"] else None,  # NSE returns IV in %
            oi=_nz_int(r["oi"]), volume=_nz_int(r["volume"]),
            snapshot_ts=ts,
        )
        for _, r in df.iterrows()
    ]
    return Chain(symbol=symbol, spot=spot, expiry=nearest, snapshot_ts=ts, quotes=quotes)


def chain_from_db(conn, symbol: str = "NIFTY", expiry: date | None = None) -> Chain:
    """Latest snapshot per strike from DB. If expiry is None, use nearest."""
    with conn.cursor() as cur:
        if expiry is None:
            cur.execute(
                """
                SELECT MIN(expiry) FROM option_chain_snapshots
                WHERE symbol = %s AND expiry >= CURRENT_DATE
                """,
                (symbol,),
            )
            row = cur.fetchone()
            expiry = row[0] if row else None
            if expiry is None:
                raise RuntimeError(f"No future expiries for {symbol} in DB")

        cur.execute(
            """
            WITH latest AS (
              SELECT DISTINCT ON (strike, option_type)
                strike, option_type, ltp, bid, ask, iv, oi, volume, spot, snapshot_ts
              FROM option_chain_snapshots
              WHERE symbol = %s AND expiry = %s
              ORDER BY strike, option_type, snapshot_ts DESC
            )
            SELECT * FROM latest ORDER BY strike, option_type
            """,
            (symbol, expiry),
        )
        rows = cur.fetchall()

    if not rows:
        raise RuntimeError(f"No DB snapshots for {symbol} expiry {expiry}")

    spot = float(rows[0][8])
    ts = rows[0][9]
    quotes = [
        Quote(
            symbol=symbol, expiry=expiry, strike=int(r[0]), option_type=r[1],
            ltp=float(r[2] or 0), bid=_nz(r[3]), ask=_nz(r[4]),
            iv=(float(r[5]) / 100.0) if r[5] else None,
            oi=_nz_int(r[6]), volume=_nz_int(r[7]),
            snapshot_ts=r[9],
        )
        for r in rows
    ]
    return Chain(symbol=symbol, spot=spot, expiry=expiry, snapshot_ts=ts, quotes=quotes)


def _nz(x) -> float | None:
    if x is None: return None
    try:
        v = float(x)
        return v if v != 0 else None
    except (TypeError, ValueError):
        return None


def _nz_int(x) -> int | None:
    if x is None: return None
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def t_years_to_expiry(expiry: date, now: datetime | None = None) -> float:
    """Calendar-days/365. For intraday precision, use minutes-to-1530-on-expiry."""
    now = now or datetime.now(tz=IST)
    expiry_close = datetime.combine(expiry, datetime.min.time(), tzinfo=IST) \
                   + timedelta(hours=15, minutes=30)
    seconds = (expiry_close - now).total_seconds()
    return max(seconds / (365.0 * 24 * 3600), 1.0 / (365.0 * 24 * 3600))
