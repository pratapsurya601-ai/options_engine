"""
Kite Connect data adapter.

Provides:
  - login()        — daily access-token refresh via request_token
  - ltp(symbol)    — last price (used for spot)
  - historical_bars(token, interval='5minute', from, to)
  - option_chain(symbol='NIFTY', expiry=...) — pulls instruments + LTPs

Auth flow (do once daily):
  1. Visit kite.zerodha.com/connect/login?api_key=YOUR_KEY
  2. Login → redirected to your redirect_url with ?request_token=XXX
  3. Run: python -m engine.data.kite_source login --request-token XXX
  4. access_token saved to ~/.kite_token.json — valid till next 6am IST

Env vars required:
  KITE_API_KEY    — from kite.trade app dashboard
  KITE_API_SECRET — same
"""
from __future__ import annotations

import json
import os
from datetime import datetime, date, time, timedelta, timezone
from pathlib import Path
from typing import Optional


IST = timezone(timedelta(hours=5, minutes=30))
TOKEN_FILE = Path.home() / ".kite_token.json"

# Instrument tokens are stable per symbol; we cache the NIFTY 50 spot token + futures.
NIFTY_SPOT_TOKEN = 256265   # NIFTY 50 index
BANKNIFTY_SPOT_TOKEN = 260105


def _load_token() -> dict | None:
    # CI / Streamlit Cloud / Railway path: env var trumps file.
    # This lets headless environments (GitHub Actions) inject the token
    # without writing to disk.
    env_token = os.environ.get("KITE_ACCESS_TOKEN")
    if env_token:
        return {
            "access_token": env_token,
            "user_id": os.environ.get("KITE_USER_ID", ""),
            "issued_at": datetime.now(tz=IST).isoformat(),
        }
    if not TOKEN_FILE.exists():
        return None
    try:
        data = json.loads(TOKEN_FILE.read_text())
        # Check expiry — Kite tokens valid till 6am IST next day
        issued = datetime.fromisoformat(data["issued_at"])
        if datetime.now(tz=IST) - issued.astimezone(IST) > timedelta(hours=18):
            return None
        return data
    except Exception:
        return None


def _save_token(access_token: str, user_id: str) -> None:
    TOKEN_FILE.write_text(json.dumps({
        "access_token": access_token,
        "user_id": user_id,
        "issued_at": datetime.now(tz=IST).isoformat(),
    }))


def get_kite():
    """Return an authenticated KiteConnect instance or raise."""
    try:
        from kiteconnect import KiteConnect
    except ImportError:
        raise RuntimeError(
            "kiteconnect not installed. Run: pip install kiteconnect\n"
            "Then add 'kiteconnect>=4.2.0' to requirements.txt"
        )
    api_key = os.environ.get("KITE_API_KEY")
    if not api_key:
        raise RuntimeError("KITE_API_KEY not set")
    token = _load_token()
    if not token:
        raise RuntimeError(
            "No valid Kite access token. Run:\n"
            "  python -m engine.data.kite_source login --request-token <token-from-redirect>"
        )
    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(token["access_token"])
    return kite


def login_with_request_token(request_token: str) -> str:
    """Exchange request_token for access_token. Run once per trading day."""
    from kiteconnect import KiteConnect
    api_key = os.environ.get("KITE_API_KEY")
    api_secret = os.environ.get("KITE_API_SECRET")
    if not api_key or not api_secret:
        raise RuntimeError("KITE_API_KEY and KITE_API_SECRET must be set")
    kite = KiteConnect(api_key=api_key)
    data = kite.generate_session(request_token, api_secret=api_secret)
    _save_token(data["access_token"], data.get("user_id", ""))
    return data["access_token"]


def login_url() -> str:
    api_key = os.environ.get("KITE_API_KEY")
    if not api_key:
        raise RuntimeError("KITE_API_KEY not set")
    return f"https://kite.zerodha.com/connect/login?api_key={api_key}&v=3"


# ---- data accessors ----

def spot_ltp(symbol: str = "NIFTY") -> float:
    kite = get_kite()
    key = "NSE:NIFTY 50" if symbol == "NIFTY" else "NSE:NIFTY BANK"
    data = kite.ltp([key])
    return float(data[key]["last_price"])


def india_vix_ltp() -> float | None:
    """Fetch the current India VIX value. Returns None on failure."""
    try:
        kite = get_kite()
        data = kite.ltp(["NSE:INDIA VIX"])
        return float(data["NSE:INDIA VIX"]["last_price"])
    except Exception:
        return None


def all_active_futures(symbol: str = "NIFTY") -> list[dict]:
    """Return all currently-active monthly futures for `symbol` (nearest
    first, up to ~3 contracts). Each row: {token, tradingsymbol, expiry}.
    """
    kite = get_kite()
    instruments = _instruments_cached(kite, "NFO")
    today = datetime.now(tz=IST).date()
    fut = [
        {
            "token": i["instrument_token"],
            "tradingsymbol": i["tradingsymbol"],
            "expiry": i["expiry"],
        }
        for i in instruments
        if i["name"] == symbol
        and i["instrument_type"] == "FUT"
        and i["expiry"] >= today
    ]
    fut.sort(key=lambda r: r["expiry"])
    return fut


def futures_quotes(symbol: str = "NIFTY") -> list[dict]:
    """Fetch LTP + OI + volume for ALL active futures of `symbol`.
    Returns list of dicts with keys: expiry, tradingsymbol, ltp, oi, volume,
    bid, ask. Empty list on failure."""
    futs = all_active_futures(symbol)
    if not futs:
        return []
    kite = get_kite()
    keys = [f"NFO:{f['tradingsymbol']}" for f in futs]
    try:
        data = kite.quote(keys)
    except Exception:
        return []
    out = []
    for f in futs:
        k = f"NFO:{f['tradingsymbol']}"
        d = data.get(k, {})
        if not d:
            continue
        depth = d.get("depth") or {}
        buy = (depth.get("buy") or [{}])[0]
        sell = (depth.get("sell") or [{}])[0]
        out.append({
            "expiry": f["expiry"],
            "tradingsymbol": f["tradingsymbol"],
            "ltp": float(d.get("last_price") or 0) or None,
            "oi": int(d.get("oi") or 0) or None,
            "volume": int(d.get("volume") or 0) or None,
            "bid": float(buy.get("price") or 0) or None,
            "ask": float(sell.get("price") or 0) or None,
        })
    return out


def nearest_future_token(symbol: str = "NIFTY") -> tuple[int, str, date]:
    """
    Return (instrument_token, tradingsymbol, expiry) for the nearest-expiry
    monthly future of `symbol`. NIFTY has monthly futures only (no weeklies).
    """
    kite = get_kite()
    instruments = _instruments_cached(kite, "NFO")
    today = datetime.now(tz=IST).date()
    futures = [
        i for i in instruments
        if i["name"] == symbol
        and i["instrument_type"] == "FUT"
        and i["expiry"] >= today
    ]
    if not futures:
        raise RuntimeError(f"No active futures for {symbol}")
    nearest = min(futures, key=lambda i: i["expiry"])
    return nearest["instrument_token"], nearest["tradingsymbol"], nearest["expiry"]


def historical_bars(instrument_token: int, interval: str = "5minute",
                    from_dt: datetime | None = None,
                    to_dt: datetime | None = None) -> list[dict]:
    """
    Kite historical API. Returns list of {date, open, high, low, close, volume}.
    For today's 5-min bars: from_dt = today 09:15 IST, to_dt = now.
    """
    kite = get_kite()
    now_ist = datetime.now(tz=IST)
    if from_dt is None:
        from_dt = datetime.combine(now_ist.date(), time(9, 15), tzinfo=IST)
    if to_dt is None:
        to_dt = now_ist
    return kite.historical_data(instrument_token, from_dt, to_dt, interval)


_INSTRUMENT_CACHE: dict[str, tuple[date, list[dict]]] = {}


def _instruments_cached(kite, exchange: str = "NFO") -> list[dict]:
    """Cache instrument dump per-day — it's ~120k rows and costs credits."""
    cached = _INSTRUMENT_CACHE.get(exchange)
    today = datetime.now(tz=IST).date()
    if cached and cached[0] == today:
        return cached[1]
    data = kite.instruments(exchange)
    _INSTRUMENT_CACHE[exchange] = (today, data)
    return data


def list_expiries(symbol: str = "NIFTY") -> list[date]:
    """Return sorted list of all future expiries for `symbol` (nearest first)."""
    kite = get_kite()
    instruments = _instruments_cached(kite, "NFO")
    today = datetime.now(tz=IST).date()
    expiries = sorted({
        i["expiry"] for i in instruments
        if i["name"] == symbol
        and i["instrument_type"] in ("CE", "PE")
        and i["expiry"] >= today
    })
    return expiries


def option_chain(symbol: str = "NIFTY", expiry: date | None = None,
                 with_oi: bool = False) -> "Chain":
    """
    Pull the full option chain via Kite.

    Steps:
      1. kite.instruments("NFO") - all NFO instruments (cached daily)
      2. Filter to (name=symbol, instrument_type in CE/PE, expiry=target)
      3. Bulk LTP or QUOTE query on filtered tokens
         - LTP (cheap): just last_price; oi/volume = None
         - QUOTE (with_oi=True): includes oi, volume, depth -> larger payload, more credits
      4. Build Chain object

    Use with_oi=True for rules that need OI (OIDirectionRule, max-pain etc.).
    """
    from ..chain import Chain, Quote
    kite = get_kite()
    instruments = _instruments_cached(kite, "NFO")
    rows = [i for i in instruments
            if i["name"] == symbol and i["instrument_type"] in ("CE", "PE")]
    # Pick nearest expiry if not specified
    if expiry is None:
        future_expiries = sorted({i["expiry"] for i in rows if i["expiry"] >= date.today()})
        if not future_expiries:
            raise RuntimeError(f"No future expiries for {symbol}")
        expiry = future_expiries[0]
    rows = [i for i in rows if i["expiry"] == expiry]

    # Bulk fetch — Kite accepts up to 500 instruments per call
    keys = [f"NFO:{i['tradingsymbol']}" for i in rows]
    spot = spot_ltp(symbol)
    ts_now = datetime.now(tz=IST)
    quotes: list[Quote] = []

    chunk_size = 250
    for chunk_start in range(0, len(keys), chunk_size):
        chunk = keys[chunk_start:chunk_start + chunk_size]
        if with_oi:
            data = kite.quote(chunk)   # full quote with oi, volume, depth
        else:
            data = kite.ltp(chunk)     # ltp only — cheaper
        for inst in rows[chunk_start:chunk_start + chunk_size]:
            key = f"NFO:{inst['tradingsymbol']}"
            d = data.get(key, {})
            ltp = float(d.get("last_price") or 0)
            if ltp <= 0:
                continue
            oi_val = None
            vol_val = None
            if with_oi:
                oi_val = int(d.get("oi") or 0) or None
                vol_val = int(d.get("volume") or 0) or None
            quotes.append(Quote(
                symbol=symbol, expiry=expiry, strike=int(inst["strike"]),
                option_type=inst["instrument_type"],
                ltp=ltp, bid=None, ask=None,
                iv=None,
                oi=oi_val, volume=vol_val,
                snapshot_ts=ts_now,
            ))
    return Chain(symbol=symbol, spot=spot, expiry=expiry, snapshot_ts=ts_now, quotes=quotes)


def populate_ivs(chain, r: float = 0.065) -> None:
    """Back out IV from each quote's LTP (Kite doesn't provide IV)."""
    from ..pricing import implied_vol
    from ..chain import t_years_to_expiry
    t = t_years_to_expiry(chain.expiry)
    for q in chain.quotes:
        iv = implied_vol(q.ltp, chain.spot, q.strike, t, r=r, q=0.0, option_type=q.option_type)
        if iv is not None and 0.02 < iv < 2.0:
            object.__setattr__(q, "iv", iv)


# ---- CLI: login ----

def _cli():
    import argparse
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("login-url", help="Print the Kite login URL")
    lp = sub.add_parser("login", help="Exchange request_token for access_token")
    lp.add_argument("--request-token", required=True)
    sub.add_parser("test", help="Verify token works (fetches NIFTY spot)")
    args = p.parse_args()

    if args.cmd == "login-url":
        print(login_url())
    elif args.cmd == "login":
        tok = login_with_request_token(args.request_token)
        print(f"Saved access_token (len={len(tok)}) to {TOKEN_FILE}")
    elif args.cmd == "test":
        print(f"NIFTY spot: {spot_ltp('NIFTY'):.2f}")


if __name__ == "__main__":
    _cli()
