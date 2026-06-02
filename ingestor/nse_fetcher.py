"""
NSE option chain fetcher.

NSE blocks bare requests — must establish session with cookie warmup
before hitting the API. This is the source of 90% of integration pain.
"""
import logging
import time
from datetime import datetime, date
from typing import Optional

import requests
import pandas as pd

logger = logging.getLogger(__name__)

NSE_HOMEPAGE = "https://www.nseindia.com"
NSE_OPTION_CHAIN = "https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"


class NSEClient:
    """Maintains a session with valid NSE cookies. Auto-refreshes on 401/403."""

    # NSE checks the FULL Chrome header set. Missing sec-ch-ua-* or sec-fetch-*
    # triggers 403. The exact set below is what current Chrome on Windows sends.
    _BROWSER_HEADERS = {
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language": "en-IN,en-GB;q=0.9,en-US;q=0.8,en;q=0.7",
        "Accept-Encoding": "gzip, deflate, br, zstd",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "sec-ch-ua": '"Chromium";v="131", "Not_A Brand";v="24", "Google Chrome";v="131"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"Windows"',
        "Cache-Control": "max-age=0",
    }

    def __init__(self, user_agent: str, timeout: int = 15):
        self.timeout = timeout
        # Use a current Chrome UA — old UAs are flagged as bots
        self.user_agent = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/131.0.0.0 Safari/537.36"
        ) if "Chrome/120" in user_agent or "Mac OS" in user_agent else user_agent
        self.session: Optional[requests.Session] = None
        self._warm()

    def _warm(self):
        """
        NSE cookie warmup. Real browsers visit homepage -> option-chain page,
        each setting cookies the API endpoint requires. We replicate that
        navigation, with the EXACT header set Chrome sends.
        """
        self.session = requests.Session()
        nav_headers = {"User-Agent": self.user_agent, **self._BROWSER_HEADERS}
        self.session.headers.update(nav_headers)

        try:
            # Step 1: homepage (sets primary cookies including nseQuoteSymbols)
            r1 = self.session.get(NSE_HOMEPAGE, timeout=self.timeout, allow_redirects=True)
            if r1.status_code == 403:
                raise RuntimeError(
                    "NSE returned 403 on homepage — your IP may be rate-limited or "
                    "blocked. Wait 5-10 min and retry, try a different network, or "
                    "use --source manual once we add it."
                )
            r1.raise_for_status()

            time.sleep(1.0)  # mimic human dwell time before next nav

            # Step 2: option-chain page (sets bm_sz/_abck — Akamai bot cookies)
            r2 = self.session.get(
                "https://www.nseindia.com/option-chain",
                timeout=self.timeout, allow_redirects=True,
            )
            r2.raise_for_status()

            time.sleep(0.5)

            # Step 3: market-data summary (final cookie set for API endpoints)
            self.session.get(
                "https://www.nseindia.com/get-quotes/derivatives?symbol=NIFTY",
                timeout=self.timeout, allow_redirects=True,
            )

            # Switch headers to API-mode for subsequent requests
            self.session.headers.update({
                "Accept": "application/json, text/plain, */*",
                "Sec-Fetch-Dest": "empty",
                "Sec-Fetch-Mode": "cors",
                "Sec-Fetch-Site": "same-origin",
                "Referer": "https://www.nseindia.com/option-chain",
                "X-Requested-With": "XMLHttpRequest",
            })

            logger.info("NSE session warmed, cookies: %d", len(self.session.cookies))
        except Exception as e:
            logger.error("Session warm failed: %s", e)
            raise

    def fetch_option_chain(self, symbol: str, retries: int = 3) -> dict:
        """Fetch raw option chain JSON for symbol (NIFTY/BANKNIFTY/etc)."""
        url = NSE_OPTION_CHAIN.format(symbol=symbol)
        last_err = None
        for attempt in range(retries):
            try:
                r = self.session.get(url, timeout=self.timeout)
                if r.status_code in (401, 403):
                    logger.warning("Got %d, re-warming session", r.status_code)
                    self._warm()
                    continue
                r.raise_for_status()
                data = r.json()
                if "records" not in data:
                    raise ValueError(f"Unexpected payload: {list(data.keys())}")
                return data
            except Exception as e:
                last_err = e
                logger.warning("Fetch attempt %d failed: %s", attempt + 1, e)
                time.sleep(2 ** attempt)
        raise RuntimeError(f"Failed after {retries} attempts: {last_err}")


def parse_chain_to_dataframe(raw: dict, symbol: str, snapshot_ts: datetime) -> pd.DataFrame:
    """Flatten NSE option chain JSON into a normalized DataFrame."""
    records = raw.get("records", {})
    spot = records.get("underlyingValue")
    if spot is None:
        raise ValueError("No underlyingValue in payload")

    rows = []
    for item in records.get("data", []):
        strike = item.get("strikePrice")
        expiry_str = item.get("expiryDate")
        if strike is None or not expiry_str:
            continue
        try:
            expiry = datetime.strptime(expiry_str, "%d-%b-%Y").date()
        except ValueError:
            logger.warning("Bad expiry format: %s", expiry_str)
            continue

        for opt_type_key, opt_type in [("CE", "CE"), ("PE", "PE")]:
            d = item.get(opt_type_key)
            if not d:
                continue
            rows.append({
                "symbol": symbol,
                "spot": spot,
                "expiry": expiry,
                "strike": int(strike),
                "option_type": opt_type,
                "ltp": d.get("lastPrice"),
                "bid": d.get("bidprice"),
                "ask": d.get("askPrice"),
                "volume": d.get("totalTradedVolume"),
                "oi": d.get("openInterest"),
                "oi_change": d.get("changeinOpenInterest"),
                "iv": d.get("impliedVolatility"),
                "snapshot_ts": snapshot_ts,
            })

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("Parsed zero rows from option chain")
    logger.info("Parsed %d rows for %s @ spot %.2f", len(df), symbol, spot)
    return df
