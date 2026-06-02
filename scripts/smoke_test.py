"""
Local smoke test. Run before deploying.

Usage:
  export DATABASE_URL='postgresql://...'
  python scripts/smoke_test.py
"""
import logging
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

from config.settings import load
from ingestor.nse_fetcher import NSEClient, parse_chain_to_dataframe
from db.writer import get_conn

def main():
    cfg = load()
    print(f"DB target: {cfg.database_url.split('@')[-1]}")  # don't print credentials

    # Test 1: DB connectivity
    print("\n[1] Testing DB connection...")
    conn = get_conn(cfg.database_url)
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) AS n FROM option_chain_snapshots")
        n = cur.fetchone()["n"]
        print(f"    Current snapshots in DB: {n}")
    conn.close()

    # Test 2: NSE fetch (read-only, no write)
    print("\n[2] Testing NSE fetch...")
    client = NSEClient(cfg.user_agent, cfg.fetch_timeout_sec)
    raw = client.fetch_option_chain("NIFTY")
    spot = raw["records"]["underlyingValue"]
    n_strikes = len(raw["records"]["data"])
    print(f"    NIFTY spot: {spot} | strikes returned: {n_strikes}")

    # Test 3: Parse
    print("\n[3] Testing parser...")
    snapshot_ts = datetime.now(ZoneInfo(cfg.timezone))
    df = parse_chain_to_dataframe(raw, "NIFTY", snapshot_ts)
    print(f"    Parsed {len(df)} rows")
    print(f"    Sample: strike={df.iloc[0].strike} {df.iloc[0].option_type} "
          f"ltp={df.iloc[0].ltp} iv={df.iloc[0].iv}")

    print("\nSmoke test passed. Run `python -m ingestor.runner option_chain` to write.")

if __name__ == "__main__":
    main()
