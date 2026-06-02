"""
Main ingestion runner.

Two modes:
  python -m ingestor.runner option_chain   # run once, exit (for cron)
  python -m ingestor.runner fii_dii        # EOD job
  python -m ingestor.runner loop           # local dev: poll every 5 min

Production: schedule via Railway cron or APScheduler.
"""
import argparse
import logging
import sys
import time
from datetime import datetime, date, time as dtime, timedelta
from zoneinfo import ZoneInfo

from config.settings import load as load_config
from ingestor.nse_fetcher import NSEClient, parse_chain_to_dataframe
from ingestor.fii_dii_fetcher import (
    fetch_fii_dii_cash, fetch_fii_derivatives, build_combined_row
)
from db.writer import (
    get_conn, write_option_chain, log_run_start, log_run_finish, write_fii_dii
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ingestor")


def is_market_open(cfg) -> bool:
    """IST market hours, Mon-Fri only."""
    now = datetime.now(ZoneInfo(cfg.timezone))
    if now.weekday() >= 5:  # Sat=5, Sun=6
        return False
    open_t = dtime.fromisoformat(cfg.market_open)
    close_t = dtime.fromisoformat(cfg.market_close)
    return open_t <= now.time() <= close_t


def run_option_chain_once(cfg):
    """Single pass: fetch all configured symbols, write to DB."""
    conn = get_conn(cfg.database_url)
    run_id = log_run_start(conn, "option_chain")
    total_rows = 0
    try:
        client = NSEClient(cfg.user_agent, cfg.fetch_timeout_sec)
        snapshot_ts = datetime.now(ZoneInfo(cfg.timezone))
        for symbol in cfg.nse_symbols:
            try:
                raw = client.fetch_option_chain(symbol)
                df = parse_chain_to_dataframe(raw, symbol, snapshot_ts)
                written = write_option_chain(conn, df)
                total_rows += written
            except Exception as e:
                logger.error("Failed %s: %s", symbol, e)
        log_run_finish(conn, run_id, total_rows, "success")
    except Exception as e:
        log_run_finish(conn, run_id, total_rows, "failed", str(e))
        raise
    finally:
        conn.close()
    return total_rows


def run_fii_dii_once(cfg, target_date: date = None):
    target_date = target_date or date.today()
    conn = get_conn(cfg.database_url)
    run_id = log_run_start(conn, "fii_dii")
    try:
        client = NSEClient(cfg.user_agent, cfg.fetch_timeout_sec)
        cash = fetch_fii_dii_cash(client)
        deriv = fetch_fii_derivatives(client)
        row = build_combined_row(cash, deriv, target_date)
        if row and row.get("fii_cash_net") is not None:
            write_fii_dii(conn, row)
            log_run_finish(conn, run_id, 1, "success")
            return 1
        log_run_finish(conn, run_id, 0, "no_data", "No row for target date")
        return 0
    except Exception as e:
        log_run_finish(conn, run_id, 0, "failed", str(e))
        raise
    finally:
        conn.close()


def loop(cfg):
    """Local dev: poll continuously, only act when market open."""
    logger.info("Loop mode started. Interval: %ds", cfg.ingestion_interval_sec)
    while True:
        if is_market_open(cfg):
            try:
                rows = run_option_chain_once(cfg)
                logger.info("Cycle complete: %d rows", rows)
            except Exception as e:
                logger.exception("Cycle failed: %s", e)
        else:
            logger.info("Market closed, skipping")
        time.sleep(cfg.ingestion_interval_sec)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["option_chain", "fii_dii", "loop"])
    args = p.parse_args()
    cfg = load_config()

    if args.mode == "option_chain":
        rows = run_option_chain_once(cfg)
        logger.info("Done. %d rows written.", rows)
    elif args.mode == "fii_dii":
        rows = run_fii_dii_once(cfg)
        logger.info("FII/DII done. %d rows.", rows)
    elif args.mode == "loop":
        loop(cfg)


if __name__ == "__main__":
    sys.exit(main())
