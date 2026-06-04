"""Database operations — bulk upserts with conflict handling."""
import logging
from datetime import datetime
from typing import Optional

import psycopg
from psycopg.rows import dict_row
import pandas as pd

logger = logging.getLogger(__name__)


def get_conn(database_url: str):
    """Open a connection. Caller is responsible for closing."""
    return psycopg.connect(database_url, row_factory=dict_row)


def write_option_chain(conn, df: pd.DataFrame) -> int:
    """Bulk insert option chain rows. Ignores conflicts on UNIQUE constraint."""
    if df.empty:
        return 0

    cols = [
        "symbol", "spot", "expiry", "strike", "option_type",
        "ltp", "bid", "ask", "volume", "oi", "oi_change", "iv", "snapshot_ts",
    ]
    records = df[cols].where(pd.notnull(df[cols]), None).values.tolist()

    sql = """
        INSERT INTO option_chain_snapshots
            (symbol, spot, expiry, strike, option_type,
             ltp, bid, ask, volume, oi, oi_change, iv, snapshot_ts)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (symbol, expiry, strike, option_type, snapshot_ts) DO NOTHING
    """
    with conn.cursor() as cur:
        cur.executemany(sql, records)
        written = cur.rowcount
    conn.commit()
    logger.info("Wrote %d option chain rows", written)
    return written


def log_run_start(conn, job_name: str) -> int:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO ingestion_runs (job_name, started_at, status) "
            "VALUES (%s, %s, 'running') RETURNING id",
            (job_name, datetime.utcnow()),
        )
        run_id = cur.fetchone()["id"]
    conn.commit()
    return run_id


def log_run_finish(conn, run_id: int, rows: int, status: str, error: Optional[str] = None):
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE ingestion_runs SET finished_at = %s, rows_written = %s, "
            "status = %s, error_message = %s WHERE id = %s",
            (datetime.utcnow(), rows, status, error, run_id),
        )
    conn.commit()


def write_futures_snapshots(conn, df: pd.DataFrame) -> int:
    """Bulk insert futures snapshot rows. Ignores conflicts on UNIQUE constraint."""
    if df is None or df.empty:
        return 0
    cols = [
        "symbol", "snapshot_ts", "expiry", "ltp", "bid", "ask",
        "volume", "oi", "oi_change", "spot", "basis", "basis_pct",
    ]
    records = df[cols].where(pd.notnull(df[cols]), None).values.tolist()
    sql = """
        INSERT INTO futures_snapshots
            (symbol, snapshot_ts, expiry, ltp, bid, ask,
             volume, oi, oi_change, spot, basis, basis_pct)
        VALUES (%s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s)
        ON CONFLICT (symbol, expiry, snapshot_ts) DO NOTHING
    """
    with conn.cursor() as cur:
        cur.executemany(sql, records)
        written = cur.rowcount
    conn.commit()
    logger.info("Wrote %d futures rows", written)
    return written


def write_vix_snapshot(conn, snapshot_ts: datetime, vix: float) -> int:
    """Insert one VIX snapshot. Ignores conflict on primary key."""
    sql = """
        INSERT INTO vix_snapshots (snapshot_ts, vix)
        VALUES (%s, %s)
        ON CONFLICT (snapshot_ts) DO NOTHING
    """
    with conn.cursor() as cur:
        cur.execute(sql, (snapshot_ts, vix))
        written = cur.rowcount
    conn.commit()
    return written


def write_fii_dii(conn, row: dict) -> bool:
    """Upsert single day's FII/DII row."""
    sql = """
        INSERT INTO fii_dii_activity (
            trade_date, fii_cash_buy, fii_cash_sell, fii_cash_net,
            dii_cash_buy, dii_cash_sell, dii_cash_net,
            fii_index_fut_long_contracts, fii_index_fut_short_contracts,
            fii_index_fut_net_contracts, fii_index_fut_long_short_ratio,
            fii_index_call_long, fii_index_call_short,
            fii_index_put_long, fii_index_put_short,
            fii_stock_fut_long, fii_stock_fut_short,
            raw_payload
        ) VALUES (
            %(trade_date)s, %(fii_cash_buy)s, %(fii_cash_sell)s, %(fii_cash_net)s,
            %(dii_cash_buy)s, %(dii_cash_sell)s, %(dii_cash_net)s,
            %(fii_index_fut_long_contracts)s, %(fii_index_fut_short_contracts)s,
            %(fii_index_fut_net_contracts)s, %(fii_index_fut_long_short_ratio)s,
            %(fii_index_call_long)s, %(fii_index_call_short)s,
            %(fii_index_put_long)s, %(fii_index_put_short)s,
            %(fii_stock_fut_long)s, %(fii_stock_fut_short)s,
            %(raw_payload)s
        )
        ON CONFLICT (trade_date) DO UPDATE SET
            fii_cash_net = EXCLUDED.fii_cash_net,
            dii_cash_net = EXCLUDED.dii_cash_net,
            fii_index_fut_net_contracts = EXCLUDED.fii_index_fut_net_contracts,
            fii_index_fut_long_short_ratio = EXCLUDED.fii_index_fut_long_short_ratio,
            raw_payload = EXCLUDED.raw_payload
    """
    with conn.cursor() as cur:
        cur.execute(sql, row)
    conn.commit()
    return True
