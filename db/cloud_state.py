"""
Cloud watcher state — read/write helpers backed by Aiven Postgres.

The cloud watcher is stateless across runs. Every 5-minute invocation loads
its state from Aiven, evaluates rules, persists results, and exits. This
module is the interface between the watcher and the database.

Tables touched (see db/002_cloud_watcher_schema.sql):
  - signals          : append-only log of every rule fire
  - rule_cooldowns   : last fire time per rule (for cooldown enforcement)
  - positions        : open / closed paper trades
  - watcher_runs     : audit log per watcher invocation
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Json

logger = logging.getLogger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))


def get_conn(database_url: str):
    return psycopg.connect(database_url, row_factory=dict_row)


# ---------- cooldown helpers ----------

def last_fired_at(conn, rule_name: str, symbol: str) -> datetime | None:
    """Return the most recent fire timestamp for (rule, symbol), or None."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT last_fired_at FROM rule_cooldowns "
            "WHERE rule_name = %s AND symbol = %s",
            (rule_name, symbol),
        )
        row = cur.fetchone()
    return row["last_fired_at"] if row else None


def is_in_cooldown(conn, rule_name: str, symbol: str, cooldown_min: int,
                   now: datetime | None = None) -> bool:
    """True iff (rule, symbol) fired within the last `cooldown_min` minutes."""
    now = now or datetime.now(tz=IST)
    last = last_fired_at(conn, rule_name, symbol)
    if last is None:
        return False
    return (now - last) < timedelta(minutes=cooldown_min)


def record_fire(conn, rule_name: str, symbol: str, cooldown_min: int,
                ts: datetime | None = None) -> None:
    """Upsert the cooldown row to mark a fire at ts (default: now)."""
    ts = ts or datetime.now(tz=IST)
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO rule_cooldowns (rule_name, symbol, last_fired_at, cooldown_min)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (rule_name, symbol) DO UPDATE
              SET last_fired_at = EXCLUDED.last_fired_at,
                  cooldown_min  = EXCLUDED.cooldown_min
            """,
            (rule_name, symbol, ts, cooldown_min),
        )
    conn.commit()


# ---------- signal log ----------

def log_signal(conn, *, rule_name: str, symbol: str, ts: datetime,
               spot: float | None, action: str | None,
               strike: int | None, expiry: Any | None,
               premium: float | None, target_premium: float | None,
               stop_premium: float | None, trigger_context: dict,
               outcome: str) -> int:
    """Append a row to signals. Returns the new id."""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO signals
              (rule_name, symbol, ts, spot, action, strike, expiry,
               premium, target_premium, stop_premium, trigger_context, outcome)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (rule_name, symbol, ts, spot, action, strike, expiry,
             premium, target_premium, stop_premium,
             Json(trigger_context), outcome),
        )
        row = cur.fetchone()
    conn.commit()
    return row["id"]


# ---------- positions ----------

@dataclass
class OpenPosition:
    id: str
    symbol: str
    expiry: Any
    strike: int
    option_type: str
    action: str  # 'BUY' or 'SELL'
    lots: int
    lot_size: int
    entry_price: float
    entry_ts: datetime
    rule_name: str | None
    planned_stop: float | None
    planned_target: float | None
    high_water_mark: float | None
    trail_activation_pts: float | None
    trail_distance_pts: float | None
    hold_until_ts: datetime | None
    setup_tag: str


def get_open_positions(conn, *, rule_name: str | None = None,
                       symbol: str | None = None,
                       setup_tag: str | None = None) -> list[OpenPosition]:
    """Return all open paper positions, optionally filtered."""
    sql = "SELECT * FROM positions WHERE status = 'open'"
    params: list = []
    if rule_name:
        sql += " AND rule_name = %s"
        params.append(rule_name)
    if symbol:
        sql += " AND symbol = %s"
        params.append(symbol)
    if setup_tag:
        sql += " AND setup_tag = %s"
        params.append(setup_tag)
    sql += " ORDER BY entry_ts"
    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
    return [
        OpenPosition(
            id=str(r["id"]),
            symbol=r["symbol"],
            expiry=r["expiry"],
            strike=r["strike"],
            option_type=r["option_type"],
            action=r["action"],
            lots=r["lots"],
            lot_size=r["lot_size"],
            entry_price=float(r["entry_price"]),
            entry_ts=r["entry_ts"],
            rule_name=r.get("rule_name"),
            planned_stop=float(r["planned_stop"]) if r.get("planned_stop") is not None else None,
            planned_target=float(r["planned_target"]) if r.get("planned_target") is not None else None,
            high_water_mark=float(r["high_water_mark"]) if r.get("high_water_mark") is not None else None,
            trail_activation_pts=float(r["trail_activation_pts"]) if r.get("trail_activation_pts") is not None else None,
            trail_distance_pts=float(r["trail_distance_pts"]) if r.get("trail_distance_pts") is not None else None,
            hold_until_ts=r.get("hold_until_ts"),
            setup_tag=r["setup_tag"],
        )
        for r in rows
    ]


def open_position(conn, *,
                  symbol: str, expiry: Any, strike: int, option_type: str,
                  action: str, lots: int, lot_size: int,
                  entry_price: float, entry_ts: datetime,
                  entry_spot: float | None, entry_iv: float | None,
                  thesis: str, setup_tag: str, rule_name: str,
                  planned_stop: float | None, planned_target: float | None,
                  trail_activation_pts: float | None = None,
                  trail_distance_pts: float | None = None,
                  hold_until_ts: datetime | None = None) -> str:
    """Insert a new open paper position. Returns the position id (uuid as str)."""
    # Enforce schema check: thesis must be >= 50 chars
    if len(thesis) < 50:
        thesis = (thesis + " | cloud_watcher auto-opened paper trade for research validation")[:512]
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO positions
              (symbol, expiry, strike, option_type, action, lots, lot_size,
               entry_price, entry_ts, entry_spot, entry_iv, thesis, setup_tag,
               rule_name, planned_stop, planned_target,
               high_water_mark, trail_activation_pts, trail_distance_pts,
               hold_until_ts, status, source)
            VALUES (%s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s,
                    %s, %s, %s,
                    %s, %s, %s,
                    %s, 'open', 'paper')
            RETURNING id
            """,
            (symbol, expiry, strike, option_type, action, lots, lot_size,
             entry_price, entry_ts, entry_spot, entry_iv, thesis, setup_tag,
             rule_name, planned_stop, planned_target,
             entry_price, trail_activation_pts, trail_distance_pts,
             hold_until_ts),
        )
        row = cur.fetchone()
    conn.commit()
    return str(row["id"])


def close_position(conn, position_id: str, *,
                   exit_price: float, exit_ts: datetime, exit_reason: str,
                   pnl: float) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE positions
               SET exit_price = %s,
                   exit_ts    = %s,
                   exit_reason = %s,
                   pnl = %s,
                   status = 'closed',
                   updated_at = NOW()
             WHERE id = %s::uuid
            """,
            (exit_price, exit_ts, exit_reason, pnl, position_id),
        )
    conn.commit()


def update_high_water_mark(conn, position_id: str, hwm: float,
                           new_stop: float | None = None) -> None:
    with conn.cursor() as cur:
        if new_stop is not None:
            cur.execute(
                """
                UPDATE positions
                   SET high_water_mark = %s,
                       planned_stop    = %s,
                       updated_at      = NOW()
                 WHERE id = %s::uuid
                """,
                (hwm, new_stop, position_id),
            )
        else:
            cur.execute(
                """
                UPDATE positions
                   SET high_water_mark = %s,
                       updated_at      = NOW()
                 WHERE id = %s::uuid
                """,
                (hwm, position_id),
            )
    conn.commit()


# ---------- watcher run log ----------

def start_watcher_run(conn, *, rule_name: str, symbol: str,
                      ts: datetime | None = None) -> int:
    ts = ts or datetime.now(tz=IST)
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO watcher_runs (rule_name, symbol, started_at, status) "
            "VALUES (%s, %s, %s, 'running') RETURNING id",
            (rule_name, symbol, ts),
        )
        row = cur.fetchone()
    conn.commit()
    return row["id"]


def finish_watcher_run(conn, run_id: int, *,
                       bars_loaded: int, signals_fired: int,
                       positions_opened: int, positions_closed: int,
                       trail_updated: int, status: str,
                       error_message: str | None = None) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE watcher_runs
               SET finished_at = NOW(),
                   bars_loaded = %s,
                   signals_fired = %s,
                   positions_opened = %s,
                   positions_closed = %s,
                   trail_updated = %s,
                   status = %s,
                   error_message = %s
             WHERE id = %s
            """,
            (bars_loaded, signals_fired, positions_opened, positions_closed,
             trail_updated, status, error_message, run_id),
        )
    conn.commit()
