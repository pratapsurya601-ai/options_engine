"""
Paper-trade logger — writes simulated trades to the existing `positions` table
using setup_tag prefixed with 'PAPER_'. Lets you compare engine-suggested
fills to your actual broker fills over time.

Persistence rules:
  - Each signal becomes one OPEN position row (action=BUY, lots=1).
  - The watcher monitors open paper positions every tick and closes them
    when EITHER target hit OR stop hit OR hold_minutes elapsed.
  - On close: exit_price, exit_ts, exit_reason, pnl set; status='closed'.

If DATABASE_URL is unset, paper writes go to a JSONL file at logs/paper.jsonl
so the watcher still works without DB.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .signals import Signal


IST = timezone(timedelta(hours=5, minutes=30))
JSONL_PATH = Path("logs/paper.jsonl")


def _ensure_jsonl():
    JSONL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not JSONL_PATH.exists():
        JSONL_PATH.touch()


def write_open(signal: Signal, entry_premium: float, lots: int = 1,
               lot_size: int = 75) -> str:
    """Write a new open paper position. Returns position id (uuid or jsonl key)."""
    db_url = os.environ.get("DATABASE_URL")
    payload = {
        "kind": "OPEN",
        "rule": signal.rule_name,
        "symbol": "NIFTY",
        "expiry": signal.trigger_context.get("expiry"),
        "strike": signal.strike,
        "option_type": signal.option_type,
        "action": "BUY",
        "lots": lots,
        "lot_size": lot_size,
        "entry_price": round(entry_premium, 2),
        "entry_ts": signal.ts.isoformat(),
        "entry_spot": signal.trigger_context.get("spot"),
        "entry_iv": signal.trigger_context.get("iv"),
        "thesis": signal.thesis,
        "setup_tag": f"PAPER_{signal.rule_name}",
        "planned_stop": signal.stop_premium,
        "planned_target": round(entry_premium + (signal.target_premium_gain or 0), 2),
        "planned_hold_minutes": signal.hold_minutes,
        "trail_activation_pts": signal.trigger_context.get("trail_activation_pts"),
        "trail_distance_pts": signal.trigger_context.get("trail_distance_pts"),
        "exit_on_ema_flip": signal.trigger_context.get("exit_on_ema_flip", False),
        "entry_direction": signal.trigger_context.get("entry_direction"),
        "ema_fast_period": signal.trigger_context.get("ema_fast_period"),
        "ema_slow_period": signal.trigger_context.get("ema_slow_period"),
        "eod_close_minutes": signal.trigger_context.get("eod_close_minutes"),
        "status": "open",
    }
    if not db_url:
        _ensure_jsonl()
        with JSONL_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")
        return f"jsonl:{signal.ts.isoformat()}:{signal.strike}{signal.option_type}"

    try:
        import psycopg
    except ImportError:
        _ensure_jsonl()
        with JSONL_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")
        return f"jsonl:{signal.ts.isoformat()}"

    try:
        with psycopg.connect(db_url) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO positions
                      (symbol, expiry, strike, option_type, action, lots, lot_size,
                       entry_price, entry_ts, entry_spot, entry_iv, thesis, setup_tag,
                       planned_stop, planned_target, status)
                    VALUES
                      (%(symbol)s, %(expiry)s, %(strike)s, %(option_type)s, %(action)s,
                       %(lots)s, %(lot_size)s, %(entry_price)s, %(entry_ts)s,
                       %(entry_spot)s, %(entry_iv)s, %(thesis)s, %(setup_tag)s,
                       %(planned_stop)s, %(planned_target)s, %(status)s)
                    RETURNING id
                    """,
                    payload,
                )
                pid = cur.fetchone()[0]
            conn.commit()
        return str(pid)
    except Exception:
        # Fallback to JSONL
        _ensure_jsonl()
        with JSONL_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")
        return f"jsonl:{signal.ts.isoformat()}:{signal.strike}{signal.option_type}"


def write_close(position_id: str, exit_premium: float, exit_ts: datetime,
                exit_reason: str, lots: int = 1, lot_size: int = 75,
                entry_premium: float | None = None) -> None:
    """Close a paper position. Idempotent — refuses to write duplicate
    CLOSE events for the same position_id."""
    db_url = os.environ.get("DATABASE_URL")
    if entry_premium is not None:
        pnl = (exit_premium - entry_premium) * lots * lot_size
    else:
        pnl = None
    if position_id.startswith("jsonl:") or not db_url:
        _ensure_jsonl()
        # Idempotency check: if this position_id already has a CLOSE event, skip
        if JSONL_PATH.exists():
            with JSONL_PATH.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        ev = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if (ev.get("kind") == "CLOSE"
                            and ev.get("position_id") == position_id):
                        return    # already closed — don't duplicate
        with JSONL_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps({
                "kind": "CLOSE",
                "position_id": position_id,
                "exit_price": round(exit_premium, 2),
                "exit_ts": exit_ts.isoformat(),
                "exit_reason": exit_reason,
                "pnl": pnl,
            }) + "\n")
        return

    try:
        import psycopg
    except ImportError:
        return
    with psycopg.connect(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE positions
                   SET exit_price = %s, exit_ts = %s, exit_reason = %s,
                       pnl = %s, status = 'closed', updated_at = NOW()
                 WHERE id = %s::uuid
                """,
                (round(exit_premium, 2), exit_ts, exit_reason, pnl, position_id),
            )
        conn.commit()


def open_paper_positions(rule_name: str | None = None) -> list[dict]:
    """
    List currently open paper positions. Tries DB first; falls back to
    reconstructing from logs/paper.jsonl. Survives watcher restarts because
    JSONL is the persistent source of truth.
    """
    db_url = os.environ.get("DATABASE_URL")
    if db_url:
        try:
            import psycopg
            from psycopg.rows import dict_row
            with psycopg.connect(db_url) as conn:
                with conn.cursor(row_factory=dict_row) as cur:
                    if rule_name:
                        cur.execute(
                            "SELECT * FROM positions WHERE status='open' AND setup_tag=%s",
                            (f"PAPER_{rule_name}",),
                        )
                    else:
                        cur.execute(
                            "SELECT * FROM positions WHERE status='open' AND setup_tag LIKE 'PAPER_%%'"
                        )
                    rows = list(cur.fetchall())
                    if rows:
                        return rows
        except Exception:
            pass   # fall through to JSONL

    # JSONL fallback — reconstruct open positions from paper.jsonl
    if not JSONL_PATH.exists():
        return []
    positions: dict[str, dict] = {}
    with JSONL_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get("kind") == "OPEN":
                pid = f"jsonl:{ev.get('entry_ts')}:{ev.get('strike')}{ev.get('option_type')}"
                ev["id"] = pid
                ev["status"] = "open"
                positions[pid] = ev
            elif ev.get("kind") == "CLOSE":
                pid = ev.get("position_id")
                if pid and pid in positions:
                    positions[pid]["status"] = "closed"
                    positions[pid]["exit_price"] = ev.get("exit_price")
                    positions[pid]["exit_ts"] = ev.get("exit_ts")
                    positions[pid]["exit_reason"] = ev.get("exit_reason")
                    positions[pid]["pnl"] = ev.get("pnl")

    open_pos = [p for p in positions.values() if p.get("status") == "open"]
    if rule_name:
        open_pos = [p for p in open_pos if p.get("rule") == rule_name
                    or (p.get("setup_tag") or "").endswith(rule_name)]
    return open_pos
