"""Tests for engine.research.rebuild_daemon."""
from __future__ import annotations

import json
import os
import subprocess
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from engine.research import rebuild_daemon as rd


IST = timezone(timedelta(hours=5, minutes=30))


# ---------------------------------------------------------------------------
# _is_trading_day
# ---------------------------------------------------------------------------

def test_is_trading_day_weekday():
    # 2026-06-01 is a Monday
    assert rd._is_trading_day(date(2026, 6, 1)) is True
    # Friday
    assert rd._is_trading_day(date(2026, 6, 5)) is True


def test_is_trading_day_weekend():
    # 2026-06-06 is a Saturday
    assert rd._is_trading_day(date(2026, 6, 6)) is False
    # Sunday
    assert rd._is_trading_day(date(2026, 6, 7)) is False


# ---------------------------------------------------------------------------
# _next_trigger
# ---------------------------------------------------------------------------

def test_next_trigger_skips_weekend():
    # Friday 17:00 IST -> next trigger Monday 16:00, not Saturday
    fri_evening = datetime(2026, 6, 5, 17, 0, tzinfo=IST)
    nxt = rd._next_trigger(fri_evening, time(16, 0))
    assert nxt.date() == date(2026, 6, 8)  # Monday
    assert nxt.hour == 16 and nxt.minute == 0


def test_next_trigger_same_day_if_before_time():
    # Weekday 09:00 IST -> same day 16:00
    monday_morning = datetime(2026, 6, 1, 9, 0, tzinfo=IST)
    nxt = rd._next_trigger(monday_morning, time(16, 0))
    assert nxt.date() == date(2026, 6, 1)
    assert nxt.hour == 16


# ---------------------------------------------------------------------------
# run_pipeline (with mocked subprocess.run)
# ---------------------------------------------------------------------------

def _mock_proc(returncode=0, stdout="ok", stderr=""):
    m = MagicMock()
    m.returncode = returncode
    m.stdout = stdout
    m.stderr = stderr
    return m


def test_run_pipeline_returns_six_step_results():
    with patch.object(rd.subprocess, "run", return_value=_mock_proc(0)):
        results = rd.run_pipeline(symbol="NIFTY", python_exe="python")
    assert len(results) == 6
    expected_names = [
        "flow_features", "labels", "regimes",
        "predictive_power", "walkforward", "conditional_edge",
    ]
    assert [r.name for r in results] == expected_names
    assert all(r.status == "OK" for r in results)
    assert all(r.return_code == 0 for r in results)


def test_run_pipeline_one_failure_does_not_halt():
    call_count = {"i": 0}

    def fake_run(*args, **kwargs):
        call_count["i"] += 1
        # Fail the 3rd step (regimes); others ok
        if call_count["i"] == 3:
            return _mock_proc(returncode=2, stderr="boom")
        return _mock_proc(0)

    with patch.object(rd.subprocess, "run", side_effect=fake_run):
        results = rd.run_pipeline(symbol="NIFTY", python_exe="python")

    assert len(results) == 6  # All steps ran
    assert results[2].status == "FAILED"
    assert results[2].return_code == 2
    assert "boom" in results[2].stderr_tail
    # Steps after the failure still executed
    assert results[3].status == "OK"
    assert results[5].status == "OK"


def test_run_pipeline_handles_subprocess_error():
    def fake_run(*args, **kwargs):
        raise subprocess.SubprocessError("kaboom")

    with patch.object(rd.subprocess, "run", side_effect=fake_run):
        results = rd.run_pipeline(symbol="NIFTY", python_exe="python")
    assert len(results) == 6
    assert all(r.status == "FAILED" for r in results)
    assert all("kaboom" in (r.error or "") for r in results)


def test_run_pipeline_handles_timeout():
    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="x", timeout=600,
                                         output=b"", stderr=b"slow")

    with patch.object(rd.subprocess, "run", side_effect=fake_run):
        results = rd.run_pipeline(symbol="NIFTY", python_exe="python")
    assert all(r.status == "FAILED" for r in results)
    assert all("TimeoutExpired" in (r.error or "") for r in results)


# ---------------------------------------------------------------------------
# write_status
# ---------------------------------------------------------------------------

def _make_results(statuses: list[str]) -> list[rd.StepResult]:
    out = []
    for i, s in enumerate(statuses):
        out.append(rd.StepResult(
            name=f"step{i}",
            cmd=["python", "-m", f"x{i}"],
            status=s,
            duration_s=1.0 + i,
            stdout_tail="hi",
            stderr_tail="" if s == "OK" else "err details",
            return_code=0 if s == "OK" else 1,
        ))
    return out


def test_write_status_writes_valid_json(tmp_path):
    results = _make_results(["OK", "OK", "OK", "OK", "OK", "OK"])
    target = tmp_path / "rebuild_status.json"
    started = datetime(2026, 6, 2, 16, 0, tzinfo=IST)
    completed = datetime(2026, 6, 2, 16, 3, 12, tzinfo=IST)
    out = rd.write_status(results, symbol="NIFTY",
                          run_started_at=started,
                          run_completed_at=completed,
                          daily_time=time(16, 0),
                          path=target)
    assert out == target
    data = json.loads(target.read_text(encoding="utf-8"))
    for k in ("symbol", "run_started_at", "run_completed_at",
              "overall_status", "steps", "next_run_at"):
        assert k in data
    assert data["symbol"] == "NIFTY"
    assert data["overall_status"] == "OK"
    assert len(data["steps"]) == 6
    # next_run_at should be a parseable ISO datetime
    datetime.fromisoformat(data["next_run_at"])


def test_overall_status_all_ok():
    r = _make_results(["OK"] * 6)
    assert rd._overall_status(r) == "OK"


def test_overall_status_one_failed_is_partial():
    r = _make_results(["OK", "OK", "FAILED", "OK", "OK", "OK"])
    assert rd._overall_status(r) == "PARTIAL"


def test_overall_status_all_failed():
    r = _make_results(["FAILED"] * 6)
    assert rd._overall_status(r) == "FAILED"


def test_write_status_partial_when_one_fails(tmp_path):
    results = _make_results(["OK", "FAILED", "OK", "OK", "OK", "OK"])
    target = tmp_path / "rebuild_status.json"
    rd.write_status(results, symbol="NIFTY", path=target,
                    run_started_at=datetime(2026, 6, 2, 16, 0, tzinfo=IST),
                    run_completed_at=datetime(2026, 6, 2, 16, 3, tzinfo=IST))
    data = json.loads(target.read_text(encoding="utf-8"))
    assert data["overall_status"] == "PARTIAL"
    # Failed step has stderr_tail in the dump
    failed = [s for s in data["steps"] if s["status"] == "FAILED"]
    assert len(failed) == 1
    assert "stderr_tail" in failed[0]


# ---------------------------------------------------------------------------
# CLI parser smoke
# ---------------------------------------------------------------------------

def test_parse_time_hhmm():
    assert rd._parse_time("16:00") == time(16, 0)
    assert rd._parse_time("9:30") == time(9, 30)


def test_parse_time_invalid():
    with pytest.raises(ValueError):
        rd._parse_time("not-a-time")
