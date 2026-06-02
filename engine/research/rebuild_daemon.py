"""
Research pipeline daemon.

Runs the 6 research modules sequentially once per day after market close:
    1. flow_features build-all --symbol NIFTY
    2. labels label-file --features data/research/flow_features/NIFTY/all.parquet
    3. regimes build --symbol NIFTY --interval day
    4. predictive_power run --symbol NIFTY
    5. walkforward run --symbol NIFTY
    6. conditional_edge run --symbol NIFTY

After completion, writes status to data/research/rebuild_status.json so the
dashboard can show "last automated rebuild" info.

Failures in a single module do NOT halt the pipeline -- we log the error,
mark the run PARTIAL, and continue. Failures across runs do NOT crash the
daemon either -- we sleep until the next scheduled time.

Holidays
--------
We do not maintain a hardcoded Indian-market-holiday list, so the daemon
WILL run on Republic Day, Diwali, etc. That just means a few extra
no-op-ish runs per year. Cheap; accepted.

# Running the daemon on Windows

Option A (cmd window in background):
  start "research_daemon" /B C:\\ProgramData\\miniconda3\\python.exe ^
      -m engine.research.rebuild_daemon run

Option B (NSSM as Windows service):
  See docs/research_daemon_service.md

Option C (Task Scheduler):
  schtasks /create /tn "OptionsResearchDaemon" ^
      /tr "C:\\ProgramData\\miniconda3\\python.exe -m engine.research.rebuild_daemon run" ^
      /sc onlogon /f

CLI
---
  python -m engine.research.rebuild_daemon run --daily-time 16:00
  python -m engine.research.rebuild_daemon once
  python -m engine.research.rebuild_daemon status
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time as _time_mod
from dataclasses import dataclass, asdict, field
from datetime import datetime, date, time, timedelta, timezone
from pathlib import Path
from typing import Optional

IST = timezone(timedelta(hours=5, minutes=30))

DEFAULT_PYTHON = r"C:\ProgramData\miniconda3\python.exe"
STATUS_PATH = Path("data/research/rebuild_status.json")
PER_STEP_TIMEOUT_SEC = 600  # 10 min per module max


# ---------------------------------------------------------------------------
# Step definitions
# ---------------------------------------------------------------------------

def _step_commands(symbol: str) -> list[tuple[str, list[str]]]:
    """The 6 pipeline steps, in order. Returns [(name, argv-tail)]."""
    flow_path = f"data/research/flow_features/{symbol}/all.parquet"
    return [
        ("flow_features",
         ["-m", "engine.research.flow_features", "build-all", "--symbol", symbol]),
        ("labels",
         ["-m", "engine.research.labels", "label-file", "--features", flow_path]),
        ("regimes",
         ["-m", "engine.research.regimes", "build", "--symbol", symbol, "--interval", "day"]),
        ("predictive_power",
         ["-m", "engine.research.predictive_power", "run", "--symbol", symbol]),
        ("walkforward",
         ["-m", "engine.research.walkforward", "run", "--symbol", symbol]),
        ("conditional_edge",
         ["-m", "engine.research.conditional_edge", "run", "--symbol", symbol]),
    ]


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class StepResult:
    """One module's outcome."""
    name: str
    cmd: list[str]
    status: str               # "OK" | "FAILED" | "SKIPPED"
    duration_s: float
    stdout_tail: str = ""     # last ~1000 chars
    stderr_tail: str = ""     # last ~1000 chars
    return_code: Optional[int] = None
    error: Optional[str] = None  # exception message if subprocess machinery failed

    def to_status_dict(self) -> dict:
        d = {
            "name": self.name,
            "status": self.status,
            "duration_s": round(self.duration_s, 2),
            "return_code": self.return_code,
        }
        if self.status == "FAILED" and self.stderr_tail:
            d["stderr_tail"] = self.stderr_tail
        if self.error:
            d["error"] = self.error
        return d


# ---------------------------------------------------------------------------
# Time / scheduling helpers
# ---------------------------------------------------------------------------

def _now_ist() -> datetime:
    return datetime.now(tz=IST)


def _ts_str(dt: Optional[datetime] = None) -> str:
    """HH:MM:SS IST string for log lines."""
    if dt is None:
        dt = _now_ist()
    return dt.strftime("%H:%M:%S IST")


def _is_trading_day(d: date) -> bool:
    """True if Mon-Fri.

    We do NOT check Indian-market holidays (no reliable hardcoded list in
    engine.events). So this returns True on e.g. Republic Day -- the
    pipeline will just run on stale-but-recent data, which is a no-op-ish
    cost we accept.
    """
    return d.weekday() < 5


def _next_trigger(now: datetime, daily_time: time) -> datetime:
    """Next datetime (>= now) when (a) it's a weekday and (b) the
    clock has reached `daily_time`."""
    target = datetime.combine(now.date(), daily_time, tzinfo=IST)
    if target <= now:
        target = target + timedelta(days=1)
    while not _is_trading_day(target.date()):
        target = target + timedelta(days=1)
        target = datetime.combine(target.date(), daily_time, tzinfo=IST)
    return target


def _wait_until(target_time: time, timezone_=IST) -> None:
    """Sleep until next occurrence of `target_time` on a trading day.

    Wakes at most hourly to recheck the clock -- defensive against DST
    or clock changes mid-sleep.
    """
    while True:
        now = datetime.now(tz=timezone_)
        target = _next_trigger(now, target_time)
        remaining = (target - now).total_seconds()
        if remaining <= 0:
            return
        sleep_s = min(remaining, 3600.0)
        _time_mod.sleep(sleep_s)


# ---------------------------------------------------------------------------
# Subprocess driver
# ---------------------------------------------------------------------------

def _tail(s: Optional[str], n: int = 1000) -> str:
    if not s:
        return ""
    return s[-n:]


def _run_step(name: str, argv_tail: list[str], python_exe: str,
              timeout_s: int = PER_STEP_TIMEOUT_SEC) -> StepResult:
    """Run one module via subprocess. Never raises."""
    cmd = [python_exe] + argv_tail
    t0 = _time_mod.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        dur = _time_mod.monotonic() - t0
        status = "OK" if proc.returncode == 0 else "FAILED"
        return StepResult(
            name=name,
            cmd=cmd,
            status=status,
            duration_s=dur,
            stdout_tail=_tail(proc.stdout),
            stderr_tail=_tail(proc.stderr),
            return_code=proc.returncode,
        )
    except subprocess.TimeoutExpired as e:
        dur = _time_mod.monotonic() - t0
        return StepResult(
            name=name,
            cmd=cmd,
            status="FAILED",
            duration_s=dur,
            stdout_tail=_tail(e.stdout.decode() if isinstance(e.stdout, bytes) else e.stdout),
            stderr_tail=_tail(e.stderr.decode() if isinstance(e.stderr, bytes) else e.stderr),
            return_code=None,
            error=f"TimeoutExpired after {timeout_s}s",
        )
    except subprocess.SubprocessError as e:
        dur = _time_mod.monotonic() - t0
        return StepResult(
            name=name, cmd=cmd, status="FAILED", duration_s=dur,
            return_code=None, error=f"{type(e).__name__}: {e}",
        )
    except OSError as e:
        dur = _time_mod.monotonic() - t0
        return StepResult(
            name=name, cmd=cmd, status="FAILED", duration_s=dur,
            return_code=None, error=f"OSError: {e}",
        )


# ---------------------------------------------------------------------------
# Pipeline driver
# ---------------------------------------------------------------------------

def _resolve_python(python_exe: Optional[str]) -> str:
    if python_exe:
        return python_exe
    env = os.environ.get("OPTIONS_ENGINE_PYTHON")
    if env:
        return env
    if Path(DEFAULT_PYTHON).exists():
        return DEFAULT_PYTHON
    # Fallback: current interpreter (keeps tests / non-Windows usable)
    return sys.executable


def run_pipeline(symbol: str = "NIFTY",
                 python_exe: Optional[str] = None) -> list[StepResult]:
    """Run the 6 modules in order. Returns per-step results.

    Default python: C:\\ProgramData\\miniconda3\\python.exe (override via
    arg or OPTIONS_ENGINE_PYTHON env var).
    """
    py = _resolve_python(python_exe)
    steps = _step_commands(symbol)
    results: list[StepResult] = []
    print(f"[{_ts_str()}] Pipeline start: symbol={symbol} python={py}",
          flush=True)
    for name, argv_tail in steps:
        print(f"[{_ts_str()}] {name}... running", flush=True)
        res = _run_step(name, argv_tail, python_exe=py)
        results.append(res)
        if res.status == "OK":
            print(f"[{_ts_str()}] {name}... OK ({res.duration_s:.1f}s)",
                  flush=True)
        else:
            err_snip = (res.stderr_tail or res.error or "")[-200:]
            err_snip = err_snip.replace("\n", " | ")
            print(f"[{_ts_str()}] {name}... FAILED ({res.duration_s:.1f}s)"
                  f" | stderr: {err_snip}", flush=True)
    ok = sum(1 for r in results if r.status == "OK")
    failed = sum(1 for r in results if r.status == "FAILED")
    print(f"[{_ts_str()}] Pipeline done: {ok}/{len(results)} OK, "
          f"{failed} FAILED", flush=True)
    return results


# ---------------------------------------------------------------------------
# Status file
# ---------------------------------------------------------------------------

def _overall_status(results: list[StepResult]) -> str:
    if not results:
        return "FAILED"
    statuses = [r.status for r in results]
    if all(s == "OK" for s in statuses):
        return "OK"
    if all(s == "FAILED" for s in statuses):
        return "FAILED"
    return "PARTIAL"


def write_status(results: list[StepResult],
                 symbol: str = "NIFTY",
                 run_started_at: Optional[datetime] = None,
                 run_completed_at: Optional[datetime] = None,
                 daily_time: time = time(16, 0),
                 path: Path = STATUS_PATH) -> Path:
    """Persist a JSON status to data/research/rebuild_status.json."""
    if run_completed_at is None:
        run_completed_at = _now_ist()
    if run_started_at is None:
        # Best-effort: subtract sum of step durations
        run_started_at = run_completed_at - timedelta(
            seconds=sum(r.duration_s for r in results)
        )
    next_run = _next_trigger(run_completed_at, daily_time)
    payload = {
        "symbol": symbol,
        "run_started_at": run_started_at.isoformat(),
        "run_completed_at": run_completed_at.isoformat(),
        "overall_status": _overall_status(results),
        "steps": [r.to_status_dict() for r in results],
        "next_run_at": next_run.isoformat(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Daemon loop
# ---------------------------------------------------------------------------

def run_daemon(symbol: str = "NIFTY",
               daily_time: time = time(16, 0),
               python_exe: Optional[str] = None,
               max_runs: Optional[int] = None) -> None:
    """Main loop. Wakes daily at `daily_time` IST on weekdays, runs the
    pipeline, persists status, sleeps until the next trigger.

    `max_runs=None` means run forever; useful for tests / once-off jobs.
    """
    print(f"Research rebuild daemon started "
          f"(symbol={symbol}, daily_time={daily_time.strftime('%H:%M')} IST, "
          f"max_runs={max_runs})", flush=True)
    runs_done = 0
    while True:
        if max_runs is not None and runs_done >= max_runs:
            print(f"[{_ts_str()}] Reached max_runs={max_runs}, exiting.",
                  flush=True)
            return
        try:
            next_at = _next_trigger(_now_ist(), daily_time)
            print(f"[{_ts_str()}] Next run at {next_at.isoformat()}",
                  flush=True)
            _wait_until(daily_time)
        except KeyboardInterrupt:
            print(f"[{_ts_str()}] Interrupted by user during sleep.",
                  flush=True)
            return
        run_started = _now_ist()
        try:
            results = run_pipeline(symbol=symbol, python_exe=python_exe)
            write_status(results, symbol=symbol,
                         run_started_at=run_started,
                         run_completed_at=_now_ist(),
                         daily_time=daily_time)
        except KeyboardInterrupt:
            print(f"[{_ts_str()}] Interrupted by user during pipeline.",
                  flush=True)
            return
        except Exception as e:
            # Defensive: never let an unhandled error kill the daemon
            print(f"[{_ts_str()}] Pipeline crashed: {type(e).__name__}: {e}",
                  flush=True)
        runs_done += 1


# ---------------------------------------------------------------------------
# Status pretty-print
# ---------------------------------------------------------------------------

def _print_status(path: Path = STATUS_PATH) -> int:
    if not path.exists():
        print(f"No status file at {path}. Run "
              f"`python -m engine.research.rebuild_daemon once` first.")
        return 1
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"Could not parse {path}: {e}")
        return 1
    print(f"Status file: {path}")
    print(f"  symbol           : {data.get('symbol')}")
    print(f"  overall_status   : {data.get('overall_status')}")
    print(f"  run_started_at   : {data.get('run_started_at')}")
    print(f"  run_completed_at : {data.get('run_completed_at')}")
    print(f"  next_run_at      : {data.get('next_run_at')}")
    print(f"  steps:")
    for s in data.get("steps", []):
        line = (f"    - {s.get('name'):<18} {s.get('status'):<8}"
                f" {s.get('duration_s'):>7.2f}s  rc={s.get('return_code')}")
        print(line)
        if s.get("status") == "FAILED" and s.get("stderr_tail"):
            tail = s["stderr_tail"].splitlines()[-3:]
            for ln in tail:
                print(f"        {ln}")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_time(s: str) -> time:
    """Parse 'HH:MM' or 'HH:MM:SS' (24h IST)."""
    parts = s.split(":")
    if len(parts) == 2:
        return time(int(parts[0]), int(parts[1]))
    if len(parts) == 3:
        return time(int(parts[0]), int(parts[1]), int(parts[2]))
    raise ValueError(f"Unrecognized time format: {s!r} (use HH:MM)")


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m engine.research.rebuild_daemon",
        description="Daemon that re-runs the research pipeline daily.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run", help="Run the daemon loop forever.")
    pr.add_argument("--symbol", default="NIFTY")
    pr.add_argument("--daily-time", default="16:00",
                    help="HH:MM IST trigger time (default 16:00).")
    pr.add_argument("--python-exe", default=None,
                    help="Python interpreter to launch modules with.")
    pr.add_argument("--max-runs", type=int, default=None,
                    help="Stop after this many runs (default: forever).")

    po = sub.add_parser("once", help="Run one pipeline pass and exit.")
    po.add_argument("--symbol", default="NIFTY")
    po.add_argument("--python-exe", default=None)

    ps = sub.add_parser("status",
                        help="Pretty-print data/research/rebuild_status.json.")

    args = p.parse_args(argv)

    if args.cmd == "run":
        run_daemon(symbol=args.symbol,
                   daily_time=_parse_time(args.daily_time),
                   python_exe=args.python_exe,
                   max_runs=args.max_runs)
        return 0
    if args.cmd == "once":
        started = _now_ist()
        results = run_pipeline(symbol=args.symbol, python_exe=args.python_exe)
        path = write_status(results, symbol=args.symbol,
                            run_started_at=started,
                            run_completed_at=_now_ist())
        print(f"[{_ts_str()}] Wrote status: {path}", flush=True)
        overall = _overall_status(results)
        # Exit 0 if all ok, 1 if partial/failed (useful for cron / smoke tests).
        return 0 if overall == "OK" else 1
    if args.cmd == "status":
        return _print_status()
    return 2


if __name__ == "__main__":
    sys.exit(main())
