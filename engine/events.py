"""
Event-risk calendar for option positioning.

Hardcoded high-impact events that historically cause IV expansion before and
crush after. The recommender uses `events_between(today, expiry)` to:
  - veto debit trades that span an event when IV-rank is already high
  - prefer credit/iron-condor structures that monetize the IV crush
  - surface a warning in the rationale when an event sits inside the trade window

Sources:
  - RBI MPC: 6 scheduled meetings/year, published ~1y in advance
  - US FOMC: 8 meetings/year
  - Union Budget: Feb 1 each year (Interim Budget in election years can shift)
  - Monthly expiry: last Thursday of month (NIFTY) — handled by expiry math, not here
  - Weekly expiry: Tuesdays (NIFTY, post-2025 SEBI norms)

Update this file once a year. Or: scrape RBI/Fed sites in a separate
ingestor and write to an `events` table — saved for later.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Iterable


@dataclass(frozen=True)
class Event:
    name: str
    event_date: date
    impact: str          # 'high' | 'medium' | 'low'
    iv_effect: str       # 'expand_before_crush_after' | 'directional' | 'mixed'
    notes: str = ""


# --- Scheduled events 2026 ---------------------------------------------------
# RBI MPC 2026 dates per RBI's schedule (Feb, Apr, Jun, Aug, Oct, Dec)
_RBI_2026 = [
    date(2026, 2, 6),
    date(2026, 4, 8),
    date(2026, 6, 5),
    date(2026, 8, 7),
    date(2026, 10, 1),
    date(2026, 12, 4),
]

# US FOMC 2026 (publish around mid-month) — verify against Fed calendar
_FOMC_2026 = [
    date(2026, 1, 28),
    date(2026, 3, 18),
    date(2026, 4, 29),
    date(2026, 6, 17),
    date(2026, 7, 29),
    date(2026, 9, 16),
    date(2026, 10, 28),
    date(2026, 12, 16),
]

# Union Budget — typically Feb 1
_BUDGET = [date(2026, 2, 1)]

# Election results / major macro — leave empty; add manually if needed.
_ELECTIONS: list[date] = []


SCHEDULED_EVENTS: list[Event] = [
    *[Event(f"RBI MPC", d, "high", "expand_before_crush_after",
            "Post-decision IV crush typically 5-10 vol points") for d in _RBI_2026],
    *[Event(f"US FOMC", d, "high", "expand_before_crush_after",
            "Overnight gap risk — affects NIFTY at next open") for d in _FOMC_2026],
    *[Event(f"Union Budget", d, "high", "directional",
            "Largest single-day vol event in Indian markets") for d in _BUDGET],
    *[Event(f"Election Result", d, "high", "directional", "") for d in _ELECTIONS],
]


def events_between(start: date, end: date,
                   min_impact: str = "high") -> list[Event]:
    """Return scheduled events with start <= event_date <= end (inclusive both sides)."""
    rank = {"low": 0, "medium": 1, "high": 2}
    threshold = rank[min_impact]
    return [
        e for e in SCHEDULED_EVENTS
        if start <= e.event_date <= end and rank[e.impact] >= threshold
    ]


def is_pre_event_window(today: date, window_days: int = 2) -> list[Event]:
    """Events occurring within `window_days` AFTER today — these inflate IV NOW."""
    end = today + timedelta(days=window_days)
    return events_between(today - timedelta(days=1), end)


@dataclass
class EventRisk:
    spans_event: bool
    events: list[Event]
    pre_event_now: list[Event]   # events imminent — IV likely inflated
    advice: str                   # one-line guidance

    def is_safe_for_debit(self, iv_rank: float | None) -> bool:
        """Debit trades through an event are unsafe if IV is already rich."""
        if not self.spans_event:
            return True
        if iv_rank is None:
            return False  # unknown IV-rank + event in path = avoid
        return iv_rank < 0.50

    def to_rationale(self) -> list[str]:
        out = []
        if self.events:
            names = ", ".join(f"{e.name} on {e.event_date}" for e in self.events)
            out.append(f"EVENT IN TRADE WINDOW: {names}")
        if self.pre_event_now:
            names = ", ".join(f"{e.name} in {(e.event_date - date.today()).days}d"
                              for e in self.pre_event_now)
            out.append(f"PRE-EVENT NOW (IV likely inflated): {names}")
        if self.advice:
            out.append(self.advice)
        return out


def assess(today: date, expiry: date, iv_rank: float | None = None) -> EventRisk:
    """One-shot event-risk assessment for a trade from `today` to `expiry`."""
    spanning = events_between(today, expiry)
    pre_now = is_pre_event_window(today, window_days=2)

    advice = ""
    if spanning and (iv_rank is None or iv_rank >= 0.50):
        advice = ("Avoid debit options across this event window — IV crush will eat "
                  "premium. Prefer credit structures (iron condor, credit vertical) "
                  "to monetize the post-event vol collapse.")
    elif spanning and iv_rank is not None and iv_rank < 0.30:
        advice = ("Event in window but IV is cheap — buying premium here is "
                  "defensible IF directional thesis is strong. Size small.")
    elif pre_now:
        advice = ("Pre-event vol expansion in progress. Wait 1 day after the event "
                  "for IV reset before committing to debit trades.")

    return EventRisk(
        spans_event=bool(spanning),
        events=spanning,
        pre_event_now=pre_now,
        advice=advice,
    )
