"""
Market state — the input that rules evaluate against.

Built up incrementally:
  - spot, IV, chain  (from any chain source)
  - OHLC bars        (intraday from Kite historical or aggregated from ticks)
  - indicators       (VWAP, ATR, RSI computed from bars)
  - time-of-day      (used by event/session rules)

The state object is the only thing rules see. Decouples rule logic from data.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time, date, timedelta, timezone
from typing import Optional


IST = timezone(timedelta(hours=5, minutes=30))


@dataclass
class Bar:
    """5-min OHLC bar."""
    ts: datetime          # bar OPEN time (IST)
    open: float
    high: float
    low: float
    close: float
    volume: int


@dataclass
class MarketState:
    symbol: str
    spot: float
    chain: object | None = None     # Chain from engine.chain (optional)
    bars_5m: list[Bar] = field(default_factory=list)
    ts: datetime = field(default_factory=lambda: datetime.now(tz=IST))

    # ---- session helpers ----
    def session_time(self) -> time:
        return self.ts.astimezone(IST).time()

    def minutes_since_open(self) -> int:
        now = self.session_time()
        if now < time(9, 15):
            return -1
        delta = (now.hour * 60 + now.minute) - (9 * 60 + 15)
        return max(delta, 0)

    def is_market_open(self) -> bool:
        t = self.session_time()
        return time(9, 15) <= t <= time(15, 30)

    # ---- bar accessors ----
    def opening_range(self, minutes: int = 15) -> tuple[float, float] | None:
        """Return (OR_low, OR_high) over the first `minutes` of the session."""
        if not self.bars_5m:
            return None
        session_date = self.ts.astimezone(IST).date()
        open_dt = datetime.combine(session_date, time(9, 15), tzinfo=IST)
        cutoff = open_dt + timedelta(minutes=minutes)
        morning = [b for b in self.bars_5m if open_dt <= b.ts < cutoff]
        if not morning:
            return None
        return (min(b.low for b in morning), max(b.high for b in morning))

    def vwap(self, since_open: bool = True) -> float | None:
        """
        Volume-weighted average price. For NIFTY INDEX bars (which have
        volume=0 from Kite), falls back to plain typical-price average —
        in practice indistinguishable for an index since each bar represents
        the same time-weighted slice.
        """
        if not self.bars_5m:
            return None
        bars = self.bars_5m
        if since_open:
            session_date = self.ts.astimezone(IST).date()
            open_dt = datetime.combine(session_date, time(9, 15), tzinfo=IST)
            bars = [b for b in bars if b.ts >= open_dt]
        if not bars:
            return None
        typical = [(b.high + b.low + b.close) / 3 for b in bars]
        total_vol = sum(b.volume for b in bars)
        if total_vol == 0:
            # Index data — equal-weight typical price (time-weighted avg)
            return sum(typical) / len(typical)
        return sum(tp * b.volume for tp, b in zip(typical, bars)) / total_vol

    def last_bar(self) -> Bar | None:
        return self.bars_5m[-1] if self.bars_5m else None

    def latest_n_bars(self, n: int) -> list[Bar]:
        return self.bars_5m[-n:]

    def volume_z(self, lookback: int = 12) -> float | None:
        """Z-score of latest bar's volume vs prior N bars. Detects volume spikes."""
        if len(self.bars_5m) < lookback + 1:
            return None
        recent = self.bars_5m[-lookback-1:-1]
        avg = sum(b.volume for b in recent) / lookback
        std = (sum((b.volume - avg) ** 2 for b in recent) / lookback) ** 0.5
        if std == 0:
            return 0.0
        return (self.bars_5m[-1].volume - avg) / std

    def range_z(self, lookback: int = 12) -> float | None:
        """
        Z-score of latest bar's RANGE (high-low) vs prior N bars. Use as a
        volume proxy when volume data is unreliable (e.g. NIFTY index bars).
        Range correlates ~0.7 with volume on liquid equities.
        """
        if len(self.bars_5m) < lookback + 1:
            return None
        recent = self.bars_5m[-lookback-1:-1]
        ranges = [b.high - b.low for b in recent]
        avg = sum(ranges) / lookback
        var = sum((r - avg) ** 2 for r in ranges) / lookback
        std = var ** 0.5
        if std == 0:
            return 0.0
        latest = self.bars_5m[-1].high - self.bars_5m[-1].low
        return (latest - avg) / std

    def activity_z(self, lookback: int = 12) -> float | None:
        """
        Best-available activity z-score: volume_z when volumes are nonzero,
        else range_z (volume proxy). One call, rule doesn't have to choose.
        """
        if len(self.bars_5m) >= lookback + 1:
            recent_vol = sum(b.volume for b in self.bars_5m[-lookback-1:-1])
            if recent_vol > 0:
                return self.volume_z(lookback)
        return self.range_z(lookback)
