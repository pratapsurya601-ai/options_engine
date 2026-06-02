"""
Signal + Rule framework.

A Rule is a pure function of MarketState -> Optional[Signal]. Rules NEVER
fetch data themselves — the watcher feeds them state. This makes rules
testable in isolation and replayable from historical state snapshots.

A Signal is what fires when a rule triggers. It carries:
  - the rule name and trigger context
  - the recommended action (strike, type, target, stop)
  - probability estimates under the touch model
  - a thesis string that goes to the paper-trade log
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Protocol

from .state import MarketState


@dataclass
class Signal:
    rule_name: str
    ts: datetime
    direction: str               # 'bullish' | 'bearish' | 'neutral'
    action: str                  # 'BUY_CE' | 'BUY_PE' | 'SELL_STRADDLE' | etc.
    strike: int | None
    option_type: str | None      # CE | PE | None for multi-leg
    target_premium_gain: float | None    # points
    target_spot: float | None
    stop_spot: float | None
    stop_premium: float | None
    hold_minutes: int | None
    prob_estimates: dict = field(default_factory=dict)   # {target_pts: p_touch}
    trigger_context: dict = field(default_factory=dict)  # what made it fire
    thesis: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["ts"] = self.ts.isoformat()
        return d


class Rule(Protocol):
    """A rule is any object with name + evaluate(state) -> Optional[Signal]."""
    name: str
    def evaluate(self, state: MarketState) -> Signal | None: ...


class BaseRule:
    """Convenience base — most rules subclass this."""
    name: str = "base"
    cooldown_minutes: int = 30   # don't re-fire within this window

    def __init__(self):
        self._last_fired: dict[str, datetime] = {}

    def evaluate(self, state: MarketState) -> Signal | None:
        if self._in_cooldown(state):
            return None
        sig = self._check(state)
        if sig is not None:
            self._last_fired[self.name] = state.ts
        return sig

    def _check(self, state: MarketState) -> Signal | None:
        raise NotImplementedError

    def _in_cooldown(self, state: MarketState) -> bool:
        last = self._last_fired.get(self.name)
        if last is None:
            return False
        from datetime import timedelta
        return (state.ts - last) < timedelta(minutes=self.cooldown_minutes)
