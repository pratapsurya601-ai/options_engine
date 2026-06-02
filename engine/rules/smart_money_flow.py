"""
Smart-Money Flow rule — intraday directional bias from FII/DII positioning.

Premise:
  Strong, persistent FII flow (cash + index-futures L/S + index-options bias)
  over the past few days has measurable short-horizon predictive power on
  NIFTY's next session direction. We use engine.smart_money.score_rolling()
  to convert recent fii_dii_activity rows into a score in [-1, +1] and bias
  intraday CE/PE entries accordingly.

Entry (within configured time window, default 9:45 - 12:00 IST):
  - Read last N days of fii_dii_activity (DB -> JSON fallback -> none).
  - Compute rolling smart-money score.
  - If |score| >= score_threshold:
      score >= +thr -> bullish bias -> BUY CE (ATM)
      score <= -thr -> bearish bias -> BUY PE (ATM)
  - Optionally require price-action confirmation:
      bullish: last 3 closes monotonically rising AND last close > EMA20
      bearish: last 3 closes monotonically falling AND last close < EMA20

Exits:
  - ATR(14)-based target/stop on spot, mapped to premium via |delta|.
  - Time exit at hold_minutes.
  - Trailing stop hints in trigger_context (executed by harness, not here).

Honest caveats:
  - FII/DII data is EOD; single prints are noisy. Rolling window of 3-5 days
    smooths some of that but still small-sample.
  - The rule does NOT fire if FII/DII data isn't available — degrades silently.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, time
from pathlib import Path
from typing import Callable, Optional

from ..signals import BaseRule, Signal
from ..state import MarketState
from ..pricing import black_scholes
from ..indicators import ema
from ..smart_money import score_rolling, SmartMoneySignal


# ---------------------------------------------------------------------------
# Default FII/DII row provider: DB -> JSON file -> empty list
# ---------------------------------------------------------------------------
def _default_fii_dii_provider(rolling_days: int) -> list[dict]:
    """
    Return up to `rolling_days` fii_dii_activity rows newest-first.

    Tries Postgres via DATABASE_URL first; on any failure falls back to
    a manual JSON file at data/fii_dii_manual.json relative to the project
    root. Returns [] if neither is available.
    """
    rows: list[dict] = []
    db_url = os.environ.get("DATABASE_URL")
    if db_url:
        try:
            import psycopg
            from psycopg.rows import dict_row
            with psycopg.connect(db_url) as conn:
                with conn.cursor(row_factory=dict_row) as cur:
                    cur.execute(
                        "SELECT * FROM fii_dii_activity "
                        "ORDER BY trade_date DESC LIMIT %s",
                        (rolling_days,),
                    )
                    rows = list(cur.fetchall())
            if rows:
                return rows
        except Exception as e:
            print(f"smart_money_flow: DB read failed ({type(e).__name__}: {e}); "
                  "falling back to JSON",
                  file=sys.stderr)

    # JSON fallback — project_root/data/fii_dii_manual.json
    # engine/rules/smart_money_flow.py -> parents[2] = project root
    try:
        project_root = Path(__file__).resolve().parents[2]
        json_path = project_root / "data" / "fii_dii_manual.json"
        if json_path.exists():
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                # Sort newest-first by trade_date if present
                try:
                    data.sort(key=lambda r: str(r.get("trade_date", "")), reverse=True)
                except Exception:
                    pass
                return data[:rolling_days]
    except Exception as e:
        print(f"smart_money_flow: JSON fallback failed ({type(e).__name__}: {e})",
              file=sys.stderr)

    return []


class SmartMoneyFlow(BaseRule):
    name = "smart_money_flow"
    cooldown_minutes = 240   # one fire per half-session max

    def __init__(
        self,
        time_window_start: int = 30,        # 9:45
        time_window_end: int = 165,         # 12:00
        rolling_window_days: int = 3,
        score_threshold: float = 0.35,
        require_price_confirmation: bool = True,
        ema_fast: int = 20,
        target_atr_mult: float = 2.0,
        stop_atr_mult: float = 1.2,
        hold_minutes: int = 90,
        strike_step: int = 50,
        strike_offset_steps: int = 0,
        trail_activation_pts: float = 8.0,
        trail_distance_pts: float = 5.0,
        r: float = 0.065,
        data_provider: Optional[Callable[[int], list[dict]]] = None,
    ):
        super().__init__()
        self.time_window_start = time_window_start
        self.time_window_end = time_window_end
        self.rolling_window_days = rolling_window_days
        self.score_threshold = score_threshold
        self.require_price_confirmation = require_price_confirmation
        self.ema_fast = ema_fast
        self.target_atr_mult = target_atr_mult
        self.stop_atr_mult = stop_atr_mult
        self.hold_minutes = hold_minutes
        self.strike_step = strike_step
        self.strike_offset_steps = strike_offset_steps
        self.trail_activation_pts = trail_activation_pts
        self.trail_distance_pts = trail_distance_pts
        self.r = r
        self.data_provider = data_provider or _default_fii_dii_provider

        # Cache the smart-money signal per session-date to avoid re-querying
        # the DB on every 5-min bar. Key = date.
        self._sm_cache: dict = {}

    # -----------------------------------------------------------------
    def _get_smart_money(self, session_date) -> SmartMoneySignal | None:
        """Cached per-day FII/DII lookup. Returns None on no data."""
        if session_date in self._sm_cache:
            return self._sm_cache[session_date]
        try:
            rows = self.data_provider(self.rolling_window_days)
        except Exception as e:
            print(f"smart_money_flow: data_provider raised "
                  f"({type(e).__name__}: {e})", file=sys.stderr)
            rows = []
        if not rows:
            self._sm_cache[session_date] = None
            return None
        try:
            sig = score_rolling(rows, days=self.rolling_window_days)
        except Exception as e:
            print(f"smart_money_flow: scoring failed ({type(e).__name__}: {e})",
                  file=sys.stderr)
            sig = None
        self._sm_cache[session_date] = sig
        return sig

    # -----------------------------------------------------------------
    def _price_confirmation(self, bars, ema_fast_series, direction: str) -> bool:
        """Last 3 closes trending + last close vs EMA20."""
        if len(bars) < 3 or ema_fast_series[-1] is None:
            return False
        c1, c2, c3 = bars[-3].close, bars[-2].close, bars[-1].close
        last_ema = ema_fast_series[-1]
        if direction == "bullish":
            return c1 < c2 < c3 and c3 > last_ema
        else:
            return c1 > c2 > c3 and c3 < last_ema

    # -----------------------------------------------------------------
    def _check(self, state: MarketState) -> Signal | None:
        if not state.is_market_open():
            return None
        mso = state.minutes_since_open()
        if mso < self.time_window_start or mso > self.time_window_end:
            return None

        # Need enough bars for ATR(14) + EMA(20) + 3 confirmation bars
        bars = state.bars_5m
        min_bars = max(self.ema_fast + 2, 14 + 2, 3)
        if len(bars) < min_bars:
            return None

        session_date = state.ts.astimezone().date() if state.ts.tzinfo else state.ts.date()
        sm = self._get_smart_money(session_date)
        if sm is None:
            return None    # no data -> don't fire

        score = sm.score
        if abs(score) < self.score_threshold:
            return None

        direction = "bullish" if score > 0 else "bearish"
        opt_type = "CE" if direction == "bullish" else "PE"

        # Price-action confirmation
        closes = [b.close for b in bars]
        ef = ema(closes, self.ema_fast)
        confirmation_passed = True
        if self.require_price_confirmation:
            confirmation_passed = self._price_confirmation(bars, ef, direction)
            if not confirmation_passed:
                return None

        spot = state.spot
        if state.chain is None:
            return None

        atm_base = round(spot / self.strike_step) * self.strike_step
        if direction == "bullish":
            atm = atm_base + self.strike_offset_steps * self.strike_step
        else:
            atm = atm_base - self.strike_offset_steps * self.strike_step

        by_strike = state.chain.by_strike()
        q = by_strike.get(atm, {}).get(opt_type)
        if q is None or q.iv is None or not q.ltp:
            avail = [k for k in by_strike if opt_type in by_strike[k]
                     and by_strike[k][opt_type].iv is not None]
            if not avail:
                return None
            atm = min(avail, key=lambda k: abs(k - spot))
            q = by_strike[atm][opt_type]

        # Time to expiry
        from ..chain import IST as CHAIN_IST
        expiry_close = datetime.combine(state.chain.expiry, time(15, 30),
                                        tzinfo=CHAIN_IST)
        t_expiry = max((expiry_close - state.ts).total_seconds()
                       / (365 * 24 * 3600), 1e-5)

        g = black_scholes(spot, atm, t_expiry, q.iv, r=self.r, q=0.0,
                          option_type=opt_type)
        entry_premium = g.price

        # ATR(14) from bar high-low
        atr_period = 14
        recent = bars[-atr_period:] if len(bars) >= atr_period else bars
        atr_14 = sum(b.high - b.low for b in recent) / max(len(recent), 1)

        target_spot_move = self.target_atr_mult * atr_14
        stop_spot_move = self.stop_atr_mult * atr_14
        delta_abs = max(abs(g.delta), 0.05)
        target_premium_gain = delta_abs * target_spot_move
        stop_premium_loss = delta_abs * stop_spot_move

        target_premium = entry_premium + target_premium_gain
        stop_premium = max(0.5, entry_premium - stop_premium_loss)
        if direction == "bullish":
            target_spot = spot + target_spot_move
            stop_spot = spot - stop_spot_move
        else:
            target_spot = spot - target_spot_move
            stop_spot = spot + stop_spot_move

        # Components are tuples (raw, score) — make JSON-friendly
        comp_summary = {}
        for k, v in (sm.components or {}).items():
            try:
                if isinstance(v, tuple) and len(v) == 2:
                    comp_summary[k] = {"raw": v[0], "score": v[1]}
                else:
                    comp_summary[k] = v
            except Exception:
                comp_summary[k] = str(v)

        rr = target_premium_gain / max(stop_premium_loss, 0.1)

        thesis = (
            f"Smart-money flow {direction}: FII/DII rolling-{self.rolling_window_days}d "
            f"score={score:+.2f} ({sm.label}), |score|>={self.score_threshold}. "
            f"Confirmation: {'PASS' if confirmation_passed else 'SKIPPED'} "
            f"(last-3 closes trend + EMA{self.ema_fast}={ef[-1]:.0f}). "
            f"BUY {opt_type} {atm} @ ~Rs {entry_premium:.0f} (delta={g.delta:.2f}, IV={q.iv*100:.1f}%). "
            f"TARGET: +{target_premium_gain:.0f}pts premium (=Rs {target_premium:.0f}) "
            f"when spot reaches {target_spot:.0f} (ATR={atr_14:.1f} x {self.target_atr_mult}). "
            f"STOP: -{stop_premium_loss:.0f}pts (=Rs {stop_premium:.0f}) at spot {stop_spot:.0f} "
            f"(ATR x {self.stop_atr_mult}). "
            f"TRAIL: after +{self.trail_activation_pts}pts, trail {self.trail_distance_pts}pts below peak. "
            f"TIME EXIT: {self.hold_minutes}m. R/R={rr:.2f}. "
            f"Premise: persistent FII positioning biases next-session direction; "
            f"price-action gate filters fade days."
        )

        return Signal(
            rule_name=self.name,
            ts=state.ts,
            direction=direction,
            action=f"BUY_{opt_type}",
            strike=atm,
            option_type=opt_type,
            target_premium_gain=round(target_premium_gain, 1),
            target_spot=round(target_spot, 1),
            stop_spot=round(stop_spot, 1),
            stop_premium=round(stop_premium, 2),
            hold_minutes=self.hold_minutes,
            prob_estimates={},
            trigger_context={
                "smart_money_score": round(score, 4),
                "smart_money_label": sm.label,
                "score_components": comp_summary,
                "direction_confidence": round(abs(score), 4),
                "confirmation_passed": confirmation_passed,
                "rolling_window_days": self.rolling_window_days,
                "score_threshold": self.score_threshold,
                "entry_premium": round(entry_premium, 2),
                "target_premium": round(target_premium, 2),
                "stop_premium": round(stop_premium, 2),
                "atr_14": round(atr_14, 2),
                "target_atr_mult": self.target_atr_mult,
                "stop_atr_mult": self.stop_atr_mult,
                "spot": spot,
                "atm_strike": atm,
                "iv": q.iv,
                "delta": round(g.delta, 4),
                "ema_fast": ef[-1],
                "trail_activation_pts": self.trail_activation_pts,
                "trail_distance_pts": self.trail_distance_pts,
                "risk_reward": round(rr, 2),
                "time_exit_minutes": self.hold_minutes,
            },
            thesis=thesis[:1900],
        )
