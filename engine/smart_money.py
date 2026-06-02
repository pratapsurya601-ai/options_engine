"""
Smart-money bias scoring from FII/DII activity.

Inputs: the latest row(s) from fii_dii_activity. Output: a bias score in
[-1.0, +1.0] where +1 = strongly bullish positioning, -1 = strongly bearish,
plus a human-readable breakdown.

Honest caveat: a SINGLE EOD print is noise. The scorer is most useful with
5+ days of history (rolling net). Until then, treat the score as directional
only, not conviction-grade.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional


@dataclass
class SmartMoneySignal:
    score: float                   # -1.0 (bearish) .. +1.0 (bullish)
    label: str                     # 'strong_bull' | 'bull' | 'neutral' | 'bear' | 'strong_bear'
    components: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def _label_for(score: float) -> str:
    if score >= 0.5: return "strong_bull"
    if score >= 0.15: return "bull"
    if score <= -0.5: return "strong_bear"
    if score <= -0.15: return "bear"
    return "neutral"


def _clip(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def score_latest(
    row: dict,
    cash_scale_cr: float = 5000.0,      # cash flow scale where ±=1 saturates (₹ crore)
    fut_ratio_anchor: float = 1.0,      # L/S ratio anchor
    fut_ratio_scale: float = 0.8,       # how far from anchor to saturate
) -> SmartMoneySignal:
    """
    Score a single fii_dii_activity row. Components combined with weights:

      0.35  FII cash net   — sign and magnitude
      0.15  DII cash net   — counter-weight (DII often takes the other side)
      0.30  FII index-fut long/short ratio vs anchor
      0.20  FII index options bias (calls long - puts long, net)
    """
    notes: list[str] = []
    comps: dict = {}

    fii_cash = float(row.get("fii_cash_net") or 0.0)
    fii_cash_score = _clip(fii_cash / cash_scale_cr)
    comps["fii_cash_net"] = (fii_cash, fii_cash_score)

    dii_cash = float(row.get("dii_cash_net") or 0.0)
    dii_cash_score = _clip(dii_cash / cash_scale_cr)
    comps["dii_cash_net"] = (dii_cash, dii_cash_score)

    fut_long = float(row.get("fii_index_fut_long_contracts") or 0.0)
    fut_short = float(row.get("fii_index_fut_short_contracts") or 0.0)
    if fut_long + fut_short > 0:
        ratio = fut_long / max(fut_short, 1.0)
        fut_score = _clip((ratio - fut_ratio_anchor) / fut_ratio_scale)
    else:
        ratio = None
        fut_score = 0.0
        notes.append("no FII index-fut data")
    comps["fii_index_fut_ratio"] = (ratio, fut_score)

    # Options bias — long calls bullish, long puts bearish.
    # Short side is also informative but noisier; weight half.
    call_long = float(row.get("fii_index_call_long") or 0.0)
    call_short = float(row.get("fii_index_call_short") or 0.0)
    put_long = float(row.get("fii_index_put_long") or 0.0)
    put_short = float(row.get("fii_index_put_short") or 0.0)
    total = call_long + put_long + 0.5 * (call_short + put_short)
    if total > 0:
        raw = (call_long - put_long) + 0.5 * (put_short - call_short)
        opt_score = _clip(raw / total)
    else:
        opt_score = 0.0
        notes.append("no FII index-options data")
    comps["fii_index_options"] = ((call_long, put_long, call_short, put_short), opt_score)

    score = (
        0.35 * fii_cash_score +
        0.15 * dii_cash_score +
        0.30 * fut_score +
        0.20 * opt_score
    )
    score = _clip(score)

    return SmartMoneySignal(
        score=score,
        label=_label_for(score),
        components=comps,
        notes=notes,
    )


def score_rolling(rows: Iterable[dict], days: int = 5) -> SmartMoneySignal:
    """
    Average the per-day scores over the last `days` rows. Rows assumed
    ordered most-recent-first.
    """
    rows = list(rows)[:days]
    if not rows:
        return SmartMoneySignal(score=0.0, label="neutral", notes=["no data"])
    sigs = [score_latest(r) for r in rows]
    avg = sum(s.score for s in sigs) / len(sigs)
    return SmartMoneySignal(
        score=avg,
        label=_label_for(avg),
        components={"window_days": len(sigs), "per_day_scores": [s.score for s in sigs]},
        notes=[f"rolling avg over {len(sigs)} day(s)"],
    )
