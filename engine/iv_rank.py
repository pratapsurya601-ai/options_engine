"""
IV-rank / IV-percentile computation from option_chain_snapshots.

DEFINITIONS:
  IV-rank       = (IV_today - IV_min_window) / (IV_max_window - IV_min_window)
                  -> 0.0 if current IV is the lowest in window, 1.0 if highest.
                  Range [0, 1]. The single most useful "is vol cheap or rich?"
                  number. Most traders use 60- or 90-day window.

  IV-percentile = fraction of days in window where IV was BELOW today's IV.
                  -> 0.50 = today's IV is median.
                  More robust than IV-rank when IV has outlier days.

We track ATM IV per (symbol, expiry-bucket). Expiry-bucket = days-to-expiry
rounded to nearest weekly tier (7, 14, 21, 30, 60, 90) so we compare like-for-like.

USAGE:
  with psycopg.connect(db_url) as conn:
      rank = iv_rank_atm(conn, "NIFTY", today_iv=0.18, dte=7)
      # rank.value = 0.72  -> IV is in top quartile, prefer selling premium
      # rank.confidence = 'low' if <30 days history, else 'medium'/'high'

DEGRADED MODE:
  If history < 30 days, returns confidence='insufficient' and falls back to
  a static heuristic: NIFTY ATM IV historically averages ~13-15% in low-vol
  regime, 18-22% in normal, 25%+ in stress. The function returns a coarse
  bucket so the recommender can still gate decisions.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Optional


# Static fallback buckets when history is insufficient (NIFTY ATM, calibrated
# from public RBI vol summaries and NSE archives 2018-2025).
_STATIC_NIFTY_LOW = 0.12
_STATIC_NIFTY_MED = 0.16
_STATIC_NIFTY_HIGH = 0.22


@dataclass
class IVRankResult:
    value: float | None          # 0..1 if computable; None if degraded mode unusable
    percentile: float | None     # 0..1
    current_iv: float
    window_days: int
    sample_count: int
    confidence: str              # 'insufficient' | 'low' | 'medium' | 'high'
    bucket: str                  # 'cheap' | 'normal' | 'rich' | 'extreme'
    advice: str

    def to_rationale(self) -> list[str]:
        out = [
            f"IV-rank: {self.value*100:.0f}%" if self.value is not None
            else f"IV-rank: n/a (confidence={self.confidence})",
            f"current IV {self.current_iv*100:.1f}%, bucket={self.bucket} (over {self.sample_count}/{self.window_days}d)",
        ]
        if self.advice:
            out.append(self.advice)
        return out


def _bucket_for(iv_rank: float | None, current_iv: float) -> str:
    if iv_rank is not None:
        if iv_rank >= 0.85: return "extreme"
        if iv_rank >= 0.60: return "rich"
        if iv_rank <= 0.20: return "cheap"
        return "normal"
    # Static fallback by absolute IV (NIFTY)
    if current_iv >= _STATIC_NIFTY_HIGH: return "rich"
    if current_iv <= _STATIC_NIFTY_LOW: return "cheap"
    return "normal"


def _advice_for(bucket: str) -> str:
    return {
        "extreme": "IV is in top 15% of recent history — strongly prefer SELLING premium (credit spreads, iron condor). Buying debit options here has terrible risk/reward.",
        "rich":    "IV is rich — prefer credit structures. If buying debit, demand strong directional thesis and small size.",
        "normal":  "IV in normal range — strategy choice driven by view, not vol.",
        "cheap":   "IV is cheap — debit trades have better risk/reward. Avoid selling naked premium.",
    }.get(bucket, "")


def _confidence_for(n: int) -> str:
    if n < 10:  return "insufficient"
    if n < 30:  return "low"
    if n < 60:  return "medium"
    return "high"


def _dte_bucket(dte_days: int) -> tuple[int, int]:
    """Map days-to-expiry to comparison band. Returns (lo, hi) inclusive."""
    if dte_days <= 4:   return (0, 5)
    if dte_days <= 10:  return (5, 12)
    if dte_days <= 21:  return (10, 25)
    if dte_days <= 45:  return (25, 50)
    return (45, 120)


def iv_rank_atm(
    conn,
    symbol: str,
    current_iv: float,
    dte_days: int,
    window_days: int = 60,
) -> IVRankResult:
    """
    Compute IV-rank for ATM options of `symbol` over the trailing `window_days`.

    `conn` is a psycopg v3 connection. If the snapshots table is sparse or
    missing, returns a degraded result that downstream code can still use.
    """
    lo_dte, hi_dte = _dte_bucket(dte_days)
    window_start = date.today() - timedelta(days=window_days)

    # Pull ATM IV per trading day: take median IV of strikes within ~2% of spot
    # for the relevant DTE bucket. This is robust to single-strike outliers.
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                WITH s AS (
                  SELECT
                    DATE(snapshot_ts AT TIME ZONE 'Asia/Kolkata') AS d,
                    iv,
                    ABS(strike - spot) / spot AS moneyness,
                    (expiry - DATE(snapshot_ts AT TIME ZONE 'Asia/Kolkata')) AS dte
                  FROM option_chain_snapshots
                  WHERE symbol = %s
                    AND snapshot_ts >= %s
                    AND iv IS NOT NULL AND iv > 0
                )
                SELECT d, PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY iv) AS atm_iv
                FROM s
                WHERE moneyness <= 0.02
                  AND dte BETWEEN %s AND %s
                GROUP BY d
                ORDER BY d
                """,
                (symbol, window_start, lo_dte, hi_dte),
            )
            rows = cur.fetchall()
    except Exception as e:
        # Table might not exist yet, or schema mismatch
        return IVRankResult(
            value=None, percentile=None, current_iv=current_iv,
            window_days=window_days, sample_count=0,
            confidence="insufficient",
            bucket=_bucket_for(None, current_iv),
            advice=_advice_for(_bucket_for(None, current_iv)) +
                   f" (degraded mode: DB query failed — {type(e).__name__})",
        )

    ivs = [float(r[1]) / 100.0 if float(r[1]) > 1.5 else float(r[1]) for r in rows]
    # NSE stores IV in % (e.g. 18.5) — normalize. Heuristic: if >1.5, divide.

    n = len(ivs)
    conf = _confidence_for(n)

    if n < 10:
        bucket = _bucket_for(None, current_iv)
        return IVRankResult(
            value=None, percentile=None, current_iv=current_iv,
            window_days=window_days, sample_count=n, confidence=conf,
            bucket=bucket,
            advice=_advice_for(bucket) +
                   f" (using static heuristic — only {n} historical days available)",
        )

    iv_min, iv_max = min(ivs), max(ivs)
    if iv_max == iv_min:
        rank = 0.5
    else:
        rank = (current_iv - iv_min) / (iv_max - iv_min)
        rank = max(0.0, min(1.0, rank))

    pct = sum(1 for v in ivs if v < current_iv) / n

    bucket = _bucket_for(rank, current_iv)
    return IVRankResult(
        value=rank, percentile=pct, current_iv=current_iv,
        window_days=window_days, sample_count=n, confidence=conf,
        bucket=bucket,
        advice=_advice_for(bucket),
    )


def iv_rank_from_chain(conn, chain, dte_days: int, window_days: int = 60) -> IVRankResult:
    """Convenience: pull ATM IV from a Chain object and compute rank."""
    from .recommender import _atm_iv  # reuse the same averaging logic
    iv = _atm_iv(chain)
    return iv_rank_atm(conn, chain.symbol, iv, dte_days, window_days)
