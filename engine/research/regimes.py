"""
Market regime detection for NIFTY (and similar) bar series.

Classifies every trading day into one of four regimes:

    TREND          — strong directional move (ADX high, EMA slope significant)
    MEAN_REVERSION — chop / no edge in either direction (ADX low, mid ATR)
    PANIC          — vol spike / cascading down days (top-percentile vol or 3 red bars)
    LOW_VOL        — compressed range (bottom percentile ATR + realized vol)

These labels are then attached to each timestamp so downstream modules
(strategy selectors, edge backtests) can compute regime-conditional
expectancy.

Design notes
------------
* Indicators (ATR, ADX) are implemented from scratch so we don't take a
  TA-Lib dependency. EMA is reused from `engine.indicators`.
* Percentile thresholds (for PANIC / LOW_VOL gates) are computed from an
  EXPANDING WINDOW of historical features only — never the full dataset.
  This prevents look-ahead bias when the timeline is consumed by a backtest.
* IV-rank is best-effort: until enough chain snapshot history exists, it
  falls back to 0.5 (neutral). The `realized_to_implied` feature is exposed
  so the labels can be re-derived once snapshot data is richer.

CLI
---
    python -m engine.research.regimes build --symbol NIFTY --interval day
    python -m engine.research.regimes show  --symbol NIFTY --date 2026-06-02
    python -m engine.research.regimes stats --symbol NIFTY
"""
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from ..indicators import ema


IST = timezone(timedelta(hours=5, minutes=30))

REGIMES = ("TREND", "MEAN_REVERSION", "PANIC", "LOW_VOL")

STORAGE_DIR = Path("data/research/regimes")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class RegimeConfig:
    """All tunables for regime classification.

    The defaults are calibrated for NIFTY daily bars. For other symbols /
    intervals you'll likely want to re-tune the percentile gates.
    """
    # Indicator windows
    atr_period: int = 14
    adx_period: int = 14
    realized_vol_days: int = 5
    iv_rank_window_days: int = 30
    ema_slope_lookback: int = 5  # bars; slope = (EMA20_t - EMA20_{t-5}) / EMA20_{t-5}
    ema_period: int = 20

    # PANIC gates
    panic_realized_vol_percentile: float = 0.75
    panic_atr_percentile: float = 0.90
    panic_consecutive_red_threshold: float = -0.015  # 3 days each <= -1.5%
    panic_consecutive_red_count: int = 3

    # TREND gates
    trend_adx_min: float = 25.0
    trend_slope_min_abs: float = 0.005  # 0.5% over `ema_slope_lookback` bars

    # MEAN_REVERSION gates
    mean_rev_adx_max: float = 20.0
    mean_rev_atr_percentile_lo: float = 0.25
    mean_rev_atr_percentile_hi: float = 0.75

    # LOW_VOL gates
    low_vol_atr_percentile: float = 0.25
    low_vol_realized_vol_percentile: float = 0.25

    # Warmup — number of bars before we'll attempt classification
    warmup_bars: int = 30


# ---------------------------------------------------------------------------
# Indicators (ATR, ADX) implemented from scratch
# ---------------------------------------------------------------------------

def true_range(highs: Sequence[float], lows: Sequence[float],
               closes: Sequence[float]) -> list[float | None]:
    """TR_t = max(H-L, |H - C_{t-1}|, |L - C_{t-1}|).

    The first value is None because TR needs the prior close.
    """
    out: list[float | None] = [None]
    for i in range(1, len(highs)):
        hl = highs[i] - lows[i]
        hc = abs(highs[i] - closes[i - 1])
        lc = abs(lows[i] - closes[i - 1])
        out.append(max(hl, hc, lc))
    return out


def _wilder_smooth(values: Sequence[float | None], period: int) -> list[float | None]:
    """Wilder's smoothing: first value is the simple average of the first
    `period` non-None entries; subsequent values use
    prev * (period-1)/period + curr/period.
    """
    out: list[float | None] = [None] * len(values)
    # Find the first index where we have `period` non-None values to seed
    buf: list[float] = []
    seed_idx: int | None = None
    for i, v in enumerate(values):
        if v is None:
            continue
        buf.append(float(v))
        if len(buf) == period:
            seed = sum(buf) / period
            out[i] = seed
            seed_idx = i
            break
    if seed_idx is None:
        return out
    prev = out[seed_idx]
    for j in range(seed_idx + 1, len(values)):
        v = values[j]
        if v is None:
            out[j] = prev  # carry forward — TR is None only at idx 0 normally
            continue
        prev = (prev * (period - 1) + float(v)) / period
        out[j] = prev
    return out


def atr(highs: Sequence[float], lows: Sequence[float],
        closes: Sequence[float], period: int = 14) -> list[float | None]:
    """Wilder's ATR. Returns None until `period+1` bars are available."""
    tr = true_range(highs, lows, closes)
    return _wilder_smooth(tr, period)


def adx(highs: Sequence[float], lows: Sequence[float],
        closes: Sequence[float], period: int = 14) -> list[float | None]:
    """Wilder's ADX. Returns a list aligned with the input bars.

    Computes +DM, -DM, TR; smooths each via Wilder; then DI+, DI-, DX, and
    finally ADX (Wilder smoothing of DX).
    """
    n = len(highs)
    if n < 2:
        return [None] * n

    plus_dm: list[float | None] = [None]
    minus_dm: list[float | None] = [None]
    for i in range(1, n):
        up_move = highs[i] - highs[i - 1]
        down_move = lows[i - 1] - lows[i]
        plus_dm.append(up_move if (up_move > down_move and up_move > 0) else 0.0)
        minus_dm.append(down_move if (down_move > up_move and down_move > 0) else 0.0)

    tr = true_range(highs, lows, closes)
    smooth_tr = _wilder_smooth(tr, period)
    smooth_plus = _wilder_smooth(plus_dm, period)
    smooth_minus = _wilder_smooth(minus_dm, period)

    dx: list[float | None] = []
    for i in range(n):
        atr_v = smooth_tr[i]
        p = smooth_plus[i]
        m = smooth_minus[i]
        if atr_v is None or p is None or m is None or atr_v == 0:
            dx.append(None)
            continue
        plus_di = 100.0 * p / atr_v
        minus_di = 100.0 * m / atr_v
        denom = plus_di + minus_di
        if denom == 0:
            dx.append(0.0)
        else:
            dx.append(100.0 * abs(plus_di - minus_di) / denom)

    return _wilder_smooth(dx, period)


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def realized_vol_annualized(closes: Sequence[float], window: int) -> float | None:
    """Stdev of log returns over the last `window+1` closes, annualized (252)."""
    if len(closes) < window + 1:
        return None
    rets: list[float] = []
    for i in range(len(closes) - window, len(closes)):
        prev = closes[i - 1]
        cur = closes[i]
        if prev <= 0 or cur <= 0:
            continue
        rets.append(math.log(cur / prev))
    if len(rets) < 2:
        return None
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
    return math.sqrt(var) * math.sqrt(252.0)


def percentile_rank(value: float, sample: Iterable[float]) -> float:
    """Fraction of `sample` strictly less than `value`. Empty -> 0.5."""
    arr = [s for s in sample if s is not None and not math.isnan(s)]
    if not arr:
        return 0.5
    below = sum(1 for s in arr if s < value)
    return below / len(arr)


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------

def _compute_features_from_bars(
    df: pd.DataFrame,
    config: RegimeConfig,
    iv_rank_value: float | None = None,
) -> dict:
    """Compute the latest-bar feature dict from a daily-bar DataFrame.

    Returns NaN-valued floats (not None) so downstream pandas ops stay vectorized.
    """
    highs = df["high"].astype(float).tolist()
    lows = df["low"].astype(float).tolist()
    closes = df["close"].astype(float).tolist()
    n = len(closes)

    spot = closes[-1] if n else float("nan")

    atr_series = atr(highs, lows, closes, period=config.atr_period)
    atr_v = atr_series[-1] if n else None
    atr_pct = (atr_v / spot * 100.0) if (atr_v and spot) else float("nan")

    adx_series = adx(highs, lows, closes, period=config.adx_period)
    adx_v = adx_series[-1] if n else None

    rv = realized_vol_annualized(closes, config.realized_vol_days)
    rv_val = rv if rv is not None else float("nan")

    ema_series = ema(closes, config.ema_period)
    slope = float("nan")
    lb = config.ema_slope_lookback
    if n > lb and ema_series[-1] is not None and ema_series[-1 - lb] is not None:
        e_now = ema_series[-1]
        e_past = ema_series[-1 - lb]
        if e_past:
            slope = (e_now - e_past) / e_past

    iv_rank = iv_rank_value if iv_rank_value is not None else 0.5

    # realized_to_implied: realized_vol / atm_iv. Without a real ATM IV series
    # we treat iv_rank as a proxy is wrong — leave as NaN unless we have IV.
    realized_to_implied = float("nan")

    return {
        "atr_pct": atr_pct if atr_pct is not None else float("nan"),
        "realized_vol_5d": rv_val,
        "iv_rank": float(iv_rank),
        "ema_slope_20": slope,
        "adx_14": float(adx_v) if adx_v is not None else float("nan"),
        "realized_to_implied": realized_to_implied,
    }


def compute_regime_features(
    symbol: str = "NIFTY",
    as_of: datetime | None = None,
    config: RegimeConfig | None = None,
) -> dict:
    """Compute the regime feature dict for `symbol` as of `as_of` (default: latest).

    Reads daily bars from cache; returns NaN for unavailable values.
    """
    from ..data.cache import load_cache

    cfg = config or RegimeConfig()
    df = load_cache(symbol, "day")
    if df.empty:
        return {
            "atr_pct": float("nan"),
            "realized_vol_5d": float("nan"),
            "iv_rank": 0.5,
            "ema_slope_20": float("nan"),
            "adx_14": float("nan"),
            "realized_to_implied": float("nan"),
        }
    df = df.sort_values("ts").reset_index(drop=True)
    if as_of is not None:
        ts_as_of = pd.Timestamp(as_of)
        if ts_as_of.tz is None and df["ts"].dt.tz is not None:
            ts_as_of = ts_as_of.tz_localize(IST)
        df = df[df["ts"] <= ts_as_of].reset_index(drop=True)

    return _compute_features_from_bars(df, cfg, iv_rank_value=_safe_iv_rank(symbol))


def _safe_iv_rank(symbol: str) -> float | None:
    """Best-effort IV-rank lookup. Returns None if unavailable so caller can
    fall back to neutral 0.5. This will become useful as chain snapshot
    history accumulates; until then it almost always returns None.
    """
    try:
        # Lazy import — don't take a DB dependency for pure regime work
        import os
        if not os.environ.get("DATABASE_URL"):
            return None
        import psycopg  # type: ignore
        from ..iv_rank import iv_rank_atm  # type: ignore
        # We don't have current_iv handy here without a chain — skip.
        return None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def _is_finite(x: float | None) -> bool:
    return x is not None and isinstance(x, (int, float)) and not math.isnan(float(x))


def classify_regime(
    features: dict,
    historical_features: pd.DataFrame,
    config: RegimeConfig | None = None,
    recent_returns: Sequence[float] | None = None,
) -> str:
    """Classify a single bar's features into one of the four regimes.

    Parameters
    ----------
    features : dict
        Output of `compute_regime_features` (or one row of the timeline).
    historical_features : pd.DataFrame
        Past feature rows used to compute percentile gates. Must contain
        columns `atr_pct` and `realized_vol_5d`. Use *only* rows with
        timestamp <= the bar being classified (no look-ahead).
    config : RegimeConfig
    recent_returns : sequence of last N daily returns (most recent last).
        Used for the "3 consecutive red days" PANIC trigger. If omitted,
        that trigger is skipped.
    """
    cfg = config or RegimeConfig()

    atr_pct = features.get("atr_pct")
    rv = features.get("realized_vol_5d")
    adx_v = features.get("adx_14")
    slope = features.get("ema_slope_20")

    # Build percentile thresholds from history (expanding window).
    hist = historical_features
    atr_series = hist["atr_pct"].dropna() if "atr_pct" in hist else pd.Series(dtype=float)
    rv_series = hist["realized_vol_5d"].dropna() if "realized_vol_5d" in hist else pd.Series(dtype=float)

    def _q(s: pd.Series, q: float) -> float | None:
        if len(s) < 5:
            return None
        return float(s.quantile(q))

    atr_p25 = _q(atr_series, cfg.mean_rev_atr_percentile_lo)
    atr_p75 = _q(atr_series, cfg.mean_rev_atr_percentile_hi)
    atr_p90 = _q(atr_series, cfg.panic_atr_percentile)
    atr_low = _q(atr_series, cfg.low_vol_atr_percentile)
    rv_p75 = _q(rv_series, cfg.panic_realized_vol_percentile)
    rv_low = _q(rv_series, cfg.low_vol_realized_vol_percentile)

    # ---------- PANIC ----------
    panic = False
    if _is_finite(rv) and rv_p75 is not None and rv > rv_p75:
        panic = True
    if _is_finite(atr_pct) and atr_p90 is not None and atr_pct > atr_p90:
        panic = True
    if recent_returns is not None and len(recent_returns) >= cfg.panic_consecutive_red_count:
        tail = list(recent_returns)[-cfg.panic_consecutive_red_count:]
        if all(r < cfg.panic_consecutive_red_threshold for r in tail):
            panic = True
    if panic:
        return "PANIC"

    # ---------- LOW_VOL ----------
    low_vol = False
    if (_is_finite(atr_pct) and atr_low is not None and atr_pct < atr_low and
            _is_finite(rv) and rv_low is not None and rv < rv_low):
        low_vol = True
    if low_vol:
        return "LOW_VOL"

    # ---------- TREND ----------
    if (_is_finite(adx_v) and adx_v > cfg.trend_adx_min and
            _is_finite(slope) and abs(slope) > cfg.trend_slope_min_abs):
        return "TREND"

    # ---------- MEAN_REVERSION ----------
    if _is_finite(adx_v) and adx_v < cfg.mean_rev_adx_max:
        if (_is_finite(atr_pct) and atr_p25 is not None and atr_p75 is not None and
                atr_p25 <= atr_pct <= atr_p75):
            return "MEAN_REVERSION"

    # Fallback: most common state
    return "MEAN_REVERSION"


# ---------------------------------------------------------------------------
# Timeline builder
# ---------------------------------------------------------------------------

def _aggregate_to_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Resample arbitrary-interval bars to daily OHLC (IST trading day)."""
    if df.empty:
        return df
    d = df.copy().sort_values("ts").reset_index(drop=True)
    if d["ts"].dt.tz is None:
        d["ts"] = d["ts"].dt.tz_localize(IST)
    d["date"] = d["ts"].dt.tz_convert(IST).dt.date
    g = d.groupby("date", as_index=False).agg(
        ts=("ts", "last"),
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    )
    return g.drop(columns=["date"])


def build_regime_timeline(
    symbol: str = "NIFTY",
    interval: str = "day",
    config: RegimeConfig | None = None,
    df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build a per-bar regime timeline with NO look-ahead.

    For `interval="day"` we classify each daily bar. For `interval="minute"`
    we first aggregate to daily, classify, then broadcast each day's regime
    onto every intraday bar of that session (intraday timestamps inherit
    the day's regime by design).

    Parameters
    ----------
    df : pd.DataFrame, optional
        Pre-loaded bar frame (for tests). If omitted, loads from cache.
    """
    from ..data.cache import load_cache

    cfg = config or RegimeConfig()

    if df is None:
        raw = load_cache(symbol, "minute" if interval == "minute" else "day")
    else:
        raw = df.copy()
    if raw.empty:
        return pd.DataFrame(columns=[
            "timestamp", "regime", "atr_pct", "realized_vol_5d",
            "iv_rank", "ema_slope_20", "adx_14",
        ])

    raw = raw.sort_values("ts").reset_index(drop=True)

    # Always classify on daily bars.
    if interval == "minute":
        daily = _aggregate_to_daily(raw)
    else:
        daily = raw.copy()
        if daily["ts"].dt.tz is None:
            daily["ts"] = daily["ts"].dt.tz_localize(IST)

    rows: list[dict] = []
    closes_running: list[float] = []
    feature_history: list[dict] = []

    for i in range(len(daily)):
        bar_slice = daily.iloc[: i + 1]
        feats = _compute_features_from_bars(bar_slice, cfg)
        feature_history.append({
            "atr_pct": feats["atr_pct"],
            "realized_vol_5d": feats["realized_vol_5d"],
        })

        # Build recent daily returns for PANIC trigger
        closes_running.append(float(daily["close"].iloc[i]))
        recent_returns: list[float] = []
        if len(closes_running) >= cfg.panic_consecutive_red_count + 1:
            for k in range(len(closes_running) - cfg.panic_consecutive_red_count,
                           len(closes_running)):
                prev = closes_running[k - 1]
                cur = closes_running[k]
                if prev:
                    recent_returns.append((cur - prev) / prev)

        if i < cfg.warmup_bars:
            regime = "MEAN_REVERSION"
        else:
            hist_df = pd.DataFrame(feature_history[:i + 1])  # expanding window
            regime = classify_regime(feats, hist_df, cfg, recent_returns=recent_returns)

        rows.append({
            "timestamp": daily["ts"].iloc[i],
            "regime": regime,
            **feats,
        })

    daily_out = pd.DataFrame(rows)[[
        "timestamp", "regime", "atr_pct", "realized_vol_5d",
        "iv_rank", "ema_slope_20", "adx_14",
    ]]

    if interval != "minute":
        return daily_out

    # Broadcast daily regime onto intraday bars.
    raw_local = raw.copy()
    if raw_local["ts"].dt.tz is None:
        raw_local["ts"] = raw_local["ts"].dt.tz_localize(IST)
    raw_local["date"] = raw_local["ts"].dt.tz_convert(IST).dt.date
    daily_out_idx = daily_out.copy()
    daily_out_idx["date"] = pd.to_datetime(daily_out_idx["timestamp"]).dt.tz_convert(IST).dt.date
    merged = raw_local.merge(
        daily_out_idx[["date", "regime", "atr_pct", "realized_vol_5d",
                       "iv_rank", "ema_slope_20", "adx_14"]],
        on="date", how="left",
    )
    merged = merged.rename(columns={"ts": "timestamp"})
    return merged[[
        "timestamp", "regime", "atr_pct", "realized_vol_5d",
        "iv_rank", "ema_slope_20", "adx_14",
    ]]


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

def _storage_path(symbol: str, interval: str) -> Path:
    sub = "daily" if interval == "day" else interval
    return STORAGE_DIR / symbol / f"timeline_{sub}.parquet"


def save_timeline(df: pd.DataFrame, symbol: str, interval: str) -> Path:
    path = _storage_path(symbol, interval)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def load_timeline(symbol: str, interval: str) -> pd.DataFrame:
    path = _storage_path(symbol, interval)
    if not path.exists():
        return pd.DataFrame(columns=[
            "timestamp", "regime", "atr_pct", "realized_vol_5d",
            "iv_rank", "ema_slope_20", "adx_14",
        ])
    return pd.read_parquet(path)


# ---------------------------------------------------------------------------
# Convenience lookup
# ---------------------------------------------------------------------------

def regime_for_timestamp(ts: datetime, symbol: str = "NIFTY") -> str:
    """Look up the regime that was in force at `ts`.

    Loads the persisted daily timeline; intraday `ts` values inherit the
    most recent prior daily regime label.
    """
    df = load_timeline(symbol, "day")
    if df.empty:
        df = build_regime_timeline(symbol=symbol, interval="day")
        if df.empty:
            return "MEAN_REVERSION"

    ts_p = pd.Timestamp(ts)
    series_ts = pd.to_datetime(df["timestamp"])
    if series_ts.dt.tz is not None:
        if ts_p.tz is None:
            ts_p = ts_p.tz_localize(IST)
        else:
            ts_p = ts_p.tz_convert(series_ts.dt.tz)
    else:
        if ts_p.tz is not None:
            ts_p = ts_p.tz_convert(IST).tz_localize(None)

    prior = df[series_ts <= ts_p]
    if prior.empty:
        return "MEAN_REVERSION"
    return str(prior.iloc[-1]["regime"])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cmd_build(args: argparse.Namespace) -> None:
    cfg = RegimeConfig()
    df = build_regime_timeline(symbol=args.symbol, interval=args.interval, config=cfg)
    if df.empty:
        print(f"No bars in cache for {args.symbol} {args.interval}")
        return
    path = save_timeline(df, args.symbol, args.interval)
    print(f"Built timeline: {len(df)} rows -> {path}")
    print(f"Range: {df['timestamp'].min()} -> {df['timestamp'].max()}")
    _print_distribution(df)


def _cmd_show(args: argparse.Namespace) -> None:
    df = load_timeline(args.symbol, "day")
    if df.empty:
        print(f"No saved timeline for {args.symbol}; run build first.")
        return
    target = pd.Timestamp(args.date)
    series_ts = pd.to_datetime(df["timestamp"])
    if series_ts.dt.tz is not None and target.tz is None:
        target = target.tz_localize(IST)
    sub = df[series_ts.dt.tz_convert(IST).dt.date == target.date()] if series_ts.dt.tz is not None \
        else df[series_ts.dt.date == target.date()]
    if sub.empty:
        # fall back to most recent prior
        prior = df[series_ts <= target]
        if prior.empty:
            print(f"No regime label on or before {args.date}")
            return
        row = prior.iloc[-1]
        print(f"No bar on {args.date}; nearest prior:")
    else:
        row = sub.iloc[0]
    print(f"  timestamp    : {row['timestamp']}")
    print(f"  regime       : {row['regime']}")
    print(f"  atr_pct      : {row['atr_pct']:.4f}")
    print(f"  realized_vol : {row['realized_vol_5d']:.4f}")
    print(f"  ema_slope_20 : {row['ema_slope_20']:.4f}")
    print(f"  adx_14       : {row['adx_14']:.2f}")
    print(f"  iv_rank      : {row['iv_rank']:.2f}")


def _cmd_stats(args: argparse.Namespace) -> None:
    df = load_timeline(args.symbol, "day")
    if df.empty:
        print(f"No saved timeline for {args.symbol}; run build first.")
        return
    _print_distribution(df)


def _print_distribution(df: pd.DataFrame) -> None:
    total = len(df)
    print(f"\nRegime distribution over {total} bars:")
    counts = df["regime"].value_counts()
    for name in REGIMES:
        c = int(counts.get(name, 0))
        pct = 100.0 * c / total if total else 0.0
        print(f"  {name:<15} {c:>6}  ({pct:5.1f}%)")


def main() -> None:
    p = argparse.ArgumentParser(prog="regimes")
    sub = p.add_subparsers(dest="cmd", required=True)

    pb = sub.add_parser("build", help="Build & persist regime timeline")
    pb.add_argument("--symbol", default="NIFTY")
    pb.add_argument("--interval", default="day", choices=["day", "minute"])
    pb.set_defaults(func=_cmd_build)

    ps = sub.add_parser("show", help="Show regime on a given date")
    ps.add_argument("--symbol", default="NIFTY")
    ps.add_argument("--date", required=True, help="YYYY-MM-DD")
    ps.set_defaults(func=_cmd_show)

    pt = sub.add_parser("stats", help="Regime distribution")
    pt.add_argument("--symbol", default="NIFTY")
    pt.set_defaults(func=_cmd_stats)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
