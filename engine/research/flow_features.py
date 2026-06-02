"""
Option-flow feature engineering.

Computes structural option-positioning features from chain snapshots written
by `engine.data.chain_snapshot`. Output is **long-format**:
    columns = [timestamp, symbol, expiry, feature_name, feature_value]
which downstream predictive_power / regression code can pivot trivially.

Storage:
    data/research/flow_features/{symbol}/{YYYY-MM-DD}.parquet  -- per day
    data/research/flow_features/{symbol}/all.parquet           -- combined

Feature groups
--------------
Per-snapshot (point-in-time):
    pcr_oi, pcr_volume, pcr_oi_atm,
    atm_iv, iv_skew_25d, iv_smile_curvature,
    max_pain, call_wall, put_wall,
    dist_spot_to_max_pain, dist_spot_to_call_wall, dist_spot_to_put_wall,
    oi_concentration_call, oi_concentration_put,
    gamma_exposure_proxy,
    total_call_oi, total_put_oi, atm_strike

Inter-snapshot (require previous, some require N-2):
    oi_delta_total_call, oi_delta_total_put,
    oi_acceleration_call, oi_acceleration_put,
    put_writing_velocity, call_writing_velocity,
    put_unwinding_velocity, call_unwinding_velocity,
    atm_iv_change, skew_change,
    max_pain_migration, call_wall_migration, put_wall_migration

CLI
---
    python -m engine.research.flow_features build --symbol NIFTY --date 2026-06-02
    python -m engine.research.flow_features build-all --symbol NIFTY
    python -m engine.research.flow_features list --symbol NIFTY

Notes on signal honesty
-----------------------
- `gamma_exposure_proxy` is a *proxy*: we don't know dealer net positioning,
  only chain-wide CE-PE OI weighted by BS gamma. Theoretical, not measured.
- `iv_skew_25d` uses the strike whose |BS-delta| is closest to 0.25; if BS
  cannot be evaluated (no IV / DTE<=0), falls back to ~200pt OTM strikes.
- `max_pain` is the classic option-writer payoff minimum, NOT a forecast.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from engine.data.chain_snapshot import (
    SNAPSHOT_ROOT,
    list_snapshots,
    read_snapshot,
)
from engine.pricing import black_scholes


IST = timezone(timedelta(hours=5, minutes=30))
FEATURES_ROOT = Path("data/research/flow_features")

# Columns of the long-format output frame
OUT_COLS = ["timestamp", "symbol", "expiry", "feature_name", "feature_value"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _coerce_expiry(x) -> str:
    """Render expiry as ISO date string YYYY-MM-DD."""
    if x is None:
        return ""
    if isinstance(x, str):
        return x[:10]
    if isinstance(x, (datetime, pd.Timestamp)):
        return x.date().isoformat()
    if isinstance(x, date):
        return x.isoformat()
    return str(x)[:10]


def _pick_nearest_expiry(df: pd.DataFrame, asof_ts: datetime | None = None) -> str:
    """Return the expiry value (as string) closest to (and >=) asof_ts."""
    exps = df["expiry"].dropna().unique().tolist()
    if not exps:
        raise ValueError("snapshot has no expiry column populated")

    def _to_date(e) -> date | None:
        try:
            if isinstance(e, str):
                return datetime.fromisoformat(e[:10]).date()
            if isinstance(e, (datetime, pd.Timestamp)):
                return e.date() if hasattr(e, "date") else None
            if isinstance(e, date):
                return e
        except Exception:
            return None
        return None

    parsed = [(_to_date(e), e) for e in exps]
    parsed = [p for p in parsed if p[0] is not None]
    if not parsed:
        return _coerce_expiry(exps[0])

    if asof_ts is None:
        asof = date.today()
    else:
        asof = asof_ts.date() if hasattr(asof_ts, "date") else asof_ts

    future = [p for p in parsed if p[0] >= asof]
    chosen = min(future, key=lambda p: p[0]) if future else min(parsed, key=lambda p: p[0])
    return _coerce_expiry(chosen[1])


def _expiry_to_date(e) -> date | None:
    try:
        if isinstance(e, str):
            return datetime.fromisoformat(e[:10]).date()
        if isinstance(e, (datetime, pd.Timestamp)):
            return e.date()
        if isinstance(e, date):
            return e
    except Exception:
        return None
    return None


def _atm_strike(spot: float, strikes: Iterable[float], strike_step: int = 50) -> float:
    """Round spot to nearest available strike."""
    strikes = sorted(set(float(s) for s in strikes))
    if not strikes:
        return float(round(spot / strike_step) * strike_step)
    return min(strikes, key=lambda s: abs(s - spot))


def _safe_div(num: float, den: float) -> float:
    if den is None or den == 0 or pd.isna(den):
        return float("nan")
    return float(num) / float(den)


def _filter_expiry(df: pd.DataFrame, expiry: str) -> pd.DataFrame:
    """Filter for rows matching the given expiry (string match on YYYY-MM-DD)."""
    return df[df["expiry"].apply(_coerce_expiry) == expiry].copy()


# ---------------------------------------------------------------------------
# Per-snapshot feature primitives (pure functions, easy to test)
# ---------------------------------------------------------------------------


def pcr_oi(df: pd.DataFrame) -> float:
    ce = df.loc[df["option_type"] == "CE", "oi"].sum()
    pe = df.loc[df["option_type"] == "PE", "oi"].sum()
    return _safe_div(pe, ce)


def pcr_volume(df: pd.DataFrame) -> float:
    ce = df.loc[df["option_type"] == "CE", "volume"].sum()
    pe = df.loc[df["option_type"] == "PE", "volume"].sum()
    return _safe_div(pe, ce)


def pcr_oi_atm(df: pd.DataFrame, atm: float, n: int = 3, strike_step: int = 50) -> float:
    """PCR restricted to ATM +/- n*strike_step."""
    lo, hi = atm - n * strike_step, atm + n * strike_step
    band = df[(df["strike"] >= lo) & (df["strike"] <= hi)]
    return pcr_oi(band)


def atm_iv(df: pd.DataFrame, atm: float) -> float:
    """Average IV of ATM CE + ATM PE. NaN-safe."""
    sub = df[df["strike"] == atm]
    ivs = sub["iv"].dropna()
    if ivs.empty:
        return float("nan")
    return float(ivs.mean())


def max_pain(df: pd.DataFrame) -> float:
    """
    Strike that minimizes total option writer payoff (== buyer payoff at expiry).
    payoff(K*) = sum_K [ max(K* - K, 0)*CE_OI(K) + max(K - K*, 0)*PE_OI(K) ]
    """
    strikes = sorted(df["strike"].unique())
    if not strikes:
        return float("nan")
    ce = df[df["option_type"] == "CE"].groupby("strike")["oi"].sum()
    pe = df[df["option_type"] == "PE"].groupby("strike")["oi"].sum()

    best_k = strikes[0]
    best_pay = float("inf")
    for k_star in strikes:
        pay = 0.0
        for k, oi in ce.items():
            if k_star > k:
                pay += (k_star - k) * float(oi)
        for k, oi in pe.items():
            if k > k_star:
                pay += (k - k_star) * float(oi)
        if pay < best_pay:
            best_pay = pay
            best_k = k_star
    return float(best_k)


def call_wall(df: pd.DataFrame) -> float:
    ce = df[df["option_type"] == "CE"].groupby("strike")["oi"].sum()
    if ce.empty:
        return float("nan")
    return float(ce.idxmax())


def put_wall(df: pd.DataFrame) -> float:
    pe = df[df["option_type"] == "PE"].groupby("strike")["oi"].sum()
    if pe.empty:
        return float("nan")
    return float(pe.idxmax())


def oi_concentration(df: pd.DataFrame, option_type: str, top: int = 3) -> float:
    """top-N OI / total OI for the given option type."""
    sub = df[df["option_type"] == option_type].groupby("strike")["oi"].sum()
    if sub.empty:
        return float("nan")
    total = float(sub.sum())
    if total <= 0:
        return float("nan")
    top_sum = float(sub.nlargest(top).sum())
    return top_sum / total


def iv_skew_25d(df: pd.DataFrame, spot: float, t_years: float, r: float = 0.065) -> float:
    """IV(25-delta put) - IV(25-delta call). Falls back to ~200pt OTM if BS fails."""
    ce = df[(df["option_type"] == "CE") & df["iv"].notna()]
    pe = df[(df["option_type"] == "PE") & df["iv"].notna()]
    if ce.empty or pe.empty or t_years <= 0:
        return _iv_skew_fallback(df, spot)

    def best_for(side_df: pd.DataFrame, want_otype: str) -> float | None:
        best = None
        best_err = float("inf")
        for _, row in side_df.iterrows():
            iv = float(row["iv"])
            if iv <= 0:
                continue
            try:
                g = black_scholes(spot, float(row["strike"]), t_years,
                                  iv, r=r, option_type=want_otype)
            except Exception:
                continue
            err = abs(abs(g.delta) - 0.25)
            if err < best_err:
                best_err = err
                best = iv
        return best

    iv_call_25 = best_for(ce, "CE")
    iv_put_25 = best_for(pe, "PE")
    if iv_call_25 is None or iv_put_25 is None:
        return _iv_skew_fallback(df, spot)
    return iv_put_25 - iv_call_25


def _iv_skew_fallback(df: pd.DataFrame, spot: float, offset: float = 200.0) -> float:
    """Fallback skew when BS-delta path isn't available."""
    put_target = spot - offset
    call_target = spot + offset
    pe = df[(df["option_type"] == "PE") & df["iv"].notna()]
    ce = df[(df["option_type"] == "CE") & df["iv"].notna()]
    if pe.empty or ce.empty:
        return float("nan")
    p_row = pe.iloc[(pe["strike"] - put_target).abs().argsort()].iloc[0]
    c_row = ce.iloc[(ce["strike"] - call_target).abs().argsort()].iloc[0]
    return float(p_row["iv"]) - float(c_row["iv"])


def iv_smile_curvature(df: pd.DataFrame, atm: float, strike_step: int = 50,
                       wing: int = 4) -> float:
    """(avg IV of strikes at ATM +/- 3..wing steps) - ATM IV. Positive => smile."""
    atm_iv_val = atm_iv(df, atm)
    if pd.isna(atm_iv_val):
        return float("nan")
    lo_strike = atm - wing * strike_step
    hi_strike = atm + wing * strike_step
    band = df[
        ((df["strike"] >= lo_strike) & (df["strike"] <= atm - 2 * strike_step))
        | ((df["strike"] >= atm + 2 * strike_step) & (df["strike"] <= hi_strike))
    ]
    ivs = band["iv"].dropna()
    if ivs.empty:
        return float("nan")
    return float(ivs.mean()) - atm_iv_val


def gamma_exposure_proxy(df: pd.DataFrame, spot: float, t_years: float,
                         r: float = 0.065) -> float:
    """
    Sum over strikes K of (CE_OI(K) - PE_OI(K)) * gamma_K.
    Theoretical, NOT a measure of true dealer positioning.
    """
    if t_years <= 0:
        return float("nan")
    ce = df[df["option_type"] == "CE"].set_index("strike")
    pe = df[df["option_type"] == "PE"].set_index("strike")
    strikes = set(ce.index) | set(pe.index)
    total = 0.0
    any_ok = False
    for k in strikes:
        oi_c = float(ce.loc[k, "oi"]) if k in ce.index else 0.0
        oi_p = float(pe.loc[k, "oi"]) if k in pe.index else 0.0
        iv = float("nan")
        if k in ce.index and pd.notna(ce.loc[k, "iv"]):
            iv = float(ce.loc[k, "iv"])
        elif k in pe.index and pd.notna(pe.loc[k, "iv"]):
            iv = float(pe.loc[k, "iv"])
        if not (iv > 0) or pd.isna(iv):
            continue
        try:
            g = black_scholes(spot, float(k), t_years, iv, r=r, option_type="CE")
        except Exception:
            continue
        total += (oi_c - oi_p) * g.gamma
        any_ok = True
    return total if any_ok else float("nan")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass
class _Ctx:
    """Internal: precomputed scalars reused across feature emit calls."""
    df: pd.DataFrame
    timestamp: datetime
    symbol: str
    expiry: str
    spot: float
    atm: float
    t_years: float
    r: float
    strike_step: int


def _build_ctx(snapshot_df: pd.DataFrame, expiry: str | None,
               r: float, strike_step: int) -> _Ctx | None:
    """Validate + slice + return a working context, or None if malformed."""
    required = {"snapshot_ts", "symbol", "spot", "expiry",
                "strike", "option_type", "oi"}
    if snapshot_df is None or snapshot_df.empty:
        return None
    missing = required - set(snapshot_df.columns)
    if missing:
        warnings.warn(f"snapshot missing columns {missing}; skipping")
        return None

    ts_raw = snapshot_df["snapshot_ts"].iloc[0]
    if isinstance(ts_raw, str):
        timestamp = pd.Timestamp(ts_raw).to_pydatetime()
    elif isinstance(ts_raw, pd.Timestamp):
        timestamp = ts_raw.to_pydatetime()
    else:
        timestamp = ts_raw

    symbol = str(snapshot_df["symbol"].iloc[0])
    if expiry is None:
        expiry = _pick_nearest_expiry(snapshot_df, timestamp)
    df = _filter_expiry(snapshot_df, expiry)
    if df.empty:
        warnings.warn(f"no rows for expiry={expiry}; skipping")
        return None

    spot = float(df["spot"].iloc[0])
    atm = _atm_strike(spot, df["strike"].tolist(), strike_step=strike_step)

    exp_d = _expiry_to_date(expiry)
    if exp_d is None:
        t_years = 0.0
    else:
        ts_date = timestamp.date() if hasattr(timestamp, "date") else date.today()
        dte = (exp_d - ts_date).days
        # If snapshot is intraday on the same day as expiry, treat as ~0.5d
        if dte <= 0:
            t_years = max(0.5 / 365.0, 0.0)
        else:
            t_years = dte / 365.0

    return _Ctx(
        df=df, timestamp=timestamp, symbol=symbol, expiry=expiry,
        spot=spot, atm=atm, t_years=t_years, r=r, strike_step=strike_step,
    )


def _rows_pointintime(ctx: _Ctx) -> list[dict]:
    """Compute and emit point-in-time features for one snapshot."""
    df = ctx.df
    feats: dict[str, float] = {}

    total_call_oi = float(df.loc[df["option_type"] == "CE", "oi"].sum())
    total_put_oi = float(df.loc[df["option_type"] == "PE", "oi"].sum())

    feats["pcr_oi"] = pcr_oi(df)
    feats["pcr_volume"] = pcr_volume(df) if "volume" in df.columns else float("nan")
    feats["pcr_oi_atm"] = pcr_oi_atm(df, ctx.atm, n=3, strike_step=ctx.strike_step)
    feats["atm_iv"] = atm_iv(df, ctx.atm)
    feats["iv_skew_25d"] = iv_skew_25d(df, ctx.spot, ctx.t_years, r=ctx.r)
    feats["iv_smile_curvature"] = iv_smile_curvature(df, ctx.atm, ctx.strike_step)
    mp = max_pain(df)
    cw = call_wall(df)
    pw = put_wall(df)
    feats["max_pain"] = mp
    feats["call_wall"] = cw
    feats["put_wall"] = pw
    feats["dist_spot_to_max_pain"] = _safe_div(ctx.spot - mp, ctx.spot) if pd.notna(mp) else float("nan")
    feats["dist_spot_to_call_wall"] = _safe_div(ctx.spot - cw, ctx.spot) if pd.notna(cw) else float("nan")
    feats["dist_spot_to_put_wall"] = _safe_div(ctx.spot - pw, ctx.spot) if pd.notna(pw) else float("nan")
    feats["oi_concentration_call"] = oi_concentration(df, "CE", top=3)
    feats["oi_concentration_put"] = oi_concentration(df, "PE", top=3)
    feats["gamma_exposure_proxy"] = gamma_exposure_proxy(df, ctx.spot, ctx.t_years, r=ctx.r)
    feats["total_call_oi"] = total_call_oi
    feats["total_put_oi"] = total_put_oi
    feats["atm_strike"] = float(ctx.atm)

    return [
        {
            "timestamp": ctx.timestamp,
            "symbol": ctx.symbol,
            "expiry": ctx.expiry,
            "feature_name": name,
            "feature_value": float(val) if val is not None else float("nan"),
        }
        for name, val in feats.items()
    ]


def _oi_by_strike(df: pd.DataFrame, otype: str) -> pd.Series:
    return df[df["option_type"] == otype].groupby("strike")["oi"].sum()


def _rows_intersnapshot(ctx: _Ctx, prev_ctx: _Ctx | None,
                        prev_prev_ctx: _Ctx | None) -> list[dict]:
    """Compute and emit inter-snapshot features. Requires same expiry."""
    if prev_ctx is None:
        return []
    if prev_ctx.expiry != ctx.expiry:
        return []

    dt_minutes = max((ctx.timestamp - prev_ctx.timestamp).total_seconds() / 60.0, 1e-9)

    feats: dict[str, float] = {}

    # Totals
    cur_ce_total = float(ctx.df.loc[ctx.df["option_type"] == "CE", "oi"].sum())
    cur_pe_total = float(ctx.df.loc[ctx.df["option_type"] == "PE", "oi"].sum())
    prv_ce_total = float(prev_ctx.df.loc[prev_ctx.df["option_type"] == "CE", "oi"].sum())
    prv_pe_total = float(prev_ctx.df.loc[prev_ctx.df["option_type"] == "PE", "oi"].sum())

    feats["oi_delta_total_call"] = cur_ce_total - prv_ce_total
    feats["oi_delta_total_put"] = cur_pe_total - prv_pe_total

    # Accelerations
    if prev_prev_ctx is not None and prev_prev_ctx.expiry == ctx.expiry:
        pp_ce_total = float(prev_prev_ctx.df.loc[
            prev_prev_ctx.df["option_type"] == "CE", "oi"].sum())
        pp_pe_total = float(prev_prev_ctx.df.loc[
            prev_prev_ctx.df["option_type"] == "PE", "oi"].sum())
        d1_ce = cur_ce_total - prv_ce_total
        d0_ce = prv_ce_total - pp_ce_total
        d1_pe = cur_pe_total - prv_pe_total
        d0_pe = prv_pe_total - pp_pe_total
        feats["oi_acceleration_call"] = d1_ce - d0_ce
        feats["oi_acceleration_put"] = d1_pe - d0_pe
    else:
        feats["oi_acceleration_call"] = float("nan")
        feats["oi_acceleration_put"] = float("nan")

    # Per-strike OI changes -> writing vs unwinding velocities
    for otype, label in (("CE", "call"), ("PE", "put")):
        cur = _oi_by_strike(ctx.df, otype)
        prv = _oi_by_strike(prev_ctx.df, otype)
        joined = cur.reindex(cur.index.union(prv.index), fill_value=0.0) - \
            prv.reindex(cur.index.union(prv.index), fill_value=0.0)
        positive = joined[joined > 0].sum()
        negative = joined[joined < 0].sum()  # negative number
        feats[f"{label}_writing_velocity"] = float(positive) / dt_minutes
        feats[f"{label}_unwinding_velocity"] = float(-negative) / dt_minutes

    # IV / skew / wall migrations -- recompute on prev for honesty
    cur_atm_iv = atm_iv(ctx.df, ctx.atm)
    prv_atm_iv = atm_iv(prev_ctx.df, prev_ctx.atm)
    feats["atm_iv_change"] = (cur_atm_iv - prv_atm_iv) if (pd.notna(cur_atm_iv) and pd.notna(prv_atm_iv)) else float("nan")

    cur_skew = iv_skew_25d(ctx.df, ctx.spot, ctx.t_years, r=ctx.r)
    prv_skew = iv_skew_25d(prev_ctx.df, prev_ctx.spot, prev_ctx.t_years, r=prev_ctx.r)
    feats["skew_change"] = (cur_skew - prv_skew) if (pd.notna(cur_skew) and pd.notna(prv_skew)) else float("nan")

    cur_mp = max_pain(ctx.df); prv_mp = max_pain(prev_ctx.df)
    cur_cw = call_wall(ctx.df); prv_cw = call_wall(prev_ctx.df)
    cur_pw = put_wall(ctx.df); prv_pw = put_wall(prev_ctx.df)
    feats["max_pain_migration"] = (cur_mp - prv_mp) if (pd.notna(cur_mp) and pd.notna(prv_mp)) else float("nan")
    feats["call_wall_migration"] = (cur_cw - prv_cw) if (pd.notna(cur_cw) and pd.notna(prv_cw)) else float("nan")
    feats["put_wall_migration"] = (cur_pw - prv_pw) if (pd.notna(cur_pw) and pd.notna(prv_pw)) else float("nan")

    return [
        {
            "timestamp": ctx.timestamp,
            "symbol": ctx.symbol,
            "expiry": ctx.expiry,
            "feature_name": name,
            "feature_value": float(val) if val is not None else float("nan"),
        }
        for name, val in feats.items()
    ]


def compute_features_for_snapshot(
    snapshot_df: pd.DataFrame,
    prev_snapshot_df: pd.DataFrame | None = None,
    prev_prev_snapshot_df: pd.DataFrame | None = None,
    r: float = 0.065,
    expiry: str | None = None,
    strike_step: int = 50,
) -> pd.DataFrame:
    """
    Compute the full long-format feature frame for one snapshot.

    Parameters
    ----------
    snapshot_df : raw chain rows for snapshot t.
    prev_snapshot_df : optional raw chain at t-1 (for delta features).
    prev_prev_snapshot_df : optional raw chain at t-2 (for accelerations).
    r : risk-free rate (annual decimal, default 6.5%).
    expiry : YYYY-MM-DD; if None, picks nearest >= snapshot date.
    strike_step : NIFTY step (default 50).

    Returns
    -------
    DataFrame[timestamp, symbol, expiry, feature_name, feature_value]
    (empty if snapshot is malformed.)

    Example
    -------
    >>> df = read_snapshot(some_path)
    >>> feats = compute_features_for_snapshot(df)
    >>> feats.pivot_table(index='timestamp', columns='feature_name',
    ...                   values='feature_value').head()
    """
    ctx = _build_ctx(snapshot_df, expiry, r, strike_step)
    if ctx is None:
        return pd.DataFrame(columns=OUT_COLS)

    prev_ctx = (
        _build_ctx(prev_snapshot_df, ctx.expiry, r, strike_step)
        if prev_snapshot_df is not None else None
    )
    prev_prev_ctx = (
        _build_ctx(prev_prev_snapshot_df, ctx.expiry, r, strike_step)
        if prev_prev_snapshot_df is not None else None
    )

    rows = _rows_pointintime(ctx)
    rows += _rows_intersnapshot(ctx, prev_ctx, prev_prev_ctx)
    out = pd.DataFrame(rows, columns=OUT_COLS)
    return out


# ---------------------------------------------------------------------------
# Day / range walkers
# ---------------------------------------------------------------------------


def _snapshots_for_day(symbol: str, day: date) -> list[Path]:
    sym_day = SNAPSHOT_ROOT / symbol / day.isoformat()
    if not sym_day.exists():
        return []
    return sorted(sym_day.glob("*.parquet"))


def _available_days(symbol: str) -> list[date]:
    root = SNAPSHOT_ROOT / symbol
    if not root.exists():
        return []
    out: list[date] = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        try:
            out.append(datetime.fromisoformat(d.name).date())
        except ValueError:
            continue
    return out


def compute_features_for_date(
    symbol: str = "NIFTY",
    day: date | None = None,
    out_path: Path | None = None,
    r: float = 0.065,
    strike_step: int = 50,
) -> Path:
    """Walk all snapshots for the day in time order, persist parquet.

    Inter-snapshot features use the previous snapshot of the SAME chosen
    expiry. No lookahead: at snapshot t we only consult t-1 and t-2.
    """
    if day is None:
        day = datetime.now(tz=IST).date()
    files = _snapshots_for_day(symbol, day)
    out_path = out_path or (FEATURES_ROOT / symbol / f"{day.isoformat()}.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not files:
        warnings.warn(f"no snapshots for {symbol} {day.isoformat()}")
        empty = pd.DataFrame(columns=OUT_COLS)
        empty.to_parquet(out_path, index=False)
        return out_path

    prev_df: pd.DataFrame | None = None
    prev_prev_df: pd.DataFrame | None = None
    all_rows: list[pd.DataFrame] = []

    for f in files:
        try:
            cur_df = read_snapshot(f)
        except Exception as e:
            warnings.warn(f"failed to read {f}: {e}")
            continue
        try:
            feats = compute_features_for_snapshot(
                cur_df,
                prev_snapshot_df=prev_df,
                prev_prev_snapshot_df=prev_prev_df,
                r=r,
                strike_step=strike_step,
            )
            if not feats.empty:
                all_rows.append(feats)
        except Exception as e:
            warnings.warn(f"feature compute failed for {f}: {e}")
            continue
        prev_prev_df = prev_df
        prev_df = cur_df

    if all_rows:
        full = pd.concat(all_rows, ignore_index=True)
    else:
        full = pd.DataFrame(columns=OUT_COLS)
    full.to_parquet(out_path, index=False)
    return out_path


def compute_features_range(
    symbol: str = "NIFTY",
    start_date: date | None = None,
    end_date: date | None = None,
    r: float = 0.065,
    strike_step: int = 50,
) -> Path:
    """Walk all snapshot days in [start_date, end_date], build per-day files,
    then rewrite the combined `all.parquet` for the symbol.
    """
    days = _available_days(symbol)
    if start_date is not None:
        days = [d for d in days if d >= start_date]
    if end_date is not None:
        days = [d for d in days if d <= end_date]

    sym_out = FEATURES_ROOT / symbol
    sym_out.mkdir(parents=True, exist_ok=True)

    for d in days:
        compute_features_for_date(symbol=symbol, day=d, r=r, strike_step=strike_step)

    combined_parts: list[pd.DataFrame] = []
    for f in sorted(sym_out.glob("*.parquet")):
        if f.name == "all.parquet":
            continue
        try:
            combined_parts.append(pd.read_parquet(f))
        except Exception as e:
            warnings.warn(f"skip {f}: {e}")
    if combined_parts:
        combined = pd.concat(combined_parts, ignore_index=True)
    else:
        combined = pd.DataFrame(columns=OUT_COLS)
    out = sym_out / "all.parquet"
    combined.to_parquet(out, index=False)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli_build(args):
    if args.date:
        day = datetime.fromisoformat(args.date).date()
    else:
        day = datetime.now(tz=IST).date()
    out = compute_features_for_date(symbol=args.symbol, day=day)
    df = pd.read_parquet(out)
    print(f"Wrote {out} ({len(df)} rows)")


def _cli_build_all(args):
    out = compute_features_range(symbol=args.symbol)
    df = pd.read_parquet(out)
    n_ts = df["timestamp"].nunique() if not df.empty else 0
    print(f"Wrote {out} ({len(df)} rows, {n_ts} unique timestamps)")


def _cli_list(args):
    days = _available_days(args.symbol)
    if not days:
        print(f"No snapshots for {args.symbol} under {SNAPSHOT_ROOT}")
        return
    print(f"Snapshot days for {args.symbol}:")
    for d in days:
        n = len(_snapshots_for_day(args.symbol, d))
        print(f"  {d.isoformat()}  ({n} snapshots)")
    feats_root = FEATURES_ROOT / args.symbol
    if feats_root.exists():
        print(f"\nFeature parquet files under {feats_root}:")
        for f in sorted(feats_root.glob("*.parquet")):
            size_kb = f.stat().st_size / 1024
            print(f"  {f.name}  ({size_kb:.0f} KB)")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="flow_features")
    sub = p.add_subparsers(dest="cmd", required=True)

    pb = sub.add_parser("build", help="Build features for one day")
    pb.add_argument("--symbol", default="NIFTY")
    pb.add_argument("--date", default=None, help="YYYY-MM-DD (default: today IST)")
    pb.set_defaults(func=_cli_build)

    pba = sub.add_parser("build-all", help="Build features for all available days")
    pba.add_argument("--symbol", default="NIFTY")
    pba.set_defaults(func=_cli_build_all)

    pl = sub.add_parser("list", help="List available snapshot days and feature files")
    pl.add_argument("--symbol", default="NIFTY")
    pl.set_defaults(func=_cli_list)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
