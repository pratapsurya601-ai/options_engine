"""Light feature computation for Streamlit pages.

Wraps `engine.research.flow_features` primitives. We re-export thin helpers so
Streamlit pages don't need to know about the full long-format pipeline.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

try:
    from engine.research.flow_features import (
        atm_iv,
        call_wall,
        iv_skew_25d,
        iv_smile_curvature,
        max_pain,
        oi_concentration,
        pcr_oi,
        pcr_oi_atm,
        pcr_volume,
        put_wall,
        _atm_strike,
    )
    _HAVE_ENGINE = True
except Exception:  # pragma: no cover - allows running streamlit without engine deps
    _HAVE_ENGINE = False


def _pick_expiry(df: pd.DataFrame) -> Optional[str]:
    """Pick nearest expiry available in the frame."""
    if df is None or df.empty or "expiry" not in df.columns:
        return None
    exps = sorted([str(e)[:10] for e in df["expiry"].dropna().unique()])
    return exps[0] if exps else None


def compute_snapshot_summary(chain: pd.DataFrame) -> dict:
    """Compute point-in-time features for a single snapshot's chain.

    Returns a flat dict ready for display. NaN-safe.
    """
    out = {
        "spot": None,
        "expiry": None,
        "atm_strike": None,
        "max_pain": None,
        "call_wall": None,
        "put_wall": None,
        "atm_iv": None,
        "iv_skew_25d": None,
        "iv_smile_curvature": None,
        "pcr_oi": None,
        "pcr_volume": None,
        "pcr_oi_atm": None,
        "oi_concentration_call": None,
        "oi_concentration_put": None,
        "total_call_oi": None,
        "total_put_oi": None,
    }
    if chain is None or chain.empty:
        return out

    expiry = _pick_expiry(chain)
    if expiry is None:
        return out
    df = chain[chain["expiry"].astype(str).str[:10] == expiry].copy()
    if df.empty:
        return out

    # Defensive: Postgres NUMERIC -> Decimal; psycopg can also return None
    # or pd.NA via empty rows. Coerce, drop bad, fall back to NaN-safe path.
    raw_spot = df["spot"].iloc[0]
    try:
        if raw_spot is None or pd.isna(raw_spot):
            return out
        spot = float(raw_spot)
    except (TypeError, ValueError):
        return out
    out["spot"] = spot
    out["expiry"] = expiry

    if not _HAVE_ENGINE:
        # Minimal inline fallback
        ce = df[df["option_type"] == "CE"]
        pe = df[df["option_type"] == "PE"]
        out["total_call_oi"] = float(ce["oi"].sum() or 0)
        out["total_put_oi"] = float(pe["oi"].sum() or 0)
        if out["total_call_oi"]:
            out["pcr_oi"] = out["total_put_oi"] / out["total_call_oi"]
        if not ce.empty:
            out["call_wall"] = float(ce.groupby("strike")["oi"].sum().idxmax())
        if not pe.empty:
            out["put_wall"] = float(pe.groupby("strike")["oi"].sum().idxmax())
        return out

    atm = _atm_strike(spot, df["strike"].tolist(), strike_step=50)
    out["atm_strike"] = float(atm)
    out["max_pain"] = float(max_pain(df))
    out["call_wall"] = float(call_wall(df))
    out["put_wall"] = float(put_wall(df))
    out["atm_iv"] = float(atm_iv(df, atm))
    out["pcr_oi"] = float(pcr_oi(df))
    out["pcr_volume"] = float(pcr_volume(df))
    out["pcr_oi_atm"] = float(pcr_oi_atm(df, atm, n=3, strike_step=50))
    out["oi_concentration_call"] = float(oi_concentration(df, "CE", top=3))
    out["oi_concentration_put"] = float(oi_concentration(df, "PE", top=3))
    out["total_call_oi"] = float(df.loc[df["option_type"] == "CE", "oi"].sum())
    out["total_put_oi"] = float(df.loc[df["option_type"] == "PE", "oi"].sum())

    # iv_skew / smile use t_years; rough estimate from expiry vs today
    try:
        from datetime import date
        exp_d = pd.to_datetime(expiry).date()
        dte = max((exp_d - date.today()).days, 0)
        t_years = max(dte / 365.0, 0.5 / 365.0)
        out["iv_skew_25d"] = float(iv_skew_25d(df, spot, t_years))
        out["iv_smile_curvature"] = float(iv_smile_curvature(df, atm))
    except Exception:
        pass

    return out


def oi_writing_velocities(snapshots: list[pd.DataFrame]) -> pd.DataFrame:
    """Compute per-snapshot writing/unwinding velocity for CE and PE.

    Input: ordered list of chain DataFrames (oldest first), each filtered to a
    single expiry. Output: DataFrame indexed by snapshot_ts with columns
    call_writing_velocity, put_writing_velocity, call_unwinding_velocity,
    put_unwinding_velocity.
    """
    rows = []
    prev: Optional[pd.DataFrame] = None
    prev_ts = None
    for snap in snapshots:
        if snap is None or snap.empty:
            continue
        ts = snap["snapshot_ts"].iloc[0]
        if prev is None:
            prev = snap
            prev_ts = ts
            continue
        dt_min = max((pd.Timestamp(ts) - pd.Timestamp(prev_ts)).total_seconds() / 60.0, 1e-9)
        row = {"snapshot_ts": ts}
        for otype, label in (("CE", "call"), ("PE", "put")):
            cur = snap[snap["option_type"] == otype].groupby("strike")["oi"].sum()
            prv = prev[prev["option_type"] == otype].groupby("strike")["oi"].sum()
            idx = cur.index.union(prv.index)
            diff = cur.reindex(idx, fill_value=0.0) - prv.reindex(idx, fill_value=0.0)
            pos = float(diff[diff > 0].sum())
            neg = float(-diff[diff < 0].sum())
            row[f"{label}_writing_velocity"] = pos / dt_min
            row[f"{label}_unwinding_velocity"] = neg / dt_min
        rows.append(row)
        prev = snap
        prev_ts = ts
    return pd.DataFrame(rows)
