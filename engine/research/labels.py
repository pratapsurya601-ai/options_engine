"""
Forward-return label generator for NIFTY/BANKNIFTY 1-minute cache.

For each input timestamp t, computes forward returns over a set of horizons
(15m / 30m / 60m / EOD by default) using cached 1-minute bars. Produces both
regression targets (ret_*) and 3-class classification targets (cls_*).

Design contract
---------------
- Pure function: same (cache, inputs) -> same outputs. No clocks, no randomness.
- NEVER fabricate a label. If the bar at t or t+horizon is unavailable, OR if
  the horizon crosses a trading-session boundary (and allow_overnight=False),
  the cell is NaN.
- Session boundary: NIFTY cash session = 09:15..15:30 IST. EOD is the close of
  the LAST bar of t's trading day (>= 15:25 ideally; whatever the cache has).
- Tolerance: caller timestamps may be 5-min snapshots, etc. We look up the
  nearest bar within `tolerance_minutes` (default 2). If the nearest bar is
  outside tolerance, NaN.
- Overnight: if t + horizon falls past the session end and allow_overnight is
  False (default), ret_<that horizon> is NaN; ret_eod is still produced
  whenever t and the session-end bar exist on the same trading day.
- If t IS the session-end bar, ret_eod = NaN (zero-horizon is not informative).

CLI
---
  python -m engine.research.labels label-timestamps --symbol NIFTY --date 2026-06-02
  python -m engine.research.labels label-file --features data/research/flow_features/NIFTY/all.parquet

Storage
-------
  data/research/labels/{symbol}/labeled_{YYYY-MM-DD}.parquet  (per day)
  data/research/labels/{symbol}/all.parquet                   (combined)

Examples
--------
>>> from datetime import datetime
>>> from engine.research.labels import label_timestamps
>>> df = label_timestamps([datetime(2026, 5, 15, 10, 0)], symbol="NIFTY")
>>> df[["timestamp", "ret_15m", "cls_15m"]]
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime, date, time as dtime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


IST = timezone(timedelta(hours=5, minutes=30))
SESSION_OPEN = dtime(9, 15)
SESSION_CLOSE = dtime(15, 30)
LABELS_ROOT = Path("data/research/labels")

DEFAULT_HORIZONS: tuple[int, ...] = (15, 30, 60)
DEFAULT_UP = 0.0025
DEFAULT_DOWN = -0.0025
DEFAULT_TOLERANCE_MIN = 2


# --------------------------------------------------------------------------
# Internal: index of bars for fast lookup
# --------------------------------------------------------------------------


@dataclass
class _BarIndex:
    """Sorted view over a bars DataFrame with helpers for tolerance lookup
    and session-end resolution.

    The DataFrame is assumed to be sorted ascending by `ts`, tz-aware IST,
    with float `close` column. We use numpy arrays for O(log n) bisect.
    """

    ts_ns: np.ndarray  # int64 nanoseconds (UTC-equivalent monotonic)
    close: np.ndarray  # float64
    trade_date: np.ndarray  # object (datetime.date)

    @classmethod
    def from_df(cls, df: pd.DataFrame) -> "_BarIndex":
        if df.empty:
            return cls(
                ts_ns=np.array([], dtype="int64"),
                close=np.array([], dtype="float64"),
                trade_date=np.array([], dtype=object),
            )
        df = df.sort_values("ts").reset_index(drop=True)
        ts = pd.to_datetime(df["ts"], utc=False)
        if ts.dt.tz is None:
            ts = ts.dt.tz_localize(IST)
        else:
            ts = ts.dt.tz_convert(IST)
        # Force nanosecond resolution; pandas may store us/ms internally.
        # Convert via per-element .value to be precision-agnostic.
        ts_ns = np.array([t.value for t in ts], dtype="int64")
        close = df["close"].to_numpy(dtype="float64")
        trade_date = np.array([t.date() for t in ts], dtype=object)
        return cls(ts_ns=ts_ns, close=close, trade_date=trade_date)

    def __len__(self) -> int:
        return int(self.ts_ns.size)

    # ---- lookup helpers ----

    def find_nearest(
        self, target_ns: int, tolerance_ns: int
    ) -> int | None:
        """Return the row index whose ts is closest to target within tolerance,
        or None if no such bar exists.

        We require the matched bar to be on the SAME trading day as the
        target's IST date so that a tolerance-window match can't silently
        jump over a weekend/holiday gap.
        """
        if self.ts_ns.size == 0:
            return None
        pos = int(np.searchsorted(self.ts_ns, target_ns, side="left"))
        candidates: list[int] = []
        if pos < self.ts_ns.size:
            candidates.append(pos)
        if pos > 0:
            candidates.append(pos - 1)
        if not candidates:
            return None
        best = min(candidates, key=lambda i: abs(self.ts_ns[i] - target_ns))
        if abs(self.ts_ns[best] - target_ns) > tolerance_ns:
            return None
        # Same-day guard
        target_dt = pd.Timestamp(target_ns, tz="UTC").tz_convert(IST)
        if self.trade_date[best] != target_dt.date():
            return None
        return best

    def session_end_index(self, on_date: date) -> int | None:
        """Index of the last bar on `on_date`, or None if no bars that day."""
        mask = self.trade_date == on_date
        if not mask.any():
            return None
        # rightmost True
        idxs = np.flatnonzero(mask)
        return int(idxs[-1])


# --------------------------------------------------------------------------
# Internal: label one timestamp
# --------------------------------------------------------------------------


def _classify(ret: float, up_thr: float, down_thr: float) -> str | float:
    """3-class label. NaN propagates."""
    if ret is None or (isinstance(ret, float) and np.isnan(ret)):
        return np.nan
    if ret >= up_thr:
        return "UP"
    if ret <= down_thr:
        return "DOWN"
    return "NEUTRAL"


def _label_one(
    target_ts: pd.Timestamp,
    idx: _BarIndex,
    horizons_minutes: Sequence[int],
    include_eod: bool,
    up_threshold: float,
    down_threshold: float,
    tolerance_minutes: int,
    allow_overnight: bool,
) -> dict:
    """Compute all labels for a single timestamp. Returns a row dict."""
    tolerance_ns = int(tolerance_minutes) * 60 * 1_000_000_000

    row: dict = {"timestamp": target_ts}

    if len(idx) == 0:
        for h in horizons_minutes:
            row[f"ret_{h}m"] = np.nan
            row[f"cls_{h}m"] = np.nan
        if include_eod:
            row["ret_eod"] = np.nan
            row["cls_eod"] = np.nan
        return row

    # Locate t-bar
    target_ts_ist = target_ts.tz_convert(IST) if target_ts.tzinfo else target_ts.tz_localize(IST)
    t_ns = int(target_ts_ist.value)
    t_pos = idx.find_nearest(t_ns, tolerance_ns)

    if t_pos is None:
        c0 = np.nan
        t_date = target_ts_ist.date()
    else:
        c0 = float(idx.close[t_pos])
        t_date = idx.trade_date[t_pos]

    # Per-horizon returns
    for h in horizons_minutes:
        col_ret = f"ret_{h}m"
        col_cls = f"cls_{h}m"
        if t_pos is None or not np.isfinite(c0) or c0 <= 0:
            row[col_ret] = np.nan
            row[col_cls] = np.nan
            continue
        target_future_ns = t_ns + h * 60 * 1_000_000_000
        future_dt = pd.Timestamp(target_future_ns, tz="UTC").tz_convert(IST)
        # Session-boundary check (only relevant when based on t-bar's day)
        crosses_session = (
            future_dt.date() != t_date
            or future_dt.time() > SESSION_CLOSE
        )
        if crosses_session and not allow_overnight:
            row[col_ret] = np.nan
            row[col_cls] = np.nan
            continue
        f_pos = idx.find_nearest(target_future_ns, tolerance_ns)
        if f_pos is None:
            row[col_ret] = np.nan
            row[col_cls] = np.nan
            continue
        # When allow_overnight=False, find_nearest already enforces same-day.
        # When allow_overnight=True, we accept whatever bar is within tolerance.
        c1 = float(idx.close[f_pos])
        if not np.isfinite(c1):
            row[col_ret] = np.nan
            row[col_cls] = np.nan
            continue
        ret = (c1 - c0) / c0
        row[col_ret] = ret
        row[col_cls] = _classify(ret, up_threshold, down_threshold)

    # EOD
    if include_eod:
        if t_pos is None or not np.isfinite(c0) or c0 <= 0:
            row["ret_eod"] = np.nan
            row["cls_eod"] = np.nan
        else:
            eod_pos = idx.session_end_index(t_date)
            if eod_pos is None or eod_pos == t_pos:
                row["ret_eod"] = np.nan
                row["cls_eod"] = np.nan
            else:
                c_eod = float(idx.close[eod_pos])
                if not np.isfinite(c_eod):
                    row["ret_eod"] = np.nan
                    row["cls_eod"] = np.nan
                else:
                    ret_eod = (c_eod - c0) / c0
                    row["ret_eod"] = ret_eod
                    row["cls_eod"] = _classify(
                        ret_eod, up_threshold, down_threshold
                    )

    return row


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------


def _load_bars_df(symbol: str) -> pd.DataFrame:
    from engine.data.cache import load_cache
    return load_cache(symbol, "minute")


def label_timestamps(
    timestamps: Iterable[datetime] | pd.Series,
    symbol: str = "NIFTY",
    horizons_minutes: Sequence[int] = DEFAULT_HORIZONS,
    include_eod: bool = True,
    up_threshold: float = DEFAULT_UP,
    down_threshold: float = DEFAULT_DOWN,
    tolerance_minutes: int = DEFAULT_TOLERANCE_MIN,
    allow_overnight: bool = False,
    bars_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute forward-return labels for each timestamp.

    Parameters
    ----------
    timestamps : iterable of datetime / pandas Series
        Timestamps to label. Naive datetimes are assumed to be in IST.
    symbol : str
        Used to load bars from cache via `engine.data.cache.load_cache`. Ignored
        if `bars_df` is provided (handy for tests).
    horizons_minutes : tuple[int, ...]
        Forward-look horizons in minutes. Default (15, 30, 60).
    include_eod : bool
        If True, also produce `ret_eod` / `cls_eod`.
    up_threshold, down_threshold : float
        Classification thresholds for UP/DOWN/NEUTRAL.
    tolerance_minutes : int
        Max distance (in minutes) between requested timestamp and matched bar.
    allow_overnight : bool
        If False (default), any horizon that would cross the session boundary
        yields NaN for that horizon.
    bars_df : pd.DataFrame | None
        Optional pre-loaded bars. Must have columns ts (tz-aware or naive IST),
        close, ... Saves a cache reload in batch / test workflows.

    Returns
    -------
    pd.DataFrame
        Columns: timestamp, symbol, ret_<h>m, cls_<h>m for each h, plus
        ret_eod, cls_eod if include_eod. NaN cells indicate unavailable label.

    Edge cases
    ----------
    - Empty `timestamps`        -> empty DataFrame with correct schema.
    - t outside cache range     -> all-NaN row (still emitted).
    - t == session_end          -> ret_eod = NaN.
    - t + h > session_close and not allow_overnight -> ret_<h>m = NaN.
    - t + h within session but next bar missing within tolerance -> NaN.
    """
    if bars_df is None:
        bars_df = _load_bars_df(symbol)
    idx = _BarIndex.from_df(bars_df)

    # Normalize timestamps to tz-aware IST pd.Timestamp series
    if isinstance(timestamps, pd.Series):
        ts_series = pd.to_datetime(timestamps)
    else:
        ts_series = pd.to_datetime(pd.Series(list(timestamps)))
    if not len(ts_series):
        cols = ["timestamp", "symbol"]
        for h in horizons_minutes:
            cols += [f"ret_{h}m", f"cls_{h}m"]
        if include_eod:
            cols += ["ret_eod", "cls_eod"]
        return pd.DataFrame(columns=cols)

    if ts_series.dt.tz is None:
        ts_series = ts_series.dt.tz_localize(IST)
    else:
        ts_series = ts_series.dt.tz_convert(IST)

    rows = []
    for ts in ts_series:
        rows.append(
            _label_one(
                target_ts=ts,
                idx=idx,
                horizons_minutes=horizons_minutes,
                include_eod=include_eod,
                up_threshold=up_threshold,
                down_threshold=down_threshold,
                tolerance_minutes=tolerance_minutes,
                allow_overnight=allow_overnight,
            )
        )

    df = pd.DataFrame(rows)
    df.insert(1, "symbol", symbol)
    # Reorder columns deterministically
    ordered = ["timestamp", "symbol"]
    for h in horizons_minutes:
        ordered += [f"ret_{h}m", f"cls_{h}m"]
    if include_eod:
        ordered += ["ret_eod", "cls_eod"]
    # Some horizons may be absent if user passes weird tuple — keep only ones present
    ordered = [c for c in ordered if c in df.columns]
    return df[ordered]


def label_features_file(
    features_path: Path,
    out_path: Path | None = None,
    horizons_minutes: Sequence[int] = DEFAULT_HORIZONS,
    include_eod: bool = True,
    symbol: str = "NIFTY",
    up_threshold: float = DEFAULT_UP,
    down_threshold: float = DEFAULT_DOWN,
    tolerance_minutes: int = DEFAULT_TOLERANCE_MIN,
    allow_overnight: bool = False,
) -> Path:
    """Read a flow_features parquet (long or wide; must contain a `timestamp`
    column), compute labels for each UNIQUE timestamp, join back, persist.

    The features file is not modified. Output is written to `out_path` (or to
    `data/research/labels/{symbol}/labeled_<features-filename>.parquet`).
    """
    features_path = Path(features_path)
    if not features_path.exists():
        raise FileNotFoundError(features_path)
    feat = pd.read_parquet(features_path)
    if "timestamp" not in feat.columns:
        raise ValueError(
            f"features file {features_path} has no 'timestamp' column "
            f"(found: {list(feat.columns)})"
        )

    unique_ts = pd.to_datetime(feat["timestamp"]).drop_duplicates().sort_values()
    labels = label_timestamps(
        unique_ts,
        symbol=symbol,
        horizons_minutes=horizons_minutes,
        include_eod=include_eod,
        up_threshold=up_threshold,
        down_threshold=down_threshold,
        tolerance_minutes=tolerance_minutes,
        allow_overnight=allow_overnight,
    )

    # Normalize timezone of join keys so the merge succeeds.
    feat_ts = pd.to_datetime(feat["timestamp"])
    if feat_ts.dt.tz is None:
        feat_ts = feat_ts.dt.tz_localize(IST)
    else:
        feat_ts = feat_ts.dt.tz_convert(IST)
    feat = feat.copy()
    feat["timestamp"] = feat_ts

    merged = feat.merge(labels.drop(columns=["symbol"]), on="timestamp", how="left")

    if out_path is None:
        out_dir = LABELS_ROOT / symbol
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"labeled_{features_path.stem}.parquet"
    else:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

    merged.to_parquet(out_path, index=False)
    return out_path


def load_labeled_dataset(symbol: str = "NIFTY") -> pd.DataFrame:
    """Read the combined features+labels parquet and pivot long-form feature
    rows into a wide DataFrame ready for predictive-power analysis.

    Looks for `data/research/labels/{symbol}/all.parquet`. If the dataset is
    already wide (no `feature_name`/`value` columns), it is returned as-is.
    """
    path = LABELS_ROOT / symbol / "all.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"No combined labeled dataset at {path}. "
            f"Run label_features_file first."
        )
    df = pd.read_parquet(path)

    if "feature_name" in df.columns and "value" in df.columns:
        label_cols = [c for c in df.columns if c.startswith(("ret_", "cls_"))]
        id_cols = [c for c in df.columns
                   if c not in ("feature_name", "value") and c not in label_cols]
        wide = df.pivot_table(
            index=id_cols,
            columns="feature_name",
            values="value",
            aggfunc="first",
        ).reset_index()
        # Re-join labels (one row per timestamp/symbol)
        labels = (
            df[id_cols + label_cols]
            .drop_duplicates(subset=id_cols)
            .reset_index(drop=True)
        )
        wide = wide.merge(labels, on=id_cols, how="left")
        return wide

    return df


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _cmd_label_timestamps(args):
    symbol = args.symbol
    bars = _load_bars_df(symbol)
    if bars.empty:
        print(f"No cache for {symbol}", file=sys.stderr)
        sys.exit(1)

    day = date.fromisoformat(args.date) if args.date else bars["ts"].max().date()
    bars_ts = pd.to_datetime(bars["ts"])
    if bars_ts.dt.tz is None:
        bars_ts = bars_ts.dt.tz_localize(IST)
    else:
        bars_ts = bars_ts.dt.tz_convert(IST)
    day_mask = bars_ts.dt.date == day
    day_bars = bars.loc[day_mask].copy()
    if day_bars.empty:
        print(f"No bars on {day}", file=sys.stderr)
        sys.exit(1)

    timestamps = pd.to_datetime(day_bars["ts"]).tolist()
    df = label_timestamps(
        timestamps,
        symbol=symbol,
        horizons_minutes=tuple(args.horizons),
        include_eod=not args.no_eod,
        bars_df=bars,
    )

    out_dir = LABELS_ROOT / symbol
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"labeled_{day.isoformat()}.parquet"
    df.to_parquet(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")
    _print_class_dist(df)


def _cmd_label_file(args):
    out = label_features_file(
        features_path=Path(args.features),
        out_path=Path(args.out) if args.out else None,
        horizons_minutes=tuple(args.horizons),
        include_eod=not args.no_eod,
        symbol=args.symbol,
        allow_overnight=args.allow_overnight,
    )
    print(f"Wrote labeled features to {out}")
    df = pd.read_parquet(out)
    _print_class_dist(df)


def _print_class_dist(df: pd.DataFrame) -> None:
    cls_cols = [c for c in df.columns if c.startswith("cls_")]
    if not cls_cols:
        return
    print("\nClassification distribution:")
    for c in cls_cols:
        vc = df[c].value_counts(dropna=False).to_dict()
        total = sum(vc.values())
        parts = []
        for k in ["UP", "NEUTRAL", "DOWN"]:
            n = vc.get(k, 0)
            pct = (100.0 * n / total) if total else 0.0
            parts.append(f"{k}={n} ({pct:.1f}%)")
        nan_n = vc.get(np.nan, 0) + sum(
            v for k, v in vc.items() if isinstance(k, float) and np.isnan(k)
        )
        parts.append(f"NaN={nan_n}")
        print(f"  {c}: " + ", ".join(parts))


def main():
    p = argparse.ArgumentParser(description="Forward-return label generator")
    sub = p.add_subparsers(dest="cmd", required=True)

    pt = sub.add_parser("label-timestamps",
                        help="Label every minute bar on a given date")
    pt.add_argument("--symbol", default="NIFTY")
    pt.add_argument("--date", default=None,
                    help="YYYY-MM-DD; default = latest day in cache")
    pt.add_argument("--horizons", type=int, nargs="+",
                    default=list(DEFAULT_HORIZONS))
    pt.add_argument("--no-eod", action="store_true")
    pt.set_defaults(func=_cmd_label_timestamps)

    pf = sub.add_parser("label-file",
                        help="Label every unique timestamp in a features parquet")
    pf.add_argument("--features", required=True)
    pf.add_argument("--out", default=None)
    pf.add_argument("--symbol", default="NIFTY")
    pf.add_argument("--horizons", type=int, nargs="+",
                    default=list(DEFAULT_HORIZONS))
    pf.add_argument("--no-eod", action="store_true")
    pf.add_argument("--allow-overnight", action="store_true")
    pf.set_defaults(func=_cmd_label_file)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
