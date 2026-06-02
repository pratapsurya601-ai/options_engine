"""
Walk-forward validation for option-flow features.

Detects lookahead bias and measures out-of-sample (OOS) feature stability by
splitting the labeled dataset along time order ONLY. No shuffling. No random
k-fold. Train indices always strictly precede test indices.

Three methods are supported:

1. rolling           — fixed train window slides forward by `test_size`.
2. expanding         — train always starts at index 0 and grows.
3. anchored_purged   — expanding, with an `embargo_minutes` gap inserted
                       between train end and test start to prevent feature/
                       label overlap (e.g. a 60-min forward return label
                       computed from a point at t leaks into any training
                       point in [t-60, t]).

Per-window we compute Spearman IC, Pearson, IC decay, and a stability score.
Per (feature, horizon) we aggregate across windows into a verdict:
    ROBUST | DECAYING | UNSTABLE | NEGATIVE_OOS | NEUTRAL | INSUFFICIENT

The output schema is JOINABLE on (feature, horizon) with predictive_power's
scores.parquet so the dashboard can show
    "IC=0.08 in-sample but walk-forward verdict=DECAYING — not tradable".

CLI
---
    python -m engine.research.walkforward run --symbol NIFTY \\
        --method rolling --train 500 --test 100
    python -m engine.research.walkforward summary --symbol NIFTY
    python -m engine.research.walkforward feature --symbol NIFTY \\
        --feature pcr_oi --horizon ret_60m

Storage
-------
    data/research/walkforward/{symbol}/summaries.parquet
    data/research/walkforward/{symbol}/windows.parquet

Honest tiny-data note
---------------------
With ~17 unique timestamps and a default train=500, walk-forward produces
ZERO windows. The module handles that cleanly and returns an empty result with
the right schema. Walk-forward becomes informative after ~30+ trading days of
5-minute snapshots (~2000 rows).
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import pandas as pd


WALKFORWARD_ROOT = Path("data/research/walkforward")

DEFAULT_HORIZONS: tuple[str, ...] = ("ret_15m", "ret_30m", "ret_60m", "ret_eod")
NON_FEATURE_COLS = {
    "timestamp", "symbol", "expiry",
    "ret_15m", "ret_30m", "ret_60m", "ret_eod",
    "cls_15m", "cls_30m", "cls_60m", "cls_eod",
}


# ---------------------------------------------------------------------------
# Config / result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class WalkForwardConfig:
    """Configuration for a walk-forward run.

    Parameters
    ----------
    train_size : int
        Number of consecutive rows in the train slice (for rolling) or initial
        train slice (for expanding / anchored_purged).
    test_size : int
        Number of consecutive rows in the test slice. The window also advances
        by `test_size` rows per step.
    method : str
        One of {"rolling", "expanding", "anchored_purged"}.
    embargo_minutes : int
        Only used for `anchored_purged`. Test rows whose timestamp is within
        `embargo_minutes` of train_end are dropped to prevent label/feature
        bleed (e.g. ret_60m at t-30 overlaps with feature at t).
    min_train_size : int
        Refuse to emit a window unless the train slice has at least this many
        rows. Useful so a small expanding window's first iteration is not
        starved.
    horizons : tuple[str, ...]
        Horizon column names to score against.
    """
    train_size: int = 500
    test_size: int = 100
    method: str = "rolling"
    embargo_minutes: int = 60
    min_train_size: int = 200
    horizons: tuple[str, ...] = DEFAULT_HORIZONS


@dataclass
class WindowResult:
    window_id: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    train_n: int
    test_n: int
    feature: str
    horizon: str
    train_ic: float
    test_ic: float
    train_pearson: float
    test_pearson: float
    ic_decay: float
    stability_score: float


@dataclass
class WalkForwardSummary:
    feature: str
    horizon: str
    n_windows: int
    mean_train_ic: float
    mean_test_ic: float
    mean_ic_decay: float
    test_ic_std: float
    pct_windows_test_ic_positive: float
    stability_score_mean: float
    verdict: str


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def _spearman_ic(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation. Returns NaN if <3 valid pairs."""
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    xr = pd.Series(x[mask]).rank().to_numpy()
    yr = pd.Series(y[mask]).rank().to_numpy()
    if np.std(xr) == 0 or np.std(yr) == 0:
        return float("nan")
    return float(np.corrcoef(xr, yr)[0, 1])


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation. NaN if <3 valid pairs or zero variance."""
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    xv = x[mask]
    yv = y[mask]
    if np.std(xv) == 0 or np.std(yv) == 0:
        return float("nan")
    return float(np.corrcoef(xv, yv)[0, 1])


def _stability_score(train_ic: float, test_ic: float) -> float:
    """|test_ic| / max(|train_ic|, eps), capped at 1.5."""
    if not np.isfinite(train_ic) or not np.isfinite(test_ic):
        return float("nan")
    return float(min(abs(test_ic) / max(abs(train_ic), 1e-6), 1.5))


# ---------------------------------------------------------------------------
# Splitters (pure)
# ---------------------------------------------------------------------------


def _iter_splits(
    n: int,
    config: WalkForwardConfig,
) -> Iterator[tuple[int, int, int, int]]:
    """Yield (train_lo, train_hi, test_lo, test_hi) index tuples.

    Indices are end-exclusive (i.e. train slice = [train_lo:train_hi]).
    No timestamp logic here — that is layered on in `walk_forward` so this
    splitter remains pure and unit-testable with synthetic n.
    """
    train_size = config.train_size
    test_size = config.test_size
    method = config.method

    if test_size <= 0 or train_size <= 0:
        return
    if n < max(config.min_train_size, train_size) + test_size:
        return

    if method == "rolling":
        test_lo = train_size
        while test_lo + test_size <= n:
            train_lo = test_lo - train_size
            train_hi = test_lo
            test_hi = test_lo + test_size
            if train_hi - train_lo < config.min_train_size:
                test_lo += test_size
                continue
            yield train_lo, train_hi, test_lo, test_hi
            test_lo += test_size

    elif method in ("expanding", "anchored_purged"):
        test_lo = train_size
        while test_lo + test_size <= n:
            train_lo = 0
            train_hi = test_lo
            test_hi = test_lo + test_size
            if train_hi - train_lo < config.min_train_size:
                test_lo += test_size
                continue
            yield train_lo, train_hi, test_lo, test_hi
            test_lo += test_size
    else:
        raise ValueError(
            f"unknown method {method!r}; expected one of "
            f"rolling/expanding/anchored_purged"
        )


# ---------------------------------------------------------------------------
# Core walk-forward
# ---------------------------------------------------------------------------


def _prepare_frame(labeled_df: pd.DataFrame) -> pd.DataFrame:
    """Sort ascending by timestamp, drop duplicate timestamps, reset index."""
    if "timestamp" not in labeled_df.columns:
        raise ValueError("labeled_df must have a 'timestamp' column")
    df = labeled_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    # If the same timestamp appears twice with conflicting feature columns,
    # we keep the FIRST (deterministic) to avoid silent reordering.
    df = df.drop_duplicates(subset=["timestamp"], keep="first").reset_index(drop=True)
    return df


def walk_forward(
    labeled_df: pd.DataFrame,
    feature: str,
    horizon: str,
    config: WalkForwardConfig | None = None,
) -> list[WindowResult]:
    """Run walk-forward for a single (feature, horizon).

    Parameters
    ----------
    labeled_df : pd.DataFrame
        Wide-format frame from `load_labeled_dataset`. Must contain `timestamp`,
        `feature`, and `horizon` columns.
    feature : str
        Feature column name (e.g. "pcr_oi").
    horizon : str
        Forward-return column name (e.g. "ret_60m").
    config : WalkForwardConfig | None
        If None, defaults are used.

    Returns
    -------
    list[WindowResult]
        Per-window results. Empty list if there is insufficient data.

    Example
    -------
    >>> from engine.research.labels import load_labeled_dataset
    >>> df = load_labeled_dataset("NIFTY")
    >>> windows = walk_forward(df, "pcr_oi", "ret_60m",
    ...                        WalkForwardConfig(train_size=300, test_size=50))
    """
    cfg = config or WalkForwardConfig()
    if feature not in labeled_df.columns:
        return []
    if horizon not in labeled_df.columns:
        return []

    df = _prepare_frame(labeled_df)
    n = len(df)
    if n == 0:
        return []

    feat_arr = df[feature].to_numpy(dtype="float64")
    label_arr = df[horizon].to_numpy(dtype="float64")
    # Strip tz so the underlying numpy array is datetime64[ns] (supports
    # vectorized subtraction). The python-side window timestamps are still
    # derived via pd.Timestamp(...).to_pydatetime() below.
    ts_series = df["timestamp"]
    if hasattr(ts_series.dt, "tz") and ts_series.dt.tz is not None:
        ts_series = ts_series.dt.tz_convert("UTC").dt.tz_localize(None)
    ts_arr = ts_series.to_numpy(dtype="datetime64[ns]")

    embargo_td = np.timedelta64(int(cfg.embargo_minutes) * 60, "s")

    results: list[WindowResult] = []
    for window_id, (tr_lo, tr_hi, te_lo, te_hi) in enumerate(_iter_splits(n, cfg)):
        train_start_ts = pd.Timestamp(ts_arr[tr_lo]).to_pydatetime()
        train_end_ts = pd.Timestamp(ts_arr[tr_hi - 1]).to_pydatetime()
        test_start_ts = pd.Timestamp(ts_arr[te_lo]).to_pydatetime()
        test_end_ts = pd.Timestamp(ts_arr[te_hi - 1]).to_pydatetime()

        # ===== Hard lookahead guard =====
        assert train_end_ts <= test_start_ts, (
            f"LOOKAHEAD VIOLATION: window {window_id} train_end={train_end_ts} "
            f"> test_start={test_start_ts}"
        )

        te_slice = slice(te_lo, te_hi)
        if cfg.method == "anchored_purged":
            # Drop test rows that fall within embargo_minutes of train_end.
            te_ts_block = ts_arr[te_lo:te_hi]
            gap = te_ts_block - ts_arr[tr_hi - 1]
            keep_mask = gap >= embargo_td
            te_idx = np.flatnonzero(keep_mask) + te_lo
            if te_idx.size == 0:
                continue
            test_feat = feat_arr[te_idx]
            test_label = label_arr[te_idx]
            # Update reported test_start to reflect post-embargo first row
            test_start_ts = pd.Timestamp(ts_arr[te_idx[0]]).to_pydatetime()
            test_end_ts = pd.Timestamp(ts_arr[te_idx[-1]]).to_pydatetime()
            # Re-assert post-embargo invariant
            gap_min = (test_start_ts - train_end_ts).total_seconds() / 60.0
            assert gap_min >= cfg.embargo_minutes - 1e-6, (
                f"EMBARGO VIOLATION: {gap_min:.3f} < {cfg.embargo_minutes}"
            )
        else:
            test_feat = feat_arr[te_slice]
            test_label = label_arr[te_slice]

        train_feat = feat_arr[tr_lo:tr_hi]
        train_label = label_arr[tr_lo:tr_hi]

        train_ic = _spearman_ic(train_feat, train_label)
        test_ic = _spearman_ic(test_feat, test_label)
        train_pearson = _pearson(train_feat, train_label)
        test_pearson = _pearson(test_feat, test_label)
        ic_decay = (
            (train_ic - test_ic)
            if np.isfinite(train_ic) and np.isfinite(test_ic)
            else float("nan")
        )
        stability = _stability_score(train_ic, test_ic)

        results.append(
            WindowResult(
                window_id=window_id,
                train_start=train_start_ts,
                train_end=train_end_ts,
                test_start=test_start_ts,
                test_end=test_end_ts,
                train_n=int(tr_hi - tr_lo),
                test_n=int(len(test_feat)),
                feature=feature,
                horizon=horizon,
                train_ic=train_ic,
                test_ic=test_ic,
                train_pearson=train_pearson,
                test_pearson=test_pearson,
                ic_decay=ic_decay,
                stability_score=stability,
            )
        )

    return results


# ---------------------------------------------------------------------------
# Summary / verdict
# ---------------------------------------------------------------------------


def _verdict(
    n_windows: int,
    mean_train_ic: float,
    mean_test_ic: float,
    mean_ic_decay: float,
    test_ic_std: float,
    pct_pos: float,
) -> str:
    if n_windows < 3:
        return "INSUFFICIENT"
    # NEGATIVE_OOS first — consistent reverse signal is its own category
    if np.isfinite(mean_test_ic) and mean_test_ic < -0.02:
        return "NEGATIVE_OOS"
    if (np.isfinite(mean_test_ic) and mean_test_ic > 0.05
            and pct_pos > 0.6
            and np.isfinite(mean_ic_decay) and mean_ic_decay < 0.05):
        return "ROBUST"
    if (np.isfinite(mean_train_ic) and mean_train_ic > 0.05
            and np.isfinite(mean_test_ic)
            and (mean_test_ic - mean_train_ic) < -0.05):
        return "DECAYING"
    if np.isfinite(test_ic_std) and test_ic_std > 0.10:
        return "UNSTABLE"
    return "NEUTRAL"


def walk_forward_summary(
    window_results: list[WindowResult],
) -> WalkForwardSummary:
    """Aggregate a list of WindowResult (same feature+horizon) into a verdict.

    If the list is empty, a synthetic INSUFFICIENT row is returned with
    feature='' / horizon='' so the caller can still serialize a uniform schema
    if desired. In practice `walk_forward_all` handles empty lists by skipping.
    """
    if not window_results:
        return WalkForwardSummary(
            feature="", horizon="", n_windows=0,
            mean_train_ic=float("nan"), mean_test_ic=float("nan"),
            mean_ic_decay=float("nan"), test_ic_std=float("nan"),
            pct_windows_test_ic_positive=float("nan"),
            stability_score_mean=float("nan"),
            verdict="INSUFFICIENT",
        )

    feat = window_results[0].feature
    horizon = window_results[0].horizon
    if any(w.feature != feat or w.horizon != horizon for w in window_results):
        raise ValueError(
            "walk_forward_summary expects window_results from a single "
            "(feature, horizon) pair"
        )

    test_ics = np.array([w.test_ic for w in window_results], dtype="float64")
    train_ics = np.array([w.train_ic for w in window_results], dtype="float64")
    decays = np.array([w.ic_decay for w in window_results], dtype="float64")
    stabs = np.array([w.stability_score for w in window_results], dtype="float64")

    mean_test_ic = float(np.nanmean(test_ics)) if np.isfinite(test_ics).any() else float("nan")
    mean_train_ic = float(np.nanmean(train_ics)) if np.isfinite(train_ics).any() else float("nan")
    mean_decay = float(np.nanmean(decays)) if np.isfinite(decays).any() else float("nan")
    test_ic_std = float(np.nanstd(test_ics, ddof=0)) if np.isfinite(test_ics).any() else float("nan")
    finite_mask = np.isfinite(test_ics)
    if finite_mask.any():
        pct_pos = float((test_ics[finite_mask] > 0).mean())
    else:
        pct_pos = float("nan")
    stab_mean = float(np.nanmean(stabs)) if np.isfinite(stabs).any() else float("nan")

    verdict = _verdict(
        n_windows=len(window_results),
        mean_train_ic=mean_train_ic,
        mean_test_ic=mean_test_ic,
        mean_ic_decay=mean_decay,
        test_ic_std=test_ic_std,
        pct_pos=pct_pos if np.isfinite(pct_pos) else 0.0,
    )

    return WalkForwardSummary(
        feature=feat,
        horizon=horizon,
        n_windows=len(window_results),
        mean_train_ic=mean_train_ic,
        mean_test_ic=mean_test_ic,
        mean_ic_decay=mean_decay,
        test_ic_std=test_ic_std,
        pct_windows_test_ic_positive=pct_pos,
        stability_score_mean=stab_mean,
        verdict=verdict,
    )


# ---------------------------------------------------------------------------
# All-features driver
# ---------------------------------------------------------------------------


def _auto_features(labeled_df: pd.DataFrame) -> list[str]:
    feats: list[str] = []
    for c in labeled_df.columns:
        if c in NON_FEATURE_COLS:
            continue
        if c.startswith(("ret_", "cls_")):
            continue
        if pd.api.types.is_numeric_dtype(labeled_df[c]):
            feats.append(c)
    return feats


def _empty_summaries_df() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "feature", "horizon", "n_windows",
        "mean_train_ic", "mean_test_ic", "mean_ic_decay",
        "test_ic_std", "pct_windows_test_ic_positive",
        "stability_score_mean", "verdict",
    ])


def _empty_windows_df() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "window_id", "train_start", "train_end", "test_start", "test_end",
        "train_n", "test_n", "feature", "horizon",
        "train_ic", "test_ic", "train_pearson", "test_pearson",
        "ic_decay", "stability_score",
    ])


def walk_forward_all(
    labeled_df: pd.DataFrame,
    features: list[str] | None = None,
    horizons: tuple[str, ...] = DEFAULT_HORIZONS,
    config: WalkForwardConfig | None = None,
    min_samples: int = 50,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run walk-forward for every (feature, horizon) combo.

    Returns
    -------
    (summaries_df, windows_df)
        Long-format frames ready to persist / load in the dashboard. Both have
        stable schemas even when input data is too small to produce any
        windows.
    """
    cfg = config or WalkForwardConfig(horizons=horizons)
    horizons = tuple(horizons) if horizons else cfg.horizons

    if labeled_df is None or labeled_df.empty:
        return _empty_summaries_df(), _empty_windows_df()

    feats = features if features is not None else _auto_features(labeled_df)
    n = len(labeled_df)
    needed = cfg.train_size + cfg.test_size
    if n < needed:
        print(
            f"Insufficient data for walk-forward "
            f"(n={n}, need >={needed}). Returning empty result."
        )
        return _empty_summaries_df(), _empty_windows_df()
    if n < min_samples:
        return _empty_summaries_df(), _empty_windows_df()

    summary_rows: list[dict] = []
    window_rows: list[dict] = []

    for feat in feats:
        for hz in horizons:
            if hz not in labeled_df.columns:
                continue
            results = walk_forward(labeled_df, feat, hz, cfg)
            if not results:
                continue
            for w in results:
                window_rows.append(asdict(w))
            summary_rows.append(asdict(walk_forward_summary(results)))

    if not window_rows:
        return _empty_summaries_df(), _empty_windows_df()

    return pd.DataFrame(summary_rows), pd.DataFrame(window_rows)


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------


def save_walkforward_results(
    summaries_df: pd.DataFrame,
    windows_df: pd.DataFrame,
    symbol: str = "NIFTY",
) -> tuple[Path, Path]:
    """Persist to data/research/walkforward/{symbol}/{summaries,windows}.parquet"""
    out_dir = WALKFORWARD_ROOT / symbol
    out_dir.mkdir(parents=True, exist_ok=True)
    summ_path = out_dir / "summaries.parquet"
    win_path = out_dir / "windows.parquet"
    summaries_df.to_parquet(summ_path, index=False)
    windows_df.to_parquet(win_path, index=False)
    return summ_path, win_path


def load_walkforward_results(symbol: str = "NIFTY") -> tuple[pd.DataFrame, pd.DataFrame]:
    out_dir = WALKFORWARD_ROOT / symbol
    summ_path = out_dir / "summaries.parquet"
    win_path = out_dir / "windows.parquet"
    summ = pd.read_parquet(summ_path) if summ_path.exists() else _empty_summaries_df()
    wins = pd.read_parquet(win_path) if win_path.exists() else _empty_windows_df()
    return summ, wins


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _load_labeled(symbol: str) -> pd.DataFrame:
    from engine.research.labels import load_labeled_dataset
    return load_labeled_dataset(symbol)


def _cmd_run(args) -> None:
    df = _load_labeled(args.symbol)
    cfg = WalkForwardConfig(
        train_size=args.train,
        test_size=args.test,
        method=args.method,
        embargo_minutes=args.embargo,
        min_train_size=args.min_train,
    )
    print(
        f"Walk-forward: symbol={args.symbol} method={cfg.method} "
        f"train={cfg.train_size} test={cfg.test_size} n_rows={len(df)}"
    )
    summaries, windows = walk_forward_all(df, config=cfg)
    if summaries.empty:
        print("No windows produced. Walk-forward becomes informative after "
              "~30+ trading days of snapshots (~2000 rows).")
    else:
        print(f"Produced {len(windows)} window-rows across "
              f"{len(summaries)} (feature, horizon) pairs.")
        print()
        verdict_counts = summaries["verdict"].value_counts().to_dict()
        print("Verdict counts:")
        for k, v in verdict_counts.items():
            print(f"  {k}: {v}")
    summ_path, win_path = save_walkforward_results(summaries, windows, args.symbol)
    print(f"\nWrote {summ_path}")
    print(f"Wrote {win_path}")


def _cmd_summary(args) -> None:
    summ, _ = load_walkforward_results(args.symbol)
    if summ.empty:
        print(f"No walk-forward summaries for {args.symbol}. Run "
              f"`walkforward run` first.")
        return
    cols = [
        "feature", "horizon", "n_windows", "mean_train_ic", "mean_test_ic",
        "mean_ic_decay", "test_ic_std", "pct_windows_test_ic_positive",
        "verdict",
    ]
    show = summ[cols].copy()
    for c in ["mean_train_ic", "mean_test_ic", "mean_ic_decay",
              "test_ic_std", "pct_windows_test_ic_positive"]:
        show[c] = show[c].astype(float).round(4)
    with pd.option_context("display.max_rows", 200, "display.width", 160):
        print(show.to_string(index=False))


def _cmd_feature(args) -> None:
    summ, wins = load_walkforward_results(args.symbol)
    if wins.empty:
        print(f"No walk-forward windows for {args.symbol}.")
        return
    sub = wins[(wins["feature"] == args.feature) & (wins["horizon"] == args.horizon)]
    if sub.empty:
        print(f"No windows for feature={args.feature} horizon={args.horizon}.")
        return
    show_cols = [
        "window_id", "train_start", "train_end", "test_start", "test_end",
        "train_n", "test_n",
        "train_ic", "test_ic", "ic_decay", "stability_score",
    ]
    print(f"Feature: {args.feature}  Horizon: {args.horizon}\n")
    with pd.option_context("display.max_rows", 200, "display.width", 200):
        print(sub[show_cols].to_string(index=False))
    s = summ[(summ["feature"] == args.feature) & (summ["horizon"] == args.horizon)]
    if not s.empty:
        r = s.iloc[0]
        print(
            f"\nSummary: n_windows={r.n_windows} "
            f"mean_train_ic={r.mean_train_ic:.4f} "
            f"mean_test_ic={r.mean_test_ic:.4f} "
            f"mean_decay={r.mean_ic_decay:.4f} "
            f"test_ic_std={r.test_ic_std:.4f} "
            f"verdict={r.verdict}"
        )


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="walkforward")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run", help="Run walk-forward over all features/horizons")
    pr.add_argument("--symbol", default="NIFTY")
    pr.add_argument("--method", default="rolling",
                    choices=["rolling", "expanding", "anchored_purged"])
    pr.add_argument("--train", type=int, default=500)
    pr.add_argument("--test", type=int, default=100)
    pr.add_argument("--embargo", type=int, default=60)
    pr.add_argument("--min-train", dest="min_train", type=int, default=200)
    pr.set_defaults(func=_cmd_run)

    ps = sub.add_parser("summary", help="Print stored summary table")
    ps.add_argument("--symbol", default="NIFTY")
    ps.set_defaults(func=_cmd_summary)

    pf = sub.add_parser("feature", help="Drill into a (feature, horizon) pair")
    pf.add_argument("--symbol", default="NIFTY")
    pf.add_argument("--feature", required=True)
    pf.add_argument("--horizon", required=True)
    pf.set_defaults(func=_cmd_feature)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
