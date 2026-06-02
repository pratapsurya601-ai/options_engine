"""
Conditional-edge discovery: regime-aware predictive power.

Hypothesis
----------
A feature with weak overall predictive signal may carry strong signal
*conditional* on the prevailing market regime. For example,
`put_writing_velocity` may be informative only when the tape is in PANIC
mode and pure noise otherwise. Averaging across regimes washes that edge
out; conditioning on the regime label exposes it.

For every (feature, horizon, regime) triplet this module computes:

    ic_overall          rank correlation across ALL samples
    ic_in_regime        rank correlation only when regime == this regime
    n_in_regime         sample count in that regime
    delta_ic            ic_in_regime - ic_overall
    is_regime_specific  |delta_ic| > 0.05 AND n_in_regime >= min_samples_per_regime
    composite_score     full composite from predictive_power (in regime)

A baseline "OVERALL" row is emitted per (feature, horizon) so callers
can read the unconditional metrics from the same DataFrame.

The actual statistics (IC, Pearson, MI, decile spread, confidence band)
are NOT re-implemented here -- we delegate to
`predictive_power.analyze_feature` for every slice. This module is purely
about the join and the conditional split.

Storage
-------
    data/research/conditional_edge/{symbol}/scores.parquet
    data/research/conditional_edge/{symbol}/scores.md

CLI
---
    python -m engine.research.conditional_edge run  --symbol NIFTY
    python -m engine.research.conditional_edge top  --symbol NIFTY --horizon ret_60m --k 10
    python -m engine.research.conditional_edge show --symbol NIFTY --feature pcr_oi

Honest caveat on tiny data
--------------------------
The current chain-snapshot dataset is a single trading day, ~17
timestamps, which means at most ONE regime label is present. The
"OVERALL" row and the one populated regime row will be identical, the
other three regime rows will have n=0, and every result lands in
INSUFFICIENT. The module handles this cleanly and the markdown report
documents it explicitly -- nothing here pretends a 17-sample edge is
real.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from engine.research.predictive_power import (
    analyze_feature,
    DEFAULT_HORIZONS,
    DEFAULT_MIN_SAMPLES,
    _auto_detect_features,
)
from engine.research.regimes import (
    REGIMES,
    STORAGE_DIR as REGIMES_DIR,
    build_regime_timeline,
    load_timeline,
)


IST = timezone(timedelta(hours=5, minutes=30))
COND_EDGE_ROOT = Path("data/research/conditional_edge")

OVERALL_LABEL = "OVERALL"
DEFAULT_DELTA_IC_THRESHOLD = 0.05


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------


@dataclass
class ConditionalEdgeResult:
    """One row of conditional-edge output.

    `regime` is either one of the four regime labels or the special string
    "OVERALL" -- the unconditional baseline. `delta_ic_vs_overall` is 0.0
    for the OVERALL row by construction.
    """

    feature: str
    horizon: str
    regime: str  # one of REGIMES or "OVERALL"
    n: int
    ic: float
    pearson: float
    mutual_info: float
    decile_spread: float
    composite_score: float
    delta_ic_vs_overall: float
    is_regime_specific: bool
    confidence_band: str

    def to_row(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Join: features + labels + regimes
# ---------------------------------------------------------------------------


def _load_labeled_wide(symbol: str) -> pd.DataFrame:
    """Load the labeled dataset and pivot defensively (handle both
    `feature_value` and `value` column names)."""
    from engine.research.labels import load_labeled_dataset, label_features_file, LABELS_ROOT
    from engine.research.flow_features import FEATURES_ROOT

    combined = LABELS_ROOT / symbol / "all.parquet"
    if not combined.exists():
        feats_all = FEATURES_ROOT / symbol / "all.parquet"
        if not feats_all.exists():
            raise FileNotFoundError(
                f"No flow_features all.parquet at {feats_all}. "
                f"Run engine.research.flow_features build-all first."
            )
        label_features_file(features_path=feats_all,
                            out_path=combined, symbol=symbol)

    df = load_labeled_dataset(symbol)

    # load_labeled_dataset() only pivots when value column is named 'value'.
    # Upstream writes 'feature_value'; pivot defensively here.
    if "feature_name" in df.columns and "feature_value" in df.columns:
        label_cols = [c for c in df.columns if c.startswith(("ret_", "cls_"))]
        id_cols = [c for c in df.columns
                   if c not in ("feature_name", "feature_value")
                   and c not in label_cols]
        wide = df.pivot_table(
            index=id_cols, columns="feature_name",
            values="feature_value", aggfunc="first",
        ).reset_index()
        labels = (df[id_cols + label_cols]
                  .drop_duplicates(subset=id_cols)
                  .reset_index(drop=True))
        df = wide.merge(labels, on=id_cols, how="left")
    return df


def _load_regime_timeline_for_join(symbol: str) -> pd.DataFrame:
    """Load the persisted daily regime timeline; build it if missing.

    Returns a DataFrame with columns ['date', 'regime'] keyed by trade date.
    """
    timeline = load_timeline(symbol, "day")
    if timeline.empty:
        try:
            timeline = build_regime_timeline(symbol=symbol, interval="day")
        except Exception:
            return pd.DataFrame(columns=["date", "regime"])
    if timeline.empty:
        return pd.DataFrame(columns=["date", "regime"])

    ts = pd.to_datetime(timeline["timestamp"])
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize(IST)
    else:
        ts = ts.dt.tz_convert(IST)
    out = pd.DataFrame({
        "date": ts.dt.date,
        "regime": timeline["regime"].astype(str).values,
    })
    out = out.drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
    return out


def join_features_labels_regimes(symbol: str = "NIFTY") -> pd.DataFrame:
    """Join features (wide), labels (ret_*) and the daily regime label.

    Returns
    -------
    pd.DataFrame
        Columns: timestamp, regime, <feature columns>, ret_15m, ret_30m,
        ret_60m, ret_eod (plus other label cols if present). Rows missing
        a regime label are DROPPED (we never fabricate a regime).

    Notes
    -----
    Regime is daily; intraday timestamps inherit the regime of their IST
    trading date. If the regime timeline does not cover a timestamp's
    day, that row is dropped -- no nearest-prior fallback here; that
    behavior is reserved for `regime_for_timestamp`.
    """
    labeled = _load_labeled_wide(symbol)
    if labeled.empty:
        return labeled

    ts = pd.to_datetime(labeled["timestamp"])
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize(IST)
    else:
        ts = ts.dt.tz_convert(IST)
    labeled = labeled.copy()
    labeled["timestamp"] = ts
    labeled["_join_date"] = ts.dt.date

    regimes = _load_regime_timeline_for_join(symbol)
    if regimes.empty:
        # No regime info at all -- return empty frame with regime column
        out = labeled.drop(columns=["_join_date"])
        out["regime"] = pd.Series([], dtype=object)
        return out.iloc[0:0]

    merged = labeled.merge(
        regimes.rename(columns={"date": "_join_date"}),
        on="_join_date",
        how="left",
    )
    merged = merged.dropna(subset=["regime"]).reset_index(drop=True)
    merged = merged.drop(columns=["_join_date"])

    # Reorder: timestamp, regime, then everything else
    other = [c for c in merged.columns if c not in ("timestamp", "regime")]
    return merged[["timestamp", "regime"] + other]


# ---------------------------------------------------------------------------
# Per-feature conditional analysis
# ---------------------------------------------------------------------------


def _score_to_result(
    score,
    regime: str,
    delta_ic_vs_overall: float,
    is_regime_specific: bool,
) -> ConditionalEdgeResult:
    return ConditionalEdgeResult(
        feature=score.feature,
        horizon=score.horizon,
        regime=regime,
        n=int(score.n),
        ic=float(score.ic),
        pearson=float(score.pearson),
        mutual_info=float(score.mutual_info),
        decile_spread=float(score.decile_spread),
        composite_score=float(score.composite_score),
        delta_ic_vs_overall=float(delta_ic_vs_overall),
        is_regime_specific=bool(is_regime_specific),
        confidence_band=str(score.confidence_band),
    )


def _empty_result(feature: str, horizon: str, regime: str) -> ConditionalEdgeResult:
    """Result row for a regime with zero samples."""
    return ConditionalEdgeResult(
        feature=feature,
        horizon=horizon,
        regime=regime,
        n=0,
        ic=float("nan"),
        pearson=float("nan"),
        mutual_info=float("nan"),
        decile_spread=float("nan"),
        composite_score=float("nan"),
        delta_ic_vs_overall=float("nan"),
        is_regime_specific=False,
        confidence_band="INSUFFICIENT",
    )


def analyze_conditional(
    labeled_df: pd.DataFrame,
    feature: str,
    horizon: str,
    min_samples_per_regime: int = 30,
    delta_ic_threshold: float = DEFAULT_DELTA_IC_THRESHOLD,
) -> list[ConditionalEdgeResult]:
    """Compute one ConditionalEdgeResult per regime plus an OVERALL baseline.

    Always returns 5 results: OVERALL + the four canonical regimes (any
    regime with no samples gets an empty INSUFFICIENT row).
    """
    results: list[ConditionalEdgeResult] = []

    if feature not in labeled_df.columns or horizon not in labeled_df.columns:
        # Feature/horizon missing -- emit empty rows for completeness
        for r in (OVERALL_LABEL,) + REGIMES:
            results.append(_empty_result(feature, horizon, r))
        return results

    # OVERALL baseline
    overall_score = analyze_feature(
        feature_values=labeled_df[feature],
        label_values=labeled_df[horizon],
        feature_name=feature,
        horizon=horizon,
        min_samples=min_samples_per_regime,
    )
    overall_ic = float(overall_score.ic) if np.isfinite(overall_score.ic) else float("nan")
    results.append(_score_to_result(
        overall_score,
        regime=OVERALL_LABEL,
        delta_ic_vs_overall=0.0,
        is_regime_specific=False,
    ))

    # Per-regime
    for regime in REGIMES:
        sub = labeled_df[labeled_df["regime"] == regime]
        if sub.empty:
            results.append(_empty_result(feature, horizon, regime))
            continue
        score = analyze_feature(
            feature_values=sub[feature],
            label_values=sub[horizon],
            feature_name=feature,
            horizon=horizon,
            min_samples=min_samples_per_regime,
        )
        if np.isfinite(score.ic) and np.isfinite(overall_ic):
            delta = float(score.ic) - overall_ic
        else:
            delta = float("nan")
        is_specific = bool(
            np.isfinite(delta)
            and abs(delta) > delta_ic_threshold
            and score.n >= min_samples_per_regime
        )
        results.append(_score_to_result(
            score,
            regime=regime,
            delta_ic_vs_overall=delta,
            is_regime_specific=is_specific,
        ))

    return results


# ---------------------------------------------------------------------------
# Full-dataset analysis
# ---------------------------------------------------------------------------


def analyze_dataset_conditional(
    labeled_df: pd.DataFrame,
    feature_columns: list[str] | None = None,
    horizons: Sequence[str] = DEFAULT_HORIZONS,
    min_samples_per_regime: int = 30,
) -> pd.DataFrame:
    """Run `analyze_conditional` for every (feature, horizon). Long-format DF."""
    if labeled_df.empty or "regime" not in labeled_df.columns:
        return pd.DataFrame()

    if feature_columns is None:
        # Exclude 'regime' and the standard skipped columns from auto-detect
        feature_columns = [c for c in _auto_detect_features(labeled_df)
                           if c != "regime"]

    rows: list[dict] = []
    for feat in feature_columns:
        if feat not in labeled_df.columns:
            continue
        for h in horizons:
            if h not in labeled_df.columns:
                continue
            for res in analyze_conditional(
                labeled_df, feature=feat, horizon=h,
                min_samples_per_regime=min_samples_per_regime,
            ):
                rows.append(res.to_row())

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    head = ["feature", "horizon", "regime", "n", "ic", "pearson",
            "mutual_info", "decile_spread", "composite_score",
            "delta_ic_vs_overall", "is_regime_specific", "confidence_band"]
    tail = [c for c in df.columns if c not in head]
    return df[[c for c in head if c in df.columns] + tail]


def rank_regime_specific(
    scores_df: pd.DataFrame,
    horizon: str = "ret_60m",
    top_k: int = 20,
) -> pd.DataFrame:
    """Filter to regime-specific rows for `horizon`, sort by |delta_ic| desc."""
    if scores_df.empty:
        return scores_df.copy()
    sub = scores_df[
        (scores_df["horizon"] == horizon)
        & (scores_df["is_regime_specific"] == True)  # noqa: E712
        & (scores_df["regime"] != OVERALL_LABEL)
    ].copy()
    if sub.empty:
        return sub
    sub["_abs_delta"] = sub["delta_ic_vs_overall"].abs()
    sub = sub.sort_values("_abs_delta", ascending=False, na_position="last")
    sub = sub.drop(columns=["_abs_delta"])
    return sub.head(top_k).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def _format_md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_(no rows)_\n"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                cells.append("nan" if not np.isfinite(v) else f"{v:.4f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _write_markdown_report(
    symbol: str,
    scores: pd.DataFrame,
    labeled_df: pd.DataFrame,
    md_path: Path,
    delta_ic_threshold: float,
) -> None:
    md_path.parent.mkdir(parents=True, exist_ok=True)

    if not labeled_df.empty and "timestamp" in labeled_df.columns:
        ts = pd.to_datetime(labeled_df["timestamp"])
        coverage = f"{ts.min()} to {ts.max()}"
    else:
        coverage = "n/a"

    # Sample size by regime (unique timestamps)
    regime_counts: dict[str, int] = {r: 0 for r in REGIMES}
    if not labeled_df.empty and "regime" in labeled_df.columns:
        unique_ts = labeled_df.drop_duplicates(subset=["timestamp"])
        vc = unique_ts["regime"].value_counts().to_dict()
        for r in REGIMES:
            regime_counts[r] = int(vc.get(r, 0))

    lines: list[str] = []
    lines.append(f"# Conditional Edge Discovery -- {symbol}")
    lines.append(f"Generated: {datetime.now(tz=IST).isoformat(timespec='seconds')}")
    lines.append(f"Coverage: {coverage}")
    lines.append("")

    lines.append("## Sample size by regime (unique timestamps)")
    for r in REGIMES:
        lines.append(f"{r}: {regime_counts[r]}")
    lines.append("")

    lines.append(f"## Top 10 regime-specific edges "
                 f"(|delta_ic| > {delta_ic_threshold}, sorted desc)")
    top = rank_regime_specific(scores, horizon="ret_60m", top_k=10)
    if top.empty:
        lines.append("_(no regime-specific edges met the threshold at this sample size)_")
    else:
        # Build overall-IC lookup for context
        overall = scores[scores["regime"] == OVERALL_LABEL][
            ["feature", "horizon", "ic"]
        ].rename(columns={"ic": "ic_overall"})
        view = top.merge(overall, on=["feature", "horizon"], how="left")
        view = view[[
            "feature", "horizon", "regime", "ic", "ic_overall",
            "delta_ic_vs_overall", "n", "confidence_band",
        ]].rename(columns={
            "feature": "Feature",
            "horizon": "Horizon",
            "regime": "Regime",
            "ic": "IC (regime)",
            "ic_overall": "IC (overall)",
            "delta_ic_vs_overall": "Delta IC",
            "n": "n",
            "confidence_band": "Confidence",
        })
        lines.append(_format_md_table(view))
    lines.append("")

    lines.append("## Caveats")
    lines.append("- Regime labels are computed from daily ATR/ADX/realized vol; "
                 "intraday timestamps inherit the day's regime.")
    lines.append("- Current sample sizes per regime are tiny "
                 "(insufficient for statistical confidence).")
    lines.append("- All findings are preliminary -- interpret with confidence bands.")
    lines.append("")

    # Honest dataset-size note
    total_unique = sum(regime_counts.values())
    populated = sum(1 for v in regime_counts.values() if v > 0)
    if populated <= 1:
        lines.append("## Honest dataset note")
        lines.append(f"Only {populated} regime is populated across "
                     f"{total_unique} unique timestamp(s). The OVERALL row and "
                     f"that single regime's row will therefore be identical, "
                     f"and every conditional comparison lands INSUFFICIENT. "
                     f"This is the correct answer at this data scale.")
        lines.append("")

    md_path.write_text("\n".join(lines), encoding="ascii", errors="replace")


# ---------------------------------------------------------------------------
# Public pipeline
# ---------------------------------------------------------------------------


def conditional_edge_report(
    symbol: str = "NIFTY",
    out_path: Path | None = None,
    horizons: Sequence[str] = DEFAULT_HORIZONS,
    min_samples_per_regime: int = 30,
    delta_ic_threshold: float = DEFAULT_DELTA_IC_THRESHOLD,
) -> Path:
    """End-to-end pipeline.

    1. Join features + labels + regime timeline.
    2. Run `analyze_dataset_conditional`.
    3. Persist scores.parquet and scores.md.
    4. Return parquet path.
    """
    labeled = join_features_labels_regimes(symbol)
    scores = analyze_dataset_conditional(
        labeled,
        horizons=horizons,
        min_samples_per_regime=min_samples_per_regime,
    )

    if out_path is None:
        out_dir = COND_EDGE_ROOT / symbol
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "scores.parquet"
    else:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

    if not scores.empty:
        scores.to_parquet(out_path, index=False)
    else:
        pd.DataFrame().to_parquet(out_path, index=False)

    md_path = out_path.with_suffix(".md")
    _write_markdown_report(symbol, scores, labeled, md_path, delta_ic_threshold)
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cmd_run(args: argparse.Namespace) -> None:
    p = conditional_edge_report(
        symbol=args.symbol,
        min_samples_per_regime=args.min_samples,
    )
    df = pd.read_parquet(p)
    print(f"Wrote {p} ({len(df)} rows)")
    print(f"Markdown: {p.with_suffix('.md')}")
    if not df.empty:
        n_specific = int((df["is_regime_specific"] == True).sum())  # noqa: E712
        print(f"Regime-specific edges found: {n_specific}")


def _cmd_top(args: argparse.Namespace) -> None:
    path = COND_EDGE_ROOT / args.symbol / "scores.parquet"
    if not path.exists():
        print(f"No scores at {path}. Run `run` first.", file=sys.stderr)
        sys.exit(1)
    df = pd.read_parquet(path)
    top = rank_regime_specific(df, horizon=args.horizon, top_k=args.k)
    if top.empty:
        print(f"No regime-specific edges for horizon {args.horizon}.")
        return
    view = top[[
        "feature", "regime", "ic", "delta_ic_vs_overall",
        "composite_score", "n", "confidence_band",
    ]]
    with pd.option_context("display.max_rows", None,
                           "display.width", 160,
                           "display.float_format", "{:.4f}".format):
        print(view.to_string(index=False))


def _cmd_show(args: argparse.Namespace) -> None:
    path = COND_EDGE_ROOT / args.symbol / "scores.parquet"
    if not path.exists():
        print(f"No scores at {path}. Run `run` first.", file=sys.stderr)
        sys.exit(1)
    df = pd.read_parquet(path)
    sub = df[df["feature"] == args.feature]
    if sub.empty:
        print(f"No rows for feature '{args.feature}'.", file=sys.stderr)
        sys.exit(1)
    with pd.option_context("display.max_rows", None,
                           "display.width", 200,
                           "display.float_format", "{:.4f}".format):
        print(sub.to_string(index=False))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="conditional_edge")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run", help="Build conditional-edge scores.parquet + scores.md")
    pr.add_argument("--symbol", default="NIFTY")
    pr.add_argument("--min-samples", type=int, default=DEFAULT_MIN_SAMPLES,
                    dest="min_samples")
    pr.set_defaults(func=_cmd_run)

    pt = sub.add_parser("top", help="Show top-k regime-specific edges")
    pt.add_argument("--symbol", default="NIFTY")
    pt.add_argument("--horizon", default="ret_60m")
    pt.add_argument("--k", type=int, default=10)
    pt.set_defaults(func=_cmd_top)

    ps = sub.add_parser("show", help="Show all conditional rows for a feature")
    ps.add_argument("--symbol", default="NIFTY")
    ps.add_argument("--feature", required=True)
    ps.set_defaults(func=_cmd_show)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
