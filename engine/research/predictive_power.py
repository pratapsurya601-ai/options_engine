"""
Predictive-power analysis of option-flow features.

For each (feature, forward-return-horizon) pair, compute a battery of
statistical metrics that answer the question: *does this feature contain
real, exploitable signal for that future return?* The output is intentionally
multi-metric and conservative -- we do not want one favourable number to
mask weakness elsewhere.

Scoring philosophy
------------------
We blend four metrics into a single non-negative composite score:

    composite_score = 0.35 * |IC|
                    + 0.25 * |Pearson|
                    + 0.20 * min(MI / 0.1, 1.0)
                    + 0.20 * min(|decile_spread| / 0.002, 1.0)

The composite is direction-agnostic on purpose: a feature that is strongly
*negatively* correlated with future returns is just as tradable as a
positively correlated one. The caller can inspect the *sign* of `pearson`
or `ic` to know direction.

Confidence bands (deliberately conservative)
--------------------------------------------
    INSUFFICIENT : n < min_samples (default 30)
    LOW          : n < 200 OR pearson_p >= 0.20
    MEDIUM       : 200 <= n < 1000 AND pearson_p < 0.20
    HIGH         : n >= 1000 AND pearson_p < 0.05 AND |IC| > 0.05

Small samples never earn HIGH confidence, no matter how good the score looks.
This codebase's current intraday snapshot data (~17 timestamps/day) will
land in INSUFFICIENT/LOW for every metric -- that is the correct answer.
Do not engineer the score to flatter small samples.

Statistical caveats
-------------------
- Pearson assumes (locally) linear relationships. IC/Spearman is more robust.
- Mutual information uses sklearn's k-NN estimator -- noisy at small N.
- Decile analysis requires enough samples per bucket; below ~100 rows the
  spread estimate is essentially noise. The composite score absorbs this
  by capping the contribution at |spread| >= 20 bps.
- Multiple-testing: we evaluate many features x many horizons. A naive 5%
  significance call will *expect* to fire on ~5% of (feature, horizon)
  cells by pure chance. The `is_significant_5pct` flag is single-test, not
  family-wise -- treat it as a screening hint, not a hypothesis test.
- IC and Spearman are computed with `nan_policy='omit'` only AFTER the
  caller-side NaN drop; sklearn MI cannot tolerate NaNs so we drop again.

CLI
---
    python -m engine.research.predictive_power run --symbol NIFTY
    python -m engine.research.predictive_power top --symbol NIFTY --horizon ret_60m --k 10
    python -m engine.research.predictive_power show --symbol NIFTY --feature pcr_oi

Storage
-------
    data/research/predictive_power/{symbol}/scores.parquet   (long-format)
    data/research/predictive_power/{symbol}/scores.md        (human report)
"""
from __future__ import annotations

import argparse
import math
import sys
import warnings
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import roc_auc_score


IST = timezone(timedelta(hours=5, minutes=30))
PREDPOW_ROOT = Path("data/research/predictive_power")
DEFAULT_HORIZONS: tuple[str, ...] = ("ret_15m", "ret_30m", "ret_60m", "ret_eod")
DEFAULT_MIN_SAMPLES = 30

# Composite-score normalization constants (see module docstring).
_MI_CAP = 0.1
_DECILE_SPREAD_CAP = 0.002  # 20 bps


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------


@dataclass
class FeaturePredictiveScore:
    """Predictive-power summary for a single (feature, horizon) pair."""

    feature: str
    horizon: str
    n: int
    nan_pct: float
    pearson: float
    pearson_p: float
    pearson_ci: tuple[float, float]
    spearman: float
    ic: float
    mutual_info: float
    decile_spread: float
    decile_t_stat: float
    decile_means: list[float] = field(default_factory=list)
    anova_f: float | None = None
    auc_up_vs_down: float | None = None
    is_significant_5pct: bool = False
    composite_score: float = float("nan")
    confidence_band: str = "INSUFFICIENT"

    def to_row(self) -> dict:
        """Flatten to a dict suitable for a DataFrame row."""
        d = asdict(self)
        ci = d.pop("pearson_ci")
        d["pearson_ci_low"] = float(ci[0]) if ci is not None else float("nan")
        d["pearson_ci_high"] = float(ci[1]) if ci is not None else float("nan")
        # Decile means become JSON-ish string for parquet friendliness
        means = d.pop("decile_means")
        d["decile_means"] = ",".join(
            "nan" if (m is None or (isinstance(m, float) and math.isnan(m)))
            else f"{m:.6g}" for m in means
        )
        return d


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _drop_nan_pair(x: pd.Series, y: pd.Series) -> tuple[np.ndarray, np.ndarray, float]:
    """Align, drop NaNs, return arrays and nan_pct (fraction dropped)."""
    df = pd.concat([x.rename("x"), y.rename("y")], axis=1)
    total = len(df)
    df = df.dropna()
    nan_pct = 0.0 if total == 0 else 1.0 - (len(df) / total)
    return df["x"].to_numpy(dtype="float64"), df["y"].to_numpy(dtype="float64"), nan_pct


def _fisher_ci(r: float, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Fisher z-transform 95% CI for a Pearson correlation."""
    if n < 4 or not np.isfinite(r) or abs(r) >= 1.0:
        return (float("nan"), float("nan"))
    z = math.atanh(r)
    se = 1.0 / math.sqrt(n - 3)
    z_crit = scipy_stats.norm.ppf(1.0 - alpha / 2.0)
    lo = math.tanh(z - z_crit * se)
    hi = math.tanh(z + z_crit * se)
    return (float(lo), float(hi))


def _decile_means(x: np.ndarray, y: np.ndarray, n_buckets: int = 10) -> list[float]:
    """Mean of y within each x-quantile bucket (low to high)."""
    if len(x) < n_buckets:
        return [float("nan")] * n_buckets
    # qcut with duplicates='drop' handles ties; we then re-bin into fixed slots.
    try:
        ranks = pd.qcut(pd.Series(x).rank(method="first"),
                        q=n_buckets, labels=False, duplicates="drop")
    except ValueError:
        return [float("nan")] * n_buckets
    means: list[float] = []
    for b in range(n_buckets):
        sel = (ranks.to_numpy() == b)
        if not sel.any():
            means.append(float("nan"))
        else:
            means.append(float(np.nanmean(y[sel])))
    return means


def _decile_spread_and_t(x: np.ndarray, y: np.ndarray,
                         n_buckets: int = 10) -> tuple[float, float, list[float]]:
    """Returns (top - bottom decile mean, Welch t-stat, all decile means)."""
    means = _decile_means(x, y, n_buckets=n_buckets)
    if len(means) < 2 or any(math.isnan(m) for m in (means[0], means[-1])):
        return (float("nan"), float("nan"), means)
    try:
        ranks = pd.qcut(pd.Series(x).rank(method="first"),
                        q=n_buckets, labels=False, duplicates="drop").to_numpy()
    except ValueError:
        return (float("nan"), float("nan"), means)
    bot = y[ranks == 0]
    top = y[ranks == (n_buckets - 1)]
    spread = float(np.nanmean(top) - np.nanmean(bot))
    if len(top) < 2 or len(bot) < 2:
        return (spread, float("nan"), means)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t_res = scipy_stats.ttest_ind(top, bot, equal_var=False, nan_policy="omit")
    t_stat = float(t_res.statistic) if np.isfinite(t_res.statistic) else float("nan")
    return (spread, t_stat, means)


def _safe_mutual_info(x: np.ndarray, y: np.ndarray) -> float:
    """sklearn MI for regression. Returns NaN on failure / constant input."""
    if len(x) < 5:
        return float("nan")
    if np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return float("nan")
    try:
        mi = mutual_info_regression(x.reshape(-1, 1), y, random_state=0)
        return float(mi[0])
    except Exception as e:
        warnings.warn(f"mutual_info_regression failed: {e}")
        return float("nan")


def _composite_score(ic: float, pearson: float, mi: float,
                     decile_spread: float) -> float:
    """Blend the four metrics; NaN-tolerant (missing metrics drop to 0)."""
    parts = [
        0.35 * (abs(ic) if np.isfinite(ic) else 0.0),
        0.25 * (abs(pearson) if np.isfinite(pearson) else 0.0),
        0.20 * (min(mi / _MI_CAP, 1.0) if np.isfinite(mi) and mi >= 0 else 0.0),
        0.20 * (min(abs(decile_spread) / _DECILE_SPREAD_CAP, 1.0)
                if np.isfinite(decile_spread) else 0.0),
    ]
    return float(sum(parts))


def _confidence_band(n: int, pearson_p: float, ic: float,
                     min_samples: int) -> str:
    """Conservative banding -- see module docstring."""
    if n < min_samples:
        return "INSUFFICIENT"
    p_ok_5 = np.isfinite(pearson_p) and pearson_p < 0.05
    p_ok_20 = np.isfinite(pearson_p) and pearson_p < 0.20
    ic_ok = np.isfinite(ic) and abs(ic) > 0.05
    if n >= 1000 and p_ok_5 and ic_ok:
        return "HIGH"
    if 200 <= n < 1000 and p_ok_20:
        return "MEDIUM"
    return "LOW"


def _auc_up_vs_down(feature: np.ndarray, cls: pd.Series) -> float | None:
    """AUC treating feature as a score for UP vs DOWN (drop NEUTRAL/NaN)."""
    if cls is None:
        return None
    s = pd.Series(cls).reset_index(drop=True)
    f = pd.Series(feature).reset_index(drop=True)
    # Align by length (caller has already aligned by index outside)
    if len(f) != len(s):
        return None
    mask = s.isin(["UP", "DOWN"]) & f.notna()
    if mask.sum() < 10:
        return None
    y = (s[mask] == "UP").astype(int).to_numpy()
    if y.sum() == 0 or y.sum() == len(y):
        return None  # one class only
    try:
        return float(roc_auc_score(y, f[mask].to_numpy()))
    except Exception:
        return None


def _anova_f(feature: np.ndarray, cls: pd.Series) -> float | None:
    """One-way ANOVA F over class groups (UP/DOWN/NEUTRAL)."""
    if cls is None:
        return None
    s = pd.Series(cls).reset_index(drop=True)
    f = pd.Series(feature).reset_index(drop=True)
    if len(f) != len(s):
        return None
    groups: list[np.ndarray] = []
    for label in ("UP", "DOWN", "NEUTRAL"):
        sel = (s == label) & f.notna()
        if sel.sum() >= 2:
            groups.append(f[sel].to_numpy())
    if len(groups) < 2:
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = scipy_stats.f_oneway(*groups)
        return float(res.statistic) if np.isfinite(res.statistic) else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def analyze_feature(
    feature_values: pd.Series,
    label_values: pd.Series,
    label_kind: str = "regression",
    cls_labels: pd.Series | None = None,
    feature_name: str = "feature",
    horizon: str = "ret",
    min_samples: int = DEFAULT_MIN_SAMPLES,
) -> FeaturePredictiveScore:
    """Compute the full metric battery for one (feature, label) pair.

    Parameters
    ----------
    feature_values : pd.Series
        Feature observations (one per timestamp).
    label_values : pd.Series
        Forward-return values (continuous, regression target). Required
        regardless of `label_kind` -- needed for IC/Pearson/MI.
    label_kind : str
        "regression" or "classification". For classification we additionally
        require `cls_labels` for ANOVA F / AUC.
    cls_labels : pd.Series | None
        UP/DOWN/NEUTRAL classification labels, same index as features.
    feature_name, horizon : str
        Cosmetic identifiers stored on the returned object.
    min_samples : int
        Below this, confidence_band -> INSUFFICIENT.

    Returns
    -------
    FeaturePredictiveScore

    Examples
    --------
    >>> # Strong synthetic signal
    >>> x = pd.Series(np.linspace(-1, 1, 500))
    >>> y = x + 0.1 * np.random.randn(500)
    >>> s = analyze_feature(x, y)
    >>> s.composite_score > 0.5
    True
    """
    feat_idx_aligned = feature_values
    label_idx_aligned = label_values
    x_arr, y_arr, nan_pct = _drop_nan_pair(feat_idx_aligned, label_idx_aligned)
    n = int(len(x_arr))

    # Constant-feature guard
    constant = (n == 0) or float(np.nanstd(x_arr)) == 0.0

    if constant or n < 2:
        if constant and n >= 1:
            warnings.warn(
                f"feature '{feature_name}' is constant or empty -- metrics NaN"
            )
        return FeaturePredictiveScore(
            feature=feature_name, horizon=horizon, n=n, nan_pct=nan_pct,
            pearson=float("nan"), pearson_p=float("nan"),
            pearson_ci=(float("nan"), float("nan")),
            spearman=float("nan"), ic=float("nan"),
            mutual_info=float("nan"),
            decile_spread=float("nan"), decile_t_stat=float("nan"),
            decile_means=[float("nan")] * 10,
            anova_f=None, auc_up_vs_down=None,
            is_significant_5pct=False,
            composite_score=float("nan"),
            confidence_band="INSUFFICIENT",
        )

    # Pearson + CI + p-value
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pr = scipy_stats.pearsonr(x_arr, y_arr)
        pearson = float(pr.statistic) if hasattr(pr, "statistic") else float(pr[0])
        pearson_p = float(pr.pvalue) if hasattr(pr, "pvalue") else float(pr[1])
    except Exception:
        pearson, pearson_p = float("nan"), float("nan")
    pearson_ci = _fisher_ci(pearson, n)

    # Spearman / IC
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sr = scipy_stats.spearmanr(x_arr, y_arr)
        spearman = float(sr.statistic) if hasattr(sr, "statistic") else float(sr[0])
    except Exception:
        spearman = float("nan")
    ic = spearman

    # MI
    mi = _safe_mutual_info(x_arr, y_arr)

    # Decile analysis
    decile_spread, decile_t, decile_means = _decile_spread_and_t(x_arr, y_arr)

    # Classification metrics (need cls_labels aligned to ORIGINAL feature index)
    anova_f_val: float | None = None
    auc_val: float | None = None
    if cls_labels is not None:
        # Align cls_labels with feature_values index (drop where either NaN)
        df_c = pd.concat(
            [feature_values.rename("f"),
             label_values.rename("y"),
             pd.Series(cls_labels).rename("c")],
            axis=1,
        ).dropna(subset=["f"])  # cls can have NaN; handled internally
        f_c = df_c["f"].to_numpy(dtype="float64")
        c_c = df_c["c"]
        anova_f_val = _anova_f(f_c, c_c)
        auc_val = _auc_up_vs_down(f_c, c_c)

    composite = _composite_score(ic, pearson, mi, decile_spread)
    band = _confidence_band(n, pearson_p, ic, min_samples)
    significant = bool(np.isfinite(pearson_p) and pearson_p < 0.05)

    return FeaturePredictiveScore(
        feature=feature_name, horizon=horizon, n=n, nan_pct=float(nan_pct),
        pearson=pearson, pearson_p=pearson_p, pearson_ci=pearson_ci,
        spearman=spearman, ic=ic, mutual_info=mi,
        decile_spread=decile_spread, decile_t_stat=decile_t,
        decile_means=decile_means,
        anova_f=anova_f_val, auc_up_vs_down=auc_val,
        is_significant_5pct=significant,
        composite_score=composite,
        confidence_band=band,
    )


def _auto_detect_features(labeled_df: pd.DataFrame) -> list[str]:
    """Columns that are neither labels nor id columns."""
    skip = {"timestamp", "symbol", "expiry"}
    out: list[str] = []
    for c in labeled_df.columns:
        if c in skip:
            continue
        if c.startswith("ret_") or c.startswith("cls_"):
            continue
        if pd.api.types.is_numeric_dtype(labeled_df[c]):
            out.append(c)
    return out


def analyze_dataset(
    labeled_df: pd.DataFrame,
    feature_columns: list[str] | None = None,
    horizons: Sequence[str] = DEFAULT_HORIZONS,
    min_samples: int = DEFAULT_MIN_SAMPLES,
) -> pd.DataFrame:
    """Run analyze_feature for every (feature, horizon) and return long DF.

    Parameters
    ----------
    labeled_df : wide-format DataFrame (timestamp + features + ret_*/cls_*).
    feature_columns : optional explicit list; otherwise auto-detected.
    horizons : forward-return labels to evaluate.
    min_samples : threshold below which a row is tagged INSUFFICIENT.

    Returns
    -------
    pd.DataFrame with one row per (feature, horizon).
    """
    if feature_columns is None:
        feature_columns = _auto_detect_features(labeled_df)

    rows: list[dict] = []
    for feat in feature_columns:
        if feat not in labeled_df.columns:
            warnings.warn(f"feature '{feat}' not in dataframe; skipped")
            continue
        for h in horizons:
            if h not in labeled_df.columns:
                continue
            cls_col = "cls_" + h[len("ret_"):] if h.startswith("ret_") else None
            cls = labeled_df[cls_col] if (cls_col and cls_col in labeled_df.columns) else None
            score = analyze_feature(
                feature_values=labeled_df[feat],
                label_values=labeled_df[h],
                cls_labels=cls,
                feature_name=feat,
                horizon=h,
                min_samples=min_samples,
            )
            rows.append(score.to_row())

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # Reorder a couple of key columns up front for readability
    head = ["feature", "horizon", "n", "nan_pct", "composite_score",
            "confidence_band", "ic", "pearson", "pearson_p", "mutual_info",
            "decile_spread", "decile_t_stat", "anova_f", "auc_up_vs_down",
            "is_significant_5pct"]
    tail = [c for c in df.columns if c not in head]
    return df[[c for c in head if c in df.columns] + tail]


def rank_features(
    scores: pd.DataFrame,
    horizon: str = "ret_60m",
    top_k: int = 20,
) -> pd.DataFrame:
    """Return the top-K rows of `scores` for `horizon`, sorted by composite."""
    sub = scores[scores["horizon"] == horizon].copy()
    sub = sub.sort_values("composite_score", ascending=False, na_position="last")
    return sub.head(top_k).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Pipeline / report
# ---------------------------------------------------------------------------


def _ensure_labeled_dataset(symbol: str) -> pd.DataFrame:
    """Try load_labeled_dataset; if absent, build it from flow_features+labels."""
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
    # load_labeled_dataset only pivots when columns are named exactly
    # ('feature_name', 'value'). Our upstream uses 'feature_value', so the
    # frame may still be long-form. Pivot defensively here.
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


def _format_md_table(df: pd.DataFrame) -> str:
    """Render a small markdown table from a DataFrame of fixed columns."""
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


def _write_markdown_report(symbol: str, scores: pd.DataFrame,
                           labeled_df: pd.DataFrame, md_path: Path,
                           horizons: Sequence[str]) -> None:
    md_path.parent.mkdir(parents=True, exist_ok=True)

    if not labeled_df.empty and "timestamp" in labeled_df.columns:
        ts = pd.to_datetime(labeled_df["timestamp"])
        coverage = f"{ts.min()} to {ts.max()}"
        n_timestamps = int(ts.nunique())
    else:
        coverage = "n/a"
        n_timestamps = 0
    n_features = int(scores["feature"].nunique()) if not scores.empty else 0

    lines: list[str] = []
    lines.append(f"# Predictive Power Analysis -- {symbol}")
    lines.append(f"Generated: {datetime.now(tz=IST).isoformat(timespec='seconds')}")
    lines.append(f"Data coverage: {coverage}")
    lines.append(f"Total samples: {n_features} features x {n_timestamps} timestamps")
    lines.append("")

    # NaN rate per horizon (median across features)
    lines.append("## NaN rate per horizon")
    if not scores.empty:
        nan_summary = scores.groupby("horizon")["nan_pct"].median().round(4)
        for h, v in nan_summary.items():
            lines.append(f"- {h}: median nan_pct = {v}")
    lines.append("")

    for h in horizons:
        lines.append(f"## Top 10 features by composite score (horizon = {h})")
        top = rank_features(scores, horizon=h, top_k=10)
        if top.empty:
            lines.append("_(no rows for this horizon)_")
            lines.append("")
            continue
        # Keep a tight, readable subset of columns
        view = top[[
            "feature", "composite_score", "ic", "pearson", "mutual_info",
            "decile_spread", "n", "confidence_band"
        ]].copy()
        view.insert(0, "Rank", np.arange(1, len(view) + 1))
        view = view.rename(columns={
            "composite_score": "Composite", "ic": "IC", "pearson": "Pearson",
            "mutual_info": "MI", "decile_spread": "DecileSpread",
            "n": "n", "confidence_band": "Confidence", "feature": "Feature",
        })
        lines.append(_format_md_table(view))
        lines.append("")

    lines.append("## Sample size warning")
    lines.append(f"N timestamps available: {n_timestamps}")
    lines.append("Statistical robustness threshold: n >= 1000 per horizon for HIGH confidence.")
    if n_timestamps < 1000:
        lines.append("Current dataset is INSUFFICIENT for HIGH confidence -- "
                     "interpret all results as preliminary.")
    lines.append("")

    lines.append("## Methodology")
    lines.append("Composite score = 0.35*|IC| + 0.25*|Pearson| + "
                 "0.20*min(MI/0.1, 1.0) + 0.20*min(|spread|/0.002, 1.0). "
                 "Direction-agnostic; check sign of IC/Pearson for direction. "
                 "Confidence bands are deliberately conservative: HIGH requires "
                 "n>=1000, p<0.05, |IC|>0.05; MEDIUM requires n in [200, 1000) "
                 "and p<0.20; everything else is LOW or INSUFFICIENT.")

    md_path.write_text("\n".join(lines), encoding="ascii", errors="replace")


def predictive_power_report(
    symbol: str = "NIFTY",
    out_path: Path | None = None,
    horizons: Sequence[str] = DEFAULT_HORIZONS,
    min_samples: int = DEFAULT_MIN_SAMPLES,
) -> Path:
    """End-to-end pipeline.

    1. Load (or rebuild) the labeled wide-form dataset for `symbol`.
    2. Run `analyze_dataset` over every feature x horizon.
    3. Persist scores.parquet and scores.md alongside it.
    4. Return the parquet path.
    """
    labeled = _ensure_labeled_dataset(symbol)
    scores = analyze_dataset(labeled, horizons=horizons, min_samples=min_samples)

    if out_path is None:
        out_dir = PREDPOW_ROOT / symbol
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "scores.parquet"
    else:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

    # Parquet does not love object columns with mixed types; ensure strings.
    if not scores.empty:
        scores_to_save = scores.copy()
        if "decile_means" in scores_to_save.columns:
            scores_to_save["decile_means"] = scores_to_save["decile_means"].astype(str)
        scores_to_save.to_parquet(out_path, index=False)
    else:
        pd.DataFrame().to_parquet(out_path, index=False)

    md_path = out_path.with_suffix(".md")
    _write_markdown_report(symbol, scores, labeled, md_path, horizons=horizons)
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cmd_run(args):
    p = predictive_power_report(symbol=args.symbol)
    df = pd.read_parquet(p)
    print(f"Wrote {p} ({len(df)} rows)")
    print(f"Markdown: {p.with_suffix('.md')}")


def _cmd_top(args):
    path = PREDPOW_ROOT / args.symbol / "scores.parquet"
    if not path.exists():
        print(f"No scores at {path}. Run `run` first.", file=sys.stderr)
        sys.exit(1)
    df = pd.read_parquet(path)
    top = rank_features(df, horizon=args.horizon, top_k=args.k)
    view = top[["feature", "composite_score", "ic", "pearson", "mutual_info",
                "decile_spread", "n", "confidence_band"]]
    with pd.option_context("display.max_rows", None,
                           "display.width", 160,
                           "display.float_format", "{:.4f}".format):
        print(view.to_string(index=False))


def _cmd_show(args):
    path = PREDPOW_ROOT / args.symbol / "scores.parquet"
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
    p = argparse.ArgumentParser(prog="predictive_power")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run", help="Build scores.parquet + scores.md")
    pr.add_argument("--symbol", default="NIFTY")
    pr.set_defaults(func=_cmd_run)

    pt = sub.add_parser("top", help="Show top-k features for a horizon")
    pt.add_argument("--symbol", default="NIFTY")
    pt.add_argument("--horizon", default="ret_60m")
    pt.add_argument("--k", type=int, default=10)
    pt.set_defaults(func=_cmd_top)

    ps = sub.add_parser("show", help="Show all rows for a feature")
    ps.add_argument("--symbol", default="NIFTY")
    ps.add_argument("--feature", required=True)
    ps.set_defaults(func=_cmd_show)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
