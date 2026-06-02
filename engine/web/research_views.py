"""
Research dashboard views.

Exposes a Flask blueprint with:
    GET  /research                    -> server-rendered HTML dashboard
    GET  /api/research/summary        -> JSON summary of all research outputs
    GET  /api/research/feature/<name> -> JSON deep-dive for one feature
    POST /api/research/rebuild        -> re-run the full research pipeline

Reads (read-only) the parquet artifacts written by engine.research.*:
    data/research/flow_features/<sym>/all.parquet      (long format)
    data/research/labels/<sym>/all.parquet
    data/research/regimes/<sym>/timeline_daily.parquet
    data/research/predictive_power/<sym>/scores.parquet
    data/research/walkforward/<sym>/summaries.parquet
    data/research/conditional_edge/<sym>/scores.parquet

Designed to fail gracefully: missing parquets render as "not yet built"
banners rather than raising.
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from flask import Blueprint, Response, jsonify, request


IST = timezone(timedelta(hours=5, minutes=30))
RESEARCH_ROOT = Path("data/research")
REGIMES = ("TREND", "MEAN_REVERSION", "PANIC", "LOW_VOL")
HORIZONS = ("ret_15m", "ret_30m", "ret_60m", "ret_eod")
MIN_CONFIDENT_SAMPLES = 1000

research_bp = Blueprint("research", __name__)


# ---------------------------------------------------------------------------
# mtime-keyed parquet cache (no stale data, no per-request recompute)
# ---------------------------------------------------------------------------

_CACHE: dict[str, tuple[float, Any]] = {}


def _cache_get(path: Path) -> pd.DataFrame | None:
    """Return cached parquet DataFrame keyed on mtime, or None if missing."""
    if not path.exists():
        return None
    key = str(path.resolve())
    mtime = path.stat().st_mtime
    cached = _CACHE.get(key)
    if cached and cached[0] == mtime:
        return cached[1]
    try:
        df = pd.read_parquet(path)
    except Exception:
        return None
    _CACHE[key] = (mtime, df)
    return df


def _path(*parts: str) -> Path:
    return RESEARCH_ROOT.joinpath(*parts)


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------


def _load_flow(symbol: str) -> pd.DataFrame | None:
    return _cache_get(_path("flow_features", symbol, "all.parquet"))


def _load_predpow(symbol: str) -> pd.DataFrame | None:
    return _cache_get(_path("predictive_power", symbol, "scores.parquet"))


def _load_walkfwd(symbol: str) -> pd.DataFrame | None:
    return _cache_get(_path("walkforward", symbol, "summaries.parquet"))


def _load_conditional(symbol: str) -> pd.DataFrame | None:
    return _cache_get(_path("conditional_edge", symbol, "scores.parquet"))


def _load_regimes(symbol: str) -> pd.DataFrame | None:
    return _cache_get(_path("regimes", symbol, "timeline_daily.parquet"))


def _load_labels(symbol: str) -> pd.DataFrame | None:
    return _cache_get(_path("labels", symbol, "all.parquet"))


def _automated_rebuild_status() -> dict | None:
    """Read data/research/rebuild_status.json if present.

    Returns the parsed dict (with a derived 'failed_steps' list) or None
    if the file is missing or unreadable. Used by the dashboard header
    to surface the result of the latest scheduled daemon run.
    """
    path = RESEARCH_ROOT / "rebuild_status.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    failed = [s.get("name") for s in data.get("steps", [])
              if s.get("status") == "FAILED"]
    data["failed_steps"] = failed
    return data


def _last_rebuild_ts(symbol: str) -> str | None:
    """Latest mtime across all research artifacts, formatted in IST."""
    paths = [
        _path("flow_features", symbol, "all.parquet"),
        _path("labels", symbol, "all.parquet"),
        _path("regimes", symbol, "timeline_daily.parquet"),
        _path("predictive_power", symbol, "scores.parquet"),
        _path("walkforward", symbol, "summaries.parquet"),
        _path("conditional_edge", symbol, "scores.parquet"),
    ]
    mtimes = [p.stat().st_mtime for p in paths if p.exists()]
    if not mtimes:
        return None
    dt = datetime.fromtimestamp(max(mtimes), tz=IST)
    return dt.strftime("%Y-%m-%d %H:%M IST")


# ---------------------------------------------------------------------------
# Summary builders
# ---------------------------------------------------------------------------


def _safe_float(v: Any) -> float | None:
    try:
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except (TypeError, ValueError):
        return None


def _coverage(flow: pd.DataFrame | None) -> dict:
    if flow is None or flow.empty:
        return {"start": None, "end": None, "n_snapshots": 0, "n_features": 0}
    ts = pd.to_datetime(flow["timestamp"])
    return {
        "start": ts.min().date().isoformat() if not ts.empty else None,
        "end": ts.max().date().isoformat() if not ts.empty else None,
        "n_snapshots": int(ts.nunique()),
        "n_features": int(flow["feature_name"].nunique()),
    }


def _regime_breakdown(regimes: pd.DataFrame | None) -> dict[str, int]:
    if regimes is None or regimes.empty or "regime" not in regimes.columns:
        return {r: 0 for r in REGIMES}
    counts = regimes["regime"].value_counts().to_dict()
    return {r: int(counts.get(r, 0)) for r in REGIMES}


def _current_regime(regimes: pd.DataFrame | None) -> dict | None:
    if regimes is None or regimes.empty:
        return None
    last = regimes.sort_values("timestamp").iloc[-1]
    return {
        "regime": str(last.get("regime")),
        "timestamp": str(last.get("timestamp")),
        "atr_pct": _safe_float(last.get("atr_pct")),
        "realized_vol_5d": _safe_float(last.get("realized_vol_5d")),
        "iv_rank": _safe_float(last.get("iv_rank")),
        "ema_slope_20": _safe_float(last.get("ema_slope_20")),
        "adx_14": _safe_float(last.get("adx_14")),
    }


def _last_n_regimes(regimes: pd.DataFrame | None, n: int = 10) -> list[dict]:
    if regimes is None or regimes.empty:
        return []
    tail = regimes.sort_values("timestamp").tail(n)
    out = []
    for _, row in tail.iterrows():
        ts = row.get("timestamp")
        out.append({
            "date": str(pd.to_datetime(ts).date()) if ts is not None else None,
            "regime": str(row.get("regime")),
        })
    return out


def _row_to_score_dict(row: pd.Series) -> dict:
    return {
        "feature": str(row.get("feature")),
        "horizon": str(row.get("horizon")),
        "n": int(row.get("n", 0) or 0),
        "ic": _safe_float(row.get("ic")),
        "pearson": _safe_float(row.get("pearson")),
        "mutual_info": _safe_float(row.get("mutual_info")),
        "decile_spread": _safe_float(row.get("decile_spread")),
        "composite_score": _safe_float(row.get("composite_score")),
        "confidence_band": str(row.get("confidence_band", "INSUFFICIENT")),
        "nan_pct": _safe_float(row.get("nan_pct")),
    }


def _top_features_by_horizon(predpow: pd.DataFrame | None,
                             k: int = 10) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {h: [] for h in HORIZONS}
    if predpow is None or predpow.empty:
        return out
    for h in HORIZONS:
        sub = predpow[predpow["horizon"] == h].copy()
        if sub.empty:
            continue
        # Insufficient rows to bottom
        sub["_ins"] = (sub["confidence_band"] == "INSUFFICIENT").astype(int)
        sub = sub.sort_values(
            by=["_ins", "composite_score"],
            ascending=[True, False],
            na_position="last",
        )
        out[h] = [_row_to_score_dict(r) for _, r in sub.head(k).iterrows()]
    return out


def _walkfwd_verdicts(wf: pd.DataFrame | None) -> dict[str, int]:
    buckets = ["ROBUST", "NEUTRAL", "DECAYING", "UNSTABLE",
               "NEGATIVE_OOS", "INSUFFICIENT"]
    out = {b: 0 for b in buckets}
    if wf is None or wf.empty or "verdict" not in wf.columns:
        return out
    counts = wf["verdict"].value_counts().to_dict()
    for k, v in counts.items():
        out[str(k)] = out.get(str(k), 0) + int(v)
    return out


def _conditional_top(cond: pd.DataFrame | None,
                     horizon: str = "ret_60m",
                     k: int = 5) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {r: [] for r in REGIMES}
    if cond is None or cond.empty:
        return out
    sub = cond[(cond["horizon"] == horizon)
               & (cond.get("is_regime_specific") == True)].copy()  # noqa: E712
    for r in REGIMES:
        rsub = sub[sub["regime"] == r].copy()
        if rsub.empty:
            out[r] = []
            continue
        rsub["abs_delta"] = rsub["delta_ic_vs_overall"].abs()
        rsub = rsub.sort_values("abs_delta", ascending=False, na_position="last")
        out[r] = [
            {
                "feature": str(row["feature"]),
                "n": int(row.get("n", 0) or 0),
                "ic": _safe_float(row.get("ic")),
                "delta_ic_vs_overall": _safe_float(row.get("delta_ic_vs_overall")),
                "composite_score": _safe_float(row.get("composite_score")),
                "confidence_band": str(row.get("confidence_band", "INSUFFICIENT")),
            }
            for _, row in rsub.head(k).iterrows()
        ]
    return out


def _conditional_regime_sample_counts(cond: pd.DataFrame | None) -> dict[str, int]:
    """Per-regime max n across features (proxy for samples in regime)."""
    out = {r: 0 for r in REGIMES}
    if cond is None or cond.empty:
        return out
    for r in REGIMES:
        rsub = cond[cond["regime"] == r]
        if not rsub.empty:
            out[r] = int(rsub["n"].max() or 0)
    return out


def _stable_across_horizons(predpow: pd.DataFrame | None,
                            k: int = 10) -> list[dict]:
    """Features ranked by number of horizons where they're top-k."""
    if predpow is None or predpow.empty:
        return []
    appearances: dict[str, list[str]] = {}
    for h in HORIZONS:
        sub = predpow[predpow["horizon"] == h].copy()
        if sub.empty:
            continue
        sub["_ins"] = (sub["confidence_band"] == "INSUFFICIENT").astype(int)
        sub = sub.sort_values(
            by=["_ins", "composite_score"],
            ascending=[True, False],
            na_position="last",
        )
        for feat in sub.head(k)["feature"].tolist():
            appearances.setdefault(feat, []).append(h)
    out = [
        {"feature": f, "horizons": hs, "n_horizons": len(hs)}
        for f, hs in appearances.items()
    ]
    out.sort(key=lambda d: (-d["n_horizons"], d["feature"]))
    return out


def _warnings(coverage: dict, predpow: pd.DataFrame | None) -> list[str]:
    out: list[str] = []
    n = coverage.get("n_snapshots", 0)
    if n < MIN_CONFIDENT_SAMPLES:
        out.append(
            f"Sample size insufficient for statistical confidence: "
            f"n={n} snapshots, recommend >= {MIN_CONFIDENT_SAMPLES}."
        )
    if predpow is not None and not predpow.empty:
        bands = predpow["confidence_band"].value_counts().to_dict()
        if bands.get("HIGH", 0) == 0 and bands.get("MEDIUM", 0) == 0:
            out.append(
                "No feature reached MEDIUM or HIGH confidence yet -- "
                "all results are preliminary."
            )
    return out


def build_summary(symbol: str = "NIFTY") -> dict:
    """Aggregate everything into a single JSON-serialisable dict."""
    flow = _load_flow(symbol)
    regimes = _load_regimes(symbol)
    predpow = _load_predpow(symbol)
    wf = _load_walkfwd(symbol)
    cond = _load_conditional(symbol)

    coverage = _coverage(flow)
    cur_regime = _current_regime(regimes)
    return {
        "symbol": symbol,
        "last_rebuild": _last_rebuild_ts(symbol),
        "coverage": coverage,
        "regime_breakdown": _regime_breakdown(regimes),
        "regime_sample_counts": _conditional_regime_sample_counts(cond),
        "current_regime": cur_regime["regime"] if cur_regime else None,
        "current_regime_detail": cur_regime,
        "recent_regimes": _last_n_regimes(regimes, n=10),
        "top_features": _top_features_by_horizon(predpow, k=10),
        "walkforward_verdicts": _walkfwd_verdicts(wf),
        "walkforward_rows": int(0 if wf is None else len(wf)),
        "conditional_edges": _conditional_top(cond, horizon="ret_60m", k=5),
        "stable_across_horizons": _stable_across_horizons(predpow, k=10),
        "warnings": _warnings(coverage, predpow),
        "automated_rebuild": _automated_rebuild_status(),
        "artifact_status": {
            "flow_features": flow is not None,
            "labels": _load_labels(symbol) is not None,
            "regimes": regimes is not None,
            "predictive_power": predpow is not None,
            "walkforward": wf is not None and not wf.empty,
            "conditional_edge": cond is not None,
        },
    }


def build_feature_detail(symbol: str, feature: str) -> dict | None:
    """All available data for one feature. Returns None if feature unknown."""
    predpow = _load_predpow(symbol)
    wf = _load_walkfwd(symbol)
    cond = _load_conditional(symbol)
    flow = _load_flow(symbol)

    # Known if it shows up in flow features OR predpow
    known = False
    if predpow is not None and not predpow.empty:
        known = known or (feature in set(predpow["feature"].unique()))
    if flow is not None and not flow.empty:
        known = known or (feature in set(flow["feature_name"].unique()))
    if not known:
        return None

    horizons_data = []
    if predpow is not None and not predpow.empty:
        sub = predpow[predpow["feature"] == feature]
        for _, row in sub.iterrows():
            d = _row_to_score_dict(row)
            # Decile means stored as comma-string
            dm = row.get("decile_means")
            if isinstance(dm, str):
                parsed = []
                for tok in dm.split(","):
                    tok = tok.strip()
                    if tok == "" or tok.lower() == "nan":
                        parsed.append(None)
                    else:
                        try:
                            parsed.append(float(tok))
                        except ValueError:
                            parsed.append(None)
                d["decile_means"] = parsed
            horizons_data.append(d)

    wf_rows = []
    if wf is not None and not wf.empty:
        sub = wf[wf["feature"] == feature]
        for _, row in sub.iterrows():
            wf_rows.append({k: _safe_float(v) if isinstance(v, (int, float))
                            else (None if pd.isna(v) else str(v))
                            for k, v in row.items()})

    cond_rows = []
    if cond is not None and not cond.empty:
        sub = cond[cond["feature"] == feature]
        for _, row in sub.iterrows():
            cond_rows.append({
                "horizon": str(row.get("horizon")),
                "regime": str(row.get("regime")),
                "n": int(row.get("n", 0) or 0),
                "ic": _safe_float(row.get("ic")),
                "delta_ic_vs_overall": _safe_float(row.get("delta_ic_vs_overall")),
                "composite_score": _safe_float(row.get("composite_score")),
                "is_regime_specific": bool(row.get("is_regime_specific", False)),
                "confidence_band": str(row.get("confidence_band", "INSUFFICIENT")),
            })

    recent_values = []
    if flow is not None and not flow.empty:
        sub = flow[flow["feature_name"] == feature].copy()
        if not sub.empty:
            sub = sub.sort_values("timestamp").tail(50)
            for _, row in sub.iterrows():
                recent_values.append({
                    "timestamp": str(row.get("timestamp")),
                    "value": _safe_float(row.get("feature_value")),
                })

    return {
        "symbol": symbol,
        "feature": feature,
        "horizons": horizons_data,
        "walkforward": wf_rows,
        "conditional": cond_rows,
        "recent_values": recent_values,
    }


# ---------------------------------------------------------------------------
# Rebuild pipeline
# ---------------------------------------------------------------------------

_REBUILD_STEPS: list[tuple[str, list[str]]] = [
    ("flow_features", [sys.executable, "-m", "engine.research.flow_features", "build-all"]),
    ("labels", [sys.executable, "-m", "engine.research.labels", "build"]),
    ("predictive_power", [sys.executable, "-m", "engine.research.predictive_power", "run"]),
    ("walkforward", [sys.executable, "-m", "engine.research.walkforward", "run"]),
    ("conditional_edge", [sys.executable, "-m", "engine.research.conditional_edge", "run"]),
]


def _run_rebuild(timeout_per_step: int = 600) -> dict:
    results = []
    overall_ok = True
    for name, cmd in _REBUILD_STEPS:
        t0 = time.time()
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=timeout_per_step, check=False,
            )
            ok = proc.returncode == 0
            results.append({
                "step": name,
                "ok": ok,
                "returncode": proc.returncode,
                "elapsed_sec": round(time.time() - t0, 2),
                "stderr_tail": (proc.stderr or "").splitlines()[-5:],
            })
            if not ok:
                overall_ok = False
        except Exception as e:
            results.append({
                "step": name,
                "ok": False,
                "error": f"{type(e).__name__}: {e}",
                "elapsed_sec": round(time.time() - t0, 2),
            })
            overall_ok = False
    return {"ok": overall_ok, "steps": results}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@research_bp.route("/api/research/snapshot_growth")
def api_snapshot_growth():
    """Cumulative snapshot counts per day (the 'patience meter')."""
    symbol = request.args.get("symbol", "NIFTY").upper()
    out: dict = {"dates": [], "cumulative": [], "daily": []}
    # Prefer parquet-folder listing (chain snapshots) if present
    snap_root = Path("data/chain_snapshots") / symbol
    if snap_root.exists():
        per_day: list[tuple[str, int]] = []
        for d in sorted(snap_root.iterdir()):
            if d.is_dir():
                n = sum(1 for _ in d.glob("*.parquet"))
                per_day.append((d.name, n))
        cum = 0
        for date_str, n in per_day:
            cum += n
            out["dates"].append(date_str)
            out["daily"].append(n)
            out["cumulative"].append(cum)
        return jsonify(out)
    # Fallback: derive from flow timestamps
    flow = _load_flow(symbol)
    if flow is None or flow.empty:
        return jsonify(out)
    ts = pd.to_datetime(flow["timestamp"]).dt.date
    counts = ts.value_counts().sort_index()
    cum = 0
    for d, n in counts.items():
        cum += int(n)
        out["dates"].append(str(d))
        out["daily"].append(int(n))
        out["cumulative"].append(cum)
    return jsonify(out)


@research_bp.route("/api/research/summary")
def api_summary():
    symbol = request.args.get("symbol", "NIFTY").upper()
    return jsonify(build_summary(symbol))


@research_bp.route("/api/research/feature/<feature>")
def api_feature(feature: str):
    symbol = request.args.get("symbol", "NIFTY").upper()
    detail = build_feature_detail(symbol, feature)
    if detail is None:
        return jsonify({"ok": False, "error": f"unknown feature '{feature}'"}), 404
    return jsonify(detail)


@research_bp.route("/api/research/rebuild", methods=["POST"])
def api_rebuild():
    symbol = request.args.get("symbol", "NIFTY").upper()
    res = _run_rebuild()
    res["symbol"] = symbol
    return jsonify(res), (200 if res["ok"] else 500)


@research_bp.route("/research")
def research_page():
    symbol = request.args.get("symbol", "NIFTY").upper()
    s = build_summary(symbol)
    return Response(_render_research_html(s), mimetype="text/html")


# ---------------------------------------------------------------------------
# HTML rendering (no Jinja file -- inline, matches the briefing page style)
# ---------------------------------------------------------------------------

def _esc(v: Any) -> str:
    if v is None:
        return "<span class='na'>--</span>"
    s = str(v)
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def _fmt_num(v: float | None, digits: int = 4, suffix: str = "") -> str:
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "<span class='na'>--</span>"
    return f"{v:.{digits}f}{suffix}"


def _band_class(band: str) -> str:
    return {
        "HIGH": "band-high",
        "MEDIUM": "band-med",
        "LOW": "band-low",
        "INSUFFICIENT": "band-ins",
    }.get(band, "band-ins")


def _verdict_class(v: str) -> str:
    return {
        "ROBUST": "v-good",
        "NEUTRAL": "v-neutral",
        "DECAYING": "v-warn",
        "UNSTABLE": "v-warn",
        "NEGATIVE_OOS": "v-bad",
        "INSUFFICIENT": "v-neutral",
    }.get(v, "v-neutral")


def _regime_class(r: str) -> str:
    return {
        "TREND": "rg-trend",
        "MEAN_REVERSION": "rg-mr",
        "PANIC": "rg-panic",
        "LOW_VOL": "rg-lowvol",
    }.get(r, "rg-other")


def _render_research_html(s: dict) -> str:
    sym = s["symbol"]
    cov = s["coverage"]
    n_snap = cov.get("n_snapshots", 0)
    n_feat = cov.get("n_features", 0)
    insufficient = n_snap < MIN_CONFIDENT_SAMPLES

    # Automated rebuild banner (data from data/research/rebuild_status.json)
    auto = s.get("automated_rebuild")
    auto_html = ""
    if auto is None:
        auto_html = (
            "<div class='banner warn'>"
            "No automated rebuild yet. Run "
            "<code>python -m engine.research.rebuild_daemon once</code> "
            "to trigger, or start the daemon with "
            "<code>python -m engine.research.rebuild_daemon run</code>."
            "</div>"
        )
    else:
        completed = auto.get("run_completed_at") or "--"
        status = auto.get("overall_status") or "UNKNOWN"
        status_cls = {
            "OK": "v-good",
            "PARTIAL": "v-warn",
            "FAILED": "v-bad",
        }.get(status, "v-neutral")
        failed_steps = auto.get("failed_steps") or []
        next_run = auto.get("run_completed_at")  # placeholder if 'next_run_at' missing
        next_run = auto.get("next_run_at") or next_run or "--"
        extra = ""
        if status in ("PARTIAL", "FAILED") and failed_steps:
            extra = (
                f"<div class='subtle' style='margin-top:4px'>"
                f"Failed steps: {_esc(', '.join(failed_steps))}"
                f"</div>"
            )
        banner_cls = "banner"
        if status == "PARTIAL":
            banner_cls = "banner warn"
        elif status == "FAILED":
            banner_cls = "banner danger"
        auto_html = (
            f"<div class='{banner_cls}'>"
            f"<strong>Last automated rebuild:</strong> {_esc(completed)} "
            f"&middot; status: <span class='verdict {status_cls}'>{_esc(status)}</span> "
            f"&middot; <span class='subtle'>next run: {_esc(next_run)}</span>"
            f"{extra}"
            f"</div>"
        )

    warning_html = ""
    if s["warnings"]:
        items = "".join(f"<li>{_esc(w)}</li>" for w in s["warnings"])
        warning_html = (
            "<div class='banner warn'><strong>Caveats</strong>"
            f"<ul>{items}</ul></div>"
        )

    big_banner = ""
    predpow_bands_all_ins = True
    for h in HORIZONS:
        for row in s["top_features"].get(h, []):
            if row.get("confidence_band") not in (None, "INSUFFICIENT"):
                predpow_bands_all_ins = False
                break
    if predpow_bands_all_ins and n_snap > 0:
        big_banner = (
            "<div class='banner danger'>"
            "Insufficient data for any statistical claims. "
            f"Current snapshot history: {n_snap} samples. "
            "Recommended minimum: 30 trading days (~510 snapshots) for "
            "preliminary findings; 1000+ for MEDIUM confidence."
            "</div>"
        )

    # ---- Section 2: regime breakdown
    rb = s["regime_breakdown"]
    rb_total = sum(rb.values()) or 1
    rb_bars = ""
    for r in REGIMES:
        n = rb.get(r, 0)
        pct = (n / rb_total) * 100
        rb_bars += (
            f"<div class='rg-row'>"
            f"<span class='rg-label {_regime_class(r)}'>{r}</span>"
            f"<div class='rg-bar-wrap'>"
            f"<div class='rg-bar {_regime_class(r)}' style='width:{pct:.1f}%'></div>"
            f"</div>"
            f"<span class='rg-n'>{n} ({pct:.1f}%)</span>"
            f"</div>"
        )
    cur = s.get("current_regime_detail") or {}
    cur_html = "<span class='na'>no regime data</span>"
    if cur:
        cur_html = (
            f"<span class='rg-pill {_regime_class(cur.get('regime',''))}'>"
            f"{_esc(cur.get('regime'))}</span> "
            f"<span class='subtle'>atr_pct={_fmt_num(cur.get('atr_pct'),4)} "
            f"realized_vol_5d={_fmt_num(cur.get('realized_vol_5d'),4)} "
            f"iv_rank={_fmt_num(cur.get('iv_rank'),3)} "
            f"adx_14={_fmt_num(cur.get('adx_14'),2)}</span>"
        )
    strip = ""
    for d in s["recent_regimes"]:
        strip += (
            f"<span class='strip-cell {_regime_class(d['regime'])}' "
            f"title='{_esc(d['date'])}: {_esc(d['regime'])}'>"
            f"{_esc((d['regime'] or '')[:2])}</span>"
        )

    # ---- Section 3: predictive features (per-horizon table, JS-switchable)
    horizon_tables: list[str] = []
    for h in HORIZONS:
        rows = s["top_features"].get(h, [])
        if not rows:
            horizon_tables.append(
                f"<table class='tbl' data-horizon='{h}' style='display:none'>"
                "<tbody><tr><td class='na'>No data for this horizon</td></tr>"
                "</tbody></table>"
            )
            continue
        body = ""
        for i, r in enumerate(rows, start=1):
            n = r.get("n") or 0
            n_label = (f"{n}" if n > 0 else "0")
            nan_pct = r.get("nan_pct")
            nan_note = ""
            if nan_pct and nan_pct > 0:
                nan_note = f" <span class='subtle'>({nan_pct*100:.0f}% nan)</span>"
            body += (
                f"<tr data-feature='{_esc(r['feature'])}'>"
                f"<td class='r'>{i}</td>"
                f"<td><a href='#' class='feat-link' data-feature='{_esc(r['feature'])}'>"
                f"{_esc(r['feature'])}</a></td>"
                f"<td>{_esc(r['horizon'])}</td>"
                f"<td class='r'>{_fmt_num(r.get('ic'),4)}</td>"
                f"<td class='r'>{_fmt_num(r.get('pearson'),4)}</td>"
                f"<td class='r'>{_fmt_num(r.get('mutual_info'),4)}</td>"
                f"<td class='r'>{_fmt_num(r.get('decile_spread'),5)}</td>"
                f"<td class='r'>{n_label}{nan_note}</td>"
                f"<td class='r'>{_fmt_num(r.get('composite_score'),4)}</td>"
                f"<td><span class='band {_band_class(r.get('confidence_band',''))}'>"
                f"{_esc(r.get('confidence_band'))}</span></td>"
                f"</tr>"
            )
        display = "" if h == "ret_60m" else "style='display:none'"
        horizon_tables.append(
            f"<table class='tbl sortable' data-horizon='{h}' {display}>"
            "<thead><tr>"
            "<th>Rank</th><th data-sort='str'>Feature</th>"
            "<th>Horizon</th>"
            "<th data-sort='num'>IC</th><th data-sort='num'>Pearson</th>"
            "<th data-sort='num'>MI</th><th data-sort='num'>DecileSpread</th>"
            "<th data-sort='num'>n</th><th data-sort='num'>Composite</th>"
            "<th>Confidence</th>"
            "</tr></thead>"
            f"<tbody>{body}</tbody></table>"
        )
    # ---- Top-10 composite scores for ret_60m (horizontal bar chart data) ----
    top60 = s["top_features"].get("ret_60m", [])[:10]
    top_chart_labels = json.dumps([r["feature"] for r in top60])
    top_chart_scores = json.dumps([r.get("composite_score") or 0 for r in top60])
    top_chart_bands = json.dumps([r.get("confidence_band") or "INSUFFICIENT"
                                  for r in top60])

    # ---- IC heatmap: top 15 features by best composite_score, across HORIZONS ----
    # Build {feature: {horizon: ic}} from the top_features dict
    feat_ic: dict[str, dict[str, float | None]] = {}
    feat_best: dict[str, float] = {}
    for h in HORIZONS:
        for row in s["top_features"].get(h, []):
            f = row["feature"]
            ic = row.get("ic")
            cs = row.get("composite_score") or 0
            feat_ic.setdefault(f, {})[h] = ic
            if cs > feat_best.get(f, -1):
                feat_best[f] = cs
    top_feats = sorted(feat_best.keys(), key=lambda f: -feat_best[f])[:15]

    def _hm_color(v):
        if v is None:
            return "#232730"
        # Scale: clamp |v| to 0.15
        m = max(min(v / 0.15, 1.0), -1.0)
        if m >= 0:
            # white -> green
            r = int(255 - m * (255 - 102))
            g = int(255 - m * (255 - 187))
            b = int(255 - m * (255 - 106))
        else:
            mm = -m
            r = int(255 - mm * (255 - 239))
            g = int(255 - mm * (255 - 83))
            b = int(255 - mm * (255 - 80))
        return f"rgb({r},{g},{b})"

    if top_feats:
        hm_cols = len(HORIZONS) + 1
        hm_html = (f"<div class='heatmap' "
                   f"style='grid-template-columns: 220px repeat({len(HORIZONS)}, 1fr)'>")
        hm_html += "<div class='hm-header'></div>"
        for h in HORIZONS:
            hm_html += f"<div class='hm-header'>{h}</div>"
        for f in top_feats:
            hm_html += f"<div class='hm-row-label' title='{_esc(f)}'>{_esc(f)}</div>"
            for h in HORIZONS:
                v = feat_ic.get(f, {}).get(h)
                cell_txt = f"{v:.3f}" if v is not None else "--"
                hm_html += (f"<div class='hm-cell' "
                            f"style='background:{_hm_color(v)}'>{cell_txt}</div>")
        hm_html += "</div>"
    else:
        hm_html = ("<div class='empty-state'><strong>No features ranked yet.</strong>"
                   " Heatmap populates once predictive_power has data across horizons.</div>")

    horizon_switcher = "".join(
        f"<option value='{h}' {'selected' if h == 'ret_60m' else ''}>{h}</option>"
        for h in HORIZONS
    )

    # ---- Section 4: walk-forward
    wf_html = ""
    if s["walkforward_rows"] == 0:
        wf_html = (
            f"<p class='na'>Walk-forward requires >= 600 samples; "
            f"current dataset has {n_snap}. "
            f"Empty summaries.parquet -- no verdicts yet.</p>"
        )
    else:
        wf_html = "<p class='subtle'>(walk-forward table)</p>"

    # Verdict counts strip
    wfv = s["walkforward_verdicts"]
    verdict_strip = " ".join(
        f"<span class='verdict {_verdict_class(k)}'>{k}: {v}</span>"
        for k, v in wfv.items()
    )

    # ---- Section 5: conditional edges
    cond_blocks = ""
    sample_counts = s.get("regime_sample_counts", {})
    for r in REGIMES:
        rows = s["conditional_edges"].get(r, [])
        n_r = sample_counts.get(r, 0)
        if n_r == 0:
            body = "<tr><td colspan='5' class='na'>No samples in regime yet</td></tr>"
        elif not rows:
            body = ("<tr><td colspan='5' class='na'>"
                    f"n={n_r} but no regime-specific features identified</td></tr>")
        else:
            body = ""
            for row in rows:
                body += (
                    f"<tr>"
                    f"<td>{_esc(row['feature'])}</td>"
                    f"<td class='r'>{row.get('n',0)}</td>"
                    f"<td class='r'>{_fmt_num(row.get('ic'),4)}</td>"
                    f"<td class='r'>{_fmt_num(row.get('delta_ic_vs_overall'),4)}</td>"
                    f"<td><span class='band {_band_class(row.get('confidence_band',''))}'>"
                    f"{_esc(row.get('confidence_band'))}</span></td>"
                    f"</tr>"
                )
        cond_blocks += (
            f"<div class='panel'>"
            f"<h3><span class='rg-pill {_regime_class(r)}'>{r}</span> "
            f"<span class='subtle'>n={n_r}</span></h3>"
            f"<table class='tbl'>"
            "<thead><tr><th>Feature</th><th>n</th><th>IC</th>"
            "<th>delta_IC</th><th>Conf</th></tr></thead>"
            f"<tbody>{body}</tbody></table></div>"
        )

    # ---- Section 6: stable features
    stable_rows = ""
    for row in s["stable_across_horizons"]:
        if row["n_horizons"] < 2:
            continue
        stable_rows += (
            f"<tr><td>{_esc(row['feature'])}</td>"
            f"<td class='r'>{row['n_horizons']} / {len(HORIZONS)}</td>"
            f"<td>{_esc(', '.join(row['horizons']))}</td></tr>"
        )
    if not stable_rows:
        stable_rows = (
            "<tr><td colspan='3' class='na'>"
            "No feature appears in top-10 for >= 2 horizons yet"
            "</td></tr>"
        )

    # ---- Header summary numbers
    cov_str = (
        f"{cov.get('start') or '--'} to {cov.get('end') or '--'}"
        if cov.get('start') else "no data"
    )
    artifact_status = s["artifact_status"]
    missing = [k for k, ok in artifact_status.items() if not ok]
    missing_html = ""
    if missing:
        missing_html = (
            "<div class='banner warn'>Missing artifacts: "
            + ", ".join(missing)
            + ". Run: <code>python -m engine.research.&lt;module&gt; run</code>"
            + "</div>"
        )

    summary_json = json.dumps(s, default=str)

    return f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Research Dashboard -- {sym}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
* {{ box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       background: #0f1115; color: #e7e9ee; margin: 0; padding: 16px;
       font-size: 13px; }}
a {{ color: #4fc3f7; text-decoration: none; }}
.top {{ display: flex; justify-content: space-between; align-items: center;
        flex-wrap: wrap; gap: 12px; margin-bottom: 16px; }}
.title {{ font-size: 22px; font-weight: 700; }}
.title .sym {{ color: #9aa3b2; font-size: 14px; margin-right: 8px; }}
.subtle {{ color: #9aa3b2; font-size: 11px; }}
.na {{ color: #5a6271; font-style: italic; }}
.banner {{ padding: 12px 14px; border-radius: 6px; margin-bottom: 12px;
           border: 1px solid #444; }}
.banner.warn {{ background: #2a230f; border-color: #6b5717; color: #ffd66e; }}
.banner.danger {{ background: #3d1414; border-color: #7a1a1a; color: #ff8585;
                  font-weight: 600; }}
.panel {{ background: #181b22; border-radius: 8px; padding: 14px;
          border: 1px solid #232730; margin-bottom: 12px; }}
.panel h2 {{ font-size: 13px; text-transform: uppercase; letter-spacing: 0.5px;
             color: #9aa3b2; margin: 0 0 10px 0; }}
.panel h3 {{ font-size: 12px; margin: 0 0 8px 0; color: #c5cad4; }}
.grid-2 {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
           gap: 10px; }}
.grid-4 {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
           gap: 10px; }}
.kv {{ display: flex; justify-content: space-between; padding: 3px 0;
       border-bottom: 1px dashed #232730; }}
.kv:last-child {{ border-bottom: none; }}
.kv .k {{ color: #9aa3b2; }}
.kv .v {{ font-weight: 600; font-variant-numeric: tabular-nums; }}
.tbl {{ width: 100%; border-collapse: collapse; font-size: 12px;
        font-variant-numeric: tabular-nums; }}
.tbl th, .tbl td {{ padding: 5px 8px; border-bottom: 1px solid #232730;
                    text-align: left; }}
.tbl th {{ color: #9aa3b2; font-weight: 500; font-size: 10px;
           text-transform: uppercase; letter-spacing: 0.5px; cursor: pointer;
           user-select: none; }}
.tbl th[data-sort]:after {{ content: ' ↕'; color: #444; font-size: 9px; }}
.tbl td.r, .tbl th.r {{ text-align: right; }}
.tbl tr:hover {{ background: #1d2129; }}
.band {{ display: inline-block; padding: 1px 6px; border-radius: 999px;
         font-size: 10px; font-weight: 600; }}
.band-high {{ background: #143d24; color: #66bb6a; }}
.band-med {{ background: #3d3514; color: #f5d76e; }}
.band-low {{ background: #3d2814; color: #ffa66e; }}
.band-ins {{ background: #232730; color: #6b7280; }}
.verdict {{ display: inline-block; padding: 2px 8px; border-radius: 999px;
            font-size: 10px; font-weight: 600; margin-right: 4px; }}
.v-good {{ background: #143d24; color: #66bb6a; }}
.v-neutral {{ background: #232730; color: #9aa3b2; }}
.v-warn {{ background: #3d2814; color: #ffa66e; }}
.v-bad {{ background: #3d1414; color: #ef5350; }}
.rg-row {{ display: flex; align-items: center; gap: 8px; margin: 4px 0; }}
.rg-label {{ width: 130px; font-size: 11px; font-weight: 600; }}
.rg-bar-wrap {{ flex: 1; background: #232730; border-radius: 4px;
                height: 12px; overflow: hidden; }}
.rg-bar {{ height: 100%; }}
.rg-n {{ width: 100px; text-align: right; font-size: 11px; color: #c5cad4; }}
.rg-trend, .rg-trend .rg-bar {{ background: #4fc3f7; }} .rg-trend.rg-label {{ background: none; color: #4fc3f7; }}
.rg-mr, .rg-mr .rg-bar {{ background: #ffa66e; }} .rg-mr.rg-label {{ background: none; color: #ffa66e; }}
.rg-panic, .rg-panic .rg-bar {{ background: #ef5350; }} .rg-panic.rg-label {{ background: none; color: #ef5350; }}
.rg-lowvol, .rg-lowvol .rg-bar {{ background: #66bb6a; }} .rg-lowvol.rg-label {{ background: none; color: #66bb6a; }}
.rg-label {{ background: none !important; }}
.rg-pill {{ display: inline-block; padding: 2px 10px; border-radius: 999px;
            font-size: 11px; font-weight: 600; color: #061419; }}
.strip-cell {{ display: inline-block; width: 26px; height: 22px; line-height: 22px;
               text-align: center; font-size: 10px; font-weight: 700; color: #061419;
               margin-right: 2px; border-radius: 3px; }}
select, button {{ background: #232730; color: #e7e9ee; border: 1px solid #333;
                  padding: 5px 10px; border-radius: 4px; font-family: inherit;
                  font-size: 12px; }}
button {{ cursor: pointer; }}
button:hover {{ background: #2d323c; }}
.decile-bar {{ display: inline-block; height: 14px; background: #4fc3f7;
               vertical-align: middle; margin-right: 1px; }}
.decile-row {{ background: #14171c; }}
.decile-row td {{ padding: 8px 12px; }}
code {{ background: #232730; padding: 1px 4px; border-radius: 3px;
        font-size: 11px; }}
.foot {{ color: #6b7280; font-size: 11px; margin-top: 24px;
         padding-top: 12px; border-top: 1px solid #232730; }}
/* Polish */
body {{ font-feature-settings: "tnum" 1, "lnum" 1; line-height: 1.45; }}
h1, h2, h3 {{ line-height: 1.2; }}
.panel {{ transition: transform 120ms ease, border-color 120ms ease; }}
.panel:hover {{ transform: translateY(-1px); border-color: #2c313c; }}
.tbl thead th {{ position: sticky; top: 0; background: #181b22; z-index: 1; }}
.title .sym {{
  background: linear-gradient(90deg, #4fc3f7 0%, #29b6f6 100%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  background-clip: text; font-weight: 700;
}}
.chart-wrap {{ position: relative; height: 220px; }}
.chart-wrap.tall {{ height: 280px; }}
.empty-state {{ color: #6b7280; font-size: 12px; padding: 24px 12px;
                text-align: center; line-height: 1.5; font-style: italic; }}
.empty-state strong {{ color: #c5cad4; display: block; margin-bottom: 4px;
                       font-style: normal; }}
.band, .verdict, .rg-pill {{ text-transform: uppercase; font-size: 10px;
                             letter-spacing: 0.3px; }}
/* IC heatmap */
.heatmap {{ display: grid; gap: 2px; font-size: 10px; }}
.heatmap .hm-cell {{ padding: 4px 6px; text-align: center; border-radius: 3px;
                     color: #061419; font-variant-numeric: tabular-nums; }}
.heatmap .hm-header {{ color: #9aa3b2; font-weight: 600; padding: 4px;
                       background: transparent; text-align: center; }}
.heatmap .hm-row-label {{ color: #c5cad4; padding: 4px 8px 4px 0;
                          text-align: right; background: transparent;
                          white-space: nowrap; overflow: hidden;
                          text-overflow: ellipsis; max-width: 200px; }}
</style></head><body>

<div class="top">
  <div>
    <div class="title"><span class="sym">RESEARCH</span>{sym}</div>
    <div class="subtle">
      Last rebuild: {_esc(s.get('last_rebuild') or 'never')}
      &middot; <a href="/">&larr; Dashboard</a>
      &middot; <a href="/briefing">Briefing</a>
      &middot; <a href="/api/research/summary">JSON</a>
    </div>
  </div>
  <div>
    <button id="rebuild-btn">Rebuild pipeline</button>
    <span id="rebuild-status" class="subtle"></span>
  </div>
</div>

{auto_html}
{big_banner}
{warning_html}
{missing_html}

<!-- Section 1: Data Coverage -->
<div class="panel">
  <h2>Data Coverage</h2>
  <div class="grid-4">
    <div><div class="kv"><span class="k">Symbol</span><span class="v">{sym}</span></div>
         <div class="kv"><span class="k">Date Range</span><span class="v">{cov_str}</span></div></div>
    <div><div class="kv"><span class="k">Snapshots</span><span class="v">{n_snap}</span></div>
         <div class="kv"><span class="k">Unique Features</span><span class="v">{n_feat}</span></div></div>
    <div><div class="kv"><span class="k">Walk-fwd rows</span><span class="v">{s['walkforward_rows']}</span></div>
         <div class="kv"><span class="k">Confidence</span><span class="v">{'INSUFFICIENT (n &lt; 1000)' if insufficient else 'See per-feature'}</span></div></div>
  </div>
</div>

<!-- Section 2: Regime Breakdown -->
<div class="panel">
  <h2>Regime Breakdown</h2>
  <div class="grid-2">
    <div>
      <h3>Distribution (daily samples)</h3>
      <div class="chart-wrap"><canvas id="regimeDonut"></canvas></div>
      <div id="regimeDonutEmpty" class="empty-state" style="display:none">
        <strong>No regime samples yet.</strong>
        Daily classifier needs at least 1 trading day to populate.
      </div>
      <div style="margin-top:10px">{rb_bars}</div>
    </div>
    <div>
      <h3>Current Regime</h3>
      <div style="margin-bottom: 8px">{cur_html}</div>
      <h3>Last 10 days</h3>
      <div>{strip or "<span class='na'>no data</span>"}</div>
    </div>
  </div>
</div>

<!-- Section 2b: Sample Size Growth (patience meter) -->
<div class="panel">
  <h2>Sample Size Growth</h2>
  <p class="subtle">
    Cumulative snapshots collected. Statistical confidence requires ~1000+
    snapshots (~30+ trading days at 17/day). Track progress here.
  </p>
  <div class="chart-wrap"><canvas id="growthChart"></canvas></div>
  <div id="growthEmpty" class="empty-state" style="display:none">
    <strong>Snapshot daemon running.</strong>
    First meaningful data in ~6 weeks. Currently building the foundation.
  </div>
</div>

<!-- Section 3a: Top Composite Scores chart + IC heatmap -->
<div class="panel">
  <h2>Top Features by Composite Score (ret_60m)</h2>
  <div class="chart-wrap tall"><canvas id="topFeatsChart"></canvas></div>
  <div id="topFeatsEmpty" class="empty-state" style="display:none">
    <strong>No ranked features yet.</strong>
    Composite scores populate once predictive_power has run with data.
  </div>
</div>

<div class="panel">
  <h2>IC by Horizon (top 15)</h2>
  <p class="subtle">
    Information Coefficient per feature x horizon. Green = positive predictive
    edge, red = negative, white/gray = near zero or unavailable.
  </p>
  {hm_html}
</div>

<!-- Section 3: Top Predictive Features -->
<div class="panel">
  <h2>Top Predictive Features</h2>
  <div style="margin-bottom: 8px">
    <label class="subtle">Horizon:</label>
    <select id="horizon-select">{horizon_switcher}</select>
    <label class="subtle" style="margin-left: 12px">
      <input type="checkbox" id="hide-insufficient" checked> Hide INSUFFICIENT rows
    </label>
  </div>
  <div id="horizon-tables">
    {''.join(horizon_tables)}
  </div>
</div>

<!-- Section 4: Walk-Forward Stability -->
<div class="panel">
  <h2>Walk-Forward Stability</h2>
  <div style="margin-bottom: 8px">{verdict_strip}</div>
  {wf_html}
</div>

<!-- Section 5: Conditional Edges by Regime -->
<div class="panel">
  <h2>Conditional Edges by Regime (horizon = ret_60m)</h2>
  <div class="grid-4">
    {cond_blocks}
  </div>
</div>

<!-- Section 6: Feature Importance Across Horizons -->
<div class="panel">
  <h2>Stability Across Horizons</h2>
  <p class="subtle">
    Features that appear in top-10 for multiple horizons are more robust
    than features that only show signal at one horizon.
  </p>
  <table class="tbl">
    <thead><tr><th>Feature</th><th>Horizons in top-10</th><th>Which</th></tr></thead>
    <tbody>{stable_rows}</tbody>
  </table>
</div>

<!-- Section 7: Methodology -->
<div class="foot">
  <strong>Methodology.</strong>
  composite_score = 0.35 * |IC| + 0.25 * |Pearson| + 0.20 * min(MI/0.1, 1)
  + 0.20 * min(|decile_spread|/0.002, 1). Direction-agnostic; check sign of
  IC/Pearson to know direction.
  <strong>Confidence bands:</strong>
  HIGH requires n &gt;= 1000, pearson_p &lt; 0.05, |IC| &gt; 0.05.
  MEDIUM requires 200 &lt;= n &lt; 1000 and p &lt; 0.20.
  LOW is everything significant-ish below MEDIUM.
  INSUFFICIENT is n &lt; 30. Small samples never earn HIGH no matter how
  good the metric looks.
  Rebuild via <a href="/api/research/rebuild">POST /api/research/rebuild</a>.
</div>

<script>
// Embedded summary -- so JS interactions don't need a second roundtrip
const SUMMARY = {summary_json};
const TOP_LABELS = {top_chart_labels};
const TOP_SCORES = {top_chart_scores};
const TOP_BANDS = {top_chart_bands};

// ---- Chart.js setup ----
if (window.Chart) {{
  Chart.defaults.color = '#9aa3b2';
  Chart.defaults.borderColor = '#232730';
  Chart.defaults.font.family = "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
}}

// Regime donut
(function() {{
  const rb = SUMMARY.regime_breakdown || {{}};
  const labels = Object.keys(rb);
  const data = labels.map(l => rb[l] || 0);
  const total = data.reduce((a, b) => a + b, 0);
  if (!total) {{
    document.getElementById('regimeDonut').style.display = 'none';
    document.getElementById('regimeDonutEmpty').style.display = 'block';
    return;
  }}
  const palette = {{ TREND: '#66bb6a', MEAN_REVERSION: '#4fc3f7',
                     PANIC: '#ef5350', LOW_VOL: '#9aa3b2' }};
  new Chart(document.getElementById('regimeDonut').getContext('2d'), {{
    type: 'doughnut',
    data: {{ labels, datasets: [{{
      data, backgroundColor: labels.map(l => palette[l] || '#9aa3b2'),
      borderColor: '#181b22', borderWidth: 2,
    }}]}},
    options: {{
      responsive: true, maintainAspectRatio: false, cutout: '60%',
      plugins: {{
        legend: {{ position: 'right' }},
        title: {{ display: true, text: total + ' total days',
                 color: '#c5cad4', font: {{ size: 14 }} }},
      }}
    }}
  }});
}})();

// Top features composite-score bar
(function() {{
  if (!TOP_LABELS.length) {{
    document.getElementById('topFeatsChart').style.display = 'none';
    document.getElementById('topFeatsEmpty').style.display = 'block';
    return;
  }}
  const bandColor = b => ({{ HIGH: '#66bb6a', MEDIUM: '#f5d76e',
                            LOW: '#ffa66e', INSUFFICIENT: '#5a6271' }}[b] || '#5a6271');
  new Chart(document.getElementById('topFeatsChart').getContext('2d'), {{
    type: 'bar',
    data: {{ labels: TOP_LABELS, datasets: [{{
      label: 'Composite Score', data: TOP_SCORES,
      backgroundColor: TOP_BANDS.map(bandColor),
    }}]}},
    options: {{
      indexAxis: 'y', responsive: true, maintainAspectRatio: false,
      plugins: {{ legend: {{ display: false }} }},
      scales: {{ x: {{ beginAtZero: true }} }}
    }}
  }});
}})();

// Snapshot growth line
fetch('/api/research/snapshot_growth').then(r => r.json()).then(d => {{
  if (!d.dates || !d.dates.length) {{
    document.getElementById('growthChart').style.display = 'none';
    document.getElementById('growthEmpty').style.display = 'block';
    return;
  }}
  new Chart(document.getElementById('growthChart').getContext('2d'), {{
    type: 'line',
    data: {{ labels: d.dates, datasets: [
      {{ label: 'Cumulative snapshots', data: d.cumulative,
        borderColor: '#4fc3f7', backgroundColor: 'rgba(79,195,247,0.15)',
        fill: true, tension: 0.2, pointRadius: 3, yAxisID: 'y' }},
      {{ label: 'Daily', data: d.daily, type: 'bar',
        backgroundColor: 'rgba(102,187,106,0.4)', yAxisID: 'y1' }},
    ]}},
    options: {{
      responsive: true, maintainAspectRatio: false,
      scales: {{
        y: {{ position: 'left', beginAtZero: true,
              title: {{ display: true, text: 'Cumulative' }} }},
        y1: {{ position: 'right', beginAtZero: true, grid: {{ display: false }},
               title: {{ display: true, text: 'Daily' }} }}
      }}
    }}
  }});
}}).catch(e => console.error(e));

// Horizon switcher
document.getElementById('horizon-select').addEventListener('change', function(e) {{
    const h = e.target.value;
    document.querySelectorAll('#horizon-tables table').forEach(function(t) {{
        t.style.display = (t.dataset.horizon === h) ? '' : 'none';
    }});
    applyInsufficientFilter();
}});

// Hide-insufficient toggle
function applyInsufficientFilter() {{
    const hide = document.getElementById('hide-insufficient').checked;
    document.querySelectorAll('#horizon-tables table tbody tr').forEach(function(tr) {{
        const band = tr.querySelector('.band');
        if (band && band.classList.contains('band-ins')) {{
            tr.style.display = hide ? 'none' : '';
        }}
    }});
}}
document.getElementById('hide-insufficient').addEventListener('change', applyInsufficientFilter);
applyInsufficientFilter();

// Column sort (vanilla JS)
document.querySelectorAll('table.sortable').forEach(function(table) {{
    const headers = table.querySelectorAll('th[data-sort]');
    headers.forEach(function(th, idx) {{
        // Find actual column index
        const cellIdx = Array.from(th.parentNode.children).indexOf(th);
        let asc = false;
        th.addEventListener('click', function() {{
            const tbody = table.querySelector('tbody');
            const rows = Array.from(tbody.querySelectorAll('tr'));
            const kind = th.dataset.sort;
            rows.sort(function(a, b) {{
                const av = a.children[cellIdx]?.innerText.trim() || '';
                const bv = b.children[cellIdx]?.innerText.trim() || '';
                let cmp;
                if (kind === 'num') {{
                    const af = parseFloat(av.replace(/[^0-9.\\-eE]/g, '')) || -1e18;
                    const bf = parseFloat(bv.replace(/[^0-9.\\-eE]/g, '')) || -1e18;
                    cmp = af - bf;
                }} else {{
                    cmp = av.localeCompare(bv);
                }}
                return asc ? cmp : -cmp;
            }});
            asc = !asc;
            rows.forEach(function(r) {{ tbody.appendChild(r); }});
        }});
    }});
}});

// Feature row click -> expand decile means
document.querySelectorAll('a.feat-link').forEach(function(a) {{
    a.addEventListener('click', function(e) {{
        e.preventDefault();
        const feat = a.dataset.feature;
        const tr = a.closest('tr');
        const next = tr.nextElementSibling;
        if (next && next.classList.contains('decile-row')) {{
            next.remove();
            return;
        }}
        fetch('/api/research/feature/' + encodeURIComponent(feat))
            .then(function(r) {{ return r.json(); }})
            .then(function(d) {{
                const newRow = document.createElement('tr');
                newRow.className = 'decile-row';
                const td = document.createElement('td');
                td.colSpan = 10;
                let html = '<strong>' + feat + '</strong> -- decile means by horizon';
                if (!d.horizons || !d.horizons.length) {{
                    html += ' <span class="na">no horizon data</span>';
                    td.innerHTML = html;
                    newRow.appendChild(td);
                    tr.parentNode.insertBefore(newRow, tr.nextSibling);
                    return;
                }}
                // Build a grid of canvases — one per horizon with data
                const wrap = document.createElement('div');
                wrap.style.cssText = 'display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:12px;margin-top:8px';
                const headerDiv = document.createElement('div');
                headerDiv.style.gridColumn = '1 / -1';
                headerDiv.innerHTML = html;
                wrap.appendChild(headerDiv);
                const canvases = [];
                d.horizons.forEach(function(h) {{
                    const dm = h.decile_means || [];
                    const cell = document.createElement('div');
                    cell.innerHTML = '<div class="subtle">' + h.horizon + ' (n=' + h.n + ')</div>';
                    const cvs = document.createElement('canvas');
                    cvs.style.height = '140px';
                    cell.appendChild(cvs);
                    wrap.appendChild(cell);
                    canvases.push({{cvs: cvs, dm: dm}});
                }});
                td.appendChild(wrap);
                newRow.appendChild(td);
                tr.parentNode.insertBefore(newRow, tr.nextSibling);
                // Render charts after DOM insertion
                canvases.forEach(function(item) {{
                    const dm = item.dm;
                    const labels = dm.map(function(_, i) {{ return 'D' + (i+1); }});
                    const colors = dm.map(function(v) {{
                        if (v == null) return '#3a3f4a';
                        return v >= 0 ? '#66bb6a' : '#ef5350';
                    }});
                    new Chart(item.cvs.getContext('2d'), {{
                        type: 'bar',
                        data: {{ labels: labels, datasets: [{{
                            label: 'Decile mean', data: dm.map(function(v) {{ return v == null ? 0 : v; }}),
                            backgroundColor: colors,
                        }}]}},
                        options: {{
                            responsive: true, maintainAspectRatio: false,
                            plugins: {{ legend: {{ display: false }} }},
                            scales: {{ y: {{ beginAtZero: true, ticks: {{ font: {{ size: 9 }} }} }},
                                       x: {{ ticks: {{ font: {{ size: 9 }} }} }} }}
                        }}
                    }});
                }});
            }})
            .catch(function(err) {{ console.error(err); }});
    }});
}});

// Rebuild button
document.getElementById('rebuild-btn').addEventListener('click', function() {{
    const btn = this;
    const status = document.getElementById('rebuild-status');
    if (!confirm('Re-run the full research pipeline? This may take several minutes.')) return;
    btn.disabled = true;
    status.textContent = ' running...';
    fetch('/api/research/rebuild', {{method: 'POST'}})
        .then(function(r) {{ return r.json(); }})
        .then(function(d) {{
            status.textContent = d.ok ? ' done -- reload page' : ' failed (see console)';
            console.log('rebuild result', d);
            btn.disabled = false;
        }})
        .catch(function(err) {{
            status.textContent = ' error: ' + err;
            btn.disabled = false;
        }});
}});
</script>

</body></html>"""


def register(app) -> None:
    """Attach the research blueprint to a Flask app."""
    app.register_blueprint(research_bp)
