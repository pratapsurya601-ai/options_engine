"""Tests for the /research dashboard and its JSON APIs."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture(scope="module")
def client():
    # Ensure cwd is project root so RESEARCH_ROOT relative paths resolve
    here = Path(__file__).resolve()
    proj = here.parents[2]
    os.chdir(proj)
    from engine.web.app import app
    app.config["TESTING"] = True
    return app.test_client()


# ---------------------------------------------------------------------------
# Existing routes still work
# ---------------------------------------------------------------------------

def test_existing_routes_not_broken(client):
    assert client.get("/").status_code == 200
    # /api/positions reads logs/paper.jsonl which may be absent; just ensure JSON
    r = client.get("/api/positions")
    assert r.status_code == 200
    assert r.is_json
    r = client.get("/api/signals")
    assert r.status_code == 200
    assert r.is_json


# ---------------------------------------------------------------------------
# /research HTML
# ---------------------------------------------------------------------------

def test_research_page_returns_200(client):
    r = client.get("/research")
    assert r.status_code == 200
    body = r.data.decode("utf-8", errors="replace")
    assert "Research Dashboard" in body
    assert "NIFTY" in body
    # Section headers
    assert "Data Coverage" in body
    assert "Regime Breakdown" in body
    assert "Top Predictive Features" in body
    assert "Walk-Forward Stability" in body
    assert "Conditional Edges by Regime" in body


def test_research_page_renders_with_missing_files(client, tmp_path, monkeypatch):
    # Point RESEARCH_ROOT at an empty dir -> all loaders return None
    from engine.web import research_views as rv
    monkeypatch.setattr(rv, "RESEARCH_ROOT", tmp_path)
    rv._CACHE.clear()
    r = client.get("/research")
    assert r.status_code == 200
    body = r.data.decode("utf-8", errors="replace")
    assert "Missing artifacts" in body or "no data" in body.lower() or "never" in body
    rv._CACHE.clear()


# ---------------------------------------------------------------------------
# /api/research/summary
# ---------------------------------------------------------------------------

REQUIRED_SUMMARY_KEYS = {
    "symbol", "coverage", "regime_breakdown", "current_regime",
    "top_features", "walkforward_verdicts", "conditional_edges",
    "warnings", "stable_across_horizons", "artifact_status",
}


def test_summary_returns_valid_json(client):
    r = client.get("/api/research/summary")
    assert r.status_code == 200
    data = r.get_json()
    assert isinstance(data, dict)
    missing = REQUIRED_SUMMARY_KEYS - set(data.keys())
    assert not missing, f"missing keys: {missing}"
    # coverage shape
    cov = data["coverage"]
    assert "n_snapshots" in cov
    assert "n_features" in cov
    # top_features keyed by horizon
    for h in ("ret_15m", "ret_30m", "ret_60m", "ret_eod"):
        assert h in data["top_features"]
        assert isinstance(data["top_features"][h], list)
    # regime breakdown has all four regimes
    for rg in ("TREND", "MEAN_REVERSION", "PANIC", "LOW_VOL"):
        assert rg in data["regime_breakdown"]
        assert rg in data["conditional_edges"]


# ---------------------------------------------------------------------------
# /api/research/feature/<name>
# ---------------------------------------------------------------------------

def test_feature_404_for_unknown(client):
    r = client.get("/api/research/feature/this_feature_does_not_exist_xyz")
    assert r.status_code == 404
    data = r.get_json()
    assert data["ok"] is False
    assert "unknown feature" in data["error"]


def test_feature_returns_data_for_known(client):
    # Discover a real feature name from predictive_power scores
    from engine.web.research_views import _load_predpow, _load_flow
    df = _load_predpow("NIFTY")
    if df is None or df.empty:
        df = _load_flow("NIFTY")
        if df is None or df.empty:
            pytest.skip("No research data on disk to test against")
        feat = df["feature_name"].iloc[0]
    else:
        feat = df["feature"].iloc[0]

    r = client.get(f"/api/research/feature/{feat}")
    assert r.status_code == 200
    data = r.get_json()
    assert data["feature"] == feat
    assert data["symbol"] == "NIFTY"
    assert "horizons" in data
    assert "walkforward" in data
    assert "conditional" in data
    assert "recent_values" in data
    assert isinstance(data["horizons"], list)
