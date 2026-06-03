"""Signals page — recent rule fires from the cloud_watcher pipeline."""
from __future__ import annotations

import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from streamlit_lib.db import recent_signals, signal_counts_by_rule

st.title("Signals")
st.caption("Every rule fire across cloud-deployed strategies. Newest first.")

# ---- Summary row ----
try:
    counts = signal_counts_by_rule(days=30)
except Exception as e:
    st.error(f"Aiven connection failed: {e}")
    st.stop()

if counts.empty:
    st.info(
        "No signals yet. Cloud watcher evaluates rules every 5 min during "
        "market hours but most rules fire rarely (htf_naked: ~1-2/month, "
        "panic_bounce_ce: regime-dependent). This page populates as signals occur."
    )
    st.stop()

cols = st.columns(min(4, len(counts) + 1))
total_fires = int(counts["total_fires"].sum())
cols[0].metric("Total fires (30d)", f"{total_fires:,}")
for i, row in counts.head(3).iterrows():
    if i + 1 < len(cols):
        cols[i + 1].metric(
            row["rule_name"],
            f"{int(row['total_fires'])} fires",
            f"{int(row['opened'])} opened",
        )

st.divider()

# ---- Per-rule counts bar chart ----
st.subheader("Fire counts by rule (last 30 days)")
fig = go.Figure()
fig.add_bar(
    x=counts["rule_name"],
    y=counts["opened"],
    name="Opened position",
    marker_color="#66bb6a",
)
fig.add_bar(
    x=counts["rule_name"],
    y=counts["cooldown_skipped"],
    name="Cooldown skip",
    marker_color="#9aa3b2",
)
fig.add_bar(
    x=counts["rule_name"],
    y=counts["alerts"],
    name="Alert only",
    marker_color="#4fc3f7",
)
fig.update_layout(
    template="plotly_dark",
    paper_bgcolor="#0f1115",
    plot_bgcolor="#0f1115",
    barmode="stack",
    height=320,
    margin=dict(l=10, r=10, t=20, b=10),
    yaxis=dict(title="Fires"),
    legend=dict(orientation="h", y=1.15),
)
st.plotly_chart(fig, use_container_width=True)

# ---- Recent signals table ----
st.subheader("Recent fires")
try:
    sigs = recent_signals(limit=200)
except Exception as e:
    st.warning(f"Could not load recent signals: {e}")
    st.stop()

if sigs.empty:
    st.info("No signal rows yet.")
    st.stop()

# Compact trigger_context preview
def _ctx_preview(v):
    if v is None:
        return ""
    try:
        d = v if isinstance(v, dict) else json.loads(v)
    except Exception:
        return str(v)[:80]
    keys = ("entry_premium", "target_premium", "stop_premium",
            "strike", "option_type", "spot", "delta")
    parts = [f"{k}={d[k]}" for k in keys if k in d]
    return ", ".join(parts)[:120]


display = sigs.copy()
display["ts"] = pd.to_datetime(display["ts"], errors="coerce")
display["context"] = display["trigger_context"].map(_ctx_preview)
display = display[
    ["ts", "rule_name", "action", "strike", "expiry", "spot",
     "premium", "target_premium", "stop_premium", "outcome", "context"]
]
display.columns = [
    "Time", "Rule", "Action", "Strike", "Expiry", "Spot",
    "Entry", "Target", "Stop", "Outcome", "Context"
]

st.dataframe(display, use_container_width=True, hide_index=True)

# ---- Methodology footer ----
st.divider()
st.caption(
    "Outcomes: `opened_position` = paper trade created in positions table. "
    "`skipped_cooldown` = rule fired but the per-rule cooldown blocked entry. "
    "`alert_only` = signal logged with no associated position (e.g. iron condor)."
)
