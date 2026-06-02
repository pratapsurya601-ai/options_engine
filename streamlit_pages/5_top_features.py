"""Top Features - live feature computation + placeholder for predictive_power scores.

The full predictive_power / walk-forward scores currently live in local parquet
(`data/research/predictive_power/<sym>/scores.parquet`) and are not yet pushed
to Aiven. So for v1, this page:

1. Computes basic structural features live from the most recent snapshots
   (PCR, max pain, walls, IV skew) and shows them with mini sparklines.
2. Documents the future enhancement: surface predictive_power scores once the
   research pipeline writes them to Aiven.
"""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from streamlit_lib.db import chain_at_timestamp, recent_snapshot_timestamps
from streamlit_lib.features import compute_snapshot_summary

st.title("Top Features")
st.caption("Live feature snapshot + recent history. Full IC/walk-forward scores pending Aiven push.")

st.info(
    "**Coming soon.** Predictive-power scores (IC, decile spread, walk-forward "
    "verdicts) currently live in local parquet "
    "(`data/research/predictive_power/...`) and are not yet pushed to Aiven. "
    "Once the research pipeline writes them to a `predictive_power_scores` "
    "table, this page will rank features by composite_score per horizon."
)

st.subheader("Live structural features (last 50 snapshots)")

try:
    timestamps = recent_snapshot_timestamps(n=50)
except Exception as e:
    st.error(f"Aiven connection failed: {e}")
    st.stop()

if not timestamps:
    st.info("No snapshot data yet.")
    st.stop()

timestamps = list(reversed(timestamps))  # oldest first

rows = []
for ts in timestamps:
    chain = chain_at_timestamp(ts)
    if chain.empty:
        continue
    feats = compute_snapshot_summary(chain)
    feats["snapshot_ts"] = ts
    rows.append(feats)

if not rows:
    st.info("No feature rows computed.")
    st.stop()

hist = pd.DataFrame(rows)
hist["snapshot_ts"] = pd.to_datetime(hist["snapshot_ts"])

latest = hist.iloc[-1]
feat_names = [
    ("pcr_oi", "PCR (OI)", 2),
    ("pcr_volume", "PCR (Volume)", 2),
    ("pcr_oi_atm", "PCR ATM band", 2),
    ("max_pain", "Max Pain", 0),
    ("call_wall", "Call Wall", 0),
    ("put_wall", "Put Wall", 0),
    ("atm_iv", "ATM IV", 2),
    ("iv_skew_25d", "IV Skew (25d)", 2),
    ("iv_smile_curvature", "Smile Curvature", 2),
    ("oi_concentration_call", "OI Concentration (CE)", 3),
    ("oi_concentration_put", "OI Concentration (PE)", 3),
]

# Display as a table with sparklines
for name, label, digits in feat_names:
    series = hist[name].dropna() if name in hist.columns else pd.Series(dtype=float)
    cur = latest.get(name)
    c1, c2 = st.columns([1, 3])
    with c1:
        if cur is None or pd.isna(cur):
            st.metric(label, "n/a")
        else:
            st.metric(label, f"{float(cur):,.{digits}f}")
    with c2:
        if len(series) < 2:
            st.caption("not enough history for sparkline")
        else:
            fig = go.Figure()
            fig.add_scatter(
                x=hist["snapshot_ts"],
                y=hist[name],
                mode="lines",
                line=dict(color="#4fc3f7", width=1.5),
            )
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0f1115",
                plot_bgcolor="#0f1115",
                height=80,
                margin=dict(l=0, r=0, t=0, b=0),
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True, key=f"spark_{name}")

with st.expander("Raw feature history"):
    st.dataframe(hist, use_container_width=True, hide_index=True)
