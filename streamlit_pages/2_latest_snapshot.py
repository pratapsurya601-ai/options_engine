"""Latest Snapshot - current OI walls, max pain, IV smile."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from streamlit_lib.db import latest_chain, latest_snapshot_summary
from streamlit_lib.features import compute_snapshot_summary

st.title("Latest Snapshot")
st.caption("Most recent option chain in Aiven, with structural features overlaid.")

try:
    summary = latest_snapshot_summary()
    chain = latest_chain()
except Exception as e:
    st.error(f"Aiven connection failed: {e}")
    st.stop()

if not summary or chain.empty:
    st.info("No snapshot data available yet.")
    st.stop()

ts = summary.get("snapshot_ts")
st.markdown(f"**Snapshot timestamp:** `{ts}`")

feats = compute_snapshot_summary(chain)

# Filter to the chosen expiry
expiry = feats.get("expiry")
if expiry:
    df = chain[chain["expiry"].astype(str).str[:10] == expiry].copy()
else:
    df = chain.copy()

# KV metrics
c1, c2, c3, c4, c5, c6 = st.columns(6)
spot = feats.get("spot") or 0
c1.metric("Spot", f"{spot:,.0f}")
c2.metric("Max Pain", f"{feats['max_pain']:,.0f}" if feats.get("max_pain") else "n/a")
c3.metric("Call Wall", f"{feats['call_wall']:,.0f}" if feats.get("call_wall") else "n/a")
c4.metric("Put Wall", f"{feats['put_wall']:,.0f}" if feats.get("put_wall") else "n/a")
c5.metric("ATM IV", f"{feats['atm_iv']:.2f}" if feats.get("atm_iv") else "n/a")
skew = feats.get("iv_skew_25d")
c6.metric("IV Skew (25d)", f"{skew:.2f}" if skew is not None else "n/a")

c1, c2, c3 = st.columns(3)
c1.metric("PCR (OI)", f"{feats['pcr_oi']:.2f}" if feats.get("pcr_oi") else "n/a")
c2.metric("PCR (Vol)", f"{feats['pcr_volume']:.2f}" if feats.get("pcr_volume") else "n/a")
c3.metric("Smile Curvature", f"{feats['iv_smile_curvature']:.2f}" if feats.get("iv_smile_curvature") else "n/a")

st.divider()
st.subheader("OI Profile")

ce = df[df["option_type"] == "CE"].groupby("strike")["oi"].sum().reset_index()
pe = df[df["option_type"] == "PE"].groupby("strike")["oi"].sum().reset_index()

fig = go.Figure()
fig.add_bar(x=ce["strike"], y=ce["oi"], name="CE OI", marker_color="#66bb6a", opacity=0.85)
fig.add_bar(x=pe["strike"], y=-pe["oi"], name="PE OI", marker_color="#ef5350", opacity=0.85)
if spot:
    fig.add_vline(x=spot, line=dict(color="#4fc3f7", width=2, dash="dash"),
                  annotation_text="Spot", annotation_position="top")
if feats.get("max_pain"):
    fig.add_vline(x=feats["max_pain"], line=dict(color="#f5d76e", width=2, dash="dot"),
                  annotation_text="Max Pain", annotation_position="bottom")
fig.update_layout(
    template="plotly_dark",
    paper_bgcolor="#0f1115",
    plot_bgcolor="#0f1115",
    barmode="relative",
    height=460,
    margin=dict(l=10, r=10, t=30, b=10),
    xaxis_title="Strike",
    yaxis_title="OI (CE positive, PE negative)",
    legend=dict(orientation="h", y=1.08),
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("IV Smile")
iv_df = df[df["iv"].notna() & (df["iv"] > 0)].copy()
if iv_df.empty:
    st.info("No IV data in this snapshot.")
else:
    fig2 = go.Figure()
    for otype, color in (("CE", "#66bb6a"), ("PE", "#ef5350")):
        sub = iv_df[iv_df["option_type"] == otype].sort_values("strike")
        if not sub.empty:
            fig2.add_scatter(
                x=sub["strike"],
                y=sub["iv"],
                mode="lines+markers",
                name=otype,
                line=dict(color=color),
            )
    if spot:
        fig2.add_vline(x=spot, line=dict(color="#4fc3f7", width=2, dash="dash"))
    fig2.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0f1115",
        plot_bgcolor="#0f1115",
        height=360,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Strike",
        yaxis_title="IV (%)",
        legend=dict(orientation="h", y=1.12),
    )
    st.plotly_chart(fig2, use_container_width=True)

with st.expander("Raw chain rows"):
    st.dataframe(df, use_container_width=True, hide_index=True)
