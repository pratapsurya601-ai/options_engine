"""OI Flow - put/call writing velocity across recent snapshots."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from streamlit_lib.db import chain_at_timestamp, recent_snapshot_timestamps
from streamlit_lib.features import oi_writing_velocities

st.title("OI Flow")
st.caption("Put/call writing and unwinding velocity across the last 10 snapshots (~50 min).")

try:
    timestamps = recent_snapshot_timestamps(n=10)
except Exception as e:
    st.error(f"Aiven connection failed: {e}")
    st.stop()

if not timestamps or len(timestamps) < 2:
    st.info(
        "Need at least 2 snapshots to compute inter-snapshot velocities. "
        f"Currently have {len(timestamps)}."
    )
    st.stop()

# Oldest first for sequential diff
timestamps = list(reversed(timestamps))

# Pick nearest expiry from the most recent snapshot
latest = chain_at_timestamp(timestamps[-1])
if latest.empty:
    st.info("Latest snapshot is empty.")
    st.stop()
exps = sorted([str(e)[:10] for e in latest["expiry"].dropna().unique()])
if not exps:
    st.info("No expiry data.")
    st.stop()
expiry = exps[0]
st.markdown(f"Using nearest expiry: `{expiry}`")

# Load all snapshots and filter to chosen expiry
snapshots = []
for ts in timestamps:
    df = chain_at_timestamp(ts)
    df = df[df["expiry"].astype(str).str[:10] == expiry]
    if not df.empty:
        snapshots.append(df)

if len(snapshots) < 2:
    st.info("Not enough snapshots share the chosen expiry.")
    st.stop()

vel = oi_writing_velocities(snapshots)
if vel.empty:
    st.info("No velocity rows computed.")
    st.stop()

vel["snapshot_ts"] = pd.to_datetime(vel["snapshot_ts"])

c1, c2, c3, c4 = st.columns(4)
last = vel.iloc[-1]
c1.metric("Put Writing (/min)", f"{last['put_writing_velocity']:,.0f}")
c2.metric("Call Writing (/min)", f"{last['call_writing_velocity']:,.0f}")
c3.metric("Put Unwinding (/min)", f"{last['put_unwinding_velocity']:,.0f}")
c4.metric("Call Unwinding (/min)", f"{last['call_unwinding_velocity']:,.0f}")

st.subheader("Velocity timeline")
fig = go.Figure()
fig.add_scatter(x=vel["snapshot_ts"], y=vel["put_writing_velocity"],
                name="Put writing", mode="lines+markers",
                line=dict(color="#ef5350"))
fig.add_scatter(x=vel["snapshot_ts"], y=vel["call_writing_velocity"],
                name="Call writing", mode="lines+markers",
                line=dict(color="#66bb6a"))
fig.add_scatter(x=vel["snapshot_ts"], y=vel["put_unwinding_velocity"],
                name="Put unwinding", mode="lines+markers",
                line=dict(color="#ef5350", dash="dot"))
fig.add_scatter(x=vel["snapshot_ts"], y=vel["call_unwinding_velocity"],
                name="Call unwinding", mode="lines+markers",
                line=dict(color="#66bb6a", dash="dot"))
fig.update_layout(
    template="plotly_dark",
    paper_bgcolor="#0f1115",
    plot_bgcolor="#0f1115",
    height=420,
    margin=dict(l=10, r=10, t=30, b=10),
    xaxis_title="Snapshot time",
    yaxis_title="OI change / minute",
    legend=dict(orientation="h", y=1.1),
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Net write/unwind balance")
vel["call_net"] = vel["call_writing_velocity"] - vel["call_unwinding_velocity"]
vel["put_net"] = vel["put_writing_velocity"] - vel["put_unwinding_velocity"]

fig2 = go.Figure()
fig2.add_bar(x=vel["snapshot_ts"], y=vel["put_net"], name="Put net (+write/-unwind)",
             marker_color="#ef5350")
fig2.add_bar(x=vel["snapshot_ts"], y=vel["call_net"], name="Call net",
             marker_color="#66bb6a")
fig2.update_layout(
    template="plotly_dark",
    paper_bgcolor="#0f1115",
    plot_bgcolor="#0f1115",
    barmode="group",
    height=340,
    margin=dict(l=10, r=10, t=30, b=10),
    legend=dict(orientation="h", y=1.1),
)
st.plotly_chart(fig2, use_container_width=True)

with st.expander("Raw velocity table"):
    st.dataframe(vel, use_container_width=True, hide_index=True)
