"""Coverage page - data accumulation over time."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from streamlit_lib.db import coverage_stats, expiries_available, snapshots_per_day

st.title("Coverage")
st.caption("How much chain data we've accumulated in Aiven.")

try:
    cov = coverage_stats()
except Exception as e:
    st.error(f"Aiven connection failed: {e}")
    st.stop()

if cov["total_snapshots"] == 0:
    st.info(
        "No snapshots in Aiven yet. Once the snapshot daemon writes its first "
        "rows to `option_chain_snapshots`, charts populate here."
    )
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Rows", f"{cov['total_rows']:,}")
c2.metric("Distinct Snapshots", f"{cov['total_snapshots']:,}")
c3.metric("Trading Days", cov["days_covered"])
c4.metric(
    "Latest",
    cov["newest"].strftime("%Y-%m-%d %H:%M") if cov.get("newest") else "n/a",
)

if cov["days_covered"] < 30:
    st.warning(
        f"Only {cov['days_covered']} trading days collected. Statistical findings "
        "require 30+ days (~510 snapshots at 17/day) for preliminary signal, "
        "and 1000+ snapshots for MEDIUM-confidence research claims."
    )

st.subheader("Snapshots per day")
df = snapshots_per_day()
if df.empty:
    st.info("No per-day data.")
else:
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    df["cumulative"] = df["n_snapshots"].cumsum()

    fig = go.Figure()
    fig.add_bar(
        x=df["date"],
        y=df["n_snapshots"],
        name="Daily",
        marker_color="#66bb6a",
        opacity=0.6,
        yaxis="y2",
    )
    fig.add_scatter(
        x=df["date"],
        y=df["cumulative"],
        name="Cumulative",
        mode="lines+markers",
        line=dict(color="#4fc3f7", width=2),
        fill="tozeroy",
        fillcolor="rgba(79,195,247,0.10)",
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0f1115",
        plot_bgcolor="#0f1115",
        height=420,
        margin=dict(l=10, r=10, t=30, b=10),
        yaxis=dict(title="Cumulative snapshots"),
        yaxis2=dict(title="Daily", overlaying="y", side="right", showgrid=False),
        legend=dict(orientation="h", y=1.1),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(df.tail(30), use_container_width=True, hide_index=True)

st.subheader("Expiries available")
try:
    exp = expiries_available()
    if exp.empty:
        st.info("No expiry data.")
    else:
        st.dataframe(exp, use_container_width=True, hide_index=True)
except Exception as e:
    st.warning(f"Could not load expiries: {e}")
