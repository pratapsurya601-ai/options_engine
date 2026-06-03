"""Paper Positions page — open + closed paper trades, PnL summary per rule."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from streamlit_lib.db import (
    open_paper_positions,
    closed_paper_positions,
    position_pnl_summary,
)

st.title("Paper Positions")
st.caption(
    "Cloud watcher opens paper trades when rule conditions hit. "
    "All PnL is paper — no real orders are placed."
)

# ---- Per-rule PnL summary ----
try:
    summary = position_pnl_summary()
except Exception as e:
    st.error(f"Aiven connection failed: {e}")
    st.stop()

if summary.empty:
    st.info(
        "No closed paper trades yet. Rules fire infrequently — wait for the "
        "cloud watcher to generate signals during market hours."
    )
else:
    st.subheader("Realized PnL summary (paper)")
    display = summary.copy()
    display["win_rate"] = (
        display["wins"] / display["n_trades"].replace(0, pd.NA)
    ).fillna(0)
    display["total_pnl"] = display["total_pnl"].astype(float).round(0)
    display["avg_pnl"] = display["avg_pnl"].astype(float).round(0)
    display["avg_win"] = display["avg_win"].astype(float).round(0)
    display["avg_loss"] = display["avg_loss"].astype(float).round(0)
    display["win_rate"] = (display["win_rate"] * 100).round(1)
    display = display[
        ["rule_name", "n_trades", "wins", "losses", "win_rate",
         "total_pnl", "avg_pnl", "avg_win", "avg_loss",
         "first_exit", "last_exit"]
    ]
    display.columns = [
        "Rule", "Trades", "Wins", "Losses", "Win %",
        "Total PnL (Rs)", "Avg PnL", "Avg Win", "Avg Loss",
        "First exit", "Last exit"
    ]
    st.dataframe(display, use_container_width=True, hide_index=True)

st.divider()

# ---- Open positions ----
st.subheader("Open positions")
try:
    opens = open_paper_positions()
except Exception as e:
    st.warning(f"Could not load open positions: {e}")
    opens = pd.DataFrame()

if opens.empty:
    st.info("No open paper positions right now.")
else:
    display_open = opens.copy()
    display_open["entry_ts"] = pd.to_datetime(display_open["entry_ts"], errors="coerce")
    display_open = display_open[
        ["entry_ts", "rule_name", "action", "option_type",
         "strike", "expiry", "lots", "entry_price",
         "planned_target", "planned_stop", "high_water_mark",
         "entry_spot", "setup_tag"]
    ]
    display_open.columns = [
        "Entry time", "Rule", "Action", "Type",
        "Strike", "Expiry", "Lots", "Entry",
        "Target", "Stop", "High WM",
        "Entry spot", "Setup"
    ]
    st.dataframe(display_open, use_container_width=True, hide_index=True)

st.divider()

# ---- Closed positions + equity curve ----
st.subheader("Closed positions")
try:
    closed = closed_paper_positions(limit=300)
except Exception as e:
    st.warning(f"Could not load closed positions: {e}")
    st.stop()

if closed.empty:
    st.info("No closed paper positions yet.")
    st.stop()

# Equity curve
curve = closed.copy()
curve["exit_ts"] = pd.to_datetime(curve["exit_ts"], errors="coerce")
curve = curve.dropna(subset=["exit_ts"]).sort_values("exit_ts")
curve["cumulative_pnl"] = curve["pnl"].astype(float).cumsum()

if not curve.empty:
    end_pnl = float(curve["cumulative_pnl"].iloc[-1])
    line_color = "#66bb6a" if end_pnl >= 0 else "#ef5350"
    fig = go.Figure()
    fig.add_scatter(
        x=curve["exit_ts"], y=curve["cumulative_pnl"],
        mode="lines+markers",
        line=dict(color=line_color, width=2),
        fill="tozeroy",
        fillcolor=("rgba(102,187,106,0.10)" if end_pnl >= 0
                   else "rgba(239,83,80,0.10)"),
        name="Cumulative PnL",
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0f1115",
        plot_bgcolor="#0f1115",
        height=320,
        margin=dict(l=10, r=10, t=20, b=10),
        yaxis=dict(title="Cumulative paper PnL (Rs)"),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

display_closed = closed.copy()
display_closed["entry_ts"] = pd.to_datetime(display_closed["entry_ts"], errors="coerce")
display_closed["exit_ts"] = pd.to_datetime(display_closed["exit_ts"], errors="coerce")
display_closed["pnl"] = display_closed["pnl"].astype(float).round(0)
display_closed = display_closed[
    ["exit_ts", "rule_name", "action", "option_type",
     "strike", "expiry", "lots", "entry_price", "exit_price",
     "exit_reason", "pnl", "entry_ts"]
]
display_closed.columns = [
    "Exit time", "Rule", "Action", "Type",
    "Strike", "Expiry", "Lots", "Entry", "Exit",
    "Reason", "PnL (Rs)", "Entry time"
]
st.dataframe(display_closed, use_container_width=True, hide_index=True)

st.divider()
st.caption(
    "Exit reasons: `target_hit` (planned target reached), `stop_hit` (planned stop), "
    "`timeout` (hold_until_ts passed), `eod_close` (15:25 IST forced close). "
    "All trades 1-lot, paper only."
)
