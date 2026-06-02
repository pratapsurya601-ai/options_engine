"""Regime - market regime classification with history.

NOTE: regime labels currently live in local parquet
(`data/research/regimes/<sym>/timeline_daily.parquet`), not in Aiven. For this
Streamlit Cloud dashboard to surface them, the research pipeline needs to
write a `regime_labels` table to Aiven. Until then, this page shows a
placeholder with the intended schema and explanation.
"""
import streamlit as st

st.title("Regime")
st.caption("Daily market regime classification (TREND / MEAN_REVERSION / PANIC / LOW_VOL).")

st.info(
    "**Coming soon.** Regime labels are currently written only to local parquet "
    "(`data/research/regimes/NIFTY/timeline_daily.parquet`) by the research "
    "pipeline. To surface them on Streamlit Cloud, the pipeline needs to also "
    "push them to Aiven (a new `regime_labels` table). Tracked as future work."
)

st.subheader("Planned schema")
st.code(
    """
CREATE TABLE regime_labels (
  symbol      TEXT NOT NULL,
  trade_date  DATE NOT NULL,
  regime      TEXT NOT NULL CHECK (regime IN
              ('TREND','MEAN_REVERSION','PANIC','LOW_VOL')),
  atr_pct           NUMERIC(10,6),
  realized_vol_5d   NUMERIC(10,6),
  iv_rank           NUMERIC(6,4),
  ema_slope_20      NUMERIC(10,6),
  adx_14            NUMERIC(6,2),
  computed_at       TIMESTAMPTZ DEFAULT NOW(),
  PRIMARY KEY (symbol, trade_date)
);
""",
    language="sql",
)

st.subheader("What this page will show once data is in Aiven")
st.markdown(
    """
- Pie chart of regime distribution across all classified days
- Current regime card with the latest `atr_pct`, `iv_rank`, `adx_14`
- Last-30-days regime strip (one cell per day, colored by regime)
- Per-regime sample counts (so we know which regimes have enough data
  for conditional-edge analysis)
"""
)
