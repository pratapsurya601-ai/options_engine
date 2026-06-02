"""
NIFTY Options Research Dashboard (Streamlit)
Reads from Aiven Postgres. Deployable to Streamlit Cloud free tier.

Local dev:
    streamlit run streamlit_app.py
"""
from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="NIFTY Options Engine",
    page_icon="N",
    layout="wide",
    initial_sidebar_state="expanded",
)


def home():
    st.title("NIFTY Options Engine")
    st.caption("Research dashboard - reads from Aiven Postgres")

    from streamlit_lib.db import latest_snapshot_summary, coverage_stats

    c1, c2, c3, c4 = st.columns(4)
    cov = None
    try:
        cov = coverage_stats()
        snap = latest_snapshot_summary()
        c1.metric("Total Snapshots", f"{cov['total_snapshots']:,}")
        c2.metric("Days Covered", cov["days_covered"])
        if snap and snap.get("spot") is not None:
            c3.metric("Latest Spot", f"Rs {float(snap['spot']):,.0f}")
            ts = snap.get("snapshot_ts")
            c4.metric("Last Update", ts.strftime("%Y-%m-%d %H:%M") if ts else "n/a")
        else:
            c3.metric("Latest Spot", "n/a")
            c4.metric("Last Update", "n/a")
    except Exception as e:
        st.error(f"Aiven connection failed: {e}")
        st.info(
            "Configure DATABASE_URL in `.streamlit/secrets.toml` (local) or via "
            "the Streamlit Cloud secrets UI (deployed)."
        )

    st.divider()

    if cov and cov.get("days_covered", 0) < 30:
        st.warning(
            f"Sample size warning: only {cov.get('days_covered', 0)} trading days collected. "
            "Until 30+ days accumulate, statistical analyses are preliminary."
        )

    st.markdown(
        """
### Pages (left sidebar)
- **Coverage** - data accumulation over time
- **Latest Snapshot** - current OI walls, max pain, IV smile
- **OI Flow** - put/call writing velocity, COI signals
- **Regime** - market regime classification with history
- **Top Features** - predictive power scores from research pipeline

### Notes
This dashboard reads directly from Aiven Postgres (`option_chain_snapshots`),
so it works from anywhere without your laptop being on. Data is cached for 5
minutes to stay within Streamlit Cloud free-tier limits.
"""
    )


PAGES_DIR = Path(__file__).parent / "streamlit_pages"

# Wire the `streamlit_pages/*.py` files via the new st.navigation / st.Page API
# (Streamlit >= 1.36). Each page module is loaded as its own Streamlit page;
# the file's top-level code is executed on selection.
_pages = [st.Page(home, title="Home", icon=":material/home:", default=True)]
if PAGES_DIR.exists():
    _files = [
        ("1_coverage.py", "Coverage"),
        ("2_latest_snapshot.py", "Latest Snapshot"),
        ("3_oi_flow.py", "OI Flow"),
        ("4_regime.py", "Regime"),
        ("5_top_features.py", "Top Features"),
    ]
    for fname, title in _files:
        p = PAGES_DIR / fname
        if p.exists():
            _pages.append(st.Page(str(p), title=title))

nav = st.navigation(_pages)
nav.run()
