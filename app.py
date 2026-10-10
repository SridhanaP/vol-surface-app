"""Volatility Surface Explorer - Phase 1: live surface page."""
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from core.charts import history_figure, skew_figure, surface_figure, term_figure
from core.surface import (atm_by_expiry, atm_iv_at, expected_moves,
                          fetch_chain_points, make_surface, skew_at)

SUMMARY = Path(__file__).parent / "data" / "summary.csv"
WINDOWS = {"Nearest expiry": None, "Up to 7 days": 7, "Up to 14 days": 14,
           "Up to 30 days": 30, "Up to 90 days": 90, "Up to 1 year": 365}

st.set_page_config(page_title="Volatility Surface Explorer", page_icon="📈",
                   layout="wide")


@st.cache_data(ttl=600, show_spinner=False)
def load_chain(ticker, use_last):
    """Cached for 10 minutes and shared by all visitors, to spare Yahoo."""
    return fetch_chain_points(ticker, min_days=1, max_days=365, use_last=use_last)


@st.cache_data(ttl=3600, show_spinner=False)
def load_history():
    if SUMMARY.exists():
        return pd.read_csv(SUMMARY, parse_dates=["date"])
    return pd.DataFrame()


def pct(x, digits=1):
    return "n/a" if x is None or not np.isfinite(x) else f"{x * 100:.{digits}f}%"


# ---------------- sidebar ----------------
with st.sidebar:
    st.header("Settings")
    ticker = st.text_input("Ticker", value="SPY",
                           help="Any US stock or ETF with listed options").strip().upper()
    window = st.selectbox("Expiries to include", list(WINDOWS),
                          index=len(WINDOWS) - 1)
    use_last = st.checkbox("Use last trade prices", value=False,
                           help="Turn on outside market hours, when Yahoo shows "
                                "no bid/ask. Results are rougher.")
    show_points = st.checkbox("Show market quotes on surface", value=True)
    if st.button("Refresh data"):
        load_chain.clear()
    st.caption("Data: Yahoo Finance, ~15 min delayed, cached for 10 minutes.")

# ---------------- main ----------------
st.title("📈 Volatility Surface Explorer")
st.caption("Implied volatility backed out of live option prices. "
           "For education only, not financial advice.")

if not ticker:
    st.info("Enter a ticker in the sidebar.")
    st.stop()

with st.spinner(f"Fetching {ticker} option chains (first load can take ~30 s)..."):
    try:
        pts, meta = load_chain(ticker, use_last)
    except Exception as e:
        st.error(f"Couldn't load {ticker}: {e}")
        st.stop()

if len(pts) < 5:
    st.warning("Too few usable option quotes. The market may be closed "
               "(try 'Use last trade prices' in the sidebar), or this ticker's "
               "options are too illiquid.")
    st.stop()

limit = WINDOWS[window]
if limit is None:
    view = pts[pts.expiry == pts.sort_values("days").expiry.iloc[0]]
else:
    view = pts[pts.days <= limit]
if view.empty:
    st.warning(f"No expiries within {limit} days for {ticker}. Pick a longer window.")
    st.stop()

spot = meta["spot"]
atm_all = atm_by_expiry(pts)
iv30 = atm_iv_at(atm_all, 30)
sk30 = skew_at(pts, 30)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Spot", f"{spot:,.2f}")
c2.metric("Risk-free rate", pct(meta["rate"], 2))
c3.metric("ATM IV (30d)", pct(iv30))
if np.isfinite(iv30):
    m = iv30 * np.sqrt(30 / 365)
    c4.metric("Expected move (30d)", f"±{spot * m:,.2f}", f"±{m * 100:.1f}%",
              delta_color="off")
else:
    c4.metric("Expected move (30d)", "n/a")
c5.metric("Skew 30d (90% − 110%)",
          "n/a" if not np.isfinite(sk30) else f"{sk30 * 100:+.1f} pts")
st.caption(f"Fetched {meta['fetched']} · {pts.expiry.nunique()} expiries · "
           f"{len(pts)} usable quotes · headline numbers use all expiries")

t_surf, t_skew, t_term, t_move, t_hist, t_raw = st.tabs(
    ["3D surface", "Skew", "Term structure", "Expected move", "History", "Raw data"])

with t_surf:
    grid = make_surface(view)
    if grid is None:
        st.info("A 3D surface needs at least two expiries. This window has one, "
                "so check the Skew tab.")
    else:
        K, D, IV = grid
        st.plotly_chart(surface_figure(view, K, D, IV,
                                       f"{ticker} implied volatility surface",
                                       show_points))
        st.caption("Drag to rotate, scroll to zoom. Black dots are real market "
                   "quotes; the surface between them is interpolated.")

with t_skew:
    exps = view.sort_values("days").expiry.unique().tolist()
    default = exps[::max(1, len(exps) // 4)][:4]
    chosen = st.multiselect("Expiries to compare", exps, default=default)
    if chosen:
        st.plotly_chart(skew_figure(view, chosen))
    st.caption("Higher IV on the left (low strikes) means traders are paying "
               "more for downside protection.")

with t_term:
    atm_view = atm_by_expiry(view)
    if atm_view.empty:
        st.info("Not enough at-the-money quotes in this window.")
    else:
        st.plotly_chart(term_figure(atm_view))
        st.caption("Upward slope: calmer now than expected later (normal). "
                   "Downward slope: the market expects near-term turbulence. "
                   "Bumps often mark earnings dates.")

with t_move:
    em = expected_moves(atm_by_expiry(view), spot)
    if em.empty:
        st.info("Not enough at-the-money quotes in this window.")
    else:
        st.dataframe(em, hide_index=True)
        st.caption("A one-standard-deviation range: if the options market is "
                   "right, the price ends inside it about 68% of the time. "
                   "In practice implied vol usually overstates actual moves a little.")

with t_hist:
    hist = load_history()
    h = hist[hist.ticker == ticker] if len(hist) else hist
    if len(h) < 2:
        st.info("No saved history for this ticker yet. The daily collector "
                "saves one snapshot per trading day for tickers in "
                "watchlist.txt, so this chart fills in over time.")
    else:
        st.plotly_chart(history_figure(h, ticker))
        st.caption(f"{len(h)} daily snapshots collected since "
                   f"{h.date.min():%Y-%m-%d}.")

with t_raw:
    st.dataframe(view, hide_index=True)
    st.download_button("Download CSV", view.to_csv(index=False),
                       file_name=f"{ticker}_iv_points.csv", mime="text/csv")
