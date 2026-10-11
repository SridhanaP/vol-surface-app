"""Cached market-data loaders for the Streamlit pages."""
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

from .surface import NY, fetch_chain_points, get_rate, get_spot

ROOT = Path(__file__).resolve().parent.parent
SUMMARY = ROOT / "data" / "summary.csv"
RAW = ROOT / "data" / "raw"


@st.cache_data(ttl=600, show_spinner=False)
def load_spot(ticker):
    return get_spot(yf.Ticker(ticker))[0]


@st.cache_data(ttl=3600, show_spinner=False)
def load_rate():
    return get_rate()


@st.cache_data(ttl=3600, show_spinner=False)
def load_summary():
    if SUMMARY.exists():
        return pd.read_csv(SUMMARY, parse_dates=["date"])
    return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def load_snapshot(ticker):
    """Latest saved collector snapshot for a ticker: (points, label) or None."""
    for f in sorted(RAW.glob("*.csv.gz"), reverse=True):
        df = pd.read_csv(f)
        d = df[df.ticker == ticker].drop(columns="ticker")
        if len(d):
            return d.reset_index(drop=True), f"saved snapshot from {f.name.split('.')[0]}"
    return None


@st.cache_data(ttl=600, show_spinner=False)
def _live_chain(ticker):
    return fetch_chain_points(ticker, min_days=1, max_days=730, use_last=True)


def load_market_chain(ticker):
    """Option IV points from live data, else the latest snapshot.

    Returns (points, source label); points is None if nothing is available.
    """
    try:
        pts, _ = _live_chain(ticker)
        if len(pts) >= 5:
            return pts, "live option prices (Yahoo, ~15 min delayed)"
    except Exception:
        pass
    snap = load_snapshot(ticker)
    return snap if snap is not None else (None, None)


@st.cache_data(ttl=3600, show_spinner=False)
def load_prices(tickers, years):
    """Adjusted daily closes, one column per ticker, common dates only."""
    start = (pd.Timestamp.today() - pd.DateOffset(days=int(years * 365) + 7))
    df = yf.download(list(tickers), start=start.strftime("%Y-%m-%d"),
                     auto_adjust=True, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    close = df["Close"]
    if isinstance(close, pd.Series):
        close = close.to_frame(name=tickers[0])
    return close.dropna(axis=1, how="all").dropna()


def latest_implied_vols(tickers):
    """30-day ATM implied vol per ticker from saved snapshots (watchlist only)."""
    summ = load_summary()
    if summ.empty:
        return {}
    last = (summ.dropna(subset=["atm_iv_30d"]).sort_values("date")
                .groupby("ticker").tail(1).set_index("ticker").atm_iv_30d)
    return {t: float(last[t]) for t in tickers if t in last.index}


def expiry_T(expiry):
    """Years from now until 4 pm New York time on the expiry date."""
    now = pd.Timestamp.now(tz=NY)
    exp_ts = pd.Timestamp(f"{expiry} 16:00").tz_localize(NY)
    return (exp_ts - now).total_seconds() / (365.0 * 24 * 3600)


def smile_iv(pts, expiry, strike):
    """Market IV at a strike, interpolated along that expiry's smile."""
    g = pts[pts.expiry == expiry].sort_values("moneyness")
    if g.empty:
        return np.nan, np.nan
    F = float(g.forward.iloc[0])
    return float(np.interp(strike / F, g.moneyness, g.iv)), F
