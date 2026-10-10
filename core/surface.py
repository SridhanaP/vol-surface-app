"""Data fetching, implied volatility and surface construction (no plotting)."""
import time

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from scipy.optimize import brentq
from scipy.stats import norm

NY = "America/New_York"
YEAR_SECONDS = 365.0 * 24 * 3600
MAX_REL_SPREAD = 0.35   # drop quotes whose (ask-bid)/mid is wider than this
MIN_PRICE = 0.05        # drop options cheaper than this
COLUMNS = ["expiry", "T", "days", "strike", "forward", "moneyness", "logm",
           "mid", "type", "iv"]


# ---------------- pricing ----------------
def black76(F, K, T, DF, sigma, is_call):
    """Black-76 price on the forward (dividends are absorbed into F)."""
    sq = sigma * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sq * sq) / sq
    d2 = d1 - sq
    if is_call:
        return DF * (F * norm.cdf(d1) - K * norm.cdf(d2))
    return DF * (K * norm.cdf(-d2) - F * norm.cdf(-d1))


def implied_vol(price, F, K, T, DF, is_call):
    intrinsic = DF * max(F - K, 0.0) if is_call else DF * max(K - F, 0.0)
    upper = DF * F if is_call else DF * K
    if price <= intrinsic + 1e-6 or price >= upper:
        return np.nan
    f = lambda s: black76(F, K, T, DF, s, is_call) - price
    lo, hi = 1e-4, 5.0
    if f(lo) * f(hi) > 0:
        return np.nan
    try:
        return brentq(f, lo, hi, xtol=1e-6, maxiter=200)
    except Exception:
        return np.nan


# ---------------- market data ----------------
def get_spot(tk):
    """Return (last price, date of last trade in New York time)."""
    hist = tk.history(period="1d", interval="1m")
    if hist.empty:
        hist = tk.history(period="5d")
    close = hist["Close"].dropna() if not hist.empty else hist
    if len(close) == 0:
        raise ValueError("no price data returned")
    ts = close.index[-1]
    ts = ts.tz_convert(NY) if ts.tzinfo else ts
    return float(close.iloc[-1]), ts.date()


def get_rate():
    """13-week T-bill yield as the risk-free rate."""
    try:
        irx = yf.Ticker("^IRX").history(period="5d")["Close"].dropna()
        return float(irx.iloc[-1]) / 100.0
    except Exception:
        return 0.04


def clean_quotes(df, use_last=False):
    df = df[["strike", "bid", "ask", "lastPrice"]].copy()
    good = df[(df.bid > 0) & (df.ask > df.bid)].copy()
    good["mid"] = (good.bid + good.ask) / 2
    good = good[(good.mid >= MIN_PRICE) &
                ((good.ask - good.bid) / good.mid <= MAX_REL_SPREAD)]
    if good.empty and use_last:  # after-hours fallback
        good = df[df.lastPrice >= MIN_PRICE].copy()
        good["mid"] = good.lastPrice
    return good[["strike", "mid"]]


def implied_forward(calls, puts, spot, DF):
    """Forward from put-call parity near the money: F = K + (C - P) / DF."""
    fallback = spot / DF
    m = calls.merge(puts, on="strike", suffixes=("_c", "_p"))
    if m.empty:
        return fallback
    m["dist"] = (m.strike - spot).abs()
    near = m.nsmallest(6, "dist")
    F = float(np.median(near.strike + (near.mid_c - near.mid_p) / DF))
    return F if abs(F / spot - 1) < 0.10 else fallback


def fetch_chain_points(symbol, min_days=1, max_days=365, moneyness=(0.7, 1.3),
                       use_last=False, pause=0.2):
    """Download option chains and return (points DataFrame, meta dict)."""
    tk = yf.Ticker(symbol)
    spot, last_date = get_spot(tk)
    r = get_rate()
    now = pd.Timestamp.now(tz=NY)
    expiries = list(tk.options)
    if not expiries:
        raise ValueError(f"{symbol} has no listed options on Yahoo Finance")

    rows = []
    for exp in expiries:
        exp_ts = pd.Timestamp(f"{exp} 16:00").tz_localize(NY)
        T = (exp_ts - now).total_seconds() / YEAR_SECONDS
        days = T * 365
        if days < min_days or days > max_days:
            continue
        try:
            chain = tk.option_chain(exp)
        except Exception:
            continue
        time.sleep(pause)

        calls = clean_quotes(chain.calls, use_last)
        puts = clean_quotes(chain.puts, use_last)
        if calls.empty or puts.empty:
            continue

        DF = np.exp(-r * T)
        F = implied_forward(calls, puts, spot, DF)
        otm = pd.concat([puts[puts.strike < F].assign(is_call=False),
                         calls[calls.strike >= F].assign(is_call=True)])
        for q in otm.itertuples():
            k = q.strike / F
            if not (moneyness[0] <= k <= moneyness[1]):
                continue
            iv = implied_vol(q.mid, F, q.strike, T, DF, q.is_call)
            if np.isfinite(iv) and 0.01 < iv < 3.0:
                rows.append((exp, T, days, q.strike, F, k, np.log(k), q.mid,
                             "C" if q.is_call else "P", iv))

    pts = pd.DataFrame(rows, columns=COLUMNS)
    meta = dict(symbol=symbol, spot=spot, rate=r, last_date=str(last_date),
                fetched=now.strftime("%Y-%m-%d %H:%M ET"))
    return pts, meta


# ---------------- analytics ----------------
def atm_by_expiry(pts):
    """At-the-money IV (moneyness = 1) for every expiry that straddles it."""
    out = []
    for exp, g in pts.groupby("expiry"):
        g = g.sort_values("logm")
        if len(g) >= 2 and g.logm.min() <= 0 <= g.logm.max():
            out.append(dict(expiry=exp, T=g["T"].iloc[0], days=g.days.iloc[0],
                            atm_iv=float(np.interp(0.0, g.logm, g.iv)),
                            forward=g.forward.iloc[0]))
    cols = ["expiry", "T", "days", "atm_iv", "forward"]
    return pd.DataFrame(out, columns=cols).sort_values("days").reset_index(drop=True)


def atm_iv_at(atm, days, tolerance=0.25):
    """ATM IV at a target horizon, interpolated in total variance."""
    if len(atm) == 0:
        return np.nan
    T = days / 365
    Ts, ivs = atm["T"].values, atm.atm_iv.values
    if Ts.min() <= T <= Ts.max():
        w = np.interp(T, Ts, ivs ** 2 * Ts)
        return float(np.sqrt(w / T))
    i = int(np.argmin(np.abs(Ts - T)))       # slightly outside: use nearest
    return float(ivs[i]) if abs(Ts[i] - T) <= tolerance * T else np.nan


def skew_at(pts, days, lo=0.9, hi=1.1):
    """IV at 90% moneyness minus IV at 110%, on the expiry closest to `days`."""
    best = None
    for exp, g in pts.groupby("expiry"):
        if g.moneyness.min() <= lo and g.moneyness.max() >= hi:
            d = abs(g.days.iloc[0] - days)
            if best is None or d < best[0]:
                g = g.sort_values("moneyness")
                s = (np.interp(lo, g.moneyness, g.iv) -
                     np.interp(hi, g.moneyness, g.iv))
                best = (d, float(s))
    return best[1] if best else np.nan


def expected_moves(atm, spot):
    """One-standard-deviation expected move per expiry (~68% range)."""
    if len(atm) == 0:
        return pd.DataFrame()
    pct = atm.atm_iv * np.sqrt(atm["T"])
    return pd.DataFrame({
        "Expiry": atm.expiry,
        "Days": atm.days.round(1),
        "ATM IV (%)": (atm.atm_iv * 100).round(1),
        "Move (±%)": (pct * 100).round(2),
        "Move (±$)": (spot * pct).round(2),
        "Low": (spot * (1 - pct)).round(2),
        "High": (spot * (1 + pct)).round(2),
    })


def make_surface(pts, n_k=60, n_t=60, smooth=1.0):
    """Interpolate total variance on a (log-moneyness, T) grid.

    Returns (moneyness grid, days grid, IV grid) or None if there isn't
    enough data for a 2D surface.
    """
    if pts.expiry.nunique() < 2 or len(pts) < 10:
        return None
    x, y = pts.logm.values, pts["T"].values
    w = pts.iv.values ** 2 * y
    sx, sy = (np.ptp(x) or 1.0), (np.ptp(y) or 1.0)
    xn, yn = (x - x.min()) / sx, (y - y.min()) / sy
    gx, gy = np.meshgrid(np.linspace(0, 1, n_k), np.linspace(0, 1, n_t))

    X = x.min() + gx * sx
    Y = y.min() + gy * sy

    # inside the data: interpolate total variance (smooth across expiries)
    try:
        W = griddata((xn, yn), w, (gx, gy), method="linear")
    except Exception:
        W = np.full(gx.shape, np.nan)
    IV = np.sqrt(np.maximum(W, 1e-10) / Y)
    # outside the data: fill with the nearest quote's IV (filling total
    # variance there would blow up IV in the short-expiry corners)
    IV_near = griddata((xn, yn), pts.iv.values, (gx, gy), method="nearest")
    IV = np.where(np.isnan(W), IV_near, IV)
    if smooth:
        IV = gaussian_filter(IV, sigma=smooth)
    return np.exp(X), Y * 365, IV
