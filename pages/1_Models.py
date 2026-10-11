"""Models page: choose a model, enter its inputs, run it."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.data import (expiry_T, latest_implied_vols, load_market_chain,
                       load_prices, load_rate, load_spot, smile_iv)
from core.models import (bsm, binomial, estimate_params, simulate_gbm,
                         simulate_portfolio, var_cvar)
from core.surface import atm_by_expiry, atm_iv_at

st.set_page_config(page_title="Models", page_icon="🧮", layout="wide")

MODELS = {
    "Black-Scholes": dict(
        what="Prices a European option (exercisable only at expiry) and its Greeks.",
        use="Fair value of an option and how sensitive it is to price, time, volatility and rates.",
        assumes="Constant volatility and interest rate, lognormal prices, no early exercise."),
    "Binomial (CRR)": dict(
        what="Prices an option on a price tree, including American options that can be exercised early.",
        use="Most US stock options are American. Shows how much early exercise is worth.",
        assumes="Same as Black-Scholes, approximated with a finite number of steps."),
    "Geometric Brownian Motion": dict(
        what="Simulates thousands of possible future price paths for one stock.",
        use="The range of likely prices, and the chance of finishing above or below a target.",
        assumes="Constant drift and volatility, normally distributed log returns, no jumps or crashes."),
    "Monte Carlo portfolio": dict(
        what="Simulates a portfolio of several stocks that move together (correlated).",
        use="Portfolio risk: Value at Risk, expected shortfall, probability of loss.",
        assumes="Past volatility and correlations hold, plus the return distribution you pick."),
}
HORIZONS = {"1 week": 5, "1 month": 21, "3 months": 63, "6 months": 126, "1 year": 252}
LOOKBACKS = {"6 months": 0.5, "1 year": 1, "3 years": 3, "5 years": 5}


# ---------------- chart helpers ----------------
def fan_chart(paths, title, y_title, n_samples=30):
    x = np.arange(paths.shape[1])
    p5, p25, p50, p75, p95 = np.percentile(paths, [5, 25, 50, 75, 95], axis=0)
    fig = go.Figure()
    for i in range(min(n_samples, len(paths))):
        fig.add_trace(go.Scatter(x=x, y=paths[i], mode="lines", line=dict(width=0.6),
                                 opacity=0.25, showlegend=False, hoverinfo="skip"))
    for lo, hi, name, alpha in [(p5, p95, "5–95%", 0.15), (p25, p75, "25–75%", 0.3)]:
        fig.add_trace(go.Scatter(x=x, y=hi, mode="lines", line=dict(width=0),
                                 showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=x, y=lo, mode="lines", line=dict(width=0),
                                 fill="tonexty", fillcolor=f"rgba(99,110,250,{alpha})",
                                 name=name))
    fig.add_trace(go.Scatter(x=x, y=p50, mode="lines", line=dict(width=2.5), name="Median"))
    fig.update_layout(title=title, height=480, xaxis_title="Trading days ahead",
                      yaxis_title=y_title)
    return fig


def hist_chart(values, title, x_title, lines):
    fig = go.Figure(go.Histogram(x=values, nbinsx=80, name="Simulations",
                                 marker_line_width=0))
    for label, x in lines.items():
        fig.add_vline(x=x, line_dash="dash", annotation_text=label)
    fig.update_layout(title=title, height=420, xaxis_title=x_title,
                      yaxis_title="Number of simulations", showlegend=False)
    return fig


def run_block(name, params, compute):
    """Run button + cached result that is only shown for matching inputs."""
    if st.button("▶ Run", type="primary", key=f"run_{name}"):
        with st.spinner("Running..."):
            try:
                st.session_state[name] = dict(params=params, result=compute())
            except Exception as e:
                st.error(f"Couldn't run the model: {e}")
                return None
    saved = st.session_state.get(name)
    if saved is None:
        st.caption("Set the inputs, then press Run.")
        return None
    if saved["params"] != params:
        st.warning("Inputs have changed. Press Run to update; showing the previous results.")
    return saved["result"]


# ---------------- option inputs (Black-Scholes and Binomial) ----------------
def option_inputs():
    c1, c2 = st.columns(2)
    ticker = c1.text_input("Ticker", "SPY").strip().upper()
    kind = c2.radio("Option type", ["call", "put"], horizontal=True)
    if not ticker:
        st.stop()

    with st.spinner(f"Loading {ticker} market data..."):
        try:
            spot = load_spot(ticker)
        except Exception:
            spot = None
        rate = load_rate()
        pts, source = load_market_chain(ticker)

    expiries = []
    if pts is not None:
        expiries = [e for e in sorted(pts.expiry.unique()) if expiry_T(e) > 1 / 365]
    if spot is None:
        st.warning("Couldn't load the spot price. Enter it manually.")
    if expiries:
        st.caption(f"Defaults come from {source}. You can change any value.")
    else:
        st.caption("No option data available right now, so enter volatility manually.")

    c1, c2, c3 = st.columns(3)
    S = c1.number_input("Spot price ($)", min_value=0.01,
                        value=round(spot, 2) if spot else 100.0, step=1.0)
    if expiries:
        expiry = c2.selectbox("Expiry", expiries,
                              format_func=lambda e: f"{e} ({expiry_T(e) * 365:.0f} days)")
        T = expiry_T(expiry)
        strikes = pts[pts.expiry == expiry].strike
        default_k = float(strikes.iloc[(strikes - S).abs().argmin()])
    else:
        expiry = None
        T = c2.number_input("Days to expiry", 1, 1095, 30) / 365
        default_k = float(round(S))
    K = c3.number_input("Strike ($)", min_value=0.01, value=default_k, step=1.0)

    mkt_iv, F = (smile_iv(pts, expiry, K) if expiry else (np.nan, np.nan))
    q_default = 0.0
    if expiry and spot and source and source.startswith("live"):
        q_default = float(np.clip(rate - np.log(F / spot) / T, 0, 0.10))

    c4, c5, c6 = st.columns(3)
    vol = c4.number_input("Volatility (%)", 0.1, 500.0,
                          round(mkt_iv * 100, 2) if np.isfinite(mkt_iv) else 25.0, 0.5,
                          help="Default: the market's implied vol at this strike and expiry")
    r = c5.number_input("Risk-free rate (%)", -5.0, 25.0, round(rate * 100, 3), 0.1,
                        help="Default: 13-week US T-bill yield")
    q = c6.number_input("Dividend yield (%)", 0.0, 20.0, round(q_default * 100, 2), 0.1,
                        help="Default: implied from the options' forward price when live")
    return dict(ticker=ticker, kind=kind, S=S, K=K, T=T, r=r / 100, q=q / 100,
                sigma=vol / 100, mkt_iv=mkt_iv)


# ---------------- model sections ----------------
def black_scholes_section():
    p = option_inputs()
    st.subheader("Step 3 · Results")
    g = bsm(p["S"], p["K"], p["T"], p["r"], p["q"], p["sigma"], p["kind"])
    cols = st.columns(6)
    cols[0].metric("Option price", f"${float(g['price']):,.2f}")
    cols[1].metric("Delta", f"{float(g['delta']):.3f}")
    cols[2].metric("Gamma", f"{float(g['gamma']):.4f}")
    cols[3].metric("Vega", f"{float(g['vega']):.3f}")
    cols[4].metric("Theta / day", f"{float(g['theta']):.3f}")
    cols[5].metric("Rho", f"{float(g['rho']):.3f}")
    with st.expander("What do these mean?"):
        st.markdown(
            "- **Delta**: how much the option price changes when the stock moves $1.\n"
            "- **Gamma**: how much delta changes when the stock moves $1.\n"
            "- **Vega**: price change when volatility rises 1 point (e.g. 20% to 21%).\n"
            "- **Theta**: price lost per day from time passing, all else equal.\n"
            "- **Rho**: price change when interest rates rise 1 point.")

    S_grid = np.linspace(p["S"] * 0.7, p["S"] * 1.3, 200)
    now = bsm(S_grid, p["K"], p["T"], p["r"], p["q"], p["sigma"], p["kind"])
    payoff = np.maximum(S_grid - p["K"], 0) if p["kind"] == "call" else np.maximum(p["K"] - S_grid, 0)
    greek = st.selectbox("Chart", ["price", "delta", "gamma", "vega", "theta"],
                         format_func=lambda x: f"Option {x} vs stock price")
    fig = go.Figure(go.Scatter(x=S_grid, y=now[greek], name="Today"))
    if greek == "price":
        fig.add_trace(go.Scatter(x=S_grid, y=payoff, name="At expiry", line=dict(dash="dash")))
    fig.add_vline(x=p["S"], line_dash="dot", annotation_text="Spot")
    fig.update_layout(height=450, xaxis_title="Stock price ($)", yaxis_title=greek.capitalize())
    st.plotly_chart(fig)


def binomial_section():
    p = option_inputs()
    steps = st.slider("Tree steps", 50, 2000, 500, 50,
                      help="More steps = more accurate but slower")
    st.subheader("Step 3 · Results")
    args = (p["S"], p["K"], p["T"], p["r"], p["q"], p["sigma"], p["kind"])
    try:
        amer = binomial(*args, steps=steps, american=True)
        euro = binomial(*args, steps=steps, american=False)
    except ValueError as e:
        st.error(str(e))
        return
    bs = float(bsm(*args)["price"])
    c = st.columns(4)
    c[0].metric("American (binomial)", f"${amer:,.2f}")
    c[1].metric("European (binomial)", f"${euro:,.2f}")
    c[2].metric("Black-Scholes", f"${bs:,.2f}")
    c[3].metric("Early-exercise premium", f"${amer - euro:,.3f}")
    st.caption("The early-exercise premium is the extra value of being able to exercise "
               "before expiry. It's near zero for calls on stocks without dividends, and "
               "largest for deep in-the-money puts.")

    step_list, vals = [], []
    for n in [10, 25, 50, 100, 200, 400, 800]:
        try:
            vals.append(binomial(*args, steps=n, american=False))
            step_list.append(n)
        except ValueError:
            pass
    fig = go.Figure(go.Scatter(x=step_list, y=vals, mode="lines+markers",
                               name="Binomial (European)"))
    fig.add_hline(y=bs, line_dash="dash", annotation_text="Black-Scholes")
    fig.update_layout(title="Convergence: more steps → closer to Black-Scholes",
                      height=420, xaxis_title="Steps", xaxis_type="log",
                      yaxis_title="Option price ($)")
    st.plotly_chart(fig)


def gbm_section():
    c1, c2, c3 = st.columns(3)
    ticker = c1.text_input("Ticker", "SPY").strip().upper()
    horizon = c2.selectbox("Horizon", list(HORIZONS), index=1)
    n_paths = c3.select_slider("Number of paths", [100, 500, 1000, 2000, 5000, 10000], 2000)
    c4, c5, c6 = st.columns(3)
    lookback = c4.selectbox("History used for estimates", list(LOOKBACKS), index=1)
    vol_src = c5.selectbox("Volatility source", ["Historical", "Implied (options market)", "Manual"])
    drift_src = c6.selectbox("Drift (expected return)",
                             ["Zero", "Risk-free rate", "Historical average", "Manual"], index=1)
    c7, c8, c9 = st.columns(3)
    manual_vol = c7.number_input("Volatility (%)", 1.0, 300.0, 25.0) if vol_src == "Manual" else None
    manual_mu = c8.number_input("Drift (% per year)", -100.0, 200.0, 8.0) if drift_src == "Manual" else None
    target = c9.number_input("Target price ($, optional)", 0.0, value=0.0,
                             help="Leave 0 to skip. Shows the chance of reaching this price.")
    if drift_src == "Historical average":
        st.caption("⚠️ Historical drift assumes the past trend continues. One great year "
                   "makes the simulation very optimistic.")

    params = dict(ticker=ticker, horizon=horizon, n=n_paths, lookback=lookback,
                  vol_src=vol_src, drift_src=drift_src, mv=manual_vol, mm=manual_mu,
                  target=target)

    def compute():
        prices = load_prices((ticker,), LOOKBACKS[lookback])
        if prices.empty or ticker not in prices:
            raise ValueError(f"no price history for {ticker}")
        _, sig_h, mu_h = estimate_params(prices)
        try:
            S0 = load_spot(ticker)
        except Exception:
            S0 = float(prices[ticker].iloc[-1])
        note = ""
        if vol_src == "Manual":
            sigma = manual_vol / 100
        elif vol_src.startswith("Implied"):
            sigma = latest_implied_vols([ticker]).get(ticker, np.nan)
            if not np.isfinite(sigma):
                pts, _ = load_market_chain(ticker)
                sigma = atm_iv_at(atm_by_expiry(pts), 30) if pts is not None else np.nan
            if not np.isfinite(sigma):
                sigma, note = float(sig_h[ticker]), "Implied vol unavailable, used historical."
        else:
            sigma = float(sig_h[ticker])
        mu = {"Zero": 0.0, "Risk-free rate": load_rate(),
              "Historical average": float(mu_h[ticker]),
              "Manual": (manual_mu or 0) / 100}[drift_src]
        paths = simulate_gbm(S0, mu, sigma, HORIZONS[horizon], n_paths, seed=42)
        return dict(paths=paths, S0=S0, sigma=sigma, mu=mu, note=note)

    st.subheader("Step 3 · Results")
    res = run_block("gbm", params, compute)
    if not res:
        return
    paths, S0, final = res["paths"], res["S0"], res["paths"][:, -1]
    if res["note"]:
        st.info(res["note"])
    st.caption(f"Using volatility {res['sigma']:.1%} and drift {res['mu']:.1%} per year, "
               f"start price ${S0:,.2f}.")
    p5, p50, p95 = np.percentile(final, [5, 50, 95])
    c = st.columns(4)
    c[0].metric("Median final price", f"${p50:,.2f}")
    c[1].metric("90% range", f"${p5:,.0f} – ${p95:,.0f}")
    c[2].metric("Chance above today's price", f"{(final > S0).mean():.0%}")
    if target > 0:
        hit = (paths >= target).any(1) if target > S0 else (paths <= target).any(1)
        side = (final >= target) if target > S0 else (final <= target)
        c[3].metric(f"Chance of touching ${target:,.0f}", f"{hit.mean():.0%}",
                    f"{side.mean():.0%} finish beyond it", delta_color="off")
    st.plotly_chart(fan_chart(paths, f"{params['ticker']} simulated price paths", "Price ($)"))
    st.plotly_chart(hist_chart(final, "Distribution of final prices", "Price ($)",
                               {"Today": S0, "Median": p50}))
    st.caption("These are possible outcomes under the model's assumptions, not predictions.")


def portfolio_section():
    st.markdown("**Your portfolio** (add or remove rows; 2 to 20 stocks)")
    default = pd.DataFrame({"Ticker": ["AAPL", "MSFT", "NVDA", "SPY"],
                            "Amount ($)": [2500.0, 2500.0, 2500.0, 2500.0]})
    table = st.data_editor(
        default, num_rows="dynamic", hide_index=True, key="portfolio_table",
        column_config={"Ticker": st.column_config.TextColumn(required=True),
                       "Amount ($)": st.column_config.NumberColumn(min_value=0.0, step=100.0,
                                                                   format="$%.0f")})
    c1, c2, c3 = st.columns(3)
    lookback = c1.selectbox("History used for estimates", list(LOOKBACKS), index=1)
    horizon = c2.selectbox("Horizon", list(HORIZONS), index=1)
    n_sims = c3.select_slider("Simulations", [1000, 5000, 10000, 20000], 5000)
    c4, c5, c6 = st.columns(3)
    vol_src = c4.selectbox("Volatility source",
                           ["Historical", "Implied (saved snapshots)"],
                           help="Implied uses the collector's latest 30-day ATM IV, so it "
                                "only covers watchlist tickers; others use historical.")
    drift_src = c5.selectbox("Drift (expected return)",
                             ["Zero", "Risk-free rate", "Historical average"], index=1)
    dist_label = c6.selectbox("Return distribution",
                              ["Normal", "Student-t (fat tails)", "Bootstrap (real past days)"])
    c7, c8 = st.columns(2)
    nu = c7.slider("Tail heaviness (degrees of freedom)", 3, 30, 5,
                   help="Lower = fatter tails = more extreme days") \
        if dist_label.startswith("Student") else 5
    conf = c8.radio("Confidence level", ["95%", "99%"], horizontal=True)

    clean = table.dropna().copy()
    clean["Ticker"] = clean["Ticker"].astype(str).str.strip().str.upper()
    clean["Amount ($)"] = pd.to_numeric(clean["Amount ($)"], errors="coerce")
    clean = clean[(clean["Ticker"] != "") & (clean["Amount ($)"] > 0)]
    clean = clean.groupby("Ticker", as_index=False)["Amount ($)"].sum()
    params = dict(holdings=tuple(map(tuple, clean.values.tolist())), lookback=lookback,
                  horizon=horizon, n=n_sims, vol=vol_src, drift=drift_src,
                  dist=dist_label, nu=nu, conf=conf)

    def compute():
        if not 2 <= len(clean) <= 20:
            raise ValueError("enter between 2 and 20 stocks with positive amounts")
        tickers = tuple(sorted(clean.Ticker))
        prices = load_prices(tickers, LOOKBACKS[lookback])
        missing = [t for t in tickers if t not in prices.columns]
        prices = prices[[t for t in tickers if t in prices.columns]]
        if prices.shape[1] < 2 or len(prices) < 60:
            raise ValueError("not enough overlapping price history; check the tickers")
        rets, sig_h, mu_h = estimate_params(prices)
        cols = list(prices.columns)
        amounts = clean.set_index("Ticker").loc[cols, "Amount ($)"].values
        sigma, src = sig_h[cols].values.copy(), ["historical"] * len(cols)
        if vol_src.startswith("Implied"):
            iv = latest_implied_vols(cols)
            for i, t in enumerate(cols):
                if t in iv and np.isfinite(iv[t]):
                    sigma[i], src[i] = iv[t], "implied"
        mu = {"Zero": np.zeros(len(cols)),
              "Risk-free rate": np.full(len(cols), load_rate()),
              "Historical average": mu_h[cols].values}[drift_src]
        corr = rets[cols].corr().values
        dist = {"Normal": "normal", "Student-t (fat tails)": "student_t",
                "Bootstrap (real past days)": "bootstrap"}[dist_label]
        values = simulate_portfolio(amounts, mu, sigma, corr, HORIZONS[horizon], n_sims,
                                    dist, nu, rets[cols].values, seed=42)
        return dict(values=values, cols=cols, amounts=amounts, sigma=sigma, mu=mu,
                    src=src, corr=corr, missing=missing, days=len(prices))

    st.subheader("Step 3 · Results")
    res = run_block("portfolio", params, compute)
    if not res:
        return
    if res["missing"]:
        st.warning(f"No price data for {', '.join(res['missing'])}; left out of the simulation.")
    values, v0 = res["values"], res["amounts"].sum()
    final = values[:, -1]
    level = 0.95 if conf == "95%" else 0.99
    var, cvar = var_cvar(final, v0, level)
    c = st.columns(5)
    c[0].metric("Starting value", f"${v0:,.0f}")
    c[1].metric("Median final value", f"${np.median(final):,.0f}")
    c[2].metric("Chance of a loss", f"{(final < v0).mean():.0%}")
    c[3].metric(f"Value at Risk ({conf})", f"${var:,.0f}")
    c[4].metric(f"Expected shortfall ({conf})", f"${cvar:,.0f}")
    st.caption(f"**Value at Risk**: in the worst {100 - level * 100:.0f}% of simulations, you "
               f"lose at least ${var:,.0f} over {horizon}. **Expected shortfall**: the "
               f"average loss within those worst cases (${cvar:,.0f}).")

    st.plotly_chart(fan_chart(values, "Simulated portfolio value", "Portfolio value ($)"))
    st.plotly_chart(hist_chart(final, f"Portfolio value after {horizon}", "Value ($)",
                               {"Start": v0, f"VaR {conf}": v0 - var}))

    left, right = st.columns(2)
    with left:
        st.markdown("**Assumptions used per stock**")
        st.dataframe(pd.DataFrame({
            "Ticker": res["cols"],
            "Weight": (res["amounts"] / v0 * 100).round(1).astype(str) + "%",
            "Volatility": (res["sigma"] * 100).round(1).astype(str) + "%",
            "Vol source": res["src"],
            "Drift": (np.asarray(res["mu"]) * 100).round(1).astype(str) + "%"}),
            hide_index=True)
        st.caption(f"Estimated from {res['days']} trading days of history.")
    with right:
        fig = go.Figure(go.Heatmap(z=res["corr"], x=res["cols"], y=res["cols"],
                                   zmin=-1, zmax=1, colorscale="RdBu_r",
                                   text=np.round(res["corr"], 2), texttemplate="%{text}"))
        fig.update_layout(title="Correlation of daily returns", height=420)
        st.plotly_chart(fig)
    st.caption("Higher correlation means the stocks fall together, so diversification "
               "helps less. These are possible outcomes under the model's assumptions, "
               "not predictions.")


# ---------------- page ----------------
st.title("🧮 Models")
st.caption("Pricing and risk models using live market data. For education only, "
           "not financial advice.")

st.subheader("Step 1 · Choose a model")
model = st.radio("Model", list(MODELS), horizontal=True, label_visibility="collapsed")
info = MODELS[model]
st.info(f"**{model}**: {info['what']}\n\n**Use it for:** {info['use']}\n\n"
        f"**Assumes:** {info['assumes']}")

st.subheader("Step 2 · Enter inputs")
{"Black-Scholes": black_scholes_section,
 "Binomial (CRR)": binomial_section,
 "Geometric Brownian Motion": gbm_section,
 "Monte Carlo portfolio": portfolio_section}[model]()
