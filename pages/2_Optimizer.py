"""Portfolio optimizer page: Markowitz mean-variance optimization."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.data import load_prices, load_rate
from core.optimizer import (capm_returns, efficient_frontier, ledoit_wolf, make_bounds,
                            max_sharpe, min_variance, realized, sample_cov, shrink_means,
                            stats, target_volatility)

st.set_page_config(page_title="Optimizer", page_icon="⚖️", layout="wide")

LOOKBACKS = {"1 year": 1, "3 years": 3, "5 years": 5}
DEFAULT_TICKERS = "AAPL, MSFT, NVDA, JPM, XOM, GLD, TLT, SPY"
RET_SOURCES = ["Shrunk historical (recommended)", "Historical average",
               "CAPM (beta × market premium)"]
COV_SOURCES = ["Ledoit-Wolf shrinkage (recommended)", "Sample covariance"]


def estimate(rets, market, cfg, rf):
    """Expected returns and covariance from daily log returns, per the chosen methods."""
    sigma2 = rets.var(ddof=1) * 252
    mu_hist = (rets.mean() * 252 + 0.5 * sigma2).values
    betas = None
    if cfg["ret"].startswith("Shrunk"):
        mu = shrink_means(mu_hist, cfg["shrink"])
    elif cfg["ret"].startswith("CAPM"):
        mu, betas = capm_returns(rets.values, market.values, rf, cfg["premium"])
    else:
        mu = mu_hist
    if cfg["cov"].startswith("Ledoit"):
        cov, lw = ledoit_wolf(rets.values)
    else:
        cov, lw = sample_cov(rets.values), None
    return mu, cov, mu_hist, betas, lw


def compute(tickers, cfg):
    need = tuple(sorted(set(tickers) | ({"SPY"} if cfg["ret"].startswith("CAPM") else set())))
    prices = load_prices(need, LOOKBACKS[cfg["lookback"]])
    missing = [t for t in tickers if t not in prices.columns]
    cols = [t for t in tickers if t in prices.columns]
    if len(cols) < 2:
        raise ValueError("need at least 2 tickers with price history")
    if cfg["ret"].startswith("CAPM") and "SPY" not in prices.columns:
        raise ValueError("couldn't load SPY, which CAPM uses as the market")
    all_rets = np.log(prices).diff().dropna()
    if len(all_rets) < 120:
        raise ValueError("not enough overlapping price history (need ~6 months)")
    rets, market = all_rets[cols], all_rets.get("SPY")
    rf = load_rate()
    bounds = make_bounds(len(cols), cfg["cap"], cfg["short"])

    mu, cov, mu_hist, betas, lw = estimate(rets, market, cfg, rf)
    frontier = efficient_frontier(mu, cov, bounds)
    w_ms = max_sharpe(mu, cov, rf, bounds, frontier)

    # out-of-sample check: estimate on the first 75%, measure on the last 25%
    split = int(len(rets) * 0.75)
    oos = None
    if split >= 120 and len(rets) - split >= 40:
        tr, te = rets.iloc[:split], rets.iloc[split:]
        mkt_tr = market.iloc[:split] if market is not None else None
        mu_t, cov_t, *_ = estimate(tr, mkt_tr, cfg, rf)
        fr_t = efficient_frontier(mu_t, cov_t, bounds)
        port = {"Minimum variance": fr_t[2][0],
                "Maximum Sharpe": max_sharpe(mu_t, cov_t, rf, bounds, fr_t),
                "Equal weight": np.full(len(cols), 1 / len(cols))}
        simple = np.expm1(te.values)
        rows = []
        for name, w in port.items():
            er, ev, es = stats(w, mu_t, cov_t, rf)
            rr, rv, rs = realized(simple, w, rf)
            rows.append({"Portfolio": name, "Predicted return": f"{er:.1%}",
                         "Actual return": f"{rr:.1%}", "Predicted volatility": f"{ev:.1%}",
                         "Actual volatility": f"{rv:.1%}", "Actual Sharpe": f"{rs:.2f}"})
        oos = dict(table=pd.DataFrame(rows), start=te.index[0], end=te.index[-1])

    return dict(cols=cols, missing=missing, mu=mu, cov=cov, mu_hist=mu_hist, betas=betas,
                lw=lw, rf=rf, bounds=bounds, frontier=frontier, w_ms=w_ms,
                days=len(rets), start=rets.index[0], oos=oos)


# ---------------- charts ----------------
def frontier_chart(res, w_sel):
    mu, cov, rf = res["mu"], res["cov"], res["rf"]
    vols, rets, _ = res["frontier"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=vols * 100, y=rets * 100, mode="lines", line=dict(width=3),
                             name="Efficient frontier"))
    a_vol = np.sqrt(np.diag(cov)) * 100
    fig.add_trace(go.Scatter(x=a_vol, y=mu * 100, mode="markers+text", text=res["cols"],
                             textposition="top center", marker=dict(size=8, opacity=0.6),
                             name="Individual stocks"))
    _, v_ms, s_ms = stats(res["w_ms"], mu, cov, rf)
    if s_ms > 0:
        x = np.array([0, max(vols.max(), a_vol.max() / 100) * 1.05])
        fig.add_trace(go.Scatter(x=x * 100, y=(rf + s_ms * x) * 100, mode="lines",
                                 line=dict(dash="dot", width=1), name="Capital market line"))
    n = len(res["cols"])
    for label, w, sym in [("Minimum variance", res["frontier"][2][0], "diamond"),
                          ("Maximum Sharpe", res["w_ms"], "star"),
                          ("Equal weight", np.full(n, 1 / n), "square"),
                          ("Your portfolio", w_sel, "circle")]:
        r, v, _ = stats(w, mu, cov, rf)
        fig.add_trace(go.Scatter(x=[v * 100], y=[r * 100], mode="markers", name=label,
                                 marker=dict(size=16 if label == "Your portfolio" else 12,
                                             symbol=sym, line=dict(width=1, color="white"))))
    fig.update_layout(title="Efficient frontier", height=520,
                      xaxis_title="Risk: annual volatility (%)",
                      yaxis_title="Expected annual return (%)")
    return fig


def transition_chart(res, short):
    vols, _, ws = res["frontier"]
    fig = go.Figure()
    for i, t in enumerate(res["cols"]):
        fig.add_trace(go.Scatter(x=vols * 100, y=ws[:, i] * 100, name=t, mode="lines",
                                 stackgroup=None if short else "one"))
    fig.update_layout(title="How the allocation changes as risk increases", height=420,
                      xaxis_title="Risk: annual volatility (%)", yaxis_title="Weight (%)")
    return fig


# ---------------- page ----------------
st.title("⚖️ Portfolio optimizer")
st.caption("Markowitz mean-variance optimization on market data. For education only, "
           "not financial advice.")
with st.expander("How this works"):
    st.markdown(
        "Markowitz optimization finds the mix of stocks with the **highest expected return "
        "for each level of risk**. Plotting those best mixes gives the **efficient "
        "frontier**: no portfolio can sit above it.\n\n"
        "- **Risk** is volatility, calculated from how the stocks moved and moved together.\n"
        "- **Expected returns** are the hardest input to estimate, and the optimizer is very "
        "sensitive to them. Shrinking them toward the average, or using CAPM, gives far more "
        "sensible portfolios than raw historical averages.\n"
        "- **Ledoit-Wolf** shrinkage makes the risk model more stable, especially with many "
        "stocks or short histories.\n"
        "- The **maximum Sharpe** portfolio gives the most return per unit of risk above the "
        "risk-free rate.")

st.subheader("Step 1 · Choose stocks and assumptions")
raw = st.text_input("Tickers (comma-separated, 2 to 20)", DEFAULT_TICKERS)
tickers = list(dict.fromkeys(t.strip().upper() for t in raw.replace(";", ",").split(",")
                             if t.strip()))
c1, c2, c3 = st.columns(3)
lookback = c1.selectbox("History used for estimates", list(LOOKBACKS), index=1)
ret_src = c2.selectbox("Expected returns", RET_SOURCES)
cov_src = c3.selectbox("Risk model", COV_SOURCES)
c4, c5, c6 = st.columns(3)
shrink, premium = 0.5, 0.05
if ret_src.startswith("Shrunk"):
    shrink = c4.slider("Shrinkage toward the average (%)", 0, 100, 50, 5,
                       help="0% = raw historical averages, 100% = every stock gets the "
                            "same expected return") / 100
elif ret_src.startswith("CAPM"):
    premium = c4.number_input("Market risk premium (% per year)", 0.0, 15.0, 5.0, 0.5,
                              help="Extra return expected from the market over the "
                                   "risk-free rate. 4–6% is a common assumption") / 100
else:
    c4.caption("⚠️ Raw historical returns assume the past repeats. Expect concentrated, "
               "unstable portfolios.")
cap = c5.slider("Maximum weight per stock (%)", 5, 100, 40, 5,
                help="Caps concentration. 100% means no limit") / 100
short = c6.checkbox("Allow short selling", False,
                    help="Lets weights go negative (betting on a fall), down to minus "
                         "the maximum weight")

cfg = dict(lookback=lookback, ret=ret_src, cov=cov_src, shrink=shrink, premium=premium,
           cap=cap, short=short)
params = dict(tickers=tuple(tickers), **cfg)

if st.button("▶ Optimize", type="primary"):
    if not 2 <= len(tickers) <= 20:
        st.error("Enter between 2 and 20 tickers.")
    else:
        with st.spinner("Loading prices and optimizing..."):
            try:
                st.session_state["opt"] = dict(params=params, result=compute(tickers, cfg))
            except Exception as e:
                st.error(f"Couldn't optimize: {e}")

saved = st.session_state.get("opt")
if saved is None:
    st.caption("Set your stocks and assumptions, then press Optimize.")
    st.stop()
if saved["params"] != params:
    st.warning("Inputs have changed. Press Optimize to update; showing the previous results.")
res = saved["result"]
if res["missing"]:
    st.warning(f"No price data for {', '.join(res['missing'])}; left out.")
mu, cov, rf, cols = res["mu"], res["cov"], res["rf"], res["cols"]
vols = res["frontier"][0]
st.caption(f"Estimated from {res['days']} trading days since {res['start']:%Y-%m-%d}. "
           f"Risk-free rate {rf:.2%}.")

st.subheader("Step 2 · Pick your risk level")
_, v_ms, _ = stats(res["w_ms"], mu, cov, rf)
lo, hi = float(vols[0] * 100), float(vols[-1] * 100)
c1, c2 = st.columns([3, 1])
if round(hi, 1) - round(lo, 1) < 0.2:
    target = lo
    c1.info("Every stock has the same expected return, so the only efficient portfolio is "
            "the minimum-variance one. Lower the shrinkage to see a range of risk levels.")
else:
    target = c1.slider("Target risk: annual volatility (%)", round(lo, 1), round(hi, 1),
                       round(float(np.clip(v_ms * 100, lo, hi)), 1), 0.1,
                       help="Left end = the lowest-risk portfolio possible. Right end = the "
                            "highest-return portfolio. Starts at the maximum-Sharpe portfolio.")
amount = c2.number_input("Amount to invest ($)", 100.0, 1e9, 10000.0, 1000.0)
w = target_volatility(mu, cov, target / 100, res["bounds"], res["frontier"])
r, v, s = stats(w, mu, cov, rf)

st.subheader("Step 3 · Your optimal allocation")
m = st.columns(4)
m[0].metric("Expected return", f"{r:.1%}")
m[1].metric("Volatility (risk)", f"{v:.1%}")
m[2].metric("Sharpe ratio", f"{s:.2f}")
m[3].metric("Stocks held", f"{int((np.abs(w) > 0.005).sum())} of {len(cols)}")
st.caption(f"A rough one-year range (±1 standard deviation): {r - v:+.1%} to {r + v:+.1%}. "
           "Losses outside this range happen about 1 year in 6.")

order = np.argsort(-w)
left, right = st.columns([1, 1])
with left:
    fig = go.Figure(go.Bar(x=w[order] * 100, y=[cols[i] for i in order], orientation="h",
                           text=[f"{x:.1%}" for x in w[order]], textposition="auto"))
    fig.update_layout(title="Allocation", height=max(260, 32 * len(cols)),
                      xaxis_title="Weight (%)", yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig)
with right:
    table = pd.DataFrame({
        "Ticker": [cols[i] for i in order],
        "Weight": [f"{w[i]:.1%}" for i in order],
        "Amount ($)": [f"{w[i] * amount:,.0f}" for i in order],
        "Expected return": [f"{mu[i]:.1%}" for i in order],
        "Volatility": [f"{np.sqrt(cov[i, i]):.1%}" for i in order]})
    st.dataframe(table, hide_index=True)
    if short and (w < 0).any():
        st.caption("Negative weights are short positions.")

st.plotly_chart(frontier_chart(res, w))
st.caption("Each point on the frontier is the best possible return for that level of risk. "
           "Notice that most individual stocks sit below and to the right of it: "
           "diversification gives more return per unit of risk.")
st.plotly_chart(transition_chart(res, short))

with st.expander("Out-of-sample check: does the optimization hold up?"):
    oos = res["oos"]
    if oos is None:
        st.info("Not enough history for a reliable split. Choose a longer history.")
    else:
        st.markdown(f"Portfolios were optimized using only the first 75% of the history, then "
                    f"tested on **{oos['start']:%Y-%m-%d} to {oos['end']:%Y-%m-%d}**, data the "
                    "optimizer never saw.")
        st.dataframe(oos["table"], hide_index=True)
        st.caption("Optimized portfolios usually look best in-sample and disappoint "
                   "out of sample, especially their predicted returns. Equal weight often "
                   "holds up surprisingly well. Risk estimates tend to be more reliable "
                   "than return estimates.")

with st.expander("Assumptions used"):
    info = pd.DataFrame({"Ticker": cols,
                         "Expected return used": [f"{x:.1%}" for x in mu],
                         "Historical return": [f"{x:.1%}" for x in res["mu_hist"]],
                         "Volatility": [f"{x:.1%}" for x in np.sqrt(np.diag(cov))]})
    if res["betas"] is not None:
        info["Beta vs SPY"] = [f"{b:.2f}" for b in res["betas"]]
    st.dataframe(info, hide_index=True)
    if res["lw"] is not None:
        st.caption(f"Ledoit-Wolf shrinkage intensity: {res['lw']:.1%} toward a "
                   "constant-variance, zero-correlation target.")

st.info("**Keep in mind:** the optimizer assumes the future looks like its estimates. "
        "Expected returns are especially uncertain, so treat the allocation as a starting "
        "point for thinking, not an instruction. It also ignores taxes, trading costs and "
        "rebalancing.")
