"""Plotly figures for the app."""
import plotly.graph_objects as go


def surface_figure(pts, K, D, IV, title, show_points=True):
    fig = go.Figure()
    fig.add_trace(go.Surface(
        x=K, y=D, z=IV * 100, colorscale="Viridis", opacity=0.92,
        colorbar=dict(title="IV %"), name="surface",
        hovertemplate="K/F %{x:.3f}<br>%{y:.0f} days<br>IV %{z:.1f}%<extra></extra>"))
    if show_points:
        fig.add_trace(go.Scatter3d(
            x=pts.moneyness, y=pts.days, z=pts.iv * 100, mode="markers",
            marker=dict(size=2, color="black"), name="market quotes",
            text=[f"{e} {t} K={k:g}" for e, t, k in zip(pts.expiry, pts.type, pts.strike)],
            hovertemplate="%{text}<br>IV %{z:.1f}%<extra></extra>"))
    fig.update_layout(
        title=title, height=720, margin=dict(l=0, r=0, t=50, b=0),
        scene=dict(xaxis_title="Moneyness K/F", yaxis_title="Days to expiry",
                   zaxis_title="IV (%)"))
    return fig


def skew_figure(pts, expiries):
    fig = go.Figure()
    for exp in expiries:
        g = pts[pts.expiry == exp].sort_values("moneyness")
        if g.empty:
            continue
        fig.add_trace(go.Scatter(
            x=g.moneyness, y=g.iv * 100, mode="lines+markers",
            marker=dict(size=4), name=f"{exp} ({g.days.iloc[0]:.0f}d)"))
    fig.add_vline(x=1.0, line_dash="dot", opacity=0.5)
    fig.update_layout(title="Volatility skew by expiry", height=500,
                      xaxis_title="Moneyness K/F", yaxis_title="IV (%)")
    return fig


def term_figure(atm):
    fig = go.Figure(go.Scatter(
        x=atm.days, y=atm.atm_iv * 100, mode="lines+markers", text=atm.expiry,
        hovertemplate="%{text}<br>%{x:.0f} days<br>ATM IV %{y:.1f}%<extra></extra>"))
    fig.update_layout(title="ATM implied volatility term structure", height=480,
                      xaxis_title="Days to expiry", yaxis_title="ATM IV (%)")
    return fig


def history_figure(hist, ticker):
    fig = go.Figure()
    for col, label in [("atm_iv_7d", "7-day"), ("atm_iv_30d", "30-day"),
                       ("atm_iv_90d", "90-day")]:
        if col in hist:
            fig.add_trace(go.Scatter(x=hist.date, y=hist[col] * 100,
                                     mode="lines+markers", name=label))
    fig.update_layout(title=f"{ticker} ATM implied volatility over time",
                      height=480, xaxis_title="Date", yaxis_title="ATM IV (%)")
    return fig
