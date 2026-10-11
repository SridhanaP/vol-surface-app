"""Markowitz mean-variance optimization (pure NumPy/SciPy, no Streamlit)."""
import numpy as np
from scipy.optimize import minimize

TRADING_DAYS = 252
_OPTS = dict(ftol=1e-12, maxiter=1000)


# ---------------- estimation ----------------
def sample_cov(rets):
    """Annualised sample covariance of daily returns (T x N array)."""
    return np.cov(np.asarray(rets, dtype=float), rowvar=False) * TRADING_DAYS


def ledoit_wolf(rets):
    """Ledoit-Wolf (2004) shrinkage toward a scaled identity.

    Returns (annualised covariance, shrinkage intensity between 0 and 1).
    """
    X = np.asarray(rets, dtype=float)
    X = X - X.mean(0)
    T, N = X.shape
    S = X.T @ X / T
    m = np.trace(S) / N
    d2 = np.sum((S - m * np.eye(N)) ** 2) / N
    b2_bar = (np.sum(np.sum(X ** 2, axis=1) ** 2) - T * np.sum(S ** 2)) / N / T ** 2
    shrink = 0.0 if d2 <= 0 else float(np.clip(min(b2_bar, d2) / d2, 0, 1))
    return (shrink * m * np.eye(N) + (1 - shrink) * S) * TRADING_DAYS, shrink


def shrink_means(mu, strength):
    """Pull each expected return toward the cross-sectional average."""
    mu = np.asarray(mu, dtype=float)
    return (1 - strength) * mu + strength * mu.mean()


def capm_returns(asset_rets, market_rets, rf, premium):
    """Expected return = rf + beta x market risk premium. Returns (mu, betas)."""
    A = np.asarray(asset_rets, dtype=float)
    m = np.asarray(market_rets, dtype=float)
    mc = m - m.mean()
    betas = ((A - A.mean(0)) * mc[:, None]).sum(0) / (mc ** 2).sum()
    return rf + betas * premium, betas


# ---------------- portfolio maths ----------------
def stats(w, mu, cov, rf=0.0):
    ret = float(w @ mu)
    vol = float(np.sqrt(max(w @ cov @ w, 0.0)))
    return ret, vol, (ret - rf) / vol if vol > 0 else np.nan


def make_bounds(n, max_weight, allow_short):
    if max_weight * n < 1 - 1e-9:
        raise ValueError(f"a maximum weight of {max_weight:.0%} is too low for {n} stocks; "
                         f"it must be at least {1 / n:.0%}")
    lo = -max_weight if allow_short else 0.0
    return [(lo, max_weight)] * n


def _sum_to_one():
    return {"type": "eq", "fun": lambda w: w.sum() - 1, "jac": lambda w: np.ones_like(w)}


def _clean(w):
    w = np.where(np.abs(w) < 1e-6, 0.0, w)
    return w / w.sum()


def min_variance(cov, bounds):
    n = len(cov)
    res = minimize(lambda w: w @ cov @ w, np.full(n, 1 / n), jac=lambda w: 2 * cov @ w,
                   bounds=bounds, constraints=[_sum_to_one()], method="SLSQP", options=_OPTS)
    if not res.success:
        raise RuntimeError(f"minimum-variance optimization failed: {res.message}")
    return _clean(res.x)


def max_return(mu, bounds):
    """Exact solution: start at every lower bound, then fill the best stocks first."""
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    w, left = lo.copy(), 1 - lo.sum()
    for i in np.argsort(-np.asarray(mu)):
        add = min(hi[i] - lo[i], left)
        w[i] += add
        left -= add
        if left <= 1e-12:
            break
    return w


def efficient_frontier(mu, cov, bounds, n_points=40):
    """Minimum-variance portfolios for evenly spaced target returns.

    Returns (vols, rets, weights) arrays running from the minimum-variance
    portfolio to the maximum-return portfolio.
    """
    w_mv, w_mx = min_variance(cov, bounds), max_return(mu, bounds)
    r_lo, r_hi = float(w_mv @ mu), float(w_mx @ mu)
    ws = [w_mv]
    if r_hi - r_lo > 1e-9:                     # otherwise every stock has the same return
        x0 = w_mv
        for target in np.linspace(r_lo, r_hi, n_points)[1:]:
            cons = [_sum_to_one(),
                    {"type": "eq", "fun": lambda w, t=target: w @ mu - t,
                     "jac": lambda w: mu}]
            res = minimize(lambda w: w @ cov @ w, x0, jac=lambda w: 2 * cov @ w,
                           bounds=bounds, constraints=cons, method="SLSQP", options=_OPTS)
            if res.success:
                ws.append(_clean(res.x))
                x0 = res.x
        if float(ws[-1] @ mu) < r_hi - 1e-6:      # make sure the top end is included
            ws.append(w_mx)
    ws = np.array(ws)
    rets = ws @ mu
    vols = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", ws, cov, ws), 0))
    return vols, rets, ws


def max_sharpe(mu, cov, rf, bounds, frontier):
    """Tangency portfolio. Starts from the best frontier point, then refines."""
    vols, rets, ws = frontier
    x0 = ws[int(np.nanargmax((rets - rf) / vols))]
    f = lambda w: -(w @ mu - rf) / np.sqrt(w @ cov @ w)
    res = minimize(f, x0, bounds=bounds, constraints=[_sum_to_one()], method="SLSQP",
                   options=_OPTS)
    w = _clean(res.x) if res.success and f(res.x) <= f(x0) + 1e-12 else x0
    return w


def target_volatility(mu, cov, target, bounds, frontier):
    """Highest-return portfolio whose volatility does not exceed `target`."""
    vols, rets, ws = frontier
    if target <= vols[0] + 1e-9:
        return ws[0]
    if target >= vols[-1] - 1e-9:
        return ws[-1]
    best = int(np.max(np.where(vols <= target)[0]))       # warm start, always feasible
    cons = [_sum_to_one(),
            {"type": "ineq", "fun": lambda w: target ** 2 - w @ cov @ w,
             "jac": lambda w: -2 * cov @ w}]
    res = minimize(lambda w: -(w @ mu), ws[best], jac=lambda w: -mu, bounds=bounds,
                   constraints=cons, method="SLSQP", options=_OPTS)
    if res.success and np.sqrt(res.x @ cov @ res.x) <= target + 1e-6:
        return _clean(res.x)
    return ws[best]


def realized(simple_rets, w, rf):
    """Annualised return, volatility and Sharpe of a daily-rebalanced portfolio."""
    p = np.asarray(simple_rets, dtype=float) @ w
    ret, vol = p.mean() * TRADING_DAYS, p.std(ddof=1) * np.sqrt(TRADING_DAYS)
    return ret, vol, (ret - rf) / vol if vol > 0 else np.nan
