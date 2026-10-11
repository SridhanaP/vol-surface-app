"""Pricing and simulation models (pure NumPy/SciPy, no Streamlit)."""
import numpy as np
from scipy.stats import norm

TRADING_DAYS = 252


# ---------------- Black-Scholes(-Merton) ----------------
def bsm(S, K, T, r, q, sigma, kind="call"):
    """Price and Greeks. Works with scalar or array S.

    vega: per 1 vol point, theta: per calendar day, rho: per 1% rate move.
    """
    S = np.asarray(S, dtype=float)
    sq = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / sq
    d2 = d1 - sq
    dq, dr = np.exp(-q * T), np.exp(-r * T)
    pdf = norm.pdf(d1)
    common_theta = -S * dq * pdf * sigma / (2 * np.sqrt(T))
    if kind == "call":
        price = S * dq * norm.cdf(d1) - K * dr * norm.cdf(d2)
        delta = dq * norm.cdf(d1)
        theta = common_theta - r * K * dr * norm.cdf(d2) + q * S * dq * norm.cdf(d1)
        rho = K * T * dr * norm.cdf(d2)
    else:
        price = K * dr * norm.cdf(-d2) - S * dq * norm.cdf(-d1)
        delta = dq * (norm.cdf(d1) - 1)
        theta = common_theta + r * K * dr * norm.cdf(-d2) - q * S * dq * norm.cdf(-d1)
        rho = -K * T * dr * norm.cdf(-d2)
    return dict(price=price, delta=delta, gamma=dq * pdf / (S * sq),
                vega=S * dq * pdf * np.sqrt(T) / 100, theta=theta / 365,
                rho=rho / 100)


# ---------------- Binomial (Cox-Ross-Rubinstein) ----------------
def binomial(S, K, T, r, q, sigma, kind="call", steps=500, american=True):
    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)
    if not 0 < p < 1:
        raise ValueError("Too few steps for these inputs; increase steps")
    disc = np.exp(-r * dt)
    payoff = (lambda x: np.maximum(x - K, 0.0)) if kind == "call" else \
             (lambda x: np.maximum(K - x, 0.0))
    V = payoff(S * u ** np.arange(steps, -steps - 1, -2.0))
    for i in range(steps - 1, -1, -1):
        V = disc * (p * V[:-1] + (1 - p) * V[1:])
        if american:
            V = np.maximum(V, payoff(S * u ** np.arange(i, -i - 1, -2.0)))
    return float(V[0])


# ---------------- estimation ----------------
def estimate_params(prices):
    """Daily log returns, annual volatility and annual expected return."""
    rets = np.log(prices).diff().dropna()
    sigma = rets.std() * np.sqrt(TRADING_DAYS)
    mu = rets.mean() * TRADING_DAYS + 0.5 * sigma ** 2
    return rets, sigma, mu


def safe_cholesky(corr):
    """Cholesky factor, nudging the matrix to positive-definite if needed."""
    corr = np.asarray(corr, dtype=float)
    try:
        return np.linalg.cholesky(corr + 1e-10 * np.eye(len(corr)))
    except np.linalg.LinAlgError:
        w, v = np.linalg.eigh(corr)
        fixed = v @ np.diag(np.clip(w, 1e-6, None)) @ v.T
        dinv = 1 / np.sqrt(np.diag(fixed))
        return np.linalg.cholesky(fixed * np.outer(dinv, dinv))


# ---------------- simulations ----------------
def simulate_gbm(S0, mu, sigma, days, n_paths, seed=None):
    """Price paths, shape (n_paths, days + 1). mu, sigma are annual."""
    rng = np.random.default_rng(seed)
    dt = 1 / TRADING_DAYS
    z = rng.standard_normal((n_paths, days))
    inc = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z
    log_paths = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(inc, axis=1)], axis=1)
    return S0 * np.exp(log_paths)


def simulate_portfolio(amounts, mu, sigma, corr, days, n_sims, dist="normal",
                       nu=5, hist_returns=None, seed=None):
    """Buy-and-hold portfolio values, shape (n_sims, days + 1).

    dist: "normal", "student_t" (fat tails, correlated) or "bootstrap"
    (resample whole historical days, rescaled to the chosen vol and drift).
    """
    rng = np.random.default_rng(seed)
    amounts, mu, sigma = (np.asarray(a, dtype=float) for a in (amounts, mu, sigma))
    m, dt = len(amounts), 1 / TRADING_DAYS
    drift = (mu - 0.5 * sigma ** 2) * dt
    L = safe_cholesky(corr)
    if dist == "bootstrap":
        h = np.asarray(hist_returns, dtype=float)
        scaled = (h - h.mean(0)) * (sigma * np.sqrt(dt) / h.std(0, ddof=1))

    log_s = np.zeros((n_sims, m))
    values = np.empty((n_sims, days + 1))
    values[:, 0] = amounts.sum()
    for t in range(days):
        if dist == "bootstrap":
            shock = scaled[rng.integers(0, len(scaled), n_sims)]
        else:
            z = rng.standard_normal((n_sims, m)) @ L.T
            if dist == "student_t":
                w = rng.chisquare(nu, size=(n_sims, 1))
                z = z / np.sqrt(w / nu) * np.sqrt((nu - 2) / nu)  # unit variance
            shock = sigma * np.sqrt(dt) * z
        log_s += drift + shock
        values[:, t + 1] = (amounts * np.exp(log_s)).sum(1)
    return values


def var_cvar(final_values, start_value, confidence=0.95):
    """Value at Risk and Conditional VaR (expected shortfall), as positive losses."""
    cutoff = np.quantile(final_values, 1 - confidence)
    tail = final_values[final_values <= cutoff]
    return float(start_value - cutoff), float(start_value - tail.mean())
