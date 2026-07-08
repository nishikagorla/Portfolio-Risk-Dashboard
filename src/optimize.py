from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

from .config import TRADING_DAYS

# Mean-variance optimization (Markowitz). Everything works on ANNUALIZED moments
# and uses arithmetic annualization (mean * periods, cov * periods): variance
# scales linearly with time, so this keeps the quadratic form w'Cw consistent.
# All portfolios here are long-only and fully invested (weights >= 0, sum to 1)
# unless allow_short is set.


def annualized_moments(asset_returns, periods=TRADING_DAYS, shrink=False):
    mu = asset_returns.mean().to_numpy() * periods # average annual return for each asset
    if shrink:
        lw = LedoitWolf().fit(asset_returns.to_numpy()) # reduces noise
        cov = lw.covariance_ * periods
    else:
        cov = asset_returns.cov().to_numpy() * periods
    return mu, cov


def portfolio_performance(weights, mu, cov, risk_free=0.0):
    w = np.asarray(weights, dtype=float)
    ret = float(w @ mu) 
    vol = float(np.sqrt(w @ cov @ w)) 
    sharpe = (ret - risk_free) / vol if vol > 0 else np.nan
    return ret, vol, sharpe


def _bounds_and_constraints(n_assets, allow_short=False):
    lower = None if allow_short else 0.0
    bounds = tuple((lower, 1.0) for _ in range(n_assets))
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    return bounds, constraints


def min_variance(mu, cov, allow_short=False):
    # weights of global minimum-variance portfolio
    n = len(mu)
    bounds, constraints = _bounds_and_constraints(n, allow_short)
    x0 = np.repeat(1.0 / n, n)
    res = minimize(
        lambda w: w @ cov @ w, x0, method="SLSQP",
        bounds=bounds, constraints=constraints,
    )
    return res.x


def max_sharpe(mu, cov, risk_free=0.0, allow_short=False):
    # weights of maximum-Sharpe (tangency) portfolio
    n = len(mu)
    bounds, constraints = _bounds_and_constraints(n, allow_short)
    x0 = np.repeat(1.0 / n, n)

    def neg_sharpe(w):
        vol = np.sqrt(w @ cov @ w)
        if vol == 0:
            return 1e6
        return -(w @ mu - risk_free) / vol

    res = minimize(
        neg_sharpe, x0, method="SLSQP",
        bounds=bounds, constraints=constraints,
    )
    return res.x


def efficient_frontier(mu, cov, n_points=50, allow_short=False):
    """Trace the frontier: for a grid of target returns, find the minimum-
    variance portfolio achieving each. Returns (vols, rets, weights) arrays.
    """
    n = len(mu)
    bounds, base_constraints = _bounds_and_constraints(n, allow_short)
    x0 = np.repeat(1.0 / n, n)

    # min-variance portfolio's return to single best-returning asset
    r_min = float(min_variance(mu, cov, allow_short) @ mu)
    r_max = float(mu.max())
    if r_max <= r_min:
        r_max = r_min + 1e-6
    targets = np.linspace(r_min, r_max, n_points)

    vols, rets, weights = [], [], []
    for target in targets:
        constraints = base_constraints + [
            {"type": "eq", "fun": lambda w, t=target: w @ mu - t}
        ]
        res = minimize(
            lambda w: w @ cov @ w, x0, method="SLSQP",
            bounds=bounds, constraints=constraints,
        )
        if res.success:
            w = res.x
            vols.append(float(np.sqrt(w @ cov @ w)))
            rets.append(float(w @ mu))
            weights.append(w)

    return np.array(vols), np.array(rets), np.array(weights)
