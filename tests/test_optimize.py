"""Tests for src/optimize.py.

Synthetic data with properties we can check by hand, so they run fast and don't
touch the network. Run with:  pytest
"""

import numpy as np
import pandas as pd

from src import optimize


def _returns(n=750, seed=0):
    # Three assets with different means/vols and mild correlation.
    rng = np.random.default_rng(seed)
    a = rng.normal(0.0006, 0.012, n)
    b = rng.normal(0.0004, 0.009, n)
    c = 0.3 * a + rng.normal(0.0003, 0.011, n)  # correlated with a
    return pd.DataFrame({"A": a, "B": b, "C": c})


def test_moments_shapes():
    r = _returns()
    mu, cov = optimize.annualized_moments(r, periods=252)
    assert mu.shape == (3,)
    assert cov.shape == (3, 3)


def test_shrinkage_covariance_symmetric_psd():
    r = _returns()
    _, cov = optimize.annualized_moments(r, periods=252, shrink=True)
    assert np.allclose(cov, cov.T)                       # symmetric
    assert np.all(np.linalg.eigvalsh(cov) > -1e-10)      # positive semidefinite


def test_min_variance_weights_valid():
    mu, cov = optimize.annualized_moments(_returns(), periods=252)
    w = optimize.min_variance(mu, cov)
    assert np.isclose(w.sum(), 1.0)          # fully invested
    assert np.all(w >= -1e-8)                # long-only


def test_min_variance_beats_equal_weight():
    mu, cov = optimize.annualized_moments(_returns(), periods=252)
    w_mv = optimize.min_variance(mu, cov)
    equal = np.repeat(1 / 3, 3)
    _, vol_mv, _ = optimize.portfolio_performance(w_mv, mu, cov)
    _, vol_eq, _ = optimize.portfolio_performance(equal, mu, cov)
    assert vol_mv <= vol_eq + 1e-9           # by definition min variance is lowest


def test_max_sharpe_weights_valid_and_best():
    mu, cov = optimize.annualized_moments(_returns(), periods=252)
    w_ms = optimize.max_sharpe(mu, cov, risk_free=0.0)
    equal = np.repeat(1 / 3, 3)
    assert np.isclose(w_ms.sum(), 1.0)
    assert np.all(w_ms >= -1e-8)
    _, _, s_ms = optimize.portfolio_performance(w_ms, mu, cov)
    _, _, s_eq = optimize.portfolio_performance(equal, mu, cov)
    assert s_ms >= s_eq - 1e-9               # no worse than equal weight


def test_frontier_monotonic_and_valid():
    mu, cov = optimize.annualized_moments(_returns(), periods=252)
    vols, rets, weights = optimize.efficient_frontier(mu, cov, n_points=25)
    assert len(vols) == len(rets) == len(weights)
    assert np.all(vols > 0)
    # target returns were built increasing, so the achieved returns increase too
    assert np.all(np.diff(rets) >= -1e-8)
    # every frontier portfolio is fully invested
    assert np.allclose(weights.sum(axis=1), 1.0)


def test_performance_hand_computed():
    # Two uncorrelated assets, equal weight: return and vol are checkable by hand.
    mu = np.array([0.10, 0.20])
    cov = np.array([[0.04, 0.0], [0.0, 0.09]])   # vols 0.2 and 0.3
    w = np.array([0.5, 0.5])
    ret, vol, sharpe = optimize.portfolio_performance(w, mu, cov, risk_free=0.0)
    assert np.isclose(ret, 0.15)
    assert np.isclose(vol, np.sqrt(0.25 * 0.04 + 0.25 * 0.09))
    assert np.isclose(sharpe, 0.15 / vol)
