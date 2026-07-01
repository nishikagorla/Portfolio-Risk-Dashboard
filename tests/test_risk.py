import numpy as np
import pandas as pd

from src import risk


def _normal_returns(n=100_000, mu=0.0, sigma=0.01, seed=0):
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(mu, sigma, n))


def test_historical_var_matches_definition():
    # Historical VaR must equal the negated (1 - alpha) percentile. This pins
    # down the sign convention and that alpha maps to the right percentile.
    r = _normal_returns()
    assert np.isclose(risk.historical_var(r, 0.95), -np.percentile(r, 5))


def test_parametric_var_normal_sample():
    # For a clean normal sample, parametric VaR at 95% is about 1.645 * sigma
    # (mean ~ 0). 1.645 is -norm.ppf(0.05).
    sigma = 0.01
    r = _normal_returns(sigma=sigma)
    assert np.isclose(risk.parametric_var(r, 0.95), 1.645 * sigma, rtol=0.05)


def test_es_at_least_var():
    # Expected Shortfall averages the tail beyond VaR, so it can never be
    # smaller than VaR.
    r = _normal_returns(seed=3)
    assert risk.expected_shortfall(r, 0.95) >= risk.historical_var(r, 0.95)


def test_var_increases_with_confidence():
    # A higher confidence level looks further into the tail -> larger loss.
    r = _normal_returns(seed=7)
    assert risk.historical_var(r, 0.99) >= risk.historical_var(r, 0.95)


def test_monte_carlo_close_to_parametric_for_normal():
    # Single normal asset: the MC generator is normal, so MC VaR should land
    # near parametric VaR. This confirms the simulation + weighting is wired up.
    sigma = 0.01
    asset_returns = pd.DataFrame({"A": _normal_returns(sigma=sigma, seed=1)})
    mc = risk.monte_carlo_var(asset_returns, [1.0], alpha=0.95, n_sims=200_000, seed=1)
    param = risk.parametric_var(asset_returns["A"], 0.95)
    assert np.isclose(mc, param, rtol=0.05)


def test_monte_carlo_weights_normalized():
    # Weights that don't sum to 1 should be normalized, not crash.
    r1 = _normal_returns(sigma=0.01, seed=10)
    r2 = _normal_returns(sigma=0.02, seed=11)
    asset_returns = pd.DataFrame({"A": r1, "B": r2})
    out = risk.monte_carlo_var(asset_returns, [2.0, 2.0], alpha=0.95, seed=5)
    assert out > 0
