"""Tests for src/stress.py and src/backtest.py.

Synthetic, deterministic, no network. Run with:  pytest
"""

import numpy as np
import pandas as pd

from src import stress, backtest


# --- stress ----------------------------------------------------------------

def _window_returns(values_by_asset):
    idx = pd.date_range("2020-02-19", periods=len(next(iter(values_by_asset.values()))), freq="B")
    return pd.DataFrame(values_by_asset, index=idx)


def test_replay_known_path():
    # One asset: +10% then -50% -> cumulative 1.1*0.5 - 1 = -0.45, drawdown -0.50.
    w = _window_returns({"A": [0.10, -0.50]})
    cum, dd = stress.replay(w, [1.0])
    assert np.isclose(cum, -0.45)
    assert np.isclose(dd, -0.50)


def test_replay_empty_window():
    w = pd.DataFrame({"A": []})
    cum, dd = stress.replay(w, [1.0])
    assert np.isnan(cum) and np.isnan(dd)


def test_beta_shock_linear():
    assert np.isclose(stress.beta_shock(1.2, -0.30), -0.36)
    assert np.isclose(stress.beta_shock(0.5, -0.20), -0.10)


# --- backtest --------------------------------------------------------------

def _normal_series(n=2000, sigma=0.01, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2015-01-01", periods=n, freq="B")
    return pd.Series(rng.normal(0.0, sigma, n), index=idx)


def test_backtest_breach_rate_near_expected():
    # For well-behaved normal data, a 95% VaR should be breached ~5% of the time.
    r = _normal_series()
    bt = backtest.var_backtest(r, alpha=0.95, window=250, method="historical")
    rate = bt["breach"].mean()
    assert 0.02 < rate < 0.09  # loose band around 5%


def test_backtest_columns_and_length():
    r = _normal_series(n=600)
    bt = backtest.var_backtest(r, alpha=0.95, window=250)
    assert list(bt.columns) == ["var", "ret", "breach"]
    assert len(bt) == 600 - 250


def test_kupiec_correct_model_not_rejected():
    # Exactly the expected number of breaches -> LR ~ 0, do not reject.
    kp = backtest.kupiec_pof(n_obs=1000, n_breaches=50, alpha=0.95)
    assert np.isclose(kp["lr_stat"], 0.0, atol=1e-6)
    assert kp["reject_model"] is False


def test_kupiec_too_many_breaches_rejected():
    # 15% breaches when 5% expected -> clearly reject.
    kp = backtest.kupiec_pof(n_obs=1000, n_breaches=150, alpha=0.95)
    assert kp["reject_model"] is True
    assert kp["p_value"] < 0.05


def test_kupiec_zero_breaches_finite():
    # Edge case: no breaches must not blow up (0*log0 handled).
    kp = backtest.kupiec_pof(n_obs=500, n_breaches=0, alpha=0.95)
    assert np.isfinite(kp["lr_stat"])
    assert kp["breach_rate"] == 0.0
