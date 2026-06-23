import numpy as np
import pandas as pd

from src import metrics


def _series(values):
    idx = pd.date_range("2024-01-01", periods=len(values), freq="B")
    return pd.Series(values, index=idx)


def test_annualized_return_constant():
    # A constant 0.1% daily return over 252 days should compound to roughly
    # 1.001 ** 252 - 1.
    r = _series([0.001] * 252)
    expected = 1.001 ** 252 - 1
    assert np.isclose(metrics.annualized_return(r), expected, rtol=1e-6)


def test_volatility_scales_with_sqrt_time():
    # Daily std times sqrt(252) is the annualization rule.
    r = _series(np.random.default_rng(0).normal(0, 0.01, 252))
    expected = r.std(ddof=1) * np.sqrt(252)
    assert np.isclose(metrics.annualized_volatility(r), expected)


def test_max_drawdown_known_path():
    # Up 10%, then down 50%: wealth goes 1.0 -> 1.1 -> 0.55, a 50% drawdown.
    r = _series([0.10, -0.50])
    assert np.isclose(metrics.max_drawdown(r), -0.50)


def test_beta_of_series_with_itself_is_one():
    r = _series(np.random.default_rng(1).normal(0, 0.01, 100))
    assert np.isclose(metrics.beta(r, r), 1.0)


def test_sharpe_zero_vol_is_nan():
    r = _series([0.0] * 50)
    assert np.isnan(metrics.sharpe_ratio(r))
