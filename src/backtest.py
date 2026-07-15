from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm

# VaR backtesting = checking a VaR model against what actually happened. We roll
# a window forward, estimate one-day VaR from the trailing window, and count the
# days the realized loss exceeded it (a "breach"). A correct 95% VaR should be
# breached about 5% of the time; the Kupiec POF test says whether the observed
# breach rate differs from that by more than chance.


def var_backtest(returns, alpha=0.95, window=250, method="historical"):
    """Walk forward and flag VaR breaches.

    Returns a DataFrame indexed by date with columns:
      var    -- the one-day VaR estimated from the trailing window (positive loss)
      ret    -- the realized return that day
      breach -- True if the loss exceeded VaR (ret < -var)
    """
    values = np.asarray(returns)
    index = returns.index if isinstance(returns, pd.Series) else pd.RangeIndex(len(values))
    z = norm.ppf(1.0 - alpha)

    records = []
    for t in range(window, len(values)):
        hist = values[t - window:t]
        if method == "historical":
            var = -np.percentile(hist, (1.0 - alpha) * 100.0)
        elif method == "parametric":
            var = -(hist.mean() + z * hist.std(ddof=1))
        else:
            raise ValueError(f"Unknown method: {method!r}")
        actual = values[t]
        records.append((index[t], var, actual, bool(actual < -var)))

    return pd.DataFrame(
        records, columns=["date", "var", "ret", "breach"]
    ).set_index("date")


def kupiec_pof(n_obs, n_breaches, alpha=0.95):
    """Kupiec proportion-of-failures test.

    Compares the observed breach rate to the expected rate (1 - alpha) with a
    likelihood-ratio statistic that is chi-square(1) under the null "the model
    is correctly calibrated." A large statistic / small p-value means reject.
    """
    p = 1.0 - alpha
    n = int(n_obs)
    x = int(n_breaches)
    pi_hat = x / n if n else 0.0

    # Log-likelihood under the null rate p and under the observed rate pi_hat.
    # At x = 0 or x = n the observed-rate term collapses to 0 (0*log0 := 0).
    ll_null = (n - x) * np.log(1.0 - p) + x * np.log(p)
    if x == 0 or x == n:
        ll_obs = 0.0
    else:
        ll_obs = (n - x) * np.log(1.0 - pi_hat) + x * np.log(pi_hat)

    lr = -2.0 * (ll_null - ll_obs)
    return {
        "n_obs": n,
        "breaches": x,
        "expected_breaches": p * n,
        "breach_rate": pi_hat,
        "expected_rate": p,
        "lr_stat": float(lr),
        "p_value": float(chi2.sf(lr, df=1)),
        "reject_model": bool(lr > chi2.ppf(0.95, df=1)),  # 5% level, crit ~ 3.841
    }
