from __future__ import annotations

import numpy as np
import pandas as pd

# Historical stress scenarios. Each carries the crisis window (for replaying the
# portfolio's actual behavior when the selected date range covers it) and the
# approximate S&P 500 peak-to-trough drawdown (for a single-factor, beta-scaled
# estimate that works even when the window predates some holdings).
SCENARIOS = {
    "2008 Financial Crisis": {"window": ("2007-10-09", "2009-03-09"), "market_shock": -0.568},
    "2020 COVID Crash": {"window": ("2020-02-19", "2020-03-23"), "market_shock": -0.339},
    "2022 Bear Market": {"window": ("2022-01-03", "2022-10-12"), "market_shock": -0.254},
    "2018 Q4 Selloff": {"window": ("2018-09-20", "2018-12-24"), "market_shock": -0.196},
}


def _normalize(weights):
    w = np.asarray(weights, dtype=float)
    total = w.sum()
    if total == 0:
        raise ValueError("Weights sum to zero.")
    return w / total


def replay(asset_returns_window, weights):
    """Actual portfolio cumulative return and worst drawdown over a window of
    per-asset daily returns. Returns (cumulative_return, max_drawdown), both as
    signed fractions (e.g. -0.31). Empty window -> (nan, nan).
    """
    if asset_returns_window.shape[0] == 0:
        return np.nan, np.nan
    w = _normalize(weights)
    port = asset_returns_window.to_numpy() @ w
    wealth = np.cumprod(1.0 + port)
    cumulative = float(wealth[-1] - 1.0)
    drawdown = float((wealth / np.maximum.accumulate(wealth) - 1.0).min())
    return cumulative, drawdown


def beta_shock(beta, market_shock):
    """Single-factor estimate of the portfolio move: beta x market move.

    A first-order approximation only -- it assumes a linear, constant-beta
    relationship and ignores idiosyncratic moves, but it applies to any
    portfolio regardless of how far back its holdings trade.
    """
    return float(beta) * float(market_shock)
