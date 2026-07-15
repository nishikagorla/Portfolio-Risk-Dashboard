from __future__ import annotations

import numpy as np
import pandas as pd

from .config import TRADING_DAYS


def annualized_return(returns, periods=TRADING_DAYS):
    n = returns.shape[0]
    if n == 0:
        return np.nan
    growth = (1.0 + returns).prod()
    cagr =  growth ** (periods / n) - 1.0
    return cagr


def annualized_volatility(returns, periods=TRADING_DAYS):
    daily_vol = returns.std(ddof=1)
    annualized_vol = daily_vol * np.sqrt(periods)
    return annualized_vol


def sharpe_ratio(returns, risk_free=0.0, periods=TRADING_DAYS):
    vol = annualized_volatility(returns, periods)
    if vol == 0 or np.isnan(vol):
        return np.nan
    sharpe = (annualized_return(returns, periods) - risk_free) / vol
    return sharpe


def max_drawdown(returns):
    wealth = (1.0 + returns).cumprod()
    running_peak = wealth.cummax()
    drawdown = wealth / running_peak - 1.0
    max_drawdown = drawdown.min()
    return max_drawdown


def beta(portfolio, benchmark):
    joined = pd.concat([portfolio, benchmark], axis=1).dropna()
    if joined.shape[0] < 2:
        return np.nan
    cov = np.cov(joined.iloc[:, 0], joined.iloc[:, 1])
    benchmark_var = cov[1, 1] 
    if benchmark_var == 0:
        return np.nan
    beta = cov[0, 1] / benchmark_var
    return beta


def summary(returns, benchmark=None, risk_free=0.0, periods=TRADING_DAYS):
    stats = {
        "Annualized Return": annualized_return(returns, periods),
        "Annualized Volatility": annualized_volatility(returns, periods),
        "Sharpe Ratio": sharpe_ratio(returns, risk_free, periods),
        "Max Drawdown": max_drawdown(returns),
    }
    if benchmark is not None:
        stats["Beta vs. Benchmark"] = beta(returns, benchmark)
    return stats
