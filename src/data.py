# data layer
from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf


def load_prices(tickers, start, end=None):
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )
    if raw is None or raw.empty:
        raise ValueError(
            "No data returned. Check the ticker symbols and date range."
        )

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"].copy()
    else:
        prices = raw[["Close"]].copy()
        prices.columns = tickers if isinstance(tickers, list) else [tickers]

    prices = prices.ffill().dropna()

    if prices.empty:
        raise ValueError("Prices were all NaN after cleaning. Check inputs.")

    return prices


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna()


def portfolio_returns(asset_returns: pd.DataFrame, weights: list[float]) -> pd.Series:
    w = np.asarray(weights, dtype=float)
    if w.sum() == 0:
        raise ValueError("Weights sum to zero.")
    w = w / w.sum()

    if asset_returns.shape[1] != w.shape[0]:
        raise ValueError(
            f"Got {asset_returns.shape[1]} return columns but {w.shape[0]} weights."
        )

    daily = asset_returns.to_numpy() @ w
    return pd.Series(daily, index=asset_returns.index, name="Portfolio")
