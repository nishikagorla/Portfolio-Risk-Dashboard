from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import config, metrics
from src.data import compute_returns, load_prices, portfolio_returns

st.set_page_config(page_title="Portfolio Risk Dashboard", layout="wide")


# cache data
@st.cache_data(ttl=3600, show_spinner=False)
def get_prices(tickers: list[str], benchmark: str, start: str) -> pd.DataFrame:
    all_symbols = list(dict.fromkeys(tickers + [benchmark]))  # de-dupe, keep order
    return load_prices(all_symbols, start=start)


# sidebar inputs
st.sidebar.header("Portfolio")

tickers_raw = st.sidebar.text_input(
    "Tickers (comma-separated)", ", ".join(config.DEFAULT_TICKERS)
)
tickers = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()]

weights_raw = st.sidebar.text_input(
    "Weights (comma-separated)", ", ".join(str(w) for w in config.DEFAULT_WEIGHTS)
)
try:
    weights = [float(w) for w in weights_raw.split(",") if w.strip()]
except ValueError:
    st.sidebar.error("Weights must be numbers.")
    st.stop()

benchmark = st.sidebar.text_input("Benchmark", config.DEFAULT_BENCHMARK).strip().upper()
start = st.sidebar.text_input("Start date (YYYY-MM-DD)", config.DEFAULT_START).strip()
risk_free = st.sidebar.number_input(
    "Risk-free rate (annual)", value=config.RISK_FREE_RATE, step=0.005, format="%.3f"
)

if len(tickers) != len(weights):
    st.sidebar.error(
        f"You entered {len(tickers)} tickers but {len(weights)} weights."
    )
    st.stop()
if not tickers:
    st.sidebar.error("Add at least one ticker.")
    st.stop()

st.sidebar.caption(
    "Weights get normalized to sum to 1."
)


# main panel
st.title("Portfolio Risk Dashboard")

try:
    prices = get_prices(tickers, benchmark, start)
except Exception as exc:  # noqa: BLE001, surface any data error to the user
    st.error(f"Could not load data: {exc}")
    st.stop()

returns = compute_returns(prices)

asset_cols = [t for t in tickers if t in returns.columns]
missing = [t for t in tickers if t not in returns.columns]
if missing:
    st.warning(f"No data for: {', '.join(missing)}. They were skipped.")
if not asset_cols:
    st.error("None of the tickers returned data.")
    st.stop()

weight_map = dict(zip(tickers, weights))
aligned_weights = [weight_map[t] for t in asset_cols]

asset_returns = returns[asset_cols]
port_returns = portfolio_returns(asset_returns, aligned_weights)
bench_returns = returns[benchmark] if benchmark in returns.columns else None


# metrics 
st.subheader("Headline Metrics")
stats = metrics.summary(port_returns, bench_returns, risk_free)

cols = st.columns(len(stats))
for col, (name, value) in zip(cols, stats.items()):
    if "Return" in name or "Volatility" in name or "Drawdown" in name:
        display = f"{value:.2%}"
    else:
        display = f"{value:.2f}"
    col.metric(name, display)

# cumulative performance vs benchmark 
st.subheader("Cumulative Return")
wealth = (1.0 + port_returns).cumprod() - 1.0
perf = pd.DataFrame({"Portfolio": wealth})
if bench_returns is not None:
    perf[benchmark] = (1.0 + bench_returns).cumprod() - 1.0

fig_perf = go.Figure()
for column in perf.columns:
    fig_perf.add_trace(
        go.Scatter(x=perf.index, y=perf[column], name=column, mode="lines")
    )
fig_perf.update_layout(
    yaxis_tickformat=".0%",
    hovermode="x unified",
    margin=dict(l=0, r=0, t=10, b=0),
    legend=dict(orientation="h"),
)
st.plotly_chart(fig_perf, width='stretch')

# correlation heatmap
st.subheader("Correlation of Daily Returns")
corr = asset_returns.corr()
fig_corr = px.imshow(
    corr,
    text_auto=".2f",
    color_continuous_scale="RdBu_r",
    zmin=-1,
    zmax=1,
    aspect="auto",
)
fig_corr.update_layout(margin=dict(l=0, r=0, t=10, b=0))
st.plotly_chart(fig_corr, width='stretch')

# per-asset breakdown 
st.subheader("Per-Asset Annualized Metrics")
per_asset = pd.DataFrame(
    {
        "Weight": pd.Series(weight_map),
        "Annualized Return": asset_returns.apply(metrics.annualized_return, periods=config.TRADING_DAYS),
        "Annualized Volatility": asset_returns.apply(metrics.annualized_volatility, periods=config.TRADING_DAYS),
        "Sharpe": asset_returns.apply(
            lambda s: metrics.sharpe_ratio(s, risk_free, config.TRADING_DAYS)
        ),
    }
).loc[asset_cols]
per_asset["Weight"] = per_asset["Weight"] / per_asset["Weight"].sum()

st.dataframe(
    per_asset.style.format(
        {
            "Weight": "{:.1%}",
            "Annualized Return": "{:.2%}",
            "Annualized Volatility": "{:.2%}",
            "Sharpe": "{:.2f}",
        }
    ),
    width='stretch',
)
