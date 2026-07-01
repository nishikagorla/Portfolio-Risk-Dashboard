from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import config, metrics, risk
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
    "Weights are auto-normalized to 100% if needed."
)

# Monte Carlo distribution controls
st.sidebar.divider()
st.sidebar.header("Monte Carlo")
mc_dist_label = st.sidebar.selectbox(
    "Simulation distribution", ["Normal", "Student-t"], index=0
)
mc_df = st.sidebar.slider(
    "Student-t degrees of freedom",
    min_value=3,
    max_value=30,
    value=config.MC_DF,
    disabled=(mc_dist_label == "Normal"),
    help="Lower = fatter tails. Around 30 is close to Normal.",
)
mc_dist = "t" if mc_dist_label == "Student-t" else "normal"


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

st.caption(
    "**Annualized Return:** compounded annual growth. "
    "**Volatility:** annualized daily fluctuations. "
    "**Sharpe:** excess return per unit of risk. "
    "**Max Drawdown:** worst peak-to-trough decline. "
    "**Beta:** sensitivity to benchmark (>1 = more volatile than benchmark, <1 = less volatile than benchmark)."
)


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


# value at risk
st.subheader("Value at Risk (1-day)")

confidence = st.slider(
    "Confidence level",
    min_value=0.90,
    max_value=0.99,
    value=config.VAR_CONFIDENCE,
    step=0.01,
)

var_stats = risk.var_summary(
    port_returns,
    asset_returns,
    aligned_weights,
    alpha=confidence,
    n_sims=config.MC_SIMULATIONS,
    seed=42,  # fixed so the Monte Carlo number is reproducible between reruns
    dist=mc_dist,
    df=mc_df,
)

var_cols = st.columns(len(var_stats))
for col, (name, value) in zip(var_cols, var_stats.items()):
    col.metric(name, f"{value:.2%}")


if mc_dist == "t":
    mc_note = (
        f"Monte Carlo here uses a Student-t with {mc_df} degrees of freedom "
        "(fatter tails), scaled to the same covariance as the normal model, so it "
        "sits further out than Parametric and shows the tail risk that the normal "
        "assumption misses."
    )
else:
    mc_note = (
        "Monte Carlo here assumes normality, so it should track Parametric closely; "
        "a gap between those two is simulation noise, not a market signal."
    )

st.caption(
    f"At {confidence:.0%} confidence, VaR is the daily loss the portfolio is not "
    f"expected to exceed; on the {(1 - confidence):.0%} of days it is exceeded, "
    "Expected Shortfall is the average loss. The gap between Historical/Expected "
    "Shortfall and Parametric is the fat tails the normal assumption misses. "
    + mc_note
)


# distribution of daily returns with VaR & ES thresholds
fig_var = go.Figure()

bin_edges = np.histogram_bin_edges(port_returns, bins=100)
counts, _ = np.histogram(port_returns, bins=bin_edges)
y_top = counts.max() * 1.08

fig_var.add_trace(
    go.Histogram(
        x=port_returns,
        xbins=dict(
            start=bin_edges[0],
            end=bin_edges[-1],
            size=bin_edges[1] - bin_edges[0],
        ),
        name="Daily returns",
        opacity=0.75,
        showlegend=False,
        hoverinfo="skip",
    )
)

var_lines = [
    ("Historical VaR", var_stats["Historical VaR"], "#1f77b4"),
    ("Parametric VaR", var_stats["Parametric VaR"], "#ff7f0e"),
    ("Monte Carlo VaR", var_stats["Monte Carlo VaR"], "#2ca02c"),
    ("Expected Shortfall", var_stats["Expected Shortfall"], "#d62728"),
]


# add line density to display value on hover
y_line = np.linspace(0, y_top, 60)
for label, value, color in var_lines:
    fig_var.add_trace(
        go.Scatter(
            x=np.full_like(y_line, -value),
            y=y_line,
            mode="lines",
            name=label,
            line=dict(color=color, dash="dash", width=2),
            hovertemplate=f"{label}: {value:.2%}<extra></extra>",
        )
    )
fig_var.update_layout(
    xaxis_tickformat=".1%",
    xaxis_title="Daily return",
    yaxis_title="Frequency",
    hovermode="closest",
    legend=dict(orientation="h"),
    margin=dict(l=0, r=0, t=10, b=0),
    bargap=0.02,
)
fig_var.update_yaxes(range=[0, y_top])
st.plotly_chart(fig_var, width='stretch')
st.caption(
    "If Historical VaR or Expected Shortfall lie further left than Parametric VaR, "
    "that suggests the return distribution may be negatively skewed with fat tails."
)


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
st.caption(
    "Correlations < 0.5 -> diversification benefit. > 0.7 -> concentration risk. Negative correlations = natural hedges. "
    "High correlation across holdings reduces diversification and increases systematic drawdown risk."
)

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


# rule-based risk read
st.subheader("Risk Read")

read = risk.risk_read(
    asset_returns,
    aligned_weights,
    var_stats,
    max_drawdown=stats["Max Drawdown"],
    beta=stats.get("Beta vs. Benchmark"),  # None if no benchmark loaded
)

if not read:
    st.markdown("_No notable flags._")
else:
    lines = []
    for item in read:
        if item["level"] == "caution":
            lines.append(f"- **Flag —** {item['text']}")
        else:
            lines.append(f"- {item['text']}")
    st.markdown("\n".join(lines))

st.caption(
    "Automated scan for concentration, tail risk, and drawdown severity. Thresholds are fixed rules, not ML predictions."
)

with st.expander("How this read is generated"):
    st.markdown(
        "Each line comes from a fixed threshold on the metrics above — no model "
        "predicts returns or evaluates the portfolio. Current rules:\n"
        f"- Historical VaR more than {risk.FAT_TAIL_GAP:.0%} above parametric → fat-tail flag\n"
        f"- Expected Shortfall above {risk.ES_SEVERITY_RATIO:.1f}× VaR → severe-tail flag\n"
        f"- Average pairwise correlation ≥ {risk.HIGH_CORR:.2f} → concentration flag "
        f"(< {risk.LOW_CORR:.2f} → well diversified)\n"
        f"- Any single weight above {risk.HEAVY_WEIGHT:.0%} → concentration flag\n"
        f"- Max drawdown beyond {risk.DEEP_DRAWDOWN:.0%} → severe-drawdown flag"
    )

st.caption(
    "Generated from metrics above — not investment advice."
)