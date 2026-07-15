from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import config, metrics, risk, optimize, stress, backtest
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
    f"expected to exceed. On the {(1 - confidence):.0%} of days it is exceeded, "
    "Expected Shortfall is the average loss. The gap between Historical/Expected "
    "Shortfall and Parametric reveals the fat tails which the normal assumption misses. "
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
    xaxis_title="Daily Return",
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
    "the return distribution may be negatively skewed with fat tails."
)


# stress testing
st.subheader("Stress Testing")

beta_val = stats.get("Beta vs. Benchmark")
has_beta = beta_val is not None and not np.isnan(beta_val)

stress_rows = []
for name, spec in stress.SCENARIOS.items():
    w_start, w_end = spec["window"]
    market_shock = spec["market_shock"]

    # historical replay if the selected date range covers the crisis window
    window_ret = asset_returns.loc[w_start:w_end]
    if window_ret.shape[0] > 1:
        hist_ret, hist_dd = stress.replay(window_ret, aligned_weights)
    else:
        hist_ret, hist_dd = np.nan, np.nan

    beta_est = stress.beta_shock(beta_val, market_shock) if has_beta else np.nan

    stress_rows.append(
        {
            "Scenario": name,
            "S&P Shock": market_shock,
            "Beta Estimate": beta_est,
            "Historical": hist_ret,
            "Max Drawdown": hist_dd,
        }
    )

stress_df = pd.DataFrame(stress_rows).set_index("Scenario")
st.dataframe(
    stress_df.style.format(
        {
            "S&P Shock": "{:.1%}",
            "Beta Estimate": "{:.1%}",
            "Historical": "{:.1%}",
            "Max Drawdown": "{:.1%}",
        },
        na_rep="—",
    ),
    width='stretch',
)

# custom hypothetical shock
custom_shock_pct = st.slider(
    "Hypothetical market shock", min_value=-50, max_value=0,
    value=-20, step=1, format="%d%%",
)
custom_shock = custom_shock_pct / 100.0
if has_beta:
    st.metric(
        "Estimated portfolio impact",
        f"{stress.beta_shock(beta_val, custom_shock):.1%}",
        help=f"beta {beta_val:.2f} x {custom_shock:.0%} market move",
    )
else:
    st.info("Add a benchmark in the sidebar to enable the beta-scaled estimate.")

st.caption(
    "**S&P Shock:** peak-to-trough drop in the S&P 500 during that period. "
    "**Beta Estimate:** a rough guess for crises your holdings predate, scaling the "
    "market's drop by your beta. **Historical:** actual loss during a past crisis "
    "(shown only if your start date covers it). "
    "**Max Drawdown:** worst peak-to-trough loss during the crisis window."
)


# var backtesting
st.subheader("VaR Backtesting")

bt_c1, bt_c2 = st.columns(2)
with bt_c1:
    bt_method = st.selectbox("VaR method to test", ["historical", "parametric"])
with bt_c2:
    bt_window = st.slider(
        "Trailing window (days)", min_value=100, max_value=500,
        value=config.BACKTEST_WINDOW, step=25,
    )

bt = backtest.var_backtest(
    port_returns, alpha=confidence, window=bt_window, method=bt_method
)

if len(bt) < 30:
    st.info(
        "Not enough history for a meaningful backtest at this window. "
        "Use an earlier start date or a smaller window."
    )
else:
    kp = backtest.kupiec_pof(len(bt), int(bt["breach"].sum()), alpha=confidence)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Observations", f"{kp['n_obs']:,}")
    k2.metric(
        "Breaches", kp["breaches"],
        help=f"expected about {kp['expected_breaches']:.0f}",
    )
    k3.metric(
        "Breach rate", f"{kp['breach_rate']:.2%}",
        help=f"expected {kp['expected_rate']:.2%}",
    )
    k4.metric("Kupiec p-value", f"{kp['p_value']:.2f}")

    if kp["reject_model"]:
        st.warning(
            f"Kupiec test **rejects** this VaR model at 5% (p = {kp['p_value']:.2f}): "
            "the breach rate differs from expected by more than chance, so the "
            "model looks mis-calibrated over this history."
        )
    else:
        st.success(
            f"Kupiec test does **not** reject the model (p = {kp['p_value']:.2f}): "
            "the breach rate is statistically consistent with a well-calibrated "
            f"{confidence:.0%} VaR."
        )

    fig_bt = go.Figure()
    fig_bt.add_trace(go.Scatter(
        x=bt.index, y=-bt["var"], mode="lines",
        name=f"-VaR ({confidence:.0%})", line=dict(color="#1f77b4"),
        hovertemplate="%{x|%Y-%m-%d}<br>VaR %{y:.2%}<extra></extra>",
    ))
    breaches = bt[bt["breach"]]
    fig_bt.add_trace(go.Scatter(
        x=breaches.index, y=breaches["ret"], mode="markers",
        name="Breaches", marker=dict(color="#d62728", size=6, symbol="x"),
        hovertemplate="%{x|%Y-%m-%d}<br>Loss %{y:.2%}<extra></extra>",
    ))
    fig_bt.update_layout(
        yaxis_tickformat=".1%",
        yaxis_title="Daily Return",
        hovermode="closest",
        legend=dict(orientation="h"),
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig_bt, width='stretch')

    st.caption(
        "Red points are days where the loss was worse than the VaR prediction. A 95 % VaR "
        "should be breached about 5% of days and the Kupiec test verifies whether the observed "
        "breach rate matches."
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
    "Correlations < 0.5 = diversification benefit. > 0.7 = concentration risk. Negative correlations = natural hedges. "
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


# portfolio optimization
st.subheader("Portfolio Optimization")

if len(asset_cols) < 2:
    st.info("Optimization needs at least two assets with data.")
else:
    use_shrinkage = st.checkbox(
        "Use Ledoit-Wolf covariance shrinkage",
        value=True,
        help="Shrinks the noisy sample covariance toward a structured target, "
        "which stabilizes the optimizer. Raw sample covariance is unstable with "
        "limited history and tends to produce extreme weights.",
    )

    mu, cov = optimize.annualized_moments(
        asset_returns, periods=config.TRADING_DAYS, shrink=use_shrinkage
    )

    # normalize current weights
    cur_w = np.asarray(aligned_weights, dtype=float)
    cur_w = cur_w / cur_w.sum()

    ms_w = optimize.max_sharpe(mu, cov, risk_free)
    mv_w = optimize.min_variance(mu, cov)
    front_vols, front_rets, _ = optimize.efficient_frontier(
        mu, cov, n_points=config.FRONTIER_POINTS
    )

    cur_ret, cur_vol, cur_sharpe = optimize.portfolio_performance(cur_w, mu, cov, risk_free)
    ms_ret, ms_vol, ms_sharpe = optimize.portfolio_performance(ms_w, mu, cov, risk_free)
    mv_ret, mv_vol, mv_sharpe = optimize.portfolio_performance(mv_w, mu, cov, risk_free)

    # efficient frontier plot
    fig_ef = go.Figure()
    fig_ef.add_trace(go.Scatter(
        x=front_vols, y=front_rets, mode="lines", name="Efficient frontier",
        line=dict(color="#1f77b4"),
        hovertemplate="Vol %{x:.1%}<br>Return %{y:.1%}<extra></extra>",
    ))

    # capital market line = risk-free rate through the tangency portfolio
    if ms_vol > 0:
        x_max = max(front_vols.max() if front_vols.size else cur_vol, cur_vol) * 1.05
        cml_x = np.array([0.0, x_max])
        cml_y = risk_free + (ms_ret - risk_free) / ms_vol * cml_x
        fig_ef.add_trace(go.Scatter(
            x=cml_x, y=cml_y, mode="lines", name="Capital market line",
            line=dict(color="#2ca02c", dash="dot", width=1), hoverinfo="skip",
        ))

    # individual assets
    asset_vols = np.sqrt(np.diag(cov))
    fig_ef.add_trace(go.Scatter(
        x=asset_vols, y=mu, mode="markers+text", name="Assets",
        text=asset_cols, textposition="top center",
        marker=dict(color="#aaaaaa", size=8),
        hovertemplate="%{text}<br>Vol %{x:.1%}<br>Return %{y:.1%}<extra></extra>",
    ))

    # key portfolios
    fig_ef.add_trace(go.Scatter(
        x=[cur_vol], y=[cur_ret], mode="markers", name="Current",
        marker=dict(color="#d62728", size=13),
        hovertemplate="Current<br>Vol %{x:.1%}<br>Return %{y:.1%}<extra></extra>",
    ))
    fig_ef.add_trace(go.Scatter(
        x=[ms_vol], y=[ms_ret], mode="markers", name="Max Sharpe",
        marker=dict(color="#2ca02c", size=16, symbol="star"),
        hovertemplate="Max Sharpe<br>Vol %{x:.1%}<br>Return %{y:.1%}<extra></extra>",
    ))
    fig_ef.add_trace(go.Scatter(
        x=[mv_vol], y=[mv_ret], mode="markers", name="Min Variance",
        marker=dict(color="#ff7f0e", size=13, symbol="diamond"),
        hovertemplate="Min Variance<br>Vol %{x:.1%}<br>Return %{y:.1%}<extra></extra>",
    ))
    fig_ef.update_layout(
        xaxis_title="Annualized Volatility",
        yaxis_title="Annualized Return",
        xaxis_tickformat=".0%",
        yaxis_tickformat=".0%",
        hovermode="closest",
        legend=dict(orientation="h"),
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig_ef, width='stretch')

    st.caption(
        "**Efficient frontier**: best achievable return for each level of "
        "volatility. **Max Sharpe**: portfolio with the best risk-adjusted"
        "return. **Min Variance**: portfolio with the lowest-risk. If **Current** "
        "sits below the frontier, the same return was reachable at lower risk."
    )

    # weights comparison
    weights_df = pd.DataFrame(
        {"Current": cur_w, "Max Sharpe": ms_w, "Min Variance": mv_w},
        index=asset_cols,
    )
    st.markdown("**Weights**")
    st.dataframe(weights_df.style.format("{:.1%}"), width='stretch')

    # performance comparison
    perf_df = pd.DataFrame(
        {
            "Return": [cur_ret, ms_ret, mv_ret],
            "Volatility": [cur_vol, ms_vol, mv_vol],
            "Sharpe": [cur_sharpe, ms_sharpe, mv_sharpe],
        },
        index=["Current", "Max Sharpe", "Min Variance"],
    )
    st.markdown("**Expected performance**")
    st.dataframe(
        perf_df.style.format(
            {"Return": "{:.2%}", "Volatility": "{:.2%}", "Sharpe": "{:.2f}"}
        ),
        width='stretch',
    )

    st.caption(
        "Returns and covariance are annualized arithmetically from the selected "
        "window. These weights are optimized **in-sample**, so they reflect what would "
        "have been optimal over this history, not a forecast. Mean-variance is sensitive "
        "to estimation error, which the shrinkage option mitigiates but does not remove."
        
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