# Portfolio Risk Dashboard

An interactive dashboard for measuring, decomposing, and stress-testing the risk
of an equity portfolio. Give it holdings and weights, and it reports how much
risk the portfolio carries, where it comes from, and how it behaves under
historical shocks. Each model states its assumptions explicitly, so you can
judge when its estimates apply.

Built in clean, testable layers: a data pipeline for price history, a library of
pure, unit-tested risk and performance functions, and a thin Streamlit UI on top.

## Quick start
```bash
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py         # opens http://localhost:8501
pytest                       # run the tests
```

Enter your portfolio in the sidebar (tickers, weights, benchmark, start date).

## Project structure
```
portfolio-risk-dashboard/
├── app.py              # Streamlit UI + layout
├── src/
│   ├── config.py       # defaults & constants
│   ├── data.py         # price download, returns, portfolio aggregation
│   ├── metrics.py      # performance & risk metrics
│   ├── risk.py         # VaR, Expected Shortfall, rule-based risk read
│   ├── optimize.py     # mean-variance optimization & efficient frontier
│   ├── stress.py       # historical stress scenarios
│   └── backtest.py     # VaR backtesting (Kupiec POF test)
├── tests/              # unit tests on the math (no Streamlit)
└── requirements.txt
```

The `src/` modules contain no Streamlit code, so they're testable in isolation
and the UI holds no business logic.

## Methodology

Prices are Yahoo Finance adjusted closes (splits and dividends reflected).
Returns are daily **simple** returns, so a portfolio's return is the weighted sum
of its holdings'.

### Performance metrics

| Metric | Definition |
|---|---|
| Annualized return | Geometric: compound daily returns, raise to `252/n`. |
| Annualized volatility | Daily std × √252. |
| Sharpe ratio | (Annualized return − risk-free) ÷ annualized volatility. |
| Max drawdown | Largest peak-to-trough drop of the cumulative wealth curve. |
| Beta | cov(portfolio, benchmark) ÷ var(benchmark). |

### Value at Risk

VaR at confidence α is the loss the portfolio is not expected to exceed on a
given day. The three methods estimate the same lower tail; their disagreement
reveals how non-normal the returns are. Expected Shortfall answers what VaR
ignores — *how bad it is when you breach VaR.*

| Method | How it's estimated |
|---|---|
| Historical VaR | Empirical (1 − α) percentile of actual returns; no distributional assumption. |
| Parametric VaR | Normal: −(μ + zσ), z = Φ⁻¹(1 − α) ≈ −1.645 at 95%. |
| Monte Carlo VaR | Simulate from the assets' mean/covariance (correlations preserved) under normal or Student-t; take the percentile of simulated P&L. |
| Expected Shortfall (CVaR) | Average loss on days that breach VaR; coherent and always ≥ VaR. |

The **risk read** turns these into plain-language flags using fixed thresholds
(not a model): fat-tail, severe-tail, concentration, and drawdown flags derived
from the VaR spread, ES multiple, correlation, position size, and max drawdown.

### Portfolio optimization

Mean-variance (Markowitz) optimization on annualized moments, solved with
`scipy.optimize` (SLSQP). Portfolios are long-only and fully invested. Moments
use **arithmetic** annualization (mean × 252, covariance × 252), keeping `wᵀΣw`
consistent since variance scales linearly with time.

| Output | How it's computed |
|---|---|
| Efficient frontier | Minimum-variance portfolio for each of a grid of target returns. |
| Max Sharpe (tangency) | Weights maximizing (return − risk-free) ÷ volatility. |
| Min variance | Weights minimizing portfolio variance. |
| Ledoit-Wolf shrinkage | Optional estimator shrinking noisy sample covariance toward a structured target, stabilizing weights. |

## Limitations

- Daily simple returns ignore intraday risk, transaction costs, taxes, and slippage.
- Sharpe, beta, and volatility are **backward-looking** over the chosen window and unstable across regimes.
- Beta assumes a linear relationship to a single benchmark.
- Parametric VaR assumes normal returns and **understates tail risk**.
- All VaR/ES estimates assume the historical window represents the future; scaling one-day VaR to longer horizons assumes returns are independent across days.
- Mean-variance optimization is **highly sensitive to estimation error**; Ledoit-Wolf shrinkage mitigates but doesn't remove this. Weights are **in-sample**, long-only, and fully invested.
- Yahoo Finance data can have gaps and survivorship issues; it is not production-grade.

## Features

- [x] Data pipeline with adjusted-price history and return calculation
- [x] Performance metrics — annualized return, volatility, Sharpe, max drawdown, beta
- [x] Correlation heatmap and benchmark-relative performance
- [x] Value at Risk (historical, parametric, Monte Carlo — normal and Student-t) and Expected Shortfall
- [x] Rule-based risk read — concentration, tail-severity, and drawdown flags
- [x] Mean-variance optimization and the efficient frontier (Ledoit-Wolf covariance shrinkage)
- [x] Historical stress testing (2008, COVID, etc.) and VaR backtesting (Kupiec POF test)
- [ ] Deployment to Streamlit Community Cloud
