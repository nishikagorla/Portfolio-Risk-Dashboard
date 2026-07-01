# Portfolio Risk Dashboard

An interactive dashboard for measuring, decomposing, and stress-testing the risk
of an equity portfolio. Give it a set of holdings and weights, and it reports how
much risk the portfolio carries, where that risk comes from, and how it would
behave under historical market shocks. Each model's assumptions are stated
explicitly, so users can judge when the risk estimates are appropriate.

Project is built in clean, testable layers: a data pipeline that pulls and
cleans price history, a library of pure, unit-tested risk and performance
functions, and a thin Streamlit interface on top.

## Quick start
```bash
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Then open the URL Streamlit prints (usually http://localhost:8501). Enter your
portfolio in the sidebar (tickers, weights, benchmark, start date).

Run the tests with:

```bash
pytest
```

## Project structure
```
portfolio-risk-dashboard/
├── app.py              # Streamlit UI + layout
├── src/
│   ├── config.py       # defaults & constants
│   ├── data.py         # price download, returns, portfolio aggregation
│   ├── metrics.py      # performance & risk metric functions
│   └── risk.py         # Value at Risk and Expected Shortfall
├── tests/
│   ├── test_metrics.py # unit tests on the performance math
│   └── test_risk.py    # unit tests on the VaR / ES math
└── requirements.txt
```

`data.py`, `metrics.py`, and `risk.py` contain no Streamlit code, so they're testable in isolation and the UI never has business logic.

## Methodology

Prices are adjusted closes from Yahoo Finance, so splits and
dividends are already reflected. Returns are daily **simple** returns, because a
portfolio's simple return is the weighted sum of its holdings' simple returns.

### Performance metrics

| Metric | Definition |
|---|---|
| Annualized return | Geometric: compound daily returns, raise to `252/n`. |
| Annualized volatility | Daily std × √252 (volatility scales with √time). |
| Sharpe ratio | (Annualized return − risk-free) ÷ annualized volatility. |
| Max drawdown | Largest peak-to-trough drop of the cumulative wealth curve. |
| Beta | cov(portfolio, benchmark) ÷ var(benchmark). |

### Value at Risk

VaR at confidence α is the loss that the portfolio is not expected to exceed on a
given day. The three methods all estimate the same lower tail of the return
distribution and report a positive loss number; their disagreement reveals how
non-normal the returns are. Expected Shortfall answers the question VaR ignores —
*how bad is it when you do breach VaR.*

| Method | How it's estimated |
|---|---|
| Historical VaR | Empirical (1 − α) percentile of actual returns; no distributional assumption. |
| Parametric VaR | Normal assumption: −(μ + zσ), where z = Φ⁻¹(1 − α) ≈ −1.645 at 95%. |
| Monte Carlo VaR | Simulate scenarios from the assets' mean/covariance (preserving correlations), take the percentile of simulated portfolio P&L. |
| Expected Shortfall (CVaR) | Average loss on the days that breach VaR; coherent and always ≥ VaR. |

## Limitations

*Expanded as features land*

- Daily simple returns assume no intraday risk and ignore transaction costs,
  taxes, and slippage.
- Sharpe, beta, and volatility are computed over the chosen window; they are
  **backward-looking** and not stable across regimes.
- Beta assumes a linear relationship to a single benchmark.
- Parametric VaR assumes normally distributed returns and so **understates tail
  risk** — real returns have fat tails.
- All VaR and ES estimates depend on the chosen historical window and assume it
  is representative of the future.
- One-day VaR scaled to longer horizons assumes returns are independent across
  days; in practice volatility clusters.
- Yahoo Finance data can have gaps and survivorship issues; it is not
  production-grade.

## Features

- [x] Data pipeline with adjusted-price history and return calculation
- [x] Performance metrics — annualized return, volatility, Sharpe, max drawdown, beta
- [x] Correlation heatmap and benchmark-relative performance
- [x] Value at Risk (historical, parametric, Monte Carlo) and Expected Shortfall
- [ ] Mean-variance optimization and the efficient frontier (Ledoit-Wolf covariance shrinkage)
- [ ] Historical stress testing (2008, COVID, etc.) and VaR backtesting (Kupiec POF test)
- [ ] Deployment to Streamlit Community Cloud