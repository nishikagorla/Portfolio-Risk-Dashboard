# Portfolio Risk Dashboard

An interactive dashboard for measuring, decomposing, and stress-testing the risk
of an equity portfolio. Give it a set of holdings and weights, and it reports how
much risk the portfolio carries, where that risk comes from, and how it would
behave under historical market shocks. Each model's assumptions are stated
explicitly, so users can judge when the risk estimates are appropriate.

The project is built in clean, testable layers: a data pipeline that pulls and
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
│   └── metrics.py      # performance & risk metrics (pure functions)
├── tests/
│   └── test_metrics.py # unit tests on the math
└── requirements.txt
```

Separation is deliberate: `data.py` and `metrics.py` contain no Streamlit
code, so they're testable in isolation and the UI never has business logic.

## Methodology

Prices are adjusted closes from Yahoo Finance (`auto_adjust=True`), so splits and
dividends are already reflected. Returns are daily **simple** returns, because a
portfolio's simple return is the weighted sum of its holdings' simple returns.

| Metric | Definition |
|---|---|
| Annualized return | Geometric: compound daily returns, raise to `252/n`. |
| Annualized volatility | Daily std × √252 (volatility scales with √time). |
| Sharpe ratio | (Annualized return − risk-free) ÷ annualized volatility. |
| Max drawdown | Largest peak-to-trough drop of the cumulative wealth curve. |
| Beta | cov(portfolio, benchmark) ÷ var(benchmark). |

## Limitations

*(This section is the point of the project — it's expanded as features land.)*

- Daily simple returns assume no intraday risk and ignore transaction costs,
  taxes, and slippage.
- Sharpe, beta, and volatility are computed over the chosen window; they are
  **backward-looking** and not stable across regimes.
- Beta assumes a linear relationship to a single benchmark.
- Yahoo Finance data can have gaps and survivorship issues; it is not
  production-grade.

## Features

- [x] Data pipeline with adjusted-price history and return calculation
- [x] Performance metrics — annualized return, volatility, Sharpe, max drawdown, beta
- [x] Correlation heatmap and benchmark-relative performance
- [ ] Value at Risk (historical, parametric, Monte Carlo) and Expected Shortfall
- [ ] Mean-variance optimization and the efficient frontier (Ledoit-Wolf covariance shrinkage)
- [ ] Historical stress testing (2008, COVID) and VaR backtesting (Kupiec POF test)
- [ ] Deployment to Streamlit Community Cloud
