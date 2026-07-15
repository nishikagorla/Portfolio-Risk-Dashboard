# Portfolio Risk Dashboard

An interactive Streamlit dashboard for measuring, decomposing, and stress-testing
the risk of an equity portfolio. Enter holdings and weights, and it reports how
much risk the portfolio carries, where it comes from, and how it behaves under
historical market shocks. Each model's assumptions are stated explicitly, so you
can judge when the estimates are appropriate.

Built in clean, testable layers: a data pipeline, a library of pure unit-tested
risk/performance functions, and a thin Streamlit UI with no business logic.

## Quick start
```bash
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py         # opens http://localhost:8501
pytest                       # run the test suite
```

Enter your portfolio in the sidebar (tickers, weights, benchmark, start date).

## Project structure
```
├── app.py               # Streamlit UI + layout
├── src/
│   ├── config.py        # defaults & constants
│   ├── data.py          # price download, returns, portfolio aggregation
│   ├── metrics.py       # performance & risk metrics (return, vol, Sharpe, drawdown, beta)
│   ├── risk.py          # VaR, Expected Shortfall, rule-based risk read
│   ├── optimize.py      # mean-variance optimization & efficient frontier
│   ├── stress.py        # historical stress scenarios & beta-scaled shocks
│   └── backtest.py      # rolling VaR backtest + Kupiec POF test
└── tests/               # unit tests for each src module
```

`src/` contains no Streamlit code, so the risk math is testable in isolation.

## Methodology

Prices are adjusted closes from Yahoo Finance (splits/dividends already
reflected). Returns are daily **simple** returns, since a portfolio's simple
return is the weighted sum of its holdings'.

**Performance** — annualized return (geometric, `252/n`), annualized volatility
(daily std × √252), Sharpe, max drawdown, and beta (cov/var vs. benchmark).

**Value at Risk** — the loss not expected to be exceeded on a given day, at
confidence α. Three estimates of the same lower tail; their disagreement reveals
how non-normal the returns are:

| Method | How it's estimated |
|---|---|
| Historical | Empirical (1 − α) percentile of actual returns; no distributional assumption. |
| Parametric | Normal: −(μ + zσ), z = Φ⁻¹(1 − α) ≈ −1.645 at 95%. |
| Monte Carlo | Simulate from the assets' mean/covariance (correlations preserved) under Normal or Student-t; take the percentile of simulated P&L. |
| Expected Shortfall (CVaR) | Average loss on days that breach VaR; coherent and always ≥ VaR. |

The **risk read** turns these into plain-language flags via fixed thresholds (not
a model): fat-tail, severe-tail, concentration, and drawdown warnings.

**Stress testing** — replays the portfolio through historical crisis windows
(2008, COVID, 2022, 2018 Q4) when the date range covers them, plus a
single-factor beta-scaled estimate for any market shock.

**VaR backtesting** — rolls a trailing window forward, counts breaches, and runs
the **Kupiec POF test** (χ² likelihood ratio) to check whether the observed
breach rate is consistent with a well-calibrated VaR.

**Optimization** — mean-variance (Markowitz) on arithmetically annualized moments
(mean × 252, cov × 252), solved with SLSQP; long-only, fully invested. Traces the
efficient frontier and finds the max-Sharpe (tangency) and min-variance
portfolios. Optional Ledoit-Wolf covariance shrinkage stabilizes the weights.

## Limitations

- Daily simple returns ignore intraday risk, transaction costs, taxes, slippage.
- Sharpe, beta, volatility are backward-looking over the chosen window and not
  stable across regimes; beta assumes a linear fit to a single benchmark.
- Parametric VaR assumes normality and **understates tail risk**; all VaR/ES
  estimates depend on the chosen window being representative.
- Mean-variance optimization is **highly sensitive to estimation error**;
  Ledoit-Wolf shrinkage mitigates but doesn't remove it. Weights are computed
  **in-sample** — optimal over the window, not a forecast.
- Yahoo Finance data has gaps and survivorship issues; not production-grade.
