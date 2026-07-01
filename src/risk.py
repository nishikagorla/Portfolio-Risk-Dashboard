from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm, multivariate_t

# all functions return VaR & ES as a POSITIVE percentage representing a loss.

def historical_var(returns, alpha=0.95):
    return -np.percentile(returns, (1.0 - alpha) * 100.0)


def parametric_var(returns, alpha=0.95):
    mu = returns.mean()
    sigma = returns.std(ddof=1)
    z = norm.ppf(1.0 - alpha)
    return -(mu + z * sigma)


def monte_carlo_var(asset_returns, weights, alpha=0.95, n_sims=10000, seed=None,
                    dist="normal", df=5):
    rng = np.random.default_rng(seed)
    mu = np.asarray(asset_returns.mean()) # compute mean return of all assets
    cov = np.asarray(asset_returns.cov()) # compute covariance matrix of all assets
    n_assets = mu.shape[0] # count assets

    w = np.asarray(weights, dtype=float)
    if w.sum() == 0:
        raise ValueError("Weights sum to zero.")
    w = w / w.sum() # normalize

    if dist == "normal":
        sims = rng.multivariate_normal(mu, cov, size=n_sims)
    elif dist == "t":
        if df <= 2:
            raise ValueError("Degrees of freedom must be > 2 for finite variance.")
        scale = cov * (df - 2) / df  # match to historical cov, isolate fat tail effect
        sims = multivariate_t.rvs(loc=mu, shape=scale, df=df, size=n_sims,
                                  random_state=rng)
        sims = np.asarray(sims).reshape(n_sims, n_assets)  # guard 1-asset squeeze
    else:
        raise ValueError(f"Unknown distribution: {dist!r}")

    port_sims = sims @ w # compute portfolio returns using similation returns
    return -np.percentile(port_sims, (1.0 - alpha) * 100.0)


def expected_shortfall(returns, alpha=0.95):
    var = historical_var(returns, alpha)
    arr = np.asarray(returns)
    tail = arr[arr <= -var]
    if tail.size == 0:
        return var
    return -tail.mean()


def var_summary(port_returns, asset_returns, weights, alpha=0.95, n_sims=10000,
                seed=None, dist="normal", df=5):
    return {
        "Historical VaR": historical_var(port_returns, alpha),
        "Parametric VaR": parametric_var(port_returns, alpha),
        "Monte Carlo VaR": monte_carlo_var(asset_returns, weights, alpha, n_sims,
                                           seed, dist=dist, df=df),
        "Expected Shortfall": expected_shortfall(port_returns, alpha),
    }


# rule-based risk read
FAT_TAIL_GAP = 0.10        # if historical VaR this fraction above parametric -> fat tails
ES_SEVERITY_RATIO = 1.30   # if ES this multiple of VaR -> breaches are severe
HIGH_CORR = 0.60           # if average pairwise correlation at/above this -> concentrated
LOW_CORR = 0.30            # if below this -> well diversified
HEAVY_WEIGHT = 0.40        # if single holding above this -> dominant position
DEEP_DRAWDOWN = 0.30       # if |max drawdown| beyond this -> severe
HIGH_BETA = 1.20
LOW_BETA = 0.80


def _avg_pairwise_corr(asset_returns):
    corr = asset_returns.corr().to_numpy()
    n = corr.shape[0]
    if n < 2:
        return None
    iu = np.triu_indices(n, k=1)
    return float(corr[iu].mean())


def risk_read(asset_returns, weights, var_stats, max_drawdown, beta=None):
    findings = []

    hist = var_stats.get("Historical VaR")
    param = var_stats.get("Parametric VaR")
    es = var_stats.get("Expected Shortfall")

    # normal model vs empirical
    if hist is not None and param and param > 0:
        gap = (hist - param) / param
        if gap > FAT_TAIL_GAP:
            findings.append({
                "level": "caution",
                "text": f"Historical VaR is {gap:.0%} above parametric VaR, "
                        "showing fat-tailed losses the normal-distribution "
                        "model understates.",
            })
        elif gap < -FAT_TAIL_GAP:
            findings.append({
                "level": "info",
                "text": f"Parametric VaR exceeds historical by {-gap:.0%} over "
                        "this window; in-sample tails were thinner than normal.",
            })
        else:
            findings.append({
                "level": "info",
                "text": "Parametric and historical VaR agree closely; tails are "
                        "near-normal in this window.",
            })

    # tail severity, how bad the average breach is
    if hist and hist > 0 and es is not None:
        ratio = es / hist
        if ratio > ES_SEVERITY_RATIO:
            findings.append({
                "level": "caution",
                "text": f"When losses breach VaR they average {ratio:.1f}x the "
                        "VaR threshold, so tail events tend to be severe.",
            })

    # diversification via average pairwise correlation
    avg_corr = _avg_pairwise_corr(asset_returns)
    if avg_corr is not None:
        n = asset_returns.shape[1]
        if avg_corr >= HIGH_CORR:
            findings.append({
                "level": "caution",
                "text": f"Average pairwise correlation is {avg_corr:.2f}: the {n} "
                        "holdings move closely together, so diversification is "
                        "weaker than the holding count suggests.",
            })
        elif avg_corr < LOW_CORR:
            findings.append({
                "level": "info",
                "text": f"Average pairwise correlation is {avg_corr:.2f}: holdings "
                        "are largely independent, giving meaningful "
                        "diversification.",
            })
        else:
            findings.append({
                "level": "info",
                "text": f"Average pairwise correlation is {avg_corr:.2f}: moderate "
                        "co-movement between holdings.",
            })

    # position concentration
    w = np.asarray(weights, dtype=float)
    if w.sum() > 0:
        w = w / w.sum()
        max_w = float(w.max())
        if max_w > HEAVY_WEIGHT:
            findings.append({
                "level": "caution",
                "text": f"A single position is {max_w:.0%} of the portfolio; risk "
                        "is concentrated in one name.",
            })

    # drawdown severity
    if max_drawdown is not None and not np.isnan(max_drawdown):
        if abs(max_drawdown) > DEEP_DRAWDOWN:
            findings.append({
                "level": "caution",
                "text": f"Worst peak-to-trough drawdown over the window was "
                        f"{max_drawdown:.0%}, a substantial historical loss.",
            })
        else:
            findings.append({
                "level": "info",
                "text": f"Worst drawdown over the window was {max_drawdown:.0%}.",
            })

    # market sensitivity
    if beta is not None and not np.isnan(beta):
        if beta > HIGH_BETA:
            findings.append({
                "level": "info",
                "text": f"Beta of {beta:.2f}: the portfolio tends to move about "
                        f"{beta - 1:.0%} more than the benchmark.",
            })
        elif beta < LOW_BETA:
            findings.append({
                "level": "info",
                "text": f"Beta of {beta:.2f}: the portfolio tends to move about "
                        f"{1 - beta:.0%} less than the benchmark.",
            })
        else:
            findings.append({
                "level": "info",
                "text": f"Beta of {beta:.2f}: the portfolio tends to move in line "
                        f"with the benchmark.",
            })

    return findings