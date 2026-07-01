import numpy as np
import pandas as pd

from src import risk


def _normal_returns(n=100_000, mu=0.0, sigma=0.01, seed=0):
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(mu, sigma, n))


def test_historical_var_matches_definition():
    # Historical VaR must equal the negated (1 - alpha) percentile. This pins
    # down the sign convention and that alpha maps to the right percentile.
    r = _normal_returns()
    assert np.isclose(risk.historical_var(r, 0.95), -np.percentile(r, 5))


def test_parametric_var_normal_sample():
    # For a clean normal sample, parametric VaR at 95% is about 1.645 * sigma
    # (mean ~ 0). 1.645 is -norm.ppf(0.05).
    sigma = 0.01
    r = _normal_returns(sigma=sigma)
    assert np.isclose(risk.parametric_var(r, 0.95), 1.645 * sigma, rtol=0.05)


def test_es_at_least_var():
    # Expected Shortfall averages the tail beyond VaR, so it can never be
    # smaller than VaR.
    r = _normal_returns(seed=3)
    assert risk.expected_shortfall(r, 0.95) >= risk.historical_var(r, 0.95)


def test_var_increases_with_confidence():
    # A higher confidence level looks further into the tail -> larger loss.
    r = _normal_returns(seed=7)
    assert risk.historical_var(r, 0.99) >= risk.historical_var(r, 0.95)


def test_monte_carlo_close_to_parametric_for_normal():
    # Single normal asset: the MC generator is normal, so MC VaR should land
    # near parametric VaR. This confirms the simulation + weighting is wired up.
    sigma = 0.01
    asset_returns = pd.DataFrame({"A": _normal_returns(sigma=sigma, seed=1)})
    mc = risk.monte_carlo_var(asset_returns, [1.0], alpha=0.95, n_sims=200_000, seed=1)
    param = risk.parametric_var(asset_returns["A"], 0.95)
    assert np.isclose(mc, param, rtol=0.05)


def test_monte_carlo_weights_normalized():
    # Weights that don't sum to 1 should be normalized, not crash.
    r1 = _normal_returns(sigma=0.01, seed=10)
    r2 = _normal_returns(sigma=0.02, seed=11)
    asset_returns = pd.DataFrame({"A": r1, "B": r2})
    out = risk.monte_carlo_var(asset_returns, [2.0, 2.0], alpha=0.95, seed=5)
    assert out > 0


# ---------------------------------------------------------------------------
# rule-based risk read
# ---------------------------------------------------------------------------

# A single-column frame makes _avg_pairwise_corr return None (n < 2), so the
# correlation branch stays silent and we can isolate every other branch.
_ONE_ASSET = pd.DataFrame({"A": [0.01, -0.01, 0.02, -0.02]})


def _texts(findings):
    # Lower-cased, joined finding text for substring assertions.
    return " ".join(f["text"] for f in findings).lower()


def _correlated_df(n=5000, base_sigma=0.01, noise_sigma=0.004, seed=0):
    # Shared factor dominates idiosyncratic noise -> high pairwise correlation.
    rng = np.random.default_rng(seed)
    base = rng.normal(0.0, base_sigma, n)
    return pd.DataFrame({
        c: base + rng.normal(0.0, noise_sigma, n) for c in ("A", "B", "C")
    })


def _independent_df(n=5000, seed=0):
    # Separate draws -> near-zero pairwise correlation.
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        c: rng.normal(0.0, 0.01, n) for c in ("A", "B", "C")
    })


# --- fat-tail branch (historical vs parametric) ---------------------------

def test_fat_tail_flagged_when_historical_far_above_parametric():
    # gap = (hist - param) / param = 0.33 > FAT_TAIL_GAP -> caution.
    stats = {"Historical VaR": 0.02, "Parametric VaR": 0.015,
             "Expected Shortfall": 0.02}
    out = risk.risk_read(_ONE_ASSET, [1.0], stats, max_drawdown=None)
    assert any(f["level"] == "caution" and "fat-tailed" in f["text"]
               for f in out)


def test_near_normal_when_historical_and_parametric_agree():
    # gap within +/- FAT_TAIL_GAP -> the "near-normal" info finding.
    stats = {"Historical VaR": 0.0152, "Parametric VaR": 0.015,
             "Expected Shortfall": 0.016}
    assert "near-normal" in _texts(
        risk.risk_read(_ONE_ASSET, [1.0], stats, max_drawdown=None))


def test_thin_tails_when_parametric_exceeds_historical():
    # gap < -FAT_TAIL_GAP -> the thinner-than-normal info finding.
    stats = {"Historical VaR": 0.012, "Parametric VaR": 0.015,
             "Expected Shortfall": 0.013}
    assert "thinner than normal" in _texts(
        risk.risk_read(_ONE_ASSET, [1.0], stats, max_drawdown=None))


def test_fat_tail_skipped_when_parametric_zero():
    # param not > 0 -> the whole fat-tail block is skipped, no crash.
    stats = {"Historical VaR": 0.02, "Parametric VaR": 0.0,
             "Expected Shortfall": 0.02}
    out = risk.risk_read(_ONE_ASSET, [1.0], stats, max_drawdown=None)
    assert not any("var" in f["text"].lower() and "parametric" in f["text"].lower()
                   for f in out)


def test_var_block_skipped_when_stats_missing():
    # Missing keys must not raise; simply produce no VaR/ES findings.
    out = risk.risk_read(_ONE_ASSET, [1.0], {}, max_drawdown=None)
    assert all("var" not in f["text"].lower() for f in out)


# --- ES severity branch ---------------------------------------------------

def test_es_severity_flagged():
    # es / hist = 1.5 > ES_SEVERITY_RATIO -> caution.
    stats = {"Historical VaR": 0.02, "Parametric VaR": 0.02,
             "Expected Shortfall": 0.03}
    assert any(f["level"] == "caution" and "severe" in f["text"]
               for f in risk.risk_read(_ONE_ASSET, [1.0], stats, max_drawdown=None))


def test_es_severity_not_flagged_below_threshold():
    # ratio = 1.2 < ES_SEVERITY_RATIO -> no severity caution.
    stats = {"Historical VaR": 0.02, "Parametric VaR": 0.02,
             "Expected Shortfall": 0.024}
    assert not any("severe" in f["text"]
                   for f in risk.risk_read(_ONE_ASSET, [1.0], stats,
                                           max_drawdown=None))


# --- correlation / diversification branch ---------------------------------

def test_high_correlation_flagged_as_concentrated():
    df = _correlated_df()
    out = risk.risk_read(df, [1, 1, 1], {}, max_drawdown=None)
    assert any(f["level"] == "caution" and "move closely together" in f["text"]
               for f in out)


def test_low_correlation_flagged_as_diversified():
    df = _independent_df()
    assert "meaningful" in _texts(
        risk.risk_read(df, [1, 1, 1], {}, max_drawdown=None))


def test_single_asset_has_no_correlation_finding():
    out = risk.risk_read(_ONE_ASSET, [1.0], {}, max_drawdown=None)
    assert "pairwise correlation" not in _texts(out)


# --- concentration branch -------------------------------------------------

def test_heavy_weight_flagged():
    # Unnormalized weights, but one name is 60% after normalizing -> caution.
    df = _independent_df()
    out = risk.risk_read(df, [6, 2, 2], {}, max_drawdown=None)
    assert any(f["level"] == "caution" and "concentrated in one name" in f["text"]
               for f in out)


def test_no_concentration_when_balanced():
    df = _independent_df()
    out = risk.risk_read(df, [1, 1, 1], {}, max_drawdown=None)
    assert "concentrated in one name" not in _texts(out)


def test_zero_weights_skip_concentration():
    # All-zero weights must not raise or divide by zero.
    out = risk.risk_read(_ONE_ASSET, [0.0], {}, max_drawdown=None)
    assert "concentrated in one name" not in _texts(out)


# --- drawdown branch ------------------------------------------------------

def test_deep_drawdown_flagged():
    out = risk.risk_read(_ONE_ASSET, [1.0], {}, max_drawdown=-0.4)
    assert any(f["level"] == "caution" and "drawdown" in f["text"] for f in out)


def test_shallow_drawdown_is_info():
    out = risk.risk_read(_ONE_ASSET, [1.0], {}, max_drawdown=-0.1)
    assert any(f["level"] == "info" and "drawdown" in f["text"] for f in out)


def test_no_drawdown_finding_when_none():
    out = risk.risk_read(_ONE_ASSET, [1.0], {}, max_drawdown=None)
    assert "drawdown" not in _texts(out)


def test_no_drawdown_finding_when_nan():
    out = risk.risk_read(_ONE_ASSET, [1.0], {}, max_drawdown=np.nan)
    assert "drawdown" not in _texts(out)


# --- beta branch ----------------------------------------------------------

def test_high_beta_finding():
    out = risk.risk_read(_ONE_ASSET, [1.0], {}, max_drawdown=None, beta=1.5)
    assert "more than the benchmark" in _texts(out)


def test_low_beta_finding():
    out = risk.risk_read(_ONE_ASSET, [1.0], {}, max_drawdown=None, beta=0.5)
    assert "less than the benchmark" in _texts(out)


def test_in_line_beta_finding():
    out = risk.risk_read(_ONE_ASSET, [1.0], {}, max_drawdown=None, beta=1.0)
    assert "in line with the benchmark" in _texts(out)


def test_no_beta_finding_when_none_or_nan():
    assert "beta" not in _texts(
        risk.risk_read(_ONE_ASSET, [1.0], {}, max_drawdown=None, beta=None))
    assert "beta" not in _texts(
        risk.risk_read(_ONE_ASSET, [1.0], {}, max_drawdown=None, beta=np.nan))
