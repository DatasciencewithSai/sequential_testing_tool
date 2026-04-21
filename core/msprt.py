"""
mSPRT (mixture Sequential Probability Ratio Test) — Bernoulli and continuous.
Based on Johari et al. (2019) "Always Valid Inference" — arXiv:1512.04922
"""

import numpy as np
import pandas as pd
from scipy import stats


def msprt_statistic(x_conversions: int, x_n: int, y_conversions: int, y_n: int, tau: float) -> float:
    """
    Compute the mSPRT likelihood ratio Λ_n for Bernoulli outcomes.

    Derived from Johari et al. (2019): Λ_n = P(data | H1 mixture) / P(data | H0).
    With mixing prior δ ~ N(0, τ²) and D_n = ȳ-x̄ ~ N(δ, SE²) under H1:

        Λ_n = sqrt(SE² / (SE² + τ²)) * exp(τ²·D_n² / (2·SE²·(SE² + τ²)))

    where SE² = p̂_pooled·(1-p̂_pooled)·(1/n_x + 1/n_y) is the variance of D_n.
    The statistic grows with n when a true effect exists, enabling early stopping.

    Args:
        x_conversions: control group conversions
        x_n:           control group total observations
        y_conversions: treatment group conversions
        y_n:           treatment group total observations
        tau:           mixing parameter (prior spread), typically set to MDE

    Returns:
        Λ_n; values >= 1/alpha indicate sufficient evidence to stop.
    """
    if x_n == 0 or y_n == 0:
        return 1.0

    x_rate = x_conversions / x_n
    y_rate = y_conversions / y_n
    pooled_rate = (x_conversions + y_conversions) / (x_n + y_n)

    observed_diff = y_rate - x_rate

    # Variance of the observed mean difference
    se_sq = pooled_rate * (1 - pooled_rate) * (1 / x_n + 1 / y_n)
    if se_sq <= 0:
        return 1.0

    tau_sq = tau ** 2
    sqrt_term = np.sqrt(se_sq / (se_sq + tau_sq))
    exp_term = np.exp(tau_sq * observed_diff ** 2 / (2 * se_sq * (se_sq + tau_sq)))

    return float(sqrt_term * exp_term)


def simulate_experiment(
    baseline_rate: float,
    mde: float,
    n_days: int,
    daily_traffic: int,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Simulate a day-by-day A/B test with a true lift equal to mde.

    Returns a DataFrame with cumulative columns:
        day, control_conversions, treatment_conversions, n_control, n_treatment
    """
    rng = np.random.default_rng(seed)
    treatment_rate = baseline_rate + mde

    rows = []
    cum_ctrl_conv = 0
    cum_trt_conv = 0
    cum_ctrl_n = 0
    cum_trt_n = 0

    for day in range(1, n_days + 1):
        ctrl_obs = rng.binomial(daily_traffic, baseline_rate)
        trt_obs = rng.binomial(daily_traffic, treatment_rate)

        cum_ctrl_conv += ctrl_obs
        cum_trt_conv += trt_obs
        cum_ctrl_n += daily_traffic
        cum_trt_n += daily_traffic

        rows.append({
            "day": day,
            "control_conversions": cum_ctrl_conv,
            "treatment_conversions": cum_trt_conv,
            "n_control": cum_ctrl_n,
            "n_treatment": cum_trt_n,
        })

    return pd.DataFrame(rows)


def compute_classical_pvalue(
    x_conversions: int, x_n: int, y_conversions: int, y_n: int
) -> float:
    """Two-proportion z-test p-value (two-sided). Returns 1.0 on degenerate inputs."""
    if x_n == 0 or y_n == 0:
        return 1.0
    x_rate = x_conversions / x_n
    y_rate = y_conversions / y_n
    pooled = (x_conversions + y_conversions) / (x_n + y_n)
    se = np.sqrt(pooled * (1 - pooled) * (1 / x_n + 1 / y_n))
    if se == 0:
        return 1.0
    z = (y_rate - x_rate) / se
    return float(2 * stats.norm.sf(abs(z)))


def msprt_statistic_continuous(
    x_mean: float, x_var: float, x_n: int,
    y_mean: float, y_var: float, y_n: int,
    tau: float,
) -> float:
    """
    mSPRT likelihood ratio Λ_n for continuous outcomes (revenue, time, etc.).

    Same formula as Bernoulli variant but SE² uses per-group sample variances:
        SE² = x_var/n_x + y_var/n_y  (Welch approximation)

    Args:
        x_mean, x_var, x_n: control cumulative mean, variance, sample size
        y_mean, y_var, y_n: treatment cumulative mean, variance, sample size
        tau:                mixing parameter — set to MDE in original metric units
    """
    if x_n < 2 or y_n < 2:
        return 1.0

    observed_diff = y_mean - x_mean
    se_sq = max(x_var, 1e-12) / x_n + max(y_var, 1e-12) / y_n

    tau_sq = tau ** 2
    sqrt_term = np.sqrt(se_sq / (se_sq + tau_sq))
    exp_term  = np.exp(tau_sq * observed_diff ** 2 / (2 * se_sq * (se_sq + tau_sq)))

    return float(sqrt_term * exp_term)


def simulate_experiment_continuous(
    baseline_mean: float,
    baseline_std: float,
    mde: float,
    n_days: int,
    daily_traffic: int,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Simulate a day-by-day continuous-metric A/B test (e.g. revenue per user).

    Returns a DataFrame with cumulative columns:
        day, ctrl_mean, ctrl_var, trt_mean, trt_var, n_control, n_treatment
    """
    rng = np.random.default_rng(seed)
    treatment_mean = baseline_mean + mde

    cum_ctrl_sum = cum_ctrl_sq = 0.0
    cum_trt_sum  = cum_trt_sq  = 0.0
    cum_n = 0
    rows = []

    for day in range(1, n_days + 1):
        ctrl_obs = rng.normal(baseline_mean,  baseline_std, daily_traffic)
        trt_obs  = rng.normal(treatment_mean, baseline_std, daily_traffic)

        cum_ctrl_sum += ctrl_obs.sum()
        cum_ctrl_sq  += (ctrl_obs ** 2).sum()
        cum_trt_sum  += trt_obs.sum()
        cum_trt_sq   += (trt_obs ** 2).sum()
        cum_n        += daily_traffic

        ctrl_mean = cum_ctrl_sum / cum_n
        trt_mean  = cum_trt_sum  / cum_n
        # Unbiased sample variance via sum-of-squares identity
        ctrl_var = max((cum_ctrl_sq - cum_n * ctrl_mean ** 2) / (cum_n - 1), 1e-12)
        trt_var  = max((cum_trt_sq  - cum_n * trt_mean  ** 2) / (cum_n - 1), 1e-12)

        rows.append({
            "day":        day,
            "ctrl_mean":  ctrl_mean,
            "ctrl_var":   ctrl_var,
            "trt_mean":   trt_mean,
            "trt_var":    trt_var,
            "n_control":  cum_n,
            "n_treatment": cum_n,
        })

    return pd.DataFrame(rows)


def compute_classical_pvalue_continuous(
    x_mean: float, x_var: float, x_n: int,
    y_mean: float, y_var: float, y_n: int,
) -> float:
    """Welch two-sample t-test p-value (two-sided) for continuous metrics."""
    if x_n < 2 or y_n < 2:
        return 1.0
    _, p = stats.ttest_ind_from_stats(
        mean1=y_mean, std1=np.sqrt(max(y_var, 1e-12)), nobs1=y_n,
        mean2=x_mean, std2=np.sqrt(max(x_var, 1e-12)), nobs2=x_n,
        equal_var=False,
    )
    return float(p)


def get_verdict(lambda_n: float, alpha: float) -> str:
    """
    Returns a PM-readable verdict string.

    "Stop — Significant":  evidence strong enough to call a winner
    "Stop — Futile":       lambda has been very low for long (not implemented here,
                           caller can use context)
    "Continue":            keep collecting data
    """
    threshold = 1.0 / alpha
    if lambda_n >= threshold:
        return "Stop — Significant"
    return "Continue"


# ---------------------------------------------------------------------------
# Inline unit tests
# ---------------------------------------------------------------------------

def _run_tests():
    # --- msprt_statistic ---
    # Equal rates → sqrt(SE²/(SE²+τ²)) < 1, exp term = 1 → Λ < 1
    lam_null = msprt_statistic(50, 1000, 50, 1000, tau=0.02)
    assert lam_null < 1.0, f"Expected Λ<1 for null (no evidence for H1), got {lam_null}"

    # Large true effect with 1000 obs/group → Λ should be enormous (well above threshold)
    lam_effect = msprt_statistic(100, 1000, 200, 1000, tau=0.1)
    assert lam_effect > 1e6, f"Expected very high lambda for strong effect, got {lam_effect}"

    # Edge: zero observations
    assert msprt_statistic(0, 0, 0, 0, tau=0.02) == 1.0

    # --- simulate_experiment ---
    df = simulate_experiment(baseline_rate=0.10, mde=0.02, n_days=30, daily_traffic=500)
    assert list(df.columns) == [
        "day", "control_conversions", "treatment_conversions", "n_control", "n_treatment"
    ]
    assert len(df) == 30
    assert df["n_control"].iloc[-1] == 30 * 500
    assert df["control_conversions"].is_monotonic_increasing
    assert df["treatment_conversions"].is_monotonic_increasing

    # --- compute_classical_pvalue ---
    p_equal = compute_classical_pvalue(100, 1000, 100, 1000)
    assert p_equal > 0.5, f"Equal groups should have high p-value, got {p_equal}"

    p_diff = compute_classical_pvalue(50, 1000, 200, 1000)
    assert p_diff < 0.001, f"Large difference should have tiny p-value, got {p_diff}"

    # --- get_verdict ---
    assert get_verdict(25.0, alpha=0.05) == "Stop — Significant"  # 1/0.05=20
    assert get_verdict(10.0, alpha=0.05) == "Continue"

    print("All msprt.py tests passed.")


if __name__ == "__main__":
    _run_tests()
