"""
Bayesian sequential testing — Beta-Binomial (binary) and t-posterior (continuous).

Binary:  conjugate Beta prior updated with daily conversions.
         P(treatment > control) via Monte Carlo over Beta posteriors.

Continuous: flat Normal prior, posterior ~ t(n-1, x̄, SE).
            P(treatment > control) via Monte Carlo over t posteriors.

Both methods use 50,000 Monte Carlo samples for ~0.2% estimation variance.
"""

import numpy as np
from scipy import stats


def bayesian_prob_binary(
    x_conversions: int, x_n: int,
    y_conversions: int, y_n: int,
    n_samples: int = 50_000,
) -> float:
    """
    P(θ_treatment > θ_control) using Beta-Binomial conjugate model.

    Prior: Beta(1, 1) — uniform / non-informative.
    Posterior after data: Beta(1 + conversions, 1 + non-conversions).

    Returns probability in [0, 1].
    """
    if x_n == 0 or y_n == 0:
        return 0.5

    alpha_c = 1 + x_conversions
    beta_c  = 1 + x_n - x_conversions
    alpha_t = 1 + y_conversions
    beta_t  = 1 + y_n - y_conversions

    rng = np.random.default_rng(None)
    ctrl_samples = rng.beta(alpha_c, beta_c, n_samples)
    trt_samples  = rng.beta(alpha_t, beta_t, n_samples)

    return float(np.mean(trt_samples > ctrl_samples))


def bayesian_prob_continuous(
    x_mean: float, x_var: float, x_n: int,
    y_mean: float, y_var: float, y_n: int,
    n_samples: int = 50_000,
) -> float:
    """
    P(μ_treatment > μ_control) using t-distribution posteriors (flat prior).

    With a flat prior on the mean, the posterior is:
        μ | data ~ t(df=n-1, loc=x̄, scale=SE)  where SE = sqrt(var/n)

    Returns probability in [0, 1].
    """
    if x_n < 2 or y_n < 2:
        return 0.5

    se_x = np.sqrt(max(x_var, 1e-12) / x_n)
    se_y = np.sqrt(max(y_var, 1e-12) / y_n)

    ctrl_samples = stats.t.rvs(df=x_n - 1, loc=x_mean, scale=se_x, size=n_samples)
    trt_samples  = stats.t.rvs(df=y_n - 1, loc=y_mean, scale=se_y, size=n_samples)

    return float(np.mean(trt_samples > ctrl_samples))


def get_bayesian_verdict(prob: float, threshold: float = 0.95) -> str:
    """
    PM-readable verdict from a posterior probability.

    threshold=0.95 means: stop when we're 95% sure which group is better.
    """
    if prob >= threshold:
        return "Stop — Treatment Wins"
    if prob <= 1.0 - threshold:
        return "Stop — Control Wins"
    return "Continue"
