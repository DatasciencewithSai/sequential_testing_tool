# Sequential Testing Tool — Project Context for Claude Code

## What This Is

An interactive Dash web app for always-valid sequential A/B testing. Combines mSPRT (frequentist) and Bayesian sequential testing, for both binary (conversion rate) and continuous (revenue, session length) metrics. Built for product and growth teams to run peek-proof experiments without a fixed sample size.

---

## Target Audience

- Product and growth data scientists
- PMs who need plain-English verdicts at every daily check

---

## Current State (as built)

### Features complete
- mSPRT statistic — binary (Bernoulli) and continuous (Normal/Welch)
- Bayesian sequential — Beta-Binomial (binary) and t-posterior (continuous)
- Classical p-value shown as education panel (z-test for binary, Welch t-test for continuous)
- Simulation tab — generate fake day-by-day data for any parameter set
- Live Test Monitor tab — enter real cumulative daily numbers from a running experiment
- Metric type toggle (Binary / Continuous) in both tabs
- 3-panel chart: mSPRT Λ, Bayesian P(treatment wins), classical p-value
- Computed results summary table in Live tab (all stats per day)
- Callout cards: mSPRT stop day, Bayesian stop day, classical false-positive day
- Inline chart guide accordion (PM decision checklist, panel-by-panel explanation)
- 72-test sanity + pressure suite in tests/test_sanity.py

### Known limitations (not yet built)
- Multiple metrics / multiplicity correction
- Multi-arm (>2 variant) tests
- Continuous metrics with extreme skew (log transform workaround suggested in README)
- Export to CSV
- Mobile layout polish

---

## Tech Stack

| Layer | Tool |
|---|---|
| App framework | Dash 4.x |
| Charts | Plotly |
| UI components | dash-bootstrap-components (FLATLY theme) |
| Math | numpy, scipy (no external mSPRT or stats packages) |
| Python | 3.9+ |
| Hosting | Render or HuggingFace Spaces (free tier) |

---

## Methods Implemented

### mSPRT (mixture Sequential Probability Ratio Test)

Based on Johari et al. (2019) "Always Valid Inference" — arXiv:1512.04922.

Core formula (same structure for binary and continuous, SE² differs):

```
Λ_n = sqrt(SE² / (SE² + τ²)) × exp(τ²·D_n² / (2·SE²·(SE² + τ²)))
```

- Binary SE²: p̂_pooled·(1−p̂)·(1/n_x + 1/n_y)
- Continuous SE²: s_x²/n_x + s_y²/n_y (Welch)
- τ = MDE (mixing parameter / prior spread)
- Stop when Λ_n ≥ 1/α

Do NOT use the `msprt` PyPI package. Everything is implemented from scratch in `core/msprt.py`.

### Bayesian sequential

- Binary: Beta-Binomial conjugate. Prior = Beta(1,1). Posterior = Beta(1+k, 1+n−k). P(treatment > control) via 50,000 Monte Carlo samples.
- Continuous: Flat Normal prior. Posterior ≈ t(df=n−1, loc=x̄, scale=SE). P(treatment > control) via 50,000 Monte Carlo samples from scipy t-distributions.
- Verdict: P ≥ 0.95 → "Stop — Treatment Wins", P ≤ 0.05 → "Stop — Control Wins", else "Continue".
- Implemented in `core/bayesian.py`.

### Classical p-value (education only)

- Binary: two-proportion z-test (manual, scipy.stats.norm.sf).
- Continuous: Welch two-sample t-test (scipy.stats.ttest_ind_from_stats).
- Shown in Panel 3. Never drives a decision. Purpose: illustrate false positive inflation from peeking.

---

## Why These Methods (design rationale)

### Why mSPRT over alpha spending

Alpha spending (O'Brien-Fleming, Pocock) requires committing upfront to specific look dates and total sample size. mSPRT lets you look any time. For teams that check dashboards daily, mSPRT is the only frequentist method that is actually compatible with real-world behaviour.

### Why mSPRT over Wald SPRT

Wald's original SPRT requires specifying the exact effect size to test against. mSPRT uses a mixture prior (τ = MDE), making it robust across a range of true effects. It degrades gracefully when the true effect differs from τ.

### Why Bayesian alongside mSPRT

mSPRT gives the formal guarantee but its output (likelihood ratio) is not intuitive for PMs. Bayesian gives a probability that is immediately interpretable ("we are 87% sure treatment is better") but lacks Type I error control. Together they cover both audiences and provide cross-validation — when both agree, confidence is very high.

### Why not Bayesian alone

Without Type I error control, a Bayesian sequential test can stop too early if the prior is miscalibrated or if the team has implicit multiple testing pressure. Most engineering organisations require frequentist guarantees for ship decisions.

### Why not a fixed-horizon test

Fixed-horizon tests require knowing the sample size in advance, waiting for the full planned duration, and accepting that any mid-experiment look is statistically invalid. This is incompatible with real product teams who check dashboards daily and sometimes need to stop early due to business constraints.

---

## Project Structure

```
sequential_testing_tool/
├── app.py                        # Dash entry point, layout, all callbacks
├── core/
│   ├── msprt.py                  # mSPRT binary + continuous, simulation, p-values
│   └── bayesian.py               # Bayesian binary + continuous
├── components/
│   ├── controls.py               # Simulation input panel (metric type toggle)
│   └── chart.py                  # 3-panel Plotly chart builder
├── tests/
│   └── test_sanity.py            # 72-test pressure suite
├── assets/
│   └── style.css
├── requirements.txt
├── CLAUDE.md                     # This file
└── README.md                     # User-facing documentation
```

---

## Key Implementation Notes for Future Claude Code Sessions

### chart.py — build_figure()
Accepts a DataFrame with columns: `day, lambda_n, bayes_prob, p_value, verdict_msprt, verdict_bayes`.
Returns a 3-panel Plotly figure (50% / 25% / 25% row heights).
Handles empty DataFrame gracefully (returns placeholder figure).

### app.py — two shared helpers
- `_compute_results_binary(sim_df, tau, alpha)` — runs all three stats on binary simulation output
- `_compute_results_continuous(sim_df, tau, alpha)` — same for continuous
Both return a standardised results DataFrame used by both chart and callout cards.

### app.py — live callback
Reads both `live-table-binary` and `live-table-continuous` as States. Uses `live-metric-type` to decide which to process. DataTable sends values as floats — always parse with `int(float(...))` for integer fields.

### Bayesian Monte Carlo seed
`bayesian_prob_binary` and `bayesian_prob_continuous` use no fixed seed (`np.random.default_rng(None)`). With 50,000 samples, variance is < 0.003 per call. This is intentional — reproducibility is not required and fixing the seed would make results identical across different data inputs if the data happened to produce the same parameters.

### mSPRT tau property
Λ is maximised when τ ≈ observed effect size. When τ >> observed effect, the sqrt term dominates and Λ < 1. Setting τ = MDE is the standard recommendation — it tunes the prior to the effect size you care about.

### Continuous simulation variance
Uses the sum-of-squares identity to compute running variance:
`var = (sum_sq - n * mean²) / (n - 1)`
This avoids storing all observations. Variance is clamped to 1e-12 to prevent division by zero.

---

## Design Principles

- Every parameter input has a tooltip in plain English (no stat jargon visible to users)
- Verdict copy must be PM-readable: "Stop — Treatment Wins" not "reject the null hypothesis"
- Panel 3 (classical p-value) must always be labelled "For Education Only"
- Chart guide accordion starts collapsed — data scientists don't need it, PMs can open it
- Single scrollable page — no routing
- Both methods shown together so PM and DS can validate against each other

---

## Next Steps (not yet built)

1. CSV export of simulation and live results
2. Alpha spending comparison panel (add as Panel 4 toggle)
3. Multi-arm / multiple comparison extension
4. Mobile layout polish
5. Deploy to HuggingFace Spaces or Render

---

## References

> Johari, R., Pekelis, L., & Walsh, D. J. (2019). *Always Valid Inference.* arXiv:1512.04922

> Wald, A. (1947). *Sequential Analysis.* Wiley.

> Deng, A., Lu, J., & Chen, S. (2016). *Continuous Monitoring of A/B Tests without Pain.* IEEE DSAA.
