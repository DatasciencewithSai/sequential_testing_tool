# Sequential Testing Tool

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Framework](https://img.shields.io/badge/Framework-Dash-orange)
![Methods](https://img.shields.io/badge/Methods-mSPRT%20%2B%20Bayesian-purple)

> **Peek at your A/B test results every day — without inflating false positives.**

An interactive web app for always-valid sequential A/B testing. Combines **mSPRT** (frequentist, always-valid) and **Bayesian sequential** testing side-by-side, for both **binary** (conversion rate) and **continuous** (revenue, session length, LTV) metrics.

Built for product and data science teams who run experiments and need a principled answer to *"is it safe to stop today?"* — without waiting for a fixed sample size.

---

## The Problem This Solves

Classical A/B tests break when you peek at results early. Every extra look inflates your false-positive rate. A team checking results daily on a planned 30-day test can easily have a true false positive rate of 20–30% while thinking it is 5%.

This tool uses methods that are **mathematically valid at every single look** — you can check results every morning with no statistical penalty.

---

## Quick Start

```bash
git clone https://github.com/DatasciencewithSai/sequential_testing_tool.git
cd sequential_testing_tool
pip install -r requirements.txt
python3 app.py
```

Open **http://localhost:8050** in your browser.

**Stop the app:** `Ctrl + C`

---

## Features

- **Two methods, side-by-side** — mSPRT (frequentist, Type I error guarantee) + Bayesian sequential (intuitive probability), always shown together for cross-validation
- **Binary and continuous metrics** — conversion rates, click-through rates, revenue per user, session length, LTV, order value
- **Simulation tab** — run a simulated day-by-day experiment before you start, to understand expected stopping patterns
- **Live Test Monitor tab** — enter your actual daily cumulative numbers from a running experiment, get today's verdict in real time
- **3-panel chart** — mSPRT Λ statistic, Bayesian P(treatment wins), and classical p-value (shown to illustrate the peeking problem)
- **PM-readable verdicts** — "Stop — Treatment Wins", "Continue" — no stat jargon
- **Inline chart guide** — collapsible decision checklist and panel explanations built into the app
- **72-test sanity suite** — math, parsing, chart building, and edge cases all covered

---

## Why These Two Methods

### The peeking problem with classical tests

Classical tests (t-test, z-test, p-value) assume you look at results exactly once. Every additional look inflates the false-positive rate. This is a well-documented problem that affects most product teams running experiments.

---

### mSPRT — frequentist, always valid

mSPRT (mixture Sequential Probability Ratio Test) produces a likelihood ratio Λ_n that is valid at **any** sample size. You can look at it every day with no penalty. This is called "always-valid inference" (Johari et al., 2019).

**Why mSPRT over other frequentist sequential methods:**

| Method | Always valid at any peek | Requires pre-planned look dates | Requires fixed sample size | Type I error guarantee |
|---|---|---|---|---|
| **mSPRT** | **Yes** | **No** | **No** | **Yes** |
| Alpha spending (O'Brien-Fleming) | Only at pre-planned looks | Yes | Yes | Yes |
| SPRT (Wald) | Yes | No | No | Yes, but requires known effect size |
| Classical z/t-test | No | No | Yes | Yes only at end |

Alpha spending requires committing upfront to specific look dates and total sample size. mSPRT lets you look whenever you want. Wald's SPRT requires specifying the exact effect size to test against — mSPRT uses a mixture prior (τ = MDE) so it is robust across a range of true effects.

**mSPRT formula:**

```
Λ_n = sqrt(SE² / (SE² + τ²)) × exp(τ²·D_n² / (2·SE²·(SE² + τ²)))
```

- `D_n` = observed mean difference (treatment − control)
- `SE²` = variance of the mean difference
- `τ` = mixing parameter, set to your MDE
- **Stop when Λ_n ≥ 1/α** (e.g. ≥ 20 when α = 0.05)

For binary: `SE² = p̂_pooled·(1−p̂)·(1/n_x + 1/n_y)`
For continuous: `SE² = s_x²/n_x + s_y²/n_y` (Welch)

---

### Bayesian sequential — intuitive, no peek penalty

The Bayesian approach updates a probability each day: *"what is the probability that treatment is better than control?"* You stop when you are confident enough (e.g. P ≥ 0.95).

**mSPRT vs Bayesian — when to use which:**

| Property | mSPRT | Bayesian sequential |
|---|---|---|
| Output | Likelihood ratio Λ | P(treatment > control) |
| Interpretation | "Data is Λ× more likely if effect exists" | "We are 87% sure treatment is better" |
| Formal Type I error control | **Yes — guaranteed ≤ α** | No — credible statement, not frequentist |
| Prior required | Only τ (set to MDE) | Uniform Beta(1,1) — non-informative |
| Best used for | Formal go/no-go decisions | Directional confidence, early signals |

When both methods cross their thresholds on the same day — very high confidence. When they disagree — mSPRT drives the formal decision, Bayesian gives the intuitive confidence level.

**Why not Bayesian alone:** without Type I error control, a purely Bayesian sequential test can stop too early if the prior is miscalibrated. Most engineering organisations require a frequentist guarantee for ship decisions.

**Bayesian model:**
- Binary: Beta-Binomial conjugate. Prior = Beta(1,1). Posterior = Beta(1 + conversions, 1 + non-conversions). 50,000 Monte Carlo samples.
- Continuous: Flat Normal prior. Posterior ≈ t(df=n−1, loc=x̄, scale=SE). 50,000 Monte Carlo samples.
- Stop threshold: P ≥ 0.95 (treatment wins) or P ≤ 0.05 (control wins).

---

## Reading the Charts

The app shows three stacked panels. Panels 1 and 2 are your decision tools. Panel 3 is for education only.

### Panel 1 — mSPRT Statistic (Λ) — act on this

The blue line is your evidence meter. It starts near 1 and climbs as data accumulates. The dashed red line is the stop threshold (= 1/α).

| What you see | What it means | What to do |
|---|---|---|
| Blue line climbing | Evidence for a real effect is building | Keep running |
| **Blue line crosses the red line** | Strong frequentist evidence — safe to stop | **Stop and ship the winner** |
| Blue line flat or drifting down | No meaningful difference emerging | Run longer or re-evaluate |
| "STOP — Day N" annotation | Threshold was crossed on that day | You could have safely stopped there |

The y-axis is log scale — Λ can grow exponentially once a real effect appears. A value of 20 (α=0.05) means the data is 20× more likely under "effect exists" than "no effect."

### Panel 2 — Bayesian P(treatment wins) — confirmation

The purple line shows the daily probability that treatment is better than control.

| What you see | What it means | What to do |
|---|---|---|
| **Purple line above 0.95** | ≥95% confident treatment is better | Stop, ship treatment |
| **Purple line below 0.05** | ≥95% confident control is better | Stop, keep control |
| Purple line between 0.05–0.95 | Still uncertain | Keep running |

**When both Panel 1 and Panel 2 agree on the same day — very high confidence. When they disagree — trust Panel 1.**

### Panel 3 — Classical p-value — education only

Shows what a standard daily p-value check would look like. Red-shaded zones show where peeking would produce a false positive. **Never use this panel to make a ship/no-ship call.**

### Callout cards

After running the simulation or clicking Compute, three cards appear:

- **Green — mSPRT stopped on Day N** — days and users saved vs running the full planned duration
- **Green — Bayesian stopped on Day N** — when Bayesian confidence crossed 95%
- **Red — Classical test would have stopped on Day N** — the false positive day you avoided

---

## How to Use — PM Decision Checklist

1. Select your metric type (Binary or Continuous).
2. Enter parameters matching your actual test plan — use the same MDE you used to calculate the original sample size.
3. Run the **Simulation** tab to see the expected stopping pattern before starting the real test.
4. Once the test is live, enter your daily cumulative numbers each morning in the **Live Test Monitor** tab.
5. If the blue line (Panel 1) crosses its threshold — stop immediately. No need to wait for the planned end date.
6. Use the purple line (Panel 2) for confidence. If both agree, act with full confidence.
7. If neither panel crosses a threshold by the planned end date — the effect is smaller than your MDE or does not exist. Extend the test or call it null.
8. Never use Panel 3 (p-value) to make a go/no-go call.

---

## Parameters Reference

### Binary metric

| Parameter | Default | Description |
|---|---|---|
| Baseline Conversion Rate | 0.10 | Control group rate before the test. E.g. 0.10 = 10% of users convert. |
| MDE (proportion) | 0.02 | Smallest lift worth detecting. E.g. 0.02 = 2pp lift. Also used as τ in mSPRT. |
| Alpha | 0.05 | Acceptable false-positive rate. mSPRT stop threshold = 1/α = 20. |
| Daily Traffic per Variant | 500 | Users assigned to each group per day. |
| Days to Simulate | 30 | Length of the simulated experiment. |
| Random Seed | 42 | Controls the random number generator for reproducibility. |

### Continuous metric (additional inputs)

| Parameter | Default | Description |
|---|---|---|
| Baseline Mean | 5.00 | Average metric value per user in control. E.g. $5.00 revenue. |
| Baseline Std Dev | 15.00 | Standard deviation across users. Typically 2–5× the mean for revenue. |
| MDE (absolute) | 0.50 | Smallest absolute lift worth detecting. E.g. $0.50 = detect +$0.50 per user. |

---

## Known Limitations

| Limitation | Workaround |
|---|---|
| Multiple metrics simultaneously | Run separate tests; apply Bonferroni correction to α |
| Continuous metrics with extreme skew (e.g. revenue with whales) | Log-transform the metric before entering mean/std, or use a ratio metric |
| Baseline rates < 1% | Increase sample size or use exact binomial methods |
| Multi-arm tests (more than 2 variants) | Split into pairwise comparisons |

---

<details>
<summary><strong>Technical Implementation Details</strong></summary>

### Binary simulation
Generates daily Binomial draws for control (p = baseline_rate) and treatment (p = baseline_rate + MDE). Tracks cumulative conversions and users.

### Continuous simulation
Generates daily Normal draws for control (μ = baseline_mean, σ = baseline_std) and treatment (μ = baseline_mean + MDE, σ = baseline_std). Tracks cumulative sum and sum-of-squares to compute running mean and unbiased variance without storing all observations.

### mSPRT — binary
`SE² = p̂_pooled·(1−p̂_pooled)·(1/n_x + 1/n_y)`

### mSPRT — continuous
`SE² = s_x²/n_x + s_y²/n_y` (Welch approximation)

### Bayesian — binary
Prior: Beta(1,1). Posterior: Beta(1+k, 1+n−k). Monte Carlo: 50,000 samples.

### Bayesian — continuous
Prior: flat Normal. Posterior: t(df=n−1, loc=x̄, scale=SE). Monte Carlo: 50,000 samples from scipy t-distributions.

### Classical p-value
Binary: two-proportion z-test. Continuous: Welch two-sample t-test (`scipy.stats.ttest_ind_from_stats`).

</details>

---

## Project Structure

```
sequential_testing_tool/
├── app.py                        # Dash entry point, layout, all callbacks
├── core/
│   ├── msprt.py                  # mSPRT (binary + continuous), simulation, classical p-values
│   └── bayesian.py               # Bayesian sequential (binary + continuous)
├── components/
│   ├── controls.py               # Simulation input panel with metric type toggle
│   └── chart.py                  # 3-panel Plotly chart builder
├── tests/
│   └── test_sanity.py            # 72-test sanity + pressure suite
├── assets/
│   └── style.css                 # Custom styles (optional)
├── requirements.txt
├── CLAUDE.md                     # Project context for Claude Code sessions
└── README.md                     # This file
```

---

## Running the Tests

```bash
python3 tests/test_sanity.py
# Expected: 72/72 passed
```

---

## Contributing

Contributions welcome. Some good first areas:

- CSV export of simulation and live results
- Alpha spending (O'Brien-Fleming) as a fourth comparison panel
- Multi-arm / multiple comparison extension
- Mobile layout improvements

Please open an issue before starting a large change so we can align on approach.

---

## Package Versions (tested)

| Package | Version |
|---|---|
| Python | 3.9+ |
| dash | 4.1.0 |
| dash-bootstrap-components | 2.0.4 |
| plotly | 6.7.0 |
| numpy | 2.4.4 |
| scipy | 1.17.1 |
| pandas | 3.0.2 |

---

## References

> Johari, R., Pekelis, L., & Walsh, D. J. (2019).
> *Always Valid Inference: Bringing Sequential Analysis to A/B Testing.*
> arXiv:1512.04922

> Wald, A. (1947). *Sequential Analysis.* Wiley.

> Deng, A., Lu, J., & Chen, S. (2016).
> *Continuous Monitoring of A/B Tests without Pain: Optional Stopping in Bayesian Testing.*
> IEEE DSAA.

---

## License

MIT — see [LICENSE](LICENSE).
