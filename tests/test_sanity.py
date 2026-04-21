"""
Comprehensive sanity + pressure tests for the sequential testing tool.
Covers: mSPRT math, simulation, live monitoring data parsing, chart building,
edge cases, and boundary conditions.
Run with: python3 tests/test_sanity.py
"""

import sys, math, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from core.msprt import (
    msprt_statistic,
    simulate_experiment,
    compute_classical_pvalue,
    get_verdict,
)
from components.chart import build_figure

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
results = []

def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    results.append(condition)
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not condition:
        print(f"         ^^^ FAILED")

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ===========================================================================
# 1. msprt_statistic
# ===========================================================================
section("1. msprt_statistic — core formula")

# Null case: equal rates → Λ < 1 (no evidence for H1)
lam = msprt_statistic(100, 1000, 100, 1000, tau=0.02)
check("Equal rates → Λ < 1", lam < 1.0, f"Λ={lam:.4f}")

# Strong effect → Λ >> threshold
lam = msprt_statistic(100, 1000, 200, 1000, tau=0.1)
check("Strong effect (10% vs 20%) → Λ >> 20", lam > 1e6, f"Λ={lam:.2e}")

# Λ grows with sample size when true effect exists
lam_small = msprt_statistic(10, 100, 15, 100, tau=0.05)
lam_large = msprt_statistic(100, 1000, 150, 1000, tau=0.05)
check("Λ grows with n when true effect present", lam_large > lam_small,
      f"n=100→{lam_small:.4f}, n=1000→{lam_large:.4f}")

# Λ decreases toward null when no effect and n grows
lam_s = msprt_statistic(50, 500, 50, 500, tau=0.05)
lam_l = msprt_statistic(500, 5000, 500, 5000, tau=0.05)
check("Λ decreases under H0 as n grows", lam_l < lam_s,
      f"n=500→{lam_s:.4f}, n=5000→{lam_l:.4f}")

# Symmetry: swapping control/treatment shouldn't change Λ
lam_ab = msprt_statistic(100, 1000, 120, 1000, tau=0.02)
lam_ba = msprt_statistic(120, 1000, 100, 1000, tau=0.02)
check("Symmetric: swap ctrl/trt → same Λ", abs(lam_ab - lam_ba) < 1e-10,
      f"ab={lam_ab:.6f}, ba={lam_ba:.6f}")

# tau=MDE: Λ is maximised when tau ≈ observed effect size.
# tau too small or too large both reduce Λ. Peak should be near tau=0.02 (observed diff).
lam_at_mde  = msprt_statistic(100, 1000, 120, 1000, tau=0.02)   # tau ≈ observed diff
lam_too_big = msprt_statistic(100, 1000, 120, 1000, tau=0.10)   # tau >> observed diff
lam_too_sml = msprt_statistic(100, 1000, 120, 1000, tau=0.001)  # tau << SE
check("Λ is highest when tau ≈ observed effect (tau=0.02 beats tau=0.10 and tau=0.001)",
      lam_at_mde > lam_too_big and lam_at_mde > lam_too_sml,
      f"tau=0.02→{lam_at_mde:.4f}, tau=0.10→{lam_too_big:.4f}, tau=0.001→{lam_too_sml:.4f}")

# Edge: zero observations → returns 1.0
lam = msprt_statistic(0, 0, 0, 0, tau=0.02)
check("Zero observations → Λ=1.0", lam == 1.0, f"Λ={lam}")

# Edge: zero conversions in both groups → valid
lam = msprt_statistic(0, 1000, 0, 1000, tau=0.02)
check("Zero conversions both groups → Λ<1 (no signal)", lam <= 1.0, f"Λ={lam:.4f}")

# Edge: 100% conversion rate
lam = msprt_statistic(1000, 1000, 1000, 1000, tau=0.02)
check("100% conversion both groups → Λ=1.0 (degenerate, no diff)", lam == 1.0, f"Λ={lam}")

# Edge: unbalanced groups
lam = msprt_statistic(50, 500, 80, 800, tau=0.02)
check("Unbalanced groups → returns finite float", math.isfinite(lam), f"Λ={lam:.4f}")

# Λ is always positive
for ctrl_c, ctrl_n, trt_c, trt_n in [(0,100,5,100),(50,100,50,100),(99,100,1,100)]:
    lam = msprt_statistic(ctrl_c, ctrl_n, trt_c, trt_n, tau=0.05)
    check(f"Λ always positive ({ctrl_c}/{ctrl_n} vs {trt_c}/{trt_n})", lam >= 0, f"Λ={lam:.4f}")


# ===========================================================================
# 2. compute_classical_pvalue
# ===========================================================================
section("2. compute_classical_pvalue")

# Equal rates → high p-value
p = compute_classical_pvalue(100, 1000, 100, 1000)
check("Equal rates → p > 0.5", p > 0.5, f"p={p:.4f}")

# Large effect → tiny p-value
p = compute_classical_pvalue(100, 1000, 200, 1000)
check("Large effect → p < 0.001", p < 0.001, f"p={p:.6f}")

# p-value always in [0, 1]
for cc, cn, tc, tn in [(0,100,10,100),(50,100,55,100),(100,100,0,100)]:
    p = compute_classical_pvalue(cc, cn, tc, tn)
    check(f"p ∈ [0,1] for ({cc}/{cn} vs {tc}/{tn})", 0 <= p <= 1, f"p={p:.4f}")

# Zero observations → 1.0
p = compute_classical_pvalue(0, 0, 0, 0)
check("Zero observations → p=1.0", p == 1.0, f"p={p}")

# Two-sided: p-value same when effect is negative
p_pos = compute_classical_pvalue(100, 1000, 130, 1000)
p_neg = compute_classical_pvalue(130, 1000, 100, 1000)
check("Two-sided: p same for positive/negative effect", abs(p_pos - p_neg) < 1e-10,
      f"p_pos={p_pos:.6f}, p_neg={p_neg:.6f}")


# ===========================================================================
# 3. get_verdict
# ===========================================================================
section("3. get_verdict")

check("Λ=20, alpha=0.05 → Stop (threshold=20)", get_verdict(20.0, 0.05) == "Stop — Significant")
check("Λ=19.99, alpha=0.05 → Continue", get_verdict(19.99, 0.05) == "Continue")
check("Λ=100, alpha=0.05 → Stop", get_verdict(100.0, 0.05) == "Stop — Significant")
check("Λ=1.0, alpha=0.05 → Continue", get_verdict(1.0, 0.05) == "Continue")
check("Λ=10, alpha=0.10 → Stop (threshold=10)", get_verdict(10.0, 0.10) == "Stop — Significant")
check("Λ=9.99, alpha=0.10 → Continue", get_verdict(9.99, 0.10) == "Continue")


# ===========================================================================
# 4. simulate_experiment
# ===========================================================================
section("4. simulate_experiment")

df = simulate_experiment(baseline_rate=0.10, mde=0.02, n_days=30, daily_traffic=500)

check("Returns DataFrame", isinstance(df, pd.DataFrame))
check("Has correct columns", list(df.columns) == ["day","control_conversions","treatment_conversions","n_control","n_treatment"])
check("Correct number of rows", len(df) == 30, f"rows={len(df)}")
check("Days are 1..30", list(df["day"]) == list(range(1, 31)))
check("n_control final = 30*500", df["n_control"].iloc[-1] == 15000)
check("Cumulative control conversions monotone increasing", df["control_conversions"].is_monotonic_increasing)
check("Cumulative treatment conversions monotone increasing", df["treatment_conversions"].is_monotonic_increasing)

# Treatment rate > control rate on average (true lift = MDE = 0.02)
ctrl_rate = df["control_conversions"].iloc[-1] / df["n_control"].iloc[-1]
trt_rate = df["treatment_conversions"].iloc[-1] / df["n_treatment"].iloc[-1]
check("Treatment rate > control rate overall", trt_rate > ctrl_rate,
      f"ctrl={ctrl_rate:.4f}, trt={trt_rate:.4f}")

# Reproducibility: same seed → same results
df2 = simulate_experiment(baseline_rate=0.10, mde=0.02, n_days=30, daily_traffic=500, seed=42)
check("Same seed → same results", df.equals(df2))

# Different seed → different results
df3 = simulate_experiment(baseline_rate=0.10, mde=0.02, n_days=30, daily_traffic=500, seed=99)
check("Different seed → different results", not df.equals(df3))

# Conversion counts bounded by traffic
check("Control conversions ≤ n_control", (df["control_conversions"] <= df["n_control"]).all())
check("Treatment conversions ≤ n_treatment", (df["treatment_conversions"] <= df["n_treatment"]).all())
check("All conversion counts ≥ 0", (df["control_conversions"] >= 0).all() and (df["treatment_conversions"] >= 0).all())

# Edge: 1 day
df1 = simulate_experiment(0.10, 0.02, n_days=1, daily_traffic=500)
check("1-day simulation works", len(df1) == 1)

# Edge: high baseline rate
df_hi = simulate_experiment(0.95, 0.02, n_days=10, daily_traffic=100)
check("High baseline (0.95) simulation works", len(df_hi) == 10)

# Edge: tiny MDE
df_tiny = simulate_experiment(0.10, 0.001, n_days=10, daily_traffic=100)
check("Tiny MDE (0.001) simulation works", len(df_tiny) == 10)


# ===========================================================================
# 5. Full simulation pipeline (simulate → msprt → chart)
# ===========================================================================
section("5. Full simulation pipeline")

for baseline, mde, alpha, traffic, days in [
    (0.10, 0.02, 0.05, 500, 30),   # standard
    (0.05, 0.01, 0.05, 1000, 60),  # low baseline, long test
    (0.50, 0.05, 0.05, 200, 14),   # high baseline
    (0.01, 0.005, 0.10, 5000, 20), # low baseline, high traffic
]:
    df = simulate_experiment(baseline, mde, days, traffic)
    records = []
    for _, row in df.iterrows():
        lam = msprt_statistic(int(row["control_conversions"]), int(row["n_control"]),
                              int(row["treatment_conversions"]), int(row["n_treatment"]), tau=mde)
        pval = compute_classical_pvalue(int(row["control_conversions"]), int(row["n_control"]),
                                        int(row["treatment_conversions"]), int(row["n_treatment"]))
        records.append({"day": row["day"], "lambda_n": lam, "p_value": pval,
                         "verdict": get_verdict(lam, alpha)})
    res = pd.DataFrame(records)

    all_finite = res["lambda_n"].apply(math.isfinite).all()
    all_positive = (res["lambda_n"] > 0).all()
    p_bounded = ((res["p_value"] >= 0) & (res["p_value"] <= 1)).all()
    check(f"Pipeline ok: baseline={baseline}, mde={mde}, alpha={alpha}, traffic={traffic}, days={days}",
          all_finite and all_positive and p_bounded,
          f"finite={all_finite}, positive={all_positive}, p_bounded={p_bounded}")


# ===========================================================================
# 6. Live monitoring — data parsing (mirrors the callback logic)
# ===========================================================================
section("6. Live monitoring — data parsing")

def parse_live_rows(table_data):
    rows = []
    for r in table_data:
        try:
            cu = int(float(r["ctrl_users"]))
            cc = int(float(r["ctrl_conv"]))
            tu = int(float(r["trt_users"]))
            tc = int(float(r["trt_conv"]))
            if cu > 0 and tu > 0 and cc >= 0 and tc >= 0:
                rows.append({"day": r["day"], "ctrl_users": cu, "ctrl_conv": cc,
                              "trt_users": tu, "trt_conv": tc})
        except (TypeError, ValueError):
            continue
    return rows

# Normal input
table = [{"day": i, "ctrl_users": i*500, "ctrl_conv": int(i*50), "trt_users": i*500, "trt_conv": int(i*60)}
         for i in range(1, 6)]
rows = parse_live_rows(table)
check("5 complete rows parsed correctly", len(rows) == 5)

# Mixed filled/empty (trailing None rows)
table2 = table + [{"day": 6, "ctrl_users": None, "ctrl_conv": None, "trt_users": None, "trt_conv": None}]
rows2 = parse_live_rows(table2)
check("None rows are skipped", len(rows2) == 5)

# String numbers from DataTable (type="numeric" can still send strings)
table3 = [{"day": 1, "ctrl_users": "500", "ctrl_conv": "50", "trt_users": "500", "trt_conv": "60"}]
rows3 = parse_live_rows(table3)
check("String numbers parsed correctly", len(rows3) == 1 and rows3[0]["ctrl_users"] == 500)

# Float numbers (DataTable sends floats)
table4 = [{"day": 1, "ctrl_users": 500.0, "ctrl_conv": 50.0, "trt_users": 500.0, "trt_conv": 60.0}]
rows4 = parse_live_rows(table4)
check("Float numbers parsed correctly", len(rows4) == 1)

# Zero users → skipped
table5 = [{"day": 1, "ctrl_users": 0, "ctrl_conv": 0, "trt_users": 500, "trt_conv": 60}]
rows5 = parse_live_rows(table5)
check("Zero users → row skipped", len(rows5) == 0)

# Zero conversions → valid (0% conversion rate is legitimate)
table6 = [{"day": 1, "ctrl_users": 500, "ctrl_conv": 0, "trt_users": 500, "trt_conv": 0}]
rows6 = parse_live_rows(table6)
check("Zero conversions → row kept (0% rate is valid)", len(rows6) == 1)

# Conversions > users → still parsed (validation is caller's concern)
table7 = [{"day": 1, "ctrl_users": 100, "ctrl_conv": 200, "trt_users": 100, "trt_conv": 50}]
rows7 = parse_live_rows(table7)
check("Conversions > users → parsed without crash", len(rows7) == 1)

# All empty → returns empty list
table8 = [{"day": i, "ctrl_users": None, "ctrl_conv": None, "trt_users": None, "trt_conv": None}
          for i in range(1, 32)]
rows8 = parse_live_rows(table8)
check("All-empty table → 0 rows", len(rows8) == 0)

# Negative effect (treatment worse than control)
table9 = [{"day": 1, "ctrl_users": 1000, "ctrl_conv": 150, "trt_users": 1000, "trt_conv": 80}]
rows9 = parse_live_rows(table9)
lam_neg = msprt_statistic(rows9[0]["ctrl_conv"], rows9[0]["ctrl_users"],
                           rows9[0]["trt_conv"], rows9[0]["trt_users"], tau=0.05)
check("Negative effect → Λ still finite positive", math.isfinite(lam_neg) and lam_neg > 0,
      f"Λ={lam_neg:.4f}")

# Very large numbers (big platform)
table10 = [{"day": 1, "ctrl_users": 5_000_000, "ctrl_conv": 250_000,
            "trt_users": 5_000_000, "trt_conv": 275_000}]
rows10 = parse_live_rows(table10)
lam_big = msprt_statistic(rows10[0]["ctrl_conv"], rows10[0]["ctrl_users"],
                           rows10[0]["trt_conv"], rows10[0]["trt_users"], tau=0.005)
check("Very large numbers (5M users) → Λ finite", math.isfinite(lam_big), f"Λ={lam_big:.2e}")

# Single day of data
table11 = [{"day": 1, "ctrl_users": 500, "ctrl_conv": 50, "trt_users": 500, "trt_conv": 60}]
rows11 = parse_live_rows(table11)
lam1 = msprt_statistic(rows11[0]["ctrl_conv"], rows11[0]["ctrl_users"],
                        rows11[0]["trt_conv"], rows11[0]["trt_users"], tau=0.02)
check("Single day of data → valid Λ", math.isfinite(lam1), f"Λ={lam1:.4f}")


# ===========================================================================
# 7. Live monitoring — mSPRT values are correct
# ===========================================================================
section("7. Live monitoring — mSPRT correctness")

# Λ should grow day by day when true effect is consistent
days_data = [
    (1,  500,  50,  500,  61),
    (2,  1000, 100, 1000, 122),
    (3,  1500, 149, 1500, 184),
    (4,  2000, 200, 2000, 244),
    (5,  2500, 250, 2500, 305),
    (10, 5000, 498, 5000, 612),
    (20, 10000, 998, 10000, 1224),
]
lambdas = [msprt_statistic(cc, cu, tc, tu, tau=0.02) for _, cu, cc, tu, tc in days_data]
check("Λ is generally increasing over days with consistent effect",
      lambdas[-1] > lambdas[0], f"Day1={lambdas[0]:.3f}, Day20={lambdas[-1]:.3f}")

# p-value decreases as more data comes in with a real effect
pvals = [compute_classical_pvalue(cc, cu, tc, tu) for _, cu, cc, tu, tc in days_data]
check("p-value decreases as n grows with real effect",
      pvals[-1] < pvals[0], f"Day1={pvals[0]:.4f}, Day20={pvals[-1]:.4f}")

# Verdicts are correct strings
for lam, expected in [(25.0, "Stop — Significant"), (5.0, "Continue"), (19.99, "Continue"), (20.0, "Stop — Significant")]:
    v = get_verdict(lam, 0.05)
    check(f"Verdict correct for Λ={lam}", v == expected, f"got='{v}'")


# ===========================================================================
# 8. Chart building — no crashes on edge cases
# ===========================================================================
section("8. Chart building — edge cases")

from core.bayesian import bayesian_prob_binary, get_bayesian_verdict

def _make_row(day, lam, p):
    """Helper: build a results row with all columns build_figure expects."""
    return {
        "day": day, "lambda_n": lam, "bayes_prob": 0.5, "p_value": p,
        "verdict_msprt": get_verdict(lam, 0.05),
        "verdict_bayes": get_bayesian_verdict(0.5),
    }

# Normal 30-day results
df = simulate_experiment(0.10, 0.02, 30, 500)
records = []
for row in df.itertuples():
    lam  = msprt_statistic(int(row.control_conversions), int(row.n_control),
                           int(row.treatment_conversions), int(row.n_treatment), tau=0.02)
    prob = bayesian_prob_binary(int(row.control_conversions), int(row.n_control),
                                int(row.treatment_conversions), int(row.n_treatment))
    pval = compute_classical_pvalue(int(row.control_conversions), int(row.n_control),
                                    int(row.treatment_conversions), int(row.n_treatment))
    records.append({"day": int(row.day), "lambda_n": lam, "bayes_prob": prob, "p_value": pval,
                    "verdict_msprt": get_verdict(lam, 0.05),
                    "verdict_bayes": get_bayesian_verdict(prob)})
res = pd.DataFrame(records)

try:
    fig = build_figure(res, alpha=0.05, tau=0.02)
    check("30-day chart builds without error", True, f"traces={len(fig.data)}")
except Exception as e:
    check("30-day chart builds without error", False, str(e))

# Single row
try:
    single = pd.DataFrame([_make_row(1, 0.91, 0.31)])
    fig = build_figure(single, alpha=0.05, tau=0.02)
    check("Single-row chart builds without error", True)
except Exception as e:
    check("Single-row chart builds without error", False, str(e))

# Threshold crossed on Day 1
try:
    early_stop = pd.DataFrame([_make_row(i, 25.0, 0.001) for i in range(1, 6)])
    fig = build_figure(early_stop, alpha=0.05, tau=0.02)
    check("Threshold crossed Day 1 → chart builds ok", True)
except Exception as e:
    check("Threshold crossed Day 1 → chart builds ok", False, str(e))

# No threshold crossing
try:
    no_stop = pd.DataFrame([_make_row(i, float(i) * 0.5, 0.1) for i in range(1, 31)])
    fig = build_figure(no_stop, alpha=0.05, tau=0.02)
    check("No threshold crossing → chart builds ok", True)
except Exception as e:
    check("No threshold crossing → chart builds ok", False, str(e))

# Very large lambda (no overflow on log scale)
try:
    big_lam = pd.DataFrame([_make_row(1, 1e30, 0.0001)])
    fig = build_figure(big_lam, alpha=0.05, tau=0.02)
    check("Very large lambda (1e30) → chart builds ok", True)
except Exception as e:
    check("Very large lambda (1e30) → chart builds ok", False, str(e))


# ===========================================================================
# 9. Alpha boundary conditions
# ===========================================================================
section("9. Alpha boundary conditions")

for alpha in [0.01, 0.05, 0.10, 0.20]:
    threshold = 1.0 / alpha
    lam_above = msprt_statistic(200, 1000, 300, 1000, tau=0.10)
    verdict = get_verdict(lam_above, alpha)
    p = compute_classical_pvalue(200, 1000, 300, 1000)
    check(f"alpha={alpha}: threshold={threshold}, p={p:.4f}, verdict='{verdict}'",
          isinstance(verdict, str))


# ===========================================================================
# Summary
# ===========================================================================
total = len(results)
passed = sum(results)
failed = total - passed

print(f"\n{'='*60}")
print(f"  RESULTS: {passed}/{total} passed", end="")
if failed:
    print(f"  — \033[91m{failed} FAILED\033[0m")
else:
    print(f"  — \033[92mAll tests passed\033[0m")
print(f"{'='*60}\n")

sys.exit(0 if failed == 0 else 1)
