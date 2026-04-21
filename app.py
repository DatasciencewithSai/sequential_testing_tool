"""Sequential Testing Tool — Dash entry point."""

import pandas as pd
import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, dcc, html, dash_table, callback

from core.msprt import (
    compute_classical_pvalue,
    compute_classical_pvalue_continuous,
    get_verdict,
    msprt_statistic,
    msprt_statistic_continuous,
    simulate_experiment,
    simulate_experiment_continuous,
)
from core.bayesian import (
    bayesian_prob_binary,
    bayesian_prob_continuous,
    get_bayesian_verdict,
)
from components.controls import build_controls
from components.chart import build_figure

BAYES_THRESHOLD = 0.95


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _compute_results_binary(sim_df, tau, alpha):
    records = []
    for _, row in sim_df.iterrows():
        lam  = msprt_statistic(int(row["control_conversions"]), int(row["n_control"]),
                               int(row["treatment_conversions"]), int(row["n_treatment"]), tau=tau)
        prob = bayesian_prob_binary(int(row["control_conversions"]), int(row["n_control"]),
                                    int(row["treatment_conversions"]), int(row["n_treatment"]))
        pval = compute_classical_pvalue(int(row["control_conversions"]), int(row["n_control"]),
                                        int(row["treatment_conversions"]), int(row["n_treatment"]))
        records.append({"day": int(row["day"]),
                         "lambda_n": lam, "bayes_prob": prob, "p_value": pval,
                         "verdict_msprt": get_verdict(lam, alpha),
                         "verdict_bayes": get_bayesian_verdict(prob, BAYES_THRESHOLD)})
    return pd.DataFrame(records)


def _compute_results_continuous(sim_df, tau, alpha):
    records = []
    for _, row in sim_df.iterrows():
        lam  = msprt_statistic_continuous(row["ctrl_mean"], row["ctrl_var"], int(row["n_control"]),
                                          row["trt_mean"],  row["trt_var"],  int(row["n_treatment"]), tau=tau)
        prob = bayesian_prob_continuous(row["ctrl_mean"], row["ctrl_var"], int(row["n_control"]),
                                        row["trt_mean"],  row["trt_var"],  int(row["n_treatment"]))
        pval = compute_classical_pvalue_continuous(row["ctrl_mean"], row["ctrl_var"], int(row["n_control"]),
                                                   row["trt_mean"],  row["trt_var"],  int(row["n_treatment"]))
        records.append({"day": int(row["day"]),
                         "lambda_n": lam, "bayes_prob": prob, "p_value": pval,
                         "verdict_msprt": get_verdict(lam, alpha),
                         "verdict_bayes": get_bayesian_verdict(prob, BAYES_THRESHOLD)})
    return pd.DataFrame(records)


def build_callout_cards(results, alpha, n_days, daily_traffic):
    threshold = 1.0 / alpha

    msprt_stop = results[results["lambda_n"] >= threshold]
    bayes_stop = results[results["verdict_bayes"] != "Continue"]
    classical_fp = results[(results["p_value"] < alpha) & (results["lambda_n"] < threshold)]

    msprt_day = int(msprt_stop["day"].iloc[0]) if not msprt_stop.empty else None
    bayes_day  = int(bayes_stop["day"].iloc[0])  if not bayes_stop.empty  else None
    fp_day     = int(classical_fp["day"].iloc[0]) if not classical_fp.empty else None

    cards = []

    # mSPRT stop card
    if msprt_day:
        saving_pct = round(100 * (n_days - msprt_day) / n_days)
        fixed_n = n_days * daily_traffic
        seq_n   = msprt_day * daily_traffic
        cards.append(dbc.Col(dbc.Alert([
            html.Strong(f"mSPRT stopped on Day {msprt_day}"), html.Br(),
            f"Saved ~{saving_pct}% of planned sample ({seq_n:,} vs {fixed_n:,} obs per variant).",
        ], color="success"), xs=12, md=4))
    else:
        cards.append(dbc.Col(dbc.Alert([
            html.Strong("mSPRT: no stop yet."), html.Br(),
            "Threshold not crossed. Keep running or re-evaluate MDE.",
        ], color="warning"), xs=12, md=4))

    # Bayesian stop card
    if bayes_day:
        bayes_verdict = bayes_stop["verdict_bayes"].iloc[0]
        cards.append(dbc.Col(dbc.Alert([
            html.Strong(f"Bayesian stopped on Day {bayes_day}"), html.Br(),
            f"P(treatment wins) crossed {BAYES_THRESHOLD} — verdict: {bayes_verdict}.",
        ], color="success"), xs=12, md=4))
    else:
        cards.append(dbc.Col(dbc.Alert([
            html.Strong("Bayesian: no stop yet."), html.Br(),
            f"P(treatment wins) hasn't crossed {BAYES_THRESHOLD}.",
        ], color="warning"), xs=12, md=4))

    # Classical false-positive card
    if fp_day:
        cards.append(dbc.Col(dbc.Alert([
            html.Strong(f"Classical test would have stopped on Day {fp_day}"), html.Br(),
            "False positive from peeking — mSPRT and Bayesian correctly continued.",
        ], color="danger"), xs=12, md=4))

    return dbc.Row(cards)


def build_chart_guide(id_suffix="sim"):
    threshold_note = "20 (when alpha=0.05)"
    return dbc.Accordion([
        dbc.AccordionItem([
            dbc.Table([
                html.Thead(html.Tr([html.Th("What you see"), html.Th("Means"), html.Th("Action")])),
                html.Tbody([
                    html.Tr([html.Td("Blue line climbing"), html.Td("Evidence building"), html.Td("Keep running")]),
                    html.Tr([html.Td(html.Strong("Blue crosses red threshold")),
                             html.Td("Strong evidence, safe to stop"),
                             html.Td(html.Strong("Stop and ship"))]),
                    html.Tr([html.Td("Blue line flat / falling"), html.Td("No meaningful effect"), html.Td("Run longer or reconsider")]),
                ]),
            ], bordered=True, hover=True, size="sm", className="mb-2"),
            dbc.Alert([html.Strong("Log scale y-axis: "), f"Threshold = 1/α = {threshold_note}. "
                       "Crossing it means data is that many times more likely under 'effect exists' than under 'no effect'."],
                      color="info", className="mb-0 py-2"),
        ], title="Panel 1 — mSPRT Λ: the line you act on"),

        dbc.AccordionItem([
            dbc.Table([
                html.Thead(html.Tr([html.Th("What you see"), html.Th("Means"), html.Th("Action")])),
                html.Tbody([
                    html.Tr([html.Td("Purple line above 0.95"), html.Td("95%+ sure treatment is better"), html.Td(html.Strong("Stop and ship treatment"))]),
                    html.Tr([html.Td("Purple line below 0.05"), html.Td("95%+ sure control is better"), html.Td(html.Strong("Stop, keep control"))]),
                    html.Tr([html.Td("Purple line between 0.05–0.95"), html.Td("Still uncertain"), html.Td("Keep running")]),
                ]),
            ], bordered=True, hover=True, size="sm", className="mb-2"),
            dbc.Alert([html.Strong("No Type I error guarantee: "),
                       "Bayesian probability is more intuitive but doesn't formally control false positives. "
                       "Use mSPRT (Panel 1) if your organisation requires frequentist guarantees."],
                      color="warning", className="mb-0 py-2"),
        ], title="Panel 2 — Bayesian P(treatment wins): intuitive but no false-positive guarantee"),

        dbc.AccordionItem([
            dbc.Alert([html.Strong("Never act on this panel. "),
                       "The orange line is a classical p-value checked daily. "
                       "Red-shaded zones show days where peeking would produce a false positive. "
                       "This panel exists to show why you need sequential methods."],
                      color="danger", className="mb-0 py-2"),
        ], title="Panel 3 — Classical p-value: shown to illustrate the peeking problem only"),

        dbc.AccordionItem([
            html.Ol([
                html.Li("Select metric type (Binary or Continuous)."),
                html.Li("Enter your parameters — use MDE from your original test plan."),
                html.Li(["Watch Panel 1 and Panel 2 daily. ",
                         html.Strong("Stop when either line crosses its threshold.")]),
                html.Li("If both agree on the same day — very high confidence."),
                html.Li("If they disagree — trust mSPRT for formal decisions; Bayesian for intuition."),
                html.Li("Never use Panel 3 to make a go/no-go call."),
            ]),
        ], title="Decision checklist for PMs"),
    ], start_collapsed=True, className="mt-3 mb-2", id=f"chart-guide-{id_suffix}")


# ---------------------------------------------------------------------------
# Live table definitions
# ---------------------------------------------------------------------------

BINARY_COLS = [
    {"name": "Day",                    "id": "day",        "type": "numeric", "editable": False},
    {"name": "Control — Total Users",  "id": "ctrl_users", "type": "numeric"},
    {"name": "Control — Conversions",  "id": "ctrl_conv",  "type": "numeric"},
    {"name": "Treatment — Total Users","id": "trt_users",  "type": "numeric"},
    {"name": "Treatment — Conversions","id": "trt_conv",   "type": "numeric"},
]

CONTINUOUS_COLS = [
    {"name": "Day",                        "id": "day",       "type": "numeric", "editable": False},
    {"name": "Control — Users",            "id": "ctrl_n",    "type": "numeric"},
    {"name": "Control — Cumulative Mean",  "id": "ctrl_mean", "type": "numeric"},
    {"name": "Control — Cumulative Std",   "id": "ctrl_std",  "type": "numeric"},
    {"name": "Treatment — Users",          "id": "trt_n",     "type": "numeric"},
    {"name": "Treatment — Cumulative Mean","id": "trt_mean",  "type": "numeric"},
    {"name": "Treatment — Cumulative Std", "id": "trt_std",   "type": "numeric"},
]

# 10 days of realistic sample data: baseline ~10% CR, treatment ~12% CR, 500/day per variant
_BINARY_SAMPLE = [
    {"day": 1,  "ctrl_users": 500,  "ctrl_conv": 48,  "trt_users": 500,  "trt_conv": 57},
    {"day": 2,  "ctrl_users": 1000, "ctrl_conv": 99,  "trt_users": 1000, "trt_conv": 118},
    {"day": 3,  "ctrl_users": 1500, "ctrl_conv": 151, "trt_users": 1500, "trt_conv": 181},
    {"day": 4,  "ctrl_users": 2000, "ctrl_conv": 198, "trt_users": 2000, "trt_conv": 244},
    {"day": 5,  "ctrl_users": 2500, "ctrl_conv": 249, "trt_users": 2500, "trt_conv": 306},
    {"day": 6,  "ctrl_users": 3000, "ctrl_conv": 301, "trt_users": 3000, "trt_conv": 369},
    {"day": 7,  "ctrl_users": 3500, "ctrl_conv": 352, "trt_users": 3500, "trt_conv": 431},
    {"day": 8,  "ctrl_users": 4000, "ctrl_conv": 403, "trt_users": 4000, "trt_conv": 493},
    {"day": 9,  "ctrl_users": 4500, "ctrl_conv": 451, "trt_users": 4500, "trt_conv": 557},
    {"day": 10, "ctrl_users": 5000, "ctrl_conv": 502, "trt_users": 5000, "trt_conv": 620},
]
BINARY_EMPTY = _BINARY_SAMPLE + [
    {"day": i, "ctrl_users": None, "ctrl_conv": None, "trt_users": None, "trt_conv": None}
    for i in range(11, 32)
]
CONTINUOUS_EMPTY = [{"day": i, "ctrl_n": None, "ctrl_mean": None, "ctrl_std": None, "trt_n": None, "trt_mean": None, "trt_std": None} for i in range(1, 32)]

TABLE_STYLE = dict(
    style_table={"overflowY": "auto", "maxHeight": "240px"},
    style_cell={"textAlign": "center", "fontSize": "13px", "padding": "4px 8px"},
    style_header={"fontWeight": "bold", "backgroundColor": "#f8f9fa"},
    style_data_conditional=[{"if": {"column_id": "day"}, "backgroundColor": "#f8f9fa", "color": "#6c757d"}],
    fixed_rows={"headers": True},
    editable=True,
    row_deletable=False,
)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY],
                title="Sequential Testing Tool")

app.layout = dbc.Container([
    # Header
    dbc.Row(dbc.Col([
        html.H2("Sequential Testing Tool", className="mb-1 mt-4"),
        html.P("Always-valid sequential A/B testing — mSPRT + Bayesian, binary and continuous metrics.",
               className="text-muted mb-3"),
    ])),

    # Tabs
    dbc.Tabs([

        # ── Tab 1: Simulation ──────────────────────────────────────────────
        dbc.Tab(dbc.Row([
            dbc.Col(build_controls(), xs=12, md=3, className="mb-4 mt-3"),
            dbc.Col([
                dcc.Loading(type="circle",
                            children=dcc.Graph(id="sim-chart", config={"displayModeBar": False})),
                build_chart_guide("sim"),
                html.Div(id="sim-callout-row", className="mt-2"),
            ], xs=12, md=9),
        ]), label="Simulation", tab_id="tab-sim"),

        # ── Tab 2: Live Test Monitor ───────────────────────────────────────
        dbc.Tab(dbc.Row([
            # Left settings
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H5("Test Settings", className="card-title mb-3"),

                dbc.Label("Metric Type"),
                dbc.RadioItems(
                    id="live-metric-type",
                    options=[
                        {"label": " Binary  (conversion rate)", "value": "binary"},
                        {"label": " Continuous  (revenue, time, etc.)", "value": "continuous"},
                    ],
                    value="binary", className="mb-3",
                    inputStyle={"marginRight": "6px"},
                ),

                dbc.Label(["MDE " , html.Span(" ⓘ", id="tip-lmde",
                           style={"cursor":"pointer","color":"#6c757d","fontSize":"0.85rem"})]),
                dbc.Tooltip("Same MDE you used to plan the test duration. For binary: proportion (e.g. 0.02). For continuous: absolute value (e.g. 0.50).", target="tip-lmde"),
                dbc.Input(id="live-mde", type="number", value=0.02, min=0.001, step=0.001, className="mb-3"),

                dbc.Label(["Alpha " , html.Span(" ⓘ", id="tip-lalpha",
                           style={"cursor":"pointer","color":"#6c757d","fontSize":"0.85rem"})]),
                dbc.Tooltip("Acceptable false-positive rate. mSPRT stop threshold = 1/alpha.", target="tip-lalpha"),
                dbc.Input(id="live-alpha", type="number", value=0.05, min=0.01, max=0.20, step=0.01, className="mb-3"),

                dbc.Label(["Planned Duration (days) " , html.Span(" ⓘ", id="tip-ldays",
                           style={"cursor":"pointer","color":"#6c757d","fontSize":"0.85rem"})]),
                dbc.Tooltip("Your original planned test duration — used to compute sample savings.", target="tip-ldays"),
                dbc.Input(id="live-days", type="number", value=30, min=1, max=180, step=1, className="mb-3"),

                dbc.Label(["Daily Traffic per Variant " , html.Span(" ⓘ", id="tip-ltraffic",
                           style={"cursor":"pointer","color":"#6c757d","fontSize":"0.85rem"})]),
                dbc.Tooltip("Your planned daily users per group — used for savings calculation.", target="tip-ltraffic"),
                dbc.Input(id="live-traffic", type="number", value=500, min=1, step=1, className="mb-3"),

                dbc.Alert([
                    html.Strong("How to fill the table: "), html.Br(),
                    html.Span(id="live-table-hint",
                              children="Enter cumulative totals as of each day. Leave future days blank."),
                ], color="info", className="mb-3", style={"fontSize": "0.82rem"}),

                dbc.Button("Compute", id="live-btn", color="primary", className="w-100", n_clicks=0),
            ]), className="shadow-sm"), xs=12, md=3, className="mb-4 mt-3"),

            # Right: tables + chart
            dbc.Col([
                dbc.Card(dbc.CardBody([
                    html.H6("Daily Data Entry", className="mb-2 text-muted"),
                    # Binary table
                    html.Div(id="binary-table-container", children=[
                        dash_table.DataTable(id="live-table-binary",
                                             columns=BINARY_COLS, data=BINARY_EMPTY,
                                             **TABLE_STYLE),
                    ]),
                    # Continuous table
                    html.Div(id="continuous-table-container", children=[
                        dash_table.DataTable(id="live-table-continuous",
                                             columns=CONTINUOUS_COLS, data=CONTINUOUS_EMPTY,
                                             **TABLE_STYLE),
                    ], style={"display": "none"}),
                ]), className="shadow-sm mb-3"),

                html.Div(id="live-results-summary", className="mb-3"),
                dcc.Loading(type="circle",
                            children=dcc.Graph(id="live-chart", config={"displayModeBar": False})),
                build_chart_guide("live"),
                html.Div(id="live-callout-row", className="mt-2"),
            ], xs=12, md=9),
        ]), label="Live Test Monitor", tab_id="tab-live"),

    ], id="main-tabs", active_tab="tab-sim", className="mb-2"),

    # Footer
    dbc.Row(dbc.Col(html.P(
        ["Reference: Johari, Pekelis & Walsh (2019) ",
         html.A("arXiv:1512.04922", href="https://arxiv.org/abs/1512.04922", target="_blank")],
        className="text-muted text-center mt-3 mb-4", style={"fontSize": "0.8rem"},
    ))),
], fluid=True, className="px-4")


# ---------------------------------------------------------------------------
# Callbacks: UI toggles
# ---------------------------------------------------------------------------

@app.callback(
    Output("binary-params",   "style"),
    Output("continuous-params", "style"),
    Input("metric-type", "value"),
)
def toggle_sim_params(metric_type):
    if metric_type == "continuous":
        return {"display": "none"}, {}
    return {}, {"display": "none"}


@app.callback(
    Output("binary-table-container",     "style"),
    Output("continuous-table-container", "style"),
    Output("live-table-hint",            "children"),
    Input("live-metric-type", "value"),
)
def toggle_live_table(metric_type):
    if metric_type == "continuous":
        return ({"display": "none"}, {},
                "Enter cumulative mean and std dev per day — values your analytics dashboard shows.")
    return ({}, {"display": "none"},
            "Enter cumulative totals as of each day. Leave future days blank.")


# ---------------------------------------------------------------------------
# Callback: Simulation tab
# ---------------------------------------------------------------------------

@app.callback(
    Output("sim-chart",       "figure"),
    Output("sim-callout-row", "children"),
    Input("run-btn",       "n_clicks"),
    State("metric-type",   "value"),
    State("baseline-rate", "value"),
    State("mde",           "value"),
    State("baseline-mean", "value"),
    State("baseline-std",  "value"),
    State("cont-mde",      "value"),
    State("alpha",         "value"),
    State("daily-traffic", "value"),
    State("n-days",        "value"),
    State("seed",          "value"),
    prevent_initial_call=False,
)
def run_simulation(n_clicks, metric_type,
                   baseline_rate, mde,
                   baseline_mean, baseline_std, cont_mde,
                   alpha, daily_traffic, n_days, seed):
    alpha         = alpha         or 0.05
    daily_traffic = int(daily_traffic or 500)
    n_days        = int(n_days        or 30)
    seed          = int(seed          or 42)

    if metric_type == "continuous":
        baseline_mean = baseline_mean or 5.0
        baseline_std  = baseline_std  or 15.0
        tau           = cont_mde      or 0.50
        sim_df  = simulate_experiment_continuous(baseline_mean, baseline_std, tau, n_days, daily_traffic, seed)
        results = _compute_results_continuous(sim_df, tau, alpha)
    else:
        baseline_rate = baseline_rate or 0.10
        tau           = mde           or 0.02
        sim_df  = simulate_experiment(baseline_rate, tau, n_days, daily_traffic, seed)
        results = _compute_results_binary(sim_df, tau, alpha)

    return build_figure(results, alpha=alpha, tau=tau), \
           build_callout_cards(results, alpha, n_days, daily_traffic)


# ---------------------------------------------------------------------------
# Callback: Live Test Monitor tab
# ---------------------------------------------------------------------------

def _empty_live_figure():
    return build_figure(pd.DataFrame(), alpha=0.05, tau=0.02)


@app.callback(
    Output("live-chart",           "figure"),
    Output("live-results-summary", "children"),
    Output("live-callout-row",     "children"),
    Input("live-btn",              "n_clicks"),
    State("live-metric-type",      "value"),
    State("live-table-binary",     "data"),
    State("live-table-continuous", "data"),
    State("live-mde",              "value"),
    State("live-alpha",            "value"),
    State("live-days",             "value"),
    State("live-traffic",          "value"),
    prevent_initial_call=False,
)
def run_live(n_clicks, metric_type, binary_data, continuous_data,
             mde, alpha, n_days, daily_traffic):
    mde           = mde           or 0.02
    alpha         = alpha         or 0.05
    n_days        = int(n_days        or 30)
    daily_traffic = int(daily_traffic or 500)
    tau           = mde
    threshold     = 1.0 / alpha

    no_data_msg = dbc.Alert("Enter data above and click Compute.", color="secondary", className="py-2")

    if n_clicks == 0:
        return _empty_live_figure(), no_data_msg, ""

    # ── Parse rows ──────────────────────────────────────────────────────────
    records = []

    if metric_type == "continuous":
        for r in (continuous_data or []):
            try:
                cn = int(float(r["ctrl_n"]))
                cm = float(r["ctrl_mean"])
                cs = float(r["ctrl_std"])
                tn = int(float(r["trt_n"]))
                tm = float(r["trt_mean"])
                ts = float(r["trt_std"])
                if cn > 1 and tn > 1 and cs > 0 and ts > 0:
                    lam  = msprt_statistic_continuous(cm, cs**2, cn, tm, ts**2, tn, tau=tau)
                    prob = bayesian_prob_continuous(cm, cs**2, cn, tm, ts**2, tn)
                    pval = compute_classical_pvalue_continuous(cm, cs**2, cn, tm, ts**2, tn)
                    records.append({"day": r["day"],
                                    "ctrl_rate": cm, "trt_rate": tm,
                                    "diff": tm - cm,
                                    "lambda_n": lam, "bayes_prob": prob, "p_value": pval,
                                    "verdict_msprt": get_verdict(lam, alpha),
                                    "verdict_bayes": get_bayesian_verdict(prob, BAYES_THRESHOLD)})
            except (TypeError, ValueError):
                continue
        rate_label, diff_label = "Ctrl Mean", "Trt Mean"
    else:
        for r in (binary_data or []):
            try:
                cu = int(float(r["ctrl_users"]))
                cc = int(float(r["ctrl_conv"]))
                tu = int(float(r["trt_users"]))
                tc = int(float(r["trt_conv"]))
                if cu > 0 and tu > 0 and cc >= 0 and tc >= 0:
                    lam  = msprt_statistic(cc, cu, tc, tu, tau=tau)
                    prob = bayesian_prob_binary(cc, cu, tc, tu)
                    pval = compute_classical_pvalue(cc, cu, tc, tu)
                    ctrl_r = cc / cu
                    trt_r  = tc / tu
                    records.append({"day": r["day"],
                                    "ctrl_rate": ctrl_r, "trt_rate": trt_r,
                                    "diff": trt_r - ctrl_r,
                                    "lambda_n": lam, "bayes_prob": prob, "p_value": pval,
                                    "verdict_msprt": get_verdict(lam, alpha),
                                    "verdict_bayes": get_bayesian_verdict(prob, BAYES_THRESHOLD)})
            except (TypeError, ValueError):
                continue
        rate_label, diff_label = "Ctrl Rate", "Trt Rate"

    if not records:
        return _empty_live_figure(), no_data_msg, ""

    results = pd.DataFrame(records)

    # ── Summary table ────────────────────────────────────────────────────────
    msprt_color  = {"Stop — Significant": "success", "Continue": "primary"}
    bayes_color  = {"Stop — Treatment Wins": "success", "Stop — Control Wins": "danger", "Continue": "primary"}
    fmt = "%.1f%%" if metric_type == "binary" else "%.3f"

    table_rows = []
    for _, row in results.iterrows():
        table_rows.append(html.Tr([
            html.Td(int(row["day"])),
            html.Td(fmt % (row["ctrl_rate"] * (100 if metric_type == "binary" else 1))),
            html.Td(fmt % (row["trt_rate"]  * (100 if metric_type == "binary" else 1))),
            html.Td(fmt % (row["diff"]      * (100 if metric_type == "binary" else 1))),
            html.Td("%.3f" % row["lambda_n"],
                    style={"fontWeight": "bold",
                           "color": "#198754" if row["lambda_n"] >= threshold else "#212529"}),
            html.Td("%.1f%%" % (row["bayes_prob"] * 100),
                    style={"color": "#6f42c1", "fontWeight": "bold"}),
            html.Td("%.4f" % row["p_value"]),
            html.Td(dbc.Badge(row["verdict_msprt"],
                              color=msprt_color.get(row["verdict_msprt"], "secondary"))),
            html.Td(dbc.Badge(row["verdict_bayes"],
                              color=bayes_color.get(row["verdict_bayes"], "secondary"))),
        ]))

    summary = dbc.Card(dbc.CardBody([
        html.H6(
            f"Results — {len(records)} days  |  mSPRT threshold Λ ≥ {threshold:.0f}  |  Bayesian threshold P ≥ {BAYES_THRESHOLD}",
            className="mb-2 text-muted",
        ),
        dbc.Table([
            html.Thead(html.Tr([
                html.Th("Day"), html.Th(rate_label), html.Th("Trt Rate/Mean"),
                html.Th("Lift"), html.Th("Λ (mSPRT)"), html.Th("P(trt wins)"),
                html.Th("p-value"), html.Th("mSPRT verdict"), html.Th("Bayes verdict"),
            ])),
            html.Tbody(table_rows),
        ], bordered=True, hover=True, size="sm", className="mb-0"),
    ]), className="shadow-sm")

    chart_df = results[["day", "lambda_n", "bayes_prob", "p_value", "verdict_msprt", "verdict_bayes"]]
    return (build_figure(chart_df, alpha=alpha, tau=tau),
            summary,
            build_callout_cards(results, alpha, n_days, daily_traffic))


if __name__ == "__main__":
    app.run(debug=True, port=8050)
