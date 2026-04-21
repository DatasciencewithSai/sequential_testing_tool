"""Input panel for the simulation tab — supports Binary and Continuous metrics."""

import dash_bootstrap_components as dbc
from dash import dcc, html


def _tip(input_id, text):
    return [
        html.Span(" ⓘ", id=f"tip-{input_id}",
                  style={"cursor": "pointer", "color": "#6c757d", "fontSize": "0.85rem"}),
        dbc.Tooltip(text, target=f"tip-{input_id}", placement="right"),
    ]


def _row(label, input_id, value, min_val, max_val, step, tooltip):
    return dbc.Row(dbc.Col([
        dbc.Label([label] + _tip(input_id, tooltip), html_for=input_id, className="mb-1"),
        dbc.Input(id=input_id, type="number", value=value,
                  min=min_val, max=max_val, step=step, className="mb-3"),
    ]))


def build_controls() -> dbc.Card:
    return dbc.Card(dbc.CardBody([
        html.H5("Simulation Parameters", className="card-title mb-3"),

        # ── Metric type ──────────────────────────────────────────────────
        dbc.Label("Metric Type", className="mb-1"),
        dbc.RadioItems(
            id="metric-type",
            options=[
                {"label": " Binary  (conversion rate, click-through rate)", "value": "binary"},
                {"label": " Continuous  (revenue, session length, LTV)", "value": "continuous"},
            ],
            value="binary",
            className="mb-3",
            inputStyle={"marginRight": "6px"},
        ),

        # ── Binary params ────────────────────────────────────────────────
        html.Div(id="binary-params", children=[
            _row("Baseline Conversion Rate", "baseline-rate", 0.10, 0.001, 0.999, 0.001,
                 "Current conversion rate in your control group. E.g. 0.10 = 10% of visitors convert."),
            _row("MDE (percentage points)", "mde", 0.02, 0.001, 0.50, 0.001,
                 "Smallest lift you care about detecting. E.g. 0.02 = 2pp lift. Also used as τ in mSPRT."),
        ]),

        # ── Continuous params ────────────────────────────────────────────
        html.Div(id="continuous-params", children=[
            _row("Baseline Mean", "baseline-mean", 5.00, 0.01, 1e6, 0.01,
                 "Average metric value per user in control today. E.g. $5.00 revenue per visitor."),
            _row("Baseline Std Dev", "baseline-std", 15.00, 0.01, 1e6, 0.01,
                 "Standard deviation of the metric across users. Typically 2–5× the mean for revenue."),
            _row("MDE (same units as mean)", "cont-mde", 0.50, 0.001, 1e6, 0.001,
                 "Smallest absolute lift you care about. E.g. $0.50 means you want to detect +$0.50 lift."),
        ], style={"display": "none"}),

        html.Hr(className="my-2"),

        # ── Common params ────────────────────────────────────────────────
        _row("Alpha (Significance Level)", "alpha", 0.05, 0.01, 0.20, 0.01,
             "Acceptable false-positive rate. 0.05 = 5%. Stop threshold = 1/alpha for mSPRT, 0.95 for Bayesian."),
        _row("Daily Traffic per Variant", "daily-traffic", 500, 10, 100_000, 10,
             "Users assigned to each group (control or treatment) per day."),
        _row("Days to Simulate", "n-days", 30, 5, 180, 1,
             "How many days of data to simulate. The chart builds day-by-day."),
        _row("Random Seed", "seed", 42, 0, 9999, 1,
             "Controls the random number generator. Change to see different simulated outcomes."),

        dbc.Button("Run Simulation", id="run-btn", color="primary",
                   className="w-100 mt-2", n_clicks=0),
    ]), className="shadow-sm")
