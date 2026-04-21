"""
Plotly chart — 3 panels: mSPRT Λ, Bayesian P(treatment wins), classical p-value.
Works for both binary and continuous metrics.
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def build_figure(
    df: pd.DataFrame,
    alpha: float,
    tau: float,
    bayes_threshold: float = 0.95,
) -> go.Figure:
    """
    Build the 3-panel sequential testing chart.

    Args:
        df:               DataFrame with columns:
                          day, lambda_n, bayes_prob, p_value, verdict_msprt, verdict_bayes
        alpha:            mSPRT significance level (stop threshold = 1/alpha)
        tau:              mixing parameter (display only)
        bayes_threshold:  Bayesian stop threshold (default 0.95)
    """
    if df.empty:
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                            subplot_titles=("mSPRT Statistic (Λ) — Always Valid",
                                            "Bayesian P(Treatment Wins)",
                                            "Classical p-value — For Education Only"),
                            vertical_spacing=0.10, row_heights=[0.5, 0.25, 0.25])
        fig.update_layout(height=580, template="plotly_white",
                          annotations=[dict(text="Run simulation or enter data to see results",
                                            xref="paper", yref="paper", x=0.5, y=0.5,
                                            showarrow=False, font=dict(size=14, color="#adb5bd"))])
        return fig

    msprt_threshold = 1.0 / alpha

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        subplot_titles=(
            "mSPRT Statistic (Λ) — Always Valid",
            "Bayesian P(Treatment Wins)",
            "Classical p-value — <b>For Education Only</b>",
        ),
        vertical_spacing=0.10,
        row_heights=[0.50, 0.25, 0.25],
    )

    # ── Panel 1: mSPRT ──────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=df["day"], y=df["lambda_n"],
        mode="lines+markers", name="mSPRT Λ",
        line=dict(color="#0d6efd", width=2.5), marker=dict(size=4),
        hovertemplate="Day %{x}<br>Λ = %{y:.2f}<extra></extra>",
    ), row=1, col=1)

    fig.add_hline(y=msprt_threshold, line=dict(color="crimson", dash="dash", width=2),
                  annotation_text=f"Stop threshold = 1/α = {msprt_threshold:.0f}",
                  annotation_position="top right",
                  annotation_font=dict(color="crimson"), row=1, col=1)

    stop_rows = df[df["lambda_n"] >= msprt_threshold]
    if not stop_rows.empty:
        stop_day = int(stop_rows["day"].iloc[0])
        stop_val = stop_rows["lambda_n"].iloc[0]
        fig.add_annotation(x=stop_day, y=stop_val,
                           text=f"<b>STOP — Day {stop_day}</b>",
                           showarrow=True, arrowhead=2, arrowcolor="crimson",
                           font=dict(color="crimson", size=12),
                           bgcolor="rgba(255,255,255,0.9)", bordercolor="crimson",
                           row=1, col=1)

    last_msprt = df["verdict_msprt"].iloc[-1]
    msprt_color = "crimson" if "Significant" in last_msprt else "#198754"
    fig.add_annotation(x=0.01, y=0.97, xref="paper", yref="paper",
                       text=f"<b>mSPRT: {last_msprt}</b>",
                       showarrow=False, font=dict(size=12, color=msprt_color),
                       bgcolor="rgba(255,255,255,0.85)", bordercolor=msprt_color,
                       borderwidth=1, borderpad=4, align="left")

    # ── Panel 2: Bayesian ───────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=df["day"], y=df["bayes_prob"],
        mode="lines+markers", name="P(treatment wins)",
        line=dict(color="#6f42c1", width=2.5), marker=dict(size=4),
        hovertemplate="Day %{x}<br>P(treatment wins) = %{y:.3f}<extra></extra>",
    ), row=2, col=1)

    fig.add_hline(y=bayes_threshold, line=dict(color="#198754", dash="dash", width=1.5),
                  annotation_text=f"Treatment wins threshold = {bayes_threshold}",
                  annotation_position="top right",
                  annotation_font=dict(color="#198754"), row=2, col=1)

    fig.add_hline(y=1 - bayes_threshold, line=dict(color="crimson", dash="dash", width=1.5),
                  annotation_text=f"Control wins threshold = {1 - bayes_threshold:.2f}",
                  annotation_position="bottom right",
                  annotation_font=dict(color="crimson"), row=2, col=1)

    # Shade the credible region between the two thresholds (continue zone)
    fig.add_hrect(y0=1 - bayes_threshold, y1=bayes_threshold,
                  fillcolor="rgba(200,200,200,0.12)", layer="below", line_width=0,
                  row=2, col=1)

    last_bayes = df["verdict_bayes"].iloc[-1]
    bayes_color = "#198754" if "Treatment" in last_bayes else ("crimson" if "Control" in last_bayes else "#6f42c1")
    fig.add_annotation(x=0.01, y=0.72, xref="paper", yref="paper",
                       text=f"<b>Bayesian: {last_bayes}</b>",
                       showarrow=False, font=dict(size=12, color=bayes_color),
                       bgcolor="rgba(255,255,255,0.85)", bordercolor=bayes_color,
                       borderwidth=1, borderpad=4, align="left")

    # ── Panel 3: Classical p-value ──────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=df["day"], y=df["p_value"],
        mode="lines+markers", name="p-value (classical)",
        line=dict(color="#fd7e14", width=2), marker=dict(size=4),
        hovertemplate="Day %{x}<br>p = %{y:.4f}<extra></extra>",
    ), row=3, col=1)

    fig.add_hline(y=alpha, line=dict(color="crimson", dash="dash", width=1.5),
                  annotation_text=f"α = {alpha}",
                  annotation_position="top right",
                  annotation_font=dict(color="crimson"), row=3, col=1)

    # Highlight false-positive zone: p < alpha but mSPRT says continue
    early_fp = df[(df["p_value"] < alpha) & (df["lambda_n"] < msprt_threshold)]
    if not early_fp.empty:
        for _, row in early_fp.iterrows():
            fig.add_vrect(x0=row["day"] - 0.5, x1=row["day"] + 0.5,
                          fillcolor="rgba(255,0,0,0.08)", layer="below",
                          line_width=0, row=3, col=1)
        fig.add_annotation(x=early_fp["day"].mean(), y=alpha * 0.5,
                           text="False positive zone<br>(peeking would mislead here)",
                           showarrow=False, font=dict(size=9, color="crimson"),
                           bgcolor="rgba(255,255,255,0.85)", bordercolor="crimson",
                           row=3, col=1)

    fig.update_layout(
        height=640,
        template="plotly_white",
        legend=dict(orientation="h", y=-0.07, x=0.5, xanchor="center"),
        margin=dict(t=60, b=20, l=60, r=20),
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Λ_n", type="log", row=1, col=1)
    fig.update_yaxes(title_text="P(trt wins)", range=[0, 1], row=2, col=1)
    fig.update_yaxes(title_text="p-value", row=3, col=1)
    fig.update_xaxes(title_text="Day", row=3, col=1)

    return fig
