import plotly.graph_objects as go
import pandas as pd
from engines.ratio_engine import safe_val

def _get_series(df: pd.DataFrame, key: str) -> list:
    periods = [c for c in df.columns if str(c).lower() not in ["metric", "category", "line_item"]]
    return [safe_val(df, key, p) for p in periods]

def build_growth_trajectory_chart(df: pd.DataFrame):
    fig = go.Figure()
    if df is None or df.empty:
        fig.update_layout(title="No Financial Data Available")
        return fig

    periods = [str(c) for c in df.columns if str(c).lower() not in ["metric", "category", "line_item"]]
    rev = _get_series(df, "revenue")
    ebitda = _get_series(df, "ebitda")
    pat = _get_series(df, "pat")

    fig.add_trace(go.Bar(x=periods, y=rev, name="Gross Revenue", marker_color="#0f172a"))
    fig.add_trace(go.Scatter(x=periods, y=ebitda, name="Operating EBITDA", mode="lines+markers", line=dict(color="#f59e0b", width=3)))
    fig.add_trace(go.Scatter(x=periods, y=pat, name="Reported PAT", mode="lines+markers", line=dict(color="#10b981", width=2, dash="dot")))

    fig.update_layout(
        title="Top-Line & Operating Earnings Trajectory",
        barmode="group",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=40, b=20),
        template="plotly_white"
    )
    return fig

def build_margin_trend_chart(clean_num_ratios: pd.DataFrame):
    fig = go.Figure()
    if clean_num_ratios is None or clean_num_ratios.empty:
        fig.update_layout(title="No Ratio Data Available")
        return fig

    periods = [str(c) for c in clean_num_ratios.columns if str(c).lower() not in ["metric", "category", "line_item"]]
    for row in clean_num_ratios.index:
        if "margin" in str(row).lower():
            vals = [float(clean_num_ratios.loc[row, p]) for p in periods]
            fig.add_trace(go.Scatter(x=periods, y=vals, name=str(row), mode="lines+markers"))

    fig.update_layout(
        title="Operational Margins Trend (%)",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=40, b=20),
        template="plotly_white"
    )
    return fig

def build_cash_waterfall(df: pd.DataFrame, period: str):
    fig = go.Figure()
    if df is None or df.empty:
        fig.update_layout(title="No Cash Flow Data Available")
        return fig

    ebitda = safe_val(df, "ebitda", period)
    cfo = safe_val(df, "cfo", period)
    capex = safe_val(df, "capex", period)
    wc_movement = cfo - ebitda

    fig.add_trace(go.Waterfall(
        orientation="v",
        measure=["absolute", "relative", "total", "relative", "total"],
        x=["EBITDA", "WC & Tax Drift", "Operating CFO", "Reinvestment Capex", "Free Cash Flow"],
        y=[ebitda, wc_movement, 0, -capex, 0],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        decreasing={"marker": {"color": "#ef4444"}},
        increasing={"marker": {"color": "#10b981"}},
        totals={"marker": {"color": "#2563eb"}}
    ))
    fig.update_layout(
        title=f"Cash Conversion Walkway ({period})",
        showlegend=False,
        margin=dict(l=20, r=20, t=40, b=20),
        template="plotly_white"
    )
    return fig