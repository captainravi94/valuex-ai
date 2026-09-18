import os
import re
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from engines.extraction_engine import DynamicReportExtractor
from engines.ratio_engine import compute_financial_ratios, safe_val
from engines.risk_engine import CFOForensicAudit
from engines.deep_analytics_engine import (
    calculate_piotroski_f_score,
    calculate_altman_z,
    perform_dupont_decomposition
)
from engines.data_ingestion_engine import ingest_from_excel_or_csv, fetch_from_ir_link
from engines.visualization_engine import (
    build_growth_trajectory_chart,
    build_margin_trend_chart
)
from engines.reconciliation_engine import FinancialReconciliationEngine
from engines.anomaly_engine import FinancialAnomalyEngine
from engines.working_capital_engine import WorkingCapitalBridgeEngine
from engines.notes_engine import FootnoteDisclosuresEngine
from engines.statistical_engine import FinancialStatisticalEngine
from ai.gemini_intelligence import GeminiFinancialAnalyst
from utils.pdf_viewer import display_evidence_drawer
from utils.pdf_exporter import InstitutionalPDFExporter

st.set_page_config(page_title="VALUEX AI | Institutional Financial Intelligence", page_icon="🏛️", layout="wide")

# ----------------- SESSION STATE PERSISTENCE -----------------
if "df_metrics" not in st.session_state:
    st.session_state.df_metrics = None
if "company_name" not in st.session_state:
    st.session_state.company_name = "Tata Motors Limited"
if "taxonomy" not in st.session_state:
    st.session_state.taxonomy = "Corporate / Industrial"
if "currency_symbol" not in st.session_state:
    st.session_state.currency_symbol = "₹"
if "unit_scale" not in st.session_state:
    st.session_state.unit_scale = "Cr"
if "active_nav_idx" not in st.session_state:
    st.session_state.active_nav_idx = 0

if "gemini_api_key" not in st.session_state:
    key_found = ""
    try:
        if "GEMINI_API_KEY" in st.secrets:
            key_found = st.secrets["GEMINI_API_KEY"]
    except Exception:
        key_found = os.environ.get("GEMINI_API_KEY", "")
    st.session_state.gemini_api_key = key_found

if "extra_meta" not in st.session_state:
    st.session_state.extra_meta = {}
if "pdf_bytes" not in st.session_state:
    st.session_state.pdf_bytes = None
if "pdf_meta" not in st.session_state:
    st.session_state.pdf_meta = {}

# ----------------- INSTITUTIONAL BENCHMARK PRESETS -----------------
PRESETS = {
    "Tata Motors Limited": {
        "name": "Tata Motors Limited",
        "taxonomy": "Corporate / Industrial",
        "currency": "₹",
        "scale": "Cr",
        "data": {
            "FY21": {"revenue": 249795.0, "ebitda": 32100.0, "pat": -13451.0, "total_borrowings": 135000.0, "cash_and_equivalents": 29000.0, "equity_share_capital": 765.0, "reserves": 54235.0, "total_equity": 55000.0, "total_assets": 260000.0, "total_liabilities": 260000.0, "trade_receivables": 12400.0, "cfo": 28900.0, "capex": 19500.0, "other_income": 2600.0},
            "FY22": {"revenue": 278454.0, "ebitda": 34800.0, "pat": -11441.0, "total_borrowings": 139000.0, "cash_and_equivalents": 30500.0, "equity_share_capital": 765.0, "reserves": 44235.0, "total_equity": 45000.0, "total_assets": 272000.0, "total_liabilities": 272000.0, "trade_receivables": 13100.0, "cfo": 30200.0, "capex": 21000.0, "other_income": 2900.0},
            "FY23": {"revenue": 345967.0, "ebitda": 45600.0, "pat": 2414.0, "total_borrowings": 125000.0, "cash_and_equivalents": 32000.0, "equity_share_capital": 765.0, "reserves": 64235.0, "total_equity": 65000.0, "total_assets": 285000.0, "total_liabilities": 285000.0, "trade_receivables": 14200.0, "cfo": 38500.0, "capex": 24000.0, "other_income": 3200.0},
            "FY24": {"revenue": 392000.0, "ebitda": 54800.0, "pat": 24200.0, "total_borrowings": 110000.0, "cash_and_equivalents": 38000.0, "equity_share_capital": 765.0, "reserves": 81235.0, "total_equity": 82000.0, "total_assets": 312000.0, "total_liabilities": 312000.0, "trade_receivables": 15800.0, "cfo": 44000.0, "capex": 28000.0, "other_income": 4100.0},
            "FY25": {"revenue": 437928.0, "ebitda": 62114.0, "pat": 31807.0, "total_borrowings": 96000.0, "cash_and_equivalents": 44000.0, "equity_share_capital": 765.0, "reserves": 104235.0, "total_equity": 105000.0, "total_assets": 345000.0, "total_liabilities": 345000.0, "trade_receivables": 21800.0, "cfo": 41200.0, "capex": 31500.0, "other_income": 4800.0}
        }
    },
    "Citigroup Inc.": {
        "name": "Citigroup Inc.",
        "taxonomy": "Banking / Financial Institution",
        "currency": "$",
        "scale": "M",
        "data": {
            "2021": {"revenue": 71884.0, "ebitda": 10200.0, "pat": 21952.0, "total_borrowings": 1280000.0, "cash_and_equivalents": 240000.0, "equity_share_capital": 18000.0, "reserves": 184000.0, "total_equity": 202000.0, "total_assets": 2290000.0, "total_liabilities": 2290000.0, "trade_receivables": 650000.0, "cfo": 16000.0, "capex": 2900.0, "other_income": 950.0},
            "2022": {"revenue": 75338.0, "ebitda": 10900.0, "pat": 14845.0, "total_borrowings": 1300000.0, "cash_and_equivalents": 250000.0, "equity_share_capital": 18000.0, "reserves": 187000.0, "total_equity": 205000.0, "total_assets": 2340000.0, "total_liabilities": 2340000.0, "trade_receivables": 670000.0, "cfo": 17100.0, "capex": 3000.0, "other_income": 1100.0},
            "2023": {"revenue": 78462.0, "ebitda": 11800.0, "pat": 9228.0, "total_borrowings": 1310000.0, "cash_and_equivalents": 260000.0, "equity_share_capital": 18000.0, "reserves": 190000.0, "total_equity": 208000.0, "total_assets": 2372000.0, "total_liabilities": 2372000.0, "trade_receivables": 687000.0, "cfo": 18200.0, "capex": 3100.0, "other_income": 1200.0},
            "2024": {"revenue": 81300.0, "ebitda": 15600.0, "pat": 12500.0, "total_borrowings": 1335000.0, "cash_and_equivalents": 275000.0, "equity_share_capital": 18000.0, "reserves": 194000.0, "total_equity": 212000.0, "total_assets": 2410000.0, "total_liabilities": 2410000.0, "trade_receivables": 712000.0, "cfo": 21400.0, "capex": 3400.0, "other_income": 1450.0},
            "2025": {"revenue": 84500.0, "ebitda": 18200.0, "pat": 14600.0, "total_borrowings": 1360000.0, "cash_and_equivalents": 290000.0, "equity_share_capital": 18000.0, "reserves": 200000.0, "total_equity": 218000.0, "total_assets": 2480000.0, "total_liabilities": 2480000.0, "trade_receivables": 745000.0, "cfo": 24500.0, "capex": 3600.0, "other_income": 1600.0}
        }
    }
}

# ----------------- SIDEBAR INGESTION HUB -----------------
with st.sidebar:
    st.header("🏛️ Ingestion Hub")
    ingest_mode = st.radio(
        "Source Pipeline:", 
        ["Upload Excel / CSV", "PDF Annual Report", "Investor Relations URL", "Benchmark Models"]
    )

    if ingest_mode == "Upload Excel / CSV":
        uploaded_sheet = st.file_uploader("Upload Statements (.xlsx / .csv)", type=["xlsx", "xls", "csv"])
        if uploaded_sheet and st.button("Ingest Spreadsheet", type="primary"):
            try:
                df_sheet, msg, comp_name, tax, cur, scale, extra = ingest_from_excel_or_csv(uploaded_sheet)
                if df_sheet is not None and not df_sheet.empty:
                    st.session_state.df_metrics = df_sheet
                    if comp_name:
                        st.session_state.company_name = comp_name
                    if tax:
                        st.session_state.taxonomy = tax
                    st.session_state.currency_symbol = cur or "₹"
                    st.session_state.unit_scale = scale or "Cr"
                    st.session_state.extra_meta = extra or {}
                    st.session_state.pdf_bytes = None
                    st.session_state.pdf_meta = {}
                    st.session_state.active_nav_idx = 0
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(f"Spreadsheet Parsing Notice: {msg}")
            except Exception as e:
                st.error(f"Ingestion Error: {str(e)}")

    elif ingest_mode == "PDF Annual Report":
        uploaded_pdf = st.file_uploader("Upload Annual Report (.pdf)", type=["pdf"])
        if uploaded_pdf and st.button("Parse PDF Document", type="primary"):
            with st.spinner("Executing document layout analysis & statement extraction..."):
                try:
                    pdf_bytes_data = uploaded_pdf.read()
                    extractor = DynamicReportExtractor(pdf_bytes_data)
                    df_extr, comp, tax, pdf_meta = extractor.extract_and_spread()
                    if df_extr is not None and not df_extr.empty:
                        st.session_state.df_metrics = df_extr
                        st.session_state.company_name = comp
                        st.session_state.taxonomy = tax
                        st.session_state.currency_symbol = "$" if "citi" in comp.lower() or "inc" in comp.lower() else "₹"
                        st.session_state.unit_scale = "M" if "citi" in comp.lower() or "inc" in comp.lower() else "Cr"
                        st.session_state.pdf_bytes = pdf_bytes_data
                        st.session_state.pdf_meta = pdf_meta
                        st.session_state.extra_meta = {
                            "unit_consistent": True,
                            "source_doc": uploaded_pdf.name,
                            "detected_pages": pdf_meta.get("detected_pages", {})
                        }
                        st.session_state.active_nav_idx = 0
                        st.success(f"Successfully extracted financial statements for {comp}")
                        st.rerun()
                    else:
                        st.error("Failed to parse tabular financial statements from this PDF.")
                except Exception as e:
                    st.error(f"PDF Extraction Error: {str(e)}")

    elif ingest_mode == "Investor Relations URL":
        ir_url = st.text_input("Enter Financial Table URL:", value="https://www.investor.gov")
        if st.button("Scrape Tables"):
            with st.spinner("Fetching tables from target domain..."):
                tbl, msg = fetch_from_ir_link(ir_url)
                if tbl is not None:
                    st.success(msg)
                    st.dataframe(tbl.head(4))
                else:
                    st.warning(msg)

    elif ingest_mode == "Benchmark Models":
        choice = st.selectbox("Select Institution:", list(PRESETS.keys()))
        if st.button("Load Institution"):
            sel = PRESETS[choice]
            st.session_state.df_metrics = pd.DataFrame(sel["data"])
            st.session_state.company_name = sel["name"]
            st.session_state.taxonomy = sel["taxonomy"]
            st.session_state.currency_symbol = sel["currency"]
            st.session_state.unit_scale = sel["scale"]
            st.session_state.extra_meta = {}
            st.session_state.pdf_bytes = None
            st.session_state.pdf_meta = {}
            st.session_state.active_nav_idx = 0
            st.rerun()

    st.markdown("---")
    st.header("🤖 AI Copilot Config")
    user_api_key = st.text_input("Gemini API Key (Optional)", type="password", value=st.session_state.gemini_api_key)
    if user_api_key != st.session_state.gemini_api_key:
        st.session_state.gemini_api_key = user_api_key

# ----------------- FALLBACK INITIALIZATION -----------------
if st.session_state.df_metrics is None:
    def_data = PRESETS["Tata Motors Limited"]
    st.session_state.df_metrics = pd.DataFrame(def_data["data"])
    st.session_state.company_name = def_data["name"]
    st.session_state.taxonomy = def_data["taxonomy"]
    st.session_state.currency_symbol = def_data["currency"]
    st.session_state.unit_scale = def_data["scale"]

df_metrics = st.session_state.df_metrics

if "metric" in df_metrics.columns:
    df_metrics = df_metrics.set_index("metric")
st.session_state.df_metrics = df_metrics

company_name = st.session_state.company_name
taxonomy = st.session_state.taxonomy
unit = st.session_state.currency_symbol
scale = st.session_state.unit_scale

periods = [
    str(c) for c in df_metrics.columns 
    if str(c).strip().lower() not in ["metric", "canonical_metric", "category", "line_item", "particulars", "nan"]
    and not str(c).lower().startswith("unnamed")
]

if not periods:
    periods = [str(c) for c in df_metrics.columns]

with st.sidebar:
    st.markdown("---")
    default_period_idx = max(0, len(periods) - 1)
    target_period = st.selectbox("Valuation Period", periods, index=default_period_idx)

# ----------------- BASE METRIC CALCULATIONS -----------------
curr_rev = safe_val(df_metrics, "revenue", target_period)
curr_ebitda = safe_val(df_metrics, "ebitda", target_period)
curr_pat = safe_val(df_metrics, "pat", target_period)
curr_cfo = safe_val(df_metrics, "cfo", target_period)
curr_capex = safe_val(df_metrics, "capex", target_period)
curr_fcf = curr_cfo - curr_capex
curr_equity = safe_val(df_metrics, "total_equity", target_period)
curr_other_inc = safe_val(df_metrics, "other_income", target_period)

df_ratios = compute_financial_ratios(df_metrics, unit_label=scale)
risk_alerts = CFOForensicAudit.analyze_health(df_metrics)
dupont = perform_dupont_decomposition(df_metrics, target_period)
altman = calculate_altman_z(df_metrics, target_period, is_bank=(taxonomy != "Corporate / Industrial"), extra_meta=st.session_state.extra_meta)
piotroski = calculate_piotroski_f_score(df_metrics)
anomalies = FinancialAnomalyEngine.detect_anomalies(df_metrics, target_period, st.session_state.extra_meta)
recon = FinancialReconciliationEngine.audit_statement_integrity(df_metrics, target_period, st.session_state.extra_meta)

# ----------------- MAIN TITLE & TOP KPI BANNER -----------------
st.title(f"🏛️ VALUEX AI | {company_name}")
st.caption(f"Taxonomy: **{taxonomy}** | Target Cycle: **{target_period}** | Document-to-Decision Institutional Platform")

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
ebitda_m = (curr_ebitda / curr_rev * 100) if curr_rev > 0 else 0.0
kpi1.metric("Reported Gross Sales", f"{unit}{curr_rev:,.0f} {scale}", delta=f"{ebitda_m:.2f}% Operating Margin" if curr_rev > 0 else None)
kpi2.metric("Operating EBITDA", f"{unit}{curr_ebitda:,.0f} {scale}")

if curr_other_inc > (curr_ebitda * 0.5) and curr_ebitda > 0:
    kpi3.metric("Reported PAT", f"{unit}{curr_pat:,.0f} {scale}", delta=f"⚠️ Incl. {unit}{curr_other_inc:,.0f} {scale} Other Income", delta_color="off")
else:
    kpi3.metric("Reported PAT", f"{unit}{curr_pat:,.0f} {scale}")

if curr_equity < 0:
    kpi4.metric("Derived Free Cash Flow", f"{unit}{curr_fcf:,.0f} {scale}", delta=f"⚠️ Net Worth: {unit}{curr_equity:,.0f} {scale}", delta_color="inverse")
else:
    cash_qual = (curr_cfo / curr_pat) if curr_pat > 0 else 0.0
    kpi4.metric("Derived Free Cash Flow", f"{unit}{curr_fcf:,.0f} {scale}", delta=f"Cash Quality: {cash_qual:.2f}x CFO/PAT" if curr_pat > 0 else None)

st.markdown("---")

# ----------------- 11-FEATURE VISIBLE COMMAND DOCK -----------------
SECTIONS = [
    "📑 Spreading",
    "🌉 Cash Flow Bridge",
    "🎛️ Scenario Analysis",
    "💰 DCF Valuation",
    "🧪 Hypothesis & Stat Tests",
    "🧠 Seven-Tier AI",
    "⚡ Anomaly Engine",
    "🔬 Model Validation",
    "📈 Visual Charts",
    "🚨 CFO Radar",
    "📄 Boardroom Dossier"
]

c_step1, c_dock, c_step2 = st.columns([1, 8, 1])

with c_step1:
    if st.button("◀ Prev", use_container_width=True, disabled=(st.session_state.active_nav_idx == 0)):
        st.session_state.active_nav_idx = max(0, st.session_state.active_nav_idx - 1)
        st.rerun()

with c_step2:
    if st.button("Next ▶", use_container_width=True, disabled=(st.session_state.active_nav_idx == len(SECTIONS) - 1)):
        st.session_state.active_nav_idx = min(len(SECTIONS) - 1, st.session_state.active_nav_idx + 1)
        st.rerun()

with c_dock:
    chosen = st.radio(
        "Feature Navigation",
        options=SECTIONS,
        index=st.session_state.active_nav_idx,
        horizontal=True,
        label_visibility="collapsed"
    )
    if chosen != SECTIONS[st.session_state.active_nav_idx]:
        st.session_state.active_nav_idx = SECTIONS.index(chosen)
        st.rerun()

selected_section = SECTIONS[st.session_state.active_nav_idx]
st.progress((st.session_state.active_nav_idx + 1) / len(SECTIONS), text=f"Feature {st.session_state.active_nav_idx + 1} of {len(SECTIONS)}: {selected_section}")
st.markdown("---")

# ----------------- 1. SPREADING -----------------
if selected_section == "📑 Spreading":
    st.subheader("📑 Interactive Statement Spreading & Line-Item Filtering")
    
    sp_col1, sp_col2 = st.columns(2)
    with sp_col1:
        max_periods = len(periods)
        spread_window = st.slider(
            "⏳ Historical Window (Periods)", 
            min_value=min(2, max_periods), 
            max_value=max_periods, 
            value=max_periods, 
            step=1
        )
    with sp_col2:
        max_val = float(df_metrics[periods].max().max()) if not df_metrics.empty else 1000.0
        mat_step = max(round(max_val / 100.0, -1), 10.0)
        materiality_thresh = st.slider(
            f"🔍 Materiality Filter: Hide Line Items with Peak < ({unit} {scale})", 
            min_value=0.0, 
            max_value=float(round(max_val * 0.25, -1)), 
            value=0.0, 
            step=float(mat_step)
        )

    selected_periods = periods[-spread_window:]
    
    if materiality_thresh > 0:
        filtered_metrics = df_metrics[df_metrics[selected_periods].abs().max(axis=1) >= materiality_thresh]
    else:
        filtered_metrics = df_metrics

    spread_display_df = filtered_metrics[selected_periods]
    ratios_display_df = df_ratios[selected_periods]

    col_s1, col_s2 = st.columns([1.2, 1], gap="large")
    with col_s1:
        st.markdown(f"**Financial Statement Matrix** ({len(spread_display_df)} Line Items Visible)")
        def format_currency_cell(val):
            try:
                num = float(val)
                if abs(num) >= 1000:
                    return f"{unit}{num:,.0f} {scale}"
                return f"{unit}{num:,.2f} {scale}"
            except (ValueError, TypeError):
                return str(val)
        st.dataframe(spread_display_df.style.format(format_currency_cell), use_container_width=True, height=450)

    with col_s2:
        st.markdown(f"**Institutional Ratios** ({spread_window} Periods)")
        st.dataframe(ratios_display_df, use_container_width=True, height=450)

# ----------------- 2. CASH FLOW & WORKING CAPITAL BRIDGE -----------------
elif selected_section == "🌉 Cash Flow Bridge":
    st.subheader("🌉 Working Capital Movement & Cash Realization Bridge")
    st.caption("Adjust operating assumptions in real-time to inspect working capital sensitivity.")

    wc_bridge = WorkingCapitalBridgeEngine.compute_bridge(df_metrics, target_period)

    wc_c1, wc_c2, wc_c3, wc_c4 = st.columns(4)
    with wc_c1:
        dso_days_shift = st.slider("Debtor Days (DSO) Delta", min_value=-45, max_value=45, value=0, step=1)
    with wc_c2:
        inv_turn_shift = st.slider("Inventory Holding Drift (Days)", min_value=-45, max_value=45, value=0, step=1)
    with wc_c3:
        dpo_days_shift = st.slider("Payable Terms (DPO) Shift", min_value=-45, max_value=45, value=0, step=1)
    with wc_c4:
        capex_modifier = st.slider("Capex Budget Scale (%)", min_value=-50.0, max_value=100.0, value=0.0, step=5.0)

    daily_sales = curr_rev / 365.0 if curr_rev > 0 else 100.0
    rec_cash_impact = - (dso_days_shift * daily_sales)
    inv_cash_impact = - (inv_turn_shift * (daily_sales * 0.65))
    pay_cash_impact = (dpo_days_shift * (daily_sales * 0.50))
    total_slider_wc_shift = rec_cash_impact + inv_cash_impact + pay_cash_impact

    base_ebitda = wc_bridge.get("ebitda", curr_ebitda)
    base_wc_movement = wc_bridge.get("net_wc_movement", curr_cfo - curr_ebitda)
    adj_wc_movement = base_wc_movement + total_slider_wc_shift
    adj_cfo = base_ebitda + adj_wc_movement
    adj_capex = curr_capex * (1.0 + capex_modifier / 100.0)
    adj_fcf = adj_cfo - adj_capex

    c_b1, c_b2 = st.columns([1.3, 1])
    with c_b1:
        fig_dyn_waterfall = go.Figure(go.Waterfall(
            orientation="v",
            measure=["absolute", "relative", "relative", "relative", "relative", "total", "relative", "total"],
            x=[
                "EBITDA", 
                "Base WC Movement", 
                "DSO Impact", 
                "Inventory Impact", 
                "DPO Terms", 
                "Simulated CFO", 
                "Capex Outlay", 
                "Simulated FCF"
            ],
            y=[
                base_ebitda, 
                base_wc_movement, 
                rec_cash_impact, 
                inv_cash_impact, 
                pay_cash_impact, 
                0, 
                -adj_capex, 
                0
            ],
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            decreasing={"marker": {"color": "#ef4444"}},
            increasing={"marker": {"color": "#10b981"}},
            totals={"marker": {"color": "#2563eb"}}
        ))
        fig_dyn_waterfall.update_layout(
            title=f"Dynamic Cash Realization Bridge ({target_period})",
            showlegend=False,
            margin=dict(l=20, r=20, t=40, b=20),
            template="plotly_white",
            height=340
        )
        st.plotly_chart(fig_dyn_waterfall, use_container_width=True)

    with c_b2:
        st.markdown("**Simulated Cash Realization Impact**")
        st.metric("Simulated Operating CFO", f"{unit}{adj_cfo:,.0f} {scale}", delta=f"{unit}{(adj_cfo - curr_cfo):+,.0f} {scale} vs Reported")
        st.metric("Simulated Free Cash Flow", f"{unit}{adj_fcf:,.0f} {scale}", delta=f"{unit}{(adj_fcf - curr_fcf):+,.0f} {scale} vs Reported")
        st.caption(f"DSO Movement Impact: **{unit}{rec_cash_impact:+,.0f} {scale}**")
        st.caption(f"Inventory Movement Impact: **{unit}{inv_cash_impact:+,.0f} {scale}**")
        st.caption(f"Creditor Payment Term Impact: **{unit}{pay_cash_impact:+,.0f} {scale}**")

    st.markdown("---")
    st.subheader("🔍 Interactive Footnote & Schedule Drilldown")
    note_choice = st.selectbox("Select Footnote Schedule to Audit:", ["Other Income Note", "Borrowings Note"])
    note_data = FootnoteDisclosuresEngine.drilldown_note(
        note_category=note_choice,
        company_name=company_name,
        context_data={"other_income": curr_other_inc, "total_borrowings": safe_val(df_metrics, "total_borrowings", target_period)}
    )

    st.markdown(f"**{note_data['note_title']}** (Reported Total: **{unit}{note_data['reported_total']:,.0f} {scale}**)")
    st.caption(f"Accounting Classification: `{note_data['accounting_treatment']}`")
    if note_data["components"]:
        comp_df = pd.DataFrame(note_data["components"])
        comp_df["Amount"] = comp_df["Amount"].apply(lambda v: f"{unit}{v:,.0f} {scale}")
        st.dataframe(comp_df, use_container_width=True)

# ----------------- 3. SCENARIO ANALYSIS -----------------
elif selected_section == "🎛️ Scenario Analysis":
    st.subheader("🎛️ FP&A Multi-Driver Stress Testing & Scenario Console")
    st.caption("Tune operational parameters to model base, upside, and downside projections.")

    b_rev = curr_rev
    b_ebitda = curr_ebitda
    b_pat = curr_pat
    b_debt = safe_val(df_metrics, "total_borrowings", target_period)
    b_cfo = curr_cfo
    b_capex = curr_capex

    col_d1, col_d2, col_d3 = st.columns(3)
    with col_d1:
        st.markdown("**1. Top-Line Trajectory & Pricing Power**")
        rev_shock = st.slider("Revenue Growth Delta (%)", min_value=-50.0, max_value=200.0, value=15.0, step=1.0)
        margin_shift = st.slider("Operating Margin Shift (bps)", min_value=-800, max_value=800, value=50, step=25)

    with col_d2:
        st.markdown("**2. Cost Structure & Working Capital**")
        cost_shock = st.slider("Input Cost / Inflation (%)", min_value=0.0, max_value=25.0, value=2.0, step=0.5)
        rec_drag = st.slider("Receivables / DSO Extension (Days)", min_value=-30, max_value=60, value=5, step=1)

    with col_d3:
        st.markdown("**3. Capital Structure & Reinvestment**")
        rate_shock = st.slider("Effective Debt Rate Shock (bps)", min_value=-150, max_value=400, value=25, step=25)
        capex_growth = st.slider("Capex Expansion Budget (%)", min_value=-50.0, max_value=150.0, value=10.0, step=5.0)

    def run_sim(rev_g, m_bps, cost_inf, dso, rate_bps, cap_g):
        s_rev = max(b_rev * (1.0 + rev_g / 100.0), 0.0)
        base_m = (b_ebitda / b_rev) if b_rev > 0 else 0.15
        s_m = max(base_m + (m_bps / 10000.0) - (cost_inf / 100.0), 0.01)
        s_ebitda = s_rev * s_m
        int_drag = b_debt * (rate_bps / 10000.0)
        ebitda_delta = s_ebitda - b_ebitda
        pretax_delta = ebitda_delta - int_drag
        tax_impact = pretax_delta * 0.25
        s_pat = b_pat + (pretax_delta - tax_impact)
        wc_drag = (s_rev / 365.0) * dso
        s_cfo = b_cfo + (s_ebitda - b_ebitda) - wc_drag
        s_capex = max(b_capex * (1.0 + cap_g / 100.0), 0.0)
        s_fcf = s_cfo - s_capex
        return {
            "Revenue": s_rev, "EBITDA": s_ebitda, "Margin": s_m * 100.0,
            "PAT": s_pat, "PAT_Margin": (s_pat / s_rev * 100.0) if s_rev > 0 else 0.0,
            "CFO": s_cfo, "Capex": s_capex, "FCF": s_fcf
        }

    worst = run_sim(rev_g=-20.0, m_bps=-300.0, cost_inf=4.0, dso=20, rate_bps=150.0, cap_g=-15.0)
    base = run_sim(rev_g=0.0, m_bps=0.0, cost_inf=0.0, dso=0, rate_bps=0.0, cap_g=0.0)
    sim = run_sim(rev_g=rev_shock, m_bps=float(margin_shift), cost_inf=cost_shock, dso=rec_drag, rate_bps=float(rate_shock), cap_g=capex_growth)
    best = run_sim(rev_g=max(rev_shock * 1.5, 30.0), m_bps=200.0, cost_inf=0.0, dso=-10, rate_bps=-50.0, cap_g=10.0)

    scenario_matrix_df = pd.DataFrame({
        "Financial Metric": [
            "Top-Line Gross Revenue", "Operating Profit (EBITDA)", "EBITDA Margin (%)",
            "Net Profit (PAT)", "PAT Margin (%)", "Operating Cash Flow (CFO)",
            "Capital Expenditure (Capex)", "Analytically Derived Free Cash Flow"
        ],
        "Downside Scenario": [
            f"{unit}{worst['Revenue']:,.0f} {scale}", f"{unit}{worst['EBITDA']:,.0f} {scale}", f"{worst['Margin']:.2f}%",
            f"{unit}{worst['PAT']:,.0f} {scale}", f"{worst['PAT_Margin']:.2f}%", f"{unit}{worst['CFO']:,.0f} {scale}",
            f"{unit}{worst['Capex']:,.0f} {scale}", f"{unit}{worst['FCF']:,.0f} {scale}"
        ],
        "Base Case (Audited)": [
            f"{unit}{base['Revenue']:,.0f} {scale}", f"{unit}{base['EBITDA']:,.0f} {scale}", f"{base['Margin']:.2f}%",
            f"{unit}{base['PAT']:,.0f} {scale}", f"{base['PAT_Margin']:.2f}%", f"{unit}{base['CFO']:,.0f} {scale}",
            f"{unit}{base['Capex']:,.0f} {scale}", f"{unit}{base['FCF']:,.0f} {scale}"
        ],
        "Dynamic Model (Tuned)": [
            f"{unit}{sim['Revenue']:,.0f} {scale}", f"{unit}{sim['EBITDA']:,.0f} {scale}", f"{sim['Margin']:.2f}%",
            f"{unit}{sim['PAT']:,.0f} {scale}", f"{sim['PAT_Margin']:.2f}%", f"{unit}{sim['CFO']:,.0f} {scale}",
            f"{unit}{sim['Capex']:,.0f} {scale}", f"{unit}{sim['FCF']:,.0f} {scale}"
        ],
        "Upside Scenario": [
            f"{unit}{best['Revenue']:,.0f} {scale}", f"{unit}{best['EBITDA']:,.0f} {scale}", f"{best['Margin']:.2f}%",
            f"{unit}{best['PAT']:,.0f} {scale}", f"{best['PAT_Margin']:.2f}%", f"{unit}{best['CFO']:,.0f} {scale}",
            f"{unit}{best['Capex']:,.0f} {scale}", f"{unit}{best['FCF']:,.0f} {scale}"
        ]
    })

    st.markdown("---")
    st.subheader("📊 Comparative Scenario Underwriting Matrix")
    st.dataframe(scenario_matrix_df, use_container_width=True)

# ----------------- 4. DCF VALUATION -----------------
elif selected_section == "💰 DCF Valuation":
    st.subheader("💰 5-Year Institutional DCF Valuation Workbench")
    st.caption("Estimate intrinsic Enterprise Value and Equity Value using interactive cost of capital parameters.")

    dcf_c1, dcf_c2, dcf_c3, dcf_c4 = st.columns(4)
    with dcf_c1:
        wacc_rate = st.slider("WACC Discount Rate (%)", min_value=7.0, max_value=20.0, value=11.5, step=0.25)
    with dcf_c2:
        terminal_growth = st.slider("Terminal Growth Rate (%)", min_value=1.5, max_value=6.0, value=4.0, step=0.25)
    with dcf_c3:
        fcf_growth_rate = st.slider("5-Yr FCF CAGR (%)", min_value=-10.0, max_value=35.0, value=8.0, step=0.5)
    with dcf_c4:
        shares_out = st.slider("Diluted Shares Outstanding (Cr)", min_value=10.0, max_value=1500.0, value=330.0, step=10.0)

    starting_fcf = max(curr_fcf, 500.0)
    years = [f"Year {i}" for i in range(1, 6)]
    fcf_projections = []
    pv_factors = []
    discounted_fcf = []

    for i in range(1, 6):
        proj = starting_fcf * ((1.0 + fcf_growth_rate / 100.0) ** i)
        pv = 1.0 / ((1.0 + wacc_rate / 100.0) ** i)
        fcf_projections.append(proj)
        pv_factors.append(pv)
        discounted_fcf.append(proj * pv)

    pv_discrete_fcf = sum(discounted_fcf)
    terminal_fcf = fcf_projections[-1] * (1.0 + terminal_growth / 100.0)
    wacc_spread = max((wacc_rate - terminal_growth) / 100.0, 0.01)
    terminal_value = terminal_fcf / wacc_spread
    pv_terminal_value = terminal_value * pv_factors[-1]

    enterprise_value = pv_discrete_fcf + pv_terminal_value
    net_debt = max(safe_val(df_metrics, "total_borrowings", target_period) - safe_val(df_metrics, "cash_and_equivalents", target_period), 0.0)
    implied_equity_value = max(enterprise_value - net_debt, 0.0)
    fair_value_per_share = (implied_equity_value / shares_out) if shares_out > 0 else 0.0

    dcf_kpi1, dcf_kpi2, dcf_kpi3, dcf_kpi4 = st.columns(4)
    dcf_kpi1.metric("Enterprise Value", f"{unit}{enterprise_value:,.0f} {scale}")
    dcf_kpi2.metric("PV of Terminal Value", f"{unit}{pv_terminal_value:,.0f} {scale}")
    dcf_kpi3.metric("Implied Equity Value", f"{unit}{implied_equity_value:,.0f} {scale}")
    dcf_kpi4.metric("Fair Value / Share", f"{unit}{fair_value_per_share:,.1f}")

    dcf_forecast_df = pd.DataFrame({
        "Period": years,
        f"Projected FCF ({unit} {scale})": [f"{v:,.0f}" for v in fcf_projections],
        "PV Discount Factor": [f"{pv:.4f}" for pv in pv_factors],
        f"Discounted PV ({unit} {scale})": [f"{dfcf:,.0f}" for dfcf in discounted_fcf]
    })
    st.markdown("---")
    st.dataframe(dcf_forecast_df, use_container_width=True)

# ----------------- 5. HYPOTHESIS & STATISTICAL TESTS -----------------
elif selected_section == "🧪 Hypothesis & Stat Tests":
    st.subheader("🧪 Financial Hypothesis & Statistical Testing Suite")
    st.caption("Apply parametric inferential statistics (Z-scores, Student's t-Test, Snedecor's F-Test) to audit earnings quality and variance stability.")

    stat_tab1, stat_tab2, stat_tab3 = st.tabs([
        "1. Metric Z-Score Outlier Analysis",
        "2. Student's t-Test (Mean & Spread Significance)",
        "3. Snedecor's F-Test (Volatility & Variance Ratio)"
    ])

    with stat_tab1:
        st.markdown("**Gaussian Z-Score Standardization ($Z = \\frac{x - \\mu}{\\sigma}$)**")
        metric_choice = st.selectbox("Select Line Item to Audit:", ["revenue", "ebitda", "pat", "cfo", "capex", "total_borrowings"])
        z_res = FinancialStatisticalEngine.compute_metric_z_scores(df_metrics, metric_choice)
        if z_res.get("status") == "SUCCESS":
            zc1, zc2 = st.columns([1, 2])
            with zc1:
                st.metric("Historical Mean (μ)", f"{unit}{z_res['mean']:,.0f} {scale}")
                st.metric("Std Deviation (σ)", f"{unit}{z_res['std']:,.0f} {scale}")
            with zc2:
                st.dataframe(pd.DataFrame(z_res["table"]), use_container_width=True)
        else:
            st.info("Insufficient historical cycles to calculate standard deviations.")

    with stat_tab2:
        st.markdown("**Student's t-Test for Statistical Significance**")
        t_type = st.radio("Test Design:", ["One-Sample: Mean Growth vs 0.0", "Two-Sample: Metric A vs Metric B Growth"], horizontal=True)
        if "One-Sample" in t_type:
            t_metric = st.selectbox("Metric to Test:", ["revenue", "ebitda", "cfo", "pat"], key="t_m1")
            t_res = FinancialStatisticalEngine.run_student_t_test(df_metrics, t_metric)
        else:
            tc1, tc2 = st.columns(2)
            with tc1:
                m_a = st.selectbox("Primary Metric (A):", ["revenue", "cfo"], key="t_ma")
            with tc2:
                m_b = st.selectbox("Benchmark Metric (B):", ["pat", "ebitda"], key="t_mb")
            t_res = FinancialStatisticalEngine.run_student_t_test(df_metrics, m_a, m_b)

        if "t_statistic" in t_res:
            tm1, tm2, tm3 = st.columns(3)
            tm1.metric("t-Statistic", t_res["t_statistic"])
            tm2.metric("p-Value", t_res["p_value"])
            tm3.metric("Null Hypothesis (H0)", "REJECTED (p < 0.05)" if t_res["h0_rejected"] else "ACCEPTED (p >= 0.05)")
            if t_res["h0_rejected"]:
                st.warning(f"👉 **Statistical Verdict:** {t_res['interpretation']}")
            else:
                st.success(f"👉 **Statistical Verdict:** {t_res['interpretation']}")

    with stat_tab3:
        st.markdown("**Snedecor's F-Test for Variance & Volatility Divergence ($F = \\frac{s_1^2}{s_2^2}$)**")
        fc1, fc2 = st.columns(2)
        with fc1:
            f_m1 = st.selectbox("Sample 1 Variance (s1²):", ["cfo", "revenue"], key="f_m1")
        with fc2:
            f_m2 = st.selectbox("Sample 2 Variance (s2²):", ["pat", "ebitda"], key="f_m2")
        f_res = FinancialStatisticalEngine.run_f_test_variance(df_metrics, f_m1, f_m2)

        if "f_statistic" in f_res:
            fm1, fm2, fm3 = st.columns(3)
            fm1.metric("F-Statistic", f_res["f_statistic"])
            fm2.metric("p-Value", f_res["p_value"])
            fm3.metric("Variance Parity", "UNEQUAL (p < 0.05)" if f_res["h0_rejected"] else "HOMOGENEOUS")
            st.info(f"• Sample 1 Variance: **{f_res['variance_a']:,.0f}** | Sample 2 Variance: **{f_res['variance_b']:,.0f}**")
            st.markdown(f"👉 **Auditor Directive:** {f_res['interpretation']}")

# ----------------- 6. SEVEN-TIER AI -----------------
elif selected_section == "🧠 Seven-Tier AI":
    st.subheader(f"🧠 Forensic Lineage Appraisal: {company_name} ({target_period})")
    analyst = GeminiFinancialAnalyst(api_key=st.session_state.gemini_api_key)
    dossier = analyst.generate_boardroom_dossier(
        company_name, target_period, df_metrics, df_ratios, altman, piotroski, risk_alerts, {"other_income": curr_other_inc, "book_equity": curr_equity}
    )
    if dossier.get("status") == "LIVE_GEMINI":
        st.success("⚡ Live Gemini AI Intelligence Active")
    else:
        st.info("ℹ️ Running on Institutional Deterministic Lineage Engine.")
    st.markdown(dossier.get("content", ""))

# ----------------- 7. ANOMALY ENGINE -----------------
elif selected_section == "⚡ Anomaly Engine":
    st.subheader("⚡ Automated Financial Anomaly Scanner")
    if not anomalies:
        st.success("✓ No critical statistical or accounting anomalies detected for this cycle.")
    else:
        for anom in anomalies:
            st.error(f"**[{anom['id']}] {anom['title']} ({anom['severity']})**")
            st.write(f"• **Discovered Metric:** `{anom['metric']}`")
            st.write(f"• **Reported Evidence:** {anom['evidence']}")
            st.caption(f"👉 **Investigation Required:** {anom['investigation']}")
            st.markdown("---")

# ----------------- 8. MODEL VALIDATION -----------------
elif selected_section == "🔬 Model Validation":
    st.subheader("🔬 Financial Model Validation & Threshold Controls")
    
    mod_c1, mod_c2 = st.columns(2)
    with mod_c1:
        z_distress_cut = st.slider("Altman Z Distress Cutoff", min_value=1.0, max_value=2.5, value=1.81, step=0.05)
    with mod_c2:
        piot_cut = st.slider("Piotroski Quality Pass Benchmark", min_value=4, max_value=9, value=6, step=1)

    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        z_val = altman.get("z_score", 0.0)
        z_status = "Safe Zone" if z_val >= 2.99 else ("Grey Zone" if z_val >= z_distress_cut else "Distress Warning")
        st.markdown("**Altman Z-Score Model**")
        st.metric(label="Calculated Z-Score", value=z_val, delta=z_status)
        st.caption(f"Distress threshold tuned to {z_distress_cut:.2f}")

    with col_m2:
        passes = piotroski.get("passing_count", 0)
        p_status = "Investment Grade" if passes >= piot_cut else "Watchlist"
        st.markdown("**Piotroski Health Index**")
        st.metric(label="Passing Signals", value=f"{passes} Passes", delta=p_status)
        st.caption(f"Benchmark threshold tuned to {piot_cut}/9")

    with col_m3:
        st.markdown("**DuPont Decomposition**")
        st.write(f"• **Net Margin:** {dupont.get('Net Profit Margin (%)', 0.0)}%")
        st.write(f"• **Asset Turnover:** {dupont.get('Asset Turnover (x)', 0.0)}x")
        st.write(f"👉 **ROE:** **{dupont.get('Decomposed ROE (%)', '0.00%')}**")

    st.markdown("---")
    if altman.get("formula_audit"):
        st.dataframe(pd.DataFrame(altman["formula_audit"]), use_container_width=True)

# ----------------- 9. VISUAL CHARTS -----------------
elif selected_section == "📈 Visual Charts":
    st.subheader("📈 Institutional Trajectory & Cash Conversion Charts")
    c_left, c_right = st.columns(2)
    with c_left:
        st.plotly_chart(build_growth_trajectory_chart(df_metrics), use_container_width=True)
        numeric_ratios = compute_financial_ratios(df_metrics)
        clean_num_ratios = pd.DataFrame(index=numeric_ratios.index, columns=numeric_ratios.columns)
        for row in numeric_ratios.index:
            for col in numeric_ratios.columns:
                val_str = str(numeric_ratios.loc[row, col]).replace("%", "").replace("x", "").replace("days", "").replace(",", "").strip()
                try:
                    clean_num_ratios.loc[row, col] = float(val_str)
                except ValueError:
                    clean_num_ratios.loc[row, col] = 0.0
        st.plotly_chart(build_margin_trend_chart(clean_num_ratios), use_container_width=True)
    with c_right:
        ebitda_val = safe_val(df_metrics, "ebitda", target_period)
        cfo_val = safe_val(df_metrics, "cfo", target_period)
        capex_val = safe_val(df_metrics, "capex", target_period)
        wc_movement_val = cfo_val - ebitda_val

        static_wf = go.Figure(go.Waterfall(
            orientation="v",
            measure=["absolute", "relative", "total", "relative", "total"],
            x=["EBITDA", "WC & Tax Drift", "Operating CFO", "Reinvestment Capex", "Free Cash Flow"],
            y=[ebitda_val, wc_movement_val, 0, -capex_val, 0],
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            decreasing={"marker": {"color": "#ef4444"}},
            increasing={"marker": {"color": "#10b981"}},
            totals={"marker": {"color": "#2563eb"}}
        ))
        static_wf.update_layout(title=f"Audited Statement Waterfall ({target_period})", showlegend=False, template="plotly_white")
        st.plotly_chart(static_wf, use_container_width=True)

# ----------------- 10. CFO FORENSIC RADAR -----------------
elif selected_section == "🚨 CFO Radar":
    st.subheader("CFO Early-Warning Radar & Reconciliation")
    if not risk_alerts:
        st.success("✓ Primary operational cash metrics within historical bounds.")
    else:
        for alert in risk_alerts:
            st.error(f"**[{alert['category']}] {alert['metric']}**")
            st.write(f"**Reported Evidence:** {alert['evidence']}")
            st.caption(f"**Forensic Implication:** {alert['implication']}")
            st.markdown("---")

# ----------------- 11. BOARDROOM EXECUTIVE DOSSIER -----------------
elif selected_section == "📄 Boardroom Dossier":
    st.subheader("📄 Formal Boardroom Dossier & Export Hub")

    analyst = GeminiFinancialAnalyst(api_key=st.session_state.gemini_api_key)
    dossier = analyst.generate_boardroom_dossier(
        company_name, target_period, df_metrics, df_ratios, altman, piotroski, risk_alerts, {"other_income": curr_other_inc, "book_equity": curr_equity}
    )

    cfo_report_md = f"""# EXECUTIVE CFO FINANCIAL DOSSIER
**ENTITY:** {company_name} | **TAXONOMY:** {taxonomy} | **CYCLE:** {target_period}
* **Gross Sales:** {unit}{curr_rev:,.0f} {scale} (EBITDA Margin: {ebitda_m:.2f}%)
* **Operating CFO:** {unit}{curr_cfo:,.0f} {scale}
* **Free Cash Flow:** {unit}{curr_fcf:,.0f} {scale}
* **Net Worth:** {unit}{curr_equity:,.0f} {scale}
* **Total Borrowings:** {unit}{safe_val(df_metrics, 'total_borrowings', target_period):,.0f} {scale}
* **Altman Z-Score:** `{altman.get('z_score', 'N/A')}` ({altman.get('zone', 'N/A')})
* **Piotroski Quality:** `{piotroski.get('summary_label', 'N/A')}`

---
{dossier.get('content', '')}
"""
    st.markdown(cfo_report_md)
    st.markdown("---")

    col_dl1, col_dl2, col_dl3 = st.columns(3)
    with col_dl1:
        st.download_button("📥 Download Dossier (.md)", cfo_report_md, file_name=f"{company_name}_{target_period}_Dossier.md", mime="text/markdown", use_container_width=True)
    with col_dl2:
        try:
            pdf_bytes = InstitutionalPDFExporter.build_pdf_dossier(
                company_name, target_period, taxonomy, unit, scale, recon, df_metrics, df_ratios, altman, piotroski, anomalies, dossier.get("content", "")
            )
            st.download_button("📥 Download PDF Briefing (.pdf)", pdf_bytes, file_name=f"{company_name}_{target_period}_Briefing.pdf", mime="application/pdf", use_container_width=True)
        except Exception:
            st.info("PDF generation ready via Markdown or CSV export.")
    with col_dl3:
        st.download_button("📊 Download Statement Matrix (.csv)", df_metrics.to_csv().encode("utf-8"), file_name=f"{company_name}_{target_period}_Spread.csv", mime="text/csv", use_container_width=True)