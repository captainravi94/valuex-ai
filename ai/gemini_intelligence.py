import os
import json
import requests
import pandas as pd
from typing import Dict, Any

def _stringify_keys(obj):
    """
    Recursively converts all dictionary keys to strings to guarantee
    safe JSON serialization, preventing TypeError on datetime/timestamp keys.
    """
    if isinstance(obj, dict):
        return {str(k): _stringify_keys(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_stringify_keys(item) for item in obj]
    elif isinstance(obj, pd.DataFrame):
        return obj.astype(str).to_dict()
    return obj

class GeminiFinancialAnalyst:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")

    def generate_boardroom_dossier(
        self, 
        company_name: str, 
        period: str, 
        df_metrics: pd.DataFrame, 
        df_ratios: pd.DataFrame, 
        altman: Dict[str, Any], 
        piotroski: Dict[str, Any],
        alerts: list,
        extra_diagnostics: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Executes the 7-Tier Lineage Protocol with Evidence Traces:
        SOURCE -> REPORTED -> CALCULATED -> OBSERVATION -> INTERPRETATION -> LIMITATION -> INVESTIGATION
        """
        diag = extra_diagnostics or {}
        raw_payload = {
            "company": company_name,
            "period": period,
            "metrics": df_metrics.to_dict() if isinstance(df_metrics, pd.DataFrame) else {},
            "ratios": df_ratios.to_dict() if isinstance(df_ratios, pd.DataFrame) else {},
            "altman_z": altman,
            "piotroski_f": piotroski,
            "extra": diag
        }

        safe_payload = _stringify_keys(raw_payload)
        json_payload_str = json.dumps(safe_payload, default=str)

        prompt = f"""
        You are a Senior Managing Director of Institutional Credit & Forensic Accounting.
        Analyze this company using the exact 7-Tier Lineage Protocol for every core finding:
        1. SOURCE: Financial report schedule and statement citation.
        2. 🔵 REPORTED: Direct unadjusted figures from source statements.
        3. 🟣 CALCULATED: Derived ratios and mathematical adjustments (label FCF as 'Analytically Derived Free Cash Flow').
        4. 🟠 ANALYTICAL OBSERVATION: Empirical anomaly or directional drift.
        5. 🟠 INTERPRETATION: Analytical context of what this may indicate.
        6. ⚠️ LIMITATION: What CANNOT be established from this data alone.
        7. 🔴 INVESTIGATION: Probing, non-generic audit questions and note inspections for the CFO/Board.

        MANDATORY RULES:
        1. Disaggregate 'Equity Share Capital' from 'Total Shareholders' Equity / Net Worth'.
        2. Replace 'deleveraged' with 'Gross borrowings decreased by [X] Cr', accompanied by full leverage metric calculations.
        3. Never say debtor days proves billing speed; call it 'Days Sales Outstanding / DSO (Receivables Collection Cycle)'.
        4. Provide Confidence Ratings on each card (Reported Data: HIGH, Calculation: HIGH, Interpretation: MEDIUM, Recurrence: UNKNOWN).

        PAYLOAD:
        {json_payload_str}
        """

        if self.api_key:
            try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={self.api_key}"
                headers = {"Content-Type": "application/json"}
                body = {
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"temperature": 0.1, "maxOutputTokens": 3500}
                }
                res = requests.post(url, headers=headers, json=body, timeout=25)
                if res.status_code == 200:
                    return {"status": "LIVE_GEMINI", "content": res.json()["candidates"][0]["content"]["parts"][0]["text"]}
            except Exception:
                pass

        return {
            "status": "DETERMINISTIC_ENGINE",
            "content": self._build_deterministic_dossier(company_name, period, df_metrics, df_ratios, altman, piotroski, diag)
        }

    def _build_deterministic_dossier(self, comp, period, df, ratios, altman, piotroski, diag) -> str:
        def _get(k):
            if k in df.index and period in df.columns:
                v = pd.to_numeric(df.loc[k, period], errors="coerce")
                return float(v) if pd.notna(v) else 0.0
            for idx in df.index:
                if k in str(idx).lower() and period in df.columns:
                    v = pd.to_numeric(df.loc[idx, period], errors="coerce")
                    if pd.notna(v):
                        return float(v)
            return 0.0

        curr_rev = _get("revenue") or _get("sales")
        curr_ebitda = _get("ebitda") or _get("operating profit")
        curr_pat = _get("pat") or _get("net profit")
        curr_cfo = _get("cfo") or _get("cash from operating")
        curr_capex = _get("capex") or _get("purchase of")
        other_inc = diag.get("other_income", 0.0)
        debt = _get("total_borrowings") or _get("borrowings")
        equity_cap = _get("equity_share_capital") or _get("share capital")
        total_equity = _get("total_equity") or _get("equity")
        reserves = _get("reserves") or (total_equity - equity_cap)
        cash = _get("cash_and_equivalents") or _get("cash")
        assets = _get("total_assets") or 1.0
        rec = _get("trade_receivables") or _get("receivables")
        rec_days = ratios.loc["Receivable Days", period] if "Receivable Days" in ratios.index else "N/A"
        fcf = curr_cfo - curr_capex

        periods = [c for c in df.columns if str(c).lower() not in ["metric", "category"]]
        prev_p = periods[periods.index(period) - 1] if period in periods and periods.index(period) > 0 else None
        
        def _get_prev(k):
            if prev_p and k in df.index and prev_p in df.columns:
                v = pd.to_numeric(df.loc[k, prev_p], errors="coerce")
                return float(v) if pd.notna(v) else 0.0
            return 0.0

        debt_prev = _get_prev("total_borrowings") or _get_prev("borrowings") or debt
        debt_change = debt - debt_prev
        rev_prev = _get_prev("revenue") or _get_prev("sales") or curr_rev
        rev_growth = ((curr_rev - rev_prev) / rev_prev * 100) if rev_prev > 0 else 0.0
        ebitda_prev = _get_prev("ebitda") or _get_prev("operating profit") or curr_ebitda
        assets_prev = _get_prev("total_assets") or assets
        cash_prev = _get_prev("cash_and_equivalents") or cash

        d_ebitda_c = (debt / curr_ebitda) if curr_ebitda > 0 else 0.0
        d_ebitda_p = (debt_prev / ebitda_prev) if ebitda_prev > 0 else 0.0
        nd_ebitda_c = (max(debt - cash, 0) / curr_ebitda) if curr_ebitda > 0 else 0.0
        nd_ebitda_p = (max(debt_prev - cash_prev, 0) / ebitda_prev) if ebitda_prev > 0 else 0.0
        d_assets_c = debt / assets if assets > 0 else 0.0
        d_assets_p = debt_prev / assets_prev if assets_prev > 0 else 0.0

        # CARD 1: REVENUE & EARNINGS COMPOSITION
        card_1 = f"""### 1. Revenue & Earnings Composition Analysis
* **SOURCE:** Ingested Statement Schedule ({period}) $\\rightarrow$ Consolidated Statement of Profit & Loss $\\rightarrow$ Revenue from Operations & Other Income.
* 🔵 **REPORTED:** Gross Sales = **₹{curr_rev:,.0f} Cr** (YoY Growth: **{rev_growth:+.2f}%**); Operating EBITDA = **₹{curr_ebitda:,.0f} Cr**; Reported PAT = **₹{curr_pat:,.0f} Cr**; Reported Other Income = **₹{other_inc:,.0f} Cr**.
* 🟣 **CALCULATED:**
  * Operating EBITDA Margin = `EBITDA / Revenue` = **{(curr_ebitda/curr_rev*100) if curr_rev else 0:.2f}%**
  * Other Income Relative to Operating Profit = `Other Income / EBITDA` = **{(other_inc/curr_ebitda*100) if curr_ebitda else 0:.1f}%**
  * Cash Realization Ratio = `CFO / PAT` = **{(curr_cfo/curr_pat) if curr_pat else 0:.2f}x**
* 🟠 **ANALYTICAL OBSERVATION:** Reported Other Income (₹{other_inc:,.0f} Cr) is {'substantially larger than operating EBITDA' if other_inc > curr_ebitda else 'a relevant component of total earnings'}, while Operating Cash Flow is approximately {(curr_cfo/curr_pat) if curr_pat else 0:.2f}x of reported PAT.
* 🟠 **INTERPRETATION:** Reported earnings conversion warrants structural inspection to isolate core operational profit from non-operating accruals.
* ⚠️ **LIMITATION:** Statement summaries alone do not decompose whether other income represents liquid cash yield, balance sheet derecognition credits, or asset liquidations.
* 🔴 **INVESTIGATION:** Review Note disclosures on Other Income to verify the underlying cash/non-cash breakdown and tax incidence.

> **Confidence Matrix:** Reported Figures: **HIGH** | Calculation: **HIGH** | Interpretation: **MEDIUM** | Note Breakdown: **REQUIRES EXPANSION**"""

        # CARD 2: BALANCE SHEET & CAPITAL STRUCTURE
        card_2 = f"""### 2. Balance Sheet & Capital Structure Diagnostics
* **SOURCE:** Ingested Statement Schedule ({period}) $\\rightarrow$ Consolidated Balance Sheet $\\rightarrow$ Share Capital, Reserves & Borrowings.
* 🔵 **REPORTED:**
  * Equity Share Capital = **₹{equity_cap:,.0f} Cr**
  * Reserves & Surplus = **₹{reserves:,.0f} Cr**
  * Total Shareholders' Equity / Net Worth = **₹{total_equity:,.0f} Cr**
  * Total Borrowings = **₹{debt:,.0f} Cr** (Prior Period: **₹{debt_prev:,.0f} Cr**)
* 🟣 **CALCULATED:**
  * Gross Borrowings Delta = `Borrowings(t) - Borrowings(t-1)` = **₹{debt_change:+,.0f} Cr**
  * Gross Debt / EBITDA = **{d_ebitda_c:.2f}x** (Prior: **{d_ebitda_p:.2f}x**)
  * Net Debt / EBITDA = **{nd_ebitda_c:.2f}x** (Prior: **{nd_ebitda_p:.2f}x**)
  * Debt / Assets = **{d_assets_c:.2f}x** (Prior: **{d_assets_p:.2f}x**)
* 🟠 **ANALYTICAL OBSERVATION:** Gross borrowings changed by ₹{abs(debt_change):,.0f} Cr over the period, adjusting Gross Debt/EBITDA from {d_ebitda_p:.2f}x to {d_ebitda_c:.2f}x.
* 🟠 **INTERPRETATION:** {"Carrying liabilities exceed total asset values, resulting in negative net worth." if total_equity < 0 else "Net worth is positive, providing solvent asset backing for senior liabilities."}
* ⚠️ **LIMITATION:** Debt balance movements do not isolate operational cash debt amortization from moratorium restructuring or debt-to-equity conversions.
* 🔴 **INVESTIGATION:** Reconcile the Cash Flow from Financing Activities and Debt Notes to verify repayment mechanics.

> **Confidence Matrix:** Reported Capital: **HIGH** | Leverage Calculations: **HIGH** | Deleveraging Mechanism: **REQUIRES CFF NOTE**"""

        # CARD 3: CASH FLOW INTEGRITY
        reinvest = (curr_capex / curr_cfo * 100) if curr_cfo > 0 else 0.0
        card_3 = f"""### 3. Cash Flow Integrity & Capital Allocation
* **SOURCE:** Ingested Statement Schedule ({period}) $\\rightarrow$ Statement of Cash Flows & Trade Receivables.
* 🔵 **REPORTED:** Operating Cash Flow (CFO) = **₹{curr_cfo:,.0f} Cr**; Capital Expenditure = **₹{curr_capex:,.0f} Cr**; Trade Receivables = **₹{rec:,.0f} Cr**.
* 🟣 **CALCULATED:**
  * Analytically Derived Free Cash Flow = `CFO - Capex` = **₹{fcf:,.0f} Cr**
  * Capital Reinvestment Rate = `Capex / CFO` = **{reinvest:.1f}%**
  * Days Sales Outstanding (DSO) = **{rec_days}**
* 🟠 **ANALYTICAL OBSERVATION:** Operating cash generation resulted in an analytically derived Free Cash Flow of ₹{fcf:,.0f} Cr after allocating ₹{curr_capex:,.0f} Cr toward capital investments.
* 🟠 **INTERPRETATION:** Receivables represent a collection timeframe of {rec_days}. Working capital movements must be audited holistically alongside inventory and payables.
* ⚠️ **LIMITATION:** Point-in-time trade receivables metrics do not capture intra-cycle billing spikes or unbilled revenue.
* 🔴 **INVESTIGATION:** Audit the working capital reconciliation schedule in the Statement of Cash Flows for customer retention and payable extensions.

> **Confidence Matrix:** Reported CFO & Capex: **HIGH** | Analytically Derived FCF: **HIGH** | Working Capital Trajectory: **MEDIUM**"""

        # CARD 4: MODEL METHODOLOGY AUDIT
        card_4 = f"""### 4. Financial Model Validation & Methodology Audit
* **SOURCE:** Academic Solvency & Accounting Quality Algorithms ({period}).
* **ALTMAN Z-SCORE AUDIT:**
  * *Model Name:* {altman.get('model_name', 'Altman Z-Score')} | *Formula:* {altman.get('formula_version', 'Manufacturing')}
  * *Calculated Value:* `{altman.get('z_score', 'N/A')}` — **{altman.get('zone', 'N/A')}**
  * *Applicability:* {altman.get('applicability_warning', 'Standard application')}
* **PIOTROSKI F-SCORE AUDIT:**
  * *Model Name:* Joseph Piotroski 9-Signal Health Index
  * *Summary Result:* **{piotroski.get('summary_label', 'N/A')}** (Verifiable: {piotroski.get('verifiable_count', 0)}/9 | Passing: {piotroski.get('passing_count', 0)} | Unavailable: {piotroski.get('unavailable_count', 0)})
  * *Validation Protocol:* Signals lacking direct statement items are classified as 'Unavailable' rather than guessed."""

        # CARD 5: BOARDROOM DIRECTIVES
        card_5 = f"""### 5. Boardroom & CFO Investigation Directives
1. 🔴 **Earnings Quality Inspection:** Decompose reported Other Income of ₹{other_inc:,.0f} Cr into operational vs non-operational components.
2. 🔴 **Capital Structure Reconciliation:** Trace the gross debt movement of ₹{abs(debt_change):,.0f} Cr directly against the Cash Flow from Financing Activities schedule.
3. 🔴 **Reinvestment Sufficiency:** Evaluate whether current capex outlays (₹{curr_capex:,.0f} Cr) sustain long-term operational maintenance requirements."""

        return f"{card_1}\n\n---\n\n{card_2}\n\n---\n\n{card_3}\n\n---\n\n{card_4}\n\n---\n\n{card_5}"