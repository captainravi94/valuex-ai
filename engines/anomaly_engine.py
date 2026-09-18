import pandas as pd
from typing import Dict, Any, List
from engines.ratio_engine import safe_val

class FinancialAnomalyEngine:
    @staticmethod
    def detect_anomalies(df: pd.DataFrame, period: str, extra_meta: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        anomalies = []
        periods = [c for c in df.columns if c not in ["metric", "category", "line_item"]]
        if not periods or period not in periods:
            return anomalies

        curr = period
        prev = periods[periods.index(curr) - 1] if periods.index(curr) > 0 else None

        rev_c = safe_val(df, "revenue", curr)
        ebitda_c = safe_val(df, "ebitda", curr)
        pat_c = safe_val(df, "pat", curr)
        other_inc = safe_val(df, "other_income", curr)
        cfo_c = safe_val(df, "cfo", curr)
        debt_c = safe_val(df, "total_borrowings", curr)
        debt_p = safe_val(df, "total_borrowings", prev) if prev else 0.0
        equity_c = safe_val(df, "total_equity", curr)
        rec_c = safe_val(df, "trade_receivables", curr)
        rec_p = safe_val(df, "trade_receivables", prev) if prev else 0.0
        rev_p = safe_val(df, "revenue", prev) if prev else 0.0

        if ebitda_c > 0 and (other_inc / ebitda_c) >= 1.0:
            anomalies.append({
                "id": "ANOMALY-001", "severity": "CRITICAL",
                "title": "Other Income Exceeds Operating Profit",
                "metric": f"Other Income / EBITDA = {(other_inc/ebitda_c):.2f}x",
                "evidence": f"Reported Other Income of ₹{other_inc:,.0f} vs Operating EBITDA of ₹{ebitda_c:,.0f}",
                "investigation": "Inspect Note on Other Income for non-operating or non-cash credits."
            })

        if pat_c > 0 and cfo_c > 0 and (cfo_c / pat_c) < 0.70:
            anomalies.append({
                "id": "ANOMALY-002", "severity": "HIGH",
                "title": "Earnings-to-Cash Realization Divergence",
                "metric": f"CFO / PAT = {(cfo_c/pat_c):.2f}x",
                "evidence": f"Reported PAT of ₹{pat_c:,.0f} realized only ₹{cfo_c:,.0f} in Operating Cash Flow",
                "investigation": "Determine if net income is driven by accruals or capital trapped in working capital."
            })

        if equity_c < 0:
            anomalies.append({
                "id": "ANOMALY-003", "severity": "CRITICAL",
                "title": "Negative Shareholders' Equity / Deficit Net Worth",
                "metric": f"Shareholders' Equity = ₹{equity_c:,.0f}",
                "evidence": f"Carrying value of liabilities exceeds assets by ₹{abs(equity_c):,.0f}",
                "investigation": "Evaluate debt maturity profile, statutory refinancing support, and capital injection plans."
            })

        if prev and (debt_p - debt_c) > (rev_c * 0.20):
            debt_reduction = debt_p - debt_c
            anomalies.append({
                "id": "ANOMALY-004", "severity": "MEDIUM",
                "title": "Substantial Gross Borrowings Contraction",
                "metric": f"Gross Debt Reduction = ₹{debt_reduction:,.0f}",
                "evidence": f"Borrowings decreased from ₹{debt_p:,.0f} to ₹{debt_c:,.0f}",
                "investigation": "Reconcile Cash Flow from Financing Activities to confirm if paid down with cash or via equity swaps."
            })

        if prev and rev_p > 0 and rec_p > 0:
            rev_g = (rev_c - rev_p) / rev_p
            rec_g = (rec_c - rec_p) / rec_p
            if (rec_g - rev_g) > 0.30:
                anomalies.append({
                    "id": "ANOMALY-005", "severity": "HIGH",
                    "title": "Receivables Growth Disconnect",
                    "metric": f"Debtors (+{rec_g*100:.1f}%) outpacing Sales (+{rev_g*100:.1f}%)",
                    "evidence": f"Trade Receivables outpaced sales growth by {(rec_g - rev_g)*100:.1f}%",
                    "investigation": "Audit customer payment terms, aging brackets, and provision for doubtful debts."
                })

        return anomalies