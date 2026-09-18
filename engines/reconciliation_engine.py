import pandas as pd
from typing import Dict, Any
from engines.ratio_engine import safe_val

class FinancialReconciliationEngine:
    @staticmethod
    def audit_statement_integrity(df: pd.DataFrame, period: str, extra_meta: Dict[str, Any] = None) -> Dict[str, Any]:
        rev = safe_val(df, "revenue", period)
        ebitda = safe_val(df, "ebitda", period)
        pat = safe_val(df, "pat", period)
        cfo = safe_val(df, "cfo", period)
        capex = safe_val(df, "capex", period)
        debt = safe_val(df, "total_borrowings", period)
        equity = safe_val(df, "total_equity", period)
        assets = safe_val(df, "total_assets", period)

        checks = []
        score = 0
        total_tests = 4

        # Test 1: Operating Conversion
        if rev > 0 and ebitda > 0:
            checks.append({"test": "Top-Line Operational Health", "status": "VERIFIED", "evidence": f"EBITDA: ₹{ebitda:,.0f} on Sales of ₹{rev:,.0f}"})
            score += 1
        else:
            checks.append({"test": "Top-Line Operational Health", "status": "FLAGGED", "evidence": "EBITDA or Sales missing/zero"})

        # Test 2: Cash Realization
        if cfo > 0:
            checks.append({"test": "Operating Cash Generation", "status": "VERIFIED", "evidence": f"Operating CFO: ₹{cfo:,.0f}"})
            score += 1
        else:
            checks.append({"test": "Operating Cash Generation", "status": "FLAGGED", "evidence": "Operating Cash Flow is negative or zero"})

        # Test 3: Capital Adequacy
        if equity > 0:
            checks.append({"test": "Net Worth Solvency", "status": "VERIFIED", "evidence": f"Positive Net Worth: ₹{equity:,.0f}"})
            score += 1
        else:
            checks.append({"test": "Net Worth Solvency", "status": "DEFICIT", "evidence": f"Negative Shareholders' Equity: ₹{equity:,.0f}"})

        # Test 4: Reinvestment Discipline
        if capex > 0:
            checks.append({"test": "Capital Reinvestment Schedule", "status": "VERIFIED", "evidence": f"Capex Outlay: ₹{capex:,.0f}"})
            score += 1
        else:
            checks.append({"test": "Capital Reinvestment Schedule", "status": "UNAVAILABLE", "evidence": "Capex schedule not isolated"})

        confidence = int((score / total_tests) * 100)
        status = "HEALTHY" if confidence >= 75 else "VARIANCE_IDENTIFIED"

        return {
            "confidence_pct": confidence,
            "overall_status": status,
            "score_points": score,
            "max_points": total_tests,
            "checks": checks
        }