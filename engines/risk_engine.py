import pandas as pd
from typing import List, Dict, Any
from engines.ratio_engine import safe_val

class CFOForensicAudit:
    @staticmethod
    def analyze_health(df: pd.DataFrame) -> List[Dict[str, Any]]:
        alerts = []
        if df is None or df.empty:
            return alerts

        periods = [c for c in df.columns if str(c).lower() not in ["metric", "category", "line_item"]]
        if not periods:
            return alerts
        p = periods[-1]

        cfo = safe_val(df, "cfo", p)
        pat = safe_val(df, "pat", p)
        debt = safe_val(df, "total_borrowings", p)
        equity = safe_val(df, "total_equity", p)
        ebitda = safe_val(df, "ebitda", p)

        if pat > 0 and cfo < pat:
            alerts.append({
                "category": "Earnings Quality",
                "metric": "CFO Lagging Reported PAT",
                "evidence": f"Reported PAT of ₹{pat:,.0f} produced only ₹{cfo:,.0f} in CFO.",
                "implication": "Working capital accumulation or non-cash paper earnings inflating reported profits."
            })

        if ebitda > 0 and debt > (ebitda * 4.0):
            alerts.append({
                "category": "Capital Structure",
                "metric": "Elevated Gross Leverage",
                "evidence": f"Total Borrowings of ₹{debt:,.0f} are {(debt/ebitda):.2f}x Operating EBITDA.",
                "implication": "Refinancing risk sensitive to interest rate fluctuations."
            })

        if equity < 0:
            alerts.append({
                "category": "Solvency Risk",
                "metric": "Negative Net Worth",
                "evidence": f"Shareholders' Equity is ₹{equity:,.0f}.",
                "implication": "Liabilities exceed book assets; requires liquidity support."
            })

        return alerts