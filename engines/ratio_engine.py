import pandas as pd
import numpy as np
import re

def safe_val(df: pd.DataFrame, metric_key: str, period: str, default: float = 0.0) -> float:
    if df is None or df.empty or period not in df.columns:
        return default
    if metric_key in df.index:
        v = pd.to_numeric(df.loc[metric_key, period], errors="coerce")
        if pd.notna(v):
            return float(v)

    norm = {str(k).strip().lower(): k for k in df.index}
    if metric_key.lower() in norm:
        v = pd.to_numeric(df.loc[norm[metric_key.lower()], period], errors="coerce")
        if pd.notna(v):
            return float(v)

    syns = {
        "revenue": ["revenue", "sales", "turnover", "total income", "gross revenue"],
        "ebitda": ["ebitda", "operating profit", "pbit", "operating margin"],
        "pat": ["pat", "profit after tax", "profit for the period", "loss for the period", "net profit"],
        "cfo": ["cfo", "cash from operating", "cash flow from operating", "operating activities"],
        "capex": ["capex", "capital expenditure", "purchase of property", "purchase of fixed assets"],
        "total_borrowings": ["borrowings", "total debt", "debt", "financial liabilities", "loans"],
        "total_equity": ["total equity", "equity", "net worth", "shareholders fund", "share capital"],
        "trade_receivables": ["trade receivables", "receivables", "debtors"],
        "total_assets": ["total assets", "assets"]
    }
    targets = syns.get(metric_key.lower(), [metric_key.lower()])
    for row in df.index:
        clean = re.sub(r'[^a-z0-9]', ' ', str(row).lower())
        if any(t in clean for t in targets):
            v = pd.to_numeric(df.loc[row, period], errors="coerce")
            if pd.notna(v):
                return float(v)
    return default

def compute_financial_ratios(df_metrics: pd.DataFrame, unit_label: str = "Cr") -> pd.DataFrame:
    periods = [c for c in df_metrics.columns if str(c).lower() not in ["metric", "category", "line_item"]]
    ratio_names = [
        "Operating EBITDA Margin (%)",
        "Net Profit Margin (%)",
        "CFO to EBITDA Ratio (x)",
        "Cash Quality (CFO / PAT) (x)",
        "Debt-to-Equity Ratio (x)",
        "Gross Debt to EBITDA (x)",
        "Receivable Days"
    ]
    res = pd.DataFrame(index=ratio_names, columns=periods)

    for p in periods:
        rev = safe_val(df_metrics, "revenue", p)
        ebitda = safe_val(df_metrics, "ebitda", p)
        pat = safe_val(df_metrics, "pat", p)
        cfo = safe_val(df_metrics, "cfo", p)
        debt = safe_val(df_metrics, "total_borrowings", p)
        equity = safe_val(df_metrics, "total_equity", p)
        rec = safe_val(df_metrics, "trade_receivables", p)

        res.loc["Operating EBITDA Margin (%)", p] = f"{(ebitda / rev * 100):.2f}%" if rev > 0 else "0.00%"
        res.loc["Net Profit Margin (%)", p] = f"{(pat / rev * 100):.2f}%" if rev > 0 else "0.00%"
        res.loc["CFO to EBITDA Ratio (x)", p] = f"{(cfo / ebitda):.2f}x" if ebitda > 0 else "N/A"
        res.loc["Cash Quality (CFO / PAT) (x)", p] = f"{(cfo / pat):.2f}x" if pat > 0 else "N/A"
        res.loc["Debt-to-Equity Ratio (x)", p] = f"{(debt / equity):.2f}x" if equity > 0 else ("Negative" if equity < 0 else "N/A")
        res.loc["Gross Debt to EBITDA (x)", p] = f"{(debt / ebitda):.2f}x" if ebitda > 0 else "N/A"
        res.loc["Receivable Days", p] = f"{int((rec / rev) * 365)} days" if rev > 0 and rec > 0 else "N/A"

    return res