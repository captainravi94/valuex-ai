import numpy as np
import pandas as pd
from typing import Dict, Any, List

METRIC_ALIASES = {
    "revenue": ["revenue", "sales", "revenue from operations", "total revenue", "turnover", "gross sales"],
    "ebitda": ["ebitda", "operating profit", "pbitda", "operating_ebitda"],
    "pat": ["pat", "net profit", "profit after tax", "net income"],
    "cfo": ["cfo", "cash from operating activity", "operating cash flow", "cash from operations"],
    "capex": ["capex", "capital expenditure", "fixed assets purchased", "purchase of fixed assets"],
    "equity_share_capital": ["equity share capital", "share capital", "equity capital"],
    "reserves": ["reserves", "reserves and surplus", "retained earnings"],
    "total_equity": ["total_equity", "net worth", "shareholders equity", "total shareholders funds", "equity"],
    "total_borrowings": ["total_borrowings", "borrowings", "total debt", "long term borrowings", "short term borrowings"],
    "cash_and_equivalents": ["cash_and_equivalents", "cash & bank", "cash and cash equivalents", "cash"],
    "total_assets": ["total_assets", "total", "total assets"],
    "total_liabilities": ["total_liabilities", "total liabilities", "total equity and liabilities"],
    "trade_receivables": ["trade_receivables", "receivables", "sundry debtors", "debtors"],
    "inventories": ["inventories", "inventory", "stock"],
    "other_income": ["other_income", "other income", "non-operating income"],
    "interest": ["interest", "finance costs", "interest expense"],
    "depreciation": ["depreciation", "depreciation and amortisation", "depreciation & amortisation"]
}

def safe_val(df: pd.DataFrame, canonical_key: str, period: str, *args, **kwargs) -> float:
    default = args[0] if args else kwargs.get("default", 0.0)
    if df is None or df.empty or period not in df.columns:
        return float(default)

    possible_keys = METRIC_ALIASES.get(canonical_key.lower(), [canonical_key.lower()])
    
    # Check exact match
    for k in possible_keys:
        if k in df.index:
            try:
                val = df.loc[k, period]
                if isinstance(val, (pd.Series, pd.DataFrame)):
                    val = val.iloc[0]
                if pd.notna(val):
                    return float(val)
            except (ValueError, TypeError):
                pass

    # Check lowercase normalized index
    lower_idx = {str(idx).strip().lower(): idx for idx in df.index}
    for k in possible_keys:
        if k in lower_idx:
            try:
                val = df.loc[lower_idx[k], period]
                if isinstance(val, (pd.Series, pd.DataFrame)):
                    val = val.iloc[0]
                if pd.notna(val):
                    return float(val)
            except (ValueError, TypeError):
                pass

    return float(default)

def compute_financial_ratios(df: pd.DataFrame, unit_label: str = "Cr") -> pd.DataFrame:
    periods = [
        str(c) for c in df.columns 
        if str(c).strip().lower() not in ["metric", "canonical_metric", "category", "line_item", "particulars", "nan"]
        and not str(c).lower().startswith("unnamed")
    ]
    if not periods:
        periods = [str(c) for c in df.columns]

    ratios = {}
    for p in periods:
        rev = safe_val(df, "revenue", p)
        other_inc = safe_val(df, "other_income", p)
        ebitda_incl = safe_val(df, "ebitda", p)
        
        # Dual EBITDA resolution
        ebitda_excl = safe_val(df, "operating_ebitda", p, default=max(ebitda_incl - other_inc, 0.0))
        if ebitda_excl == 0.0 and ebitda_incl > 0.0 and other_inc > 0.0:
            ebitda_excl = max(ebitda_incl - other_inc, 0.0)

        pat = safe_val(df, "pat", p)
        
        # Net Worth = Share Capital + Reserves (Fallback reconciliation)
        cap = safe_val(df, "equity_share_capital", p)
        res = safe_val(df, "reserves", p)
        reported_equity = safe_val(df, "total_equity", p)
        equity = (cap + res) if (cap + res) > 0 and abs(reported_equity - (cap + res)) > 10.0 else reported_equity

        debt = safe_val(df, "total_borrowings", p)
        cfo = safe_val(df, "cfo", p)
        capex = safe_val(df, "capex", p)
        
        assets = safe_val(df, "total_assets", p)
        rec = safe_val(df, "trade_receivables", p)
        inv = safe_val(df, "inventories", p)
        interest = safe_val(df, "interest", p)

        # Profitability Ratios (Dual Margins)
        margin_excl = (ebitda_excl / rev * 100.0) if rev > 0 else 0.0
        margin_incl = (ebitda_incl / rev * 100.0) if rev > 0 else 0.0
        pat_margin = (pat / rev * 100.0) if rev > 0 else 0.0
        roe = (pat / equity * 100.0) if equity > 0 else 0.0
        roa = (pat / assets * 100.0) if assets > 0 else 0.0

        # Solvency & Leverage
        de_ratio = (debt / equity) if equity > 0 else 0.0
        debt_to_ebitda_excl = (debt / ebitda_excl) if ebitda_excl > 0 else 0.0
        icr = (ebitda_excl / interest) if interest > 0 else (99.0 if ebitda_excl > 0 else 0.0)

        # Working Capital Velocity
        daily_sales = (rev / 365.0) if rev > 0 else 1.0
        dso = (rec / daily_sales) if rev > 0 else 0.0
        dio = (inv / (daily_sales * 0.65)) if rev > 0 else 0.0

        # FCF Flagging
        fcf_str = f"{cfo - capex:,.1f}" if capex > 0 else "Pending Capex Audit"

        ratios[p] = {
            "Core EBITDA Margin (Excl. Other Income)": f"{margin_excl:.2f}%",
            "Consolidated EBITDA Margin (Incl. Other Income)": f"{margin_incl:.2f}%",
            "Net Profit Margin (%)": f"{pat_margin:.2f}%",
            "Return on Equity (ROE) (%)": f"{roe:.2f}%",
            "Return on Assets (ROA) (%)": f"{roa:.2f}%",
            "Debt-to-Equity (x)": f"{de_ratio:.2f}x",
            "Debt-to-EBITDA (Core, x)": f"{debt_to_ebitda_excl:.2f}x",
            "Interest Coverage Ratio (x)": f"{icr:.2f}x",
            "Approx. DSO (Year-End Receivables)": f"{dso:.0f} days",
            "Inventory Days (DIO)": f"{dio:.0f} days",
            f"Verified Free Cash Flow ({unit_label})": fcf_str
        }

    return pd.DataFrame(ratios)