import pandas as pd
import numpy as np
from typing import Dict, Any
from engines.ratio_engine import safe_val

def calculate_piotroski_f_score(df: pd.DataFrame) -> Dict[str, Any]:
    periods = [col for col in df.columns if str(col).lower() not in ["metric", "category", "line_item"]]
    if len(periods) < 2:
        return {
            "summary_label": "N/A — Insufficient Multi-Cycle History",
            "verifiable_count": 0, "passing_count": 0, "unavailable_count": 9, "signals": {}
        }

    curr, prev = periods[-1], periods[-2]
    signals = {}
    passes, verifiable, unavailable = 0, 0, 0

    def record_signal(name: str, passed: bool = None, detail: str = ""):
        nonlocal passes, verifiable, unavailable
        if passed is None:
            signals[name] = {"status": "UNAVAILABLE", "label": "N/A — Insufficient source data", "detail": detail}
            unavailable += 1
        elif passed:
            signals[name] = {"status": "PASS", "label": "Pass (1)", "detail": detail}
            passes += 1
            verifiable += 1
        else:
            signals[name] = {"status": "FLAG", "label": "Flag (0)", "detail": detail}
            verifiable += 1

    pat_c = safe_val(df, "pat", curr, None)
    ast_c = safe_val(df, "total_assets", curr, None)
    if pat_c is not None and ast_c and ast_c > 0:
        roa = pat_c / ast_c
        record_signal("1. Positive Return on Assets (ROA)", roa > 0, f"ROA: {roa*100:.2f}%")
    else:
        record_signal("1. Positive Return on Assets (ROA)", None, "Missing PAT or Assets")

    cfo_c = safe_val(df, "cfo", curr, None)
    if cfo_c is not None:
        record_signal("2. Positive Operating Cash Flow (CFO)", cfo_c > 0, f"CFO: {cfo_c:,.0f}")
    else:
        record_signal("2. Positive Operating Cash Flow (CFO)", None, "Missing Cash Flow Statement")

    pat_p = safe_val(df, "pat", prev, None)
    ast_p = safe_val(df, "total_assets", prev, None)
    if all(v is not None for v in [pat_c, ast_c, pat_p, ast_p]) and ast_c > 0 and ast_p > 0:
        record_signal("3. Year-over-Year ROA Improvement", (pat_c / ast_c) > (pat_p / ast_p), "ROA improved YoY")
    else:
        record_signal("3. Year-over-Year ROA Improvement", None, "Missing comparative cycle items")

    if cfo_c is not None and pat_c is not None:
        record_signal("4. Accrual Quality (CFO > Net Income)", cfo_c > pat_c, f"CFO: {cfo_c:,.0f} vs PAT: {pat_c:,.0f}")
    else:
        record_signal("4. Accrual Quality (CFO > Net Income)", None, "Missing CFO or PAT")

    d_c = safe_val(df, "total_borrowings", curr, None)
    d_p = safe_val(df, "total_borrowings", prev, None)
    if all(v is not None for v in [d_c, ast_c, d_p, ast_p]) and ast_c > 0 and ast_p > 0:
        record_signal("5. Deleveraging Trend (Debt / Assets)", (d_c / ast_c) <= (d_p / ast_p), f"Current: {(d_c/ast_c):.2f}x vs Prev: {(d_p/ast_p):.2f}x")
    else:
        record_signal("5. Deleveraging Trend (Debt / Assets)", None, "Borrowings or Assets incomplete")

    record_signal("6. Liquidity Improvement (Current Ratio)", None, "Condensed statements lack isolated Current Assets / Liabilities breakdown")
    record_signal("7. Share Dilution Guardrail", None, "Share count not reported in canonical schedule")

    r_c, r_p = safe_val(df, "revenue", curr, None), safe_val(df, "revenue", prev, None)
    e_c, e_p = safe_val(df, "ebitda", curr, None), safe_val(df, "ebitda", prev, None)
    if all(v is not None for v in [r_c, r_p, e_c, e_p]) and r_c > 0 and r_p > 0:
        record_signal("8. Operating Margin Improvement", (e_c / r_c) > (e_p / r_p), f"Margin: {(e_c/r_c)*100:.2f}% vs Prev: {(e_p/r_p)*100:.2f}%")
    else:
        record_signal("8. Operating Margin Improvement", None, "Missing comparative revenue/EBITDA")

    if all(v is not None for v in [r_c, r_p, ast_c, ast_p]) and ast_c > 0 and ast_p > 0:
        record_signal("9. Asset Turnover Improvement", (r_c / ast_c) > (r_p / ast_p), "Turnover expanded")
    else:
        record_signal("9. Asset Turnover Improvement", None, "Turnover components incomplete")

    return {
        "summary_label": f"{passes} / {verifiable} Verifiable ({unavailable} Unavailable)",
        "verifiable_count": verifiable,
        "passing_count": passes,
        "unavailable_count": unavailable,
        "signals": signals
    }

def calculate_altman_z(df: pd.DataFrame, period: str, is_bank: bool = False, extra_meta: Dict[str, Any] = None) -> Dict[str, Any]:
    if is_bank:
        return {
            "model_name": "Altman Z-Score", "formula_version": "Inapplicable",
            "z_score": "N/A", "zone": "Inapplicable",
            "applicability_warning": "Altman Z is conceptually invalid for banks and deposit-taking institutions.", "formula_audit": []
        }

    assets = max(safe_val(df, "total_assets", period, 1.0), 1.0)
    rev = safe_val(df, "revenue", period, 0.0)
    ebitda = safe_val(df, "ebitda", period, 0.0)
    debt = safe_val(df, "total_borrowings", period, 0.0)
    equity = safe_val(df, "total_equity", period, 0.0)
    cash = safe_val(df, "cash_and_equivalents", period, 0.0)
    rec = safe_val(df, "trade_receivables", period, 0.0)

    reserves_val = safe_val(df, "reserves", period, None)
    if reserves_val is None:
        eq_cap = safe_val(df, "equity_share_capital", period, 0.0)
        reserves_val = equity - eq_cap if eq_cap else equity

    retained_earnings = float(reserves_val)
    wc_proxy = (cash + rec) - (debt * 0.15)

    x1 = wc_proxy / assets
    x2 = retained_earnings / assets
    x3 = ebitda / assets
    x4 = equity / max(debt, 1.0) if debt > 0 else 1.0
    x5 = rev / assets

    raw_z = 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 0.999 * x5
    z_score = round(float(raw_z), 2)

    formula_audit = [
        {"Term": "A: Working Capital / Assets", "Formula": "Working Capital Proxy / Total Assets", "Value": f"{x1:.4f}", "Contribution": f"{1.2 * x1:.2f}"},
        {"Term": "B: Retained Earnings / Assets", "Formula": "Reserves & Surplus / Total Assets", "Value": f"{x2:.4f}", "Contribution": f"{1.4 * x2:.2f}"},
        {"Term": "C: Operating Profit / Assets", "Formula": "Operating EBITDA / Total Assets", "Value": f"{x3:.4f}", "Contribution": f"{3.3 * x3:.2f}"},
        {"Term": "D: Book Equity / Liabilities", "Formula": "Total Shareholders' Equity / Gross Debt", "Value": f"{x4:.4f}", "Contribution": f"{0.6 * x4:.2f}"},
        {"Term": "E: Sales / Assets", "Formula": "Gross Revenue / Total Assets", "Value": f"{x5:.4f}", "Contribution": f"{0.999 * x5:.2f}"}
    ]

    zone = "Safe Zone (Statistical Low Distress Probability)" if z_score >= 2.99 else ("Grey Zone (Financial Leverage Strain)" if z_score >= 1.81 else "Distress Zone (High Leverage Profile)")

    return {
        "model_name": "Altman Z-Score",
        "formula_version": "Original Z (Manufacturing Calibration with Working Capital Proxy)",
        "z_score": z_score,
        "zone": zone,
        "applicability_warning": "Original Z-score was calibrated on manufacturing firms; capital-intensive entities should be interpreted alongside operational liquidity.",
        "formula_audit": formula_audit
    }

def perform_dupont_decomposition(df: pd.DataFrame, period: str) -> Dict[str, Any]:
    rev = safe_val(df, "revenue", period, 0.0)
    pat = safe_val(df, "pat", period, 0.0)
    assets = max(safe_val(df, "total_assets", period, 1.0), 1.0)
    equity = safe_val(df, "total_equity", period, 0.0)

    if equity < 0:
        return {
            "Net Profit Margin (%)": round((pat / rev * 100) if rev > 0 else 0.0, 2),
            "Asset Turnover (x)": round((rev / assets) if assets > 0 else 0.0, 2),
            "Financial Leverage Multiplier (x)": "Negative (Capital Deficit)",
            "Decomposed ROE (%)": "Inappropriate (Negative Book Equity)",
            "is_negative_equity": True
        }

    net_m = (pat / rev * 100) if rev > 0 else 0.0
    at = rev / assets if assets > 0 else 0.0
    fl = assets / equity if equity > 0 else 0.0
    roe = (pat / equity) * 100 if equity > 0 else 0.0

    return {
        "Net Profit Margin (%)": round(net_m, 2),
        "Asset Turnover (x)": round(at, 2),
        "Financial Leverage Multiplier (x)": round(fl, 2),
        "Decomposed ROE (%)": f"{round(roe, 2)}%",
        "is_negative_equity": False
    }