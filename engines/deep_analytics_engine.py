import numpy as np
import pandas as pd
from typing import Dict, Any, List
from engines.ratio_engine import safe_val

def calculate_piotroski_f_score(df: pd.DataFrame) -> Dict[str, Any]:
    periods = [
        str(c) for c in df.columns 
        if str(c).strip().lower() not in ["metric", "canonical_metric", "category", "line_item", "particulars", "nan"]
        and not str(c).lower().startswith("unnamed")
    ]
    if len(periods) < 2:
        return {
            "score": 0, "passing_count": 0, "total_signals": 9, "unavailable_count": 9,
            "summary_label": "Piotroski F-Score: 0/9 signals passed | 9 signals unavailable",
            "breakdown": []
        }

    curr_p = periods[-1]
    prev_p = periods[-2]

    pat_c = safe_val(df, "pat", curr_p)
    pat_p = safe_val(df, "pat", prev_p)
    cfo_c = safe_val(df, "cfo", curr_p)
    assets_c = safe_val(df, "total_assets", curr_p)
    assets_p = safe_val(df, "total_assets", prev_p)
    debt_c = safe_val(df, "total_borrowings", curr_p)
    debt_p = safe_val(df, "total_borrowings", prev_p)
    rev_c = safe_val(df, "revenue", curr_p)
    rev_p = safe_val(df, "revenue", prev_p)
    ebitda_c = safe_val(df, "ebitda", curr_p)
    ebitda_p = safe_val(df, "ebitda", prev_p)

    roa_c = (pat_c / assets_c) if assets_c > 0 else 0.0
    roa_p = (pat_p / assets_p) if assets_p > 0 else 0.0
    gm_c = (ebitda_c / rev_c) if rev_c > 0 else 0.0
    gm_p = (ebitda_p / rev_p) if rev_p > 0 else 0.0
    at_c = (rev_c / assets_c) if assets_c > 0 else 0.0
    at_p = (rev_p / assets_p) if assets_p > 0 else 0.0

    signals = [
        # Profitability
        {"signal": "1. Positive Net Income (ROA > 0)", "status": "PASS" if roa_c > 0 else "FAIL", "evidence": f"ROA = {roa_c*100:.2f}%"},
        {"signal": "2. Positive Operating Cash Flow (CFO > 0)", "status": "PASS" if cfo_c > 0 else "FAIL", "evidence": f"CFO = ₹{cfo_c:,.0f} Cr"},
        {"signal": "3. Higher Return on Assets YoY (ΔROA > 0)", "status": "PASS" if roa_c > roa_p else "FAIL", "evidence": f"{roa_c*100:.2f}% vs {roa_p*100:.2f}%"},
        {"signal": "4. Cash Quality Accrual Test (CFO > PAT)", "status": "PASS" if cfo_c > pat_c else "FAIL", "evidence": f"CFO ₹{cfo_c:,.0f} Cr vs PAT ₹{pat_c:,.0f} Cr"},
        # Leverage, Liquidity & Source of Funds
        {"signal": "5. Lower Long-Term Gearing YoY (ΔLeverage ≤ 0)", "status": "PASS" if (debt_c/assets_c) <= (debt_p/assets_p) else "FAIL", "evidence": f"{(debt_c/assets_c)*100:.2f}% vs {(debt_p/assets_p)*100:.2f}%"},
        {"signal": "6. Higher Current Ratio YoY (ΔLiquidity > 0)", "status": "UNAVAILABLE", "evidence": "Detailed current assets schedule not isolated"},
        {"signal": "7. Zero Share Dilution (Shares_t ≤ Shares_t-1)", "status": "UNAVAILABLE", "evidence": "Diluted share count series pending footnote audit"},
        # Operating Efficiency
        {"signal": "8. Higher Operating Margin YoY (ΔMargin > 0)", "status": "PASS" if gm_c > gm_p else "FAIL", "evidence": f"{gm_c*100:.2f}% vs {gm_p*100:.2f}%"},
        {"signal": "9. Higher Asset Turnover YoY (ΔAssetTurnover > 0)", "status": "PASS" if at_c >= at_p else "FAIL", "evidence": f"{at_c:.2f}x vs {at_p:.2f}x"}
    ]

    passing = sum(1 for s in signals if s["status"] == "PASS")
    unavailable = sum(1 for s in signals if s["status"] == "UNAVAILABLE")

    return {
        "score": passing,
        "passing_count": passing,
        "total_signals": 9,
        "unavailable_count": unavailable,
        "summary_label": f"Piotroski F-Score: {passing}/9 signals passed | {9 - unavailable} evaluated | {unavailable} unavailable",
        "breakdown": signals
    }

def calculate_altman_z(df: pd.DataFrame, target_period: str, is_bank: bool = False, extra_meta: Dict[str, Any] = None) -> Dict[str, Any]:
    if is_bank:
        return {"z_score": 0.0, "zone": "Not Applicable (Financial Institution)", "formula_audit": []}

    assets = safe_val(df, "total_assets", target_period, default=1.0)
    if assets <= 0: assets = 1.0

    ebitda = safe_val(df, "ebitda", target_period)
    dep = safe_val(df, "depreciation", target_period)
    ebit = ebitda - dep
    sales = safe_val(df, "revenue", target_period)
    
    # Net Worth check
    cap = safe_val(df, "equity_share_capital", target_period)
    res = safe_val(df, "reserves", target_period)
    equity = (cap + res) if (cap + res) > 0 else safe_val(df, "total_equity", target_period)

    total_liab = safe_val(df, "total_liabilities", target_period, default=1.0)
    if total_liab <= 0: total_liab = 1.0

    rec = safe_val(df, "trade_receivables", target_period)
    inv = safe_val(df, "inventories", target_period)
    cash = safe_val(df, "cash_and_equivalents", target_period)
    borrowings = safe_val(df, "total_borrowings", target_period)
    
    current_assets = rec + inv + cash
    current_liab = max(total_liab - borrowings - equity, total_liab * 0.20)
    working_capital = current_assets - current_liab

    # Factors
    x1 = working_capital / assets
    x2 = res / assets  # Retained earnings / Assets
    x3 = ebit / assets
    x4 = equity / total_liab
    x5 = sales / assets

    # Emerging Market Z'' Model (1993): Z'' = 6.56*X1 + 3.26*X2 + 6.72*X3 + 1.05*X4
    # (Excludes X5 Sales/Assets to remove sector asset-turnover distortion)
    z_double_prime = (6.56 * x1) + (3.26 * x2) + (6.72 * x3) + (1.05 * x4)
    z_val = round(float(z_double_prime), 2)
    zone = "Safe Zone" if z_val >= 2.60 else ("Grey Zone" if z_val >= 1.10 else "Model-indicated Distress Range")

    audit = [
        {"Factor": "X1: Working Capital / Total Assets", "Weight": "6.56", "Calculated Value": f"{x1:.4f}", "Product Contribution": f"{6.56 * x1:.2f}"},
        {"Factor": "X2: Retained Earnings / Total Assets", "Weight": "3.26", "Calculated Value": f"{x2:.4f}", "Product Contribution": f"{3.26 * x2:.2f}"},
        {"Factor": "X3: EBIT / Total Assets", "Weight": "6.72", "Calculated Value": f"{x3:.4f}", "Product Contribution": f"{6.72 * x3:.2f}"},
        {"Factor": "X4: Book Net Worth / Total Liabilities", "Weight": "1.05", "Calculated Value": f"{x4:.4f}", "Product Contribution": f"{1.05 * x4:.2f}"},
        {"Factor": "X5: Asset Turnover (Sales / Assets)", "Weight": "Baseline Context", "Calculated Value": f"{x5:.4f}", "Product Contribution": "Diagnostic Only"}
    ]

    return {
        "z_score": z_val,
        "zone": zone,
        "formula_audit": audit,
        "methodology": "Altman Z''-Score Emerging Market Model (1993 calibration). Values > 2.60 denote investment-grade balance sheet liquidity."
    }

def perform_dupont_decomposition(df: pd.DataFrame, target_period: str) -> Dict[str, Any]:
    pat = safe_val(df, "pat", target_period)
    rev = safe_val(df, "revenue", target_period)
    assets = safe_val(df, "total_assets", target_period, default=1.0)
    
    cap = safe_val(df, "equity_share_capital", target_period)
    res = safe_val(df, "reserves", target_period)
    equity = (cap + res) if (cap + res) > 0 else safe_val(df, "total_equity", target_period, default=1.0)

    if rev <= 0 or assets <= 0 or equity <= 0:
        return {"Net Profit Margin (%)": "0.00%", "Asset Turnover (x)": "0.00x", "Financial Leverage (x)": "0.00x", "Decomposed ROE (%)": "0.00%"}

    net_margin = (pat / rev) * 100.0
    asset_turnover = rev / assets
    leverage = assets / equity
    roe = (net_margin / 100.0) * asset_turnover * leverage * 100.0

    return {
        "Net Profit Margin (%)": f"{net_margin:.2f}%",
        "Asset Turnover (x)": f"{asset_turnover:.2f}x",
        "Financial Leverage (x)": f"{leverage:.2f}x",
        "Decomposed ROE (%)": f"{roe:.2f}%"
    }