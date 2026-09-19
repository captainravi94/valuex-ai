import numpy as np
import pandas as pd
from typing import Dict, Any

class MSMECreditRiskEngine:
    """
    Institutional MSME Credit Underwriting Engine:
    - Probability of Default (PD) scoring based on financial and banking behavioral features.
    - Loss Given Default (LGD) calculated via collateral coverage & asset haircuts.
    - Exposure at Default (EAD) based on limit utilization and credit conversion factor (CCF).
    - Staged Expected Credit Loss (ECL) under IFRS 9 / Ind AS 109.
    """

    @staticmethod
    def calculate_credit_risk(
        annual_revenue: float,
        ebitda_margin_pct: float,
        debt_to_equity: float,
        interest_coverage_ratio: float,
        gst_bank_turnover_ratio: float,  # Ratio of GST sales to Banking credits (ideally 0.95 - 1.05)
        bureau_commercial_score: int,    # Commercial CIBIL / Experian score (300 - 900)
        sanctioned_limit: float,
        outstanding_balance: float,
        undrawn_limit: float,
        primary_collateral_value: float,
        collateral_haircut_pct: float = 25.0
    ) -> Dict[str, Any]:
        
        # 1. Base Financial & Banking Quality Score (0 to 100)
        score = 0.0

        # DSCR / Interest Coverage
        if interest_coverage_ratio >= 3.0:
            score += 20.0
        elif interest_coverage_ratio >= 1.75:
            score += 15.0
        elif interest_coverage_ratio >= 1.25:
            score += 10.0
        else:
            score += 2.0

        # Leverage (Debt-to-Equity)
        if debt_to_equity <= 1.0:
            score += 20.0
        elif debt_to_equity <= 2.0:
            score += 15.0
        elif debt_to_equity <= 3.5:
            score += 8.0
        else:
            score += 0.0

        # EBITDA Margin Stability
        if ebitda_margin_pct >= 15.0:
            score += 15.0
        elif ebitda_margin_pct >= 8.0:
            score += 10.0
        elif ebitda_margin_pct >= 3.0:
            score += 5.0
        else:
            score += 0.0

        # Commercial Bureau Score
        if bureau_commercial_score >= 750:
            score += 25.0
        elif bureau_commercial_score >= 675:
            score += 18.0
        elif bureau_commercial_score >= 600:
            score += 10.0
        else:
            score += 2.0

        # GST vs Banking Reconciliation Compliance
        gst_gap = abs(1.0 - gst_bank_turnover_ratio)
        if gst_gap <= 0.08:
            score += 20.0
        elif gst_gap <= 0.18:
            score += 12.0
        elif gst_gap <= 0.30:
            score += 4.0
        else:
            score += 0.0  # Significant GST vs Bank divergence (circular transaction flag)

        # 2. Probability of Default (PD) calibration via Logistic Link
        # Mapping Score (0-100) to continuous PD curve
        pd_raw = 1.0 / (1.0 + np.exp((score - 52.0) / 11.0))
        pd_pct = max(min(round(float(pd_raw * 100), 2), 99.0), 0.5)

        # Rating Grade
        if score >= 80:
            rating = "CMR-1 (Prime / Investment Grade)"
            stage = "Stage 1 (Normal Performing)"
        elif score >= 68:
            rating = "CMR-2 (Standard Good)"
            stage = "Stage 1 (Normal Performing)"
        elif score >= 55:
            rating = "CMR-3 (Standard Satisfactory)"
            stage = "Stage 1 (Normal Performing)"
        elif score >= 42:
            rating = "CMR-4 (Moderate Risk / Watchlist)"
            stage = "Stage 2 (Underperforming - SICR)"
        elif score >= 30:
            rating = "CMR-5 (Sub-Standard / Vulnerable)"
            stage = "Stage 2 (Underperforming - SICR)"
        else:
            rating = "CMR-6 (Impaired / High Default Risk)"
            stage = "Stage 3 (Credit Impaired)"

        # 3. Exposure at Default (EAD)
        # Credit Conversion Factor (CCF) on undrawn revolving lines is standard 40% (Basel)
        ccf = 0.40
        ead = outstanding_balance + (undrawn_limit * ccf)

        # 4. Loss Given Default (LGD)
        net_recoverable_collateral = max(primary_collateral_value * (1.0 - collateral_haircut_pct / 100.0), 0.0)
        unsecured_portion = max(ead - net_recoverable_collateral, 0.0)
        
        # Secured recovery rate ~85%, unsecured recovery rate ~15%
        secured_loss_rate = 0.15
        unsecured_loss_rate = 0.85

        total_loss_nominal = (min(ead, net_recoverable_collateral) * secured_loss_rate) + (unsecured_portion * unsecured_loss_rate)
        lgd_pct = max(min(round(float((total_loss_nominal / ead) * 100), 2), 90.0), 10.0) if ead > 0 else 45.0

        # 5. Expected Credit Loss (ECL = PD × LGD × EAD)
        ecl_amount = round((pd_pct / 100.0) * (lgd_pct / 100.0) * ead, 2)

        return {
            "score": round(score, 1),
            "internal_rating": rating,
            "ifrs9_stage": stage,
            "pd_pct": pd_pct,
            "ead_amount": round(ead, 2),
            "lgd_pct": lgd_pct,
            "ecl_amount": ecl_amount,
            "net_collateral": round(net_recoverable_collateral, 2),
            "gst_bank_gap_pct": round(gst_gap * 100, 1),
            "underwriting_directive": (
                "Approved within standard policy limits." if score >= 65 and gst_gap <= 0.15 else (
                    "Conditional sanction: Higher collateral cover or promoter personal guarantee required."
                    if score >= 45 else
                    "Rejected / Elevated Risk: Material divergence between tax filings and audited balances."
                )
            )
        }