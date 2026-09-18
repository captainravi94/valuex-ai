import pandas as pd
from typing import Dict, Any
from engines.ratio_engine import safe_val

class WorkingCapitalBridgeEngine:
    @staticmethod
    def compute_bridge(df: pd.DataFrame, period: str) -> Dict[str, Any]:
        ebitda = safe_val(df, "ebitda", period)
        cfo = safe_val(df, "cfo", period)
        net_movement = cfo - ebitda

        bridge_steps = [
            {"Step": "1. Reported Operating EBITDA", "Impact": ebitda, "Nature": "Base Operational Conversion"},
            {"Step": "2. Working Capital & Tax Movement", "Impact": net_movement, "Nature": "Working Capital Drag / Release"},
            {"Step": "3. Final Operating CFO Realized", "Impact": cfo, "Nature": "Audited Cash Flow Generation"}
        ]

        return {
            "status": "SUCCESS",
            "ebitda": ebitda,
            "cfo": cfo,
            "net_wc_movement": net_movement,
            "bridge": bridge_steps
        }