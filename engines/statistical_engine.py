import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any, List
from engines.ratio_engine import safe_val

class FinancialStatisticalEngine:
    """
    Executes formal statistical inference across multi-cycle financial statements:
    - Time-series Metric Z-Scores (Gaussian Outlier Detection)
    - Student's t-Test (Mean Growth & Margin Spread Significance)
    - Fisher / Snedecor F-Test (Variance & Volatility Ratio Test)
    """

    @staticmethod
    def compute_metric_z_scores(df: pd.DataFrame, metric_key: str) -> Dict[str, Any]:
        periods = [c for c in df.columns if str(c).lower() not in ["metric", "category", "line_item"]]
        if len(periods) < 3:
            return {"status": "INSUFFICIENT_DATA", "series": []}

        vals = [safe_val(df, metric_key, p) for p in periods]
        arr = np.array(vals, dtype=float)
        mean_val = np.mean(arr)
        std_val = np.std(arr, ddof=1) if len(arr) > 1 else 0.0

        results = []
        for p, v in zip(periods, vals):
            z = (v - mean_val) / std_val if std_val > 0 else 0.0
            outlier_tag = "Critical Anomaly (|Z| > 2)" if abs(z) >= 2.0 else ("Moderate Drift (|Z| > 1)" if abs(z) >= 1.0 else "Normal Range")
            results.append({
                "Period": p,
                "Reported Value": v,
                "Mean Baseline": round(mean_val, 2),
                "Z-Score": round(z, 2),
                "Statistical Diagnosis": outlier_tag
            })

        return {
            "status": "SUCCESS",
            "metric": metric_key,
            "mean": round(mean_val, 2),
            "std": round(std_val, 2),
            "table": results
        }

    @staticmethod
    def run_student_t_test(df: pd.DataFrame, metric_a: str, metric_b: str = None) -> Dict[str, Any]:
        periods = [c for c in df.columns if str(c).lower() not in ["metric", "category", "line_item"]]
        if len(periods) < 3:
            return {"status": "INSUFFICIENT_DATA", "detail": "At least 3 financial periods required."}

        vals_a = [safe_val(df, metric_a, p) for p in periods]
        growth_a = [(vals_a[i] - vals_a[i-1]) / vals_a[i-1] for i in range(1, len(vals_a)) if vals_a[i-1] > 0]

        if len(growth_a) < 2:
            return {"status": "INSUFFICIENT_DATA", "detail": "Insufficient consecutive growth data points."}

        if metric_b:
            vals_b = [safe_val(df, metric_b, p) for p in periods]
            growth_b = [(vals_b[i] - vals_b[i-1]) / vals_b[i-1] for i in range(1, len(vals_b)) if vals_b[i-1] > 0]

            t_stat, p_val = stats.ttest_ind(growth_a, growth_b, equal_var=False)
            h0_rejected = p_val < 0.05
            interp = (
                f"Statistically significant divergence (p = {p_val:.4f} < 0.05). "
                f"Trajectory of {metric_a} differs from {metric_b}."
                if h0_rejected else
                f"No statistically significant difference in trajectories (p = {p_val:.4f} >= 0.05). H0 accepted."
            )
            return {
                "test_type": f"Two-Sample Welch's t-Test ({metric_a} vs {metric_b})",
                "t_statistic": round(float(t_stat), 4),
                "p_value": round(float(p_val), 4),
                "h0_rejected": h0_rejected,
                "interpretation": interp
            }
        else:
            t_stat, p_val = stats.ttest_1samp(growth_a, 0.0)
            h0_rejected = p_val < 0.05
            interp = (
                f"Mean growth of {metric_a} ({np.mean(growth_a)*100:.1f}%) is statistically non-zero (p = {p_val:.4f})."
                if h0_rejected else
                f"Growth trend is statistically indistinguishable from zero (p = {p_val:.4f})."
            )
            return {
                "test_type": f"One-Sample t-Test (Mean Growth of {metric_a} vs 0.0)",
                "t_statistic": round(float(t_stat), 4),
                "p_value": round(float(p_val), 4),
                "h0_rejected": h0_rejected,
                "interpretation": interp
            }

    @staticmethod
    def run_f_test_variance(df: pd.DataFrame, metric_a: str, metric_b: str) -> Dict[str, Any]:
        periods = [c for c in df.columns if str(c).lower() not in ["metric", "category", "line_item"]]
        if len(periods) < 3:
            return {"status": "INSUFFICIENT_DATA", "detail": "At least 3 cycles needed for F-test."}

        vals_a = np.array([safe_val(df, metric_a, p) for p in periods], dtype=float)
        vals_b = np.array([safe_val(df, metric_b, p) for p in periods], dtype=float)

        var_a = np.var(vals_a, ddof=1)
        var_b = np.var(vals_b, ddof=1)

        if var_b <= 0 or var_a <= 0:
            return {"status": "INSUFFICIENT_VARIANCE", "detail": "One of the metrics exhibits zero variance."}

        f_stat = var_a / var_b
        df1 = len(vals_a) - 1
        df2 = len(vals_b) - 1
        p_val = 2 * min(stats.f.cdf(f_stat, df1, df2), 1 - stats.f.cdf(f_stat, df1, df2))

        h0_rejected = p_val < 0.05
        interp = (
            f"Significant variance inequality (F = {f_stat:.2f}, p = {p_val:.4f} < 0.05). "
            f"Volatility profile of {metric_a} diverges from {metric_b}."
            if h0_rejected else
            f"Equal variance hypothesis accepted (F = {f_stat:.2f}, p = {p_val:.4f} >= 0.05). Consistent variance profiles."
        )

        return {
            "test_type": f"Two-Tailed F-Test for Equality of Variances ({metric_a} vs {metric_b})",
            "f_statistic": round(float(f_stat), 4),
            "df1": df1,
            "df2": df2,
            "variance_a": round(float(var_a), 2),
            "variance_b": round(float(var_b), 2),
            "p_value": round(float(p_val), 4),
            "h0_rejected": h0_rejected,
            "interpretation": interp
        }