"""
Credit Risk Evaluation Suite:
- AUC-ROC
- Kolmogorov-Smirnov (KS) statistic
- Gini Coefficient (2 * AUC - 1)
- Risk Band Calibration & Lift
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix

def calculate_ks_statistic(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float, pd.DataFrame]:
    """
    Calculates Kolmogorov-Smirnov (KS) statistic, which measures the maximum
    separation between the cumulative distribution functions of Good and Bad loans.

    Returns:
    - ks_stat (float): Maximum difference (0 to 1, typically 0.30 - 0.50 in credit)
    - ks_threshold (float): Cutoff probability where KS is maximized
    - ks_df (pd.DataFrame): Dataframe of deciles/quantiles with cumulative goods and bads
    """
    df = pd.DataFrame({"target": y_true, "prob": y_prob})
    # Decile binning
    df["decile"] = pd.qcut(df["prob"], q=10, duplicates="drop")
    
    grouped = df.groupby("decile", observed=False).agg(
        total=("target", "count"),
        bads=("target", "sum")
    ).reset_index()

    grouped["goods"] = grouped["total"] - grouped["bads"]
    grouped["cum_bads"] = grouped["bads"].cumsum() / grouped["bads"].sum()
    grouped["cum_goods"] = grouped["goods"].cumsum() / grouped["goods"].sum()
    grouped["ks"] = (grouped["cum_bads"] - grouped["cum_goods"]).abs()

    max_ks_row = grouped.loc[grouped["ks"].idxmax()]
    ks_stat = float(max_ks_row["ks"])
    
    # Detailed ROC-based KS
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    ks_curve = tpr - fpr
    best_idx = np.argmax(ks_curve)
    exact_ks = float(ks_curve[best_idx])
    exact_threshold = float(thresholds[best_idx])

    return exact_ks, exact_threshold, grouped

def evaluate_credit_model(y_true: np.ndarray, y_prob: np.ndarray, model_name: str = "Model") -> Dict[str, Any]:
    """
    Computes all standard regulatory credit risk metrics.
    """
    auc = float(roc_auc_score(y_true, y_prob))
    gini = float(2.0 * auc - 1.0)
    ks_stat, ks_thresh, ks_table = calculate_ks_statistic(y_true, y_prob)

    # ROC curve data for plotting
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    # Downsample points for lightweight storage/plotting
    step = max(1, len(fpr) // 100)
    roc_points = [
        {"fpr": round(float(f), 4), "tpr": round(float(t), 4)}
        for f, t in zip(fpr[::step], tpr[::step])
    ]

    return {
        "model_name": model_name,
        "auc_roc": round(auc, 4),
        "gini_coefficient": round(gini, 4),
        "ks_statistic": round(ks_stat, 4),
        "ks_optimal_threshold": round(ks_thresh, 4),
        "roc_curve": roc_points
    }
