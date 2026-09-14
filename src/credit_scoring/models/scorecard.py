"""
Credit Risk Regulatory Scorecard Model and Points Scaler.
Standard FICO / Basel II scaling formula:
Score = Offset - Factor * ln(odds of bad)
Score = Offset + Factor * ln(odds of good)

Where:
Factor = PDO / ln(2)
Offset = Target_Score - Factor * ln(Target_Odds)
Points per attribute bin = -(beta_i * WOE_ij + alpha / n) * Factor + Offset / n
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from sklearn.linear_model import LogisticRegression

class RegulatoryScorecard:
    """
    Standard Credit Scorecard with points per characteristic.
    Fully interpretable, additive scoring model.
    """

    def __init__(self, pdo: float = 40.0, target_score: float = 650.0, target_odds: float = 5.0):
        """
        Parameters:
        - pdo: Points to Double the Odds (e.g. 40 points increases odds of good by 2x)
        - target_score: Base score (e.g. 650)
        - target_odds: Good:Bad odds at target score (e.g. 5:1, matching ~16-20% default portfolio baseline)
        """
        self.pdo = pdo
        self.target_score = target_score
        self.target_odds = target_odds
        
        # Factor and Offset
        self.factor = self.pdo / np.log(2)
        self.offset = self.target_score - (self.factor * np.log(self.target_odds))
        
        # Underlying Logistic Regression
        self.lr_model = LogisticRegression(penalty="l2", C=1.0, max_iter=1000, solver="lbfgs")
        self.feature_names: List[str] = []
        self.coefficients: Dict[str, float] = {}
        self.intercept: float = 0.0
        self.points_table: Dict[str, Dict[str, float]] = {}

    def fit(self, X_woe: pd.DataFrame, y: pd.Series, woe_dict: Dict[str, Any]):
        """
        Fits Logistic Regression on WOE-transformed features and creates the scorecard point table.
        Note: WOE values represent ln(%good / %bad).
        """
        self.feature_names = list(X_woe.columns)
        self.lr_model.fit(X_woe, y)
        
        self.intercept = float(self.lr_model.intercept_[0])
        self.coefficients = {col: float(coef) for col, coef in zip(self.feature_names, self.lr_model.coef_[0])}
        
        # Build Points Table
        # Formula for points allocated to bin j of feature i:
        # Points_ij = - (beta_i * WOE_ij + (alpha / n)) * Factor + (Offset / n)
        n_features = len(self.feature_names)
        self.points_table = {}

        for raw_col, info in woe_dict.items():
            woe_col = f"{raw_col}_woe"
            if woe_col not in self.coefficients:
                continue

            beta = self.coefficients[woe_col]
            woe_map = info["woe_map"]
            
            bin_points = {}
            for bin_label, woe_val in woe_map.items():
                # In standard credit scoring:
                # Higher score = lower risk (Good)
                # Bad log-odds = alpha + sum(beta_i * WOE_i)
                # Points contribution:
                pt = - (beta * woe_val + (self.intercept / n_features)) * self.factor + (self.offset / n_features)
                bin_points[bin_label] = round(float(pt), 1)

            self.points_table[raw_col] = bin_points

        return self

    def predict_proba(self, X_woe: pd.DataFrame) -> np.ndarray:
        """Returns predicted probability [P(Good), P(Bad)]"""
        return self.lr_model.predict_proba(X_woe)

    def calculate_score_from_proba(self, prob_bad: np.ndarray) -> np.ndarray:
        """
        Converts probability of default (bad) to standard FICO-like credit score (300 - 850).
        odds_good = (1 - prob_bad) / prob_bad
        score = offset + factor * ln(odds_good)
        """
        prob_bad = np.clip(prob_bad, 1e-6, 1.0 - 1e-6)
        odds_good = (1.0 - prob_bad) / prob_bad
        score = self.offset + self.factor * np.log(odds_good)
        # Standard Credit Bureau FICO scale: 300 to 850
        score = np.clip(score, 300.0, 850.0)
        return score

    def calculate_score_and_breakdown(self, applicant_bins: Dict[str, str]) -> Dict[str, Any]:
        """
        Calculates exact additive score and itemized points for a single applicant
        given their assigned bins per characteristic.
        """
        total_score = 0.0
        breakdown = {}

        for feature, bin_val in applicant_bins.items():
            if feature in self.points_table:
                bins_dict = self.points_table[feature]
                pts = bins_dict.get(bin_val, list(bins_dict.values())[0])
                breakdown[feature] = {
                    "bin": bin_val,
                    "points": pts
                }
                total_score += pts

        # Round and clip
        final_score = int(np.clip(round(total_score), 300, 850))

        # Risk band tiering
        if final_score >= 780:
            risk_tier = "Exceptional"
            rating = "A+"
            approval = "Approved (Prime+)"
        elif final_score >= 720:
            risk_tier = "Very Good"
            rating = "A"
            approval = "Approved (Prime)"
        elif final_score >= 660:
            risk_tier = "Good"
            rating = "B"
            approval = "Approved (Standard)"
        elif final_score >= 600:
            risk_tier = "Fair"
            rating = "C"
            approval = "Manual Review / Conditional"
        else:
            risk_tier = "Poor / High Risk"
            rating = "D/E"
            approval = "Declined"

        return {
            "credit_score": final_score,
            "risk_band": risk_tier,
            "credit_grade": rating,
            "recommendation": approval,
            "points_breakdown": breakdown
        }
