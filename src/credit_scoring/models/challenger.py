"""
Challenger Machine Learning Model: LightGBM with SHAP Interpretability.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
import lightgbm as lgb
import shap

class ChallengerModel:
    """
    LightGBM Challenger model with native categorical handling
    and tree SHAP value computation.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model = lgb.LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=6,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            objective="binary",
            n_jobs=-1,
            verbose=-1
        )
        self.explainer: Optional[shap.TreeExplainer] = None
        self.feature_names: List[str] = []
        self.categorical_cols: List[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series):
        X = X.copy()
        self.feature_names = list(X.columns)
        
        # Prepare categoricals
        self.categorical_cols = []
        for col in X.columns:
            if X[col].dtype == object or X[col].dtype.name == "category":
                X[col] = X[col].astype("category")
                self.categorical_cols.append(col)

        self.model.fit(X, y)
        
        # Initialize SHAP explainer
        try:
            self.explainer = shap.TreeExplainer(self.model)
        except Exception as e:
            print(f"Warning: could not initialize TreeExplainer: {e}")
            self.explainer = None

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X = X.copy()
        for col in self.categorical_cols:
            if col in X.columns:
                X[col] = X[col].astype("category")
        return self.model.predict_proba(X)

    def explain_instance(self, single_row_df: pd.DataFrame, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Computes SHAP contributions for a single applicant row.
        """
        if self.explainer is None:
            return []

        df = single_row_df.copy()
        for col in self.categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype("category")

        shap_values = self.explainer.shap_values(df)
        
        # Depending on shap version, shap_values can be a list [vals_class0, vals_class1] or array
        if isinstance(shap_values, list):
            sv = shap_values[1][0] # class 1 (Bad/Default)
        elif len(shap_values.shape) == 3:
            sv = shap_values[0, :, 1]
        else:
            sv = shap_values[0]

        contributions = []
        for feat, val, s in zip(self.feature_names, df.iloc[0].values, sv):
            contributions.append({
                "feature": feat,
                "value": str(val),
                "shap_impact": round(float(s), 4),
                "risk_direction": "Increases Risk" if s > 0 else "Decreases Risk"
            })

        # Sort by absolute impact
        contributions.sort(key=lambda x: abs(x["shap_impact"]), reverse=True)
        return contributions[:top_k]
