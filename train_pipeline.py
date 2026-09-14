"""
End-to-end training and serialization pipeline for Credit Scoring Platform.
"""
import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from src.credit_scoring.data.loader import load_or_create_dataset
from src.credit_scoring.features.woe_iv import WOEIVTransformer
from src.credit_scoring.models.scorecard import RegulatoryScorecard
from src.credit_scoring.models.challenger import ChallengerModel
from src.credit_scoring.evaluation.metrics import evaluate_credit_model

def run_pipeline(n_samples: int = 30000, artifacts_dir: str = "artifacts"):
    os.makedirs(artifacts_dir, exist_ok=True)
    print("=" * 60)
    print("1. Ingesting & Preparing LendingClub Dataset...")
    print("=" * 60)
    X, y = load_or_create_dataset(n_samples=n_samples)
    print(f"Total Portfolio: {len(X)} accounts | Default Rate: {y.mean()*100:.2f}%")

    # Train / Test split (80/20) with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print("\n" + "=" * 60)
    print("2. Performing Basel II WOE Binning & Information Value Analysis...")
    print("=" * 60)
    woe_engine = WOEIVTransformer(max_bins=5, min_sample_pct=0.05)
    woe_engine.fit(X_train, y_train)

    # Transform to WOE matrices
    X_train_woe = woe_engine.transform(X_train)
    X_test_woe = woe_engine.transform(X_test)

    iv_df = woe_engine.get_iv_summary_df()
    print("\nInformation Value (IV) Summary:")
    print(iv_df.to_string(index=False))

    print("\n" + "=" * 60)
    print("3. Training Champion Model: Regulatory Scorecard (Logistic Regression)...")
    print("=" * 60)
    scorecard = RegulatoryScorecard(pdo=40.0, target_score=650.0, target_odds=5.0)
    scorecard.fit(X_train_woe, y_train, woe_engine.woe_dict)

    # Scorecard evaluation
    y_test_pred_lr = scorecard.predict_proba(X_test_woe)[:, 1]
    lr_metrics = evaluate_credit_model(y_test.values, y_test_pred_lr, model_name="Regulatory Scorecard (Logistic Regression)")
    print(f"Scorecard AUC-ROC: {lr_metrics['auc_roc']} | Gini: {lr_metrics['gini_coefficient']} | KS Statistic: {lr_metrics['ks_statistic']}")

    print("\n" + "=" * 60)
    print("4. Training Challenger Model: LightGBM Tree Booster...")
    print("=" * 60)
    challenger = ChallengerModel(random_state=42)
    challenger.fit(X_train, y_train)

    # Challenger evaluation
    y_test_pred_lgb = challenger.predict_proba(X_test)[:, 1]
    lgb_metrics = evaluate_credit_model(y_test.values, y_test_pred_lgb, model_name="Challenger (LightGBM)")
    print(f"Challenger AUC-ROC: {lgb_metrics['auc_roc']} | Gini: {lgb_metrics['gini_coefficient']} | KS Statistic: {lgb_metrics['ks_statistic']}")

    print("\n" + "=" * 60)
    print("5. Serializing Artifacts for Production Service & Dashboard...")
    print("=" * 60)
    # Save models and transformers
    joblib.dump(woe_engine, os.path.join(artifacts_dir, "woe_transformer.joblib"))
    joblib.dump(scorecard, os.path.join(artifacts_dir, "scorecard_model.joblib"))
    joblib.dump(challenger, os.path.join(artifacts_dir, "challenger_model.joblib"))

    # Save reference test sample for dashboard preview
    sample_df = X_test.head(100).copy()
    sample_df["actual_default"] = y_test.head(100).values
    sample_df["scorecard_score"] = scorecard.calculate_score_from_proba(y_test_pred_lr[:100]).round().astype(int)
    sample_df["scorecard_pd"] = y_test_pred_lr[:100].round(4)
    sample_df["challenger_pd"] = y_test_pred_lgb[:100].round(4)
    sample_df.to_csv(os.path.join(artifacts_dir, "sample_evaluated_applicants.csv"), index=False)

    # Save metrics and points table
    with open(os.path.join(artifacts_dir, "scorecard_points.json"), "w") as f:
        json.dump(scorecard.points_table, f, indent=2)

    with open(os.path.join(artifacts_dir, "iv_summary.json"), "w") as f:
        json.dump(iv_df.to_dict(orient="records"), f, indent=2)

    with open(os.path.join(artifacts_dir, "metrics.json"), "w") as f:
        json.dump({
            "scorecard": lr_metrics,
            "challenger": lgb_metrics
        }, f, indent=2)

    print(f"All pipeline artifacts successfully saved in '{artifacts_dir}/'!")

if __name__ == "__main__":
    run_pipeline()
