"""
FastAPI Microservice for Real-Time Credit Scoring and Risk Analysis.
"""
import os
import json
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional

app = FastAPI(
    title="LendingClub Credit Scoring & Risk Decision API",
    description="Regulatory WOE Scorecard (FICO points) and Challenger ML Real-Time Scoring Service",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global artifacts holders
ARTIFACTS_DIR = os.environ.get("ARTIFACTS_DIR", "artifacts")
woe_transformer = None
scorecard_model = None
challenger_model = None
metrics_data = None
points_data = None

def load_models():
    global woe_transformer, scorecard_model, challenger_model, metrics_data, points_data
    try:
        woe_path = os.path.join(ARTIFACTS_DIR, "woe_transformer.joblib")
        sc_path = os.path.join(ARTIFACTS_DIR, "scorecard_model.joblib")
        cl_path = os.path.join(ARTIFACTS_DIR, "challenger_model.joblib")
        metrics_path = os.path.join(ARTIFACTS_DIR, "metrics.json")
        points_path = os.path.join(ARTIFACTS_DIR, "scorecard_points.json")

        if os.path.exists(woe_path):
            woe_transformer = joblib.load(woe_path)
        if os.path.exists(sc_path):
            scorecard_model = joblib.load(sc_path)
        if os.path.exists(cl_path):
            challenger_model = joblib.load(cl_path)
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                metrics_data = json.load(f)
        if os.path.exists(points_path):
            with open(points_path, "r") as f:
                points_data = json.load(f)
    except Exception as e:
        print(f"Artifact loading status: {e}")

@app.on_event("startup")
def startup_event():
    load_models()

class ApplicantRequest(BaseModel):
    loan_amnt: float = Field(..., example=15000.0, description="Requested loan amount ($)")
    term: float = Field(..., example=36.0, description="Loan term in months (36 or 60)")
    int_rate: float = Field(..., example=11.5, description="Interest rate percentage (%)")
    grade: str = Field(..., example="B", description="LendingClub credit grade (A-G)")
    sub_grade: str = Field(..., example="B3", description="LendingClub subgrade (A1-G5)")
    emp_length: Optional[str] = Field("5 years", example="5 years", description="Employment duration")
    home_ownership: str = Field(..., example="MORTGAGE", description="Home ownership: MORTGAGE, RENT, OWN, OTHER")
    annual_inc: float = Field(..., example=75000.0, description="Annual verifiable income ($)")
    purpose: str = Field(..., example="debt_consolidation", description="Loan purpose")
    dti: float = Field(..., example=16.5, description="Debt-to-income ratio (%)")
    delinq_2yrs: int = Field(0, example=0, description="Delinquencies in the past 2 years")
    inq_last_6mths: int = Field(1, example=1, description="Credit inquiries in the past 6 months")
    pub_rec: int = Field(0, example=0, description="Derogatory public records count")
    revol_util: float = Field(..., example=42.0, description="Revolving line utilization rate (%)")
    total_acc: int = Field(..., example=22, description="Total credit lines in credit file")

class ScoringResponse(BaseModel):
    credit_score: int
    score_range: str = "300 - 850"
    probability_of_default: float
    risk_band: str
    credit_grade: str
    recommendation: str
    challenger_pd: Optional[float] = None
    points_breakdown: Optional[Dict[str, Any]] = None
    top_risk_factors: Optional[List[Dict[str, Any]]] = None

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "models_loaded": {
            "scorecard": scorecard_model is not None,
            "woe_transformer": woe_transformer is not None,
            "challenger": challenger_model is not None
        }
    }

@app.get("/metrics")
def get_model_metrics():
    if not metrics_data:
        raise HTTPException(status_code=404, detail="Pipeline metrics not generated yet.")
    return metrics_data

@app.get("/scorecard-points")
def get_scorecard_points_table():
    if not points_data:
        raise HTTPException(status_code=404, detail="Scorecard points table not found.")
    return points_data

@app.post("/score", response_model=ScoringResponse)
def score_applicant(applicant: ApplicantRequest):
    if scorecard_model is None or woe_transformer is None:
        load_models()
        if scorecard_model is None:
            raise HTTPException(status_code=503, detail="Models not trained or loaded. Run train_pipeline.py first.")

    data_dict = applicant.model_dump()
    
    # Process emp_length_num
    emp_str = str(data_dict.get("emp_length", ""))
    digits = "".join([c for c in emp_str if c.isdigit()])
    if "<" in emp_str:
        data_dict["emp_length_num"] = 0.0
    elif digits:
        data_dict["emp_length_num"] = float(digits)
    else:
        data_dict["emp_length_num"] = -1.0

    df_input = pd.DataFrame([data_dict])

    # 1. Champion: WOE transform and Scorecard prediction
    df_woe = woe_transformer.transform(df_input)
    prob_bad_sc = float(scorecard_model.predict_proba(df_woe)[0, 1])
    credit_score = int(round(float(scorecard_model.calculate_score_from_proba(np.array([prob_bad_sc]))[0])))

    # Identify individual bins and points
    points_breakdown = {}
    for col in df_input.columns:
        if col in woe_transformer.woe_dict:
            info = woe_transformer.woe_dict[col]
            vtype = info["var_type"]
            val = df_input[col].iloc[0]
            
            # find matching bin
            matched_bin = "Missing"
            if pd.isna(val):
                matched_bin = "Missing"
            elif vtype == "numeric":
                for b_name in info["woe_map"].keys():
                    if b_name in ["Missing", "Other"]:
                        continue
                    cleaned = b_name.strip("[]() ")
                    parts = cleaned.split(",")
                    if len(parts) == 2:
                        try:
                            l, r = float(parts[0]), float(parts[1])
                            if l < float(val) <= r:
                                matched_bin = b_name
                                break
                        except Exception:
                            pass
            else:
                sval = str(val)
                matched_bin = sval if sval in info["woe_map"] else "Other"

            pts = scorecard_model.points_table.get(col, {}).get(matched_bin, 0.0)
            points_breakdown[col] = {
                "input_value": str(val),
                "assigned_bin": matched_bin,
                "points": pts
            }

    # Assign risk tier
    if credit_score >= 780:
        risk_band = "Exceptional"
        grade = "A+"
        rec = "Approved (Prime+)"
    elif credit_score >= 720:
        risk_band = "Very Good"
        grade = "A"
        rec = "Approved (Prime)"
    elif credit_score >= 660:
        risk_band = "Good"
        grade = "B"
        rec = "Approved (Standard)"
    elif credit_score >= 600:
        risk_band = "Fair"
        grade = "C"
        rec = "Manual Review / Conditional"
    else:
        risk_band = "Poor / High Risk"
        grade = "D/E"
        rec = "Declined"

    # 2. Challenger Model prediction & SHAP
    challenger_pd = None
    top_shap = None
    if challenger_model is not None:
        try:
            # align columns
            cl_cols = challenger_model.feature_names
            aligned_input = df_input.reindex(columns=cl_cols)
            challenger_pd = round(float(challenger_model.predict_proba(aligned_input)[0, 1]), 4)
            top_shap = challenger_model.explain_instance(aligned_input, top_k=5)
        except Exception as ex:
            print(f"Challenger scoring note: {ex}")

    return ScoringResponse(
        credit_score=credit_score,
        probability_of_default=round(prob_bad_sc, 4),
        risk_band=risk_band,
        credit_grade=grade,
        recommendation=rec,
        challenger_pd=challenger_pd,
        points_breakdown=points_breakdown,
        top_risk_factors=top_shap
    )
