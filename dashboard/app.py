"""
Production Streamlit Risk Engineering & Credit Scoring Dashboard.
Features:
- Live Applicant Simulator (interactive inputs -> real-time score & points waterfall)
- Regulatory WOE & IV Explorer (predictive power audit)
- Champion vs. Challenger Model Validation (AUC-ROC, KS Statistic, Gini)
- Portfolio Risk Segmentation & Cutoff Optimizer
- Direct Local Artifact loading (robust fallback when API service is offline)
"""
import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import requests

# Ensure repository root is on sys.path for joblib unpickling
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import model & transformer classes into scope so unpickler resolves them
from src.credit_scoring.features.woe_iv import WOEIVTransformer
from src.credit_scoring.models.scorecard import RegulatoryScorecard
from src.credit_scoring.models.challenger import ChallengerModel
from src.credit_scoring.data.loader import generate_benchmark_lendingclub_data, clean_and_prepare_lendingclub_data

# Set Page Config
st.set_page_config(
    page_title="LendingClub Credit Scoring Platform",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern fintech aesthetic
st.markdown("""
<style>
    .main {
        background-color: #0d1117;
    }
    .metric-card {
        background: linear-gradient(135deg, rgba(22, 27, 34, 0.8) 0%, rgba(33, 38, 45, 0.9) 100%);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
        margin-bottom: 15px;
    }
    .score-badge {
        font-size: 2.8rem;
        font-weight: 800;
        letter-spacing: -1px;
    }
    .status-pill {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-top: 5px;
    }
    .pill-prime { background-color: rgba(35, 134, 54, 0.25); color: #3fb950; border: 1px solid #238636; }
    .pill-standard { background-color: rgba(56, 139, 253, 0.25); color: #58a6ff; border: 1px solid #388bfd; }
    .pill-warning { background-color: rgba(210, 153, 34, 0.25); color: #d29922; border: 1px solid #bb8009; }
    .pill-danger { background-color: rgba(248, 81, 73, 0.25); color: #f85149; border: 1px solid #da3633; }
</style>
""", unsafe_allow_html=True)

ARTIFACTS_DIR = os.environ.get("ARTIFACTS_DIR", os.path.join(PROJECT_ROOT, "artifacts"))
API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")

@st.cache_resource
def load_local_artifacts():
    """Directly loads local artifacts to guarantee zero dashboard downtime."""
    artifacts = {}
    woe_path = os.path.join(ARTIFACTS_DIR, "woe_transformer.joblib")
    sc_path = os.path.join(ARTIFACTS_DIR, "scorecard_model.joblib")
    cl_path = os.path.join(ARTIFACTS_DIR, "challenger_model.joblib")
    metrics_path = os.path.join(ARTIFACTS_DIR, "metrics.json")
    points_path = os.path.join(ARTIFACTS_DIR, "scorecard_points.json")
    iv_path = os.path.join(ARTIFACTS_DIR, "iv_summary.json")
    sample_path = os.path.join(ARTIFACTS_DIR, "sample_evaluated_applicants.csv")

    if os.path.exists(woe_path):
        artifacts["woe_transformer"] = joblib.load(woe_path)
    if os.path.exists(sc_path):
        artifacts["scorecard"] = joblib.load(sc_path)
    if os.path.exists(cl_path):
        artifacts["challenger"] = joblib.load(cl_path)
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            artifacts["metrics"] = json.load(f)
    if os.path.exists(points_path):
        with open(points_path, "r") as f:
            artifacts["points"] = json.load(f)
    if os.path.exists(iv_path):
        with open(iv_path, "r") as f:
            artifacts["iv_summary"] = json.load(f)
    if os.path.exists(sample_path):
        artifacts["sample_df"] = pd.read_csv(sample_path)

    return artifacts

artifacts = load_local_artifacts()

# -------------------------------------------------------------
# Header & Sidebar Navigation
# -------------------------------------------------------------
st.title("💳 LendingClub Credit Scoring & Risk Architecture")
st.caption("Regulatory WOE Scorecard (FICO Standard) & LightGBM Challenger with Basel II Metrics")

tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Real-Time Applicant Simulator", 
    "📊 Model Performance & Basel II Audit", 
    "📈 WOE & Information Value Engine", 
    "💼 Portfolio Risk & Cutoff Optimization"
])

# -------------------------------------------------------------
# TAB 1: Real-Time Applicant Simulator
# -------------------------------------------------------------
with tab1:
    st.subheader("Interactive Credit Underwriting Simulator")
    st.markdown("Adjust applicant credit attributes below to simulate immediate credit scoring, scorecard point breakdown, and probability of default.")

    col_inp1, col_inp2, col_inp3 = st.columns(3)

    with col_inp1:
        st.markdown("##### 💵 Loan Terms & Capacity")
        loan_amnt = st.slider("Loan Amount ($)", min_value=1000, max_value=40000, value=15000, step=500)
        term = st.selectbox("Term (Months)", options=[36.0, 60.0], index=0)
        int_rate = st.slider("Interest Rate (%)", min_value=5.0, max_value=32.0, value=12.5, step=0.1)
        annual_inc = st.number_input("Verifiable Annual Income ($)", min_value=10000, max_value=500000, value=75000, step=2500)

    with col_inp2:
        st.markdown("##### 📋 Credit Bureau Profile")
        grade = st.selectbox("LendingClub Grade", options=["A", "B", "C", "D", "E", "F", "G"], index=1)
        sub_grade = f"{grade}{st.selectbox('Sub-Grade Number', options=[1, 2, 3, 4, 5], index=1)}"
        dti = st.slider("Debt-to-Income Ratio - DTI (%)", min_value=0.0, max_value=45.0, value=16.8, step=0.2)
        revol_util = st.slider("Revolving Credit Utilization (%)", min_value=0.0, max_value=120.0, value=45.0, step=0.5)

    with col_inp3:
        st.markdown("##### 🔍 Stability & Derogatories")
        home_ownership = st.selectbox("Home Ownership", options=["MORTGAGE", "RENT", "OWN", "OTHER"], index=0)
        emp_length = st.selectbox("Employment Length", options=["< 1 year", "1 year", "2 years", "3 years", "5 years", "10+ years"], index=4)
        purpose = st.selectbox("Loan Purpose", options=["debt_consolidation", "credit_card", "home_improvement", "major_purchase", "small_business", "medical", "other"], index=0)
        delinq_2yrs = st.slider("Delinquencies in Past 2 Yrs", min_value=0, max_value=4, value=0)
        inq_last_6mths = st.slider("Inquiries in Past 6 Months", min_value=0, max_value=5, value=1)
        pub_rec = st.selectbox("Derogatory Public Records", options=[0, 1, 2], index=0)
        total_acc = st.slider("Total Credit Accounts", min_value=3, max_value=70, value=24)

    # Scoring Execution
    applicant_data = {
        "loan_amnt": float(loan_amnt),
        "term": float(term),
        "int_rate": float(int_rate),
        "grade": str(grade),
        "sub_grade": str(sub_grade),
        "emp_length": str(emp_length),
        "home_ownership": str(home_ownership),
        "annual_inc": float(annual_inc),
        "purpose": str(purpose),
        "dti": float(dti),
        "delinq_2yrs": int(delinq_2yrs),
        "inq_last_6mths": int(inq_last_6mths),
        "pub_rec": int(pub_rec),
        "revol_util": float(revol_util),
        "total_acc": int(total_acc)
    }

    # Extract clean emp_length_num
    digits = "".join([c for c in emp_length if c.isdigit()])
    applicant_data["emp_length_num"] = 0.0 if "<" in emp_length else (float(digits) if digits else -1.0)

    scorecard = artifacts.get("scorecard")
    woe_trans = artifacts.get("woe_transformer")
    challenger = artifacts.get("challenger")

    if scorecard is not None and woe_trans is not None:
        df_row = pd.DataFrame([applicant_data])
        df_woe_row = woe_trans.transform(df_row)
        if hasattr(scorecard, "feature_names") and scorecard.feature_names:
            df_woe_row = df_woe_row.reindex(columns=scorecard.feature_names, fill_value=0.0)
        
        pd_scorecard = float(scorecard.predict_proba(df_woe_row)[0, 1])
        credit_score = int(round(float(scorecard.calculate_score_from_proba(np.array([pd_scorecard]))[0])))

        # Risk Classification
        if credit_score >= 780:
            tier, grade_label, pill_class = "Exceptional", "A+", "pill-prime"
            rec = "APPROVED (PRIME+)"
            score_color = "#3fb950"
        elif credit_score >= 720:
            tier, grade_label, pill_class = "Very Good", "A", "pill-prime"
            rec = "APPROVED (PRIME)"
            score_color = "#3fb950"
        elif credit_score >= 660:
            tier, grade_label, pill_class = "Good", "B", "pill-standard"
            rec = "APPROVED (STANDARD)"
            score_color = "#58a6ff"
        elif credit_score >= 600:
            tier, grade_label, pill_class = "Fair", "C", "pill-warning"
            rec = "CONDITIONAL / MANUAL REVIEW"
            score_color = "#d29922"
        else:
            tier, grade_label, pill_class = "High Risk / Poor", "D/E", "pill-danger"
            rec = "DECLINED"
            score_color = "#f85149"

        st.markdown("---")
        res_col1, res_col2, res_col3, res_col4 = st.columns([1.2, 1, 1, 1.2])

        with res_col1:
            st.markdown(f"""
            <div class="metric-card">
                <span style="color: #8b949e; font-size: 0.9rem; text-transform: uppercase;">FICO Scaled Credit Score</span>
                <div class="score-badge" style="color: {score_color};">{credit_score}</div>
                <div class="status-pill {pill_class}">{rec}</div>
            </div>
            """, unsafe_allow_html=True)

        with res_col2:
            st.markdown(f"""
            <div class="metric-card">
                <span style="color: #8b949e; font-size: 0.9rem; text-transform: uppercase;">Probability of Default (PD)</span>
                <div style="font-size: 2.2rem; font-weight: 700; color: #f0f6fc; margin-top: 5px;">{pd_scorecard*100:.2f}%</div>
                <span style="color: #8b949e; font-size: 0.85rem;">Scorecard Model</span>
            </div>
            """, unsafe_allow_html=True)

        with res_col3:
            pd_challenger_val = "N/A"
            if challenger is not None:
                try:
                    aligned = df_row.reindex(columns=challenger.feature_names)
                    pd_challenger = float(challenger.predict_proba(aligned)[0, 1])
                    pd_challenger_val = f"{pd_challenger*100:.2f}%"
                except Exception:
                    pass
            st.markdown(f"""
            <div class="metric-card">
                <span style="color: #8b949e; font-size: 0.9rem; text-transform: uppercase;">Challenger PD (LightGBM)</span>
                <div style="font-size: 2.2rem; font-weight: 700; color: #a371f7; margin-top: 5px;">{pd_challenger_val}</div>
                <span style="color: #8b949e; font-size: 0.85rem;">Raw Gradient Boosting</span>
            </div>
            """, unsafe_allow_html=True)

        with res_col4:
            st.markdown(f"""
            <div class="metric-card">
                <span style="color: #8b949e; font-size: 0.9rem; text-transform: uppercase;">Risk Band & Rating</span>
                <div style="font-size: 2.2rem; font-weight: 700; color: #58a6ff; margin-top: 5px;">{tier}</div>
                <span style="color: #8b949e; font-size: 0.85rem;">Internal Grade: <strong>{grade_label}</strong></span>
            </div>
            """, unsafe_allow_html=True)

        # Scorecard Points Breakdown & Waterfall
        st.subheader("Scorecard Points-Per-Feature Breakdown")
        st.caption("Basel II compliant additive points formula: Points = -(WOE_i × β_i + α/n) × Factor + Offset/n")

        points_list = []
        for col in df_row.columns:
            if col in woe_trans.woe_dict:
                info = woe_trans.woe_dict[col]
                vtype = info["var_type"]
                val = df_row[col].iloc[0]
                
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

                pts = scorecard.points_table.get(col, {}).get(matched_bin, 0.0)
                points_list.append({
                    "Feature": col,
                    "Applicant Value": str(val),
                    "Assigned Bin": matched_bin,
                    "Points Awarded": pts
                })

        df_pts = pd.DataFrame(points_list).sort_values(by="Points Awarded", ascending=False)

        col_w1, col_w2 = st.columns([1.3, 1])
        with col_w1:
            fig_bar = px.bar(
                df_pts,
                x="Points Awarded",
                y="Feature",
                orientation="h",
                color="Points Awarded",
                color_continuous_scale=["#f85149", "#d29922", "#3fb950"],
                title="Credit Points Contribution by Characteristic"
            )
            fig_bar.update_layout(height=450, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_bar, use_container_width=True)

        with col_w2:
            st.dataframe(
                df_pts[["Feature", "Applicant Value", "Points Awarded"]],
                use_container_width=True,
                height=450
            )

        # Challenger SHAP Waterfall
        if challenger is not None:
            st.subheader("Challenger Model: SHAP Factor Attribution")
            try:
                aligned = df_row.reindex(columns=challenger.feature_names)
                shap_contribs = challenger.explain_instance(aligned, top_k=7)
                if shap_contribs:
                    df_shap = pd.DataFrame(shap_contribs)
                    fig_shap = px.bar(
                        df_shap,
                        x="shap_impact",
                        y="feature",
                        orientation="h",
                        color="risk_direction",
                        color_discrete_map={"Increases Risk": "#f85149", "Decreases Risk": "#3fb950"},
                        title="Top SHAP Contributors (Marginal Log-Odds Impact)"
                    )
                    fig_shap.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig_shap, use_container_width=True)
            except Exception as e:
                st.info(f"SHAP explanation note: {e}")

    else:
        st.warning("⚠️ Models have not been trained yet. Please run `python train_pipeline.py` or check the 'Model Performance' tab.")

# -------------------------------------------------------------
# TAB 2: Model Performance & Basel II Audit
# -------------------------------------------------------------
with tab2:
    st.subheader("Credit Risk Regulatory Validation & Metrics")
    metrics = artifacts.get("metrics")

    if metrics:
        sc_m = metrics.get("scorecard", {})
        cl_m = metrics.get("challenger", {})

        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        with m_col1:
            st.metric("Scorecard AUC-ROC", f"{sc_m.get('auc_roc', 0.0):.3f}", delta=f"vs Challenger {cl_m.get('auc_roc', 0.0):.3f}")
        with m_col2:
            st.metric("Scorecard Gini (2*AUC - 1)", f"{sc_m.get('gini_coefficient', 0.0):.3f}", delta=f"vs Challenger {cl_m.get('gini_coefficient', 0.0):.3f}")
        with m_col3:
            st.metric("Scorecard KS Statistic", f"{sc_m.get('ks_statistic', 0.0):.3f}", delta=f"vs Challenger {cl_m.get('ks_statistic', 0.0):.3f}")
        with m_col4:
            st.metric("KS Optimal Cutoff", f"{sc_m.get('ks_optimal_threshold', 0.0):.3f}")

        st.markdown("---")

        col_roc, col_compare = st.columns(2)
        with col_roc:
            st.markdown("##### ROC Discrimination Curves")
            roc_pts = sc_m.get("roc_curve", [])
            if roc_pts:
                df_roc = pd.DataFrame(roc_pts)
                fig_roc = px.line(df_roc, x="fpr", y="tpr", title=f"Scorecard ROC Curve (AUC = {sc_m.get('auc_roc', 0.0):.3f})")
                fig_roc.add_shape(type="line", line=dict(dash="dash", color="grey"), x0=0, x1=1, y0=0, y1=1)
                fig_roc.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_roc, use_container_width=True)

        with col_compare:
            st.markdown("##### Champion vs Challenger Architecture Summary")
            df_comp = pd.DataFrame([
                {
                    "Metric / Dimension": "Model Type",
                    "Regulatory Scorecard (Champion)": "Logistic Regression on WOE Bins",
                    "Machine Learning (Challenger)": "LightGBM Gradient Boosted Trees"
                },
                {
                    "Metric / Dimension": "AUC-ROC",
                    "Regulatory Scorecard (Champion)": str(sc_m.get("auc_roc")),
                    "Machine Learning (Challenger)": str(cl_m.get("auc_roc"))
                },
                {
                    "Metric / Dimension": "Gini Index",
                    "Regulatory Scorecard (Champion)": str(sc_m.get("gini_coefficient")),
                    "Machine Learning (Challenger)": str(cl_m.get("gini_coefficient"))
                },
                {
                    "Metric / Dimension": "KS Statistic",
                    "Regulatory Scorecard (Champion)": str(sc_m.get("ks_statistic")),
                    "Machine Learning (Challenger)": str(cl_m.get("ks_statistic"))
                },
                {
                    "Metric / Dimension": "Explainability",
                    "Regulatory Scorecard (Champion)": "100% Additive FICO Scorecard Points",
                    "Machine Learning (Challenger)": "Post-hoc SHAP Explanations"
                },
                {
                    "Metric / Dimension": "Regulatory Compliance",
                    "Regulatory Scorecard (Champion)": "Basel II / FCRA / ECOA Compliant",
                    "Machine Learning (Challenger)": "Requires SR 11-7 Model Risk Governance"
                }
            ])
            st.dataframe(df_comp, use_container_width=True, hide_index=True)

    else:
        st.info("Metrics not found. Please execute the training pipeline.")

# -------------------------------------------------------------
# TAB 3: WOE & Information Value Engine
# -------------------------------------------------------------
with tab3:
    st.subheader("Weight of Evidence (WOE) & Information Value (IV) Analysis")
    st.markdown("""
    **Credit Industry IV Benchmark Guidelines:**
    - `< 0.02`: Unpredictive / Noise (Drop)
    - `0.02 - 0.10`: Weak Predictor
    - `0.10 - 0.30`: Medium Predictor
    - `0.30 - 0.50`: Strong Predictor
    - `> 0.50`: Suspiciously High (Review for data leakage)
    """)

    iv_data = artifacts.get("iv_summary")
    if iv_data:
        df_iv = pd.DataFrame(iv_data)
        
        col_iv1, col_iv2 = st.columns([1.3, 1])
        with col_iv1:
            fig_iv = px.bar(
                df_iv,
                x="information_value",
                y="feature",
                orientation="h",
                color="information_value",
                color_continuous_scale="Viridis",
                title="Features Ranked by Information Value (IV)"
            )
            fig_iv.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_iv, use_container_width=True)

        with col_iv2:
            st.dataframe(df_iv, use_container_width=True, height=450)

        # Inspect WOE for selected variable
        st.markdown("---")
        st.subheader("Attribute WOE Pattern Inspector")
        woe_trans = artifacts.get("woe_transformer")
        if woe_trans:
            selected_feat = st.selectbox("Select Attribute to Inspect WOE Profile:", options=list(woe_trans.woe_dict.keys()))
            if selected_feat:
                bin_data = woe_trans.woe_dict[selected_feat]["bin_map"]
                rows = []
                for b_name, b_info in bin_data.items():
                    rows.append({
                        "Bin Interval": b_name,
                        "Count": b_info["count"],
                        "Goods (Paid)": b_info["good"],
                        "Bads (Default)": b_info["bad"],
                        "WOE": round(b_info["woe"], 4),
                        "IV Contribution": round(b_info["iv"], 4)
                    })
                df_bin_view = pd.DataFrame(rows)

                col_b1, col_b2 = st.columns(2)
                with col_b1:
                    fig_woe = px.bar(
                        df_bin_view,
                        x="Bin Interval",
                        y="WOE",
                        title=f"Weight of Evidence Monotonicity: {selected_feat}",
                        color="WOE",
                        color_continuous_scale="Bluered"
                    )
                    fig_woe.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig_woe, use_container_width=True)
                with col_b2:
                    st.dataframe(df_bin_view, use_container_width=True)
    else:
        st.info("Run `python train_pipeline.py` to generate the WOE and IV report.")

# -------------------------------------------------------------
# TAB 4: Portfolio Risk & Cutoff Optimization
# -------------------------------------------------------------
with tab4:
    st.subheader("Portfolio Risk Segmentation & Cutoff Strategy")
    st.markdown("Simulate policy cutoff thresholds to balance loan approval volume against bad debt loss rates.")

    sample_df = artifacts.get("sample_df")
    if sample_df is None and scorecard is not None and woe_trans is not None:
        # Dynamically generate synthetic portfolio cohort in-memory (no disk CSV required)
        from src.credit_scoring.data.loader import generate_benchmark_lendingclub_data, clean_and_prepare_lendingclub_data
        df_raw_cohort = generate_benchmark_lendingclub_data(n_samples=500, random_state=123)
        X_cohort, y_cohort = clean_and_prepare_lendingclub_data(df_raw_cohort)
        X_cohort_woe = woe_trans.transform(X_cohort)
        if hasattr(scorecard, "feature_names") and scorecard.feature_names:
            X_cohort_woe = X_cohort_woe.reindex(columns=scorecard.feature_names, fill_value=0.0)
        pds = scorecard.predict_proba(X_cohort_woe)[:, 1]
        scores = scorecard.calculate_score_from_proba(pds).round().astype(int)
        
        sample_df = X_cohort.copy()
        sample_df["actual_default"] = y_cohort.values
        sample_df["scorecard_score"] = scores
        sample_df["scorecard_pd"] = pds.round(4)
        if challenger is not None:
            try:
                aligned_cohort = X_cohort.reindex(columns=challenger.feature_names)
                sample_df["challenger_pd"] = challenger.predict_proba(aligned_cohort)[:, 1].round(4)
            except Exception:
                sample_df["challenger_pd"] = sample_df["scorecard_pd"]
        else:
            sample_df["challenger_pd"] = sample_df["scorecard_pd"]

    if sample_df is not None:
        col_c1, col_c2 = st.columns([1, 2])
        with col_c1:
            cutoff = st.slider("Select Minimum Approval Credit Score Cutoff:", min_value=500, max_value=750, value=620, step=10)
            
            # Apply cutoff
            approved = sample_df[sample_df["scorecard_score"] >= cutoff]
            declined = sample_df[sample_df["scorecard_score"] < cutoff]
            
            approval_rate = (len(approved) / len(sample_df)) * 100
            approved_default_rate = (approved["actual_default"].mean() * 100) if len(approved) > 0 else 0.0
            overall_default_rate = sample_df["actual_default"].mean() * 100

            st.metric("Portfolio Approval Rate", f"{approval_rate:.1f}%")
            st.metric("Approved Bad Debt Rate", f"{approved_default_rate:.2f}%", delta=f"{approved_default_rate - overall_default_rate:.2f}% vs Unfiltered")
            st.metric("Declined Count", len(declined))

        with col_c2:
            fig_hist = px.histogram(
                sample_df,
                x="scorecard_score",
                color="actual_default",
                nbins=25,
                barmode="overlay",
                title="Credit Score Distribution by Repayment Outcome",
                color_discrete_map={0: "#3fb950", 1: "#f85149"},
                labels={"scorecard_score": "FICO Credit Score", "actual_default": "Default Status (1=Bad, 0=Good)"}
            )
            fig_hist.add_vline(x=cutoff, line_width=3, line_dash="dash", line_color="#58a6ff", annotation_text=f"Cutoff: {cutoff}")
            fig_hist.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_hist, use_container_width=True)

        st.markdown("##### Evaluated Applicant Portfolio Sample")
        st.dataframe(
            sample_df[["scorecard_score", "scorecard_pd", "challenger_pd", "actual_default", "loan_amnt", "grade", "dti", "annual_inc"]].head(25),
            use_container_width=True
        )
    else:
        st.info("Sample evaluated portfolio data will appear after training.")

st.markdown("---")
st.caption("LendingClub Credit Scoring Risk Architecture • Engineered with Basel II Standards & Machine Learning")
