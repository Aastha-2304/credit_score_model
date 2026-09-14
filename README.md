# LendingClub Credit Scoring & Risk Decision Platform

An end-to-end, production-grade Credit Scoring Platform following Basel II / regulatory credit risk modeling standards (Weight of Evidence / Information Value binning, Logistic Regression Scorecard with standard points scaling) alongside a high-performance gradient boosting challenger model (LightGBM) with SHAP explanations, a high-speed FastAPI prediction engine, and an interactive Streamlit simulation & portfolio risk analytics dashboard.

---

## 🏛️ System Architecture

```
                                  +---------------------------------------+
                                  |     LendingClub Accepted Loans        |
                                  |   (data/raw/accepted_loans.csv)       |
                                  +-------------------+-------------------+
                                                      |
                                                      v
                                  +---------------------------------------+
                                  |     Data Layer & Preprocessing        |
                                  |    Target: 0 = Good, 1 = Default      |
                                  +-------------------+-------------------+
                                                      |
                                                      v
                                  +---------------------------------------+
                                  |     WOE Binning & IV Engine           |
                                  |    ln(%Good/%Bad), IV ranking         |
                                  +---------+-------------------+---------+
                                            |                   |
                     (WOE Transformed Data) |                   | (Raw Categorical/Numerical)
                                            v                   v
                     +----------------------+----+    +---------+-------------------+
                     |  Regulatory Scorecard     |    | Challenger ML Model         |
                     |  (Logistic Regression)    |    | (LightGBM Booster)          |
                     +--------------+------------+    +---------+-------------------+
                                    |                           |
                                    v                           v
                     +--------------+------------+    +---------+-------------------+
                     |  FICO Scorecard Scaling   |    | Tree SHAP Explainability    |
                     |  PDO=40, Base=650 @ 5:1   |    | Feature attribution impact  |
                     +--------------+------------+    +---------+-------------------+
                                    \                           /
                                     \                         /
                                      v                       v
                         +-----------------------------------------------+
                         |   Credit Risk Evaluation & Basel II Metrics   |
                         |   AUC-ROC (0.738), Gini (0.476), KS (0.357)   |
                         +-----------------------+-----------------------+
                                                 |
                                                 v
                         +-----------------------------------------------+
                         |           FastAPI Real-Time Service           |
                         |     POST /score, GET /metrics, /points        |
                         +-----------------------+-----------------------+
                                                 |
                                                 v
                         +-----------------------------------------------+
                         |     Streamlit Risk Analytics Dashboard        |
                         | Underwriting Simulator, Audit, Portfolio Risk |
                         +-----------------------------------------------+
```

---

## 🚀 Key Features

1. **Credit Industry Standard Feature Engineering (WOE & IV)**:
   - Coarse and fine binning for continuous and categorical variables.
   - Weight of Evidence: $\text{WOE}_i = \ln\left(\frac{\% \text{Goods}_i}{\% \text{Bads}_i}\right)$.
   - Information Value: $\text{IV} = \sum (\% \text{Goods}_i - \% \text{Bads}_i) \times \text{WOE}_i$.
   - Feature power ranking (<0.02 unpredictive, 0.02–0.10 weak, 0.10–0.30 medium, 0.30–0.50 strong).

2. **Champion vs. Challenger Modeling Strategy**:
   - **Champion (Scorecard)**: Regulated Logistic Regression on WOE variables. Calibrated to the standard FICO scale (300–850) with points allocated per attribute bin.
   - **Challenger (LightGBM)**: Non-linear gradient boosted tree with fast inference and SHAP tree attribution.

3. **Regulatory Scorecard Scaling**:
   - $\text{Factor} = \frac{PDO}{\ln(2)}$
   - $\text{Offset} = \text{Target\_Score} - \text{Factor} \times \ln(\text{Target\_Odds})$
   - $\text{Points}_{ij} = -\left(\beta_i \times \text{WOE}_{ij} + \frac{\alpha}{n}\right) \times \text{Factor} + \frac{\text{Offset}}{n}$

4. **Comprehensive Basel II Credit Metrics**:
   - **AUC-ROC**: Discrimination performance ($0.738$ scorecard, $0.741$ challenger).
   - **Kolmogorov-Smirnov (KS) Statistic**: Maximum separation between cumulative Good vs. Bad distributions ($KS = 0.357$).
   - **Gini Coefficient**: $2 \times AUC - 1 = 0.476$.

5. **Production FastAPI Service**:
   - `/score`: Accepts applicant parameters and returns FICO credit score, probability of default, risk band, points breakdown, and top SHAP risk factors.
   - `/metrics`: Model validation metrics.
   - `/scorecard-points`: Full points-per-bin lookup dictionary.

6. **Interactive Streamlit Risk Dashboard**:
   - **Real-Time Underwriting Simulator**: Adjust loan amount, income, DTI, revol util, grade, and delinquencies with live gauge updates and points waterfall chart.
   - **Model Performance & Audit**: Interactive ROC curves, KS curves, and regulatory comparison tables.
   - **WOE & IV Engine**: Attribute binning profiles and monotonicity checks.
   - **Portfolio Risk Strategy**: Dynamic cutoff threshold slider showing portfolio approval rate vs bad debt rate trade-offs.

---

## ⚡ Quickstart Guide

### 1. Installation
Ensure Python 3.10+ is installed:
```bash
pip install -r requirements.txt
```

### 2. Run the End-to-End Pipeline
Train models, calculate WOE/IV, calibrate the scorecard, and generate artifacts:
```bash
python train_pipeline.py
```

### 3. Launch the FastAPI Prediction Microservice
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive Swagger docs will be accessible at: `http://127.0.0.1:8000/docs`.

### 4. Launch the Streamlit Dashboard
```bash
streamlit run dashboard/app.py
```
Dashboard will open at: `http://localhost:8501`.

---

## 📁 Repository Structure
```
Credit Scoring Model/
├── api/
│   └── main.py                     # FastAPI REST scoring endpoints
├── dashboard/
│   └── app.py                      # Streamlit fintech risk platform
├── data/
│   └── raw/                        # LendingClub accepted loans storage
├── src/
│   └── credit_scoring/
│       ├── data/
│       │   └── loader.py           # Ingestion, cleaning & benchmark generator
│       ├── features/
│       │   └── woe_iv.py           # WOE binning and IV calculation
│       ├── models/
│       │   ├── scorecard.py        # Regulatory Scorecard & Points Scaler
│       │   └── challenger.py       # LightGBM & SHAP explainer
│       └── evaluation/
│           └── metrics.py          # AUC-ROC, KS statistic, Gini index
├── artifacts/                      # Serialized models, metrics & point tables
├── train_pipeline.py               # Orchestration pipeline CLI
├── test_api.py                     # Automated integration test script
└── requirements.txt                # Project dependencies
```
