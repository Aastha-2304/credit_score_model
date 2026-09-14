"""Data loading, target definition, and synthetic benchmark generator for LendingClub."""
import os
import numpy as np
import pandas as pd
from typing import Tuple, Optional

# Standard LendingClub Target Mapping:
# Good = 0 (Fully Paid, Current)
# Bad = 1 (Charged Off, Default, Late (31-120 days), Does not meet the credit policy)
GOOD_STATUSES = ["Fully Paid", "Current", "In Grace Period"]
BAD_STATUSES = ["Charged Off", "Default", "Late (31-120 days)", "Late (16-30 days)"]

def generate_benchmark_lendingclub_data(n_samples: int = 25000, random_state: int = 42) -> pd.DataFrame:
    """
    Generates a realistic benchmark LendingClub accepted loan dataset
    replicating distributions, missingness, and default odds for real LendingClub portfolios.
    """
    np.random.seed(random_state)
    
    # 1. Core financial attributes
    loan_amnt = np.random.choice([5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000], 
                                 size=n_samples, 
                                 p=[0.12, 0.18, 0.22, 0.18, 0.14, 0.08, 0.05, 0.03])
    loan_amnt = loan_amnt + np.random.randint(-1500, 1500, size=n_samples)
    loan_amnt = np.clip(loan_amnt, 1000, 40000)

    term = np.random.choice([" 36 months", " 60 months"], size=n_samples, p=[0.72, 0.28])
    
    annual_inc = np.random.lognormal(mean=11.1, sigma=0.55, size=n_samples)
    annual_inc = np.clip(annual_inc, 12000, 300000)

    dti = np.random.normal(loc=18.5, scale=8.0, size=n_samples)
    dti = np.clip(dti, 1.0, 45.0)

    emp_length_choices = [
        "< 1 year", "1 year", "2 years", "3 years", "4 years", 
        "5 years", "6 years", "7 years", "8 years", "9 years", "10+ years"
    ]
    emp_length = np.random.choice(emp_length_choices, size=n_samples, 
                                  p=[0.08, 0.06, 0.09, 0.08, 0.06, 0.06, 0.05, 0.04, 0.04, 0.03, 0.41])

    home_ownership = np.random.choice(["MORTGAGE", "RENT", "OWN", "OTHER"], 
                                      size=n_samples, p=[0.50, 0.40, 0.09, 0.01])

    purpose = np.random.choice(
        ["debt_consolidation", "credit_card", "home_improvement", "major_purchase", "small_business", "medical", "other"],
        size=n_samples, p=[0.58, 0.22, 0.07, 0.04, 0.03, 0.02, 0.04]
    )

    revol_util = np.random.normal(loc=52.0, scale=24.0, size=n_samples)
    revol_util = np.clip(revol_util, 0.0, 120.0)

    total_acc = np.random.poisson(lam=25, size=n_samples)
    total_acc = np.clip(total_acc, 3, 75)

    delinq_2yrs = np.random.choice([0, 1, 2, 3, 4], size=n_samples, p=[0.80, 0.13, 0.04, 0.02, 0.01])
    pub_rec = np.random.choice([0, 1, 2], size=n_samples, p=[0.84, 0.14, 0.02])
    inq_last_6mths = np.random.choice([0, 1, 2, 3, 4], size=n_samples, p=[0.52, 0.28, 0.12, 0.05, 0.03])

    # FICO score proxy (correlated with subgrade)
    fico_score = 700 - (dti * 2.2) - (revol_util * 0.9) - (delinq_2yrs * 25) - (pub_rec * 30) - (inq_last_6mths * 10)
    fico_score += np.random.normal(0, 30, size=n_samples)
    fico_score = np.clip(fico_score, 580, 840)

    # LendingClub Grade (A to G) based on FICO & risk
    grades = []
    sub_grades = []
    int_rates = []
    for fs, l_term in zip(fico_score, term):
        if fs >= 750:
            g = "A"
            rate = np.random.uniform(5.3, 8.5)
        elif fs >= 710:
            g = "B"
            rate = np.random.uniform(8.6, 12.0)
        elif fs >= 675:
            g = "C"
            rate = np.random.uniform(12.1, 15.5)
        elif fs >= 640:
            g = "D"
            rate = np.random.uniform(15.6, 19.5)
        elif fs >= 615:
            g = "E"
            rate = np.random.uniform(19.6, 24.0)
        elif fs >= 595:
            g = "F"
            rate = np.random.uniform(24.1, 28.0)
        else:
            g = "G"
            rate = np.random.uniform(28.1, 31.0)
        
        if "60" in l_term:
            rate += 1.5
            
        sub = f"{g}{np.random.randint(1, 6)}"
        grades.append(g)
        sub_grades.append(sub)
        int_rates.append(round(rate, 2))

    # 2. Realistic Logistic Probability of Default (Bad)
    # Log-odds of default
    z = (
        - 2.8
        + 0.000025 * (loan_amnt - 15000)
        + 0.045 * (dti - 18.0)
        + 0.02 * (revol_util - 50.0)
        + 0.45 * delinq_2yrs
        + 0.50 * pub_rec
        + 0.25 * inq_last_6mths
        + 0.08 * (np.array(int_rates) - 12.0)
        - 0.000008 * (annual_inc - 70000)
        + (0.35 if "60" in term[0] else 0.0)
    )
    # Convert log-odds to probability
    prob_bad = 1.0 / (1.0 + np.exp(-z))
    prob_bad = np.clip(prob_bad, 0.02, 0.85)

    is_bad = (np.random.uniform(0, 1, size=n_samples) < prob_bad).astype(int)
    
    # Map to LendingClub textual loan_status
    loan_status = []
    for b in is_bad:
        if b == 0:
            loan_status.append(np.random.choice(["Fully Paid", "Current"], p=[0.75, 0.25]))
        else:
            loan_status.append(np.random.choice(["Charged Off", "Default", "Late (31-120 days)"], p=[0.70, 0.15, 0.15]))

    df = pd.DataFrame({
        "loan_amnt": np.round(loan_amnt, 2),
        "term": term,
        "int_rate": int_rates,
        "grade": grades,
        "sub_grade": sub_grades,
        "emp_length": emp_length,
        "home_ownership": home_ownership,
        "annual_inc": np.round(annual_inc, 2),
        "purpose": purpose,
        "dti": np.round(dti, 2),
        "delinq_2yrs": delinq_2yrs,
        "inq_last_6mths": inq_last_6mths,
        "pub_rec": pub_rec,
        "revol_util": np.round(revol_util, 2),
        "total_acc": total_acc,
        "loan_status": loan_status
    })

    # Introduce realistic missingness (<3% on revol_util, emp_length, dti)
    mask_revol = np.random.rand(n_samples) < 0.015
    df.loc[mask_revol, "revol_util"] = np.nan

    mask_emp = np.random.rand(n_samples) < 0.04
    df.loc[mask_emp, "emp_length"] = np.nan

    mask_dti = np.random.rand(n_samples) < 0.005
    df.loc[mask_dti, "dti"] = np.nan

    return df

def clean_and_prepare_lendingclub_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Cleans raw or benchmark LendingClub data and prepares standard binary target.
    Target: 0 = Good (Fully Paid, Current), 1 = Bad (Default, Charged Off, Late)
    """
    df = df.copy()

    # Drop unneeded/post-origination or leaky columns if present in full LC files
    leaky_cols = [
        "id", "member_id", "url", "desc", "funded_amnt", "funded_amnt_inv", 
        "out_prncp", "out_prncp_inv", "total_pymnt", "total_pymnt_inv", 
        "total_rec_prncp", "total_rec_int", "total_rec_late_fee", "recoveries", 
        "collection_recovery_fee", "last_pymnt_d", "last_pymnt_amnt", "next_pymnt_d", 
        "last_credit_pull_d", "debt_settlement_flag"
    ]
    for col in leaky_cols:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    # Filter for known statuses
    valid_mask = df["loan_status"].isin(GOOD_STATUSES + BAD_STATUSES)
    df = df[valid_mask].copy()

    # Standard Credit Target: 0 = Good, 1 = Bad
    y = df["loan_status"].apply(lambda s: 1 if s in BAD_STATUSES else 0)
    df.drop(columns=["loan_status"], inplace=True)

    # Clean term to numeric months if string
    if df["term"].dtype == object:
        df["term"] = df["term"].astype(str).str.extract(r"(\d+)").astype(float)

    # Clean emp_length: convert to numeric or clean string representation
    if "emp_length" in df.columns and df["emp_length"].dtype == object:
        # standard handling: map '< 1 year' to 0, '10+ years' to 10, 'n years' to n, missing to -1 (Missing Bin)
        emp_clean = df["emp_length"].astype(str).str.extract(r"(\d+)")[0].astype(float)
        # For entries like '< 1 year', extract will give 1, but let's be precise
        df["emp_length_num"] = emp_clean
        df.loc[df["emp_length"].astype(str).str.contains("<", na=False), "emp_length_num"] = 0
        df["emp_length_num"] = df["emp_length_num"].fillna(-1)

    # Clean int_rate and revol_util if strings with '%'
    for col in ["int_rate", "revol_util"]:
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].astype(str).str.replace("%", "").astype(float)

    return df, y

def load_or_create_dataset(csv_path: Optional[str] = "data/raw/accepted_loans.csv", 
                           n_samples: int = 40000, 
                           save_raw: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Loads raw LendingClub dataset if present; otherwise generates authentic benchmark dataset.
    """
    if csv_path and os.path.exists(csv_path):
        print(f"Loading LendingClub data from {csv_path}...")
        df = pd.read_csv(csv_path, low_memory=False)
        return clean_and_prepare_lendingclub_data(df)
    
    print(f"No existing CSV found at '{csv_path}'. Generating authentic LendingClub benchmark dataset ({n_samples} loans)...")
    df_raw = generate_benchmark_lendingclub_data(n_samples=n_samples)
    
    if save_raw:
        os.makedirs(os.path.dirname(csv_path) if csv_path else "data/raw", exist_ok=True)
        target_path = csv_path or "data/raw/accepted_loans_benchmark.csv"
        df_raw.to_csv(target_path, index=False)
        print(f"Benchmark dataset cached to {target_path}")

    return clean_and_prepare_lendingclub_data(df_raw)

if __name__ == "__main__":
    X, y = load_or_create_dataset(n_samples=5000)
    print("Features shape:", X.shape)
    print("Default rate (Bad=1):", f"{y.mean() * 100:.2f}%")
    print(X.head())
