# ============================================================
# src/eda_features.py — EDA + Feature Engineering
# ============================================================

import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime
from config import PROCESSED_DATA_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_eda(df: pd.DataFrame) -> dict:
    """Exploratory Data Analysis — key churn driver statistics."""
    insights = {}

    # Overall churn rate
    insights["overall_churn_rate"] = round(df["churned"].mean() * 100, 2)

    # Churn by country
    insights["churn_by_country"] = df.groupby("country")["churned"].mean().round(4).to_dict()

    # Churn by num_products
    insights["churn_by_products"] = df.groupby("num_products")["churned"].mean().round(4).to_dict()

    # Churn by activity
    insights["churn_active"]   = round(df[df["is_active_member"]==1]["churned"].mean() * 100, 2)
    insights["churn_inactive"] = round(df[df["is_active_member"]==0]["churned"].mean() * 100, 2)

    # Churn by satisfaction
    insights["churn_by_satisfaction"] = df.groupby("satisfaction_score")["churned"].mean().round(4).to_dict()

    # Balance: churned vs retained
    insights["avg_balance_churned"]   = round(df[df["churned"]==1]["balance"].mean(), 2)
    insights["avg_balance_retained"]  = round(df[df["churned"]==0]["balance"].mean(), 2)

    # Credit score: churned vs retained
    insights["avg_credit_churned"]    = round(df[df["churned"]==1]["credit_score"].mean(), 2)
    insights["avg_credit_retained"]   = round(df[df["churned"]==0]["credit_score"].mean(), 2)

    logger.info(f"📊 EDA complete:")
    logger.info(f"   Overall churn rate:     {insights['overall_churn_rate']}%")
    logger.info(f"   Active member churn:    {insights['churn_active']}%")
    logger.info(f"   Inactive member churn:  {insights['churn_inactive']}%")
    logger.info(f"   Avg balance churned:    ${insights['avg_balance_churned']:,.2f}")
    logger.info(f"   Avg balance retained:   ${insights['avg_balance_retained']:,.2f}")

    return insights


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new features to improve model performance:
    - Balance per product (financial engagement ratio)
    - Credit score tier (categorical)
    - Age group
    - High risk flag (combines multiple risk factors)
    - Tenure group
    - Transaction rate (transactions per year of tenure)
    """
    df = df.copy()

    # Balance per product — measures financial engagement
    df["balance_per_product"] = (df["balance"] / df["num_products"]).round(2)

    # Has zero balance flag
    df["has_zero_balance"] = (df["balance"] == 0).astype(int)

    # Credit score tiers
    df["credit_tier"] = pd.cut(
        df["credit_score"],
        bins   = [0, 499, 579, 669, 739, 799, 850],
        labels = ["Very Poor", "Poor", "Fair", "Good", "Very Good", "Exceptional"]
    ).astype(str)

    # Age groups
    df["age_group"] = pd.cut(
        df["age"],
        bins   = [0, 25, 35, 50, 65, 100],
        labels = ["18-25", "26-35", "36-50", "51-65", "65+"]
    ).astype(str)

    # Tenure groups
    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins   = [0, 2, 5, 10],
        labels = ["New (1-2yr)", "Mid (3-5yr)", "Long (6-10yr)"]
    ).astype(str)

    # Transaction rate (per tenure year)
    df["transaction_rate"] = (df["num_transactions"] / df["tenure"].clip(1)).round(2)

    # Salary to balance ratio
    df["salary_balance_ratio"] = (df["balance"] / df["estimated_salary"].clip(1)).round(4)

    # High risk composite flag
    df["high_risk_flag"] = (
        (df["is_active_member"] == 0).astype(int) +
        (df["satisfaction_score"] <= 2).astype(int) +
        (df["months_inactive"] >= 4).astype(int) +
        (df["num_products"] == 1).astype(int) +
        (df["credit_score"] < 550).astype(int)
    )

    # Encode categoricals for ML
    df["gender_enc"]   = (df["gender"] == "Male").astype(int)
    df["germany_enc"]  = (df["country"] == "Germany").astype(int)
    df["france_enc"]   = (df["country"] == "France").astype(int)

    logger.info(f"✅ Feature engineering: {df.shape[1]} total columns")
    return df


def prepare_ml_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Select and return feature matrix X and target y."""
    feature_cols = [
        "credit_score", "age", "tenure", "balance", "num_products",
        "has_credit_card", "is_active_member", "estimated_salary",
        "num_transactions", "avg_transaction_amt", "months_inactive",
        "num_contacts", "satisfaction_score",
        # Engineered features
        "balance_per_product", "has_zero_balance", "transaction_rate",
        "salary_balance_ratio", "high_risk_flag",
        # Encoded categoricals
        "gender_enc", "germany_enc", "france_enc"
    ]
    X = df[feature_cols].fillna(0)
    y = df["churned"]
    logger.info(f"✅ ML features prepared: {X.shape[1]} features, {len(y):,} samples")
    return X, y


def save_processed(df: pd.DataFrame, ts: str) -> str:
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
    path = os.path.join(PROCESSED_DATA_PATH, f"customers_processed_{ts}.csv")
    df.to_csv(path, index=False)
    logger.info(f"💾 Processed data → {path}")
    return path


def run_eda_and_features(df: pd.DataFrame, ts: str) -> tuple[pd.DataFrame, dict]:
    insights = run_eda(df)
    df_feat  = engineer_features(df)
    save_processed(df_feat, ts)
    return df_feat, insights


if __name__ == "__main__":
    import glob
    from config import RAW_DATA_PATH
    files = sorted(glob.glob(os.path.join(RAW_DATA_PATH, "customers_*.csv")))
    if files:
        df = pd.read_csv(files[-1])
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        df_feat, insights = run_eda_and_features(df, ts)
        print(f"\nFeature engineered shape: {df_feat.shape}")
        print(f"High risk customers: {(df_feat['high_risk_flag'] >= 3).sum()}")
