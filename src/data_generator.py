# ============================================================
# src/data_generator.py — Simulate bank customer churn data
# 7,000 customers with realistic churn patterns
# ============================================================

import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime
from config import (NUM_CUSTOMERS, CHURN_RATE, AGE_RANGE, TENURE_RANGE,
                    PRODUCTS, COUNTRIES, RAW_DATA_PATH, RANDOM_SEED)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
np.random.seed(RANDOM_SEED)


def generate_customers() -> pd.DataFrame:
    """
    Generate realistic bank customer data with churn labels.
    Churn probability is influenced by:
    - Low balance, low credit score → higher churn
    - Inactive members → higher churn
    - Germany customers → slightly higher churn
    - Single product holders → higher churn
    - Age 30-50 → lower churn (more settled)
    """
    countries   = list(COUNTRIES.keys())
    country_probs = list(COUNTRIES.values())

    records = []
    for i in range(1, NUM_CUSTOMERS + 1):

        # Demographics
        age        = np.random.randint(*AGE_RANGE)
        country    = np.random.choice(countries, p=country_probs)
        gender     = np.random.choice(["Male", "Female"])
        tenure     = np.random.randint(*TENURE_RANGE)

        # Financial profile
        credit_score = int(np.random.normal(650, 80))
        credit_score = np.clip(credit_score, 350, 850)

        balance = np.random.choice(
            [0, np.random.uniform(1000, 250000)],
            p=[0.25, 0.75]
        )
        balance = round(float(balance), 2)

        num_products   = np.random.choice([1, 2, 3, 4], p=[0.50, 0.35, 0.10, 0.05])
        has_credit_card = np.random.choice([0, 1], p=[0.30, 0.70])
        is_active       = np.random.choice([0, 1], p=[0.35, 0.65])
        estimated_salary = round(np.random.uniform(20000, 150000), 2)

        # Transaction behavior
        num_transactions    = np.random.poisson(12 * tenure)
        avg_transaction_amt = round(np.random.uniform(50, 2000), 2)
        months_inactive     = np.random.choice([0, 1, 2, 3, 4, 5, 6], p=[0.30, 0.25, 0.18, 0.12, 0.08, 0.05, 0.02])
        num_contacts        = np.random.choice([0, 1, 2, 3, 4, 5, 6], p=[0.35, 0.28, 0.18, 0.10, 0.05, 0.03, 0.01])

        # Satisfaction score (1-5)
        satisfaction = np.random.choice([1, 2, 3, 4, 5], p=[0.10, 0.15, 0.25, 0.30, 0.20])

        # ── Churn probability model ──────────────────────────
        churn_score = 0.0

        # Strong negative baseline — most customers retained (~21% churn target)
        churn_score = -2.5

        # Credit score — strong signal
        if credit_score < 450:    churn_score += 1.20
        elif credit_score < 550:  churn_score += 0.70
        elif credit_score < 650:  churn_score += 0.25
        elif credit_score > 750:  churn_score -= 0.50

        # Balance — strong signal
        if balance == 0:          churn_score += 1.00
        elif balance < 5000:      churn_score += 0.40
        elif balance > 100000:    churn_score -= 0.40

        # Active member — very strong signal
        if is_active == 0:        churn_score += 1.50
        else:                     churn_score -= 0.60

        # Num products — very strong signal
        if num_products == 1:     churn_score += 0.80
        elif num_products == 2:   churn_score -= 0.30
        elif num_products >= 3:   churn_score -= 0.90

        # Satisfaction — strongest signal
        if satisfaction == 1:     churn_score += 2.00
        elif satisfaction == 2:   churn_score += 1.20
        elif satisfaction == 3:   churn_score += 0.10
        elif satisfaction == 4:   churn_score -= 0.60
        elif satisfaction == 5:   churn_score -= 1.20

        # Months inactive
        if months_inactive >= 5:  churn_score += 1.00
        elif months_inactive >= 3: churn_score += 0.50
        elif months_inactive == 0: churn_score -= 0.50

        # Contacts — dissatisfied customers call more
        if num_contacts >= 5:     churn_score += 0.80
        elif num_contacts >= 3:   churn_score += 0.30

        # Tenure
        if tenure <= 1:           churn_score += 0.50
        elif tenure >= 8:         churn_score -= 0.60

        # Germany
        if country == "Germany":  churn_score += 0.30

        # Age
        if 35 <= age <= 55:       churn_score -= 0.30
        elif age < 25:            churn_score += 0.30

        # Convert to probability with steeper sigmoid
        churn_prob = 1 / (1 + np.exp(-churn_score * 1.6))
        churn_prob = np.clip(churn_prob, 0.01, 0.99)
        churned    = int(np.random.random() < churn_prob)

        records.append({
            "customer_id":         f"CUST_{i:05d}",
            "age":                 age,
            "gender":              gender,
            "country":             country,
            "credit_score":        int(credit_score),
            "tenure":              tenure,
            "balance":             balance,
            "num_products":        num_products,
            "has_credit_card":     has_credit_card,
            "is_active_member":    is_active,
            "estimated_salary":    estimated_salary,
            "num_transactions":    num_transactions,
            "avg_transaction_amt": avg_transaction_amt,
            "months_inactive":     months_inactive,
            "num_contacts":        num_contacts,
            "satisfaction_score":  satisfaction,
            "churned":             churned
        })

    df = pd.DataFrame(records)
    actual_churn_rate = df["churned"].mean()
    logger.info(f"✅ Generated {len(df):,} customers | Churn rate: {actual_churn_rate:.1%}")
    return df


def save_raw_data(df: pd.DataFrame) -> str:
    os.makedirs(RAW_DATA_PATH, exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(RAW_DATA_PATH, f"customers_{ts}.csv")
    df.to_csv(path, index=False)
    logger.info(f"💾 Raw data saved → {path}")
    return path


def run_data_generation() -> tuple[pd.DataFrame, str]:
    logger.info("🏦 Generating bank customer churn data...")
    df = generate_customers()
    path = save_raw_data(df)
    return df, path


if __name__ == "__main__":
    df, _ = run_data_generation()
    print(f"\nShape: {df.shape}")
    print(f"Churn rate: {df['churned'].mean():.1%}")
    print(f"\nChurn by country:\n{df.groupby('country')['churned'].mean().round(3)}")
    print(f"\nChurn by num_products:\n{df.groupby('num_products')['churned'].mean().round(3)}")
