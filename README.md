# Customer Churn Prediction — Banking

> End-to-end ML pipeline predicting bank customer churn using 3 models (Logistic Regression, Random Forest, XGBoost) on 7,000+ customers with 20 features — achieving 85%+ accuracy and ROC-AUC, with churn risk scoring and Tableau-ready exports.

---

## Project Overview

This project builds a complete churn prediction system for a banking institution. It simulates 7,000 bank customers with realistic behavioral and financial profiles, engineers 20 features, trains and compares 3 ML models, and scores every customer with a churn risk tier (Low / Medium / High / Critical).

---

## Architecture

```
Data Generation (7,000 bank customers)
        |
        v
[ Stage 1: Generate     ]  src/data_generator.py  → realistic churn patterns
        |
        v
[ Stage 2: EDA+Features ]  src/eda_features.py    → 20 engineered features
        |
        v
[ Stage 3: ML Training  ]  src/ml_models.py       → LR + RF + XGBoost
        |
        v
[ Stage 4: DB Load      ]  src/db_loader.py       → SQLite with predictions
        |
        v
[ Stage 5: Validate     ]  Query summary          → Tableau CSV exports
```

---

## ML Model Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 85%+ | - | - | - | 85%+ |
| Random Forest | 85%+ | - | - | - | 87%+ |
| XGBoost | 86%+ | - | - | - | 88%+ |

**Top churn drivers:** satisfaction_score, is_active_member, high_risk_flag, balance, months_inactive

---

## Key Business Insights

- Inactive members churn at 3x the rate of active members
- Customers with 1 product churn at 2x the rate of multi-product holders
- Germany customers show 10% higher churn tendency than other regions
- Low satisfaction (score ≤ 2) is the single strongest churn predictor
- Critical risk customers identified for immediate retention intervention

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| ML | Scikit-learn 1.8, XGBoost 3.2 |
| Data Processing | Pandas 2.2, NumPy 1.26 |
| Database | SQLite → PostgreSQL upgrade path |
| Visualization | CSV export → Tableau / Power BI |
| Version Control | Git / GitHub |

---

## Setup & Run

```bash
git clone https://github.com/shreyayeruvareddy/customer-churn-prediction.git
cd customer-churn-prediction
py -3.11 -m pip install -r requirements.txt
py -3.11 run_pipeline.py
```

---

## Author

**Yeruva Bala Shreya Reddy**
M.S. Computer Science (Data Science) — UNC Charlotte
[GitHub](https://github.com/shreyayeruvareddy) | [Email](mailto:yeruvabalashreyareddy@gmail.com)
