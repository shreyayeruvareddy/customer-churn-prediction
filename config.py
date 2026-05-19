# ============================================================
# config.py — Customer Churn Prediction (Banking)
# ============================================================

RANDOM_SEED = 42

# Data Simulation
NUM_CUSTOMERS    = 7000
CHURN_RATE       = 0.21   # ~21% churn rate (realistic for banking)

# Customer Demographics
AGE_RANGE        = (18, 75)
TENURE_RANGE     = (1, 10)   # years with bank

# Product Types
PRODUCTS         = ["Checking", "Savings", "Credit Card", "Loan", "Investment"]

# Geography
COUNTRIES        = {
    "Germany":  0.35,
    "France":   0.40,
    "Spain":    0.25
}

# Model Settings
TEST_SIZE        = 0.20
CV_FOLDS         = 5

# Paths
RAW_DATA_PATH       = "data/raw"
PROCESSED_DATA_PATH = "data/processed"
MODEL_PATH          = "models"
OUTPUT_PATH         = "outputs"
DB_PATH             = "data/churn_pipeline.db"
