# ============================================================
# src/db_loader.py — Load churn data into SQLite
# ============================================================

import sqlite3
import pandas as pd
import logging
import os
from datetime import datetime
from config import DB_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    return sqlite3.connect(DB_PATH)


def create_tables():
    conn = get_connection()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS customers (
            customer_id         TEXT PRIMARY KEY,
            age                 INTEGER,
            gender              TEXT,
            country             TEXT,
            credit_score        INTEGER,
            tenure              INTEGER,
            balance             REAL,
            num_products        INTEGER,
            has_credit_card     INTEGER,
            is_active_member    INTEGER,
            estimated_salary    REAL,
            num_transactions    INTEGER,
            avg_transaction_amt REAL,
            months_inactive     INTEGER,
            num_contacts        INTEGER,
            satisfaction_score  INTEGER,
            contract_type       TEXT,
            balance_per_product REAL,
            has_zero_balance    INTEGER,
            transaction_rate    REAL,
            high_risk_flag      INTEGER,
            churned             INTEGER
        );

        CREATE TABLE IF NOT EXISTS churn_predictions (
            prediction_id       INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id         TEXT,
            churn_probability   REAL,
            predicted_churn     INTEGER,
            risk_tier           TEXT,
            prediction_correct  INTEGER,
            model_used          TEXT DEFAULT 'XGBoost',
            predicted_at        TEXT
        );

        CREATE TABLE IF NOT EXISTS model_performance (
            perf_id             INTEGER PRIMARY KEY AUTOINCREMENT,
            model               TEXT,
            accuracy            REAL,
            precision_score     REAL,
            recall_score        REAL,
            f1_score            REAL,
            roc_auc             REAL,
            cv_roc_auc_mean     REAL,
            cv_roc_auc_std      REAL,
            true_positives      INTEGER,
            true_negatives      INTEGER,
            false_positives     INTEGER,
            false_negatives     INTEGER,
            evaluated_at        TEXT
        );

        CREATE TABLE IF NOT EXISTS feature_importance (
            importance_id   INTEGER PRIMARY KEY AUTOINCREMENT,
            model           TEXT,
            feature         TEXT,
            importance      REAL,
            importance_rank INTEGER
        );

        CREATE TABLE IF NOT EXISTS pipeline_run_log (
            run_id            INTEGER PRIMARY KEY AUTOINCREMENT,
            run_timestamp     TEXT,
            stage             TEXT,
            status            TEXT,
            records_processed INTEGER DEFAULT 0,
            error_message     TEXT,
            duration_sec      REAL
        );
    """)
    conn.commit()
    conn.close()
    logger.info("✅ Database schema created/verified")


def load_customers(df: pd.DataFrame) -> int:
    conn = get_connection()
    cols = ["customer_id","age","gender","country","credit_score","tenure","balance",
            "num_products","has_credit_card","is_active_member","estimated_salary",
            "num_transactions","avg_transaction_amt","months_inactive","num_contacts",
            "satisfaction_score","contract_type","balance_per_product","has_zero_balance",
            "transaction_rate","high_risk_flag","churned"]
    inserted = 0
    for _, r in df[cols].iterrows():
        try:
            conn.execute(f"INSERT OR IGNORE INTO customers VALUES ({','.join(['?']*len(cols))})",
                         tuple(r))
            inserted += 1
        except: pass
    conn.commit()
    conn.close()
    logger.info(f"✅ Inserted {inserted:,} customers")
    return inserted


def load_predictions(churn_scores: pd.DataFrame):
    conn = get_connection()
    now  = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    for _, r in churn_scores.iterrows():
        conn.execute("""
            INSERT INTO churn_predictions
            (customer_id, churn_probability, predicted_churn, risk_tier, prediction_correct, predicted_at)
            VALUES (?,?,?,?,?,?)
        """, (r.customer_id, float(r.churn_probability), int(r.predicted_churn),
              r.risk_tier, int(r.prediction_correct), now))
    conn.commit()
    conn.close()
    logger.info(f"✅ Inserted {len(churn_scores):,} predictions")


def load_model_performance(metrics_list: list):
    conn = get_connection()
    now  = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    for m in metrics_list:
        conn.execute("""
            INSERT INTO model_performance
            (model, accuracy, precision_score, recall_score, f1_score, roc_auc,
             cv_roc_auc_mean, cv_roc_auc_std, true_positives, true_negatives,
             false_positives, false_negatives, evaluated_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (m["model"], m["accuracy"], m["precision"], m["recall"],
              m["f1_score"], m["roc_auc"], m.get("cv_roc_auc_mean"),
              m.get("cv_roc_auc_std"), m["true_positives"], m["true_negatives"],
              m["false_positives"], m["false_negatives"], now))
    conn.commit()
    conn.close()
    logger.info(f"✅ Inserted {len(metrics_list)} model performance records")


def load_feature_importance(importance_df: pd.DataFrame):
    conn = get_connection()
    for model_name in importance_df["model"].unique():
        sub = importance_df[importance_df["model"] == model_name].reset_index(drop=True)
        for rank, (_, row) in enumerate(sub.iterrows(), 1):
            conn.execute("""
                INSERT INTO feature_importance (model, feature, importance, importance_rank)
                VALUES (?,?,?,?)
            """, (row["model"], row["feature"], float(row["importance"]), rank))
    conn.commit()
    conn.close()
    logger.info(f"✅ Feature importance loaded")


def log_run(stage, status, records=0, error=None, duration=None):
    conn = get_connection()
    conn.execute("INSERT INTO pipeline_run_log VALUES (?,?,?,?,?,?,?)",
        (None, datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
         stage, status, records, error, duration))
    conn.commit()
    conn.close()


def query_summary() -> pd.DataFrame:
    conn = get_connection()
    df = pd.read_sql_query("""
        SELECT
            c.country,
            COUNT(c.customer_id)                                AS total_customers,
            SUM(c.churned)                                      AS actual_churned,
            ROUND(AVG(c.churned)*100, 1)                        AS actual_churn_pct,
            ROUND(AVG(p.churn_probability)*100, 1)              AS avg_churn_prob_pct,
            SUM(CASE WHEN p.risk_tier='Critical Risk' THEN 1 ELSE 0 END) AS critical_risk_count,
            ROUND(AVG(c.credit_score), 0)                       AS avg_credit_score,
            ROUND(AVG(c.balance), 2)                            AS avg_balance
        FROM customers c
        LEFT JOIN churn_predictions p ON c.customer_id = p.customer_id
        GROUP BY c.country
        ORDER BY actual_churn_pct DESC
    """, conn)
    conn.close()
    return df


def run_db_load(df_feat, churn_scores, metrics_list, importance_df):
    import time
    t = time.time()
    try:
        create_tables()
        load_customers(df_feat)
        load_predictions(churn_scores)
        load_model_performance(metrics_list)
        load_feature_importance(importance_df)
        duration = round(time.time() - t, 2)
        log_run("db_load", "SUCCESS", len(df_feat), duration=duration)
        logger.info(f"✅ DB load complete in {duration}s")
    except Exception as e:
        log_run("db_load", "FAILED", error=str(e))
        logger.error(f"❌ DB load failed: {e}")
        raise
