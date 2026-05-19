# ============================================================
# run_pipeline.py — Customer Churn Prediction Pipeline
# Stages: Generate >> EDA+Features >> ML Training >> DB Load
# Usage: py -3.11 run_pipeline.py
# ============================================================

import time, logging, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_pipeline():
    start  = time.time()
    run_id = __import__("datetime").datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    ts     = run_id

    logger.info("=" * 60)
    logger.info(f"🏦 CHURN PIPELINE STARTED  |  run_id: {run_id}")
    logger.info("=" * 60)

    # STAGE 1 — DATA GENERATION
    logger.info("\n📊 STAGE 1: Customer Data Generation")
    logger.info("-" * 40)
    t = time.time()
    try:
        from src.data_generator import run_data_generation
        df_raw, _ = run_data_generation()
        logger.info(f"✅ Stage 1 complete in {round(time.time()-t,2)}s | {len(df_raw):,} customers")
    except Exception as e:
        logger.error(f"❌ Stage 1 FAILED: {e}"); return False

    # STAGE 2 — EDA + FEATURE ENGINEERING
    logger.info("\n🔍 STAGE 2: EDA & Feature Engineering")
    logger.info("-" * 40)
    t = time.time()
    try:
        from src.eda_features import run_eda_and_features, prepare_ml_features
        df_feat, insights = run_eda_and_features(df_raw, ts)
        X, y = prepare_ml_features(df_feat)
        logger.info(f"✅ Stage 2 complete in {round(time.time()-t,2)}s | {X.shape[1]} features")
    except Exception as e:
        logger.error(f"❌ Stage 2 FAILED: {e}"); return False

    # STAGE 3 — ML TRAINING
    logger.info("\n🤖 STAGE 3: ML Model Training (LR + RF + XGBoost)")
    logger.info("-" * 40)
    t = time.time()
    try:
        from src.ml_models import run_ml_pipeline
        metrics_list, importance_df, churn_scores = run_ml_pipeline(X, y, df_feat)
        logger.info(f"✅ Stage 3 complete in {round(time.time()-t,2)}s")
    except Exception as e:
        logger.error(f"❌ Stage 3 FAILED: {e}"); return False

    # STAGE 4 — DATABASE LOAD
    logger.info("\n🗄️  STAGE 4: Database Load")
    logger.info("-" * 40)
    t = time.time()
    try:
        from src.db_loader import run_db_load, query_summary
        run_db_load(df_feat, churn_scores, metrics_list, importance_df)
        logger.info(f"✅ Stage 4 complete in {round(time.time()-t,2)}s")
    except Exception as e:
        logger.error(f"❌ Stage 4 FAILED: {e}"); return False

    # STAGE 5 — VALIDATION
    logger.info("\n✅ STAGE 5: Validation Summary")
    logger.info("-" * 40)
    try:
        from src.db_loader import query_summary
        print("\n" + query_summary().to_string())

        # Best model
        best = max(metrics_list, key=lambda x: x["roc_auc"])
        logger.info(f"\n📊 FINAL RESULTS:")
        logger.info(f"   Customers analyzed:  {len(df_feat):,}")
        logger.info(f"   Actual churn rate:   {df_feat['churned'].mean():.1%}")
        logger.info(f"   Best model:          {best['model']}")
        logger.info(f"   Best ROC-AUC:        {best['roc_auc']}%")
        logger.info(f"   Best Accuracy:       {best['accuracy']}%")
        logger.info(f"   Best F1 Score:       {best['f1_score']}%")
        logger.info(f"   Critical risk:       {(churn_scores['risk_tier']=='Critical Risk').sum()} customers")
        logger.info(f"   High risk:           {(churn_scores['risk_tier']=='High Risk').sum()} customers")

    except Exception as e:
        logger.warning(f"⚠️  Validation warning: {e}")

    total = round(time.time() - start, 2)
    logger.info("\n" + "=" * 60)
    logger.info(f"🎉 PIPELINE COMPLETE  |  Total time: {total}s")
    logger.info("=" * 60)
    return True


if __name__ == "__main__":
    run_pipeline()
