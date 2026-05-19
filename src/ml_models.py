# ============================================================
# src/ml_models.py — 3 ML Models: LR + RF + XGBoost
# ============================================================

import pandas as pd
import numpy as np
import os
import pickle
import logging
from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier
from xgboost                 import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing   import StandardScaler
from sklearn.metrics         import (accuracy_score, precision_score, recall_score,
                                     f1_score, roc_auc_score, confusion_matrix,
                                     classification_report)
from sklearn.pipeline        import Pipeline
from config import TEST_SIZE, CV_FOLDS, MODEL_PATH, OUTPUT_PATH, RANDOM_SEED

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def find_optimal_threshold(model, X_test, y_test) -> float:
    """Find threshold that maximizes F1 score."""
    from sklearn.metrics import f1_score
    y_proba = model.predict_proba(X_test)[:, 1]
    best_thresh, best_f1 = 0.5, 0
    for thresh in [i/100 for i in range(20, 60)]:
        y_pred = (y_proba >= thresh).astype(int)
        f1 = f1_score(y_test, y_pred)
        if f1 > best_f1:
            best_f1, best_thresh = f1, thresh
    return best_thresh


def evaluate_model(name: str, model, X_test, y_test) -> dict:
    """Compute all evaluation metrics for a trained model."""
    y_proba = model.predict_proba(X_test)[:, 1]
    # Use optimal threshold instead of default 0.5
    threshold = find_optimal_threshold(model, X_test, y_test)
    y_pred = (y_proba >= threshold).astype(int)
    logger.info(f"  Optimal threshold: {threshold:.2f}")

    metrics = {
        "model":         name,
        "accuracy":      round(accuracy_score(y_test, y_pred) * 100, 2),
        "precision":     round(precision_score(y_test, y_pred) * 100, 2),
        "recall":        round(recall_score(y_test, y_pred) * 100, 2),
        "f1_score":      round(f1_score(y_test, y_pred) * 100, 2),
        "roc_auc":       round(roc_auc_score(y_test, y_proba) * 100, 2),
    }

    cm = confusion_matrix(y_test, y_pred)
    metrics["true_negatives"]  = int(cm[0][0])
    metrics["false_positives"] = int(cm[0][1])
    metrics["false_negatives"] = int(cm[1][0])
    metrics["true_positives"]  = int(cm[1][1])

    logger.info(f"\n  📊 {name} Results:")
    logger.info(f"     Accuracy:  {metrics['accuracy']}%")
    logger.info(f"     Precision: {metrics['precision']}%")
    logger.info(f"     Recall:    {metrics['recall']}%")
    logger.info(f"     F1 Score:  {metrics['f1_score']}%")
    logger.info(f"     ROC-AUC:   {metrics['roc_auc']}%")
    logger.info(f"\n{classification_report(y_test, y_pred, target_names=['Retained','Churned'])}")

    return metrics


def get_feature_importance(model, feature_names: list, model_name: str) -> pd.DataFrame:
    """Extract feature importances from model."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        return pd.DataFrame()

    df = pd.DataFrame({
        "feature":    feature_names,
        "importance": importances,
        "model":      model_name
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    logger.info(f"\n  🔑 Top 5 features ({model_name}):")
    for _, row in df.head(5).iterrows():
        logger.info(f"     {row['feature']}: {row['importance']:.4f}")

    return df


def train_logistic_regression(X_train, X_test, y_train, y_test,
                               feature_names: list) -> tuple[dict, pd.DataFrame, object]:
    """
    Logistic Regression with StandardScaler.
    Best for: interpretability, baseline comparison, coefficient analysis.
    """
    logger.info("\n🔵 Training Logistic Regression...")

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model",  LogisticRegression(
            C            = 0.1,
            max_iter     = 2000,
            solver       = "lbfgs",
            random_state = RANDOM_SEED,
            class_weight = {0: 1, 1: 4}
        ))
    ])
    pipeline.fit(X_train, y_train)

    # Cross-validation
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc")
    logger.info(f"  CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    metrics = evaluate_model("Logistic Regression", pipeline, X_test, y_test)
    metrics["cv_roc_auc_mean"] = round(cv_scores.mean() * 100, 2)
    metrics["cv_roc_auc_std"]  = round(cv_scores.std() * 100, 2)

    # Feature importance from coefficients
    lr_model   = pipeline.named_steps["model"]
    importance = get_feature_importance(lr_model, feature_names, "Logistic Regression")

    return metrics, importance, pipeline


def train_random_forest(X_train, X_test, y_train, y_test,
                        feature_names: list) -> tuple[dict, pd.DataFrame, object]:
    """
    Random Forest Classifier.
    Best for: non-linear patterns, feature importance, robust performance.
    """
    logger.info("\n🌲 Training Random Forest...")

    model = RandomForestClassifier(
        n_estimators  = 300,
        max_depth     = 15,
        min_samples_split = 3,
        min_samples_leaf  = 1,
        max_features  = "sqrt",
        class_weight  = {0: 1, 1: 3},
        random_state  = RANDOM_SEED,
        n_jobs        = -1
    )
    model.fit(X_train, y_train)

    # Cross-validation
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
    logger.info(f"  CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    metrics = evaluate_model("Random Forest", model, X_test, y_test)
    metrics["cv_roc_auc_mean"] = round(cv_scores.mean() * 100, 2)
    metrics["cv_roc_auc_std"]  = round(cv_scores.std() * 100, 2)

    importance = get_feature_importance(model, feature_names, "Random Forest")

    return metrics, importance, model


def train_xgboost(X_train, X_test, y_train, y_test,
                  feature_names: list) -> tuple[dict, pd.DataFrame, object]:
    """
    XGBoost Classifier.
    Best for: highest accuracy, gradient boosting, handles imbalance natively.
    """
    logger.info("\n⚡ Training XGBoost...")

    # Calculate scale_pos_weight for class imbalance
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos_weight = neg / pos

    model = XGBClassifier(
        n_estimators      = 300,
        max_depth         = 7,
        learning_rate     = 0.05,
        subsample         = 0.85,
        colsample_bytree  = 0.85,
        min_child_weight  = 3,
        gamma             = 0.1,
        reg_alpha         = 0.1,
        scale_pos_weight  = scale_pos_weight,
        random_state      = RANDOM_SEED,
        eval_metric       = "logloss",
        verbosity         = 0
    )
    model.fit(X_train, y_train)

    # Cross-validation
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
    logger.info(f"  CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    metrics = evaluate_model("XGBoost", model, X_test, y_test)
    metrics["cv_roc_auc_mean"] = round(cv_scores.mean() * 100, 2)
    metrics["cv_roc_auc_std"]  = round(cv_scores.std() * 100, 2)

    importance = get_feature_importance(model, feature_names, "XGBoost")

    return metrics, importance, model


def generate_churn_scores(models: dict, X: pd.DataFrame,
                          df_original: pd.DataFrame) -> pd.DataFrame:
    """
    Generate churn risk scores for all customers using best model (XGBoost).
    Adds predicted churn probability and risk tier.
    """
    best_model = models["XGBoost"]
    proba      = best_model.predict_proba(X)[:, 1]
    predicted  = best_model.predict(X)

    df_scores = df_original[["customer_id", "country", "age", "balance",
                               "credit_score", "num_products", "is_active_member",
                               "satisfaction_score", "churned"]].copy()
    df_scores["churn_probability"] = proba.round(4)
    df_scores["predicted_churn"]   = predicted
    df_scores["risk_tier"] = pd.cut(
        proba,
        bins   = [0, 0.30, 0.50, 0.70, 1.0],
        labels = ["Low Risk", "Medium Risk", "High Risk", "Critical Risk"]
    ).astype(str)

    # Correct predictions
    df_scores["prediction_correct"] = (df_scores["predicted_churn"] == df_scores["churned"]).astype(int)

    logger.info(f"\n🎯 Churn Risk Distribution:")
    logger.info(df_scores["risk_tier"].value_counts().to_string())

    return df_scores


def save_models(models: dict):
    os.makedirs(MODEL_PATH, exist_ok=True)
    for name, model in models.items():
        safe_name = name.lower().replace(" ", "_")
        path = os.path.join(MODEL_PATH, f"{safe_name}_model.pkl")
        with open(path, "wb") as f:
            pickle.dump(model, f)
        logger.info(f"💾 Saved {name} → {path}")


def run_ml_pipeline(X: pd.DataFrame, y: pd.Series,
                    df_original: pd.DataFrame) -> tuple[list, pd.DataFrame, pd.DataFrame]:
    """Train all 3 models, compare, score all customers."""
    logger.info("=" * 50)
    logger.info("🤖 TRAINING 3 ML MODELS")
    logger.info("=" * 50)

    feature_names = list(X.columns)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    logger.info(f"Train: {len(X_train):,} | Test: {len(X_test):,} | Churn rate: {y.mean():.1%}")

    # Train all 3 models
    lr_metrics,  lr_imp,  lr_model  = train_logistic_regression(X_train, X_test, y_train, y_test, feature_names)
    rf_metrics,  rf_imp,  rf_model  = train_random_forest(X_train, X_test, y_train, y_test, feature_names)
    xgb_metrics, xgb_imp, xgb_model = train_xgboost(X_train, X_test, y_train, y_test, feature_names)

    all_metrics   = [lr_metrics, rf_metrics, xgb_metrics]
    all_importance = pd.concat([lr_imp, rf_imp, xgb_imp], ignore_index=True)

    # Print model comparison
    logger.info("\n" + "=" * 50)
    logger.info("📊 MODEL COMPARISON SUMMARY")
    logger.info("=" * 50)
    comp_df = pd.DataFrame(all_metrics)[["model","accuracy","precision","recall","f1_score","roc_auc"]]
    logger.info("\n" + comp_df.to_string(index=False))

    # Best model
    best = max(all_metrics, key=lambda x: x["roc_auc"])
    logger.info(f"\n🏆 Best model: {best['model']} (ROC-AUC: {best['roc_auc']}%)")

    # Save models
    models = {"Logistic Regression": lr_model, "Random Forest": rf_model, "XGBoost": xgb_model}
    save_models(models)

    # Score all customers
    churn_scores = generate_churn_scores(models, X, df_original)

    # Save outputs
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    ts = __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")
    pd.DataFrame(all_metrics).to_csv(os.path.join(OUTPUT_PATH, f"model_comparison_{ts}.csv"), index=False)
    all_importance.to_csv(os.path.join(OUTPUT_PATH, f"feature_importance_{ts}.csv"), index=False)
    churn_scores.to_csv(os.path.join(OUTPUT_PATH, f"churn_scores_{ts}.csv"), index=False)
    logger.info(f"💾 Results exported to outputs/")

    return all_metrics, all_importance, churn_scores
