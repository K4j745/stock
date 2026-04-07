import logging

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger("stock_ml")


def evaluate_model(y_true, y_pred, y_prob=None) -> dict:
    """Compute classification metrics.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        y_prob: Predicted probabilities for the positive class (optional).

    Returns:
        Dictionary with accuracy, precision, recall, f1, roc_auc, mcc.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }

    if y_prob is not None and len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
    else:
        metrics["roc_auc"] = np.nan

    return metrics


def print_classification_report(results: dict) -> None:
    """Log formatted classification results for all models."""
    for model_name, model_results in results.items():
        logger.info("=" * 60)
        logger.info("Model: %s", model_name)
        logger.info("-" * 60)

        for i, fold_metrics in enumerate(model_results["fold_metrics"]):
            logger.info(
                "  Fold %d: acc=%.4f prec=%.4f rec=%.4f f1=%.4f auc=%.4f mcc=%.4f",
                i + 1,
                fold_metrics["accuracy"],
                fold_metrics["precision"],
                fold_metrics["recall"],
                fold_metrics["f1"],
                fold_metrics.get("roc_auc", float("nan")),
                fold_metrics["mcc"],
            )

        mean = model_results["mean_metrics"]
        logger.info(
            "  MEAN:   acc=%.4f prec=%.4f rec=%.4f f1=%.4f auc=%.4f mcc=%.4f",
            mean["accuracy"],
            mean["precision"],
            mean["recall"],
            mean["f1"],
            mean.get("roc_auc", float("nan")),
            mean["mcc"],
        )
        logger.info("=" * 60)


def evaluate_saved_models(ticker: str, label_version: str = "A"):
    """
    Load saved models, run evaluation on test split (last 20%), print metrics.
    """
    import os
    import joblib
    import pandas as pd
    from features.pipeline import build_feature_matrix
    from config import MODEL_DIR

    X, y = build_feature_matrix(ticker, label_version)
    split_idx = int(len(X) * 0.8)
    X_test = X.iloc[split_idx:]
    y_test = y.iloc[split_idx:]

    scaler_path = os.path.join(MODEL_DIR, f"scaler_{ticker}_{label_version}.joblib")
    if not os.path.exists(scaler_path):
        logger.error(f"Scaler not found: {scaler_path}")
        return
    scaler = joblib.load(scaler_path)
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )

    model_configs = [
        ("logistic_regression", "logistic_regression", ".joblib"),
        ("random_forest", "random_forest", ".joblib"),
        ("xgboost", "xgboost", ".json"),
        ("lightgbm", "lightgbm", ".txt"),
    ]

    all_results = {}
    for model_name, file_prefix, ext in model_configs:
        model_path = os.path.join(MODEL_DIR, f"{file_prefix}_{ticker}_{label_version}{ext}")
        if not os.path.exists(model_path):
            logger.warning(f"Model not found: {model_path}, skipping")
            continue

        if model_name == "xgboost":
            import xgboost as xgb
            model = xgb.XGBClassifier()
            model.load_model(model_path)
            preds = model.predict(X_test_scaled)
            proba = model.predict_proba(X_test_scaled)[:, 1]
        elif model_name == "lightgbm":
            import lightgbm as lgb
            booster = lgb.Booster(model_file=model_path)
            proba = booster.predict(X_test_scaled)
            preds = (proba > 0.5).astype(int)
        else:
            model = joblib.load(model_path)
            preds = model.predict(X_test_scaled)
            proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else None

        metrics = evaluate_model(y_test, preds, proba)
        all_results[model_name] = metrics
        logger.info(f"{model_name:25s} | acc={metrics['accuracy']:.4f} | f1={metrics['f1']:.4f} | auc={metrics.get('roc_auc', 0):.4f} | mcc={metrics['mcc']:.4f}")

    return all_results
