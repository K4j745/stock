import logging
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from config import RANDOM_STATE, N_SPLITS, MODEL_DIR
from features.pipeline import build_feature_matrix
from models.evaluate import evaluate_model, print_classification_report
from reports.generate import export_results

logger = logging.getLogger("stock_ml")


def _get_models() -> dict:
    """Return dictionary of model name -> model instance."""
    return {
        "logistic_regression": LogisticRegression(
            random_state=RANDOM_STATE, max_iter=1000
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE
        ),
        "xgboost": XGBClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            use_label_encoder=False,
        ),
        "lightgbm": LGBMClassifier(
            n_estimators=200, random_state=RANDOM_STATE, verbose=-1
        ),
    }


def train_all_models(ticker: str, label_version: str = "A", refresh: bool = False) -> dict:
    """Train all models for a given ticker and label version using expanding window CV.

    Steps:
        1. Build feature matrix
        2. TimeSeriesSplit cross-validation (no data leakage)
        3. Evaluate per fold
        4. Retrain on full data and save model + scaler

    Returns:
        Dictionary: {model_name: {fold_metrics: [...], mean_metrics: {...}}}
    """
    logger.info("Starting training pipeline for %s (label version %s)", ticker, label_version)

    # 1. Build feature matrix
    X, y = build_feature_matrix(ticker, label_version, refresh=refresh)

    # 2. Setup TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    models = _get_models()
    results = {}

    # 3. Cross-validation for each model
    for model_name, model in models.items():
        logger.info("Training %s on %s...", model_name, ticker)
        fold_metrics = []

        for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # Fit scaler ONLY on train fold (no data leakage)
            scaler = StandardScaler()
            X_train_scaled = pd.DataFrame(
                scaler.fit_transform(X_train),
                columns=X_train.columns, index=X_train.index,
            )
            X_test_scaled = pd.DataFrame(
                scaler.transform(X_test),
                columns=X_test.columns, index=X_test.index,
            )

            # Train model
            model_clone = _clone_model(model_name)
            model_clone.fit(X_train_scaled, y_train)

            # Predict
            y_pred = model_clone.predict(X_test_scaled)
            y_prob = None
            if hasattr(model_clone, "predict_proba"):
                y_prob = model_clone.predict_proba(X_test_scaled)[:, 1]

            # Evaluate
            metrics = evaluate_model(y_test, y_pred, y_prob)
            fold_metrics.append(metrics)

            logger.info(
                "  %s fold %d/%d: acc=%.4f f1=%.4f auc=%.4f mcc=%.4f",
                model_name, fold_idx + 1, N_SPLITS,
                metrics["accuracy"], metrics["f1"],
                metrics.get("roc_auc", float("nan")),
                metrics["mcc"],
            )

        # Compute mean metrics
        mean_metrics = {}
        for key in fold_metrics[0]:
            values = [m[key] for m in fold_metrics if not np.isnan(m[key])]
            mean_metrics[key] = np.mean(values) if values else np.nan

        results[model_name] = {
            "fold_metrics": fold_metrics,
            "mean_metrics": mean_metrics,
        }

        logger.info(
            "  %s MEAN: acc=%.4f f1=%.4f auc=%.4f mcc=%.4f",
            model_name,
            mean_metrics["accuracy"], mean_metrics["f1"],
            mean_metrics.get("roc_auc", float("nan")),
            mean_metrics["mcc"],
        )

    # 4. Retrain on full data and save
    logger.info("Retraining all models on full dataset for %s...", ticker)
    final_scaler = StandardScaler()
    X_scaled_full = pd.DataFrame(
        final_scaler.fit_transform(X), columns=X.columns, index=X.index,
    )

    # Save the scaler
    scaler_path = os.path.join(MODEL_DIR, f"scaler_{ticker}_{label_version}.joblib")
    joblib.dump(final_scaler, scaler_path)
    logger.info("Saved scaler to %s", scaler_path)

    for model_name in models:
        final_model = _clone_model(model_name)
        final_model.fit(X_scaled_full, y)
        _save_model(final_model, model_name, ticker, label_version)

    # 5. Print report and export results
    print_classification_report(results)
    export_results(results, ticker, label_version)

    logger.info("Training pipeline complete for %s (label version %s)", ticker, label_version)
    return results


def _clone_model(model_name: str):
    """Create a fresh model instance by name."""
    return _get_models()[model_name]


def _save_model(model, model_name: str, ticker: str, label_version: str) -> None:
    """Save a trained model to disk."""
    if model_name == "xgboost":
        path = os.path.join(MODEL_DIR, f"{model_name}_{ticker}_{label_version}.json")
        model.save_model(path)
    elif model_name == "lightgbm":
        path = os.path.join(MODEL_DIR, f"{model_name}_{ticker}_{label_version}.txt")
        model.booster_.save_model(path)
    else:
        path = os.path.join(MODEL_DIR, f"{model_name}_{ticker}_{label_version}.joblib")
        joblib.dump(model, path)

    logger.info("Saved %s model to %s", model_name, path)
