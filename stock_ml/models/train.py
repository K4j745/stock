import logging
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from config import (
    RANDOM_STATE,
    MODEL_DIR,
    WF_TRAIN_SIZE,
    WF_TEST_SIZE,
    WF_STEP,
    WF_MODE,
    DEFAULT_LABEL_MODE,
    LABEL_THRESHOLD,
    model_tag,
)
from features.pipeline import build_feature_matrix
from features.validation import get_time_series_folds, count_folds
from models.evaluate import evaluate_model, print_classification_report
from reports.generate import export_results

logger = logging.getLogger("stock_ml")


def _get_models(label_mode: str = DEFAULT_LABEL_MODE) -> dict:
    """Return dictionary of model name -> model instance.

    When ``label_mode='multiclass'``, classifiers are configured with
    ``class_weight='balanced'`` to mitigate strong class imbalance from the
    dominant 'hold' class.
    """
    use_balanced = label_mode == "multiclass"
    lr_kwargs = {"random_state": RANDOM_STATE, "max_iter": 1000}
    rf_kwargs = {"n_estimators": 200, "random_state": RANDOM_STATE}
    if use_balanced:
        lr_kwargs["class_weight"] = "balanced"
        rf_kwargs["class_weight"] = "balanced"

    return {
        "logistic_regression": LogisticRegression(**lr_kwargs),
        "random_forest": RandomForestClassifier(**rf_kwargs),
        "xgboost": XGBClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            eval_metric="logloss",
        ),
        "lightgbm": LGBMClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            verbose=-1,
            class_weight="balanced" if use_balanced else None,
        ),
    }


def train_all_models(
    ticker: str,
    label_version: str = "A",
    refresh: bool = False,
    label_mode: str = DEFAULT_LABEL_MODE,
    threshold: float = LABEL_THRESHOLD,
) -> dict:
    """Train all models for a given ticker using rolling walk-forward CV.

    Steps:
        1. Build feature matrix
        2. Rolling walk-forward cross-validation (no leakage, fixed train window)
        3. Evaluate per fold
        4. Retrain on the last walk-forward train slice and save model + scaler

    Saved artifacts are keyed by ``model_tag`` (legacy ``A``/``B`` or a
    threshold-derived ``bin5``/``bin10`` tag) so both binary thresholds coexist.

    Returns:
        Dictionary: {model_name: {fold_metrics: [...], mean_metrics: {...}}}
    """
    tag = model_tag(label_mode, label_version, threshold)
    logger.info("Starting training pipeline for %s (label_mode=%s, tag=%s)", ticker, label_mode, tag)

    # 1. Build feature matrix
    X, y = build_feature_matrix(
        ticker, label_version, refresh=refresh, label_mode=label_mode, threshold=threshold,
    )

    # 2. Walk-forward folds (rolling window by default)
    n_folds = count_folds(len(X), WF_TRAIN_SIZE, WF_TEST_SIZE, WF_STEP)
    folds = list(get_time_series_folds(
        X, train_size=WF_TRAIN_SIZE, test_size=WF_TEST_SIZE, step=WF_STEP, mode=WF_MODE,
    ))
    if not folds:
        raise ValueError(
            f"Not enough data for {ticker}: {len(X)} rows < "
            f"{WF_TRAIN_SIZE + WF_TEST_SIZE} required"
        )
    models = _get_models(label_mode=label_mode)
    results = {}

    # 3. Cross-validation for each model
    for model_name, model in models.items():
        logger.info("Training %s on %s (%d folds)...", model_name, ticker, len(folds))
        fold_metrics = []

        for fold_idx, (train_idx, test_idx) in enumerate(folds):
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
            model_clone = _clone_model(model_name, label_mode=label_mode)
            model_clone.fit(X_train_scaled, y_train)

            # Predict
            y_pred = model_clone.predict(X_test_scaled)
            y_prob = None
            if hasattr(model_clone, "predict_proba"):
                proba = model_clone.predict_proba(X_test_scaled)
                if proba.shape[1] == 2:
                    y_prob = proba[:, 1]

            # Evaluate
            metrics = evaluate_model(y_test, y_pred, y_prob)
            fold_metrics.append(metrics)

            logger.info(
                "  %s fold %d/%d: acc=%.4f prec=%.4f rec=%.4f f1=%.4f mcc=%.4f",
                model_name, fold_idx + 1, len(folds),
                metrics["accuracy"], metrics["precision"], metrics["recall"],
                metrics["f1"], metrics["mcc"],
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
            "  %s MEAN: acc=%.4f prec=%.4f rec=%.4f f1=%.4f mcc=%.4f",
            model_name,
            mean_metrics["accuracy"], mean_metrics["precision"],
            mean_metrics["recall"], mean_metrics["f1"], mean_metrics["mcc"],
        )

    # 4. Train final saved models on the last walk-forward train window
    logger.info("Training final saved models on last walk-forward train window for %s...", ticker)
    last_train_idx, _ = folds[-1]
    X_train_final = X.iloc[last_train_idx]
    y_train_final = y.iloc[last_train_idx]

    final_scaler = StandardScaler()
    X_train_final_scaled = pd.DataFrame(
        final_scaler.fit_transform(X_train_final),
        columns=X_train_final.columns,
        index=X_train_final.index,
    )

    # Save the scaler
    scaler_path = os.path.join(MODEL_DIR, f"scaler_{ticker}_{tag}.joblib")
    joblib.dump(final_scaler, scaler_path)
    logger.info("Saved scaler to %s", scaler_path)

    for model_name in models:
        final_model = _clone_model(model_name, label_mode=label_mode)
        final_model.fit(X_train_final_scaled, y_train_final)
        _save_model(final_model, model_name, ticker, tag)

    # 5. Print report and export results
    print_classification_report(results)
    export_results(results, ticker, tag)

    logger.info("Training pipeline complete for %s (label version %s)", ticker, label_version)
    return results


def _clone_model(model_name: str, label_mode: str = DEFAULT_LABEL_MODE):
    """Create a fresh model instance by name."""
    return _get_models(label_mode=label_mode)[model_name]


def _save_model(model, model_name: str, ticker: str, tag: str) -> None:
    """Save a trained model to disk, keyed by ``tag`` (label version/threshold)."""
    if model_name == "xgboost":
        path = os.path.join(MODEL_DIR, f"{model_name}_{ticker}_{tag}.json")
        model.save_model(path)
    elif model_name == "lightgbm":
        path = os.path.join(MODEL_DIR, f"{model_name}_{ticker}_{tag}.txt")
        model.booster_.save_model(path)
    else:
        path = os.path.join(MODEL_DIR, f"{model_name}_{ticker}_{tag}.joblib")
        joblib.dump(model, path)

    logger.info("Saved %s model to %s", model_name, path)
