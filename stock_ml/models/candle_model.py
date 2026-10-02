"""Candlestick model — a separate classifier trained only on candle-shape
features plus a small set of base technical indicators.

This is intentionally kept apart from the main model lineup so the thesis can
answer a focused question: *how much predictive signal is in the raw candle
geometry alone?* It reuses the same leakage-safe walk-forward protocol and the
same binary BUY/SELL labels as the main pipeline.

CLI entry point is wired in ``main.py`` (``python main.py candle ...``); the
dashboard consumes :func:`predict_candle_proba` to add a "Candle Signal".
"""
from __future__ import annotations

import logging
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from config import LABEL_THRESHOLD, MODEL_DIR, RANDOM_STATE, WF_MODE, WF_STEP, WF_TEST_SIZE, WF_TRAIN_SIZE, model_tag
from data.download import download_single
from data.preprocess import clean_data
from features.candle_features import CANDLE_FEATURE_COLUMNS, compute_candle_features
from features.indicators import add_technical_indicators
from features.labels import create_binary_labels
from features.validation import count_folds, get_time_series_folds
from models.evaluate import evaluate_model

logger = logging.getLogger("stock_ml")

# Small "base technical" context the candle model is allowed to see alongside
# the pure candle-shape features. Kept deliberately compact.
BASE_TECH_COLUMNS = ["RSI_14", "SMA_20", "MACD_hist", "ATR_14", "Volume_ratio", "HL_spread", "ROC_5"]

CANDLE_MODEL_COLUMNS = CANDLE_FEATURE_COLUMNS + BASE_TECH_COLUMNS


def build_candle_feature_matrix(
    ticker: str,
    threshold: float = LABEL_THRESHOLD,
    refresh: bool = False,
) -> tuple[pd.DataFrame, pd.Series]:
    """Assemble ``(X, y)`` for the candle model.

    X = candle-shape features + a compact base-technical set. y = binary BUY/SELL
    label at the given decision ``threshold``.
    """
    df = download_single(ticker, refresh=refresh)
    if df.empty:
        raise ValueError(f"No data available for ticker {ticker}")

    df = clean_data(df)
    candles = compute_candle_features(df)
    tech = add_technical_indicators(df, ticker=ticker)

    # Align on the technical frame's index (indicators drop warm-up rows).
    feat = tech.join(candles, how="inner")
    feat["label"] = create_binary_labels(feat, threshold=threshold)
    feat = feat.dropna(subset=CANDLE_MODEL_COLUMNS + ["label"])

    y = feat["label"].astype(int)
    X = feat[CANDLE_MODEL_COLUMNS].copy()
    logger.info(
        "Candle feature matrix for %s: X=%s, BUY class %.1f%% (threshold=%.4f)",
        ticker, X.shape, 100 * (y == 1).mean(), threshold,
    )
    return X, y


def _make_model(model_type: str):
    if model_type == "xgboost":
        return XGBClassifier(n_estimators=200, random_state=RANDOM_STATE, eval_metric="logloss")
    if model_type == "random_forest":
        return RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
    raise ValueError(f"Unknown candle model_type '{model_type}'. Use 'xgboost' or 'random_forest'.")


def train_candle_model(
    ticker: str,
    threshold: float = LABEL_THRESHOLD,
    model_type: str = "xgboost",
    refresh: bool = False,
) -> dict:
    """Train the candle model with rolling walk-forward CV and save it.

    Returns a dict with per-fold and mean metrics.
    """
    X, y = build_candle_feature_matrix(ticker, threshold=threshold, refresh=refresh)

    if count_folds(len(X), WF_TRAIN_SIZE, WF_TEST_SIZE, WF_STEP) == 0:
        raise ValueError(
            f"Not enough data for {ticker}: {len(X)} rows < {WF_TRAIN_SIZE + WF_TEST_SIZE} required"
        )
    folds = list(get_time_series_folds(
        X, train_size=WF_TRAIN_SIZE, test_size=WF_TEST_SIZE, step=WF_STEP, mode=WF_MODE,
    ))

    fold_metrics = []
    for i, (train_idx, test_idx) in enumerate(folds):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        scaler = StandardScaler()
        X_tr_s = pd.DataFrame(scaler.fit_transform(X_tr), columns=X_tr.columns, index=X_tr.index)
        X_te_s = pd.DataFrame(scaler.transform(X_te), columns=X_te.columns, index=X_te.index)

        model = _make_model(model_type)
        model.fit(X_tr_s, y_tr)
        y_pred = model.predict(X_te_s)
        y_prob = model.predict_proba(X_te_s)[:, 1] if hasattr(model, "predict_proba") else None
        m = evaluate_model(y_te, y_pred, y_prob)
        fold_metrics.append(m)
        logger.info(
            "  candle_%s fold %d/%d: acc=%.4f f1=%.4f auc=%.4f",
            model_type, i + 1, len(folds), m["accuracy"], m["f1"], m.get("roc_auc", float("nan")),
        )

    mean_metrics = {
        k: float(np.nanmean([m[k] for m in fold_metrics])) for k in fold_metrics[0]
    }
    logger.info(
        "  candle_%s MEAN: acc=%.4f prec=%.4f rec=%.4f f1=%.4f auc=%.4f",
        model_type, mean_metrics["accuracy"], mean_metrics["precision"],
        mean_metrics["recall"], mean_metrics["f1"], mean_metrics.get("roc_auc", float("nan")),
    )

    # Final artifact: fit on the last walk-forward train window.
    last_train_idx, _ = folds[-1]
    X_final = X.iloc[last_train_idx]
    y_final = y.iloc[last_train_idx]
    final_scaler = StandardScaler()
    X_final_s = pd.DataFrame(
        final_scaler.fit_transform(X_final), columns=X_final.columns, index=X_final.index,
    )
    final_model = _make_model(model_type)
    final_model.fit(X_final_s, y_final)
    _save_candle_model(final_model, final_scaler, ticker, threshold, model_type)

    return {"fold_metrics": fold_metrics, "mean_metrics": mean_metrics}


def _candle_paths(ticker: str, threshold: float, model_type: str) -> tuple[str, str]:
    tag = model_tag("binary", "", threshold)
    if model_type == "xgboost":
        model_path = os.path.join(MODEL_DIR, f"candle_xgboost_{ticker}_{tag}.json")
    else:
        model_path = os.path.join(MODEL_DIR, f"candle_{model_type}_{ticker}_{tag}.joblib")
    scaler_path = os.path.join(MODEL_DIR, f"candle_scaler_{ticker}_{tag}.joblib")
    return model_path, scaler_path


def _save_candle_model(model, scaler, ticker: str, threshold: float, model_type: str) -> None:
    model_path, scaler_path = _candle_paths(ticker, threshold, model_type)
    if model_type == "xgboost":
        model.save_model(model_path)
    else:
        joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    logger.info("Saved candle model -> %s (+ scaler)", model_path)


def predict_candle_proba(
    ticker: str,
    df: pd.DataFrame,
    threshold: float = LABEL_THRESHOLD,
    model_type: str = "xgboost",
) -> pd.Series | None:
    """Return the BUY probability series for a saved candle model.

    ``df`` is an OHLCV frame (indicators are recomputed internally). Returns
    ``None`` when no trained artifact exists, so callers (e.g. the dashboard)
    can transparently fall back to a proxy signal.
    """
    model_path, scaler_path = _candle_paths(ticker, threshold, model_type)
    if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
        return None

    clean = clean_data(df.copy())
    candles = compute_candle_features(clean)
    tech = add_technical_indicators(clean, ticker=ticker)
    feat = tech.join(candles, how="inner").dropna(subset=CANDLE_MODEL_COLUMNS)
    if feat.empty:
        return None

    X = feat[CANDLE_MODEL_COLUMNS]
    scaler = joblib.load(scaler_path)
    X_s = pd.DataFrame(scaler.transform(X), columns=X.columns, index=X.index)

    if model_type == "xgboost":
        model = XGBClassifier()
        model.load_model(model_path)
    else:
        model = joblib.load(model_path)
    proba = model.predict_proba(X_s)[:, 1]
    return pd.Series(proba, index=X.index, name="candle_prob_up")
