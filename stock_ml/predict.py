#!/usr/bin/env python3
"""Daily prediction script.

Loads the model saved by ``retrain.py``, computes the latest feature row for
each ticker and emits an UP/DOWN signal for the next session. Designed to
run every trading day.

Usage:
    python stock_ml/predict.py                       # all tickers, default xgboost
    python stock_ml/predict.py --model lightgbm
    python stock_ml/predict.py --ticker AAPL --refresh
    python stock_ml/predict.py --out reports/signals_2026-05-15.csv
"""
import argparse
import logging
import os
import sys
from datetime import datetime

import joblib
import pandas as pd

# Make stock_ml importable when this file is run directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    DEFAULT_LABEL_MODE,
    MODEL_DIR,
    REPORTS_DIR,
    TICKERS,
    logger,
)
from features.pipeline import build_feature_matrix


def _load_model(model_name: str, ticker: str, label_version: str):
    if model_name == "xgboost":
        import xgboost as xgb
        path = os.path.join(MODEL_DIR, f"xgboost_{ticker}_{label_version}.json")
        if not os.path.exists(path):
            return None
        model = xgb.XGBClassifier()
        model.load_model(path)
        return model
    if model_name == "lightgbm":
        import lightgbm as lgb
        path = os.path.join(MODEL_DIR, f"lightgbm_{ticker}_{label_version}.txt")
        if not os.path.exists(path):
            return None
        return lgb.Booster(model_file=path)
    path = os.path.join(MODEL_DIR, f"{model_name}_{ticker}_{label_version}.joblib")
    if not os.path.exists(path):
        return None
    return joblib.load(path)


def predict_one(
    ticker: str,
    model_name: str,
    label_version: str = "A",
    label_mode: str = DEFAULT_LABEL_MODE,
    refresh: bool = False,
) -> dict:
    """Compute today's signal for ``ticker`` using the saved model.

    Returns a dict with date, ticker, model, signal (UP/DOWN) and class
    probability when the model supports it.
    """
    X, _ = build_feature_matrix(
        ticker, label_version=label_version, refresh=refresh, label_mode=label_mode
    )
    if X.empty:
        raise ValueError(f"Empty feature matrix for {ticker}")

    last_row = X.iloc[[-1]]
    as_of = last_row.index[-1]

    scaler_path = os.path.join(MODEL_DIR, f"scaler_{ticker}_{label_version}.joblib")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler missing: {scaler_path}. Run retrain.py first.")
    scaler = joblib.load(scaler_path)
    last_scaled = scaler.transform(last_row)

    model = _load_model(model_name, ticker, label_version)
    if model is None:
        raise FileNotFoundError(
            f"Model {model_name} missing for {ticker}. Run retrain.py first."
        )

    if model_name == "lightgbm":
        proba = float(model.predict(last_scaled)[0])
        pred = int(proba > 0.5)
    else:
        pred = int(model.predict(last_scaled)[0])
        proba = float(model.predict_proba(last_scaled)[0, 1]) if hasattr(
            model, "predict_proba"
        ) else None

    return {
        "date": pd.Timestamp(as_of).date().isoformat(),
        "ticker": ticker,
        "model": model_name,
        "label_mode": label_mode,
        "prediction": int(pred),
        "signal": "UP" if pred == 1 else "DOWN",
        "prob_up": proba,
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }


def predict_all(
    tickers,
    model_name: str,
    label_version: str = "A",
    label_mode: str = DEFAULT_LABEL_MODE,
    refresh: bool = False,
) -> pd.DataFrame:
    """Run prediction for each ticker; failures are logged and skipped."""
    rows = []
    for ticker in tickers:
        try:
            rows.append(predict_one(
                ticker, model_name, label_version, label_mode, refresh
            ))
        except Exception:
            logger.exception("Prediction failed for %s", ticker)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Daily prediction for stock_ml models")
    parser.add_argument("--ticker", type=str, default=None,
                        help="Predict a single ticker (default: all from config)")
    parser.add_argument("--model", type=str, default="xgboost",
                        choices=["logistic_regression", "random_forest", "xgboost", "lightgbm"])
    parser.add_argument("--label-mode", type=str, default=DEFAULT_LABEL_MODE,
                        choices=["binary", "multiclass", "legacy"])
    parser.add_argument("--label-version", type=str, default="A", choices=["A", "B"])
    parser.add_argument("--refresh", action="store_true",
                        help="Force fresh OHLCV download (recommended for daily runs)")
    parser.add_argument("--out", type=str, default=None,
                        help="Output CSV path (default: reports/signals_YYYY-MM-DD.csv)")
    args = parser.parse_args()

    tickers = [args.ticker] if args.ticker else TICKERS
    df = predict_all(
        tickers,
        model_name=args.model,
        label_version=args.label_version,
        label_mode=args.label_mode,
        refresh=args.refresh,
    )

    if df.empty:
        logger.warning("No predictions generated.")
        return

    out_path = args.out or os.path.join(
        REPORTS_DIR, f"signals_{datetime.utcnow().strftime('%Y-%m-%d')}.csv"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info("Wrote %d signals -> %s", len(df), out_path)
    for _, row in df.iterrows():
        prob = row["prob_up"]
        prob_str = f"{prob:.3f}" if prob is not None else "n/a"
        logger.info(
            "  %s %s -> %s (prob_up=%s)",
            row["date"], row["ticker"], row["signal"], prob_str,
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
