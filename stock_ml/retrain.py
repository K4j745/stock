#!/usr/bin/env python3
"""Monthly retrain script.

Pulls the most recent ~252 trading days of OHLCV for each ticker, rebuilds
the feature matrix, trains the configured ML models and writes them to disk
under ``MODEL_DIR``. Intended to be invoked once a month (e.g. on the 1st
via a scheduled GitHub Actions workflow).

Usage:
    python stock_ml/retrain.py                # all tickers, binary mode
    python stock_ml/retrain.py --ticker AAPL  # single ticker
    python stock_ml/retrain.py --label-mode multiclass
"""
import argparse
import logging
import os
import sys

import joblib
from sklearn.preprocessing import StandardScaler

# Make stock_ml importable when this file is run directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    DEFAULT_LABEL_MODE,
    MODEL_DIR,
    TICKERS,
    WF_TRAIN_SIZE,
    logger,
)
from features.pipeline import build_feature_matrix
from models.train import _clone_model, _get_models, _save_model


def retrain_ticker(
    ticker: str,
    label_mode: str = DEFAULT_LABEL_MODE,
    label_version: str = "A",
    refresh: bool = True,
) -> None:
    """Retrain all models for a single ticker and save them to MODEL_DIR.

    Uses the last ``WF_TRAIN_SIZE`` (~252) sessions for fitting. Scaler and
    each trained model are written next to the previous artefacts; existing
    files are overwritten.
    """
    logger.info(
        "Retraining %s | label_mode=%s | refresh=%s", ticker, label_mode, refresh
    )

    X, y = build_feature_matrix(
        ticker, label_version=label_version, refresh=refresh, label_mode=label_mode
    )
    if len(X) < WF_TRAIN_SIZE:
        logger.warning(
            "Not enough data for %s (%d rows < %d); using all available",
            ticker, len(X), WF_TRAIN_SIZE,
        )
        X_train, y_train = X, y
    else:
        X_train, y_train = X.iloc[-WF_TRAIN_SIZE:], y.iloc[-WF_TRAIN_SIZE:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    scaler_path = os.path.join(MODEL_DIR, f"scaler_{ticker}_{label_version}.joblib")
    joblib.dump(scaler, scaler_path)
    logger.info("Saved scaler -> %s", scaler_path)

    feature_path = os.path.join(MODEL_DIR, f"features_{ticker}_{label_version}.joblib")
    joblib.dump(list(X.columns), feature_path)

    for model_name in _get_models(label_mode=label_mode):
        model = _clone_model(model_name, label_mode=label_mode)
        model.fit(X_train_scaled, y_train)
        _save_model(model, model_name, ticker, label_version)

    logger.info("Retrain complete for %s", ticker)


def main():
    parser = argparse.ArgumentParser(description="Monthly retrain for stock_ml models")
    parser.add_argument("--ticker", type=str, default=None,
                        help="Retrain a single ticker (default: all from config)")
    parser.add_argument("--label-mode", type=str, default=DEFAULT_LABEL_MODE,
                        choices=["binary", "multiclass", "legacy"])
    parser.add_argument("--label-version", type=str, default="A", choices=["A", "B"],
                        help="Only used when --label-mode=legacy")
    parser.add_argument("--no-refresh", action="store_true",
                        help="Use cached OHLCV instead of redownloading")
    args = parser.parse_args()

    tickers = [args.ticker] if args.ticker else TICKERS
    for tk in tickers:
        try:
            retrain_ticker(
                tk,
                label_mode=args.label_mode,
                label_version=args.label_version,
                refresh=not args.no_refresh,
            )
        except Exception:
            logger.exception("Retrain failed for %s", tk)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
