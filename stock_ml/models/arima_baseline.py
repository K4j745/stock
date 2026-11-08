"""ARIMA baseline for next-day direction.

A classical time-series reference point for the ML classifiers. It sees only the
return series (no technical indicators) and is scored on the same walk-forward
folds, labels and metrics as the classifiers in ``models.train``.

ARIMA predicts a number, not a class, so the forecast is converted to the binary
label: BUY when the expected next-day return exceeds the decision threshold, with
``P(return > threshold)`` taken from the normal forecast distribution for ROC AUC.

Parameters are estimated once per fold on the training window. Test forecasts then
use those fixed parameters, each one seeing only observations before its own day.
"""
import json
import logging
import os
from typing import Tuple

import numpy as np
import pandas as pd

from config import (
    ARIMA_ORDER,
    DEFAULT_LABEL_MODE,
    LABEL_THRESHOLD,
    REPORTS_DIR,
    WF_MODE,
    WF_STEP,
    WF_TEST_SIZE,
    WF_TRAIN_SIZE,
    model_tag,
)
from data.download import download_single
from data.preprocess import clean_data
from features.pipeline import build_feature_matrix
from features.validation import get_time_series_folds
from models.arima_core import OneStepForecast, probability_above
from models.arima_core import forecast_one_step as _forecast_one_step
from models.evaluate import evaluate_model
from reports.generate import export_results

logger = logging.getLogger("stock_ml")


def forecast_one_step(
    train: np.ndarray,
    test: np.ndarray,
    order: Tuple[int, int, int] = ARIMA_ORDER,
) -> OneStepForecast:
    """Forecast each ``test`` value one step ahead, ARIMA_ORDER by default."""
    return _forecast_one_step(train, test, order)


def _mean_metrics(fold_metrics: list) -> dict:
    """Average each metric across folds, ignoring undefined (NaN) values."""
    means = {}
    for key in fold_metrics[0]:
        values = [m[key] for m in fold_metrics if not np.isnan(m[key])]
        means[key] = float(np.mean(values)) if values else float("nan")
    return means


def walk_forward_arima(
    next_returns: pd.Series,
    labels: pd.Series,
    threshold: float,
    train_size: int = WF_TRAIN_SIZE,
    test_size: int = WF_TEST_SIZE,
    step: int = WF_STEP,
    mode: str = WF_MODE,
    order: Tuple[int, int, int] = ARIMA_ORDER,
) -> dict:
    """Score ARIMA on walk-forward folds.

    Args:
        next_returns: Next-day return for each date, aligned with ``labels``.
        labels: Binary labels (1 when the next-day return exceeds ``threshold``).
        threshold: Decision threshold on the next-day return.

    Returns:
        Dict with ``fold_metrics`` and ``mean_metrics``, as in ``train_all_models``.
    """
    if not next_returns.index.equals(labels.index):
        raise ValueError("next_returns and labels must be aligned on the same index")

    returns = next_returns.to_numpy()
    targets = labels.to_numpy()
    fold_metrics = []
    unconverged = 0

    for train_idx, test_idx in get_time_series_folds(
        returns, train_size=train_size, test_size=test_size, step=step, mode=mode,
    ):
        forecast = forecast_one_step(returns[train_idx], returns[test_idx], order)
        unconverged += not forecast.converged
        probabilities = probability_above(forecast.mean, forecast.std, threshold)
        predictions = (forecast.mean > threshold).astype(int)
        fold_metrics.append(evaluate_model(targets[test_idx], predictions, probabilities))

    if unconverged:
        logger.warning(
            "ARIMA%s did not converge on %d/%d folds", order, unconverged, len(fold_metrics),
        )
    return {"fold_metrics": fold_metrics, "mean_metrics": _mean_metrics(fold_metrics)}


def run_arima_baseline(
    ticker: str,
    label_version: str = "A",
    label_mode: str = DEFAULT_LABEL_MODE,
    threshold: float = LABEL_THRESHOLD,
) -> dict:
    """Evaluate ARIMA for one ticker and add it to the ticker's results report.

    Results are merged under the ``arima`` key, so ``main.py report`` shows ARIMA
    next to the ML models. ``main.py train`` rewrites the report without this key;
    run this baseline again after retraining.
    """
    if label_mode != "binary":
        raise ValueError(f"ARIMA baseline supports binary labels only, got '{label_mode}'")

    tag = model_tag(label_mode, label_version, threshold)
    _, labels = build_feature_matrix(
        ticker, label_version, label_mode=label_mode, threshold=threshold,
    )
    close = clean_data(download_single(ticker))["Close"]
    next_returns = close.pct_change().shift(-1).reindex(labels.index)

    # The newest row has no next-day return yet, so it cannot be scored.
    known = next_returns.notna()
    result = walk_forward_arima(next_returns[known], labels[known], threshold)
    logger.info(
        "ARIMA%s %s (%s): acc=%.4f f1=%.4f auc=%.4f mcc=%.4f (%d folds)",
        ARIMA_ORDER, ticker, tag,
        result["mean_metrics"]["accuracy"], result["mean_metrics"]["f1"],
        result["mean_metrics"]["roc_auc"], result["mean_metrics"]["mcc"],
        len(result["fold_metrics"]),
    )

    report_path = os.path.join(REPORTS_DIR, f"{ticker}_{tag}_results.json")
    existing = {}
    if os.path.exists(report_path):
        with open(report_path) as report_file:
            existing = json.load(report_file)
    export_results({**existing, "arima": result}, ticker, tag)
    return result
