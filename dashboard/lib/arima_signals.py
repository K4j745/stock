"""ARIMA signal stream for the dashboard.

Unlike the ML models, ARIMA needs no saved artifacts: it is refitted on a
rolling window of past daily returns while the signals are generated, so it
runs the same way locally and on CI.

For each day ``t`` the target is the next-day return. A fit uses only returns
known at the close of the day before the first forecast, and each forecast uses
fixed parameters plus the returns up to its own day, so nothing looks ahead.
Days before the first fit (``min_train``) have no forecast and become HOLD.
"""
from __future__ import annotations

import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from . import ml_loader
from . import signals as signals_mod

logger = logging.getLogger("dashboard.arima")

MODEL_NAME = "arima"
WARMUP_RULE = "arima_warmup"
DEFAULT_SETTINGS = {"order": [1, 0, 1], "train_window": 252, "min_train": 60, "refit_every": 21}


def _settings(settings: Dict) -> Tuple[Tuple[int, int, int], int, int, int]:
    merged = {**DEFAULT_SETTINGS, **(settings or {})}
    order = tuple(int(v) for v in merged["order"])
    window = int(merged["train_window"])
    min_train = int(merged["min_train"])
    refit_every = int(merged["refit_every"])
    if len(order) != 3 or min_train < 1 or refit_every < 1 or window < min_train:
        raise ValueError(f"Invalid ARIMA settings: {merged}")
    return order, window, min_train, refit_every


def _load_core():
    if not ml_loader.ensure_stock_ml_on_path():
        raise ImportError("stock_ml directory not found")
    from models import arima_core  # type: ignore
    return arima_core


def model_version(settings: Dict) -> str:
    order, window, _, refit_every = _settings(settings)
    return f"arima{order}-window{window}-refit{refit_every}"


def arima_probabilities(close: pd.Series, threshold: float, settings: Dict, ticker: str) -> pd.Series:
    """Return P(next-day return > ``threshold``) per day, NaN while warming up.

    Raises ImportError when statsmodels or stock_ml is unavailable, so the
    caller can leave ARIMA out instead of publishing made-up signals.
    """
    order, window, min_train, refit_every = _settings(settings)
    core = _load_core()

    next_returns = close.pct_change().shift(-1).to_numpy()
    probabilities = np.full(len(next_returns), np.nan)
    unconverged, failed, fits = 0, 0, 0

    for start in range(min_train, len(next_returns), refit_every):
        stop = min(start + refit_every, len(next_returns))
        train = next_returns[max(0, start - window):start]
        fits += 1
        try:
            forecast = core.forecast_one_step(train, next_returns[start:stop], order)
        except (ValueError, np.linalg.LinAlgError) as exc:
            failed += 1
            logger.warning("ARIMA fit failed for %s at %s: %s", ticker, close.index[start].date(), exc)
            continue
        unconverged += not forecast.converged
        probabilities[start:stop] = core.probability_above(forecast.mean, forecast.std, threshold)

    if unconverged or failed:
        logger.warning("ARIMA%s %s: %d of %d fits did not converge, %d failed (left as HOLD)",
                       order, ticker, unconverged, fits, failed)
    return pd.Series(probabilities, index=close.index)


def to_signals(probabilities: pd.Series, buy_threshold: float) -> pd.DataFrame:
    """Binary BUY/SELL like the ML models, with HOLD for days without a forecast."""
    signals = signals_mod.from_probabilities(probabilities, buy_threshold, 1 - buy_threshold)
    signals.loc[probabilities.isna(), "triggered_rules"] = WARMUP_RULE
    return signals
