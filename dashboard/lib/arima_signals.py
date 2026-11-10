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
from collections import Counter
from typing import Dict, NamedTuple, Tuple

import numpy as np
import pandas as pd

from . import ml_loader
from . import signals as signals_mod

logger = logging.getLogger("dashboard.arima")

MODEL_NAME = "arima"
WARMUP_RULE = "arima_warmup"
DEFAULT_SETTINGS = {
    "order_selection": "bic",
    "order": [1, 0, 1],
    "max_p": 2,
    "max_q": 2,
    "adf_alpha": 0.05,
    "train_window": 252,
    "min_train": 60,
    "refit_every": 21,
}


class ArimaSettings(NamedTuple):
    selection: str
    order: Tuple[int, int, int]
    max_p: int
    max_q: int
    adf_alpha: float
    window: int
    min_train: int
    refit_every: int


def _settings(settings: Dict) -> ArimaSettings:
    merged = {**DEFAULT_SETTINGS, **(settings or {})}
    parsed = ArimaSettings(
        selection=str(merged["order_selection"]),
        order=tuple(int(v) for v in merged["order"]),
        max_p=int(merged["max_p"]),
        max_q=int(merged["max_q"]),
        adf_alpha=float(merged["adf_alpha"]),
        window=int(merged["train_window"]),
        min_train=int(merged["min_train"]),
        refit_every=int(merged["refit_every"]),
    )
    if (parsed.selection not in {"bic", "fixed"} or len(parsed.order) != 3
            or parsed.min_train < 1 or parsed.refit_every < 1 or parsed.window < parsed.min_train):
        raise ValueError(f"Invalid ARIMA settings: {merged}")
    return parsed


def _load_core():
    if not ml_loader.ensure_stock_ml_on_path():
        raise ImportError("stock_ml directory not found")
    from models import arima_core  # type: ignore
    return arima_core


def model_version(settings: Dict) -> str:
    s = _settings(settings)
    spec = (f"bic-p{s.max_p}q{s.max_q}-adf{s.adf_alpha:g}" if s.selection == "bic" else f"{s.order}")
    return f"arima-{spec}-window{s.window}-refit{s.refit_every}"


def _forecast(core, s: ArimaSettings, train: np.ndarray, test: np.ndarray):
    """One-step forecasts for ``test`` plus the order used, chosen on ``train`` if BIC."""
    if s.selection == "bic":
        forecast, chosen = core.forecast_auto(train, test, s.max_p, s.max_q, s.adf_alpha)
        return forecast, chosen.order
    return core.forecast_one_step(train, test, s.order), s.order


def arima_probabilities(close: pd.Series, threshold: float, settings: Dict, ticker: str) -> pd.Series:
    """Return P(next-day return > ``threshold``) per day, NaN while warming up.

    With ``order_selection = "bic"`` each refit first picks the order on its own
    training window (ADF test for d, lowest BIC for p and q).

    Raises ImportError when statsmodels or stock_ml is unavailable, so the
    caller can leave ARIMA out instead of publishing made-up signals.
    """
    s = _settings(settings)
    core = _load_core()

    next_returns = close.pct_change().shift(-1).to_numpy()
    probabilities = np.full(len(next_returns), np.nan)
    unconverged, failed, fits = 0, 0, 0
    orders: Counter = Counter()

    for start in range(s.min_train, len(next_returns), s.refit_every):
        stop = min(start + s.refit_every, len(next_returns))
        train = next_returns[max(0, start - s.window):start]
        fits += 1
        try:
            forecast, order = _forecast(core, s, train, next_returns[start:stop])
        except (ValueError, np.linalg.LinAlgError) as exc:
            failed += 1
            logger.warning("ARIMA fit failed for %s at %s: %s", ticker, close.index[start].date(), exc)
            continue
        orders[order] += 1
        unconverged += not forecast.converged
        probabilities[start:stop] = core.probability_above(forecast.mean, forecast.std, threshold)

    logger.info("ARIMA %s: orders used %s", ticker, dict(orders.most_common()))
    if unconverged or failed:
        logger.warning("ARIMA %s: %d of %d fits did not converge, %d failed (left as HOLD)",
                       ticker, unconverged, fits, failed)
    return pd.Series(probabilities, index=close.index)


def to_signals(probabilities: pd.Series, buy_threshold: float) -> pd.DataFrame:
    """Binary BUY/SELL like the ML models, with HOLD for days without a forecast."""
    signals = signals_mod.from_probabilities(probabilities, buy_threshold, 1 - buy_threshold)
    signals.loc[probabilities.isna(), "triggered_rules"] = WARMUP_RULE
    return signals
