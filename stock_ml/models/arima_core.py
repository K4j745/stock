"""ARIMA forecasting core shared by the thesis baseline and the dashboard.

Kept free of the rest of ``stock_ml`` (config, data download, feature pipeline)
so the dashboard can import it with only numpy, scipy and statsmodels installed.

Order selection follows the Box-Jenkins steps on the training window only:
an augmented Dickey-Fuller test decides whether to difference the series
(d = 0 or 1), then every (p, q) on a small grid is fitted and the lowest BIC
wins. Nothing outside the training window is used to choose the order.
"""
import warnings
from typing import NamedTuple, Tuple

import numpy as np
from scipy.stats import norm
from statsmodels.tools.sm_exceptions import ConvergenceWarning, EstimationWarning
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

# Daily returns are ~0.01; fitting in percent keeps the optimiser well conditioned.
PERCENT_SCALE = 100.0


class OneStepForecast(NamedTuple):
    """One-step-ahead forecasts for a test window, in decimal return units."""

    mean: np.ndarray
    std: np.ndarray
    converged: bool


class OrderSelection(NamedTuple):
    """The order chosen for one training window and the evidence behind it."""

    order: Tuple[int, int, int]
    bic: float
    adf_pvalue: float


def _fit(series_scaled: np.ndarray, order: Tuple[int, int, int]):
    # Non-convergence is reported through ``converged``, not as a warning per fit.
    # EstimationWarning only says statsmodels reset its starting parameters to zero.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", EstimationWarning)
        return ARIMA(series_scaled, order=order).fit()


def _forecast(fitted, train_len: int, test_scaled: np.ndarray) -> OneStepForecast:
    extended = fitted.append(test_scaled, refit=False)
    prediction = extended.get_prediction(start=train_len, end=train_len + len(test_scaled) - 1)
    return OneStepForecast(
        mean=np.asarray(prediction.predicted_mean) / PERCENT_SCALE,
        std=np.asarray(prediction.se_mean) / PERCENT_SCALE,
        converged=bool(fitted.mle_retvals.get("converged", True)),
    )


def forecast_one_step(
    train: np.ndarray,
    test: np.ndarray,
    order: Tuple[int, int, int],
) -> OneStepForecast:
    """Fit ARIMA of a fixed ``order`` on ``train`` and forecast ``test`` one step ahead.

    Parameters are estimated on ``train`` only. Each test forecast uses those
    fixed parameters and the observations before its own position. A NaN in
    ``test`` is treated as a missing observation.
    """
    train_scaled = np.asarray(train, dtype=float) * PERCENT_SCALE
    fitted = _fit(train_scaled, order)
    return _forecast(fitted, len(train_scaled), np.asarray(test, dtype=float) * PERCENT_SCALE)


def adf_pvalue(series: np.ndarray) -> float:
    """p-value of the augmented Dickey-Fuller test (H0: the series has a unit root)."""
    values = np.asarray(series, dtype=float)
    try:
        return float(adfuller(values, autolag="AIC", result_object=True).pvalue)
    except TypeError:
        # statsmodels < 0.15 has no ``result_object`` and returns a tuple.
        return float(adfuller(values, autolag="AIC")[1])


def _check_search(max_p: int, max_q: int, adf_alpha: float) -> None:
    if max_p < 0 or max_q < 0 or not 0 < adf_alpha < 1:
        raise ValueError(f"Invalid order search: max_p={max_p}, max_q={max_q}, adf_alpha={adf_alpha}")


def _candidates(train_scaled: np.ndarray, max_p: int, max_q: int, adf_alpha: float):
    """Fitted models on the grid, best BIC first, plus the ADF p-value.

    Only fits whose likelihood optimiser converged are ranked: the BIC of a
    fit that did not converge is not a valid basis for comparison.
    """
    _check_search(max_p, max_q, adf_alpha)
    pvalue = adf_pvalue(train_scaled)
    d = 0 if pvalue < adf_alpha else 1
    ranked = []
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            try:
                fitted = _fit(train_scaled, (p, d, q))
            except (ValueError, np.linalg.LinAlgError):
                continue
            converged = bool(fitted.mle_retvals.get("converged", True))
            if converged and np.isfinite(fitted.bic):
                ranked.append((float(fitted.bic), (p, d, q), fitted))
    if not ranked:
        raise ValueError("No ARIMA order on the grid converged")
    ranked.sort(key=lambda item: item[0])
    return ranked, pvalue


def _is_valid(forecast: OneStepForecast) -> bool:
    return bool(np.all(np.isfinite(forecast.mean)) and np.all(np.isfinite(forecast.std))
                and np.all(forecast.std > 0))


def select_order(train: np.ndarray, max_p: int, max_q: int, adf_alpha: float) -> OrderSelection:
    """Choose (p, d, q) for ``train``: ADF decides d, the lowest BIC among
    converged fits picks p and q."""
    ranked, pvalue = _candidates(np.asarray(train, dtype=float) * PERCENT_SCALE, max_p, max_q, adf_alpha)
    bic, order, _ = ranked[0]
    return OrderSelection(order, bic, pvalue)


def forecast_auto(
    train: np.ndarray,
    test: np.ndarray,
    max_p: int,
    max_q: int,
    adf_alpha: float,
) -> Tuple[OneStepForecast, OrderSelection]:
    """Select the order on ``train`` and forecast ``test`` with that fitted model.

    If the best-BIC model yields a degenerate forecast (non-finite or zero
    standard error), the next model by BIC is used instead.
    """
    train_scaled = np.asarray(train, dtype=float) * PERCENT_SCALE
    test_scaled = np.asarray(test, dtype=float) * PERCENT_SCALE
    ranked, pvalue = _candidates(train_scaled, max_p, max_q, adf_alpha)
    for bic, order, fitted in ranked:
        forecast = _forecast(fitted, len(train_scaled), test_scaled)
        if _is_valid(forecast):
            return forecast, OrderSelection(order, bic, pvalue)
    raise ValueError("No converged ARIMA order gave a valid forecast")


def probability_above(mean: np.ndarray, std: np.ndarray, threshold: float) -> np.ndarray:
    """Return P(return > threshold) under the normal forecast distribution."""
    return norm.sf(threshold, loc=mean, scale=std)
