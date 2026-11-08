"""ARIMA forecasting core shared by the thesis baseline and the dashboard.

Kept free of the rest of ``stock_ml`` (config, data download, feature pipeline)
so the dashboard can import it with only numpy, scipy and statsmodels installed.
"""
import warnings
from typing import NamedTuple, Tuple

import numpy as np
from scipy.stats import norm
from statsmodels.tools.sm_exceptions import ConvergenceWarning, EstimationWarning
from statsmodels.tsa.arima.model import ARIMA

# Daily returns are ~0.01; fitting in percent keeps the optimiser well conditioned.
PERCENT_SCALE = 100.0


class OneStepForecast(NamedTuple):
    """One-step-ahead forecasts for a test window, in decimal return units."""

    mean: np.ndarray
    std: np.ndarray
    converged: bool


def forecast_one_step(
    train: np.ndarray,
    test: np.ndarray,
    order: Tuple[int, int, int],
) -> OneStepForecast:
    """Fit ARIMA on ``train`` and forecast each ``test`` value one step ahead.

    Parameters are estimated on ``train`` only. Each test forecast uses those
    fixed parameters and the observations before its own position. A NaN in
    ``test`` is treated as a missing observation.
    """
    train_scaled = np.asarray(train, dtype=float) * PERCENT_SCALE
    test_scaled = np.asarray(test, dtype=float) * PERCENT_SCALE

    # Non-convergence is reported through ``converged``, not as a warning per fit.
    # EstimationWarning only says statsmodels reset its starting parameters to zero.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", EstimationWarning)
        fitted = ARIMA(train_scaled, order=order).fit()

    extended = fitted.append(test_scaled, refit=False)
    prediction = extended.get_prediction(
        start=len(train_scaled),
        end=len(train_scaled) + len(test_scaled) - 1,
    )
    return OneStepForecast(
        mean=np.asarray(prediction.predicted_mean) / PERCENT_SCALE,
        std=np.asarray(prediction.se_mean) / PERCENT_SCALE,
        converged=bool(fitted.mle_retvals.get("converged", True)),
    )


def probability_above(mean: np.ndarray, std: np.ndarray, threshold: float) -> np.ndarray:
    """Return P(return > threshold) under the normal forecast distribution."""
    return norm.sf(threshold, loc=mean, scale=std)
