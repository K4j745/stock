"""Tests for the Box-Jenkins steps: ADF stationarity test and BIC order choice."""
import numpy as np
import pandas as pd
import pytest

from models.arima_baseline import walk_forward_arima
from models.arima_core import adf_pvalue, forecast_auto, select_order

TRAIN_SIZE = 252
TEST_SIZE = 21
ADF_ALPHA = 0.05


def ar1(n: int, coefficient: float = 0.7, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = np.zeros(n)
    for t in range(1, n):
        values[t] = coefficient * values[t - 1] + rng.normal(0, 0.01)
    return values


def random_walk(n: int, seed: int = 5) -> np.ndarray:
    return np.cumsum(np.random.default_rng(seed).normal(0, 0.01, n))


def test_adf_rejects_unit_root_for_stationary_series():
    assert adf_pvalue(ar1(TRAIN_SIZE)) < ADF_ALPHA


def test_adf_keeps_unit_root_for_random_walk():
    assert adf_pvalue(random_walk(TRAIN_SIZE)) > ADF_ALPHA


def test_stationary_series_is_not_differenced_and_gets_an_ar_term():
    selection = select_order(ar1(TRAIN_SIZE), max_p=2, max_q=2, adf_alpha=ADF_ALPHA)

    assert selection.order[1] == 0
    assert selection.order[0] + selection.order[2] >= 1
    assert np.isfinite(selection.bic)


def test_random_walk_is_differenced_once():
    selection = select_order(random_walk(TRAIN_SIZE), max_p=2, max_q=2, adf_alpha=ADF_ALPHA)

    assert selection.order[1] == 1
    assert selection.adf_pvalue > ADF_ALPHA


def test_selected_order_has_the_lowest_bic_of_the_grid():
    from statsmodels.tsa.arima.model import ARIMA

    train = ar1(TRAIN_SIZE)
    selection = select_order(train, max_p=1, max_q=1, adf_alpha=ADF_ALPHA)
    bics = {
        (p, 0, q): ARIMA(train * 100, order=(p, 0, q)).fit().bic
        for p in range(2) for q in range(2)
    }

    assert selection.order == min(bics, key=bics.get)


def test_auto_forecast_does_not_use_future_observations():
    values = ar1(TRAIN_SIZE + TEST_SIZE)
    train, test = values[:TRAIN_SIZE], values[TRAIN_SIZE:]
    altered = test.copy()
    altered[10:] += 0.5

    original, chosen = forecast_auto(train, test, max_p=2, max_q=2, adf_alpha=ADF_ALPHA)
    changed, chosen_again = forecast_auto(train, altered, max_p=2, max_q=2, adf_alpha=ADF_ALPHA)

    assert chosen.order == chosen_again.order
    np.testing.assert_allclose(original.mean[:11], changed.mean[:11])


def test_walk_forward_reports_the_order_chosen_in_every_fold():
    returns = pd.Series(ar1(TRAIN_SIZE + 2 * TEST_SIZE), index=pd.date_range("2020-01-01", periods=TRAIN_SIZE + 2 * TEST_SIZE, freq="B"))
    labels = (returns > 0).astype(int)

    result = walk_forward_arima(
        returns, labels, threshold=0.0, train_size=TRAIN_SIZE, test_size=TEST_SIZE,
        step=TEST_SIZE, order_selection="bic",
    )

    assert len(result["order_selection"]) == 2
    first = result["order_selection"][0]
    assert set(first) == {"fold", "train_start", "train_end", "order", "bic", "adf_pvalue"}
    assert first["train_end"] == str(returns.index[TRAIN_SIZE - 1].date())


@pytest.mark.parametrize("bad", [{"max_p": -1}, {"max_q": -1}, {"adf_alpha": 0}, {"adf_alpha": 1}])
def test_rejects_invalid_search_settings(bad):
    settings = {"max_p": 2, "max_q": 2, "adf_alpha": ADF_ALPHA, **bad}
    with pytest.raises(ValueError):
        select_order(ar1(TRAIN_SIZE), **settings)


def test_auto_forecast_skips_a_best_bic_model_with_an_invalid_forecast(monkeypatch):
    from models import arima_core

    real_forecast = arima_core._forecast
    calls = []

    def degenerate_first(fitted, train_len, test_scaled):
        forecast = real_forecast(fitted, train_len, test_scaled)
        calls.append(fitted.model.order)
        if len(calls) == 1:
            return forecast._replace(std=np.zeros_like(forecast.std))
        return forecast

    monkeypatch.setattr(arima_core, "_forecast", degenerate_first)
    values = ar1(TRAIN_SIZE + TEST_SIZE)

    forecast, chosen = forecast_auto(values[:TRAIN_SIZE], values[TRAIN_SIZE:], max_p=1, max_q=1, adf_alpha=ADF_ALPHA)

    assert len(calls) == 2
    assert chosen.order == calls[1]
    assert np.all(forecast.std > 0)


def test_selection_ignores_fits_that_did_not_converge(monkeypatch):
    from models import arima_core

    real_fit = arima_core._fit

    def unconverged_ar1(series, order):
        fitted = real_fit(series, order)
        if order == (1, 0, 0):
            fitted.mle_retvals["converged"] = False
        return fitted

    monkeypatch.setattr(arima_core, "_fit", unconverged_ar1)

    selection = select_order(ar1(TRAIN_SIZE), max_p=1, max_q=1, adf_alpha=ADF_ALPHA)

    assert selection.order != (1, 0, 0)
