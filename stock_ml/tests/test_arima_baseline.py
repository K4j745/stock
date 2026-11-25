"""Tests for the ARIMA baseline (one-step forecasts and walk-forward scoring)."""
import numpy as np
import pandas as pd
import pytest

from models.arima_baseline import (
    forecast_one_step,
    probability_above,
    walk_forward_arima,
)

TRAIN_SIZE = 200
TEST_SIZE = 20
AR_COEFFICIENT = 0.8
NOISE_STD = 0.01


def make_ar1_returns(n: int, seed: int = 42) -> pd.Series:
    """Build a strongly autocorrelated return series that ARIMA can predict."""
    rng = np.random.default_rng(seed)
    values = np.zeros(n)
    for t in range(1, n):
        values[t] = AR_COEFFICIENT * values[t - 1] + rng.normal(0, NOISE_STD)
    index = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.Series(values, index=index)


def test_forecast_returns_one_value_per_test_observation():
    returns = make_ar1_returns(TRAIN_SIZE + TEST_SIZE).to_numpy()

    forecast = forecast_one_step(returns[:TRAIN_SIZE], returns[TRAIN_SIZE:])

    assert forecast.mean.shape == (TEST_SIZE,)
    assert forecast.std.shape == (TEST_SIZE,)
    assert np.all(forecast.std > 0)


def test_forecast_does_not_use_future_observations():
    returns = make_ar1_returns(TRAIN_SIZE + TEST_SIZE).to_numpy()
    train, test = returns[:TRAIN_SIZE], returns[TRAIN_SIZE:]
    altered_test = test.copy()
    altered_test[10:] += 0.5

    original = forecast_one_step(train, test)
    altered = forecast_one_step(train, altered_test)

    # Forecast k depends only on observations before k, so changing the test
    # series from position 10 onward may only affect forecasts after position 10.
    np.testing.assert_allclose(original.mean[:11], altered.mean[:11])
    assert not np.allclose(original.mean[11:], altered.mean[11:])


def test_forecast_tracks_predictable_series():
    returns = make_ar1_returns(TRAIN_SIZE + 100).to_numpy()
    train, test = returns[:TRAIN_SIZE], returns[TRAIN_SIZE:]

    forecast = forecast_one_step(train, test)

    assert np.corrcoef(forecast.mean, test)[0, 1] > 0.5


def test_probability_above_is_half_when_forecast_equals_threshold():
    probability = probability_above(np.array([0.002]), np.array([0.01]), threshold=0.002)

    assert probability[0] == pytest.approx(0.5)


def test_probability_above_grows_with_forecast_mean():
    means = np.array([-0.01, 0.0, 0.01])

    probabilities = probability_above(means, np.full(3, 0.01), threshold=0.0)

    assert np.all(np.diff(probabilities) > 0)


def test_walk_forward_returns_metrics_for_every_fold():
    returns = make_ar1_returns(TRAIN_SIZE + 3 * TEST_SIZE)
    labels = (returns > 0).astype(int)

    result = walk_forward_arima(
        returns, labels, threshold=0.0,
        train_size=TRAIN_SIZE, test_size=TEST_SIZE, step=TEST_SIZE,
    )

    assert len(result["fold_metrics"]) == 3
    assert set(result["mean_metrics"]) == {
        "accuracy", "precision", "recall", "f1", "mcc", "roc_auc",
    }
    assert result["mean_metrics"]["accuracy"] > 0.6


def test_walk_forward_rejects_misaligned_inputs():
    returns = make_ar1_returns(TRAIN_SIZE + TEST_SIZE)
    labels = (returns > 0).astype(int).iloc[:-1]

    with pytest.raises(ValueError, match="aligned"):
        walk_forward_arima(returns, labels, threshold=0.0)
