"""Tests for the rolling ARIMA signal stream used by the dashboard."""
import numpy as np
import pandas as pd
import pytest

from lib import arima_signals

THRESHOLD = 0.002
SETTINGS = {"order": [1, 0, 1], "train_window": 120, "min_train": 60, "refit_every": 21}
SESSIONS = 200
AR_COEFFICIENT = 0.6
NOISE_STD = 0.01


def make_close(n: int = SESSIONS, seed: int = 7) -> pd.Series:
    """Prices whose daily returns follow an AR(1), so ARIMA has something to find."""
    rng = np.random.default_rng(seed)
    returns = np.zeros(n)
    for t in range(1, n):
        returns[t] = AR_COEFFICIENT * returns[t - 1] + rng.normal(0, NOISE_STD)
    index = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.Series(100 * np.cumprod(1 + returns), index=index)


def test_probabilities_are_aligned_and_warm_up_with_nan():
    close = make_close()

    probabilities = arima_signals.arima_probabilities(close, THRESHOLD, SETTINGS, "TEST")

    assert probabilities.index.equals(close.index)
    assert probabilities.iloc[:SETTINGS["min_train"]].isna().all()
    assert probabilities.iloc[SETTINGS["min_train"]:].notna().all()
    assert probabilities.dropna().between(0, 1).all()


def test_probabilities_do_not_use_future_prices():
    close = make_close()
    cutoff = 150
    altered = close.copy()
    altered.iloc[cutoff:] *= 1.25

    original = arima_signals.arima_probabilities(close, THRESHOLD, SETTINGS, "TEST")
    changed = arima_signals.arima_probabilities(altered, THRESHOLD, SETTINGS, "TEST")

    # A price change on day `cutoff` first alters the return known at the close
    # of that day, so every forecast made before it must be unchanged.
    pd.testing.assert_series_equal(original.iloc[:cutoff - 1], changed.iloc[:cutoff - 1])


def test_signals_hold_during_warm_up_then_follow_the_threshold():
    close = make_close()
    probabilities = arima_signals.arima_probabilities(close, THRESHOLD, SETTINGS, "TEST")

    signals = arima_signals.to_signals(probabilities, buy_threshold=0.5)

    warm_up = signals.iloc[:SETTINGS["min_train"]]
    assert (warm_up["signal"] == "HOLD").all()
    assert (warm_up["triggered_rules"] == "arima_warmup").all()
    active = signals.iloc[SETTINGS["min_train"]:]
    expected = np.where(probabilities.iloc[SETTINGS["min_train"]:] >= 0.5, "BUY", "SELL")
    assert (active["signal"].to_numpy() == expected).all()


def test_series_shorter_than_min_train_stays_in_warm_up():
    close = make_close(n=SETTINGS["min_train"])

    probabilities = arima_signals.arima_probabilities(close, THRESHOLD, SETTINGS, "TEST")

    assert probabilities.isna().all()


@pytest.mark.parametrize("bad", [{"min_train": 0}, {"refit_every": 0}, {"train_window": 10, "min_train": 60}])
def test_rejects_invalid_settings(bad):
    with pytest.raises(ValueError):
        arima_signals.arima_probabilities(make_close(), THRESHOLD, {**SETTINGS, **bad}, "TEST")
