"""Tests for turning model probabilities into BUY/SELL/HOLD signals."""
import numpy as np
import pandas as pd

from lib import signals


def test_missing_probability_becomes_hold_not_sell():
    index = pd.date_range("2025-01-01", periods=4, freq="B")
    probabilities = pd.Series([np.nan, 0.7, 0.3, np.nan], index=index)

    result = signals.from_probabilities(probabilities, buy_threshold=0.5, sell_threshold=0.5)

    assert result["signal"].tolist() == ["HOLD", "BUY", "SELL", "HOLD"]
    assert result["triggered_rules"].tolist()[0] == "no_prediction"
    assert result["score"].isna().tolist() == [True, False, False, True]


def test_ensemble_holds_when_no_model_has_a_prediction():
    index = pd.date_range("2025-01-01", periods=3, freq="B")
    per_model = {
        "a": signals.from_probabilities(pd.Series([np.nan, 0.8, 0.2], index=index), 0.5, 0.5),
        "b": signals.from_probabilities(pd.Series([np.nan, 0.6, 0.4], index=index), 0.5, 0.5),
    }

    result = signals.ensemble_majority(per_model)

    assert result["signal"].tolist() == ["HOLD", "BUY", "SELL"]
    assert result["triggered_rules"].tolist()[0] == "no_votes"


def test_threshold_is_inclusive_for_buy():
    index = pd.date_range("2025-01-01", periods=2, freq="B")
    probabilities = pd.Series([0.5, 0.4999], index=index)

    result = signals.from_probabilities(probabilities, buy_threshold=0.5, sell_threshold=0.5)

    assert result["signal"].tolist() == ["BUY", "SELL"]
