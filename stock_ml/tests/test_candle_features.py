"""Unit tests for candlestick-shape features.

Uses small, hand-built OHLC frames so every expected value can be computed by
hand — no network, no randomness.
"""
import numpy as np
import pandas as pd
import pytest

from features.candle_features import (
    CANDLE_FEATURE_COLUMNS,
    DOJI_BODY_THRESHOLD,
    compute_candle_features,
)


def _frame(rows):
    idx = pd.date_range("2020-01-01", periods=len(rows), freq="D")
    return pd.DataFrame(rows, columns=["Open", "High", "Low", "Close"], index=idx)


def test_returns_all_expected_columns_and_shape():
    df = _frame([[100, 110, 90, 105], [105, 108, 100, 101]])
    out = compute_candle_features(df)
    assert list(out.columns) == CANDLE_FEATURE_COLUMNS
    assert len(out) == len(df)
    assert out.index.equals(df.index)


def test_body_and_shadow_math():
    # Open 100, High 110, Low 90, Close 104 -> bullish candle.
    df = _frame([[100, 110, 90, 104]])
    out = compute_candle_features(df).iloc[0]
    # body = |104-100| / 100 = 0.04
    assert out["body_length"] == pytest.approx(0.04)
    # upper shadow = (110 - max(100,104)) / 100 = 6/100
    assert out["upper_shadow"] == pytest.approx(0.06)
    # lower shadow = (min(100,104) - 90) / 100 = 10/100
    assert out["lower_shadow"] == pytest.approx(0.10)
    # close position = (104 - 90) / (110 - 90) = 14/20 = 0.7
    assert out["close_position"] == pytest.approx(0.7)


def test_doji_flag_triggers_on_tiny_body():
    # Body of 0.0005 * open < DOJI threshold (0.001).
    df = _frame([[100, 101, 99, 100.05]])
    out = compute_candle_features(df).iloc[0]
    assert 0.0005 < DOJI_BODY_THRESHOLD
    assert out["doji_flag"] == 1


def test_long_lower_shadow_flag():
    # Small body (1), long lower shadow (10) -> long_lower_shadow_flag = 1.
    df = _frame([[100, 101, 89, 101]])
    out = compute_candle_features(df).iloc[0]
    assert out["long_lower_shadow_flag"] == 1
    assert out["long_upper_shadow_flag"] == 0


def test_bullish_engulfing_detection():
    # Day 1: bearish (open 105, close 100). Day 2: bullish body 99->106 engulfs it.
    df = _frame([[105, 106, 99, 100], [99, 107, 98, 106]])
    out = compute_candle_features(df)
    assert out["bullish_engulfing"].iloc[0] == 0  # no previous candle
    assert out["bullish_engulfing"].iloc[1] == 1
    assert out["bearish_engulfing"].iloc[1] == 0


def test_bearish_engulfing_detection():
    # Day 1: bullish (100->105). Day 2: bearish body 106->99 engulfs it.
    df = _frame([[100, 106, 99, 105], [106, 107, 98, 99]])
    out = compute_candle_features(df)
    assert out["bearish_engulfing"].iloc[1] == 1
    assert out["bullish_engulfing"].iloc[1] == 0


def test_flat_range_does_not_produce_nan():
    # High == Low (zero range) must not yield NaN/inf in close_position.
    df = _frame([[100, 100, 100, 100]])
    out = compute_candle_features(df).iloc[0]
    assert np.isfinite(out["close_position"])
    assert out["close_position"] == pytest.approx(0.5)


def test_missing_columns_raise():
    bad = pd.DataFrame({"Open": [1], "High": [2]})
    with pytest.raises(ValueError):
        compute_candle_features(bad)
