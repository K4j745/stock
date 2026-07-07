"""Candlestick (OHLC-shape) features.

These features describe the *geometry* of each daily candle — the size of the
body, the length of the shadows, where the close sits inside the day's range —
plus a few classic single/two-candle formation flags (doji, long shadows,
bullish/bearish engulfing).

All continuous features are normalised by the open price so they are
comparable across tickers with very different absolute price levels. The public
entry point is :func:`compute_candle_features`.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger("stock_ml")

# A candle whose body is smaller than this fraction of the open is "doji-like".
DOJI_BODY_THRESHOLD = 0.001
# A shadow longer than this multiple of the body counts as a "long" shadow.
LONG_SHADOW_BODY_RATIO = 2.0

CANDLE_FEATURE_COLUMNS = [
    "body_length",
    "upper_shadow",
    "lower_shadow",
    "close_position",
    "doji_flag",
    "long_lower_shadow_flag",
    "long_upper_shadow_flag",
    "bullish_engulfing",
    "bearish_engulfing",
]


def compute_candle_features(df_ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Compute candlestick-shape features from an OHLCV DataFrame.

    Args:
        df_ohlcv: DataFrame with at least ``Open``, ``High``, ``Low`` and
            ``Close`` columns, indexed by date.

    Returns:
        DataFrame indexed identically to the input, containing the columns in
        :data:`CANDLE_FEATURE_COLUMNS`. No rows are dropped; the engulfing
        flags are ``0`` on the first row (no previous candle to compare).
    """
    required = {"Open", "High", "Low", "Close"}
    missing = required - set(df_ohlcv.columns)
    if missing:
        raise ValueError(f"compute_candle_features requires columns {sorted(missing)}")

    o = df_ohlcv["Open"].astype(float)
    h = df_ohlcv["High"].astype(float)
    low = df_ohlcv["Low"].astype(float)
    c = df_ohlcv["Close"].astype(float)

    # Guard against non-positive opens (should never happen for equities, but a
    # single bad tick shouldn't blow up the whole pipeline with inf/NaN).
    safe_open = o.replace(0.0, np.nan)
    day_range = (h - low).replace(0.0, np.nan)

    body = (c - o).abs()
    body_length = body / safe_open
    upper_shadow = (h - np.maximum(o, c)) / safe_open
    lower_shadow = (np.minimum(o, c) - low) / safe_open
    # Where the close sits within the day's range: 0 = at the low, 1 = at the high.
    close_position = (c - low) / day_range

    doji_flag = (body_length < DOJI_BODY_THRESHOLD).astype(int)
    long_lower_shadow_flag = (lower_shadow > LONG_SHADOW_BODY_RATIO * body_length).astype(int)
    long_upper_shadow_flag = (upper_shadow > LONG_SHADOW_BODY_RATIO * body_length).astype(int)

    prev_o = o.shift(1)
    prev_c = c.shift(1)
    prev_bearish = prev_c < prev_o
    prev_bullish = prev_c > prev_o
    cur_bullish = c > o
    cur_bearish = c < o

    # Bullish engulfing: today's up-body fully engulfs yesterday's down-body.
    bullish_engulfing = (
        cur_bullish & prev_bearish & (c >= prev_o) & (o <= prev_c)
    ).astype(int)
    # Bearish engulfing: today's down-body fully engulfs yesterday's up-body.
    bearish_engulfing = (
        cur_bearish & prev_bullish & (o >= prev_c) & (c <= prev_o)
    ).astype(int)

    out = pd.DataFrame(
        {
            "body_length": body_length,
            "upper_shadow": upper_shadow.clip(lower=0),
            "lower_shadow": lower_shadow.clip(lower=0),
            "close_position": close_position.fillna(0.5).clip(0, 1),
            "doji_flag": doji_flag,
            "long_lower_shadow_flag": long_lower_shadow_flag,
            "long_upper_shadow_flag": long_upper_shadow_flag,
            "bullish_engulfing": bullish_engulfing,
            "bearish_engulfing": bearish_engulfing,
        },
        index=df_ohlcv.index,
    )

    # Continuous features can still be NaN if the open was non-positive; fill the
    # shape ratios with 0 so downstream models get a clean, finite matrix.
    for col in ("body_length", "upper_shadow", "lower_shadow"):
        out[col] = out[col].fillna(0.0)

    logger.info(
        "Candle features: %d rows, %d bullish / %d bearish engulfing, %d doji",
        len(out),
        int(out["bullish_engulfing"].sum()),
        int(out["bearish_engulfing"].sum()),
        int(out["doji_flag"].sum()),
    )
    return out[CANDLE_FEATURE_COLUMNS]
