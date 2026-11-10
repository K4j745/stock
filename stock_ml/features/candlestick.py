"""Candlestick pattern features (binary).

Computes a handful of widely-used patterns from OHLC only - zero leakage.
Tries ``pandas-ta`` first (uses its TA-Lib wrappers when available), and
falls back to a manual implementation of doji, hammer, shooting star, and
bullish/bearish engulfing when the dependency is missing.
"""
import logging
from typing import List

import numpy as np
import pandas as pd

logger = logging.getLogger("stock_ml")

# 5-10 of the most popular patterns. Names match TA-Lib / pandas-ta.
DEFAULT_PATTERNS: List[str] = [
    "doji",
    "hammer",
    "invertedhammer",
    "shootingstar",
    "engulfing",
    "hangingman",
    "morningstar",
    "eveningstar",
]


def _manual_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Pure-pandas fallback for the most common candlestick patterns."""
    o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]
    body = (c - o).abs()
    rng = (h - l).replace(0, np.nan)
    upper_shadow = h - c.where(c >= o, o)
    lower_shadow = c.where(c <= o, o) - l

    prev_o = o.shift(1)
    prev_c = c.shift(1)

    out = pd.DataFrame(index=df.index)

    # Doji: tiny body relative to range
    out["cdl_doji"] = ((body / rng) < 0.1).fillna(False).astype(int)

    # Hammer: small body, long lower shadow, little upper shadow, near top
    out["cdl_hammer"] = (
        (body / rng < 0.3)
        & (lower_shadow > 2 * body)
        & (upper_shadow < body)
    ).fillna(False).astype(int)

    # Inverted hammer: small body near low, long upper shadow
    out["cdl_invertedhammer"] = (
        (body / rng < 0.3)
        & (upper_shadow > 2 * body)
        & (lower_shadow < body)
    ).fillna(False).astype(int)

    # Shooting star: like inverted hammer after an up move (we drop the
    # trend filter here - the model can pick up context from other features)
    out["cdl_shootingstar"] = out["cdl_invertedhammer"]

    # Bullish engulfing
    out["cdl_engulfing_bull"] = (
        (prev_c < prev_o)
        & (c > o)
        & (c >= prev_o)
        & (o <= prev_c)
    ).fillna(False).astype(int)

    # Bearish engulfing
    out["cdl_engulfing_bear"] = (
        (prev_c > prev_o)
        & (c < o)
        & (o >= prev_c)
        & (c <= prev_o)
    ).fillna(False).astype(int)

    # Combined engulfing signal
    out["cdl_engulfing"] = (out["cdl_engulfing_bull"] | out["cdl_engulfing_bear"]).astype(int)

    return out


def add_candlestick_patterns(df: pd.DataFrame, patterns: List[str] = None) -> pd.DataFrame:
    """Append binary candlestick pattern columns.

    Tries pandas-ta first (so we get TA-Lib's well-tested implementations
    when present); falls back to a manual implementation otherwise.
    """
    patterns = patterns or DEFAULT_PATTERNS
    df = df.copy()

    try:
        import pandas_ta as ta  # noqa: F401
        used_lib = True
    except Exception:
        used_lib = False

    if used_lib:
        try:
            cdl = df.ta.cdl_pattern(name=patterns)
            if cdl is None or cdl.empty:
                raise RuntimeError("pandas-ta returned no candlestick output")
            # pandas-ta returns -100/0/100 ints; convert to binary
            cdl_bin = (cdl != 0).astype(int)
            cdl_bin.columns = [c.lower() for c in cdl_bin.columns]
            df = pd.concat([df, cdl_bin], axis=1)
            logger.info("Added %d candlestick patterns via pandas-ta", cdl_bin.shape[1])
            return df
        except Exception as exc:
            logger.warning("pandas-ta candlestick failed (%s); using manual fallback", exc)

    manual = _manual_patterns(df)
    df = pd.concat([df, manual], axis=1)
    logger.info("Added %d candlestick patterns via manual fallback", manual.shape[1])
    return df
