"""Technical indicators used by the dashboard.

Kept deliberately aligned with the indicators computed by ``stock_ml`` so the
technical "view" of a ticker is consistent between the dashboard and the ML
pipeline.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger("dashboard.indicators")


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with the standard indicator set attached.

    Adds: RSI(14), MACD/signal/hist, Bollinger Bands(20,2), SMA20, SMA50,
    EMA12, EMA26, log_return, return_1d.
    """
    import ta  # local import so the module is importable without ta installed

    out = df.copy()
    close = out["Close"]

    out["RSI"] = ta.momentum.RSIIndicator(close, window=14).rsi()

    macd = ta.trend.MACD(close)
    out["MACD"] = macd.macd()
    out["MACD_signal"] = macd.macd_signal()
    out["MACD_hist"] = macd.macd_diff()

    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    out["BB_upper"] = bb.bollinger_hband()
    out["BB_lower"] = bb.bollinger_lband()
    out["BB_middle"] = bb.bollinger_mavg()

    out["SMA20"] = ta.trend.SMAIndicator(close, window=20).sma_indicator()
    out["SMA50"] = ta.trend.SMAIndicator(close, window=50).sma_indicator()
    out["EMA12"] = ta.trend.EMAIndicator(close, window=12).ema_indicator()
    out["EMA26"] = ta.trend.EMAIndicator(close, window=26).ema_indicator()

    # Log return + simple pct return (used to label past direction and to
    # measure 1d/5d/20d outcomes after each signal).
    out["log_return"] = np.log(close / close.shift(1))
    out["return_1d"] = close.pct_change(1)
    out["return_5d"] = close.pct_change(5)
    out["return_20d"] = close.pct_change(20)

    return out
