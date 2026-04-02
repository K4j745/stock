import logging

import numpy as np
import pandas as pd
import ta

logger = logging.getLogger("stock_ml")


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add all technical indicators to OHLCV DataFrame.

    Uses the `ta` library for standard indicators and manual calculations
    for price-based features. Drops rows with NaN after indicator computation.
    """
    df = df.copy()

    # --- Trend indicators ---
    # SMA
    df["SMA_5"] = ta.trend.SMAIndicator(close=df["Close"], window=5).sma_indicator()
    df["SMA_10"] = ta.trend.SMAIndicator(close=df["Close"], window=10).sma_indicator()
    df["SMA_20"] = ta.trend.SMAIndicator(close=df["Close"], window=20).sma_indicator()
    df["SMA_50"] = ta.trend.SMAIndicator(close=df["Close"], window=50).sma_indicator()

    # EMA
    df["EMA_12"] = ta.trend.EMAIndicator(close=df["Close"], window=12).ema_indicator()
    df["EMA_26"] = ta.trend.EMAIndicator(close=df["Close"], window=26).ema_indicator()

    # MACD
    macd = ta.trend.MACD(close=df["Close"])
    df["MACD"] = macd.macd()
    df["MACD_signal"] = macd.macd_signal()
    df["MACD_hist"] = macd.macd_diff()

    # --- Momentum indicators ---
    # RSI
    df["RSI_14"] = ta.momentum.RSIIndicator(close=df["Close"], window=14).rsi()

    # Stochastic Oscillator
    stoch = ta.momentum.StochasticOscillator(
        high=df["High"], low=df["Low"], close=df["Close"]
    )
    df["Stoch_K"] = stoch.stoch()
    df["Stoch_D"] = stoch.stoch_signal()

    # Rate of Change
    df["ROC_5"] = ta.momentum.ROCIndicator(close=df["Close"], window=5).roc()
    df["ROC_10"] = ta.momentum.ROCIndicator(close=df["Close"], window=10).roc()

    # --- Volatility indicators ---
    # Bollinger Bands
    bb = ta.volatility.BollingerBands(close=df["Close"])
    df["BB_upper"] = bb.bollinger_hband()
    df["BB_middle"] = bb.bollinger_mavg()
    df["BB_lower"] = bb.bollinger_lband()
    df["BB_width"] = bb.bollinger_wband()

    # ATR
    df["ATR_14"] = ta.volatility.AverageTrueRange(
        high=df["High"], low=df["Low"], close=df["Close"], window=14
    ).average_true_range()

    # --- Volume indicators ---
    df["OBV"] = ta.volume.OnBalanceVolumeIndicator(
        close=df["Close"], volume=df["Volume"]
    ).on_balance_volume()

    df["Volume_ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()

    # --- Price-based features (manual calculation) ---
    df["Log_return"] = np.log(df["Close"] / df["Close"].shift(1))

    # Lagged returns
    df["Return_lag_1"] = df["Log_return"].shift(1)
    df["Return_lag_2"] = df["Log_return"].shift(2)
    df["Return_lag_3"] = df["Log_return"].shift(3)
    df["Return_lag_5"] = df["Log_return"].shift(5)
    df["Return_lag_10"] = df["Log_return"].shift(10)

    # High-Low spread
    df["HL_spread"] = (df["High"] - df["Low"]) / df["Close"]

    # Drop rows with NaN from indicator warm-up periods
    rows_before = len(df)
    df = df.dropna()
    rows_after = len(df)
    logger.info("Added technical indicators: dropped %d warm-up rows (%d -> %d)",
                rows_before - rows_after, rows_before, rows_after)

    return df
