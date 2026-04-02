import logging

import pandas as pd

from data.download import download_single
from data.preprocess import clean_data
from features.indicators import add_technical_indicators
from features.labels import create_labels_version_a, create_labels_version_b

logger = logging.getLogger("stock_ml")

# Columns to exclude from feature matrix (raw OHLCV data)
OHLCV_COLS = ["Open", "High", "Low", "Close", "Volume"]


def build_feature_matrix(
    ticker: str, label_version: str = "A", refresh: bool = False
) -> tuple:
    """Build feature matrix X and labels y for a given ticker.

    Steps:
        1. Load data from cache (or download)
        2. Clean data
        3. Add technical indicators
        4. Add labels (version A or B)
        5. Remove NaN rows
        6. Split into X (features only) and y (labels)

    Returns:
        tuple of (X: pd.DataFrame, y: pd.Series) with date index preserved.
    """
    logger.info("Building feature matrix for %s (label version %s)", ticker, label_version)

    # 1. Load data
    df = download_single(ticker, refresh=refresh)
    if df.empty:
        raise ValueError(f"No data available for ticker {ticker}")

    # 2. Clean data
    df = clean_data(df)

    # 3. Add technical indicators
    df = add_technical_indicators(df)

    # 4. Add labels
    if label_version == "A":
        df["label"] = create_labels_version_a(df)
    elif label_version == "B":
        df["label"] = create_labels_version_b(df)
    else:
        raise ValueError(f"Unknown label version: {label_version}. Use 'A' or 'B'.")

    # 5. Remove NaN rows (last row will be NaN due to shift(-1) in labels)
    df = df.dropna()

    # 6. Split into X and y
    y = df["label"].astype(int)
    X = df.drop(columns=OHLCV_COLS + ["label"])

    logger.info("Feature matrix for %s: X shape %s, y shape %s, positive class %.1f%%",
                ticker, X.shape, y.shape, 100 * y.mean())

    return X, y
