import logging

import pandas as pd

from config import DEFAULT_LABEL_MODE
from data.download import download_single
from data.preprocess import clean_data
from features.indicators import add_technical_indicators
from features.labels import (
    create_binary_labels,
    create_multiclass_labels,
    create_labels_version_a,
    create_labels_version_b,
)

logger = logging.getLogger("stock_ml")

# Columns to exclude from feature matrix (raw OHLCV data + leakage columns)
OHLCV_COLS = ["Open", "High", "Low", "Close", "Volume"]

# Log_return = log(Close_t / Close_t-1), which is essentially the same as
# pct_change used in label creation (shifted by -1). Keeping it would cause
# the model to see the future return indirectly -> data leakage -> acc=1.0.
# Only LAGGED returns (Return_lag_1 .. Return_lag_10) are safe features.
LEAKAGE_COLS = ["Log_return"]


def build_feature_matrix(
    ticker: str,
    label_version: str = "A",
    refresh: bool = False,
    label_mode: str = DEFAULT_LABEL_MODE,
) -> tuple:
    """Build feature matrix X and labels y for a given ticker.

    Steps:
    1. Load data from cache (or download)
    2. Clean data
    3. Add technical indicators (+ VIX, earnings dummy, candlestick patterns)
    4. Add labels (binary by default; legacy A/B or multiclass available)
    5. Remove NaN rows
    6. Split into X (features only) and y (labels)

    Args:
        ticker: Stock symbol.
        label_version: 'A' or 'B' for legacy labels. Ignored when label_mode is
            'binary' (default) or 'multiclass'; the LABEL_THRESHOLD from config
            is used instead.
        refresh: Force re-download from yfinance.
        label_mode: 'binary' (default) or 'multiclass'. 'legacy' will fall back
            to label_version A/B for backwards compatibility.

    Returns:
        Tuple of (X, y) with the date index preserved.
    """
    logger.info(
        "Building feature matrix for %s (label_mode=%s, label_version=%s)",
        ticker, label_mode, label_version,
    )

    # 1. Load data
    df = download_single(ticker, refresh=refresh)
    if df.empty:
        raise ValueError(f"No data available for ticker {ticker}")

    # 2. Clean data
    df = clean_data(df)

    # 3. Add technical indicators (incl. external & candlestick if available)
    df = add_technical_indicators(df, ticker=ticker)

    # 4. Add labels
    if label_mode == "binary":
        df["label"] = create_binary_labels(df)
    elif label_mode == "multiclass":
        df["label"] = create_multiclass_labels(df)
    elif label_mode == "legacy":
        if label_version == "A":
            df["label"] = create_labels_version_a(df)
        elif label_version == "B":
            df["label"] = create_labels_version_b(df)
        else:
            raise ValueError(f"Unknown legacy label version: {label_version}. Use 'A' or 'B'.")
    else:
        raise ValueError(
            f"Unknown label_mode '{label_mode}'. Use 'binary', 'multiclass', or 'legacy'."
        )

    # 5. Remove NaN rows (last row will be NaN due to shift(-1) in labels)
    df = df.dropna()

    # 6. Split into X and y - drop OHLCV, label, and leakage columns
    drop_cols = OHLCV_COLS + LEAKAGE_COLS + ["label"]
    y = df["label"].astype(int)
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])

    pos_pct = 100 * (y == 1).mean()
    logger.info(
        "Feature matrix for %s: X shape %s, y shape %s, positive class %.1f%%",
        ticker, X.shape, y.shape, pos_pct,
    )

    return X, y
