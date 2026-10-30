import logging

import pandas as pd

logger = logging.getLogger("stock_ml")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean OHLCV data: handle NaN, duplicates, validate types."""
    initial_rows = len(df)

    # Remove duplicate dates
    duplicates = df.index.duplicated(keep="first")
    if duplicates.any():
        n_dups = duplicates.sum()
        df = df[~duplicates]
        logger.info("Removed %d duplicate date rows", n_dups)

    # Handle NaN: forward fill -> backward fill -> drop remaining
    nan_before = df.isna().sum().sum()
    df = df.ffill()
    df = df.bfill()
    nan_after = df.isna().sum().sum()
    if nan_before > 0:
        logger.info("Filled %d NaN values (ffill+bfill)", nan_before - nan_after)

    # Drop any remaining NaN rows
    rows_before_drop = len(df)
    df = df.dropna()
    dropped = rows_before_drop - len(df)
    if dropped > 0:
        logger.info("Dropped %d rows with remaining NaN", dropped)

    # Validate column types (ensure numeric)
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Remove rows with negative prices or volumes
    mask = (
        (df["Open"] >= 0)
        & (df["High"] >= 0)
        & (df["Low"] >= 0)
        & (df["Close"] >= 0)
        & (df["Volume"] >= 0)
    )
    invalid_rows = (~mask).sum()
    if invalid_rows > 0:
        df = df[mask]
        logger.info("Removed %d rows with negative prices/volumes", invalid_rows)

    final_rows = len(df)
    logger.info("Preprocessing complete: %d -> %d rows (%d removed)",
                initial_rows, final_rows, initial_rows - final_rows)
    return df
