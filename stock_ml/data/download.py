import os
import logging

import pandas as pd
import yfinance as yf

from config import DATA_DIR, DATA_START, DATA_END, DATA_INTERVAL, ALL_TICKERS

logger = logging.getLogger("stock_ml")


def download_single(ticker: str, refresh: bool = False) -> pd.DataFrame:
    """Download OHLCV data for a single ticker, with Parquet+CSV caching."""
    parquet_path = os.path.join(DATA_DIR, f"{ticker}.parquet")
    csv_path = os.path.join(DATA_DIR, f"{ticker}.csv")

    if os.path.exists(parquet_path) and not refresh:
        logger.info("Loading cached data for %s from %s", ticker, parquet_path)
        df = pd.read_parquet(parquet_path)
    else:
        logger.info("Downloading data for %s from yfinance", ticker)
        df = yf.download(
            ticker,
            start=DATA_START,
            end=DATA_END,
            interval=DATA_INTERVAL,
            auto_adjust=True,
            progress=False,
        )

        if df.empty:
            logger.warning("No data returned for %s", ticker)
            return df

        # Flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Keep only standard OHLCV columns
        df = df[["Open", "High", "Low", "Close", "Volume"]]

        # Ensure index is named Date
        df.index.name = "Date"

        # Save cache as Parquet and CSV
        df.to_parquet(parquet_path)
        df.to_csv(csv_path)
        logger.info("Saved %s data: %d rows -> %s", ticker, len(df), parquet_path)

    logger.info("Loaded %s: %d rows, date range %s to %s",
                ticker, len(df), df.index.min(), df.index.max())
    return df


def download_all(refresh: bool = False) -> dict:
    """Download OHLCV data for all tickers."""
    data = {}
    for ticker in ALL_TICKERS:
        df = download_single(ticker, refresh=refresh)
        if not df.empty:
            data[ticker] = df
        else:
            logger.warning("Skipping %s - empty dataset", ticker)
    return data
