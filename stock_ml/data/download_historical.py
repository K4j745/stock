"""Historical OHLCV backup downloader.

Downloads the full daily history for every ticker in the Stock4caster universe
(plus the SPY benchmark) and writes one committed CSV per ticker into
``data/historical/``. Unlike ``data/download.py`` (which caches into the
gitignored ``data/raw/`` for training), these CSVs are **committed to the repo**
so the thesis stays reproducible even if the yfinance feed changes or goes
offline.

Run:
    python data/download_historical.py                # all tickers
    python data/download_historical.py --ticker AAPL  # single ticker
    python data/download_historical.py --start 2015-01-01
"""
import argparse
import logging
import os
import sys

import pandas as pd
import yfinance as yf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ALL_TICKERS, DATA_INTERVAL, DATA_START, HISTORICAL_DIR

logger = logging.getLogger("stock_ml")

# Column order written to each CSV. auto_adjust=False keeps raw Close AND the
# split/dividend-adjusted Adj Close so downstream users can pick either.
CSV_COLUMNS = ["Open", "High", "Low", "Close", "Volume", "Adj Close"]


def download_historical_single(ticker: str, start: str = DATA_START, end: str | None = None) -> pd.DataFrame:
    """Download the full daily history for one ticker and save it as a CSV.

    Returns the saved DataFrame (indexed by Date), or an empty DataFrame when
    yfinance returns nothing.
    """
    logger.info("Downloading historical data for %s (start=%s)", ticker, start)
    df = yf.download(
        ticker,
        start=start,
        end=end,
        interval=DATA_INTERVAL,
        auto_adjust=False,
        progress=False,
    )

    if df.empty:
        logger.warning("No data returned for %s - skipping", ticker)
        return df

    # Flatten the (field, ticker) MultiIndex yfinance returns for single symbols.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    missing = [c for c in CSV_COLUMNS if c not in df.columns]
    if missing:
        logger.warning("%s missing columns %s - keeping what is available", ticker, missing)
    df = df[[c for c in CSV_COLUMNS if c in df.columns]]
    df.index.name = "Date"

    out_path = os.path.join(HISTORICAL_DIR, f"{ticker}_historical.csv")
    df.to_csv(out_path)
    logger.info(
        "Saved %s: %d rows (%s to %s) -> %s",
        ticker, len(df), df.index.min().date(), df.index.max().date(), out_path,
    )
    return df


def download_all_historical(tickers: list[str] | None = None, start: str = DATA_START,
                            end: str | None = None) -> dict:
    """Download and persist historical CSVs for every requested ticker."""
    tickers = tickers or ALL_TICKERS
    saved = {}
    for ticker in tickers:
        df = download_historical_single(ticker, start=start, end=end)
        if not df.empty:
            saved[ticker] = len(df)
    logger.info("Historical backup complete: %d/%d tickers saved to %s",
                len(saved), len(tickers), HISTORICAL_DIR)
    return saved


def main():
    parser = argparse.ArgumentParser(description="Download committed historical OHLCV CSV backups")
    parser.add_argument("--ticker", type=str, default=None,
                        help="Single ticker (default: all from config)")
    parser.add_argument("--start", type=str, default=DATA_START,
                        help=f"Start date YYYY-MM-DD (default: {DATA_START})")
    parser.add_argument("--end", type=str, default=None,
                        help="End date YYYY-MM-DD (default: today)")
    args = parser.parse_args()

    tickers = [args.ticker] if args.ticker else None
    download_all_historical(tickers=tickers, start=args.start, end=args.end)


if __name__ == "__main__":
    main()
