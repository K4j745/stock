"""Historical OHLCV backup downloader.

Downloads the full daily history for every ticker in the StocKK Forecast universe
(plus the SPY benchmark) and writes one committed CSV per ticker into
``data/historical/``. Unlike ``data/download.py`` (which caches into the
gitignored ``data/raw/`` for training), these CSVs are **committed to the repo**
so the thesis stays reproducible even if the yfinance feed changes or goes
offline.

New rows are **merged** into the existing CSV rather than overwriting it, so a
scheduled CI run can keep extending the archive (and a short download window is
enough) without ever truncating history. Pass ``--full-refresh`` to replace the
file instead.

Run:
    python data/download_historical.py                # all tickers
    python data/download_historical.py --ticker AAPL  # single ticker
    python data/download_historical.py --start 2015-01-01
    python data/download_historical.py --full-refresh # ignore existing CSVs
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


def _read_existing(path: str) -> pd.DataFrame | None:
    """Load an already-archived CSV, or ``None`` if it is absent/unreadable."""
    if not os.path.exists(path):
        return None
    try:
        existing = pd.read_csv(path, index_col=0, parse_dates=True)
    except Exception as exc:
        logger.warning("Could not read existing archive %s: %s", path, exc)
        return None
    if existing.empty:
        return None
    existing.index.name = "Date"
    return existing


def _merge(existing: pd.DataFrame, fresh: pd.DataFrame) -> pd.DataFrame:
    """Combine archive and fresh rows, preferring the freshly downloaded ones."""
    combined = pd.concat([existing, fresh])
    combined = combined[~combined.index.duplicated(keep="last")]
    return combined.sort_index()


def download_historical_single(ticker: str, start: str = DATA_START, end: str | None = None,
                               full_refresh: bool = False) -> pd.DataFrame:
    """Download the daily history for one ticker and save it as a CSV.

    Unless ``full_refresh`` is set, the download is merged into any existing
    archive so previously captured rows survive a short download window.

    Returns the saved DataFrame (indexed by Date), or an empty DataFrame when
    yfinance returns nothing and no archive exists.
    """
    out_path = os.path.join(HISTORICAL_DIR, f"{ticker}_historical.csv")
    existing = None if full_refresh else _read_existing(out_path)

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
        if existing is None:
            logger.warning("No data returned for %s - skipping", ticker)
            return df
        logger.warning("No data returned for %s - keeping existing archive intact", ticker)
        return existing

    # Flatten the (field, ticker) MultiIndex yfinance returns for single symbols.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    missing = [c for c in CSV_COLUMNS if c not in df.columns]
    if missing:
        logger.warning("%s missing columns %s - keeping what is available", ticker, missing)
    df = df[[c for c in CSV_COLUMNS if c in df.columns]]
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"

    if existing is not None:
        added = len(df.index.difference(existing.index))
        df = _merge(existing, df)
        logger.info("Merged %s: %d new rows into %d archived", ticker, added, len(existing))

    df.to_csv(out_path)
    logger.info(
        "Saved %s: %d rows (%s to %s) -> %s",
        ticker, len(df), df.index.min().date(), df.index.max().date(), out_path,
    )
    return df


def download_all_historical(tickers: list[str] | None = None, start: str = DATA_START,
                            end: str | None = None, full_refresh: bool = False) -> dict:
    """Download and persist historical CSVs for every requested ticker."""
    tickers = tickers or ALL_TICKERS
    saved = {}
    for ticker in tickers:
        df = download_historical_single(ticker, start=start, end=end, full_refresh=full_refresh)
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
    parser.add_argument("--full-refresh", action="store_true",
                        help="Overwrite the archive instead of merging into it")
    args = parser.parse_args()

    tickers = [args.ticker] if args.ticker else None
    download_all_historical(tickers=tickers, start=args.start, end=args.end,
                            full_refresh=args.full_refresh)


if __name__ == "__main__":
    main()
