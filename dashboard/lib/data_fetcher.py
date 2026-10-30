"""yfinance wrapper with a tiny in-memory cache and an offline CSV fallback.

The generator is run from CI on a schedule, so persistent caching would just
get stale. We only cache within a single ``generate.py`` run so that the
benchmark series is not downloaded once per portfolio.

Because the upstream feed is an unofficial, rate-limited endpoint, a download
is retried with exponential backoff and — if it still fails or comes back
malformed — the committed historical CSV archive is used instead. That keeps
the dashboard buildable even when the external source is unreachable.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Dict, Optional

import pandas as pd

logger = logging.getLogger("dashboard.data")


_cache: Dict[str, pd.DataFrame] = {}

_LIB_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_LIB_DIR))
HISTORICAL_DIR = os.path.join(_REPO_ROOT, "stock_ml", "data", "historical")

REQUIRED_COLUMNS = ("Open", "High", "Low", "Close", "Volume")

MAX_ATTEMPTS = 3
BACKOFF_BASE_SECONDS = 2.0

# Approximate calendar days per ``period`` string, used to trim the CSV
# fallback to the same window the caller asked yfinance for.
_PERIOD_DAYS = {
    "1mo": 31, "3mo": 92, "6mo": 183,
    "1y": 365, "2y": 730, "3y": 1095, "5y": 1825, "10y": 3650,
}


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten yfinance's single-symbol MultiIndex and sort by date."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def _is_valid(df: Optional[pd.DataFrame], symbol: str) -> bool:
    """Reject empty frames, missing OHLCV columns and all-NaN closes."""
    if df is None or df.empty:
        logger.warning("No data returned for %s", symbol)
        return False
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        logger.warning("%s is missing required columns %s", symbol, missing)
        return False
    if df["Close"].isna().all():
        logger.warning("%s returned only NaN closing prices", symbol)
        return False
    return True


def _download_from_yfinance(symbol: str, period: str, interval: str) -> Optional[pd.DataFrame]:
    """Fetch from yfinance, retrying transient failures with backoff."""
    try:
        import yfinance as yf  # local import keeps unit tests light
    except ImportError as exc:  # pragma: no cover
        logger.error("yfinance not available: %s", exc)
        return None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        logger.info("Downloading %s (%s %s), attempt %d/%d…",
                    symbol, period, interval, attempt, MAX_ATTEMPTS)
        try:
            df = yf.download(symbol, period=period, interval=interval,
                             progress=False, auto_adjust=False)
        except Exception as exc:  # pragma: no cover — network failures
            logger.warning("yfinance download failed for %s: %s", symbol, exc)
            df = None

        if df is not None and not df.empty:
            df = _normalize(df)
            if _is_valid(df, symbol):
                return df

        if attempt < MAX_ATTEMPTS:
            time.sleep(BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)))

    logger.error("yfinance exhausted %d attempts for %s", MAX_ATTEMPTS, symbol)
    return None


def _load_from_archive(symbol: str, period: str) -> Optional[pd.DataFrame]:
    """Read the committed historical CSV, trimmed to roughly ``period``."""
    path = os.path.join(HISTORICAL_DIR, f"{symbol}_historical.csv")
    if not os.path.exists(path):
        logger.error("No archived CSV for %s at %s", symbol, path)
        return None

    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
    except Exception as exc:
        logger.error("Could not read archived CSV for %s: %s", symbol, exc)
        return None

    df = _normalize(df)
    if not _is_valid(df, symbol):
        return None

    days = _PERIOD_DAYS.get(period)
    if days and not df.empty:
        df = df[df.index >= df.index.max() - pd.Timedelta(days=days)]

    logger.warning("Using offline archive for %s: %d rows up to %s",
                   symbol, len(df), df.index.max().date())
    return df


def download(symbol: str, period: str = "3y", interval: str = "1d") -> Optional[pd.DataFrame]:
    """Return OHLCV for ``symbol``, falling back to the CSV archive.

    Returns ``None`` only when both the live feed and the archive fail.
    """
    key = f"{symbol}|{period}|{interval}"
    if key in _cache:
        return _cache[key].copy()

    df = _download_from_yfinance(symbol, period, interval)
    if df is None and interval == "1d":
        df = _load_from_archive(symbol, period)
    if df is None:
        return None

    _cache[key] = df.copy()
    return df


def clear_cache() -> None:
    _cache.clear()
