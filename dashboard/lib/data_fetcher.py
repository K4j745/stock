"""yfinance wrapper with a tiny in-memory cache.

The generator is run from CI on a schedule, so persistent caching would just
get stale. We only cache within a single ``generate.py`` run so that the
benchmark series is not downloaded once per portfolio.
"""
from __future__ import annotations

import logging
from typing import Dict, Optional

import pandas as pd

logger = logging.getLogger("dashboard.data")


_cache: Dict[str, pd.DataFrame] = {}


def download(symbol: str, period: str = "3y", interval: str = "1d") -> Optional[pd.DataFrame]:
    """Download OHLCV for ``symbol`` from yfinance. Returns ``None`` on failure."""
    key = f"{symbol}|{period}|{interval}"
    if key in _cache:
        return _cache[key].copy()

    try:
        import yfinance as yf  # local import keeps unit tests light
    except ImportError as exc:  # pragma: no cover
        logger.error("yfinance not available: %s", exc)
        return None

    logger.info("Downloading %s (%s %s)…", symbol, period, interval)
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=False)
    except Exception as exc:  # pragma: no cover — network failures
        logger.error("yfinance download failed for %s: %s", symbol, exc)
        return None

    if df is None or df.empty:
        logger.warning("No data returned for %s", symbol)
        return None

    # yfinance sometimes returns a MultiIndex when downloading a single symbol;
    # flatten it so downstream code can rely on plain string column names.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    _cache[key] = df.copy()
    return df


def clear_cache() -> None:
    _cache.clear()
