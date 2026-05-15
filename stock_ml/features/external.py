"""External features: VIX and earnings-window dummies.

Both features are leakage-free:

- ``vix_close`` is the close of ^VIX on day T. Used to predict the move from
  T to T+1, so no future information enters.
- ``earnings_flag`` is 1 when an earnings event falls within +/- 5 sessions
  of the current row. Earnings calendars are public ahead of the event, so
  the flag is known at day T.
"""
import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("stock_ml")


def add_vix_feature(df: pd.DataFrame, start_date=None, end_date=None) -> pd.DataFrame:
    """Attach a ``vix_close`` column aligned to df's index.

    Pulls ^VIX from yfinance and aligns by date. NaNs are forward-filled
    (markets may be open while VIX data is missing) and then back-filled at
    the head to avoid dropping the warm-up window.
    """
    df = df.copy()
    try:
        import yfinance as yf
    except ImportError:
        logger.warning("yfinance not available; skipping VIX feature")
        df["vix_close"] = np.nan
        return df

    start_date = start_date or df.index.min()
    end_date = end_date or (df.index.max() + pd.Timedelta(days=1))

    try:
        vix = yf.download("^VIX", start=start_date, end=end_date, progress=False, auto_adjust=True)
    except Exception as exc:
        logger.warning("VIX download failed (%s); filling NaN", exc)
        df["vix_close"] = np.nan
        return df

    if vix.empty:
        logger.warning("VIX download returned empty frame; filling NaN")
        df["vix_close"] = np.nan
        return df

    close = vix["Close"]
    if isinstance(close, pd.DataFrame):
        # Flatten if yfinance returned a MultiIndex
        close = close.iloc[:, 0]
    close.index = pd.to_datetime(close.index).normalize()

    idx = pd.to_datetime(df.index).normalize()
    vix_series = close.reindex(idx).ffill().bfill()
    df["vix_close"] = vix_series.values

    logger.info("Added vix_close (%d rows, %d NaN)", len(df), df["vix_close"].isna().sum())
    return df


def add_earnings_dummy(
    df: pd.DataFrame, ticker: str, window: int = 5
) -> pd.DataFrame:
    """Mark sessions within +/- ``window`` trading days of an earnings date.

    Uses ``yfinance.Ticker(ticker).get_earnings_dates()`` (preferred over the
    legacy ``.calendar`` which only returns the next event). Falls back to
    ``.calendar`` if that fails. On any failure the flag is set to 0 and
    training continues.
    """
    df = df.copy()
    df["earnings_flag"] = 0

    try:
        import yfinance as yf
    except ImportError:
        logger.warning("yfinance not available; earnings_flag=0")
        return df

    dates: Optional[pd.DatetimeIndex] = None
    try:
        tk = yf.Ticker(ticker)
        ed = None
        if hasattr(tk, "get_earnings_dates"):
            try:
                ed = tk.get_earnings_dates(limit=80)
            except Exception:
                ed = None
        if ed is not None and not ed.empty:
            dates = pd.to_datetime(ed.index).tz_localize(None).normalize()
        else:
            cal = getattr(tk, "calendar", None)
            if isinstance(cal, dict):
                vals = cal.get("Earnings Date") or []
                if vals:
                    dates = pd.to_datetime(pd.Series(vals)).dt.tz_localize(None).dt.normalize()
                    dates = pd.DatetimeIndex(dates)
            elif isinstance(cal, pd.DataFrame) and "Earnings Date" in cal.index:
                vals = cal.loc["Earnings Date"].dropna().values
                dates = pd.to_datetime(pd.Series(vals)).dt.tz_localize(None).dt.normalize()
                dates = pd.DatetimeIndex(dates)
    except Exception as exc:
        logger.warning("Earnings fetch failed for %s (%s); earnings_flag=0", ticker, exc)
        return df

    if dates is None or len(dates) == 0:
        logger.info("No earnings dates available for %s; earnings_flag=0", ticker)
        return df

    sessions = pd.DatetimeIndex(pd.to_datetime(df.index).normalize())
    flag = np.zeros(len(sessions), dtype=int)

    for ev_date in dates:
        # Locate the nearest session at or after the earnings event; mark
        # +/- window sessions around it.
        pos = sessions.searchsorted(ev_date)
        lo = max(pos - window, 0)
        hi = min(pos + window + 1, len(sessions))
        flag[lo:hi] = 1

    df["earnings_flag"] = flag
    logger.info(
        "earnings_flag for %s: %d events, %d sessions flagged (%.1f%%)",
        ticker, len(dates), flag.sum(), 100 * flag.mean(),
    )
    return df
