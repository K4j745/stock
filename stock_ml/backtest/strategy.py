"""
Backtest strategy: convert model predictions to portfolio positions and compute P&L.
"""
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def build_signals(predictions: pd.Series) -> pd.Series:
    """
    Convert binary predictions to trading signals.
    1 -> go long (buy)
    0 -> go to cash (no position)
    Returns a Series of {1, 0} aligned with predictions index.
    """
    return predictions.astype(int)


def compute_strategy_returns(
    price_series: pd.Series,
    signals: pd.Series
) -> pd.Series:
    """
    Compute daily strategy returns given price series and signals.

    Signal on day t is based on prediction made on day t,
    predicting direction for day t+1.
    So we shift signals by 1 to align position with next-day return.

    Args:
        price_series: Close prices (aligned with signals index)
        signals: 0/1 signals (1 = long, 0 = cash)

    Returns:
        Daily strategy returns as pd.Series
    """
    daily_returns = price_series.pct_change()
    # Shift signal by 1: signal on day t applies to return on day t+1
    position = signals.shift(1).fillna(0)
    strategy_returns = position * daily_returns
    return strategy_returns.dropna()


def compute_buyhold_returns(price_series: pd.Series) -> pd.Series:
    """Compute buy-and-hold daily returns."""
    return price_series.pct_change().dropna()
