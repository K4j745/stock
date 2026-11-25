"""Label generation for next-day price direction.

Default: binary mode. Label = 1 if next-day return > LABEL_THRESHOLD (0.5%),
else 0. The threshold lives in ``config.LABEL_THRESHOLD`` and is the single
source of truth.

A 3-class 'multiclass' mode is kept for experimental use only: it returns
{-1, 0, 1} for down/hold/up. Hold dominates and models in this mode must use
``class_weight='balanced'``.
"""
import logging

import pandas as pd

from config import LABEL_THRESHOLD, LABEL_THRESHOLD_B

logger = logging.getLogger("stock_ml")


def _next_day_return(df: pd.DataFrame) -> pd.Series:
    return df["Close"].pct_change().shift(-1)


def _only_known(labels: pd.Series, next_day_return: pd.Series) -> pd.Series:
    """NaN where the next-day return is unknown (the last row), so dropna() removes it."""
    return labels.astype(float).where(next_day_return.notna())


def create_binary_labels(df: pd.DataFrame, threshold: float = LABEL_THRESHOLD) -> pd.Series:
    """Binary label: 1 if next-day return > threshold, else 0; NaN without a next day."""
    next_day_return = _next_day_return(df)
    labels = _only_known(next_day_return > threshold, next_day_return)
    logger.info(
        "Binary labels (threshold=%.4f): %d positive (%.1f%%), %d negative",
        threshold, labels.sum(), 100 * labels.mean(), labels.count() - labels.sum(),
    )
    return labels


def create_multiclass_labels(df: pd.DataFrame, threshold: float = LABEL_THRESHOLD) -> pd.Series:
    """3-class label: 1 if next-day return > +threshold, -1 if < -threshold, else 0.

    The 'hold' class (0) dominates with realistic thresholds and is treated as
    a separate class. Train with ``class_weight='balanced'``.
    """
    next_day_return = _next_day_return(df)
    labels = pd.Series(0, index=df.index, dtype=int)
    labels[next_day_return > threshold] = 1
    labels[next_day_return < -threshold] = -1
    labels = _only_known(labels, next_day_return)
    counts = labels.value_counts().to_dict()
    logger.info(
        "Multiclass labels (threshold=%.4f): up=%d, hold=%d, down=%d",
        threshold, counts.get(1, 0), counts.get(0, 0), counts.get(-1, 0),
    )
    return labels


def create_labels_version_a(df: pd.DataFrame) -> pd.Series:
    """Legacy: label = 1 if next day return >= 0, else 0; NaN without a next day."""
    next_day_return = _next_day_return(df)
    labels = _only_known(next_day_return >= 0, next_day_return)
    logger.info("Labels version A: %d positive (%.1f%%), %d negative",
                labels.sum(), 100 * labels.mean(), labels.count() - labels.sum())
    return labels


def create_labels_version_b(df: pd.DataFrame, threshold: float = LABEL_THRESHOLD_B) -> pd.Series:
    """Legacy: label = 1 if next day return > threshold, else 0; NaN without a next day."""
    next_day_return = _next_day_return(df)
    labels = _only_known(next_day_return > threshold, next_day_return)
    logger.info("Labels version B (threshold=%.4f): %d positive (%.1f%%), %d negative",
                threshold, labels.sum(), 100 * labels.mean(), labels.count() - labels.sum())
    return labels
