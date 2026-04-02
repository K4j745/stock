import logging

import pandas as pd

from config import LABEL_THRESHOLD_B

logger = logging.getLogger("stock_ml")


def create_labels_version_a(df: pd.DataFrame) -> pd.Series:
    """Label = 1 if next day return >= 0, else 0."""
    next_day_return = df["Close"].pct_change().shift(-1)
    labels = (next_day_return >= 0).astype(int)
    logger.info("Labels version A: %d positive (%.1f%%), %d negative",
                labels.sum(), 100 * labels.mean(), len(labels) - labels.sum())
    return labels


def create_labels_version_b(df: pd.DataFrame, threshold: float = LABEL_THRESHOLD_B) -> pd.Series:
    """Label = 1 if next day return > threshold, else 0."""
    next_day_return = df["Close"].pct_change().shift(-1)
    labels = (next_day_return > threshold).astype(int)
    logger.info("Labels version B (threshold=%.4f): %d positive (%.1f%%), %d negative",
                threshold, labels.sum(), 100 * labels.mean(), len(labels) - labels.sum())
    return labels
