"""Walk-forward time-series cross-validation.

Provides a rolling-window fold generator suitable for financial time series.
Default mode is 'rolling' (fixed-size training window, older data discarded);
'expanding' is offered as an opt-in alternative.
"""
import logging
from typing import Iterator, Tuple

import numpy as np
import pandas as pd

from config import WF_TRAIN_SIZE, WF_TEST_SIZE, WF_STEP, WF_MODE

logger = logging.getLogger("stock_ml")


def get_time_series_folds(
    df,
    train_size: int = WF_TRAIN_SIZE,
    test_size: int = WF_TEST_SIZE,
    step: int = WF_STEP,
    mode: str = WF_MODE,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Generate walk-forward train/test index pairs.

    Args:
        df: DataFrame, Series, or array-like with a length (rows are ordered in time).
        train_size: Number of training observations per fold (sessions).
        test_size: Number of test observations per fold (sessions).
        step: Stride between successive folds.
        mode: 'rolling' (fixed-size train window) or 'expanding' (growing train window).

    Yields:
        Tuples of (train_idx, test_idx) as numpy integer arrays.
    """
    if mode not in {"rolling", "expanding"}:
        raise ValueError(f"Unknown mode '{mode}'. Use 'rolling' or 'expanding'.")

    n = len(df)
    if n < train_size + test_size:
        raise ValueError(
            f"Not enough samples ({n}) for train_size={train_size} + test_size={test_size}."
        )

    start = 0
    fold_idx = 0
    while start + train_size + test_size <= n:
        train_start = 0 if mode == "expanding" else start
        train_end = start + train_size
        test_end = train_end + test_size

        train_idx = np.arange(train_start, train_end)
        test_idx = np.arange(train_end, test_end)
        yield train_idx, test_idx

        fold_idx += 1
        start += step

    logger.info(
        "Generated %d walk-forward folds (mode=%s, train=%d, test=%d, step=%d, n=%d)",
        fold_idx, mode, train_size, test_size, step, n,
    )


def count_folds(
    n: int,
    train_size: int = WF_TRAIN_SIZE,
    test_size: int = WF_TEST_SIZE,
    step: int = WF_STEP,
) -> int:
    """Return how many walk-forward folds fit in a series of length ``n``."""
    if n < train_size + test_size:
        return 0
    return 1 + (n - train_size - test_size) // step
