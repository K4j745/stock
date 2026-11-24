"""Tests for the next-day labels: a day without a known next-day return has no label."""
import numpy as np
import pandas as pd
import pytest

from features.labels import (
    create_binary_labels,
    create_labels_version_a,
    create_labels_version_b,
    create_multiclass_labels,
)

CLOSES = [100.0, 101.0, 100.5, 102.0, 101.0]  # next-day returns: +1%, -0.5%, +1.49%, -0.98%, unknown


def prices() -> pd.DataFrame:
    return pd.DataFrame({"Close": CLOSES}, index=pd.bdate_range("2026-03-25", periods=len(CLOSES)))


@pytest.mark.parametrize("make_labels", [
    lambda df: create_binary_labels(df, threshold=0.002),
    lambda df: create_multiclass_labels(df, threshold=0.002),
    create_labels_version_a,
    lambda df: create_labels_version_b(df, threshold=0.002),
])
def test_last_day_has_no_label(make_labels):
    labels = make_labels(prices())

    assert np.isnan(labels.iloc[-1])
    assert labels.iloc[:-1].notna().all()


def test_binary_labels_compare_the_next_day_return_with_the_threshold():
    labels = create_binary_labels(prices(), threshold=0.002)

    assert labels.iloc[:-1].tolist() == [1, 0, 1, 0]


def test_multiclass_labels_mark_up_hold_and_down():
    labels = create_multiclass_labels(prices(), threshold=0.006)

    assert labels.iloc[:-1].tolist() == [1, 0, 1, -1]


def test_dropna_removes_the_unlabelled_day():
    df = prices()
    df["label"] = create_binary_labels(df, threshold=0.002)

    assert df.dropna().index[-1] == df.index[-2]
