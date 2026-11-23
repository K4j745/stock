"""Tests for the extra thesis tables: pooled metrics, costs and random baselines."""
import math

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score, matthews_corrcoef, precision_score, recall_score

from backtest.run import compute_portfolio_metrics
from thesis_extras import (
    _confusion, _net_returns, _path_metrics, _pooled_metrics, _runs, _shifted_positions, _shuffled_runs, _turnover,
)


def test_pooled_metrics_match_sklearn_on_one_confusion_matrix():
    rng = np.random.default_rng(0)
    y_true = pd.Series(rng.integers(0, 2, 500))
    y_pred = pd.Series(rng.integers(0, 2, 500))

    metrics = _pooled_metrics(_confusion(y_true, y_pred))

    assert metrics["precision"] == pytest.approx(precision_score(y_true, y_pred))
    assert metrics["recall"] == pytest.approx(recall_score(y_true, y_pred))
    assert metrics["f1"] == pytest.approx(f1_score(y_true, y_pred))
    assert metrics["mcc"] == pytest.approx(matthews_corrcoef(y_true, y_pred))
    assert metrics["always_sell_accuracy"] == pytest.approx(1 - y_true.mean())


def test_precision_is_undefined_not_zero_without_buy_predictions():
    metrics = _pooled_metrics(_confusion(pd.Series([1, 0, 1]), pd.Series([0, 0, 0])))

    assert math.isnan(metrics["precision"])
    assert metrics["recall"] == 0


def test_turnover_counts_the_opening_entry_and_every_change():
    position = pd.Series([1.0, 1.0, 0.0, 1.0, 1.0])

    assert _turnover(position).tolist() == [1, 0, 1, 1, 0]


def test_net_returns_charge_cost_per_half_turn():
    position = pd.Series([1.0, 0.0])
    next_return = pd.Series([0.01, 0.02])

    net = _net_returns(position, next_return, cost_bps=5)

    assert net.tolist() == pytest.approx([0.01 - 0.0005, -0.0005])


def test_shuffled_runs_keep_exposure_and_number_of_trades():
    position = np.array([0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1], dtype=float)
    ones, zeros = _runs(position)
    rng = np.random.default_rng(1)

    for _ in range(50):
        shuffled = _shuffled_runs(ones, zeros, rng)
        assert len(shuffled) == len(position)
        assert shuffled.sum() == position.sum()
        assert np.abs(np.diff(shuffled)).sum() == np.abs(np.diff(position)).sum()


def test_shuffled_runs_handle_a_strategy_that_never_buys():
    position = np.zeros(10)
    ones, zeros = _runs(position)

    assert _shuffled_runs(ones, zeros, np.random.default_rng(0)).tolist() == position.tolist()


def test_shared_shift_moves_every_ticker_by_the_same_lag():
    position = np.array([1, 1, 0, 0, 0, 1], dtype=float)
    offsets = np.array([1, 2])

    shifted = _shifted_positions(position, None, offsets)

    assert shifted.tolist() == [np.roll(position, 1).tolist(), np.roll(position, 2).tolist()]
    assert (shifted.sum(axis=1) == position.sum()).all()


def test_path_metrics_match_compute_portfolio_metrics():
    returns = pd.Series(np.random.default_rng(2).normal(0.0004, 0.01, 800))

    vectorised = _path_metrics(returns.to_numpy())
    reference = compute_portfolio_metrics(returns)

    # compute_portfolio_metrics rounds to 6 decimals.
    assert vectorised["sharpe"][0] == pytest.approx(reference["sharpe_ratio"], abs=1e-6)
    assert vectorised["cagr"][0] == pytest.approx(reference["annualized_return"], abs=1e-6)
    assert vectorised["max_drawdown"][0] == pytest.approx(reference["max_drawdown"], abs=1e-6)
