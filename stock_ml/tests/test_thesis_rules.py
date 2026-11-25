"""Tests for the technical-rules strategy in the thesis backtest."""
import numpy as np
import pandas as pd

import thesis_results


def random_walk_prices(n: int, seed: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.015, n)))
    open_ = close * (1 + rng.normal(0, 0.003, n))
    return pd.DataFrame(
        {"Open": open_, "High": np.maximum(open_, close) * 1.01, "Low": np.minimum(open_, close) * 0.99,
         "Close": close, "Volume": rng.integers(1_000_000, 2_000_000, n)},
        index=pd.bdate_range("2020-01-01", periods=n),
    )


def positions_for(prices: pd.DataFrame, monkeypatch) -> pd.Series:
    thesis_results.technical_rule_positions.cache_clear()
    monkeypatch.setattr(thesis_results, "download_single", lambda ticker: prices)
    monkeypatch.setattr(thesis_results, "clean_data", lambda df: df)
    try:
        return thesis_results.technical_rule_positions("TEST")
    finally:
        thesis_results.technical_rule_positions.cache_clear()


def test_rule_positions_are_binary(monkeypatch):
    positions = positions_for(random_walk_prices(300), monkeypatch)

    assert set(positions.unique()) <= {0.0, 1.0}
    assert 0 < positions.mean() < 1


def test_rule_positions_do_not_use_later_data(monkeypatch):
    prices = random_walk_prices(300)

    full = positions_for(prices, monkeypatch)
    truncated = positions_for(prices.iloc[:200], monkeypatch)

    pd.testing.assert_series_equal(full.iloc[:200], truncated)


def test_rules_are_a_backtest_strategy_but_not_a_classifier():
    assert thesis_results.RULES in thesis_results.BACKTEST_ORDER
    assert thesis_results.RULES not in thesis_results.MODEL_ORDER
