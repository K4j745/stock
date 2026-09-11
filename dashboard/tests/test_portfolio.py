"""Unit tests for the portfolio backtest trading mechanics.

Covers the opt-in ``seed_shares`` + ``trade_mode: incremental`` + ``trade_size``
behaviour introduced for the model-monitoring portfolios, and verifies that the
legacy ``all_in`` behaviour is preserved when those rules are absent.
"""
import numpy as np
import pandas as pd
import pytest

from lib import portfolio


def _prices(dates, opens, closes):
    """Build a minimal OHLCV frame indexed by ``dates``."""
    return pd.DataFrame(
        {
            "Open": opens,
            "High": [c * 1.01 for c in closes],
            "Low": [o * 0.99 for o in opens],
            "Close": closes,
            "Volume": [1_000_000] * len(dates),
        },
        index=pd.to_datetime(dates),
    )


def _signals(dates, sigs):
    """Build a signal frame with the columns the engine reads."""
    return pd.DataFrame(
        {
            "signal": sigs,
            "score": [0.6] * len(dates),
            "confidence": [0.6] * len(dates),
            "probability_up": [0.6] * len(dates),
            "threshold_used": [0.5] * len(dates),
            "triggered_rules": ["test"] * len(dates),
        },
        index=pd.to_datetime(dates),
    )


DATES = ["2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07"]


def _run(rules, sigs):
    prices = {"AAPL": _prices(DATES, opens=[100, 110, 120, 130], closes=[105, 115, 125, 135])}
    signals = {"AAPL": {"logistic_regression": _signals(DATES, sigs)}}
    cfg = {
        "id": "t",
        "name": "Test",
        "tickers": ["AAPL"],
        "initial_capital": 100000,
        "signal_source": "logistic_regression",
        "note": "unit-test portfolio",
        "rules": rules,
    }
    return portfolio.run_portfolio_backtest(cfg, prices, signals)


@pytest.mark.unit
def test_seed_buys_starter_position_on_first_day():
    # No trading signals — pure seed. Expect exactly 10 shares bought at day-1 open.
    res = _run(
        {"seed_shares": 10, "trade_mode": "incremental", "trade_size": 1},
        sigs=["HOLD", "HOLD", "HOLD", "HOLD"],
    )
    seed_txs = [t for t in res.transactions if t["action"] == "SEED"]
    assert len(seed_txs) == 1
    assert seed_txs[0]["quantity"] == 10
    assert seed_txs[0]["execution_price"] == 100  # first day's Open
    assert res.positions[0]["shares"] == 10
    # cash reduced by 10 * 100
    assert res.metadata["seed_shares"] == 10
    assert res.metadata["trade_mode"] == "incremental"


@pytest.mark.unit
def test_incremental_buy_adds_one_share_per_signal():
    res = _run(
        {"seed_shares": 10, "trade_mode": "incremental", "trade_size": 1},
        sigs=["BUY", "BUY", "HOLD", "HOLD"],
    )
    # seed 10 + two incremental BUYs of 1 share each = 12
    assert res.positions[0]["shares"] == 12
    buy_txs = [t for t in res.transactions if t["action"] == "BUY"]
    assert len(buy_txs) == 2
    assert all(t["quantity"] == 1 for t in buy_txs)


@pytest.mark.unit
def test_incremental_sell_trims_one_share_and_keeps_rest():
    res = _run(
        {"seed_shares": 10, "trade_mode": "incremental", "trade_size": 1},
        sigs=["HOLD", "SELL", "SELL", "HOLD"],
    )
    # seed 10 - two incremental SELLs of 1 share = 8 (partial sell, position stays open)
    assert res.positions[0]["shares"] == 8
    sell_txs = [t for t in res.transactions if t["action"] == "SELL"]
    assert len(sell_txs) == 2
    assert all(t["quantity"] == 1 for t in sell_txs)
    # realised P&L recorded on the sold shares
    assert all(t["trade_pnl_abs"] is not None for t in sell_txs)


@pytest.mark.unit
def test_trade_size_greater_than_one():
    res = _run(
        {"seed_shares": 5, "trade_mode": "incremental", "trade_size": 3},
        sigs=["BUY", "HOLD", "SELL", "HOLD"],
    )
    # seed 5 + 3 - 3 = 5
    assert res.positions[0]["shares"] == 5


@pytest.mark.unit
def test_legacy_all_in_is_default_when_no_trade_rules():
    # No seed, no trade_mode -> legacy all-in: BUY enters the full slot, SELL exits fully.
    res = _run(
        {"max_positions": 1},
        sigs=["BUY", "HOLD", "SELL", "HOLD"],
    )
    # After SELL the position is fully liquidated.
    assert res.positions == [] or all(p["shares"] == 0 for p in res.positions)
    buy_txs = [t for t in res.transactions if t["action"] == "BUY"]
    sell_txs = [t for t in res.transactions if t["action"] == "SELL"]
    assert len(buy_txs) == 1
    assert len(sell_txs) == 1
    # All-in bought many shares (~ capital / price), not just 1.
    assert buy_txs[0]["quantity"] > 1
    # Legacy default reported in metadata.
    assert res.metadata["trade_mode"] == "all_in"
    assert res.metadata["seed_shares"] == 0


@pytest.mark.unit
def test_seed_capped_by_available_cash():
    # 10 shares * 100 = 1000 needed; give only 250 capital -> can seed at most 2.
    prices = {"AAPL": _prices(DATES, opens=[100, 110, 120, 130], closes=[105, 115, 125, 135])}
    signals = {"AAPL": {"logistic_regression": _signals(DATES, ["HOLD"] * 4)}}
    cfg = {
        "id": "t",
        "name": "Test",
        "tickers": ["AAPL"],
        "initial_capital": 250,
        "signal_source": "logistic_regression",
        "rules": {"seed_shares": 10, "trade_mode": "incremental", "trade_size": 1},
    }
    res = portfolio.run_portfolio_backtest(cfg, prices, signals)
    assert res.positions[0]["shares"] == 2  # floor(250 / 100)
