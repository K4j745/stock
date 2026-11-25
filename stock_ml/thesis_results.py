#!/usr/bin/env python3
"""Result tables for the thesis, built from walk-forward out-of-sample predictions.

``collect`` re-runs the walk-forward folds for one ticker and threshold for the
logistic regression, random forest, XGBoost, LightGBM, candle and ARIMA models,
exactly as ``models.train``, ``models.candle_model`` and ``models.arima_baseline``
do, but saves nothing to ``models/saved``. It writes every out-of-sample
prediction and the metrics of every fold.

``summarize`` turns those files into the thesis tables: class balance,
classification metrics with their spread, and a backtest of each model's
strategy on its out-of-sample predictions only. The backtest also includes
the dashboard's technical rules (no ML), computed from data up to each close.

Usage:
    python thesis_results.py collect --ticker AAPL --threshold 0.002
    python thesis_results.py summarize
"""
import argparse
import functools
import json
import logging
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtest.run import compute_portfolio_metrics  # noqa: E402
from config import (  # noqa: E402
    ARIMA_ADF_ALPHA, ARIMA_MAX_P, ARIMA_MAX_Q, BENCHMARK, TICKERS,
    WF_MODE, WF_STEP, WF_TEST_SIZE, WF_TRAIN_SIZE, model_tag,
)
from data.download import download_single  # noqa: E402
from data.preprocess import clean_data  # noqa: E402
from features.pipeline import build_feature_matrix  # noqa: E402
from features.validation import get_time_series_folds  # noqa: E402
from models.arima_core import forecast_auto, probability_above  # noqa: E402
from models.candle_model import _make_model as make_candle_model  # noqa: E402
from models.candle_model import build_candle_feature_matrix  # noqa: E402
from models.evaluate import evaluate_model  # noqa: E402
from models.train import _clone_model, _get_models  # noqa: E402

logger = logging.getLogger("stock_ml")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO_ROOT, "thesis_results")
RAW_DIR = os.path.join(OUT_DIR, "raw")
THRESHOLDS = (0.002, 0.005, 0.01)
METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc", "mcc")
MODEL_ORDER = ("logistic_regression", "random_forest", "xgboost", "lightgbm", "candle", "arima")
RULES = "technical_rules"
BACKTEST_ORDER = MODEL_ORDER + (RULES,)  # the rules are not a classifier of the labels
DASHBOARD_DIR = os.path.join(REPO_ROOT, "dashboard")


def _tag(threshold: float) -> str:
    return model_tag("binary", "A", threshold)


def _folds(n_rows: int):
    return list(get_time_series_folds(
        np.arange(n_rows), train_size=WF_TRAIN_SIZE, test_size=WF_TEST_SIZE, step=WF_STEP, mode=WF_MODE,
    ))


def _fold_record(fold: int, dates: pd.Index, train_idx, test_idx, metrics: dict) -> dict:
    return {
        "fold": fold,
        "train_start": str(dates[train_idx[0]].date()),
        "train_end": str(dates[train_idx[-1]].date()),
        "test_start": str(dates[test_idx[0]].date()),
        "test_end": str(dates[test_idx[-1]].date()),
        **{k: (None if pd.isna(v) else float(v)) for k, v in metrics.items()},
    }


def _run_classifier(X: pd.DataFrame, y: pd.Series, make_model):
    """Walk-forward with a scaler fitted on each training window only."""
    preds, probs, folds = pd.Series(dtype=float), pd.Series(dtype=float), []
    for fold, (train_idx, test_idx) in enumerate(_folds(len(X)), start=1):
        scaler = StandardScaler()
        x_train = pd.DataFrame(scaler.fit_transform(X.iloc[train_idx]), columns=X.columns)
        x_test = pd.DataFrame(scaler.transform(X.iloc[test_idx]), columns=X.columns)
        model = make_model()
        model.fit(x_train, y.iloc[train_idx])
        y_pred = model.predict(x_test)
        y_prob = model.predict_proba(x_test)[:, 1] if hasattr(model, "predict_proba") else None
        test_dates = X.index[test_idx]
        preds = pd.concat([preds, pd.Series(y_pred, index=test_dates, dtype=float)])
        if y_prob is not None:
            probs = pd.concat([probs, pd.Series(y_prob, index=test_dates)])
        folds.append(_fold_record(fold, X.index, train_idx, test_idx,
                                  evaluate_model(y.iloc[test_idx], y_pred, y_prob)))
    return preds, probs, folds


def _run_arima(next_returns: pd.Series, labels: pd.Series, threshold: float):
    returns = next_returns.to_numpy()
    preds, probs, folds = pd.Series(dtype=float), pd.Series(dtype=float), []
    for fold, (train_idx, test_idx) in enumerate(_folds(len(returns)), start=1):
        forecast, chosen = forecast_auto(returns[train_idx], returns[test_idx],
                                         ARIMA_MAX_P, ARIMA_MAX_Q, ARIMA_ADF_ALPHA)
        y_prob = probability_above(forecast.mean, forecast.std, threshold)
        y_pred = (forecast.mean > threshold).astype(int)
        test_dates = next_returns.index[test_idx]
        preds = pd.concat([preds, pd.Series(y_pred, index=test_dates, dtype=float)])
        probs = pd.concat([probs, pd.Series(y_prob, index=test_dates)])
        record = _fold_record(fold, next_returns.index, train_idx, test_idx,
                              evaluate_model(labels.iloc[test_idx], y_pred, y_prob))
        record["order"] = "({},{},{})".format(*chosen.order)
        folds.append(record)
    return preds, probs, folds


def collect(ticker: str, threshold: float) -> None:
    os.makedirs(RAW_DIR, exist_ok=True)
    tag = _tag(threshold)
    close = clean_data(download_single(ticker))["Close"]
    next_return = close.pct_change().shift(-1)

    X, y = build_feature_matrix(ticker, "A", label_mode="binary", threshold=threshold)
    predictions, all_folds = {}, {}
    for name in _get_models():
        preds, probs, folds = _run_classifier(X, y, lambda name=name: _clone_model(name))
        predictions[name] = (preds, probs)
        all_folds[name] = folds
        logger.info("%s %s %s: mean acc=%.4f", ticker, tag, name, np.nanmean([f["accuracy"] for f in folds]))

    Xc, yc = build_candle_feature_matrix(ticker, threshold=threshold)
    preds, probs, folds = _run_classifier(Xc, yc, lambda: make_candle_model("xgboost"))
    predictions["candle"] = (preds, probs)
    all_folds["candle"] = folds

    known = next_return.reindex(y.index).notna()
    labels = y[known]
    preds, probs, folds = _run_arima(next_return.reindex(labels.index), labels, threshold)
    predictions["arima"] = (preds, probs)
    all_folds["arima"] = folds

    dates = sorted(set().union(*[set(p.index) for p, _ in predictions.values()]))
    table = pd.DataFrame(index=pd.DatetimeIndex(dates, name="date"))
    table["next_return"] = next_return.reindex(table.index)
    table["y_true"] = (table["next_return"] > threshold).astype(int)
    for name, (preds, probs) in predictions.items():
        table[f"{name}_pred"] = preds.reindex(table.index)
        table[f"{name}_prob"] = probs.reindex(table.index)
    table.to_csv(os.path.join(RAW_DIR, f"{ticker}_{tag}_predictions.csv"), float_format="%.6g")

    balance = {"ticker": ticker, "tag": tag, "threshold": threshold, "n": int(len(y)),
               "n_buy": int(y.sum()), "start": str(y.index[0].date()), "end": str(y.index[-1].date())}
    with open(os.path.join(RAW_DIR, f"{ticker}_{tag}_folds.json"), "w", encoding="utf-8") as fh:
        json.dump({"class_balance": balance, "folds": all_folds}, fh, indent=1)
    logger.info("Collected %s %s: %d out-of-sample days", ticker, tag, len(table))


@functools.lru_cache(maxsize=None)
def technical_rule_positions(ticker: str) -> pd.Series:
    """1 where the dashboard's technical rules say BUY on day t, else 0.

    Uses ``dashboard/lib/signals.py::technical_rule_based`` with the rule
    settings from ``dashboard/config.json``. The indicators are trailing
    windows, so day t uses data up to its close only. The rules do not depend
    on the label threshold.
    """
    if DASHBOARD_DIR not in sys.path:
        sys.path.insert(0, DASHBOARD_DIR)
    from lib import indicators as dashboard_indicators
    from lib import signals as dashboard_signals

    with open(os.path.join(DASHBOARD_DIR, "config.json"), encoding="utf-8") as fh:
        rules = json.load(fh)["technical_signals"]
    prices = clean_data(download_single(ticker))
    signals = dashboard_signals.technical_rule_based(dashboard_indicators.compute(prices), rules)
    return (signals["signal"] == "BUY").astype(float)


def _load(tag: str):
    folds, preds = {}, {}
    for ticker in TICKERS:
        with open(os.path.join(RAW_DIR, f"{ticker}_{tag}_folds.json"), encoding="utf-8") as fh:
            folds[ticker] = json.load(fh)
        table = pd.read_csv(os.path.join(RAW_DIR, f"{ticker}_{tag}_predictions.csv"),
                            index_col="date", parse_dates=True)
        table[f"{RULES}_pred"] = technical_rule_positions(ticker).reindex(table.index)
        preds[ticker] = table
    return folds, preds


def _classification_tables(folds_by_ticker: dict, tag: str):
    per_ticker, summary = [], []
    for model in MODEL_ORDER:
        ticker_means, pooled = [], []
        for ticker, data in folds_by_ticker.items():
            rows = pd.DataFrame(data["folds"][model])[list(METRICS)].astype(float)
            pooled.append(rows)
            means = rows.mean(skipna=True)
            ticker_means.append(means)
            per_ticker.append({"tag": tag, "model": model, "ticker": ticker, **means.round(4).to_dict()})
        by_ticker = pd.DataFrame(ticker_means)
        all_folds = pd.concat(pooled)
        row = {"tag": tag, "model": model, "n_folds": len(all_folds)}
        for metric in METRICS:
            row[metric] = round(by_ticker[metric].mean(), 4)
            row[f"{metric}_sd_spolki"] = round(by_ticker[metric].std(), 4)
            row[f"{metric}_sd_foldy"] = round(all_folds[metric].std(), 4)
        summary.append(row)
    return per_ticker, summary


def _strategy_metrics(returns: pd.Series, position: pd.Series = None) -> dict:
    metrics = compute_portfolio_metrics(returns)
    if position is not None:
        metrics["exposure"] = round(float(position.mean()), 4)
        metrics["position_changes"] = int(position.diff().abs().sum())
    metrics["days"] = int(len(returns))
    return metrics


def _portfolio_activity(positions: list) -> dict:
    """Mean exposure over tickers and total position changes, for portfolio rows."""
    return {
        "exposure": round(float(np.mean([p.mean() for p in positions])), 4),
        "position_changes": int(sum(p.diff().abs().sum() for p in positions)),
    }


def _backtest_tables(preds_by_ticker: dict, tag: str):
    """Long when the model predicts BUY, cash otherwise; equal weight across tickers.

    The decision on day t takes the return from close t to close t+1. Only days
    with an out-of-sample prediction from every model are used, so all
    strategies cover the same period. No transaction costs are charged.
    """
    model_cols = [f"{m}_pred" for m in BACKTEST_ORDER]
    strategy_returns = {m: [] for m in BACKTEST_ORDER}
    strategy_positions = {m: [] for m in BACKTEST_ORDER}
    buy_hold, per_ticker = [], []
    for ticker, table in preds_by_ticker.items():
        common = table.dropna(subset=model_cols + ["next_return"])
        buy_hold.append(common["next_return"].rename(ticker))
        per_ticker.append({"tag": tag, "ticker": ticker, "strategy": "buy_and_hold",
                           **_strategy_metrics(common["next_return"])})
        for model in BACKTEST_ORDER:
            position = common[f"{model}_pred"]
            returns = (position * common["next_return"]).rename(ticker)
            strategy_returns[model].append(returns)
            strategy_positions[model].append(position)
            per_ticker.append({"tag": tag, "ticker": ticker, "strategy": model,
                               **_strategy_metrics(returns, position)})

    basket_bh = pd.concat(buy_hold, axis=1).dropna().mean(axis=1)
    period = basket_bh.index
    spy_close = clean_data(download_single(BENCHMARK))["Close"]
    spy = spy_close.pct_change().shift(-1).reindex(period).dropna()
    summary = [
        {"tag": tag, "strategy": "spy_buy_and_hold", **_strategy_metrics(spy)},
        {"tag": tag, "strategy": "basket_buy_and_hold", **_strategy_metrics(basket_bh)},
    ]
    for model in BACKTEST_ORDER:
        returns = pd.concat(strategy_returns[model], axis=1).reindex(period).mean(axis=1)
        summary.append({"tag": tag, "strategy": model, **_strategy_metrics(returns),
                        **_portfolio_activity(strategy_positions[model])})
    return per_ticker, summary, (str(period[0].date()), str(period[-1].date()))


def summarize() -> None:
    balance, cls_ticker, cls_summary, bt_ticker, bt_summary, periods = [], [], [], [], [], {}
    for threshold in THRESHOLDS:
        tag = _tag(threshold)
        folds, preds = _load(tag)
        balance += [data["class_balance"] for data in folds.values()]
        rows, summary = _classification_tables(folds, tag)
        cls_ticker += rows
        cls_summary += summary
        rows, summary, periods[tag] = _backtest_tables(preds, tag)
        bt_ticker += rows
        bt_summary += summary

    balance_df = pd.DataFrame(balance)
    balance_df["buy_share"] = (balance_df["n_buy"] / balance_df["n"]).round(4)
    totals = balance_df.groupby(["tag", "threshold"], as_index=False)[["n", "n_buy"]].sum()
    totals["ticker"] = "RAZEM"
    totals["buy_share"] = (totals["n_buy"] / totals["n"]).round(4)
    balance_df = pd.concat([balance_df, totals], ignore_index=True)
    balance_df["sell_share"] = (1 - balance_df["buy_share"]).round(4)

    balance_df.to_csv(os.path.join(OUT_DIR, "udzial_klas.csv"), index=False)
    pd.DataFrame(cls_summary).to_csv(os.path.join(OUT_DIR, "klasyfikacja_podsumowanie.csv"), index=False)
    pd.DataFrame(cls_ticker).to_csv(os.path.join(OUT_DIR, "klasyfikacja_spolki.csv"), index=False)
    pd.DataFrame(bt_summary).to_csv(os.path.join(OUT_DIR, "backtest_podsumowanie.csv"), index=False)
    pd.DataFrame(bt_ticker).to_csv(os.path.join(OUT_DIR, "backtest_spolki.csv"), index=False)
    with open(os.path.join(OUT_DIR, "metadane.json"), "w", encoding="utf-8") as fh:
        json.dump({
            "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "thresholds": list(THRESHOLDS),
            "walk_forward": {"train": WF_TRAIN_SIZE, "test": WF_TEST_SIZE, "step": WF_STEP, "mode": WF_MODE},
            "backtest_periods": periods,
            "models": list(MODEL_ORDER),
            "backtest_strategies": list(BACKTEST_ORDER),
        }, fh, indent=2)
    logger.info("Thesis tables written to %s", OUT_DIR)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p_collect = sub.add_parser("collect")
    p_collect.add_argument("--ticker", required=True)
    p_collect.add_argument("--threshold", type=float, required=True)
    sub.add_parser("summarize")
    args = parser.parse_args()
    if args.command == "collect":
        collect(args.ticker, args.threshold)
    else:
        summarize()


if __name__ == "__main__":
    main()
