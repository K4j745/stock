"""Multi-portfolio backtester.

Runs the saved strategy for each ticker in parallel and compares them
against the SPY buy-and-hold benchmark. Produces:

- One equity curve per ticker plus one for SPY.
- A long-format CSV with columns: date, ticker, portfolio_value,
  benchmark_value, signal.
- A summary metrics table (CAGR, Sharpe, max drawdown, total return).
- A combined matplotlib chart of all equity curves on a single axis.
"""
from __future__ import annotations

import logging
import os
from typing import Dict, Iterable, Optional

import joblib
import numpy as np
import pandas as pd

from config import BENCHMARK, MODEL_DIR, REPORTS_DIR, TICKERS
from backtest.strategy import (
    build_signals,
    compute_buyhold_returns,
    compute_strategy_returns,
)

logger = logging.getLogger("stock_ml")


# --- Portfolio metrics ----------------------------------------------------

def _portfolio_metrics(returns: pd.Series, periods_per_year: int = 252) -> Dict[str, float]:
    """CAGR, Sharpe, max drawdown, total return, volatility from a daily return series."""
    returns = returns.dropna()
    if returns.empty:
        return {
            "cagr": np.nan,
            "sharpe": np.nan,
            "max_drawdown": np.nan,
            "total_return": np.nan,
            "volatility": np.nan,
        }

    total_return = float((1 + returns).prod() - 1)
    n_years = len(returns) / periods_per_year
    cagr = float((1 + total_return) ** (1 / n_years) - 1) if n_years > 0 else np.nan
    volatility = float(returns.std() * np.sqrt(periods_per_year))
    sharpe = float(returns.mean() / returns.std() * np.sqrt(periods_per_year)) if returns.std() > 0 else np.nan

    equity = (1 + returns).cumprod()
    drawdown = (equity - equity.cummax()) / equity.cummax()
    max_dd = float(drawdown.min())

    return {
        "cagr": cagr,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "total_return": total_return,
        "volatility": volatility,
    }


# --- Single-ticker backtest ----------------------------------------------

def _load_model(model_name: str, ticker: str, label_version: str):
    if model_name == "xgboost":
        import xgboost as xgb
        path = os.path.join(MODEL_DIR, f"xgboost_{ticker}_{label_version}.json")
        if not os.path.exists(path):
            return None
        m = xgb.XGBClassifier()
        m.load_model(path)
        return m
    if model_name == "lightgbm":
        import lightgbm as lgb
        path = os.path.join(MODEL_DIR, f"lightgbm_{ticker}_{label_version}.txt")
        if not os.path.exists(path):
            return None
        return lgb.Booster(model_file=path)
    path = os.path.join(MODEL_DIR, f"{model_name}_{ticker}_{label_version}.joblib")
    if not os.path.exists(path):
        return None
    return joblib.load(path)


def _backtest_ticker(
    ticker: str,
    model_name: str,
    label_version: str,
) -> Optional[Dict[str, object]]:
    """Backtest one ticker on its walk-forward out-of-sample window."""
    from data.download import download_single
    from features.pipeline import build_feature_matrix

    X, _ = build_feature_matrix(ticker, label_version)
    # Use the last 20% as out-of-sample to match existing semantics.
    split_idx = int(len(X) * 0.8)
    X_test = X.iloc[split_idx:]
    if X_test.empty:
        logger.warning("No out-of-sample rows for %s; skipping", ticker)
        return None

    scaler_path = os.path.join(MODEL_DIR, f"scaler_{ticker}_{label_version}.joblib")
    if not os.path.exists(scaler_path):
        logger.warning("Scaler missing for %s; skipping", ticker)
        return None
    scaler = joblib.load(scaler_path)
    X_test_scaled = scaler.transform(X_test)

    model = _load_model(model_name, ticker, label_version)
    if model is None:
        logger.warning("Model %s missing for %s; skipping", model_name, ticker)
        return None

    if model_name == "lightgbm":
        proba = model.predict(X_test_scaled)
        preds = (proba > 0.5).astype(int)
    else:
        preds = model.predict(X_test_scaled)

    predictions = pd.Series(preds, index=X_test.index)
    signals = build_signals(predictions)

    raw = download_single(ticker)
    price = raw["Close"].reindex(X_test.index)

    strat_ret = compute_strategy_returns(price, signals)
    bh_ret = compute_buyhold_returns(price)

    equity = (1 + strat_ret).cumprod()
    benchmark_equity = (1 + bh_ret.reindex(strat_ret.index)).cumprod()

    metrics = _portfolio_metrics(strat_ret)
    metrics["start_value"] = 1.0
    metrics["end_value"] = float(equity.iloc[-1]) if not equity.empty else np.nan

    return {
        "ticker": ticker,
        "equity_curve": equity,
        "benchmark_equity": benchmark_equity,
        "signals": signals.reindex(strat_ret.index).fillna(0).astype(int),
        "returns": strat_ret,
        "cagr": metrics["cagr"],
        "sharpe": metrics["sharpe"],
        "max_drawdown": metrics["max_drawdown"],
        "total_return": metrics["total_return"],
        "volatility": metrics["volatility"],
    }


def _spy_buyhold(start, end) -> Dict[str, object]:
    """Compute SPY buy-and-hold equity curve for the union of test windows."""
    from data.download import download_single

    spy = download_single(BENCHMARK)
    price = spy["Close"].loc[(spy.index >= start) & (spy.index <= end)]
    ret = price.pct_change().dropna()
    equity = (1 + ret).cumprod()
    metrics = _portfolio_metrics(ret)
    return {
        "ticker": f"{BENCHMARK}_benchmark",
        "equity_curve": equity,
        "returns": ret,
        "cagr": metrics["cagr"],
        "sharpe": metrics["sharpe"],
        "max_drawdown": metrics["max_drawdown"],
        "total_return": metrics["total_return"],
        "volatility": metrics["volatility"],
    }


# --- Top-level entry point -----------------------------------------------

def run_multi_backtest(
    tickers: Iterable[str] = None,
    model_name: str = "xgboost",
    label_version: str = "A",
    out_dir: str = None,
) -> Dict[str, dict]:
    """Backtest every ticker, add the SPY buy-and-hold benchmark, save artefacts.

    Returns a dict keyed by ticker (plus ``SPY_benchmark``) with each entry
    containing the equity curve, daily returns, and headline metrics.
    """
    tickers = list(tickers) if tickers is not None else list(TICKERS)
    out_dir = out_dir or REPORTS_DIR
    os.makedirs(out_dir, exist_ok=True)

    results: Dict[str, dict] = {}
    long_rows = []
    union_start, union_end = None, None

    for ticker in tickers:
        if ticker == BENCHMARK:
            continue  # SPY is benchmark-only - never trained
        logger.info("Backtesting %s (%s)...", ticker, model_name)
        res = _backtest_ticker(ticker, model_name, label_version)
        if res is None:
            continue
        results[ticker] = res

        eq = res["equity_curve"]
        bench = res["benchmark_equity"]
        sig = res["signals"]

        if union_start is None or eq.index.min() < union_start:
            union_start = eq.index.min()
        if union_end is None or eq.index.max() > union_end:
            union_end = eq.index.max()

        for date, val in eq.items():
            long_rows.append({
                "date": pd.Timestamp(date).date().isoformat(),
                "ticker": ticker,
                "portfolio_value": float(val),
                "benchmark_value": float(bench.loc[date]) if date in bench.index else np.nan,
                "signal": int(sig.loc[date]) if date in sig.index else 0,
            })

    if union_start is not None:
        spy = _spy_buyhold(union_start, union_end)
        results[f"{BENCHMARK}_benchmark"] = spy
        for date, val in spy["equity_curve"].items():
            long_rows.append({
                "date": pd.Timestamp(date).date().isoformat(),
                "ticker": f"{BENCHMARK}_benchmark",
                "portfolio_value": float(val),
                "benchmark_value": float(val),
                "signal": 1,
            })

    # --- Long-format CSV ----------------------------------------------
    if long_rows:
        long_df = pd.DataFrame(long_rows)
        csv_path = os.path.join(out_dir, f"portfolio_equity_{model_name}_{label_version}.csv")
        long_df.to_csv(csv_path, index=False)
        logger.info("Saved equity table -> %s", csv_path)

    # --- Summary metrics ----------------------------------------------
    summary_rows = []
    for ticker, res in results.items():
        summary_rows.append({
            "ticker": ticker,
            "cagr": res["cagr"],
            "sharpe": res["sharpe"],
            "max_drawdown": res["max_drawdown"],
            "total_return": res["total_return"],
            "volatility": res["volatility"],
        })
    summary_df = pd.DataFrame(summary_rows).set_index("ticker")
    summary_path = os.path.join(out_dir, f"portfolio_summary_{model_name}_{label_version}.csv")
    summary_df.to_csv(summary_path)
    logger.info("Saved summary -> %s", summary_path)

    # --- Combined equity chart ----------------------------------------
    try:
        _plot_combined(results, model_name, label_version, out_dir)
    except Exception:
        logger.exception("Combined equity plot failed")

    return results


def _plot_combined(results, model_name, label_version, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 7))
    for ticker, res in results.items():
        eq = res["equity_curve"]
        if eq is None or eq.empty:
            continue
        ls = "--" if ticker.endswith("_benchmark") else "-"
        lw = 2.0 if ticker.endswith("_benchmark") else 1.2
        ax.plot(eq.index, eq.values, label=ticker, linestyle=ls, linewidth=lw)

    ax.set_title(f"Strategy Equity Curves vs SPY benchmark ({model_name}, label={label_version})")
    ax.set_ylabel("Portfolio value (start=1.0)")
    ax.set_xlabel("Date")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = os.path.join(plots_dir, f"portfolio_combined_{model_name}_{label_version}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved combined equity chart -> %s", out_path)
