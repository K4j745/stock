#!/usr/bin/env python3
"""Extra thesis tables built from the out-of-sample predictions in thesis_results/raw.

pooled  precision, recall, F1 and MCC from one confusion matrix summed over all
        folds, next to the accuracy of a classifier that always predicts SELL
costs   the backtest of thesis_results.py with 0, 5 and 10 bps per half-turn
random  random strategies with each model's exposure and number of trades
shap    SHAP plots for LightGBM and a top-10 feature comparison with XGBoost

Run ``python thesis_results.py collect`` for every ticker and threshold first.

Usage:
    python thesis_extras.py {pooled,costs,random,shap,all}
"""
import argparse
import logging
import os
import shutil
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import BENCHMARK, TICKERS  # noqa: E402
from data.download import download_single  # noqa: E402
from data.preprocess import clean_data  # noqa: E402
from thesis_results import (  # noqa: E402
    BACKTEST_ORDER, MODEL_ORDER, OUT_DIR, THRESHOLDS, _load, _portfolio_activity, _strategy_metrics, _tag,
)

logger = logging.getLogger("stock_ml")

COST_BPS = (0, 5, 10)
N_DRAWS = 1000
SEED = 20261120
TRADING_DAYS = 252
SHAP_THRESHOLDS = (0.002, 0.005)  # the dashboard models exist only for these
SHAP_MODELS = ("xgboost", "lightgbm")
TOP_FEATURES = 10
SHAP_PLOTS_DIR = os.path.join(OUT_DIR, "wykresy_shap_lightgbm")


# --- pooled classification metrics -------------------------------------------

def _confusion(y_true: pd.Series, y_pred: pd.Series) -> dict:
    return {
        "tp": int(((y_pred == 1) & (y_true == 1)).sum()),
        "fp": int(((y_pred == 1) & (y_true == 0)).sum()),
        "fn": int(((y_pred == 0) & (y_true == 1)).sum()),
        "tn": int(((y_pred == 0) & (y_true == 0)).sum()),
    }


def _ratio(numerator: float, denominator: float) -> float:
    """NaN instead of 0 when the metric is undefined (e.g. no BUY prediction at all)."""
    return numerator / denominator if denominator else float("nan")


def _pooled_metrics(c: dict) -> dict:
    tp, fp, fn, tn = (float(c[k]) for k in ("tp", "fp", "fn", "tn"))
    n = tp + fp + fn + tn
    mcc_den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {
        **c,
        "n": int(n),
        "accuracy": _ratio(tp + tn, n),
        "precision": _ratio(tp, tp + fp),
        "recall": _ratio(tp, tp + fn),
        "f1": _ratio(2 * tp, 2 * tp + fp + fn),
        "mcc": _ratio(tp * tn - fp * fn, mcc_den),
        "always_sell_accuracy": _ratio(tn + fp, n),
        "buy_share_true": _ratio(tp + fn, n),
        "buy_share_predicted": _ratio(tp + fp, n),
    }


def _evaluated_rows(table: pd.DataFrame, model: str) -> pd.DataFrame:
    """The days on which the model made an out-of-sample prediction with a known outcome."""
    rows = table.dropna(subset=[f"{model}_pred", "next_return"])
    return pd.DataFrame({"y_true": rows["y_true"].astype(int), "y_pred": rows[f"{model}_pred"].astype(int)})


def _fold_mean_auc(folds: dict, model: str) -> float:
    return float(pd.Series([f["roc_auc"] for f in folds["folds"][model]], dtype=float).mean())


def pooled() -> None:
    per_ticker, summary, baseline = [], [], []
    for threshold in THRESHOLDS:
        tag = _tag(threshold)
        folds, preds = _load(tag)
        for model in MODEL_ORDER:
            total = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
            aucs = []
            for ticker in TICKERS:
                rows = _evaluated_rows(preds[ticker], model)
                c = _confusion(rows["y_true"], rows["y_pred"])
                total = {k: total[k] + c[k] for k in total}
                aucs.append(_fold_mean_auc(folds[ticker], model))
                per_ticker.append({"tag": tag, "model": model, "ticker": ticker,
                                   **_pooled_metrics(c), "roc_auc_fold_mean": aucs[-1]})
            summary.append({"tag": tag, "model": model, **_pooled_metrics(total),
                            "roc_auc_fold_mean": float(np.mean(aucs))})
        # The four main models share the same evaluation days; the always-SELL
        # classifier is scored on those days.
        for ticker in TICKERS + ["RAZEM"]:
            tickers = TICKERS if ticker == "RAZEM" else [ticker]
            y = pd.concat([_evaluated_rows(preds[t], "logistic_regression")["y_true"] for t in tickers])
            baseline.append({"tag": tag, "ticker": ticker, "n": len(y),
                             "buy_share": float(y.mean()), "always_sell_accuracy": float(1 - y.mean())})

    _write(pd.DataFrame(summary), "klasyfikacja_laczna_podsumowanie.csv")
    _write(pd.DataFrame(per_ticker), "klasyfikacja_laczna_spolki.csv")
    _write(pd.DataFrame(baseline), "klasyfikator_zawsze_sell.csv")


# --- backtest with transaction costs ------------------------------------------

def _common_rows(preds_by_ticker: dict) -> dict:
    """Per ticker, the days on which every model has a prediction (as in thesis_results)."""
    cols = [f"{m}_pred" for m in BACKTEST_ORDER] + ["next_return"]
    return {t: table.dropna(subset=cols) for t, table in preds_by_ticker.items()}


def _turnover(position: pd.Series) -> pd.Series:
    """Half-turns per day: entering or leaving a position is one, opening on day one counts too."""
    return position.diff().fillna(position.iloc[0]).abs()


def _net_returns(position: pd.Series, next_return: pd.Series, cost_bps: float) -> pd.Series:
    return position * next_return - cost_bps / 1e4 * _turnover(position)


def _portfolio(series_by_ticker: list, period: pd.Index) -> pd.Series:
    return pd.concat(series_by_ticker, axis=1).reindex(period).mean(axis=1)


def _period(common: dict) -> pd.Index:
    return pd.concat([t["next_return"] for t in common.values()], axis=1).dropna().index


def _spy_returns(period: pd.Index) -> pd.Series:
    close = clean_data(download_single(BENCHMARK))["Close"]
    return close.pct_change().shift(-1).reindex(period).dropna()


def costs() -> None:
    summary, per_ticker = [], []
    for threshold in THRESHOLDS:
        tag = _tag(threshold)
        common = _common_rows(_load(tag)[1])
        period = _period(common)
        spy = _spy_returns(period)
        for cost in COST_BPS:
            base = {"tag": tag, "cost_bps": cost}
            summary.append({**base, "strategy": "spy_buy_and_hold",
                            **_strategy_metrics(_net_returns(pd.Series(1.0, index=spy.index), spy, cost))})
            strategies = {"buy_and_hold": lambda t: pd.Series(1.0, index=t.index)}
            strategies.update({m: (lambda t, m=m: t[f"{m}_pred"]) for m in BACKTEST_ORDER})
            for name, position_of in strategies.items():
                series, positions = [], []
                for ticker, table in common.items():
                    position = position_of(table)
                    returns = _net_returns(position, table["next_return"], cost).rename(ticker)
                    series.append(returns)
                    positions.append(position)
                    per_ticker.append({**base, "ticker": ticker, "strategy": name,
                                       **_strategy_metrics(returns, position)})
                label = "basket_buy_and_hold" if name == "buy_and_hold" else name
                summary.append({**base, "strategy": label, **_strategy_metrics(_portfolio(series, period)),
                                **_portfolio_activity(positions)})
    _write(pd.DataFrame(summary), "backtest_koszty_podsumowanie.csv")
    _write(pd.DataFrame(per_ticker), "backtest_koszty_spolki.csv")


# --- random strategies with the model's exposure -------------------------------

def _runs(position: np.ndarray) -> tuple:
    """Lengths of the in-position runs and of the cash runs."""
    starts = np.concatenate([[0], np.flatnonzero(np.diff(position)) + 1])
    lengths = np.diff(np.concatenate([starts, [len(position)]]))
    states = position[starts]
    return lengths[states == 1], lengths[states == 0]


def _shuffled_runs(ones: np.ndarray, zeros: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Same runs in a random order: the exposure and the number of trades stay the same."""
    if len(ones) == len(zeros):
        first = int(rng.integers(2))
    else:
        first = int(len(ones) > len(zeros))
    lead, follow = (rng.permutation(ones), rng.permutation(zeros)) if first else \
                   (rng.permutation(zeros), rng.permutation(ones))
    lengths = np.empty(len(lead) + len(follow), dtype=int)
    lengths[0::2], lengths[1::2] = lead, follow
    states = np.zeros(len(lengths), dtype=float)
    states[0::2] = first
    states[1::2] = 1 - first
    return np.repeat(states, lengths)


def _path_metrics(returns: np.ndarray) -> dict:
    """Sharpe, CAGR and max drawdown for each row, defined as in compute_portfolio_metrics."""
    returns = np.atleast_2d(returns)
    growth = np.exp(np.log1p(returns).sum(axis=1))
    cumulative = np.cumprod(1 + returns, axis=1)
    drawdown = cumulative / np.maximum.accumulate(cumulative, axis=1) - 1
    std = returns.std(axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sharpe = np.where(std > 0, returns.mean(axis=1) / std * np.sqrt(TRADING_DAYS), 0.0)
    return {
        "sharpe": sharpe,
        "cagr": growth ** (TRADING_DAYS / returns.shape[1]) - 1,
        "max_drawdown": drawdown.min(axis=1),
    }


def _shuffled_positions(position: np.ndarray, rng: np.random.Generator, _offsets) -> np.ndarray:
    """N_DRAWS run shuffles of one ticker, each drawn independently of the other tickers."""
    ones, zeros = _runs(position)
    return np.stack([_shuffled_runs(ones, zeros, rng) for _ in range(N_DRAWS)])


def _shifted_positions(position: np.ndarray, _rng, offsets: np.ndarray) -> np.ndarray:
    """The model's own positions moved by the same circular lag on every ticker."""
    index = (np.arange(len(position))[None, :] - offsets[:, None]) % len(position)
    return position[index]


# Two nulls with the model's exposure and number of trades. Shuffling runs per
# ticker also breaks the model's habit of holding many tickers at once, so it
# lowers the portfolio's volatility; one lag shared by all tickers keeps that.
NULLS = {"shared_shift": _shifted_positions, "run_shuffle": _shuffled_positions}
MIN_SHIFT = 21  # skip lags shorter than one test window


def _random_paths(common: dict, model: str, period: pd.Index, rng: np.random.Generator, null) -> tuple:
    """Gross returns and half-turns of N_DRAWS random portfolios, summed over tickers."""
    shortest = min(len(t) for t in common.values())
    offsets = rng.choice(np.arange(MIN_SHIFT, shortest - MIN_SHIFT), size=N_DRAWS, replace=False)
    gross = np.zeros((N_DRAWS, len(period)))
    turns = np.zeros((N_DRAWS, len(period)))
    for table in common.values():
        positions = null(table[f"{model}_pred"].to_numpy(), rng, offsets)
        keep = table.index.isin(period)
        change = np.abs(np.diff(positions, axis=1, prepend=0.0))
        gross += positions[:, keep] * table["next_return"].to_numpy()[keep]
        turns += change[:, keep]
    return gross, turns


def _model_paths(common: dict, model: str, period: pd.Index) -> tuple:
    gross, turns = np.zeros(len(period)), np.zeros(len(period))
    for table in common.values():
        keep = table.index.isin(period)
        position = table[f"{model}_pred"]
        gross += (position * table["next_return"]).to_numpy()[keep]
        turns += _turnover(position).to_numpy()[keep]
    return gross, turns


def random_baseline() -> None:
    # The models keep the stream they had before the rules were added, so their
    # draws do not change; the rules get a stream of their own.
    model_rng = np.random.default_rng(SEED)
    rules_rng = np.random.default_rng(SEED + 1)
    rows = []
    for threshold in THRESHOLDS:
        tag = _tag(threshold)
        common = _common_rows(_load(tag)[1])
        period = _period(common)
        n_tickers = len(common)
        for model in BACKTEST_ORDER:
            model_gross, model_turns = _model_paths(common, model, period)
            exposure = float(np.mean([t[f"{model}_pred"].mean() for t in common.values()]))
            for null_name, null in NULLS.items():
                rng = model_rng if model in MODEL_ORDER else rules_rng
                gross, turns = _random_paths(common, model, period, rng, null)
                for cost in COST_BPS:
                    c = cost / 1e4
                    rows.append({
                        "tag": tag, "model": model, "null": null_name, "cost_bps": cost,
                        "exposure": round(exposure, 4), "draws": N_DRAWS,
                        **_compare(_path_metrics((model_gross - c * model_turns) / n_tickers),
                                   _path_metrics((gross - c * turns) / n_tickers)),
                    })
            logger.info("Random baseline %s %s done", tag, model)
    _write(pd.DataFrame(rows), "losowe_strategie.csv")


def _compare(model_metrics: dict, random_metrics: dict) -> dict:
    row = {}
    for metric, values in random_metrics.items():
        own = float(model_metrics[metric][0])
        row[f"{metric}_model"] = own
        row[f"{metric}_random_mean"] = float(values.mean())
        row[f"{metric}_random_p5"] = float(np.percentile(values, 5))
        row[f"{metric}_random_p50"] = float(np.percentile(values, 50))
        row[f"{metric}_random_p95"] = float(np.percentile(values, 95))
        row[f"{metric}_percentile"] = float(100 * (values <= own).mean())
    return row


# --- SHAP for LightGBM ---------------------------------------------------------

def shap_comparison() -> None:
    from reports.shap_analysis import PLOTS_DIR, run_shap_analysis

    os.makedirs(SHAP_PLOTS_DIR, exist_ok=True)
    rows = []
    for threshold in SHAP_THRESHOLDS:
        tag = _tag(threshold)
        for ticker in TICKERS:
            top = {}
            for model in SHAP_MODELS:
                importance = run_shap_analysis(ticker, model, tag, label_mode="binary", threshold=threshold)
                if importance is None:
                    raise FileNotFoundError(f"No saved {model} model for {ticker} {tag}")
                top[model] = importance.head(TOP_FEATURES)
            for rank in range(TOP_FEATURES):
                rows.append({
                    "tag": tag, "ticker": ticker, "rank": rank + 1,
                    **{f"{m}_feature": top[m].index[rank] for m in SHAP_MODELS},
                    **{f"{m}_mean_abs_shap": float(top[m].iloc[rank]) for m in SHAP_MODELS},
                })
            for kind in ("summary", "bar"):
                name = f"shap_{kind}_{ticker}_lightgbm_{tag}.png"
                shutil.copy2(os.path.join(PLOTS_DIR, name), os.path.join(SHAP_PLOTS_DIR, name))

    table = pd.DataFrame(rows)
    _write(table, "shap_top10_xgboost_lightgbm.csv")
    _write(_shap_overlap(table), "shap_top10_zgodnosc.csv")


def _shap_overlap(table: pd.DataFrame) -> pd.DataFrame:
    """How many of the top-10 features the two models share, per ticker and threshold."""
    rows = []
    for (tag, ticker), group in table.groupby(["tag", "ticker"], sort=False):
        xgb, lgbm = set(group["xgboost_feature"]), set(group["lightgbm_feature"])
        rows.append({"tag": tag, "ticker": ticker, "shared_top10": len(xgb & lgbm),
                     "shared_features": ", ".join(sorted(xgb & lgbm)),
                     "same_top1": group.iloc[0]["xgboost_feature"] == group.iloc[0]["lightgbm_feature"]})
    return pd.DataFrame(rows)


# --- entry point -----------------------------------------------------------------

def _write(table: pd.DataFrame, name: str) -> None:
    path = os.path.join(OUT_DIR, name)
    table.to_csv(path, index=False, float_format="%.6g")
    logger.info("Wrote %s (%d rows)", path, len(table))


COMMANDS = {"pooled": pooled, "costs": costs, "random": random_baseline, "shap": shap_comparison}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=[*COMMANDS, "all"])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    for name, command in COMMANDS.items():
        if args.command in (name, "all"):
            command()


if __name__ == "__main__":
    main()
