"""
Visualization module: equity curves, drawdown, confusion matrices, feature importance,
and model comparison charts. All plots saved as PNG to reports/plots/.
"""
import matplotlib
matplotlib.use('Agg')

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from config import MODEL_DIR, REPORTS_DIR, TICKERS, BENCHMARK_TICKER

logger = logging.getLogger(__name__)

PLOTS_DIR = os.path.join(REPORTS_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


def plot_equity_curve(ticker: str, model_name: str, label_version: str = "A"):
    """
    Plot equity curve: strategy vs. buy-and-hold vs. SPY.
    Loads backtest results JSON.
    """
    import joblib
    from sklearn.preprocessing import StandardScaler
    from data.download import download_single
    from backtest.strategy import compute_strategy_returns, compute_buyhold_returns, build_signals
    from features.pipeline import build_feature_matrix

    # Load model predictions (same logic as backtest/run.py)
    X, y = build_feature_matrix(ticker, label_version)
    split_idx = int(len(X) * 0.8)
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]

    # Re-fit scaler on train portion only to avoid data leakage
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )

    if model_name == "xgboost":
        import xgboost as xgb
        model_path = os.path.join(MODEL_DIR, f"xgboost_{ticker}_{label_version}.json")
        model = xgb.XGBClassifier()
        model.load_model(model_path)
    elif model_name == "lightgbm":
        import lightgbm as lgb
        model_path = os.path.join(MODEL_DIR, f"lightgbm_{ticker}_{label_version}.txt")
        model = lgb.Booster(model_file=model_path)
        preds = (model.predict(X_test_scaled) > 0.5).astype(int)
        predictions = pd.Series(preds, index=X_test.index)
    else:
        model = joblib.load(os.path.join(MODEL_DIR, f"{model_name}_{ticker}_{label_version}.joblib"))

    if model_name != "lightgbm":
        predictions = pd.Series(model.predict(X_test_scaled), index=X_test.index)

    raw_df = download_single(ticker)
    price_series = raw_df['Close'].loc[X_test.index]
    spy_df = download_single(BENCHMARK_TICKER)

    signals = build_signals(predictions)
    strat_ret = compute_strategy_returns(price_series, signals)
    bh_ret = compute_buyhold_returns(price_series)
    spy_ret = spy_df['Close'].loc[X_test.index].pct_change().dropna()

    common = strat_ret.index.intersection(bh_ret.index).intersection(spy_ret.index)

    strat_equity = (1 + strat_ret.loc[common]).cumprod()
    bh_equity = (1 + bh_ret.loc[common]).cumprod()
    spy_equity = (1 + spy_ret.loc[common]).cumprod()

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                              gridspec_kw={"height_ratios": [3, 1]})

    axes[0].plot(strat_equity.index, strat_equity.values, label=f"Strategy ({model_name})", linewidth=1.5)
    axes[0].plot(bh_equity.index, bh_equity.values, label=f"Buy & Hold ({ticker})", linewidth=1.5, linestyle="--")
    axes[0].plot(spy_equity.index, spy_equity.values, label="SPY", linewidth=1.5, linestyle=":")
    axes[0].set_ylabel("Portfolio Value (normalized)")
    axes[0].set_title(f"Equity Curve - {ticker} | {model_name} | label={label_version}")
    axes[0].legend()
    axes[0].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))

    # Drawdown
    rolling_max = strat_equity.cummax()
    drawdown = (strat_equity - rolling_max) / rolling_max
    axes[1].fill_between(drawdown.index, drawdown.values, 0, color="red", alpha=0.4, label="Drawdown")
    axes[1].set_ylabel("Drawdown")
    axes[1].set_xlabel("Date")
    axes[1].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))
    axes[1].legend()

    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, f"equity_curve_{ticker}_{model_name}_{label_version}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Equity curve saved: {out}")


def plot_confusion_matrix(ticker: str, model_name: str, label_version: str = "A"):
    """Plot confusion matrix for a model on the test period."""
    import joblib
    from sklearn.preprocessing import StandardScaler
    from features.pipeline import build_feature_matrix

    X, y = build_feature_matrix(ticker, label_version)
    split_idx = int(len(X) * 0.8)
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_test = y.iloc[split_idx:]

    # Re-fit scaler on train portion only to avoid data leakage
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )

    if model_name == "xgboost":
        import xgboost as xgb
        model = xgb.XGBClassifier()
        model.load_model(os.path.join(MODEL_DIR, f"xgboost_{ticker}_{label_version}.json"))
    elif model_name == "lightgbm":
        import lightgbm as lgb
        booster = lgb.Booster(model_file=os.path.join(MODEL_DIR, f"lightgbm_{ticker}_{label_version}.txt"))
        preds = (booster.predict(X_test_scaled) > 0.5).astype(int)
    else:
        model = joblib.load(os.path.join(MODEL_DIR, f"{model_name}_{ticker}_{label_version}.joblib"))

    if model_name != "lightgbm":
        preds = model.predict(X_test_scaled)

    cm = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Down (0)", "Up (1)"])

    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix\n{ticker} | {model_name} | label={label_version}")
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, f"confusion_matrix_{ticker}_{model_name}_{label_version}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Confusion matrix saved: {out}")


def plot_feature_importance(ticker: str, model_name: str = "random_forest", label_version: str = "A", top_n: int = 20):
    """Plot feature importance from tree-based models."""
    import joblib
    from features.pipeline import build_feature_matrix

    X, y = build_feature_matrix(ticker, label_version)

    if model_name == "xgboost":
        import xgboost as xgb
        model = xgb.XGBClassifier()
        model.load_model(os.path.join(MODEL_DIR, f"xgboost_{ticker}_{label_version}.json"))
        importances = pd.Series(model.feature_importances_, index=X.columns)
    elif model_name == "lightgbm":
        import lightgbm as lgb
        booster = lgb.Booster(model_file=os.path.join(MODEL_DIR, f"lightgbm_{ticker}_{label_version}.txt"))
        importances = pd.Series(booster.feature_importance(importance_type="gain"), index=booster.feature_name())
    else:
        model = joblib.load(os.path.join(MODEL_DIR, f"{model_name}_{ticker}_{label_version}.joblib"))
        importances = pd.Series(model.feature_importances_, index=X.columns)

    importances = importances.nlargest(top_n).sort_values()

    fig, ax = plt.subplots(figsize=(10, max(6, top_n // 2)))
    ax.barh(importances.index, importances.values, color="steelblue", edgecolor="white")
    ax.set_xlabel("Feature Importance")
    ax.set_title(f"Top {top_n} Feature Importances\n{ticker} | {model_name} | label={label_version}")
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, f"feature_importance_{ticker}_{model_name}_{label_version}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Feature importance plot saved: {out}")


def plot_model_comparison(ticker: str, label_version: str = "A"):
    """
    Load metrics from reports JSON and plot bar chart comparing models.
    """
    import json

    results_path = os.path.join(REPORTS_DIR, f"{ticker}_{label_version}_results.json")
    if not os.path.exists(results_path):
        logger.warning(f"Results not found: {results_path}. Run train first.")
        return

    with open(results_path) as f:
        data = json.load(f)

    # Expect structure: {model_name: {mean_metrics: {...}}}
    models = []
    metrics_list = []
    for m_name, model_data in data.items():
        if "mean_metrics" in model_data:
            models.append(m_name)
            metrics_list.append(model_data["mean_metrics"])

    if not models:
        logger.warning("No mean_metrics found in results JSON.")
        return

    df = pd.DataFrame(metrics_list, index=models)
    metric_cols = [c for c in ["accuracy", "f1", "roc_auc", "mcc"] if c in df.columns]

    fig, axes = plt.subplots(1, len(metric_cols), figsize=(4 * len(metric_cols), 5))
    if len(metric_cols) == 1:
        axes = [axes]

    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    for ax, metric, color in zip(axes, metric_cols, colors):
        ax.bar(df.index, df[metric], color=color, edgecolor="white", width=0.6)
        ax.set_title(metric.upper())
        ax.set_ylim(0, 1)
        ax.set_ylabel(metric)
        ax.set_xticks(range(len(df.index)))
        ax.set_xticklabels(df.index, rotation=30, ha="right")
        for i, val in enumerate(df[metric]):
            ax.text(i, val + 0.01, f"{val:.3f}", ha="center", fontsize=9)

    plt.suptitle(f"Model Comparison - {ticker} | label={label_version}", fontsize=13, y=1.02)
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, f"model_comparison_{ticker}_{label_version}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Model comparison plot saved: {out}")
