"""
Execute backtest and compute portfolio metrics.
"""
import pandas as pd
import numpy as np
import logging
import json
import os

from config import MODEL_DIR, REPORTS_DIR, DATA_DIR, TICKERS, BENCHMARK_TICKER

logger = logging.getLogger(__name__)


def compute_portfolio_metrics(returns: pd.Series, annualization_factor: int = 252) -> dict:
    """
    Compute standard portfolio performance metrics.

    Args:
        returns: Daily return series
        annualization_factor: Trading days per year (252)

    Returns:
        dict with: total_return, annualized_return, sharpe_ratio,
                   max_drawdown, calmar_ratio, volatility
    """
    if len(returns) == 0:
        return {}

    # Total return
    total_return = (1 + returns).prod() - 1

    # Annualized return
    n_years = len(returns) / annualization_factor
    annualized_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0

    # Volatility (annualized)
    volatility = returns.std() * np.sqrt(annualization_factor)

    # Sharpe Ratio (assume risk-free rate = 0 for simplicity)
    sharpe_ratio = annualized_return / volatility if volatility > 0 else 0

    # Max Drawdown
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    # Calmar Ratio
    calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0

    return {
        "total_return": round(float(total_return), 6),
        "annualized_return": round(float(annualized_return), 6),
        "volatility": round(float(volatility), 6),
        "sharpe_ratio": round(float(sharpe_ratio), 6),
        "max_drawdown": round(float(max_drawdown), 6),
        "calmar_ratio": round(float(calmar_ratio), 6),
    }


def run_backtest(ticker: str, model_name: str, label_version: str = "A") -> dict:
    """
    Load saved model, generate predictions on full dataset (out-of-sample last fold),
    compute strategy vs buy-and-hold vs SPY metrics.

    Returns dict with metrics for strategy, buy_hold, and spy.
    """
    import joblib
    import pandas as pd
    from features.pipeline import build_feature_matrix
    from data.download import download_single

    logger.info(f"Running backtest: {ticker} | {model_name} | label_version={label_version}")

    # Load feature matrix and labels
    X, y = build_feature_matrix(ticker, label_version)

    # Use last 20% of data as out-of-sample test period
    split_idx = int(len(X) * 0.8)
    X_test = X.iloc[split_idx:]
    y_test = y.iloc[split_idx:]

    # Load scaler
    scaler_path = os.path.join(MODEL_DIR, f"scaler_{ticker}_{label_version}.joblib")
    if not os.path.exists(scaler_path):
        logger.error(f"Scaler not found: {scaler_path}. Run train first.")
        return {}
    scaler = joblib.load(scaler_path)
    X_test_scaled = scaler.transform(X_test)

    # Load model
    if model_name == "xgboost":
        ext = ".json"
        model_path = os.path.join(MODEL_DIR, f"{model_name}_{ticker}_{label_version}{ext}")
    elif model_name == "lightgbm":
        ext = ".txt"
        model_path = os.path.join(MODEL_DIR, f"{model_name}_{ticker}_{label_version}{ext}")
    else:
        model_path = os.path.join(MODEL_DIR, f"{model_name}_{ticker}_{label_version}.joblib")

    if not os.path.exists(model_path):
        logger.error(f"Model not found: {model_path}. Run train first.")
        return {}

    if model_name == "xgboost":
        import xgboost as xgb
        model = xgb.XGBClassifier()
        model.load_model(model_path)
    elif model_name == "lightgbm":
        import lightgbm as lgb
        model = lgb.Booster(model_file=model_path)
        # For LightGBM Booster, use predict directly
        preds_raw = model.predict(X_test_scaled)
        predictions = pd.Series((preds_raw > 0.5).astype(int), index=X_test.index)
    else:
        model = joblib.load(model_path)

    if model_name != "lightgbm":
        predictions = pd.Series(model.predict(X_test_scaled), index=X_test.index)

    # Get price series for test period
    raw_df = download_single(ticker)
    price_series = raw_df['Close'].loc[X_test.index]

    # Compute strategy returns
    from backtest.strategy import compute_strategy_returns, compute_buyhold_returns, build_signals
    signals = build_signals(predictions)
    strategy_returns = compute_strategy_returns(price_series, signals)
    buyhold_returns = compute_buyhold_returns(price_series)

    # SPY benchmark
    spy_df = download_single(BENCHMARK_TICKER)
    spy_returns = spy_df['Close'].loc[X_test.index].pct_change().dropna()
    # Align dates
    common_dates = strategy_returns.index.intersection(spy_returns.index)

    results = {
        "ticker": ticker,
        "model": model_name,
        "label_version": label_version,
        "test_start": str(X_test.index[0].date()),
        "test_end": str(X_test.index[-1].date()),
        "n_days": len(X_test),
        "strategy": compute_portfolio_metrics(strategy_returns),
        "buy_hold": compute_portfolio_metrics(buyhold_returns.loc[common_dates]),
        "spy": compute_portfolio_metrics(spy_returns.loc[common_dates]),
    }

    # Save results
    os.makedirs(REPORTS_DIR, exist_ok=True)
    out_path = os.path.join(REPORTS_DIR, f"backtest_{ticker}_{model_name}_{label_version}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Backtest results saved to {out_path}")
    logger.info(f"Strategy total return: {results['strategy'].get('total_return', 'N/A'):.4f}")
    logger.info(f"Buy&Hold total return: {results['buy_hold'].get('total_return', 'N/A'):.4f}")
    logger.info(f"SPY total return:      {results['spy'].get('total_return', 'N/A'):.4f}")

    return results
