#!/usr/bin/env python3
"""
Stock4caster ML Pipeline - CLI Runner

Binary next-day direction (BUY vs SELL) at two decision thresholds
(0.5% and 1.0%). Pick a threshold with --threshold (default 0.005).

Usage:
    python main.py train [--ticker TICKER] [--threshold 0.005|0.01] [--refresh] [--tune]
    python main.py candle [--ticker TICKER] [--threshold 0.005|0.01] [--model-type xgboost|random_forest]
    python main.py backtest [--ticker TICKER] [--model MODEL] [--label-version {A,B}]
    python main.py evaluate [--ticker TICKER] [--threshold 0.005|0.01]
    python main.py report [--ticker TICKER] [--label-version {A,B}]
    python main.py shap [--ticker TICKER] [--model MODEL] [--label-version {A,B}] [--tuned]
    python main.py plots [--ticker TICKER] [--model MODEL] [--label-version {A,B}]
"""
import argparse
import logging
import sys
import os

# Ensure the stock_ml directory is on the path so imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import TICKERS, DEFAULT_LABEL_MODE, LABEL_THRESHOLD, model_tag, logger


def cmd_train(args):
    from models.train import train_all_models
    tickers = [args.ticker] if args.ticker else TICKERS
    for ticker in tickers:
        logger.info(
            f"=== Training {ticker} | label_mode={args.label_mode} | "
            f"threshold={args.threshold} | label_version={args.label_version} ==="
        )
        train_all_models(
            ticker,
            args.label_version,
            refresh=args.refresh,
            label_mode=args.label_mode,
            threshold=args.threshold,
        )
        if args.tune:
            from models.tune import tune_and_retrain
            logger.info(f"=== Tuning {ticker} | label_version={args.label_version} ===")
            tune_and_retrain(ticker, args.label_version, n_trials=args.n_trials)


def cmd_candle(args):
    from models.candle_model import train_candle_model
    tickers = [args.ticker] if args.ticker else TICKERS
    for ticker in tickers:
        logger.info(
            f"=== Candle model {ticker} | threshold={args.threshold} | "
            f"model_type={args.model_type} ==="
        )
        train_candle_model(
            ticker,
            threshold=args.threshold,
            model_type=args.model_type,
            refresh=args.refresh,
        )


def cmd_backtest(args):
    from backtest.run import run_backtest
    tickers = [args.ticker] if args.ticker else TICKERS
    models = [args.model] if args.model else ["logistic_regression", "random_forest", "xgboost", "lightgbm"]
    for ticker in tickers:
        for model_name in models:
            run_backtest(ticker, model_name, args.label_version)


def cmd_evaluate(args):
    from models.evaluate import evaluate_saved_models
    tickers = [args.ticker] if args.ticker else TICKERS
    for ticker in tickers:
        evaluate_saved_models(
            ticker,
            label_mode=args.label_mode,
            label_version=args.label_version,
            threshold=args.threshold,
        )


def cmd_report(args):
    from reports.plots import plot_model_comparison
    # Resolve the on-disk artifact tag (e.g. binary @ 0.5% -> "bin5") so plots
    # load the models that ``train`` actually saved, not the legacy A/B naming.
    tag = model_tag(args.label_mode, args.label_version, args.threshold)
    tickers = [args.ticker] if args.ticker else TICKERS
    for ticker in tickers:
        plot_model_comparison(ticker, tag)


def cmd_shap(args):
    from reports.shap_analysis import run_shap_analysis
    tickers = [args.ticker] if args.ticker else TICKERS
    model = args.model or "xgboost"
    for ticker in tickers:
        run_shap_analysis(ticker, model, args.label_version, tuned=args.tuned)


def cmd_plots(args):
    from reports.plots import plot_equity_curve, plot_confusion_matrix, plot_feature_importance
    # Resolve the on-disk artifact tag (e.g. binary @ 0.5% -> "bin5") so plots
    # load the models that ``train`` actually saved, not the legacy A/B naming.
    tag = model_tag(args.label_mode, args.label_version, args.threshold)
    tickers = [args.ticker] if args.ticker else TICKERS
    model = args.model or "xgboost"
    for ticker in tickers:
        plot_equity_curve(ticker, model, tag)
        plot_confusion_matrix(ticker, model, tag)
        plot_feature_importance(ticker, model, tag)


def main():
    parser = argparse.ArgumentParser(
        description="Stock4caster ML Pipeline - predict next-day BUY/SELL direction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Shared args
    def add_common(p):
        p.add_argument("--ticker", type=str, default=None, help="Single ticker (default: all from config)")
        p.add_argument("--label-version", type=str, default="A", choices=["A", "B"],
                       help="Legacy label version (only used when --label-mode=legacy)")
        p.add_argument("--label-mode", type=str, default=DEFAULT_LABEL_MODE,
                       choices=["binary", "multiclass", "legacy"],
                       help="Label mode (default: binary, threshold=0.5%%)")
        p.add_argument("--threshold", type=float, default=LABEL_THRESHOLD,
                       help="Binary decision threshold on next-day return "
                            "(default: 0.005 = 0.5%%; the study also uses 0.01 = 1.0%%)")

    # train
    p_train = subparsers.add_parser("train", help="Train all models and save to disk")
    add_common(p_train)
    p_train.add_argument("--refresh", action="store_true", help="Re-download data from yfinance")
    p_train.add_argument("--tune", action="store_true", help="Run Optuna hyperparameter tuning after training")
    p_train.add_argument("--n-trials", type=int, default=50, help="Number of Optuna trials (default: 50)")

    # candle
    p_candle = subparsers.add_parser(
        "candle", help="Train the candlestick-shape model (candle + base technical features)")
    add_common(p_candle)
    p_candle.add_argument("--model-type", type=str, default="xgboost",
                          choices=["xgboost", "random_forest"],
                          help="Candle model estimator (default: xgboost)")
    p_candle.add_argument("--refresh", action="store_true", help="Re-download data from yfinance")

    # backtest
    p_bt = subparsers.add_parser("backtest", help="Run portfolio backtest")
    add_common(p_bt)
    p_bt.add_argument("--model", type=str, default=None,
                      help="Model name (default: all). Options: logistic_regression, random_forest, xgboost, lightgbm")

    # evaluate
    p_eval = subparsers.add_parser("evaluate", help="Evaluate saved models on test split")
    add_common(p_eval)

    # report
    p_rep = subparsers.add_parser("report", help="Generate CSV/JSON reports and comparison plots")
    add_common(p_rep)

    # shap
    p_shap = subparsers.add_parser("shap", help="Generate SHAP analysis plots")
    add_common(p_shap)
    p_shap.add_argument("--model", type=str, default="xgboost", help="Model to analyze (default: xgboost)")
    p_shap.add_argument("--tuned", action="store_true", help="Use tuned model variant")

    # plots
    p_plots = subparsers.add_parser("plots", help="Generate all visualization plots")
    add_common(p_plots)
    p_plots.add_argument("--model", type=str, default="xgboost", help="Model for plots (default: xgboost)")

    args = parser.parse_args()

    dispatch = {
        "train": cmd_train,
        "candle": cmd_candle,
        "backtest": cmd_backtest,
        "evaluate": cmd_evaluate,
        "report": cmd_report,
        "shap": cmd_shap,
        "plots": cmd_plots,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
