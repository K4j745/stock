#!/usr/bin/env python3
"""
Stock ML Pipeline - CLI Runner

Usage:
    python main.py train [--ticker TICKER] [--label-version {A,B}] [--refresh] [--tune]
    python main.py backtest [--ticker TICKER] [--model MODEL] [--label-version {A,B}]
    python main.py evaluate [--ticker TICKER] [--label-version {A,B}]
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

from config import TICKERS, logger


def cmd_train(args):
    from models.train import train_all_models
    tickers = [args.ticker] if args.ticker else TICKERS
    for ticker in tickers:
        logger.info(f"=== Training {ticker} | label_version={args.label_version} ===")
        results = train_all_models(ticker, args.label_version, refresh=args.refresh)
        if args.tune:
            from models.tune import tune_and_retrain
            logger.info(f"=== Tuning {ticker} | label_version={args.label_version} ===")
            tune_and_retrain(ticker, args.label_version, n_trials=args.n_trials)
        from reports.generate import export_results
        export_results(results, ticker, args.label_version)


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
        evaluate_saved_models(ticker, args.label_version)


def cmd_report(args):
    from reports.plots import plot_model_comparison
    tickers = [args.ticker] if args.ticker else TICKERS
    for ticker in tickers:
        plot_model_comparison(ticker, args.label_version)


def cmd_shap(args):
    from reports.shap_analysis import run_shap_analysis
    tickers = [args.ticker] if args.ticker else TICKERS
    model = args.model or "xgboost"
    for ticker in tickers:
        run_shap_analysis(ticker, model, args.label_version, tuned=args.tuned)


def cmd_plots(args):
    from reports.plots import plot_equity_curve, plot_confusion_matrix, plot_feature_importance
    tickers = [args.ticker] if args.ticker else TICKERS
    model = args.model or "xgboost"
    for ticker in tickers:
        plot_equity_curve(ticker, model, args.label_version)
        plot_confusion_matrix(ticker, model, args.label_version)
        plot_feature_importance(ticker, model, args.label_version)


def main():
    parser = argparse.ArgumentParser(
        description="Stock ML Pipeline — predict next-day price direction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Shared args
    def add_common(p):
        p.add_argument("--ticker", type=str, default=None, help="Single ticker (default: all from config)")
        p.add_argument("--label-version", type=str, default="A", choices=["A", "B"],
                       help="Label version: A (threshold=0) or B (threshold=0.2%%)")

    # train
    p_train = subparsers.add_parser("train", help="Train all models and save to disk")
    add_common(p_train)
    p_train.add_argument("--refresh", action="store_true", help="Re-download data from yfinance")
    p_train.add_argument("--tune", action="store_true", help="Run Optuna hyperparameter tuning after training")
    p_train.add_argument("--n-trials", type=int, default=50, help="Number of Optuna trials (default: 50)")

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
        "backtest": cmd_backtest,
        "evaluate": cmd_evaluate,
        "report": cmd_report,
        "shap": cmd_shap,
        "plots": cmd_plots,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
