import argparse
import sys
import os

# Ensure the stock_ml directory is on the path so imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import logger, TICKERS
from models.train import train_all_models


def main():
    parser = argparse.ArgumentParser(description="Stock ML Pipeline")
    parser.add_argument("command", choices=["train", "evaluate", "report"])
    parser.add_argument(
        "--ticker", type=str, default=None,
        help="Single ticker symbol (default: all tickers)"
    )
    parser.add_argument(
        "--label-version", type=str, default="A", choices=["A", "B"],
        help="Label version: A (threshold=0) or B (threshold=0.2%%)"
    )
    parser.add_argument(
        "--refresh", action="store_true",
        help="Re-download data from yfinance (ignore cache)"
    )
    args = parser.parse_args()

    tickers = [args.ticker] if args.ticker else TICKERS

    if args.command == "train":
        for ticker in tickers:
            logger.info("=" * 60)
            logger.info("Processing ticker: %s", ticker)
            logger.info("=" * 60)
            results = train_all_models(
                ticker, args.label_version, refresh=args.refresh
            )
        logger.info("All training complete.")

    elif args.command == "evaluate":
        logger.info("Evaluate command - use 'train' to run full pipeline with evaluation.")

    elif args.command == "report":
        logger.info("Report command - use 'train' to run full pipeline with report generation.")


if __name__ == "__main__":
    main()
