# Stock ML - Stock Price Direction Prediction

A modular ML system for predicting next-day stock price direction using technical indicators and binary classification. Built as an engineering thesis project (BSc in CS & Data Engineering).

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Train models

```bash
# Train all models for a single ticker (label version A)
python main.py train --ticker JNJ --label-version A

# Train with Optuna hyperparameter tuning
python main.py train --ticker JNJ --label-version A --tune --n-trials 50

# Train all models for all tickers
python main.py train --label-version A

# Force re-download of data
python main.py train --ticker JNJ --label-version A --refresh
```

### Backtest

```bash
# Backtest a single model
python main.py backtest --ticker JNJ --model xgboost --label-version A

# Backtest all models for a ticker
python main.py backtest --ticker JNJ --label-version A
```

### Evaluate saved models

```bash
# Evaluate all saved models on test split (last 20%)
python main.py evaluate --ticker JNJ --label-version A
```

### Report (model comparison plots)

```bash
# Generate model comparison bar charts
python main.py report --ticker JNJ --label-version A
```

### SHAP analysis

```bash
# Generate SHAP plots for XGBoost (default)
python main.py shap --ticker JNJ --label-version A

# SHAP for a specific model
python main.py shap --ticker JNJ --model random_forest --label-version A

# SHAP for a tuned model
python main.py shap --ticker JNJ --model xgboost --label-version A --tuned
```

### Visualization plots

```bash
# Generate equity curve, confusion matrix, feature importance
python main.py plots --ticker JNJ --model xgboost --label-version A
```

## Project Structure

```
stock_ml/
├── config.py                  # paths, parameters, Colab/local detection
├── main.py                    # CLI runner (argparse subparsers)
│
├── data/
│   ├── __init__.py
│   ├── download.py            # yfinance download and Parquet+CSV caching
│   ├── preprocess.py          # cleaning, NaN handling, validation
│   └── raw/                   # cached Parquet + CSV data (gitignored)
│
├── features/
│   ├── __init__.py
│   ├── indicators.py          # technical indicators (ta library)
│   ├── labels.py              # label generation (version A and B)
│   └── pipeline.py            # feature matrix assembly
│
├── models/
│   ├── __init__.py
│   ├── train.py               # training and model saving
│   ├── evaluate.py            # classification metrics + evaluate_saved_models
│   ├── tune.py                # Optuna hyperparameter tuning (XGBoost, LightGBM)
│   └── saved/                 # trained models (gitignored)
│
├── backtest/
│   ├── __init__.py
│   ├── strategy.py            # signals, strategy returns, buy-and-hold returns
│   └── run.py                 # backtest execution and portfolio metrics
│
├── reports/
│   ├── __init__.py
│   ├── generate.py            # CSV/JSON result export
│   ├── plots.py               # equity curves, confusion matrix, feature importance, model comparison
│   ├── shap_analysis.py       # SHAP summary, bar, and waterfall plots
│   └── plots/                 # saved PNG plots (gitignored)
│
├── requirements.txt
├── .gitignore
└── README.md
```

## Models

- Logistic Regression
- Random Forest (200 estimators)
- XGBoost (200 estimators)
- LightGBM (200 estimators)

## Label Versions

- **A**: next-day return >= 0 (up or flat)
- **B**: next-day return > 0.2% (significant upward move)

## Cross-Validation

Uses expanding window (TimeSeriesSplit with 5 folds) to avoid data leakage. StandardScaler is fit only on training data per fold.

## Hyperparameter Tuning

Optuna-based Bayesian optimization for XGBoost and LightGBM. Tuned models saved with `_tuned` suffix.

## SHAP Analysis

TreeExplainer-based SHAP analysis for tree models (XGBoost, LightGBM, Random Forest). Generates summary, bar, and waterfall plots.

## Portfolio Metrics

Backtest computes: total return, annualized return, Sharpe ratio, max drawdown, Calmar ratio, and volatility - compared against buy-and-hold and SPY benchmark.
