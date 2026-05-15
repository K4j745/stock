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
├── retrain.py                 # monthly retrain entrypoint
├── predict.py                 # daily prediction entrypoint
│
├── data/
│   ├── __init__.py
│   ├── download.py            # yfinance download and Parquet+CSV caching
│   ├── preprocess.py          # cleaning, NaN handling, validation
│   └── raw/                   # cached Parquet + CSV data (gitignored)
│
├── features/
│   ├── __init__.py
│   ├── indicators.py          # technical indicators (ta library) + VIX/earnings/candlestick
│   ├── labels.py              # label generation (binary / multiclass / legacy)
│   ├── validation.py          # rolling walk-forward fold generator
│   ├── external.py            # VIX + earnings-window dummy features
│   ├── candlestick.py         # binary candlestick pattern features
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
│   ├── portfolio.py           # multi-ticker portfolio simulator + SPY benchmark
│   └── run.py                 # single-ticker backtest execution
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

## Label Modes

- **binary** (default): label = 1 if next-day return > `LABEL_THRESHOLD` (0.5%), else 0.
- **multiclass**: {-1, 0, +1} for down / hold / up around the threshold. Trains models with `class_weight='balanced'`.
- **legacy**: original version A (>= 0) or B (> 0.2%) for back-compat.

## Cross-Validation

Rolling-window walk-forward validation via `features.validation.get_time_series_folds`:
`train_size=252`, `test_size=21`, `step=21`, `mode='rolling'` (fixed-size training window). Produces ~10–12 folds on three years of data. StandardScaler is fit only on training data per fold.

## Hyperparameter Tuning

Optuna-based Bayesian optimization for XGBoost and LightGBM. Tuned models saved with `_tuned` suffix.

## SHAP Analysis

TreeExplainer-based SHAP analysis for tree models (XGBoost, LightGBM, Random Forest). Generates summary, bar, and waterfall plots.

## Portfolio Metrics

Backtest computes: total return, annualized return, Sharpe ratio, max drawdown, Calmar ratio, and volatility - compared against buy-and-hold and SPY benchmark.
