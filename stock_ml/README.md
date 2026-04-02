# Stock ML - Stock Price Direction Prediction

A modular ML system for predicting next-day stock price direction using technical indicators and binary classification. Built as an engineering thesis project (BSc in CS & Data Engineering).

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Train all models for a single ticker (label version A)
python main.py train --ticker JNJ --label-version A

# Train all models for a single ticker (label version B, 0.2% threshold)
python main.py train --ticker JNJ --label-version B

# Train all models for all tickers
python main.py train --label-version A

# Force re-download of data
python main.py train --ticker JNJ --label-version A --refresh
```

## Project Structure

```
stock_ml/
├── config.py                  # paths, parameters, Colab/local detection
├── main.py                    # CLI runner (argparse): train / evaluate
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
│   ├── evaluate.py            # classification metrics
│   └── saved/                 # trained models (gitignored)
│
├── reports/
│   ├── __init__.py
│   └── generate.py            # CSV/JSON result export
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
