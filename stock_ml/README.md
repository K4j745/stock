# Stock4caster — Stock Price Direction Prediction

A modular ML system that predicts the **next-day price direction** of a stock as a
**binary decision — BUY vs SELL** — from technical indicators and candlestick-shape
features. Built as an engineering thesis project (praca inżynierska), Uniwersytet
Przyrodniczy w Poznaniu.

## Binary classification (BUY vs SELL)

The label is binary:

- **BUY (1)** — next-day return **>** the decision threshold
- **SELL (0)** — everything else

The project studies **two decision thresholds** so the thesis can compare a
permissive vs. a strict "worth buying" bar:

| Threshold | Value | Tag | Meaning |
|-----------|-------|-----|---------|
| T1 (default) | 0.5% (`0.005`) | `bin5` | any meaningful up move counts as BUY |
| T2 (strong)  | 1.0% (`0.010`) | `bin10` | only clearly strong up moves count as BUY |

Saved models and scalers are keyed by the threshold tag (`bin5` / `bin10`), so both
thresholds coexist on disk as separate artifacts.

Reported metrics are for the **BUY (positive) class**: accuracy, precision, recall,
F1, ROC-AUC, and MCC.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Train models

```bash
# Train all models for one ticker at the default 0.5% threshold
python main.py train --ticker AAPL

# Train at the strict 1.0% threshold
python main.py train --ticker AAPL --threshold 0.01

# Train the whole ticker universe
python main.py train

# Force a fresh yfinance download, then train
python main.py train --ticker AAPL --refresh

# Train with Optuna hyperparameter tuning
python main.py train --ticker AAPL --tune --n-trials 50
```

### Candle model (candlestick-shape signal)

A separate classifier trained **only on candlestick-shape features plus a compact
set of base technical indicators**, to answer: *how much signal is in the raw candle
geometry alone?* It reuses the same leakage-safe walk-forward protocol and the same
binary BUY/SELL labels.

```bash
# Train the candle model (XGBoost by default) at 0.5%
python main.py candle --ticker AAPL

# Random-forest variant at the strict 1.0% threshold
python main.py candle --ticker AAPL --threshold 0.01 --model-type random_forest
```

### Evaluate saved models

```bash
# Evaluate saved models on the last 20% test split at a given threshold
python main.py evaluate --ticker AAPL --threshold 0.005
```

### Backtest / report / SHAP / plots

```bash
python main.py backtest --ticker AAPL --model xgboost
python main.py report   --ticker AAPL
python main.py shap     --ticker AAPL --model xgboost
python main.py plots    --ticker AAPL --model xgboost
```

## Models

- Logistic Regression
- Random Forest (200 estimators)
- XGBoost (200 estimators)
- LightGBM (200 estimators)
- **Candle model** — XGBoost or Random Forest on candle-shape + base-technical features

## Features

### Technical indicators (`features/indicators.py`)

Trend, momentum, volatility, and volume indicators computed with the `ta` library
(SMA, MACD, RSI, ROC, ATR, Volume ratio, HL spread, and more).

### Candlestick-shape features (`features/candle_features.py`)

Geometry of each daily candle, normalised by the open price so they are comparable
across price levels:

| Feature | Definition |
|---------|------------|
| `body_length` | \|Close − Open\| / Open |
| `upper_shadow` | (High − max(Open, Close)) / Open |
| `lower_shadow` | (min(Open, Close) − Low) / Open |
| `close_position` | (Close − Low) / (High − Low), 0.5 on a zero-range candle |
| `doji_flag` | 1 when the body is a tiny fraction of the open |
| `long_lower_shadow_flag` / `long_upper_shadow_flag` | long shadow relative to the body |
| `bullish_engulfing` / `bearish_engulfing` | two-candle engulfing patterns |

## Temporal validation (walk-forward, no leakage)

Rolling-window **walk-forward cross-validation** (`features/validation.py`):

- Fixed rolling train window (`WF_TRAIN_SIZE = 252` ≈ 1 trading year),
  test window (`WF_TEST_SIZE = 21` ≈ 1 month), sliding by `WF_STEP = 21`.
- `StandardScaler` is fit **only on each train fold** — the test fold is never seen
  during scaling or fitting, so there is no look-ahead leakage.
- The `Log_return` column is dropped from the feature matrix because it mirrors the
  next-day return used to build the label; only **lagged** returns are safe features.

## Data source

Daily OHLCV from [yfinance](https://github.com/ranaroussi/yfinance) (free, no API
key). Training data is cached in the gitignored `data/raw/`.

## Historical data (committed backup)

`data/download_historical.py` downloads the full daily history for every ticker in
the universe (plus the SPY benchmark) and writes one CSV per ticker into
`data/historical/`. These CSVs are **committed to the repository** (not gitignored),
so the thesis stays reproducible even if the yfinance feed changes or goes offline.

```bash
# Refresh the committed CSV backup for the whole universe
python data/download_historical.py

# A single ticker, or a custom start date
python data/download_historical.py --ticker AAPL
python data/download_historical.py --start 2015-01-01
```

Each file is `data/historical/{TICKER}_historical.csv` with columns
`Date, Open, High, Low, Close, Volume, Adj Close` and covers ~15 years of daily bars.

## Dashboard

The static dashboard (GitHub Pages) is generated from `dashboard/generate.py` into
`docs/data/` and served as self-contained HTML pages. See the repository root for
dashboard details.

## Project Structure

```
stock_ml/
├── config.py                  # paths, thresholds, walk-forward params, model_tag()
├── main.py                    # CLI runner (train / candle / evaluate / backtest / …)
│
├── data/
│   ├── download.py            # yfinance download + Parquet/CSV cache (data/raw, gitignored)
│   ├── download_historical.py # committed CSV backup -> data/historical/
│   ├── preprocess.py          # cleaning, NaN handling, validation
│   ├── raw/                   # cached training data (gitignored)
│   └── historical/            # committed OHLCV CSV backup
│
├── features/
│   ├── indicators.py          # technical indicators (ta library)
│   ├── candle_features.py     # candlestick-shape features
│   ├── labels.py              # binary BUY/SELL labels (+ legacy A/B, multiclass)
│   ├── validation.py          # walk-forward fold generation
│   └── pipeline.py            # feature matrix assembly
│
├── models/
│   ├── train.py               # walk-forward training + saving (keyed by threshold tag)
│   ├── candle_model.py        # candlestick model (XGBoost / Random Forest)
│   ├── evaluate.py            # BUY-class metrics + evaluate_saved_models
│   ├── tune.py                # Optuna tuning (XGBoost, LightGBM)
│   └── saved/                 # trained models (gitignored)
│
├── backtest/                  # portfolio backtest + strategy returns
├── reports/                   # CSV/JSON export, plots, SHAP analysis
├── tests/                     # pytest unit tests (candle features, …)
├── requirements.txt
├── .gitignore
└── README.md
```

---

© 2025 Kajetan Kaczyński — Praca inżynierska / Uniwersytet Przyrodniczy w Poznaniu / Wszelkie prawa zastrzeżone.
