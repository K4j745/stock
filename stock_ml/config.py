import os
import logging

# --- Environment detection ---
IS_COLAB = os.path.exists("/content/drive")

if IS_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    BASE_DIR = "/content/drive/MyDrive/stock_ml"
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Paths ---
DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
MODEL_DIR = os.path.join(BASE_DIR, "models", "saved")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

# Ensure directories exist
for d in [DATA_DIR, MODEL_DIR, REPORTS_DIR]:
    os.makedirs(d, exist_ok=True)

# --- Tickers ---
# Diversified ticker universe across sectors. SPY is used ONLY as a benchmark
# for buy-and-hold comparison in the backtest module - it is never used as a
# training input.
TICKERS = [
    "AAPL",  # Technology
    "MSFT",  # Technology
    "JPM",   # Financials
    "XOM",   # Energy
    "JNJ",   # Healthcare
    "UNH",   # Healthcare
    "PG",    # Consumer Staples
    "WMT",   # Consumer Staples
    "KO",    # Consumer Staples
    "NEE",   # Utilities
]
BENCHMARK = "SPY"
BENCHMARK_TICKER = BENCHMARK  # kept for backwards compatibility
ALL_TICKERS = TICKERS + [BENCHMARK]

# --- Data params ---
DATA_START = "2011-04-01"
DATA_END = "2026-04-01"
DATA_INTERVAL = "1d"

# --- Label params ---
# Binary label: 1 if next-day return > LABEL_THRESHOLD, else 0.
LABEL_THRESHOLD = 0.005  # 0.5% - main binary threshold (default mode)
LABEL_THRESHOLD_B = 0.002  # 0.2% for legacy version B
DEFAULT_LABEL_MODE = "binary"  # 'binary' (default) or 'multiclass'

# --- Model params ---
RANDOM_STATE = 42

# --- Walk-forward params ---
# Rolling window walk-forward CV (used by features.validation.get_time_series_folds)
WF_TRAIN_SIZE = 252  # ~1 trading year
WF_TEST_SIZE = 21    # ~1 trading month
WF_STEP = 21         # window slide step
WF_MODE = "rolling"  # 'rolling' (default) or 'expanding'

# Legacy: number of TimeSeriesSplit folds (deprecated, kept for transitional use)
N_SPLITS = 5

# --- Logging ---
def setup_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    return logging.getLogger("stock_ml")

logger = setup_logging()
