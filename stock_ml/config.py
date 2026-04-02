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
TICKERS = ["JNJ", "UNH", "ABT", "PG", "KO", "WMT", "NEE", "DUK"]
BENCHMARK_TICKER = "SPY"
ALL_TICKERS = TICKERS + [BENCHMARK_TICKER]

# --- Data params ---
DATA_START = "2011-04-01"
DATA_END = "2026-04-01"
DATA_INTERVAL = "1d"

# --- Label params ---
LABEL_THRESHOLD_B = 0.002  # 0.2% for version B

# --- Model params ---
RANDOM_STATE = 42

# --- Walk-forward params ---
N_SPLITS = 5  # minimum 5 folds for TimeSeriesSplit

# --- Logging ---
def setup_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    return logging.getLogger("stock_ml")

logger = setup_logging()
