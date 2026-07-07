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
# Committed CSV backup of the raw OHLCV history (NOT gitignored), so the thesis
# results stay reproducible even if yfinance data changes or goes offline.
HISTORICAL_DIR = os.path.join(BASE_DIR, "data", "historical")
MODEL_DIR = os.path.join(BASE_DIR, "models", "saved")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

# Ensure directories exist
for d in [DATA_DIR, HISTORICAL_DIR, MODEL_DIR, REPORTS_DIR]:
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
# Binary classification: BUY (1) if next-day return > threshold, else SELL (0).
# The project studies TWO decision thresholds so the thesis can compare a
# permissive vs. a strict "worth buying" bar:
#   T1 = 0.5% (primary / default) — captures any meaningful up move.
#   T2 = 1.0% (strong)            — only clearly strong up moves count as BUY.
LABEL_THRESHOLD = 0.005         # T1: 0.5% - primary binary threshold (default)
LABEL_THRESHOLD_STRONG = 0.010  # T2: 1.0% - strict "strong move" threshold
LABEL_THRESHOLD_B = 0.002       # legacy version B threshold (kept for compat)

# Named thresholds for the two-threshold study (used by the CLI/dashboard).
BINARY_THRESHOLDS = {
    "t05": LABEL_THRESHOLD,         # 0.5%
    "t10": LABEL_THRESHOLD_STRONG,  # 1.0%
}

DEFAULT_LABEL_MODE = "binary"  # 'binary' (default) or 'multiclass'

# Positive class label = BUY, negative class = SELL (binary mode).
CLASS_BUY = 1
CLASS_SELL = 0


def model_tag(label_mode: str, label_version: str, threshold: float = LABEL_THRESHOLD) -> str:
    """Filename tag used to key saved models/scalers on disk.

    Legacy mode keeps the ``A``/``B`` version letter; binary/multiclass modes
    derive a stable tag from the threshold (e.g. 0.5% -> ``bin5``, 1.0% ->
    ``bin10``) so both thresholds can coexist as separate artifacts.
    """
    if label_mode == "legacy":
        return label_version
    return f"bin{int(round(threshold * 1000))}"

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
