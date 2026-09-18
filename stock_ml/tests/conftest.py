"""Pytest configuration: make ``stock_ml`` importable in tests.

The package uses flat imports (``from config import ...``,
``from features.candle_features import ...``) that assume ``stock_ml`` is on
``sys.path`` — main.py does this at runtime. Tests replicate that here.
"""
import os
import sys

STOCK_ML_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if STOCK_ML_DIR not in sys.path:
    sys.path.insert(0, STOCK_ML_DIR)
