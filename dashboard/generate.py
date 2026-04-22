#!/usr/bin/env python3
"""Dashboard data generator.

Reads dashboard/config.json, downloads OHLCV data via yfinance,
computes technical indicators (RSI, MACD, Bollinger Bands, SMA),
generates trading signals, and writes JSON files to docs/data/.
"""

import json
import os
import shutil
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import ta
import yfinance as yf

# Paths — resolve relative to this script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
CONFIG_PATH = os.path.join(SCRIPT_DIR, "config.json")
DOCS_DATA_DIR = os.path.join(REPO_ROOT, "docs", "data")
DOCS_CHARTS_DIR = os.path.join(REPO_ROOT, "docs", "charts", "plots")
REPORTS_PLOTS_DIR = os.path.join(REPO_ROOT, "reports", "plots")


def load_config():
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)


def download_ticker(ticker, period="2y", interval="1d"):
    """Download OHLCV data for a single ticker."""
    print(f"  Downloading {ticker}...")
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if df.empty:
        print(f"  WARNING: No data for {ticker}")
        return None

    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    df.index = pd.to_datetime(df.index)
    return df


def compute_indicators(df):
    """Compute RSI, MACD, Bollinger Bands, SMA20, SMA50."""
    close = df["Close"]

    # RSI (14)
    df["RSI"] = ta.momentum.RSIIndicator(close, window=14).rsi()

    # MACD
    macd_ind = ta.trend.MACD(close)
    df["MACD"] = macd_ind.macd()
    df["MACD_signal"] = macd_ind.macd_signal()
    df["MACD_hist"] = macd_ind.macd_diff()

    # Bollinger Bands (20, 2)
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    df["BB_upper"] = bb.bollinger_hband()
    df["BB_lower"] = bb.bollinger_lband()
    df["BB_middle"] = bb.bollinger_mavg()

    # SMA
    df["SMA20"] = ta.trend.SMAIndicator(close, window=20).sma_indicator()
    df["SMA50"] = ta.trend.SMAIndicator(close, window=50).sma_indicator()

    return df


def compute_signal(row, prev_row, config):
    """Compute BUY/SELL/HOLD signal based on scoring system."""
    sig_cfg = config["signals"]
    buy_score = 0
    sell_score = 0

    # RSI conditions
    if pd.notna(row["RSI"]):
        if row["RSI"] < sig_cfg["rsi_buy"]:
            buy_score += 1
        if row["RSI"] > sig_cfg["rsi_sell"]:
            sell_score += 1

    # MACD crossover
    if pd.notna(row["MACD"]) and pd.notna(row["MACD_signal"]):
        if prev_row is not None and pd.notna(prev_row["MACD"]) and pd.notna(prev_row["MACD_signal"]):
            # BUY: MACD crosses above signal
            if prev_row["MACD"] <= prev_row["MACD_signal"] and row["MACD"] > row["MACD_signal"]:
                buy_score += 1
            # SELL: MACD crosses below signal
            if prev_row["MACD"] >= prev_row["MACD_signal"] and row["MACD"] < row["MACD_signal"]:
                sell_score += 1

    # Bollinger Band conditions
    if pd.notna(row["BB_lower"]) and pd.notna(row["BB_upper"]):
        if row["Close"] < row["BB_lower"]:
            buy_score += 1
        if row["Close"] > row["BB_upper"]:
            sell_score += 1

    # SMA conditions
    if pd.notna(row["SMA20"]) and pd.notna(row["SMA50"]):
        if row["Close"] > row["SMA20"] and row["Close"] > row["SMA50"]:
            buy_score += 1
        if row["Close"] < row["SMA20"] and row["Close"] < row["SMA50"]:
            sell_score += 1

    if buy_score >= sig_cfg["buy_threshold"]:
        return "BUY", buy_score
    elif sell_score >= sig_cfg["sell_threshold"]:
        return "SELL", sell_score
    else:
        return "HOLD", max(buy_score, sell_score)


def generate_ohlcv_json(df, ticker):
    """Write OHLCV + indicators JSON for a ticker."""
    records = []
    for idx, row in df.iterrows():
        record = {
            "date": idx.strftime("%Y-%m-%d"),
            "open": round(float(row["Open"]), 2) if pd.notna(row["Open"]) else None,
            "high": round(float(row["High"]), 2) if pd.notna(row["High"]) else None,
            "low": round(float(row["Low"]), 2) if pd.notna(row["Low"]) else None,
            "close": round(float(row["Close"]), 2) if pd.notna(row["Close"]) else None,
            "volume": int(row["Volume"]) if pd.notna(row["Volume"]) else None,
            "rsi": round(float(row["RSI"]), 2) if pd.notna(row["RSI"]) else None,
            "macd": round(float(row["MACD"]), 4) if pd.notna(row["MACD"]) else None,
            "macd_signal": round(float(row["MACD_signal"]), 4) if pd.notna(row["MACD_signal"]) else None,
            "macd_hist": round(float(row["MACD_hist"]), 4) if pd.notna(row["MACD_hist"]) else None,
            "bb_upper": round(float(row["BB_upper"]), 2) if pd.notna(row["BB_upper"]) else None,
            "bb_lower": round(float(row["BB_lower"]), 2) if pd.notna(row["BB_lower"]) else None,
            "bb_middle": round(float(row["BB_middle"]), 2) if pd.notna(row["BB_middle"]) else None,
            "sma20": round(float(row["SMA20"]), 2) if pd.notna(row["SMA20"]) else None,
            "sma50": round(float(row["SMA50"]), 2) if pd.notna(row["SMA50"]) else None,
        }
        records.append(record)

    out_path = os.path.join(DOCS_DATA_DIR, f"{ticker}_ohlcv.json")
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"  Wrote {out_path} ({len(records)} records)")
    return records


def generate_signal_entry(df, ticker, config):
    """Generate the signal summary for the latest data point."""
    if len(df) < 2:
        return None

    last = df.iloc[-1]
    prev = df.iloc[-2]
    signal, score = compute_signal(last, prev, config)

    # Compute daily change
    prev_close = prev["Close"]
    curr_close = last["Close"]
    change_pct = ((curr_close - prev_close) / prev_close * 100) if prev_close else 0

    return {
        "ticker": ticker,
        "price": round(float(curr_close), 2),
        "change_pct": round(float(change_pct), 2),
        "rsi": round(float(last["RSI"]), 2) if pd.notna(last["RSI"]) else None,
        "macd": round(float(last["MACD"]), 4) if pd.notna(last["MACD"]) else None,
        "macd_signal_value": round(float(last["MACD_signal"]), 4) if pd.notna(last["MACD_signal"]) else None,
        "signal": signal,
        "score": score,
        "date": df.index[-1].strftime("%Y-%m-%d"),
    }


def copy_report_plots():
    """Copy PNG plots from reports/plots/ to docs/charts/plots/ if they exist."""
    if not os.path.isdir(REPORTS_PLOTS_DIR):
        print("  No reports/plots/ directory found — skipping plot copy.")
        return []

    os.makedirs(DOCS_CHARTS_DIR, exist_ok=True)
    copied = []
    for fname in sorted(os.listdir(REPORTS_PLOTS_DIR)):
        if fname.lower().endswith(".png"):
            src = os.path.join(REPORTS_PLOTS_DIR, fname)
            dst = os.path.join(DOCS_CHARTS_DIR, fname)
            shutil.copy2(src, dst)
            copied.append(fname)
    print(f"  Copied {len(copied)} plot(s) to docs/charts/plots/")
    return copied


def main():
    print("=" * 60)
    print("STOCK TERMINAL — Dashboard Data Generator")
    print("=" * 60)

    config = load_config()
    tickers = config["tickers"]
    all_tickers = tickers + [config["benchmark"]]

    os.makedirs(DOCS_DATA_DIR, exist_ok=True)

    signals = []
    ticker_list = []

    for ticker in all_tickers:
        print(f"\n[{ticker}]")
        df = download_ticker(
            ticker,
            period=config.get("data_period", "2y"),
            interval=config.get("data_interval", "1d"),
        )
        if df is None:
            continue

        df = compute_indicators(df)
        generate_ohlcv_json(df, ticker)

        entry = generate_signal_entry(df, ticker, config)
        if entry:
            signals.append(entry)
            ticker_list.append(ticker)

    # Write signals.json
    signals_path = os.path.join(DOCS_DATA_DIR, "signals.json")
    with open(signals_path, "w") as f:
        json.dump(signals, f, indent=2)
    print(f"\nWrote {signals_path} ({len(signals)} signals)")

    # Write tickers.json
    tickers_path = os.path.join(DOCS_DATA_DIR, "tickers.json")
    with open(tickers_path, "w") as f:
        json.dump(ticker_list, f, indent=2)
    print(f"Wrote {tickers_path}")

    # Write last_updated.json
    now = datetime.now(timezone.utc)
    last_updated = {
        "timestamp": now.isoformat(),
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S UTC"),
    }
    last_updated_path = os.path.join(DOCS_DATA_DIR, "last_updated.json")
    with open(last_updated_path, "w") as f:
        json.dump(last_updated, f, indent=2)
    print(f"Wrote {last_updated_path}")

    # Copy report plots
    print("\n[PLOTS]")
    copy_report_plots()

    print("\n" + "=" * 60)
    print("Dashboard data generation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
