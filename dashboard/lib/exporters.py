"""Static-file writers for the new ``docs/data/`` layout.

Layout produced::

    docs/data/
        meta.json
        last_updated.json
        registry/
            tickers.json
            models.json
        signals/
            latest.json
            history.json
            by_ticker/{TICKER}.json
            by_model/{MODEL}.json
        tickers/{TICKER}/
            ohlcv.json
            signals.json
            indicators.json
        models/
            index.json
            {MODEL}/summary.json
        portfolios/
            index.json
            {PID}/
                summary.json
                transactions.json
                decisions.json
                equity_curve.json
                positions.json
                metrics.json
                trades.csv
        exports/
            all_signals.csv
            all_transactions.csv
            portfolio_metrics.csv
            model_metrics.csv
        # legacy compatibility (kept so docs/index.html keeps working
        # immediately after a partial deploy):
        signals.json
        tickers.json
        {TICKER}_ohlcv.json
"""
from __future__ import annotations

import csv
import json
import os
from typing import Any, Dict, List

from . import audit


# ---------------------------------------------------------------------------
# Core writers
# ---------------------------------------------------------------------------

def write_meta(out_dir: str, metadata: Dict[str, Any]) -> None:
    audit.write_json(os.path.join(out_dir, "meta.json"), metadata)


def write_last_updated(out_dir: str, metadata: Dict[str, Any]) -> None:
    # mirror the legacy file shape so docs/index.html still works
    audit.write_json(os.path.join(out_dir, "last_updated.json"), {
        "timestamp": metadata["generated_at_utc"],
        "date": metadata["generated_at_utc"][:10],
        "time": metadata["generated_at_utc"][11:19] + " UTC",
        "git_commit": metadata["git_commit"],
        "data_version": metadata["data_version"],
        "schema_version": metadata["schema_version"],
    })


def write_registry(out_dir: str, registry_dict: Dict[str, Any]) -> None:
    base = os.path.join(out_dir, "registry")
    audit.write_json(os.path.join(base, "tickers.json"), registry_dict["tickers"])
    audit.write_json(os.path.join(base, "models.json"), registry_dict["models"])
    audit.write_json(os.path.join(base, "benchmark.json"), {"benchmark": registry_dict["benchmark"]})


# ---------------------------------------------------------------------------
# Tickers
# ---------------------------------------------------------------------------

def write_ticker_ohlcv(out_dir: str, ticker: str, df) -> None:
    """Write OHLCV + indicators for one ticker.

    Writes BOTH the new path ``tickers/{T}/ohlcv.json`` and the legacy path
    ``{T}_ohlcv.json`` for backward compatibility with the current frontend.
    """
    records = _ohlcv_records(df)

    new_path = os.path.join(out_dir, "tickers", ticker, "ohlcv.json")
    audit.write_json(new_path, records)

    legacy_path = os.path.join(out_dir, f"{ticker}_ohlcv.json")
    audit.write_json(legacy_path, records)


def write_ticker_signals(out_dir: str, ticker: str, signals_per_model: Dict[str, List[Dict]]) -> None:
    """Write per-ticker rolled-up signals: ``tickers/{T}/signals.json``."""
    path = os.path.join(out_dir, "tickers", ticker, "signals.json")
    audit.write_json(path, signals_per_model)


def _ohlcv_records(df) -> List[Dict[str, Any]]:
    import pandas as pd

    cols_round = {
        "open": ("Open", 4),
        "high": ("High", 4),
        "low": ("Low", 4),
        "close": ("Close", 4),
        "rsi": ("RSI", 2),
        "macd": ("MACD", 4),
        "macd_signal": ("MACD_signal", 4),
        "macd_hist": ("MACD_hist", 4),
        "bb_upper": ("BB_upper", 4),
        "bb_lower": ("BB_lower", 4),
        "bb_middle": ("BB_middle", 4),
        "sma20": ("SMA20", 4),
        "sma50": ("SMA50", 4),
        "ema12": ("EMA12", 4),
        "ema26": ("EMA26", 4),
        "return_1d": ("return_1d", 6),
        "return_5d": ("return_5d", 6),
        "return_20d": ("return_20d", 6),
    }

    records = []
    for idx, row in df.iterrows():
        rec: Dict[str, Any] = {"date": idx.strftime("%Y-%m-%d")}
        for out_key, (col, prec) in cols_round.items():
            if col in df.columns:
                v = row[col]
                rec[out_key] = round(float(v), prec) if pd.notna(v) else None
        if "Volume" in df.columns:
            v = row["Volume"]
            rec["volume"] = int(v) if pd.notna(v) else None
        records.append(rec)
    return records


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------

def write_signals_bundle(out_dir: str, signal_records: List[Dict[str, Any]]) -> None:
    """Write the rich signals dataset to the new ``signals/`` subtree.

    Also writes a legacy ``signals.json`` containing the latest technical
    signal per ticker so docs/index.html keeps working unmodified.
    """
    base = os.path.join(out_dir, "signals")
    audit.write_json(os.path.join(base, "history.json"), signal_records)

    # latest per (ticker, model)
    latest: Dict[str, Dict[str, Any]] = {}
    for rec in signal_records:
        key = f"{rec['ticker']}|{rec['model_name']}"
        prev = latest.get(key)
        if prev is None or rec["signal_date"] > prev["signal_date"]:
            latest[key] = rec
    audit.write_json(os.path.join(base, "latest.json"), list(latest.values()))

    # group by ticker / model
    by_ticker: Dict[str, List[Dict[str, Any]]] = {}
    by_model: Dict[str, List[Dict[str, Any]]] = {}
    for rec in signal_records:
        by_ticker.setdefault(rec["ticker"], []).append(rec)
        by_model.setdefault(rec["model_name"], []).append(rec)
    for tk, recs in by_ticker.items():
        audit.write_json(os.path.join(base, "by_ticker", f"{tk}.json"), recs)
    for m, recs in by_model.items():
        audit.write_json(os.path.join(base, "by_model", f"{m}.json"), recs)

    # legacy signals.json (latest technical signal per ticker, in the old shape)
    legacy = []
    for v in latest.values():
        if v["model_name"] != "technical_rule_based":
            continue
        legacy.append({
            "ticker": v["ticker"],
            "price": v.get("close"),
            "change_pct": _change_pct(v.get("close"), v.get("previous_close")),
            "rsi": v.get("rsi"),
            "macd": v.get("macd"),
            "macd_signal_value": v.get("macd_signal"),
            "signal": v.get("signal"),
            "score": v.get("score"),
            "date": v.get("signal_date"),
        })
    audit.write_json(os.path.join(out_dir, "signals.json"), legacy)


def _change_pct(close, prev):
    if close is None or prev in (None, 0):
        return None
    try:
        return round((close - prev) / prev * 100, 2)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def write_model_summaries(out_dir: str, model_summaries: Dict[str, Dict]) -> None:
    base = os.path.join(out_dir, "models")
    audit.write_json(os.path.join(base, "index.json"), list(model_summaries.keys()))
    for m, summary in model_summaries.items():
        audit.write_json(os.path.join(base, m, "summary.json"), summary)


# ---------------------------------------------------------------------------
# Portfolios
# ---------------------------------------------------------------------------

def write_portfolio(out_dir: str, result) -> None:
    base = os.path.join(out_dir, "portfolios", result.portfolio_id)

    audit.write_json(os.path.join(base, "summary.json"), {
        "portfolio_id": result.portfolio_id,
        "portfolio_name": result.portfolio_name,
        "description": result.description,
        "initial_capital": result.initial_capital,
        "final_value": result.final_value,
        "cash": result.cash,
        "metadata": result.metadata,
        "metrics": result.metrics,
        "benchmark_metrics": result.benchmark_metrics,
    })
    audit.write_json(os.path.join(base, "decisions.json"), result.decisions)
    audit.write_json(os.path.join(base, "transactions.json"), result.transactions)
    audit.write_json(os.path.join(base, "equity_curve.json"), result.equity_curve)
    audit.write_json(os.path.join(base, "positions.json"), result.positions)
    audit.write_json(os.path.join(base, "metrics.json"), {
        "performance": result.metrics,
        "benchmark": result.benchmark_metrics,
    })
    write_csv(os.path.join(base, "trades.csv"), result.transactions)


def write_portfolio_index(out_dir: str, portfolio_summaries: List[Dict[str, Any]]) -> None:
    audit.write_json(os.path.join(out_dir, "portfolios", "index.json"), portfolio_summaries)


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    """Write a list of dicts to CSV. Empty list produces a header-less empty file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("")
        return
    # Union of all keys to be defensive against heterogeneous records.
    keys: List[str] = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _scalar(row.get(k)) for k in keys})


def _scalar(v):
    if v is None:
        return ""
    if isinstance(v, (dict, list)):
        return json.dumps(v, default=str)
    return v


def write_global_exports(out_dir: str, all_signals: List[Dict], all_transactions: List[Dict],
                          portfolio_metrics: List[Dict], model_metrics: List[Dict]) -> None:
    base = os.path.join(out_dir, "exports")
    write_csv(os.path.join(base, "all_signals.csv"), all_signals)
    write_csv(os.path.join(base, "all_transactions.csv"), all_transactions)
    write_csv(os.path.join(base, "portfolio_metrics.csv"), portfolio_metrics)
    write_csv(os.path.join(base, "model_metrics.csv"), model_metrics)
