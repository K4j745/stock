"""Configuration loader and validator for ``dashboard/config.json``.

Accepts the new v2.0 schema (rich ticker objects + portfolios) and gracefully
falls back to the legacy v1 layout (flat list of ticker strings, no portfolios)
so old configs keep working.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List


SCHEMA_VERSION = "2.0"


class ConfigError(ValueError):
    """Raised when the dashboard config is malformed."""


def load_config(path: str) -> Dict[str, Any]:
    """Load and normalise the dashboard config.

    Returns a dict guaranteed to contain the v2 keys:
    ``tickers`` (list of dicts), ``benchmark``, ``data_period``,
    ``data_interval``, ``label``, ``technical_signals``, ``ml_signals``,
    ``portfolios``, ``schema_version``, ``data_version``.
    """
    if not os.path.exists(path):
        raise ConfigError(f"Config not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    return _normalise(raw)


def _normalise(raw: Dict[str, Any]) -> Dict[str, Any]:
    schema_version = str(raw.get("schema_version", "1.0"))

    tickers = _normalise_tickers(raw.get("tickers", []))
    benchmark = raw.get("benchmark", "SPY")
    data_period = raw.get("data_period", "2y")
    data_interval = raw.get("data_interval", "1d")

    label = raw.get("label") or {
        "mode": "binary",
        "threshold": 0.005,
        "version": "binary-legacy",
    }
    label.setdefault("mode", "binary")
    label.setdefault("threshold", 0.005)
    label.setdefault("version", "binary-legacy")

    # technical_signals — also accept the legacy "signals" key
    tech = raw.get("technical_signals") or raw.get("signals") or {}
    tech.setdefault("buy_threshold", 3)
    tech.setdefault("sell_threshold", 3)
    tech.setdefault("rsi_buy", 35)
    tech.setdefault("rsi_sell", 65)

    ml = raw.get("ml_signals") or {}
    ml.setdefault("models", ["logistic_regression", "random_forest", "xgboost", "lightgbm"])
    ml.setdefault("probability_threshold_buy", 0.55)
    ml.setdefault("probability_threshold_sell", 0.45)
    ml.setdefault("models_dir", "stock_ml/models/saved")
    ml.setdefault("reports_dir", "stock_ml/reports")
    ml.setdefault("use_proxy_when_artifacts_missing", True)

    # An *explicit* empty list means "no portfolios" — respect it. Only fall
    # back to the legacy default when the key is missing entirely.
    raw_portfolios = raw.get("portfolios")
    if raw_portfolios is None:
        raw_portfolios = _default_portfolios([t["symbol"] for t in tickers])
    portfolios = [_normalise_portfolio(p) for p in raw_portfolios]

    return {
        "schema_version": schema_version,
        "data_version": raw.get("data_version", "unversioned"),
        "strategy_version": raw.get("strategy_version", "1.0"),
        "tickers": tickers,
        "benchmark": benchmark,
        # ``benchmark_tickers`` is the full list of symbols that should be
        # treated as benchmarks (i.e. excluded from model-driven signal
        # generation). It defaults to ``[benchmark]`` for backwards compat.
        "benchmark_tickers": raw.get("benchmark_tickers", [benchmark]),
        # Stable colour mapping for sectors used by the frontend. Optional.
        "sector_palette": raw.get("sector_palette", {}),
        "data_period": data_period,
        "data_interval": data_interval,
        "label": label,
        "technical_signals": tech,
        "ml_signals": ml,
        "portfolios": portfolios,
    }


def _normalise_tickers(raw: List[Any]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for entry in raw:
        if isinstance(entry, str):
            out.append({"symbol": entry, "name": entry, "sector": "Other"})
        elif isinstance(entry, dict):
            sym = entry.get("symbol") or entry.get("ticker")
            if not sym:
                raise ConfigError(f"Ticker entry missing 'symbol': {entry}")
            out.append({
                "symbol": sym,
                "name": entry.get("name", sym),
                "sector": entry.get("sector", "Other"),
            })
        else:
            raise ConfigError(f"Unsupported ticker entry: {entry!r}")
    if not out:
        raise ConfigError("Config has empty tickers list")
    return out


def _default_portfolios(tickers: List[str]) -> List[Dict[str, Any]]:
    """When the legacy config has no portfolios, build a single default one."""
    return [{
        "id": "default",
        "name": "Default Portfolio",
        "description": "Auto-generated default portfolio from legacy config.",
        "tickers": tickers,
        "weights": "equal",
        "initial_capital": 100000,
        "signal_source": "technical_rule_based",
        "rules": {"buy_score": 3, "sell_score": 3, "rebalance": "on_signal"},
        "benchmark": "SPY",
    }]


def _normalise_portfolio(p: Dict[str, Any]) -> Dict[str, Any]:
    if "id" not in p or "tickers" not in p:
        raise ConfigError(f"Portfolio missing id or tickers: {p}")
    rules = dict(p.get("rules", {}) or {})

    # Normalise the optional trading-mechanics rules so the engine always
    # receives well-typed values (they are all opt-in; defaults preserve the
    # legacy all-in/all-out behaviour).
    if "seed_shares" in rules:
        try:
            rules["seed_shares"] = max(0, int(rules["seed_shares"]))
        except (TypeError, ValueError):
            raise ConfigError(f"Portfolio {p['id']}: seed_shares must be an integer")
    trade_mode = str(rules.get("trade_mode", "all_in"))
    if trade_mode not in ("all_in", "incremental"):
        raise ConfigError(
            f"Portfolio {p['id']}: trade_mode must be 'all_in' or 'incremental', got {trade_mode!r}"
        )
    rules["trade_mode"] = trade_mode
    if "trade_size" in rules:
        try:
            rules["trade_size"] = max(1, int(rules["trade_size"]))
        except (TypeError, ValueError):
            raise ConfigError(f"Portfolio {p['id']}: trade_size must be an integer")

    return {
        "id": p["id"],
        "name": p.get("name", p["id"]),
        "description": p.get("description", ""),
        "note": p.get("note", ""),
        "tickers": list(p["tickers"]),
        "weights": p.get("weights", "equal"),
        "initial_capital": float(p.get("initial_capital", 100000)),
        "signal_source": p.get("signal_source", "technical_rule_based"),
        "rules": rules,
        "benchmark": p.get("benchmark", "SPY"),
    }
