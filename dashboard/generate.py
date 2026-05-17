#!/usr/bin/env python3
"""Dashboard data generator (v2).

End-to-end pipeline:

1. Load dashboard/config.json and build the ticker + model registry.
2. Download OHLCV via yfinance and compute the standard indicator set.
3. Generate signals from every configured model (technical + ML real-or-proxy)
   for every ticker.
4. Run the portfolio backtest for every portfolio defined in the config,
   producing decisions, transactions, equity curves, positions and metrics.
5. Write a rich set of JSON + CSV artifacts to ``docs/data/`` using the new
   layout while keeping the legacy paths around for backward compatibility.

The output is *static* and consumed by the GitHub-Pages frontend — no server
backend is required.
"""
from __future__ import annotations

import logging
import os
import sys
import uuid
from typing import Dict, List

import pandas as pd

# Make ``dashboard.lib`` importable whether we run via ``python dashboard/generate.py``
# (cwd = repo root) or directly (cwd = dashboard/).
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from lib import (  # type: ignore  # noqa: E402
    audit, config_loader, data_fetcher, exporters, indicators,
    metrics as metrics_mod, ml_loader, portfolio as portfolio_mod, registry as registry_mod,
    signals as signals_mod,
)


CONFIG_PATH = os.path.join(SCRIPT_DIR, "config.json")
DOCS_DATA_DIR = os.path.join(REPO_ROOT, "docs", "data")
DOCS_CHARTS_DIR = os.path.join(REPO_ROOT, "docs", "charts", "plots")
REPORTS_PLOTS_DIR = os.path.join(REPO_ROOT, "reports", "plots")


logging.basicConfig(
    level=os.environ.get("DASHBOARD_LOGLEVEL", "INFO"),
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("dashboard")


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

def fetch_all_prices(reg: registry_mod.Registry, period: str, interval: str) -> Dict[str, pd.DataFrame]:
    """Download OHLCV for every ticker in the registry plus the benchmark."""
    out: Dict[str, pd.DataFrame] = {}
    for symbol in reg.all_symbols_with_benchmark:
        df = data_fetcher.download(symbol, period=period, interval=interval)
        if df is None or df.empty:
            logger.warning("Skipping %s (no data).", symbol)
            continue
        out[symbol] = indicators.compute(df)
    return out


def generate_all_signals(reg: registry_mod.Registry, prices: Dict[str, pd.DataFrame],
                          config: Dict) -> Dict[str, Dict[str, pd.DataFrame]]:
    """For each ticker, compute one signal DataFrame per model."""
    tech_rules = config["technical_signals"]
    ml_cfg = config["ml_signals"]
    p_buy = ml_cfg["probability_threshold_buy"]
    p_sell = ml_cfg["probability_threshold_sell"]

    out: Dict[str, Dict[str, pd.DataFrame]] = {}
    for ticker, df in prices.items():
        per_model: Dict[str, pd.DataFrame] = {}

        # 1) Technical rule based
        per_model["technical_rule_based"] = signals_mod.technical_rule_based(df, tech_rules)

        # 2) ML models (real or proxy)
        ml_signals_by_model: Dict[str, pd.DataFrame] = {}
        for model in ml_cfg["models"]:
            probs, signal_source, model_version = ml_loader.model_probabilities(model, ticker, df, ml_cfg)
            sigs = signals_mod.from_probabilities(probs.reindex(df.index), p_buy, p_sell)
            sigs.attrs["signal_source"] = signal_source
            sigs.attrs["model_version"] = model_version
            per_model[model] = sigs
            ml_signals_by_model[model] = sigs

        # 3) Ensemble across ML models
        per_model["ensemble_majority"] = signals_mod.ensemble_majority(
            ml_signals_by_model, min_votes=2,
        )

        # 4) Buy-and-hold for benchmark portfolio
        per_model["buy_and_hold"] = signals_mod.buy_and_hold(df)

        # Attach a stable signal_id to every (ticker, model, date)
        for model_name, sigdf in per_model.items():
            sigdf["signal_id"] = [
                f"S-{ticker}-{model_name}-{idx.strftime('%Y%m%d')}-{uuid.uuid4().hex[:6]}"
                for idx in sigdf.index
            ]
        out[ticker] = per_model
    return out


def build_signal_records(reg: registry_mod.Registry, prices: Dict[str, pd.DataFrame],
                          signals_by_ticker: Dict[str, Dict[str, pd.DataFrame]],
                          executed_signal_ids: set, executed_portfolio_map: Dict[str, List[str]],
                          generated_at: str, data_version: str, label_mode: str,
                          label_version: str) -> List[Dict]:
    """Convert per-model signal DataFrames into the rich JSON records used by
    the signals panel and CSV exports."""
    records: List[Dict] = []
    for ticker, per_model in signals_by_ticker.items():
        df_prices = prices[ticker]
        ticker_info = reg.ticker(ticker)
        for model_name, sigdf in per_model.items():
            signal_source = sigdf.attrs.get("signal_source", model_name)
            model_version = sigdf.attrs.get("model_version", "rule-1.0")
            for idx, row in sigdf.iterrows():
                if idx not in df_prices.index:
                    continue
                price_row = df_prices.loc[idx]
                signal_id = row["signal_id"]
                rec = {
                    "signal_id": signal_id,
                    "generated_at_utc": generated_at,
                    "signal_date": idx.strftime("%Y-%m-%d"),
                    "ticker": ticker,
                    "company_name": ticker_info.name,
                    "sector": ticker_info.sector,
                    "model_name": model_name,
                    "signal_source": signal_source,
                    "model_version": model_version,
                    "signal": row["signal"],
                    "score": _f(row.get("score"), 4),
                    "probability_up": _f(row.get("probability_up"), 4),
                    "probability_down": _f(1 - row["probability_up"], 4) if pd.notna(row.get("probability_up")) else None,
                    "confidence": _f(row.get("confidence"), 4),
                    "threshold_used": _f(row.get("threshold_used"), 4),
                    "triggered_rules": row.get("triggered_rules", ""),
                    "execution_candidate": row["signal"] in ("BUY", "SELL"),
                    "executed_in_portfolio": signal_id in executed_signal_ids,
                    "portfolio_ids": executed_portfolio_map.get(signal_id, []),
                    "current_price": _f(price_row.get("Close"), 4),
                    "previous_close": _f(_previous_close(df_prices, idx), 4),
                    "open": _f(price_row.get("Open"), 4),
                    "high": _f(price_row.get("High"), 4),
                    "low": _f(price_row.get("Low"), 4),
                    "close": _f(price_row.get("Close"), 4),
                    "volume": int(price_row["Volume"]) if pd.notna(price_row.get("Volume")) else None,
                    "rsi": _f(price_row.get("RSI"), 2),
                    "macd": _f(price_row.get("MACD"), 4),
                    "macd_signal": _f(price_row.get("MACD_signal"), 4),
                    "macd_hist": _f(price_row.get("MACD_hist"), 4),
                    "bb_upper": _f(price_row.get("BB_upper"), 4),
                    "bb_lower": _f(price_row.get("BB_lower"), 4),
                    "bb_middle": _f(price_row.get("BB_middle"), 4),
                    "sma20": _f(price_row.get("SMA20"), 4),
                    "sma50": _f(price_row.get("SMA50"), 4),
                    "return_1d_after": _forward_return(df_prices, idx, 1),
                    "return_5d_after": _forward_return(df_prices, idx, 5),
                    "return_20d_after": _forward_return(df_prices, idx, 20),
                    "label_mode": label_mode,
                    "label_version": label_version,
                    "data_version": data_version,
                }
                records.append(rec)
    return records


def run_all_portfolios(config: Dict, prices: Dict[str, pd.DataFrame],
                        signals_by_ticker: Dict[str, Dict[str, pd.DataFrame]]) -> List[portfolio_mod.BacktestResult]:
    """Run the backtest for every portfolio in the config."""
    benchmark_df = prices.get(config["benchmark"])
    label = config["label"]
    results = []
    for pcfg in config["portfolios"]:
        logger.info("Backtest portfolio: %s (%s)", pcfg["id"], pcfg["signal_source"])
        per_ticker_filtered = {t: prices[t] for t in pcfg["tickers"] if t in prices}
        sigs_filtered = {t: signals_by_ticker[t] for t in pcfg["tickers"] if t in signals_by_ticker}
        result = portfolio_mod.run_portfolio_backtest(
            portfolio_cfg=pcfg,
            prices_by_ticker=per_ticker_filtered,
            signals_by_ticker_model=sigs_filtered,
            benchmark_prices=benchmark_df,
            data_version=config.get("data_version", "unknown"),
            strategy_version=config.get("strategy_version", "1.0"),
            label_mode=label.get("mode", "binary"),
            label_version=label.get("version", "binary-2026-05"),
        )
        results.append(result)
    return results


def build_model_summaries(reg: registry_mod.Registry, signal_records: List[Dict],
                           prices: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
    """Aggregate signal effectiveness metrics per model."""
    out: Dict[str, Dict] = {}
    by_model: Dict[str, List[Dict]] = {}
    for rec in signal_records:
        by_model.setdefault(rec["model_name"], []).append(rec)

    for model_name, recs in by_model.items():
        # Compute classification metrics on the *full* per-ticker stream and
        # also on a per-ticker basis so the thesis can show breakdowns.
        all_sig = pd.Series([r["signal"] for r in recs])
        all_y = pd.Series([1 if (r["return_1d_after"] or 0) > 0 else 0 for r in recs])
        overall = metrics_mod.classification_metrics(all_sig, all_y)

        per_ticker: Dict[str, Dict] = {}
        for ticker in reg.symbols + [reg.benchmark]:
            sub = [r for r in recs if r["ticker"] == ticker]
            if not sub:
                continue
            s_sig = pd.Series([r["signal"] for r in sub])
            s_y = pd.Series([1 if (r["return_1d_after"] or 0) > 0 else 0 for r in sub])
            per_ticker[ticker] = metrics_mod.classification_metrics(s_sig, s_y)

        out[model_name] = {
            "model_name": model_name,
            "n_signals": len(recs),
            "overall": overall,
            "per_ticker": per_ticker,
        }
    return out


# ---------------------------------------------------------------------------
# Plumbing
# ---------------------------------------------------------------------------

def _previous_close(df, idx):
    if idx not in df.index:
        return None
    loc = df.index.get_loc(idx)
    if loc == 0:
        return None
    val = df["Close"].iloc[loc - 1]
    return float(val) if pd.notna(val) else None


def _forward_return(df, idx, n):
    if idx not in df.index:
        return None
    loc = df.index.get_loc(idx)
    closes = df["Close"]
    if loc + n >= len(closes):
        return None
    base = closes.iloc[loc]
    fwd = closes.iloc[loc + n]
    if pd.notna(base) and pd.notna(fwd) and base > 0:
        return round(float(fwd / base - 1), 6)
    return None


def _f(x, n=4):
    if x is None:
        return None
    try:
        if isinstance(x, float) and (x != x):
            return None
    except Exception:
        pass
    try:
        return round(float(x), n)
    except (TypeError, ValueError):
        return None


def copy_report_plots() -> List[str]:
    """Copy PNG plots from stock_ml/reports/plots and reports/plots to docs/."""
    import shutil
    sources = [
        os.path.join(REPO_ROOT, "stock_ml", "reports", "plots"),
        REPORTS_PLOTS_DIR,
    ]
    os.makedirs(DOCS_CHARTS_DIR, exist_ok=True)
    copied = []
    for src in sources:
        if not os.path.isdir(src):
            continue
        for fname in sorted(os.listdir(src)):
            if fname.lower().endswith(".png"):
                shutil.copy2(os.path.join(src, fname), os.path.join(DOCS_CHARTS_DIR, fname))
                copied.append(fname)
    return copied


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logger.info("=" * 60)
    logger.info("Dashboard generator v2 — start")
    logger.info("=" * 60)

    config = config_loader.load_config(CONFIG_PATH)
    reg = registry_mod.build_registry(config)
    artifact_status = ml_loader.detect_real_artifacts(config["ml_signals"])
    run_meta = audit.build_run_metadata(REPO_ROOT, config, artifact_status)

    logger.info("Tickers: %s", ", ".join(reg.symbols))
    logger.info("Benchmark: %s", reg.benchmark)
    logger.info("Models: %s", ", ".join(reg.model_names))
    logger.info("Portfolios: %s", ", ".join(p["id"] for p in config["portfolios"]))
    logger.info("Real ML artifacts present: %s", artifact_status)

    os.makedirs(DOCS_DATA_DIR, exist_ok=True)

    # 1) prices + indicators
    prices = fetch_all_prices(reg, config["data_period"], config["data_interval"])
    if not prices:
        logger.error("No prices downloaded — aborting.")
        return

    # 2) per-ticker per-model signals
    signals_by_ticker = generate_all_signals(reg, prices, config)

    # 3) portfolios — produces executed signal_ids
    portfolio_results = run_all_portfolios(config, prices, signals_by_ticker)
    executed_signal_ids: set = set()
    executed_portfolio_map: Dict[str, List[str]] = {}
    for r in portfolio_results:
        for tx in r.transactions:
            sid = tx.get("signal_id")
            if sid:
                executed_signal_ids.add(sid)
                executed_portfolio_map.setdefault(sid, []).append(r.portfolio_id)

    # 4) signal records (rich)
    signal_records = build_signal_records(
        reg, prices, signals_by_ticker,
        executed_signal_ids, executed_portfolio_map,
        generated_at=run_meta["generated_at_utc"],
        data_version=run_meta["data_version"],
        label_mode=config["label"]["mode"],
        label_version=config["label"]["version"],
    )

    # 5) model summaries
    model_summaries = build_model_summaries(reg, signal_records, prices)

    # 6) write everything
    exporters.write_meta(DOCS_DATA_DIR, run_meta)
    exporters.write_last_updated(DOCS_DATA_DIR, run_meta)
    exporters.write_registry(DOCS_DATA_DIR, reg.to_dict())

    for ticker, df in prices.items():
        exporters.write_ticker_ohlcv(DOCS_DATA_DIR, ticker, df)
        per_model_signals = {
            m: [r for r in signal_records if r["ticker"] == ticker and r["model_name"] == m]
            for m in reg.model_names
        }
        exporters.write_ticker_signals(DOCS_DATA_DIR, ticker, per_model_signals)

    exporters.write_signals_bundle(DOCS_DATA_DIR, signal_records)
    exporters.write_model_summaries(DOCS_DATA_DIR, model_summaries)

    portfolio_index = []
    portfolio_metrics_rows: List[Dict] = []
    all_transactions: List[Dict] = []
    for r in portfolio_results:
        exporters.write_portfolio(DOCS_DATA_DIR, r)
        portfolio_index.append({
            "portfolio_id": r.portfolio_id,
            "portfolio_name": r.portfolio_name,
            "description": r.description,
            "signal_source": r.metadata.get("signal_source"),
            "initial_capital": r.initial_capital,
            "final_value": r.final_value,
            "total_return": r.metrics.get("total_return"),
            "sharpe": r.metrics.get("sharpe_ratio"),
            "max_drawdown": r.metrics.get("max_drawdown"),
            "n_decisions": r.metadata.get("n_decisions"),
            "n_transactions": r.metadata.get("n_transactions"),
        })
        portfolio_metrics_rows.append({
            "portfolio_id": r.portfolio_id,
            **r.metrics,
            **{"benchmark_" + k: v for k, v in r.benchmark_metrics.items()},
        })
        all_transactions.extend(r.transactions)
    exporters.write_portfolio_index(DOCS_DATA_DIR, portfolio_index)

    model_metrics_rows = []
    for m, s in model_summaries.items():
        row = {"model_name": m, "n_signals": s["n_signals"]}
        for k, v in (s.get("overall") or {}).items():
            row[k] = v
        model_metrics_rows.append(row)

    exporters.write_global_exports(
        DOCS_DATA_DIR,
        all_signals=signal_records,
        all_transactions=all_transactions,
        portfolio_metrics=portfolio_metrics_rows,
        model_metrics=model_metrics_rows,
    )

    # Legacy tickers.json (a plain list of symbols for the old frontend)
    audit.write_json(os.path.join(DOCS_DATA_DIR, "tickers.json"), reg.all_symbols_with_benchmark)

    # Optional plot copy
    copied = copy_report_plots()
    if copied:
        logger.info("Copied %d plot(s).", len(copied))

    logger.info("Dashboard data generation complete.")


if __name__ == "__main__":
    main()
