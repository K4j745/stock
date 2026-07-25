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


# Fallback used only when yfinance refuses to return a value for USDPLN=X.
# Documented + made visible to the UI via the ``source`` field of fx.json.
_FX_FALLBACK_USD_PLN = 4.0


def fetch_fx_usd_pln(period: str = "1mo", interval: str = "1d") -> Dict:
    """Fetch the latest USD→PLN close from yfinance.

    Returns a dict with the rate, the as-of date, and the source. Falls back
    to the documented constant when yfinance is unreachable. The dashboard
    frontend reads ``docs/data/fx.json`` and uses ``rate`` directly — so
    when the rate is "real" (source = "yfinance") the simulator's USD→PLN
    conversion is accurate; otherwise it transparently uses the constant.
    """
    df = data_fetcher.download("USDPLN=X", period=period, interval=interval)
    if df is None or df.empty or "Close" not in df.columns:
        return {
            "rate": _FX_FALLBACK_USD_PLN,
            "as_of": None,
            "source": "fallback_constant",
            "note": f"USDPLN=X unavailable — using documented fallback {_FX_FALLBACK_USD_PLN} PLN/USD.",
        }
    closes = df["Close"].dropna()
    if closes.empty:
        return {
            "rate": _FX_FALLBACK_USD_PLN,
            "as_of": None,
            "source": "fallback_constant",
            "note": "USDPLN=X returned an empty close series.",
        }
    last_idx = closes.index[-1]
    return {
        "rate": round(float(closes.iloc[-1]), 4),
        "as_of": last_idx.strftime("%Y-%m-%d") if hasattr(last_idx, "strftime") else str(last_idx),
        "source": "yfinance",
        "symbol": "USDPLN=X",
    }


def generate_all_signals(reg: registry_mod.Registry, prices: Dict[str, pd.DataFrame],
                          config: Dict) -> Dict[str, Dict[str, pd.DataFrame]]:
    """For each ticker, compute the per-model signal DataFrames.

    Rules:
      * **non-benchmark** tickers get the *full* model lineup (technical
        rules + every ML model + ensemble + buy_and_hold).
      * **benchmark** tickers (e.g. SPY) only get the model-free streams:
        ``technical_rule_based`` (purely rule-based, no training) and
        ``buy_and_hold`` (trivial baseline). ML models are NOT trained on
        the benchmark by design — see docs/README.md.
    """
    tech_rules = config["technical_signals"]
    ml_cfg = config["ml_signals"]
    p_buy = ml_cfg["probability_threshold_buy"]
    p_sell = ml_cfg["probability_threshold_sell"]
    benchmark_set = set(config.get("benchmark_tickers", [config.get("benchmark", "SPY")]))

    out: Dict[str, Dict[str, pd.DataFrame]] = {}
    for ticker, df in prices.items():
        is_benchmark = ticker in benchmark_set
        per_model: Dict[str, pd.DataFrame] = {}

        # 1) Technical rule based — runs for every ticker, including benchmarks.
        per_model["technical_rule_based"] = signals_mod.technical_rule_based(df, tech_rules)

        # 2) ML models — skipped for benchmark tickers (SPY is not a training
        #    asset). All other tickers get the full ML lineup.
        if not is_benchmark:
            ml_signals_by_model: Dict[str, pd.DataFrame] = {}
            for model in ml_cfg["models"]:
                probs, signal_source, model_version = ml_loader.model_probabilities(model, ticker, df, ml_cfg)
                sigs = signals_mod.from_probabilities(probs.reindex(df.index), p_buy, p_sell)
                sigs.attrs["signal_source"] = signal_source
                sigs.attrs["model_version"] = model_version
                per_model[model] = sigs
                ml_signals_by_model[model] = sigs

            # 3) Ensemble — only meaningful when we have ML models, so also
            #    skipped for benchmarks.
            per_model["ensemble_majority"] = signals_mod.ensemble_majority(
                ml_signals_by_model, min_votes=2,
            )

        # 4) Buy-and-hold — runs for every ticker (baseline strategy).
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
                signal_reason = _classify_signal_reason(model_name, row)
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
                    # signal_reason explains *why* this signal fired — useful
                    # in the UI for distinguishing "HOLD because the model
                    # says so" from "HOLD because confidence is low".
                    "signal_reason": signal_reason,
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

_ML_MODELS = {"logistic_regression", "random_forest", "xgboost", "lightgbm", "candle"}


def _classify_signal_reason(model_name: str, row) -> str:
    """Map (model, row) → a short, UI-friendly reason code.

    The reason answers "why this label?" — distinct from the *source*
    (which is the model name) and the *triggered rules* (which is the raw
    trace).

    Classification is **binary** (BUY/SELL) for every predictive stream, so
    there is no HOLD dead-band any more. The only stream that still emits
    HOLD is ``buy_and_hold`` (the benchmark carry between entry and exit).

    Codes:
      * ``rule_triggered``         — technical_rule_based fired BUY/SELL
      * ``model_prediction``       — ML model (incl. candle) picked BUY/SELL
      * ``majority_vote``          — ensemble_majority decided BUY/SELL
      * ``buy_and_hold_entry`` / ``buy_and_hold_carry`` — benchmark stream
    """
    sig = str(row.get("signal", "HOLD"))
    if model_name == "technical_rule_based":
        return "rule_triggered" if sig in ("BUY", "SELL") else "rule_threshold_not_met"
    if model_name == "ensemble_majority":
        return "majority_vote" if sig in ("BUY", "SELL") else "model_disagreement"
    if model_name == "buy_and_hold":
        return "buy_and_hold_entry" if sig == "BUY" else "buy_and_hold_carry"
    if model_name in _ML_MODELS:
        return "model_prediction" if sig in ("BUY", "SELL") else "low_confidence"
    return "unknown"


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
    """Copy PNG plots from stock_ml/reports/plots and reports/plots to docs/.

    Always ensures the destination directory exists (so the frontend can
    fetch ``plots_index.json`` even on a clean repo with no models trained
    yet — it will simply be an empty list).
    """
    import shutil
    sources = [
        os.path.join(REPO_ROOT, "stock_ml", "reports", "plots"),
        REPORTS_PLOTS_DIR,
    ]
    os.makedirs(DOCS_CHARTS_DIR, exist_ok=True)
    copied = []
    for src in sources:
        if not os.path.isdir(src):
            logger.info("Plot source not found: %s (skipping)", src)
            continue
        for fname in sorted(os.listdir(src)):
            if fname.lower().endswith(".png"):
                dst = os.path.join(DOCS_CHARTS_DIR, fname)
                logger.info("Copying chart %s -> %s", fname, dst)
                shutil.copy2(os.path.join(src, fname), dst)
                copied.append(fname)
    # Also pick up any *.png already in the destination (e.g. committed
    # by hand) so the index is always authoritative.
    if os.path.isdir(DOCS_CHARTS_DIR):
        for fname in sorted(os.listdir(DOCS_CHARTS_DIR)):
            if fname.lower().endswith(".png") and fname not in copied:
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

    # FX rate — the portfolio simulator uses it to convert USD prices to PLN.
    fx_payload = fetch_fx_usd_pln()
    fx_payload["generated_at_utc"] = run_meta["generated_at_utc"]
    audit.write_json(os.path.join(DOCS_DATA_DIR, "fx.json"), fx_payload)
    logger.info("FX USD→PLN: %.4f (source=%s)", fx_payload["rate"], fx_payload["source"])

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

    # Optional plot copy + a stable plots_index.json so the frontend doesn't
    # have to probe a hardcoded list of filenames.
    copied = copy_report_plots()
    plots_index = {
        "generated_at_utc": run_meta["generated_at_utc"],
        "plots": copied,
        "count": len(copied),
        "source_directories_checked": [
            os.path.relpath(os.path.join(REPO_ROOT, "stock_ml", "reports", "plots"), REPO_ROOT),
            os.path.relpath(REPORTS_PLOTS_DIR, REPO_ROOT),
        ],
    }
    audit.write_json(os.path.join(DOCS_CHARTS_DIR, "..", "plots_index.json"), plots_index)
    audit.write_json(os.path.join(DOCS_DATA_DIR, "plots_index.json"), plots_index)
    if copied:
        logger.info("Indexed %d plot(s).", len(copied))
    else:
        logger.info("No ML plots available — frontend will show empty state.")

    logger.info("Dashboard data generation complete.")


if __name__ == "__main__":
    main()
