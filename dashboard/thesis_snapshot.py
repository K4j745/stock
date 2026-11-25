#!/usr/bin/env python3
"""Freeze the dashboard portfolios for the thesis.

Downloads prices as ``generate.py`` does, cuts every series at ``--end`` before
computing anything (so only complete sessions are used, as if the generator ran
after that day's close), computes the signals, runs every portfolio and writes
its metrics for the whole window and from ``--split``.

``--compare-same-day`` also runs the portfolios with same-day execution
(``signal_lag=0``, the behaviour before the lookahead fix) and writes the
difference, which measures the size of that bias on the same data.

Nothing is written to ``docs/data``.

Usage:
    python dashboard/thesis_snapshot.py --end 2026-11-19 --out thesis_results/dashboard_migawka_2026-11-19_v2 --compare-same-day
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from typing import Dict, List

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

import generate  # noqa: E402
from lib import config_loader, data_fetcher, indicators, metrics as metrics_mod  # noqa: E402
from lib import portfolio as portfolio_mod  # noqa: E402
from lib import registry as registry_mod  # noqa: E402

logger = logging.getLogger("dashboard.snapshot")

METRIC_COLUMNS = {
    "zwrot": ("total_return", 4),
    "cagr": ("cagr", 4),
    "zmiennosc": ("volatility_annualised", 4),
    "sharpe": ("sharpe_ratio", 3),
    "max_obsuniecie": ("max_drawdown", 4),
    "calmar": ("calmar_ratio", 3),
}


def load_prices(reg: registry_mod.Registry, config: Dict, end: pd.Timestamp) -> Dict[str, pd.DataFrame]:
    prices = {}
    for symbol in reg.all_symbols_with_benchmark:
        df = data_fetcher.download(symbol, period=config["data_period"], interval=config["data_interval"])
        if df is None or df.empty:
            raise RuntimeError(f"No prices for {symbol}")
        prices[symbol] = indicators.compute(df.loc[:end])
    return prices


def run_portfolios(config: Dict, prices: Dict, signals: Dict, signal_lag: int) -> List:
    benchmark = prices.get(config["benchmark"])
    results = []
    for cfg in config["portfolios"]:
        results.append(portfolio_mod.run_portfolio_backtest(
            portfolio_cfg=cfg,
            prices_by_ticker={t: prices[t] for t in cfg["tickers"] if t in prices},
            signals_by_ticker_model={t: signals[t] for t in cfg["tickers"] if t in signals},
            benchmark_prices=benchmark,
            signal_lag=signal_lag,
        ))
    return results


def equity_series(result) -> pd.Series:
    curve = result.equity_curve
    return pd.Series([p["portfolio_value"] for p in curve], index=pd.to_datetime([p["date"] for p in curve]))


def metric_rows(portfolio_id: str, name: str, source: str, equity: pd.Series, split: pd.Timestamp) -> List[Dict]:
    """Metrics for the whole window and from ``split`` (starting from that day's portfolio value)."""
    rows = []
    for period, series in (("cale_okno", equity), (f"od_{split.date()}", equity.loc[split:])):
        m = metrics_mod.portfolio_metrics(series)
        rows.append({
            "portfolio_id": portfolio_id, "portfel": name, "zrodlo_sygnalu": source, "okres": period,
            "od": str(series.index[0].date()), "do": str(series.index[-1].date()), "sesje": len(series),
            **{col: round(float(m[key]), digits) for col, (key, digits) in METRIC_COLUMNS.items()},
        })
    return rows


def snapshot_table(config: Dict, results: List, split: pd.Timestamp) -> pd.DataFrame:
    sources = {p["id"]: p["signal_source"] for p in config["portfolios"]}
    rows = []
    for r in results:
        rows += metric_rows(r.portfolio_id, r.portfolio_name, sources[r.portfolio_id], equity_series(r), split)
    return pd.DataFrame(rows)


def comparison(after: pd.DataFrame, before: pd.DataFrame) -> pd.DataFrame:
    keys = ["portfolio_id", "portfel", "okres"]
    merged = before.merge(after, on=keys, suffixes=("_ten_sam_dzien", "_nastepna_sesja"))
    out = merged[keys].copy()
    for col in METRIC_COLUMNS:
        out[f"{col}_ten_sam_dzien"] = merged[f"{col}_ten_sam_dzien"]
        out[f"{col}_nastepna_sesja"] = merged[f"{col}_nastepna_sesja"]
        out[f"{col}_roznica"] = (merged[f"{col}_nastepna_sesja"] - merged[f"{col}_ten_sam_dzien"]).round(4)
    return out


def signal_sources(signals: Dict) -> Dict[str, List[str]]:
    found: Dict[str, set] = {}
    for per_model in signals.values():
        for model, frame in per_model.items():
            found.setdefault(model, set()).add(frame.attrs.get("signal_source", model))
    return {model: sorted(values) for model, values in sorted(found.items())}


def git_state() -> Dict:
    def run(*args):
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()
    return {"commit": run("rev-parse", "--short", "HEAD"), "uncommitted_changes": bool(run("status", "--porcelain"))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--end", required=True, help="last complete session, YYYY-MM-DD")
    parser.add_argument("--split", default="2026-02-24", help="start of the period new to the final models")
    parser.add_argument("--out", required=True, help="output folder")
    parser.add_argument("--compare-same-day", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

    end, split = pd.Timestamp(args.end), pd.Timestamp(args.split)
    config = config_loader.load_config(generate.CONFIG_PATH)
    reg = registry_mod.build_registry(config)
    prices = load_prices(reg, config, end)
    signals = generate.generate_all_signals(reg, prices, config)

    os.makedirs(args.out, exist_ok=True)
    after = snapshot_table(config, run_portfolios(config, prices, signals, signal_lag=1), split)
    after.to_csv(os.path.join(args.out, "portfele_metryki.csv"), index=False)
    if args.compare_same_day:
        before = snapshot_table(config, run_portfolios(config, prices, signals, signal_lag=0), split)
        before.to_csv(os.path.join(args.out, "portfele_metryki_ten_sam_dzien.csv"), index=False)
        comparison(after, before).to_csv(os.path.join(args.out, "porownanie_przed_po_poprawce.csv"), index=False)

    with open(os.path.join(args.out, "zrodlo.json"), "w", encoding="utf-8") as fh:
        json.dump({
            "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end": args.end, "split_date": args.split, "data_period": config["data_period"],
            "execution": "signal of day d traded at the open of the next session (signal_lag=1); "
                         "buy_and_hold traded on the day itself",
            **git_state(),
            "signal_sources": signal_sources(signals),
        }, fh, indent=1, ensure_ascii=False)
    logger.info("Snapshot written to %s", args.out)


if __name__ == "__main__":
    main()
