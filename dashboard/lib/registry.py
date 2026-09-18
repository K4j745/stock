"""Registry — single source of truth for the universe of tickers and models.

The registry is built from the loaded config and made available to every other
module. This means the dashboard frontend, the signals panel, the portfolio
configurations, and the benchmark all reference the *same* ticker list.

The registry exposes simple helpers so callers never have to re-parse the
config dict manually.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class TickerInfo:
    symbol: str
    name: str
    sector: str


@dataclass(frozen=True)
class ModelInfo:
    name: str
    family: str  # "technical" | "ml"
    description: str


@dataclass
class Registry:
    tickers: List[TickerInfo] = field(default_factory=list)
    benchmark: str = "SPY"
    models: List[ModelInfo] = field(default_factory=list)

    @property
    def symbols(self) -> List[str]:
        return [t.symbol for t in self.tickers]

    @property
    def all_symbols_with_benchmark(self) -> List[str]:
        if self.benchmark and self.benchmark not in self.symbols:
            return self.symbols + [self.benchmark]
        return list(self.symbols)

    @property
    def model_names(self) -> List[str]:
        return [m.name for m in self.models]

    def ticker(self, symbol: str) -> TickerInfo:
        for t in self.tickers:
            if t.symbol == symbol:
                return t
        return TickerInfo(symbol=symbol, name=symbol, sector="Other")

    def to_dict(self) -> Dict:
        return {
            "tickers": [t.__dict__ for t in self.tickers],
            "benchmark": self.benchmark,
            "models": [m.__dict__ for m in self.models],
        }


_MODEL_DESCRIPTIONS = {
    "technical_rule_based": ("technical", "Score-based rule signal using RSI/MACD/BB/SMA."),
    "logistic_regression": ("ml", "Linear classifier trained on technical features."),
    "random_forest":       ("ml", "Bagged decision tree ensemble."),
    "xgboost":             ("ml", "Gradient boosted trees (XGBoost)."),
    "lightgbm":            ("ml", "Gradient boosted trees (LightGBM)."),
    "ensemble_majority":   ("ml", "Majority vote across ML models."),
    "buy_and_hold":        ("technical", "Buy on first day, hold to end (benchmark)."),
}


def build_registry(config: Dict) -> Registry:
    tickers = [TickerInfo(**t) for t in config["tickers"]]

    model_names = ["technical_rule_based"] + list(config["ml_signals"]["models"]) + [
        "ensemble_majority",
    ]
    # de-duplicate while preserving order
    seen, ordered = set(), []
    for m in model_names:
        if m not in seen:
            ordered.append(m)
            seen.add(m)

    models = []
    for name in ordered:
        family, desc = _MODEL_DESCRIPTIONS.get(name, ("other", name))
        models.append(ModelInfo(name=name, family=family, description=desc))

    return Registry(
        tickers=tickers,
        benchmark=config.get("benchmark", "SPY"),
        models=models,
    )
