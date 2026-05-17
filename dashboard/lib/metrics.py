"""Classification + portfolio performance metrics.

Kept in one file because the metric formulas are short and there is significant
overlap between "signal effectiveness" and "trade effectiveness".

All functions return JSON-serialisable plain dicts.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Classification metrics (signal vs realised next-day direction)
# ---------------------------------------------------------------------------

def classification_metrics(signals: pd.Series, realised_up: pd.Series) -> Dict[str, Optional[float]]:
    """Compute accuracy / precision / recall / F1 / MCC / balanced accuracy / hit rate.

    Treats "BUY" as positive prediction, "SELL" or "HOLD" as negative.
    ``realised_up`` is a 0/1 series for whether the next-day return was positive.
    """
    df = pd.DataFrame({"sig": signals, "y": realised_up}).dropna()
    if df.empty:
        return _empty_classification()

    y_pred = (df["sig"] == "BUY").astype(int)
    y_true = df["y"].astype(int)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    n = tp + tn + fp + fn
    if n == 0:
        return _empty_classification()

    accuracy = (tp + tn) / n
    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    recall = tp / (tp + fn) if (tp + fn) > 0 else None
    f1 = 2 * precision * recall / (precision + recall) if precision and recall else None

    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    balanced_accuracy = (sens + spec) / 2

    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denom if denom > 0 else None

    hit_rate = (df.loc[df["sig"] == "BUY", "y"].mean()
                if (df["sig"] == "BUY").any() else None)

    return {
        "n_samples": n,
        "n_buy": int((df["sig"] == "BUY").sum()),
        "n_sell": int((df["sig"] == "SELL").sum()),
        "n_hold": int((df["sig"] == "HOLD").sum()),
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "accuracy": _r(accuracy),
        "balanced_accuracy": _r(balanced_accuracy),
        "precision": _r(precision),
        "recall": _r(recall),
        "f1": _r(f1),
        "mcc": _r(mcc),
        "hit_rate_on_buy": _r(hit_rate),
    }


def _empty_classification() -> Dict[str, Optional[float]]:
    return {
        "n_samples": 0, "n_buy": 0, "n_sell": 0, "n_hold": 0,
        "true_positive": 0, "true_negative": 0, "false_positive": 0, "false_negative": 0,
        "accuracy": None, "balanced_accuracy": None,
        "precision": None, "recall": None, "f1": None, "mcc": None, "hit_rate_on_buy": None,
    }


# ---------------------------------------------------------------------------
# Portfolio / trade metrics
# ---------------------------------------------------------------------------

def portfolio_metrics(equity_curve: pd.Series, daily_returns: Optional[pd.Series] = None,
                       annualisation: int = 252) -> Dict[str, Optional[float]]:
    """Compute total return / CAGR / vol / Sharpe / Calmar / max drawdown."""
    if equity_curve is None or len(equity_curve) < 2:
        return _empty_portfolio()

    if daily_returns is None:
        daily_returns = equity_curve.pct_change().dropna()
    daily_returns = daily_returns.dropna()
    if daily_returns.empty:
        return _empty_portfolio()

    total_return = float(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1)
    n_years = len(daily_returns) / annualisation
    cagr = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0.0

    vol = float(daily_returns.std() * np.sqrt(annualisation))
    sharpe = (cagr / vol) if vol > 0 else 0.0

    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max
    max_dd = float(drawdown.min()) if not drawdown.empty else 0.0
    calmar = (cagr / abs(max_dd)) if max_dd != 0 else 0.0

    return {
        "total_return": _r(total_return),
        "cagr": _r(cagr),
        "volatility_annualised": _r(vol),
        "sharpe_ratio": _r(sharpe),
        "max_drawdown": _r(max_dd),
        "calmar_ratio": _r(calmar),
        "n_observations": len(daily_returns),
    }


def trade_metrics(trade_pnls: List[float]) -> Dict[str, Optional[float]]:
    """Win rate / profit factor / average trade return / expectancy."""
    if not trade_pnls:
        return {
            "n_trades": 0, "n_wins": 0, "n_losses": 0,
            "win_rate": None, "avg_trade_pnl": None,
            "avg_win": None, "avg_loss": None,
            "profit_factor": None, "expectancy": None,
        }
    arr = np.array(trade_pnls, dtype=float)
    wins = arr[arr > 0]
    losses = arr[arr < 0]
    win_rate = len(wins) / len(arr)
    avg = float(arr.mean())
    avg_w = float(wins.mean()) if len(wins) else 0.0
    avg_l = float(losses.mean()) if len(losses) else 0.0
    sum_w = float(wins.sum()) if len(wins) else 0.0
    sum_l = float(-losses.sum()) if len(losses) else 0.0
    profit_factor = sum_w / sum_l if sum_l > 0 else None
    expectancy = win_rate * avg_w + (1 - win_rate) * avg_l
    return {
        "n_trades": len(arr),
        "n_wins": int(len(wins)),
        "n_losses": int(len(losses)),
        "win_rate": _r(win_rate),
        "avg_trade_pnl": _r(avg),
        "avg_win": _r(avg_w),
        "avg_loss": _r(avg_l),
        "profit_factor": _r(profit_factor),
        "expectancy": _r(expectancy),
    }


def _empty_portfolio() -> Dict[str, Optional[float]]:
    return {
        "total_return": None, "cagr": None, "volatility_annualised": None,
        "sharpe_ratio": None, "max_drawdown": None, "calmar_ratio": None,
        "n_observations": 0,
    }


def _r(x, n=6):
    if x is None:
        return None
    try:
        if math.isnan(x) or math.isinf(x):
            return None
    except TypeError:
        return x
    return round(float(x), n)
