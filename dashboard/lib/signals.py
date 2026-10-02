"""Signal generators — one function per "source".

Each generator takes a DataFrame with OHLCV + indicators attached (see
``indicators.compute``) and returns a DataFrame with the columns:

    signal        — "BUY" / "SELL"  (binary; buy_and_hold benchmark also emits "HOLD" carry)
    score         — integer score for technical, count of triggered rules
    probability_up — probability of an up move tomorrow (0..1), where applicable
    confidence    — abs(probability - 0.5) * 2, or score / max_score for technical
    threshold_used — buy threshold (probability or score) actually used
    triggered_rules — short comma-separated explanation of *why* the signal fired

All generators are *deterministic and side-effect free* so the same input
yields the same signals — important for reproducing results in a thesis.

Probability proxies
-------------------
When the real stock_ml artifacts are not available on disk, ``ml_loader`` falls
back to a *probability proxy*: a transparent deterministic mapping from a few
indicators to a probability. The proxies are explicitly labelled so a reader
of the dashboard always knows when ML outputs are real vs proxy.
"""
from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger("dashboard.signals")


# ---------------------------------------------------------------------------
# Technical rule based
# ---------------------------------------------------------------------------

TECH_RULE_NAMES = [
    "rsi_oversold",
    "rsi_overbought",
    "macd_cross_up",
    "macd_cross_down",
    "bb_break_lower",
    "bb_break_upper",
    "above_sma20_50",
    "below_sma20_50",
]


def technical_rule_based(df: pd.DataFrame, rules: Dict) -> pd.DataFrame:
    """Binary BUY/SELL from RSI / MACD / Bollinger / SMA.

    Each side has four rules (max score = 4). The signal is decided by the
    *net score* (``buy_score - sell_score``):

        BUY  if buy_score > sell_score
        SELL if sell_score > buy_score
        tie  → trend tie-break (Close > SMA50 ⇒ BUY else SELL)

    This is a **binary classifier** (BUY/SELL only) — there is no HOLD class.
    The ``buy_threshold`` / ``sell_threshold`` config values are kept only for
    backward-compatible ``threshold_used`` reporting; they no longer gate a
    HOLD dead-zone. See docs/README.md "How signals are produced".
    """
    rsi_buy = rules.get("rsi_buy", 35)
    rsi_sell = rules.get("rsi_sell", 65)
    buy_thr = rules.get("buy_threshold", 2)
    sell_thr = rules.get("sell_threshold", 2)

    out = pd.DataFrame(index=df.index)
    buy_score = pd.Series(0, index=df.index, dtype=int)
    sell_score = pd.Series(0, index=df.index, dtype=int)
    triggered_buy = pd.Series("", index=df.index, dtype=object)
    triggered_sell = pd.Series("", index=df.index, dtype=object)

    # RSI
    rsi_lo = df["RSI"] < rsi_buy
    rsi_hi = df["RSI"] > rsi_sell
    buy_score = buy_score + rsi_lo.fillna(False).astype(int)
    sell_score = sell_score + rsi_hi.fillna(False).astype(int)
    triggered_buy = triggered_buy.where(~rsi_lo.fillna(False), triggered_buy + "rsi_oversold;")
    triggered_sell = triggered_sell.where(~rsi_hi.fillna(False), triggered_sell + "rsi_overbought;")

    # MACD cross
    macd_up = (df["MACD"].shift(1) <= df["MACD_signal"].shift(1)) & (df["MACD"] > df["MACD_signal"])
    macd_dn = (df["MACD"].shift(1) >= df["MACD_signal"].shift(1)) & (df["MACD"] < df["MACD_signal"])
    buy_score = buy_score + macd_up.fillna(False).astype(int)
    sell_score = sell_score + macd_dn.fillna(False).astype(int)
    triggered_buy = triggered_buy.where(~macd_up.fillna(False), triggered_buy + "macd_cross_up;")
    triggered_sell = triggered_sell.where(~macd_dn.fillna(False), triggered_sell + "macd_cross_down;")

    # Bollinger break
    bb_lo = df["Close"] < df["BB_lower"]
    bb_hi = df["Close"] > df["BB_upper"]
    buy_score = buy_score + bb_lo.fillna(False).astype(int)
    sell_score = sell_score + bb_hi.fillna(False).astype(int)
    triggered_buy = triggered_buy.where(~bb_lo.fillna(False), triggered_buy + "bb_break_lower;")
    triggered_sell = triggered_sell.where(~bb_hi.fillna(False), triggered_sell + "bb_break_upper;")

    # SMA trend
    sma_up = (df["Close"] > df["SMA20"]) & (df["Close"] > df["SMA50"])
    sma_dn = (df["Close"] < df["SMA20"]) & (df["Close"] < df["SMA50"])
    buy_score = buy_score + sma_up.fillna(False).astype(int)
    sell_score = sell_score + sma_dn.fillna(False).astype(int)
    triggered_buy = triggered_buy.where(~sma_up.fillna(False), triggered_buy + "above_sma20_50;")
    triggered_sell = triggered_sell.where(~sma_dn.fillna(False), triggered_sell + "below_sma20_50;")

    # Binary decision on the net score, with a trend tie-break so every day
    # gets a concrete BUY or SELL (no HOLD class).
    net = buy_score - sell_score
    trend_up = (df["Close"] > df["SMA50"]).fillna(True)
    signal = np.where(
        net > 0, "BUY",
        np.where(net < 0, "SELL",
                 np.where(trend_up, "BUY", "SELL")),
    )
    score = np.where(signal == "SELL", sell_score, buy_score)

    # Confidence = score / max_possible (4 rules per side)
    confidence = np.clip(score / 4.0, 0, 1)

    # Probability of "up" — derived heuristically: BUY => high, SELL => low
    prob_up = np.where(
        signal == "BUY", 0.5 + confidence * 0.4,
        0.5 - confidence * 0.4,
    )

    triggered = np.where(signal == "BUY", triggered_buy, triggered_sell)
    triggered = np.where(triggered == "", np.where(signal == "BUY", "trend_tiebreak_up", "trend_tiebreak_down"), triggered)

    out["signal"] = signal
    out["score"] = score
    out["probability_up"] = np.round(prob_up.astype(float), 4)
    out["confidence"] = np.round(confidence.astype(float), 4)
    out["threshold_used"] = buy_thr
    out["triggered_rules"] = triggered
    return out


# ---------------------------------------------------------------------------
# ML probability → BUY/SELL/HOLD
# ---------------------------------------------------------------------------

def from_probabilities(prob_up: pd.Series, buy_threshold: float, sell_threshold: float) -> pd.DataFrame:
    """Convert a probability-of-up series into **binary** BUY/SELL signals.

    Used by every ML model. Binary task (``label_mode = binary``): the model
    predicts P(next-day return > LABEL_THRESHOLD). The mapping is a single
    decision threshold (``buy_threshold``, default 0.5):

        BUY  if probability_up >= buy_threshold
        SELL otherwise

    There is no HOLD dead-band any more — the classifier is strictly binary.
    ``sell_threshold`` is accepted for signature compatibility but ignored.
    See README "How signals are produced".
    """
    sig = np.where(prob_up >= buy_threshold, "BUY", "SELL")
    confidence = np.abs(prob_up - 0.5) * 2.0
    return pd.DataFrame({
        "signal": sig,
        "score": np.round(prob_up.fillna(0.5).astype(float) * 100, 2),
        "probability_up": np.round(prob_up.astype(float), 4),
        "confidence": np.round(confidence.astype(float), 4),
        "threshold_used": buy_threshold,
        "triggered_rules": np.where(sig == "BUY", "prob_above_buy_threshold",
                                    "prob_below_buy_threshold"),
    }, index=prob_up.index)


# ---------------------------------------------------------------------------
# Probability proxies — used when stock_ml artifacts are missing.
#
# Every proxy uses ONLY the technical indicators that the model would have seen
# (no future leak). The mappings are deliberately simple and *deterministic*
# so results are reproducible.
# ---------------------------------------------------------------------------

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def proxy_logistic(df: pd.DataFrame) -> pd.Series:
    """RSI-distance + MACD-hist proxy. Smooth, calibrated around 0.5."""
    rsi = df["RSI"].fillna(50)
    macd_hist = df["MACD_hist"].fillna(0)
    # rescale features
    z_rsi = -(rsi - 50) / 20.0          # under-bought (low rsi) → positive
    z_macd = macd_hist / (macd_hist.abs().rolling(60).mean().fillna(1) + 1e-6)
    z = 0.6 * z_rsi + 0.4 * z_macd
    return _sigmoid(z).clip(0.02, 0.98)


def proxy_random_forest(df: pd.DataFrame) -> pd.Series:
    """Step-wise proxy: simulates the chunky decision-tree output (between 0.3 and 0.7)."""
    rsi = df["RSI"].fillna(50)
    close = df["Close"]
    sma20 = df["SMA20"]
    above = (close > sma20).fillna(False).astype(int)
    bb_lo_break = (close < df["BB_lower"]).fillna(False).astype(int)
    bb_hi_break = (close > df["BB_upper"]).fillna(False).astype(int)
    raw = 0.45 + 0.10 * above + 0.08 * bb_lo_break - 0.08 * bb_hi_break + (35 - rsi).clip(0, None) * 0.004 - (rsi - 65).clip(0, None) * 0.004
    return raw.clip(0.05, 0.95)


def proxy_xgboost(df: pd.DataFrame) -> pd.Series:
    """Slightly more aggressive proxy weighted toward momentum (MACD + return_5d)."""
    macd_hist = df["MACD_hist"].fillna(0)
    ret5 = df["return_5d"].fillna(0)
    rsi = df["RSI"].fillna(50)
    z = 1.2 * macd_hist / (macd_hist.abs().rolling(60).mean().fillna(1) + 1e-6) + 5 * ret5 - (rsi - 50) / 30.0
    return _sigmoid(z).clip(0.02, 0.98)


def proxy_lightgbm(df: pd.DataFrame) -> pd.Series:
    """Mean-reversion biased proxy."""
    rsi = df["RSI"].fillna(50)
    bb_mid = df["BB_middle"]
    close = df["Close"]
    dev = (close - bb_mid) / (bb_mid + 1e-9)
    z = -1.5 * dev - (rsi - 50) / 25.0
    return _sigmoid(z).clip(0.02, 0.98)


def proxy_candle(df: pd.DataFrame) -> pd.Series:
    """Candle-shape proxy — bullish when the close sits near the session high
    with a positive (green) body.

    This is the deterministic stand-in for the trained candlestick model
    (``stock_ml.models.candle_model``). It reads *only* the raw OHLC of each
    day, mirroring the intuition the real model learns from candle geometry:

        close_position — where Close falls inside the day's [Low, High] range
                         (0 = closed on the low, 1 = closed on the high)
        body_fraction  — signed body size (Close - Open) / range
                         (>0 green candle, <0 red candle)

    Both are combined through a sigmoid so the output is a smooth P(up) in
    (0, 1), reproducible and free of any future leak.
    """
    hi = df["High"]
    lo = df["Low"]
    op = df["Open"]
    cl = df["Close"]
    rng = (hi - lo).replace(0, np.nan)
    close_pos = ((cl - lo) / rng).fillna(0.5)
    body_frac = ((cl - op) / rng).fillna(0.0)
    z = 2.0 * (close_pos - 0.5) + 1.5 * body_frac
    return _sigmoid(z).clip(0.02, 0.98)


PROXY_FUNCTIONS = {
    "logistic_regression": proxy_logistic,
    "random_forest": proxy_random_forest,
    "xgboost": proxy_xgboost,
    "lightgbm": proxy_lightgbm,
    "candle": proxy_candle,
}


# ---------------------------------------------------------------------------
# Buy & Hold (used for benchmark portfolio)
# ---------------------------------------------------------------------------

def buy_and_hold(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    if n == 0:
        return pd.DataFrame(columns=["signal", "score", "probability_up", "confidence", "threshold_used", "triggered_rules"])
    sig = np.array(["HOLD"] * n, dtype=object)
    sig[0] = "BUY"  # enter on day 1
    return pd.DataFrame({
        "signal": sig,
        "score": 0,
        "probability_up": 1.0,
        "confidence": 1.0,
        "threshold_used": 0,
        "triggered_rules": np.where(sig == "BUY", "buy_and_hold_entry", "buy_and_hold_carry"),
    }, index=df.index)


# ---------------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------------

def ensemble_majority(per_model_signals: Dict[str, pd.DataFrame], min_votes: int = 2) -> pd.DataFrame:
    """Compute a **binary** consensus signal across the per-model DataFrames.

    Each model casts a BUY or SELL vote. The ensemble picks whichever side
    has more votes; on a tie it falls back to the mean predicted
    ``probability_up`` (>= 0.5 ⇒ BUY else SELL). There is no HOLD class —
    the consensus is always a concrete BUY or SELL.

    ``min_votes`` is retained for signature compatibility and reported in
    ``threshold_used`` but no longer creates a HOLD dead-zone.
    """
    if not per_model_signals:
        raise ValueError("ensemble_majority needs at least one model")
    idx = next(iter(per_model_signals.values())).index
    buy = pd.Series(0, index=idx)
    sell = pd.Series(0, index=idx)
    prob_sum = pd.Series(0.0, index=idx)
    for sigdf in per_model_signals.values():
        buy = buy + (sigdf["signal"].reindex(idx) == "BUY").astype(int)
        sell = sell + (sigdf["signal"].reindex(idx) == "SELL").astype(int)
        prob_sum = prob_sum + sigdf["probability_up"].reindex(idx).fillna(0.5).astype(float)
    n_models = len(per_model_signals)
    mean_prob = prob_sum / max(n_models, 1)
    signal = np.where(
        buy > sell, "BUY",
        np.where(sell > buy, "SELL",
                 np.where(mean_prob >= 0.5, "BUY", "SELL")),
    )
    score = np.where(signal == "SELL", sell, buy)
    confidence = np.clip(np.abs(buy - sell) / max(n_models, 1), 0, 1)
    prob_up = np.where(signal == "BUY", 0.5 + confidence * 0.4,
                       0.5 - confidence * 0.4)
    return pd.DataFrame({
        "signal": signal,
        "score": score,
        "probability_up": np.round(prob_up.astype(float), 4),
        "confidence": np.round(confidence.astype(float), 4),
        "threshold_used": min_votes,
        "triggered_rules": np.where(signal == "BUY", "majority_buy",
                                    np.where(score == sell, "majority_sell", "prob_tiebreak")),
    }, index=idx)
