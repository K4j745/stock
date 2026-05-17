"""Signal generators — one function per "source".

Each generator takes a DataFrame with OHLCV + indicators attached (see
``indicators.compute``) and returns a DataFrame with the columns:

    signal        — "BUY" / "SELL" / "HOLD"
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
    """Score-based BUY/SELL/HOLD from RSI / MACD / Bollinger / SMA.

    Each side has four rules (max score = 4). The signal fires when
    ``buy_score >= rules["buy_threshold"]`` (default 3) or the SELL mirror.
    HOLD here means "not enough triggered rules" — it is *not* a third class.

    KNOWN ISSUE (default thresholds = 3): the BUY rules are partly mutually
    exclusive (e.g. ``Close < BB_lower`` and ``Close > SMA20 && Close > SMA50``
    rarely co-occur), so on a typical 3-year universe the score almost never
    reaches 3 → the stream sits at ~100% HOLD. Lower the threshold to 2 or
    relax one of the rules to make this generator useful. See the README
    section "How signals are produced" for the full discussion + empirical
    distribution per model.
    """
    rsi_buy = rules.get("rsi_buy", 35)
    rsi_sell = rules.get("rsi_sell", 65)
    buy_thr = rules.get("buy_threshold", 3)
    sell_thr = rules.get("sell_threshold", 3)

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

    signal = np.where(
        buy_score >= buy_thr, "BUY",
        np.where(sell_score >= sell_thr, "SELL", "HOLD"),
    )
    score = np.where(signal == "SELL", sell_score, buy_score)

    # Confidence = score / max_possible (4 rules per side)
    confidence = np.clip(score / 4.0, 0, 1)

    # Probability of "up" — derived heuristically: BUY => high, SELL => low
    prob_up = np.where(
        signal == "BUY", 0.5 + confidence * 0.4,
        np.where(signal == "SELL", 0.5 - confidence * 0.4, 0.5),
    )

    triggered = np.where(signal == "BUY", triggered_buy,
                         np.where(signal == "SELL", triggered_sell, ""))

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
    """Convert a probability-of-up series into BUY/SELL/HOLD signals.

    Used by every ML model. Binary task (``label_mode = binary``): the model
    predicts P(next-day return > LABEL_THRESHOLD). The mapping is:

        BUY  if probability_up >= buy_threshold   (default 0.55)
        SELL if probability_up <= sell_threshold  (default 0.45)
        HOLD otherwise                            (the 0.45–0.55 dead band)

    For ML signals **HOLD == "model unsure"** (probability in the dead band),
    *not* a separately predicted class. See README "How signals are produced".
    """
    sig = np.where(prob_up >= buy_threshold, "BUY",
                   np.where(prob_up <= sell_threshold, "SELL", "HOLD"))
    confidence = np.abs(prob_up - 0.5) * 2.0
    return pd.DataFrame({
        "signal": sig,
        "score": np.round(prob_up.fillna(0.5).astype(float) * 100, 2),
        "probability_up": np.round(prob_up.astype(float), 4),
        "confidence": np.round(confidence.astype(float), 4),
        "threshold_used": buy_threshold,
        "triggered_rules": np.where(sig == "BUY", "prob_above_buy_threshold",
                                    np.where(sig == "SELL", "prob_below_sell_threshold", "no_threshold_crossed")),
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


PROXY_FUNCTIONS = {
    "logistic_regression": proxy_logistic,
    "random_forest": proxy_random_forest,
    "xgboost": proxy_xgboost,
    "lightgbm": proxy_lightgbm,
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
    """Compute a consensus signal across the supplied per-model DataFrames.

    BUY requires ``min_votes`` ML models voting BUY *and zero* SELL votes
    (and the mirror for SELL). Anything else is HOLD. Here HOLD means
    "models disagree, or fewer than ``min_votes`` agree" — i.e. it is an
    explicit *disagreement* label, not a model's individual decision.
    """
    if not per_model_signals:
        raise ValueError("ensemble_majority needs at least one model")
    idx = next(iter(per_model_signals.values())).index
    buy = pd.Series(0, index=idx)
    sell = pd.Series(0, index=idx)
    for sigdf in per_model_signals.values():
        buy = buy + (sigdf["signal"].reindex(idx) == "BUY").astype(int)
        sell = sell + (sigdf["signal"].reindex(idx) == "SELL").astype(int)
    n_models = len(per_model_signals)
    signal = np.where(
        (buy >= min_votes) & (sell == 0), "BUY",
        np.where((sell >= min_votes) & (buy == 0), "SELL", "HOLD"),
    )
    score = np.where(signal == "SELL", sell, buy)
    confidence = np.clip(score / max(n_models, 1), 0, 1)
    prob_up = np.where(signal == "BUY", 0.5 + confidence * 0.4,
                       np.where(signal == "SELL", 0.5 - confidence * 0.4, 0.5))
    return pd.DataFrame({
        "signal": signal,
        "score": score,
        "probability_up": np.round(prob_up.astype(float), 4),
        "confidence": np.round(confidence.astype(float), 4),
        "threshold_used": min_votes,
        "triggered_rules": np.where(signal == "BUY", f"majority_buy>={min_votes}",
                                    np.where(signal == "SELL", f"majority_sell>={min_votes}", "no_majority")),
    }, index=idx)
