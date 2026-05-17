"""Bridge between the dashboard and the ``stock_ml`` pipeline.

When trained model artifacts exist on disk we load them and use them to
generate real probability-of-up series. When they don't (the common case in
fresh checkouts and on CI without GPUs), we fall back to the deterministic
*probability proxies* defined in :mod:`signals`.

The fallback path is logged loudly and tagged in the resulting record's
``signal_source`` so any reader of the dashboard can tell a real ML signal
from a proxy at a glance.
"""
from __future__ import annotations

import logging
import os
from typing import Dict, Optional, Tuple

import pandas as pd

from . import signals as sig_module

logger = logging.getLogger("dashboard.ml")


def _model_path(models_dir: str, model_name: str, ticker: str, label_version: str = "A") -> Optional[str]:
    """Resolve the on-disk filename pattern used by ``stock_ml.models.train``."""
    if not os.path.isdir(models_dir):
        return None
    ext_by_model = {"xgboost": ".json", "lightgbm": ".txt"}
    ext = ext_by_model.get(model_name, ".joblib")
    candidate = os.path.join(models_dir, f"{model_name}_{ticker}_{label_version}{ext}")
    return candidate if os.path.exists(candidate) else None


def _load_real_probabilities(model_name: str, ticker: str, df_indicators: pd.DataFrame,
                             ml_cfg: Dict) -> Optional[pd.Series]:
    """Attempt to load a trained model and emit probability-of-up for ``df_indicators.index``.

    Returns ``None`` if any artifact is missing — the caller should then fall
    back to a proxy.
    """
    models_dir = ml_cfg.get("models_dir", "stock_ml/models/saved")
    label_version = ml_cfg.get("label_version", "A")

    model_file = _model_path(models_dir, model_name, ticker, label_version)
    scaler_file = os.path.join(models_dir, f"scaler_{ticker}_{label_version}.joblib")
    if model_file is None or not os.path.exists(scaler_file):
        return None

    # Defer heavy imports — we may go the proxy route on most runs.
    try:
        import joblib
    except ImportError:  # pragma: no cover
        return None

    try:
        scaler = joblib.load(scaler_file)
    except Exception as exc:  # pragma: no cover
        logger.warning("Could not load scaler for %s: %s", ticker, exc)
        return None

    # The real models expect the same feature columns that stock_ml.features
    # produced. To stay loosely coupled we only attempt the call if the
    # scaler exposes a feature_names_in_ attribute (sklearn ≥ 1.0).
    feature_names = getattr(scaler, "feature_names_in_", None)
    if feature_names is None:
        logger.info("Scaler for %s missing feature_names_in_; skipping real model use.", ticker)
        return None

    missing = [f for f in feature_names if f not in df_indicators.columns]
    if missing:
        logger.info("Skipping real %s for %s (missing features: %s)", model_name, ticker, missing[:3])
        return None

    X = df_indicators[list(feature_names)].dropna()
    if X.empty:
        return None
    X_scaled = scaler.transform(X)

    try:
        if model_name == "xgboost":
            import xgboost as xgb
            model = xgb.XGBClassifier()
            model.load_model(model_file)
            prob = model.predict_proba(X_scaled)[:, 1]
        elif model_name == "lightgbm":
            import lightgbm as lgb
            model = lgb.Booster(model_file=model_file)
            prob = model.predict(X_scaled)
        else:
            model = joblib.load(model_file)
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(X_scaled)[:, 1]
            else:
                prob = model.predict(X_scaled).astype(float)
    except Exception as exc:  # pragma: no cover
        logger.warning("Real %s inference failed for %s: %s", model_name, ticker, exc)
        return None

    return pd.Series(prob, index=X.index)


def model_probabilities(model_name: str, ticker: str, df_indicators: pd.DataFrame,
                         ml_cfg: Dict) -> Tuple[pd.Series, str, str]:
    """Return (probabilities, signal_source, model_version).

    ``signal_source`` is one of ``"<model_name>"`` (real) or
    ``"<model_name>_proxy"`` (deterministic stand-in).
    """
    if ml_cfg.get("use_proxy_when_artifacts_missing", True):
        real = _load_real_probabilities(model_name, ticker, df_indicators, ml_cfg)
        if real is not None and not real.empty:
            return real, model_name, "stock_ml-artifact"

    fn = sig_module.PROXY_FUNCTIONS.get(model_name)
    if fn is None:
        raise ValueError(f"No proxy defined for model '{model_name}'")
    return fn(df_indicators), f"{model_name}_proxy", "proxy-1.0"


def detect_real_artifacts(ml_cfg: Dict) -> Dict[str, bool]:
    """Light check used for audit metadata: which models have artifacts on disk."""
    models_dir = ml_cfg.get("models_dir", "stock_ml/models/saved")
    out = {}
    for model in ml_cfg.get("models", []):
        ext_by_model = {"xgboost": ".json", "lightgbm": ".txt"}
        ext = ext_by_model.get(model, ".joblib")
        # any file matching pattern?
        present = False
        if os.path.isdir(models_dir):
            for fname in os.listdir(models_dir):
                if fname.startswith(f"{model}_") and fname.endswith(ext):
                    present = True
                    break
        out[model] = present
    return out
