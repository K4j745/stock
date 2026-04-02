"""
SHAP analysis: generate summary plot, bar plot, and waterfall plot.
Loads saved model and scaler, computes SHAP values, saves plots as PNG.
"""
import matplotlib
matplotlib.use('Agg')

import os
import logging
import numpy as np
import pandas as pd

from config import MODEL_DIR, REPORTS_DIR, RANDOM_STATE

logger = logging.getLogger(__name__)

PLOTS_DIR = os.path.join(REPORTS_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


def run_shap_analysis(ticker: str, model_name: str = "xgboost", label_version: str = "A",
                      tuned: bool = False, n_samples: int = 500):
    """
    Generate SHAP plots for a trained model.

    Plots saved to reports/plots/:
    - shap_summary_{ticker}_{model}_{label_version}.png
    - shap_bar_{ticker}_{model}_{label_version}.png
    - shap_waterfall_{ticker}_{model}_{label_version}.png

    Args:
        n_samples: Number of samples to use for SHAP (subsample for speed)
    """
    import shap
    import joblib
    import matplotlib.pyplot as plt
    from features.pipeline import build_feature_matrix

    suffix = "_tuned" if tuned else ""
    logger.info(f"Running SHAP analysis: {ticker} | {model_name}{suffix} | label_version={label_version}")

    X, y = build_feature_matrix(ticker, label_version)

    # Use last portion for SHAP (more representative of recent behavior)
    X_sample = X.iloc[-n_samples:] if len(X) > n_samples else X

    # Load scaler
    scaler_path = os.path.join(MODEL_DIR, f"scaler_{ticker}_{label_version}{suffix}.joblib")
    if not os.path.exists(scaler_path):
        logger.error(f"Scaler not found: {scaler_path}")
        return
    scaler = joblib.load(scaler_path)
    X_scaled = scaler.transform(X_sample)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X_sample.columns, index=X_sample.index)

    # Load model and create explainer
    if model_name == "xgboost":
        import xgboost as xgb
        model_path = os.path.join(MODEL_DIR, f"xgboost_{ticker}_{label_version}{suffix}.json")
        if not os.path.exists(model_path):
            logger.error(f"Model not found: {model_path}")
            return
        model = xgb.XGBClassifier()
        model.load_model(model_path)
        explainer = shap.TreeExplainer(model)
    elif model_name == "lightgbm":
        import lightgbm as lgb
        model_path = os.path.join(MODEL_DIR, f"lightgbm_{ticker}_{label_version}{suffix}.txt")
        if not os.path.exists(model_path):
            logger.error(f"Model not found: {model_path}")
            return
        model = lgb.Booster(model_file=model_path)
        explainer = shap.TreeExplainer(model)
    elif model_name == "random_forest":
        model_path = os.path.join(MODEL_DIR, f"random_forest_{ticker}_{label_version}{suffix}.joblib")
        if not os.path.exists(model_path):
            logger.error(f"Model not found: {model_path}")
            return
        model = joblib.load(model_path)
        explainer = shap.TreeExplainer(model)
    else:
        logger.error(f"SHAP not supported for model: {model_name}")
        return

    shap_values = explainer.shap_values(X_scaled_df)

    # For binary classifiers that return list of arrays, take class 1
    if isinstance(shap_values, list):
        shap_vals = shap_values[1]
    else:
        shap_vals = shap_values

    label = f"{ticker} | {model_name}{suffix} | label={label_version}"

    # 1. Summary plot (beeswarm)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_vals, X_scaled_df, show=False, plot_size=None)
    plt.title(f"SHAP Summary Plot\n{label}", fontsize=12)
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, f"shap_summary_{ticker}_{model_name}_{label_version}{suffix}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close("all")
    logger.info(f"SHAP summary plot saved: {out}")

    # 2. Bar plot (mean absolute SHAP)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_vals, X_scaled_df, plot_type="bar", show=False, plot_size=None)
    plt.title(f"SHAP Feature Importance\n{label}", fontsize=12)
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, f"shap_bar_{ticker}_{model_name}_{label_version}{suffix}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close("all")
    logger.info(f"SHAP bar plot saved: {out}")

    # 3. Waterfall for a single prediction (last sample)
    try:
        exp = shap.Explanation(
            values=shap_vals[-1],
            base_values=explainer.expected_value if not isinstance(explainer.expected_value, list)
                        else explainer.expected_value[1],
            data=X_scaled_df.iloc[-1].values,
            feature_names=list(X_scaled_df.columns)
        )
        plt.figure(figsize=(12, 6))
        shap.waterfall_plot(exp, show=False, max_display=15)
        plt.title(f"SHAP Waterfall (last sample)\n{label}", fontsize=12)
        plt.tight_layout()
        out = os.path.join(PLOTS_DIR, f"shap_waterfall_{ticker}_{model_name}_{label_version}{suffix}.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close("all")
        logger.info(f"SHAP waterfall plot saved: {out}")
    except Exception as e:
        logger.warning(f"Waterfall plot failed: {e}")

    logger.info(f"SHAP analysis complete for {ticker} | {model_name}")
