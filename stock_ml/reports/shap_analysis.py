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

from config import MODEL_DIR, REPORTS_DIR, RANDOM_STATE, DEFAULT_LABEL_MODE, LABEL_THRESHOLD

logger = logging.getLogger(__name__)

PLOTS_DIR = os.path.join(REPORTS_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# The shap library draws English axis labels; the thesis charts are in Polish.
SUMMARY_XLABEL = "Wartość SHAP (wpływ na wynik modelu)"
BAR_XLABEL = "Średnia |wartość SHAP| (średni wpływ na wynik modelu)"
COLORBAR_LABEL = "Wartość cechy"
COLORBAR_ENDS = ("Niska", "Wysoka")


def _polish_summary_labels(fig) -> None:
    """Translate the axis and colour-bar labels of a SHAP beeswarm plot."""
    main_ax, *other_axes = fig.axes
    main_ax.set_xlabel(SUMMARY_XLABEL)
    if other_axes:
        colorbar_ax = other_axes[-1]
        colorbar_ax.set_ylabel(COLORBAR_LABEL)
        if len(colorbar_ax.get_yticks()) == 2:
            colorbar_ax.set_yticklabels(COLORBAR_ENDS)


def run_shap_analysis(ticker: str, model_name: str = "xgboost", label_version: str = "A",
                      tuned: bool = False, n_samples: int = 500,
                      label_mode: str = DEFAULT_LABEL_MODE,
                      threshold: float = LABEL_THRESHOLD) -> pd.Series | None:
    """
    Generate SHAP plots for a trained model.

    Returns the mean absolute SHAP value of each feature, sorted from the most
    to the least important, or None when the model or scaler is missing.

    Plots saved to reports/plots/:
    - shap_summary_{ticker}_{model}_{label_version}.png
    - shap_bar_{ticker}_{model}_{label_version}.png
    - shap_waterfall_{ticker}_{model}_{label_version}.png

    Args:
        label_version: artifact tag on disk (e.g. "bin2" for binary @ 0.2%).
        n_samples: Number of samples to use for SHAP (subsample for speed)
        label_mode / threshold: passed through so the SHAP feature matrix is
            built with the same labels the model was trained on.
    """
    import shap
    import joblib
    import matplotlib.pyplot as plt
    from features.pipeline import build_feature_matrix

    suffix = "_tuned" if tuned else ""
    logger.info(f"Running SHAP analysis: {ticker} | {model_name}{suffix} | label_version={label_version}")

    X, y = build_feature_matrix(ticker, label_version, label_mode=label_mode, threshold=threshold)

    # Use last portion for SHAP (more representative of recent behavior)
    X_sample = X.iloc[-n_samples:] if len(X) > n_samples else X

    # Load scaler
    scaler_path = os.path.join(MODEL_DIR, f"scaler_{ticker}_{label_version}{suffix}.joblib")
    if not os.path.exists(scaler_path):
        logger.error(f"Scaler not found: {scaler_path}")
        return None
    scaler = joblib.load(scaler_path)
    X_scaled = scaler.transform(X_sample)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X_sample.columns, index=X_sample.index)

    # Load model and create explainer
    if model_name == "xgboost":
        import xgboost as xgb
        model_path = os.path.join(MODEL_DIR, f"xgboost_{ticker}_{label_version}{suffix}.json")
        if not os.path.exists(model_path):
            logger.error(f"Model not found: {model_path}")
            return None
        model = xgb.XGBClassifier()
        model.load_model(model_path)
        explainer = shap.TreeExplainer(model)
    elif model_name == "lightgbm":
        import lightgbm as lgb
        model_path = os.path.join(MODEL_DIR, f"lightgbm_{ticker}_{label_version}{suffix}.txt")
        if not os.path.exists(model_path):
            logger.error(f"Model not found: {model_path}")
            return None
        model = lgb.Booster(model_file=model_path)
        explainer = shap.TreeExplainer(model)
    elif model_name == "random_forest":
        model_path = os.path.join(MODEL_DIR, f"random_forest_{ticker}_{label_version}{suffix}.joblib")
        if not os.path.exists(model_path):
            logger.error(f"Model not found: {model_path}")
            return None
        model = joblib.load(model_path)
        explainer = shap.TreeExplainer(model)
    else:
        logger.error(f"SHAP not supported for model: {model_name}")
        return None

    shap_values = explainer.shap_values(X_scaled_df)

    # For binary classifiers that return list of arrays, take class 1
    if isinstance(shap_values, list):
        shap_vals = shap_values[1]
    else:
        shap_vals = shap_values

    label = f"{ticker} | {model_name}{suffix} | etykieta={label_version}"

    # 1. Summary plot (beeswarm)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_vals, X_scaled_df, show=False, plot_size=None)
    _polish_summary_labels(plt.gcf())
    plt.title(f"Wykres zbiorczy SHAP\n{label}", fontsize=12)
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, f"shap_summary_{ticker}_{model_name}_{label_version}{suffix}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close("all")
    logger.info(f"SHAP summary plot saved: {out}")

    # 2. Bar plot (mean absolute SHAP)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_vals, X_scaled_df, plot_type="bar", show=False, plot_size=None)
    plt.gca().set_xlabel(BAR_XLABEL)
    plt.title(f"Istotność cech (SHAP)\n{label}", fontsize=12)
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, f"shap_bar_{ticker}_{model_name}_{label_version}{suffix}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close("all")
    logger.info(f"SHAP bar plot saved: {out}")

    # 3. Waterfall for a single prediction (last sample)
    try:
        expected_val = explainer.expected_value
        if isinstance(expected_val, list):
            expected_val = expected_val[1]
        elif isinstance(expected_val, np.ndarray):
            expected_val = float(expected_val[1]) if len(expected_val) > 1 else float(expected_val[0])

        exp = shap.Explanation(
            values=shap_vals[-1],
            base_values=expected_val,
            data=X_scaled_df.iloc[-1].values,
            feature_names=list(X_scaled_df.columns)
        )
        plt.figure(figsize=(12, 6))
        shap.waterfall_plot(exp, show=False, max_display=15)
        plt.title(f"Wykres wodospadowy SHAP (ostatnia próbka)\n{label}", fontsize=12)
        plt.tight_layout()
        out = os.path.join(PLOTS_DIR, f"shap_waterfall_{ticker}_{model_name}_{label_version}{suffix}.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close("all")
        logger.info(f"SHAP waterfall plot saved: {out}")
    except Exception as e:
        logger.warning(f"Waterfall plot failed: {e}")

    logger.info(f"SHAP analysis complete for {ticker} | {model_name}")
    importance = np.abs(np.asarray(shap_vals)).mean(axis=0)
    return pd.Series(importance, index=X_scaled_df.columns, name="mean_abs_shap").sort_values(ascending=False)
