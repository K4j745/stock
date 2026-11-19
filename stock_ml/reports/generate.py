import json
import logging
import os

import pandas as pd

from config import REPORTS_DIR

logger = logging.getLogger("stock_ml")


def export_results(results: dict, ticker: str, label_version: str) -> None:
    """Export cross-validation results to CSV and JSON.

    Creates a table with columns: model, fold, accuracy, precision, recall, f1, roc_auc, mcc
    Plus a 'mean' row per model.

    Args:
        results: Dict of {model_name: {fold_metrics: [...], mean_metrics: {...}}}
        ticker: Stock ticker symbol.
        label_version: Label version ("A" or "B").
    """
    rows = []

    for model_name, model_results in results.items():
        # Individual fold metrics
        for i, fold_metrics in enumerate(model_results["fold_metrics"]):
            row = {
                "model": model_name,
                "fold": i + 1,
                "accuracy": fold_metrics["accuracy"],
                "precision": fold_metrics["precision"],
                "recall": fold_metrics["recall"],
                "f1": fold_metrics["f1"],
                "roc_auc": fold_metrics.get("roc_auc"),
                "mcc": fold_metrics["mcc"],
            }
            rows.append(row)

        # Mean metrics row
        mean = model_results["mean_metrics"]
        mean_row = {
            "model": model_name,
            "fold": "mean",
            "accuracy": mean["accuracy"],
            "precision": mean["precision"],
            "recall": mean["recall"],
            "f1": mean["f1"],
            "roc_auc": mean.get("roc_auc"),
            "mcc": mean["mcc"],
        }
        rows.append(mean_row)

    df = pd.DataFrame(rows)

    # Save CSV
    csv_path = os.path.join(REPORTS_DIR, f"{ticker}_{label_version}_results.csv")
    df.to_csv(csv_path, index=False)
    logger.info("Exported results CSV to %s", csv_path)

    # Save JSON
    json_path = os.path.join(REPORTS_DIR, f"{ticker}_{label_version}_results.json")
    # Convert to serializable format
    json_data = {}
    for model_name, model_results in results.items():
        json_data[model_name] = {
            "fold_metrics": model_results["fold_metrics"],
            "mean_metrics": model_results["mean_metrics"],
        }

    # Handle NaN for JSON serialization
    json_str = json.dumps(json_data, indent=2, default=_json_serializer)
    with open(json_path, "w") as f:
        f.write(json_str)
    logger.info("Exported results JSON to %s", json_path)


def _json_serializer(obj):
    """Handle NaN and other non-serializable types for JSON."""
    import math
    if isinstance(obj, float) and math.isnan(obj):
        return None
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
