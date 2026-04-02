import logging

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger("stock_ml")


def evaluate_model(y_true, y_pred, y_prob=None) -> dict:
    """Compute classification metrics.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        y_prob: Predicted probabilities for the positive class (optional).

    Returns:
        Dictionary with accuracy, precision, recall, f1, roc_auc, mcc.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }

    if y_prob is not None and len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
    else:
        metrics["roc_auc"] = np.nan

    return metrics


def print_classification_report(results: dict) -> None:
    """Log formatted classification results for all models."""
    for model_name, model_results in results.items():
        logger.info("=" * 60)
        logger.info("Model: %s", model_name)
        logger.info("-" * 60)

        for i, fold_metrics in enumerate(model_results["fold_metrics"]):
            logger.info(
                "  Fold %d: acc=%.4f prec=%.4f rec=%.4f f1=%.4f auc=%.4f mcc=%.4f",
                i + 1,
                fold_metrics["accuracy"],
                fold_metrics["precision"],
                fold_metrics["recall"],
                fold_metrics["f1"],
                fold_metrics.get("roc_auc", float("nan")),
                fold_metrics["mcc"],
            )

        mean = model_results["mean_metrics"]
        logger.info(
            "  MEAN:   acc=%.4f prec=%.4f rec=%.4f f1=%.4f auc=%.4f mcc=%.4f",
            mean["accuracy"],
            mean["precision"],
            mean["recall"],
            mean["f1"],
            mean.get("roc_auc", float("nan")),
            mean["mcc"],
        )
        logger.info("=" * 60)
