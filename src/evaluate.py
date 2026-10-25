import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


def evaluate_model(y_true, y_proba, threshold=0.5):
    y_pred = (y_proba >= threshold).astype(int)

    metrics = {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    return metrics


def print_classification_summary(y_true, y_proba, threshold=0.5):
    y_pred = (y_proba >= threshold).astype(int)

    print("Classification Report")
    print(classification_report(y_true, y_pred, zero_division=0))

    print("Confusion Matrix")
    print(confusion_matrix(y_true, y_pred))


def save_metrics(metrics: dict, path: Path):
    with open(path, "w") as f:
        json.dump(metrics, f, indent=4)