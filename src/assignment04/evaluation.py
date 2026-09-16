"""Metric helpers; metrics are not used to compute training gradients."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(y_true, logits):
    logits = np.asarray(logits)
    prediction = logits.argmax(axis=1)
    shifted = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    try:
        roc_auc = float(roc_auc_score(y_true, probabilities[:, 1]))
    except ValueError:
        roc_auc = float("nan")
    return {
        "accuracy": float(accuracy_score(y_true, prediction)),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "roc_auc": roc_auc,
        "confusion_matrix": confusion_matrix(y_true, prediction, labels=[0, 1]).tolist(),
    }


def regression_metrics(y_true, prediction):
    y_true = np.asarray(y_true, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    # A transient House checkpoint can exponentiate beyond float64 on an
    # extreme row. It is scientifically the worst possible validation
    # candidate, not a reason to abort before restoring the best checkpoint.
    if not np.isfinite(y_true).all() or not np.isfinite(prediction).all():
        return {"mae": float("inf"), "rmse": float("inf"), "r2": float("-inf")}
    mse = mean_squared_error(y_true, prediction)
    return {
        "mae": float(mean_absolute_error(y_true, prediction)),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y_true, prediction)),
    }
