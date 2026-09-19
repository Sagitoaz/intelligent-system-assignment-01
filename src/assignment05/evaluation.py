"""Evaluation metrics and visualization helpers for Assignment 05.

Computes:
- Accuracy, Precision (macro), Recall (macro), F1 (macro)
- ROC-AUC (binary or multi-class OvR)
- Confusion Matrix

Visualizations:
- Training and validation loss curves
- Training and validation metric curves
- Confusion matrix heatmaps
- Multi-model comparison bar charts
"""

import os
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def compute_metrics(
    y_true: np.ndarray,
    logits: np.ndarray,
    is_binary: bool = False,
    compute_auc: bool = True,
) -> Dict[str, Union[float, np.ndarray]]:
    """Compute standard classification metrics from logits."""
    y_true = np.asarray(y_true, dtype=np.int64)
    logits = np.asarray(logits, dtype=np.float32)

    # Softmax probabilities
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    y_pred = np.argmax(probs, axis=1)

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    roc_auc = float("nan")
    if compute_auc:
        try:
            if is_binary:
                # Column 1 is probability of positive class
                roc_auc = float(roc_auc_score(y_true, probs[:, 1]))
            else:
                roc_auc = float(
                    roc_auc_score(y_true, probs, multi_class="ovr", average="macro")
                )
        except Exception:
            pass

    cm = confusion_matrix(y_true, y_pred)

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": cm,
        "predictions": y_pred,
        "probabilities": probs,
    }


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    is_binary: bool = False,
    compute_auc: bool = True,
) -> Dict[str, Union[float, np.ndarray]]:
    """Evaluate PyTorch model on a DataLoader."""
    model.eval()
    all_logits = []
    all_targets = []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            logits = model(x)
            all_logits.append(logits.cpu().numpy())
            all_targets.append(y.numpy())

    all_logits = np.concatenate(all_logits, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    return compute_metrics(
        all_targets, all_logits, is_binary=is_binary, compute_auc=compute_auc
    )


# =====================================================================
# PLOTTING HELPERS
# =====================================================================


def plot_training_curves(
    history: Dict[str, List[float]],
    title_prefix: str = "Model",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot Loss and F1 / Accuracy curves across training epochs."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Loss curve
    ax1.plot(epochs, history["train_loss"], "o-", label="Train Loss", color="#1f77b4")
    ax1.plot(epochs, history["val_loss"], "s--", label="Val Loss", color="#ff7f0e")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("CrossEntropy Loss", fontsize=11)
    ax1.set_title(f"{title_prefix} - Loss Curve", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend()

    # Metric curve (F1 or Accuracy)
    metric_key = "val_f1" if "val_f1" in history else "val_acc"
    train_metric_key = "train_f1" if "train_f1" in history else "train_acc"
    label_name = "Macro F1" if "f1" in metric_key else "Accuracy"

    ax2.plot(
        epochs,
        history[train_metric_key],
        "o-",
        label=f"Train {label_name}",
        color="#2ca02c",
    )
    ax2.plot(
        epochs,
        history[metric_key],
        "s--",
        label=f"Val {label_name}",
        color="#d62728",
    )
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel(label_name, fontsize=11)
    ax2.set_title(f"{title_prefix} - {label_name} Curve", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot annotated confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel="True Label",
        xlabel="Predicted Label",
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                f"{cm[i, j]:d}",
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=9 if len(class_names) > 5 else 12,
            )

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_comparison_bar(
    df: pd.DataFrame,
    metrics: List[str],
    title: str = "Model Comparison",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot grouped bar chart comparing multiple models across key metrics."""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(df))
    width = 0.8 / len(metrics)

    for i, metric in enumerate(metrics):
        if metric in df.columns:
            offset = (i - len(metrics) / 2) * width + width / 2
            values = df[metric].values
            bars = ax.bar(x + offset, values, width, label=metric.upper())
            # add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                if not np.isnan(height):
                    ax.annotate(
                        f"{height:.3f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(df["Model"], fontsize=10, fontweight="medium")
    ax.set_ylabel("Score", fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    ax.legend(loc="lower right")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig
