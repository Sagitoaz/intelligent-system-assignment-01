"""Training routines, early stopping, and checkpointing for Assignment 05."""

import copy
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .evaluation import compute_metrics


def set_seed(seed: int = 42) -> None:
    """Ensure reproducibility across random, numpy, and torch."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Enforce deterministic algorithms where possible
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 10,
    lr: float = 0.001,
    weight_decay: float = 1e-4,
    device: Optional[torch.device] = None,
    early_stopping_patience: int = 3,
    metric_monitor: str = "val_loss",  # 'val_loss' (min) or 'val_f1' (max)
    checkpoint_path: Optional[str] = None,
    verbose: bool = True,
    is_binary: bool = False,
    class_weights: Optional[torch.Tensor] = None,
) -> Tuple[nn.Module, Dict[str, List[float]], int, float]:
    """Train a PyTorch model with early stopping, history tracking, and checkpointing.

    Returns:
        best_model: Model loaded with the weights from the best epoch.
        history: Dict of train/val losses and metrics per epoch.
        best_epoch: The 1-indexed epoch where the best metric was recorded.
        total_time: Training time in seconds.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = (
        nn.CrossEntropyLoss(weight=class_weights.to(device))
        if class_weights is not None
        else nn.CrossEntropyLoss()
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "train_f1": [],
        "val_f1": [],
    }

    best_score = float("inf") if metric_monitor == "val_loss" else float("-inf")
    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 1
    no_improve_count = 0

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # -------------------------------------------------------------
        # TRAINING PHASE
        # -------------------------------------------------------------
        model.train()
        train_loss_sum = 0.0
        train_samples = 0
        all_train_logits = []
        all_train_targets = []

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            batch_size = x_batch.size(0)
            train_loss_sum += loss.item() * batch_size
            train_samples += batch_size

            all_train_logits.append(logits.detach().cpu().numpy())
            all_train_targets.append(y_batch.detach().cpu().numpy())

        epoch_train_loss = train_loss_sum / train_samples
        train_metrics = compute_metrics(
            np.concatenate(all_train_targets, axis=0),
            np.concatenate(all_train_logits, axis=0),
            is_binary=is_binary,
            compute_auc=is_binary,
        )

        # -------------------------------------------------------------
        # VALIDATION PHASE
        # -------------------------------------------------------------
        model.eval()
        val_loss_sum = 0.0
        val_samples = 0
        all_val_logits = []
        all_val_targets = []

        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val = x_val.to(device)
                y_val = y_val.to(device)
                logits = model(x_val)
                loss = criterion(logits, y_val)

                batch_size = x_val.size(0)
                val_loss_sum += loss.item() * batch_size
                val_samples += batch_size

                all_val_logits.append(logits.cpu().numpy())
                all_val_targets.append(y_val.cpu().numpy())

        epoch_val_loss = val_loss_sum / val_samples
        val_metrics = compute_metrics(
            np.concatenate(all_val_targets, axis=0),
            np.concatenate(all_val_logits, axis=0),
            is_binary=is_binary,
            compute_auc=is_binary,
        )

        # Record metrics
        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["train_acc"].append(train_metrics["accuracy"])
        history["val_acc"].append(val_metrics["accuracy"])
        history["train_f1"].append(train_metrics["f1"])
        history["val_f1"].append(val_metrics["f1"])

        if verbose:
            print(
                f"Epoch {epoch:02d}/{epochs:02d} | "
                f"Train Loss: {epoch_train_loss:.4f} - Acc: {train_metrics['accuracy']:.4f} - F1: {train_metrics['f1']:.4f} | "
                f"Val Loss: {epoch_val_loss:.4f} - Acc: {val_metrics['accuracy']:.4f} - F1: {val_metrics['f1']:.4f}"
            )

        # Check for best epoch
        current_score = (
            epoch_val_loss if metric_monitor == "val_loss" else val_metrics["f1"]
        )
        improved = (
            current_score < best_score
            if metric_monitor == "val_loss"
            else current_score > best_score
        )

        if improved:
            best_score = current_score
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())
            no_improve_count = 0
            if checkpoint_path:
                os.makedirs(os.path.dirname(os.path.abspath(checkpoint_path)), exist_ok=True)
                torch.save(best_weights, checkpoint_path)
        else:
            no_improve_count += 1
            if no_improve_count >= early_stopping_patience:
                if verbose:
                    print(
                        f"Early stopping triggered at epoch {epoch}. Best epoch was {best_epoch}."
                    )
                break

    total_time = time.time() - start_time
    model.load_state_dict(best_weights)
    return model, history, best_epoch, total_time
