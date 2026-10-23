"""Training loops and checkpoint management for PyTorch and Keras RNN models.

Includes:
- Train/Validation loss monitoring
- Gradient clipping for PyTorch to mitigate exploding gradients during BPTT
- Early stopping with best checkpoint restoration
- Model artifact serialization
"""

import copy
import os
import time
from typing import Any, Dict, List, Optional, Tuple

# Suppress excessive tensorflow logs and oneDNN warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader



def train_pytorch_rnn(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 50,
    lr: float = 0.001,
    weight_decay: float = 1e-5,
    clip_grad_norm: float = 1.0,
    patience: int = 10,
    device: str = "cpu",
    save_path: Optional[str] = None,
    verbose: bool = True,
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """Train Vanilla RNN model in PyTorch with validation monitoring and early stopping.

    Args:
        model: VanillaRNNPyTorch model
        train_loader: DataLoader for training set
        val_loader: DataLoader for validation set
        epochs: Maximum number of epochs
        lr: Learning rate for Adam optimizer
        weight_decay: L2 regularization
        clip_grad_norm: Maximum norm for gradient clipping (crucial for RNN stability)
        patience: Early stopping patience epochs
        device: 'cpu' or 'cuda'
        save_path: Path to save the best model weights (.pt)
        verbose: Print progress per epoch

    Returns:
        (best_model, history_dict)
    """
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_losses: List[float] = []
    val_losses: List[float] = []

    best_val_loss = float("inf")
    best_weights = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # 1. Training Phase
        model.train()
        running_train_loss = 0.0
        total_train_samples = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()

            # Gradient clipping to prevent exploding gradients during BPTT
            if clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)

            optimizer.step()

            batch_sz = batch_x.size(0)
            running_train_loss += loss.item() * batch_sz
            total_train_samples += batch_sz

        epoch_train_loss = running_train_loss / total_train_samples
        train_losses.append(epoch_train_loss)

        # 2. Validation Phase
        model.eval()
        running_val_loss = 0.0
        total_val_samples = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                preds = model(batch_x)
                loss = criterion(preds, batch_y)

                batch_sz = batch_x.size(0)
                running_val_loss += loss.item() * batch_sz
                total_val_samples += batch_sz

        epoch_val_loss = running_val_loss / total_val_samples
        val_losses.append(epoch_val_loss)

        if verbose and (epoch % 5 == 0 or epoch == 1 or epoch == epochs):
            print(
                f"Epoch [{epoch:03d}/{epochs:03d}] | "
                f"Train MSE: {epoch_train_loss:.6f} | "
                f"Val MSE: {epoch_val_loss:.6f}"
            )

        # 3. Early Stopping Check
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_weights = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            if save_path:
                os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
                torch.save(best_weights, save_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                if verbose:
                    print(f"Early stopping triggered at epoch {epoch}. Best Val MSE: {best_val_loss:.6f}")
                break

    total_time = time.time() - start_time
    if verbose:
        print(f"PyTorch training completed in {total_time:.2f}s. Best Val MSE: {best_val_loss:.6f}")

    # Load best weights
    model.load_state_dict(best_weights)
    model.eval()

    history = {
        "train_loss": train_losses,
        "val_loss": val_losses,
        "best_val_loss": best_val_loss,
        "stopped_epoch": len(train_losses),
        "training_time_seconds": total_time,
    }
    return model, history


def train_keras_rnn(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 50,
    batch_size: int = 32,
    patience: int = 10,
    save_path: Optional[str] = None,
    verbose: int = 0,
) -> Tuple[Any, Dict[str, List[float]]]:
    """Train Vanilla RNN model in Keras with validation monitoring and early stopping.

    Args:
        model: Compiled Keras Sequential RNN model
        X_train: Training features array (N, seq_len, input_size)
        y_train: Training targets array (N, 1)
        X_val: Validation features array
        y_val: Validation targets array
        epochs: Maximum number of epochs
        batch_size: Mini-batch size
        patience: Early stopping patience epochs
        save_path: Path to save the best model weights (.keras)
        verbose: Verbosity level (0 = silent, 1 = progress bar, 2 = one line per epoch)

    Returns:
        (best_model, history_dict)
    """
    import keras

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1 if verbose > 0 else 0,
        )
    ]

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        callbacks.append(
            keras.callbacks.ModelCheckpoint(
                filepath=save_path,
                monitor="val_loss",
                save_best_only=True,
                verbose=0,
            )
        )

    start_time = time.time()
    history_obj = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=verbose,
    )
    total_time = time.time() - start_time

    history = {
        "train_loss": [float(x) for x in history_obj.history["loss"]],
        "val_loss": [float(x) for x in history_obj.history["val_loss"]],
        "best_val_loss": float(min(history_obj.history["val_loss"])),
        "stopped_epoch": len(history_obj.history["loss"]),
        "training_time_seconds": total_time,
    }

    # Ensure best model is saved to save_path even if restore_best_weights was used
    if save_path:
        model.save(save_path)

    return model, history
