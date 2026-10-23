"""Evaluation metrics, predictions, plotting, and reporting for Assignment 06 (RNN).

Computes:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Coefficient of Determination (R²)
- Mean Absolute Percentage Error (MAPE)

Generates:
- Loss curves (Train vs Validation for PyTorch & Keras)
- Actual vs. Predicted time-series charts on Test split
- Residual error distributions and time series
- Comparison summary tables
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple

# Suppress oneDNN messages and excessive tensorflow logs
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader



def calculate_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """Calculate regression metrics on unscaled (original unit) values.

    Args:
        y_true: Ground truth target values (N, 1) or (N,)
        y_pred: Predicted target values (N, 1) or (N,)

    Returns:
        Dict with MAE, RMSE, R2, and MAPE
    """
    y_true_flat = np.array(y_true).ravel()
    y_pred_flat = np.array(y_pred).ravel()

    mae = float(mean_absolute_error(y_true_flat, y_pred_flat))
    mse = float(mean_squared_error(y_true_flat, y_pred_flat))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true_flat, y_pred_flat))

    # MAPE calculation with epsilon clipping to avoid division by zero
    non_zero_mask = np.abs(y_true_flat) > 1e-4
    if np.any(non_zero_mask):
        mape = float(
            np.mean(
                np.abs(
                    (y_true_flat[non_zero_mask] - y_pred_flat[non_zero_mask])
                    / y_true_flat[non_zero_mask]
                )
            )
            * 100.0
        )
    else:
        mape = float("nan")

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "MAPE": mape,
    }


def predict_pytorch(
    model: nn.Module,
    loader: DataLoader,
    target_scaler: Optional[MinMaxScaler] = None,
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate predictions from a PyTorch model and inverse-transform to original scale.

    Args:
        model: Trained PyTorch model
        loader: DataLoader for target dataset
        target_scaler: Fitted MinMaxScaler for target (if None, returns scaled)
        device: Device to run inference on

    Returns:
        (pred_scaled, pred_original)
    """
    model.eval()
    model.to(device)
    preds_list = []

    with torch.no_grad():
        for batch_x, _ in loader:
            batch_x = batch_x.to(device)
            p = model(batch_x)
            preds_list.append(p.cpu().numpy())

    pred_scaled = np.vstack(preds_list)
    if target_scaler is not None:
        pred_original = target_scaler.inverse_transform(pred_scaled)
    else:
        pred_original = pred_scaled.copy()

    return pred_scaled, pred_original


def predict_keras(
    model: Any,
    X: np.ndarray,
    target_scaler: Optional[MinMaxScaler] = None,
    batch_size: int = 32,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate predictions from a Keras model and inverse-transform to original scale.

    Args:
        model: Trained Keras model
        X: Feature sequence array (N, seq_len, input_size)
        target_scaler: Fitted MinMaxScaler for target
        batch_size: Batch size for inference

    Returns:
        (pred_scaled, pred_original)
    """
    pred_scaled = model.predict(X, batch_size=batch_size, verbose=0)
    if target_scaler is not None:
        pred_original = target_scaler.inverse_transform(pred_scaled)
    else:
        pred_original = pred_scaled.copy()

    return pred_scaled, pred_original


def plot_loss_curves(
    pytorch_history: Dict[str, List[float]],
    keras_history: Dict[str, List[float]],
    dataset_name: str,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot training and validation MSE loss curves across epochs for PyTorch and Keras.

    Args:
        pytorch_history: History dict from train_pytorch_rnn
        keras_history: History dict from train_keras_rnn
        dataset_name: Name of dataset (e.g. 'Apple Stock' or 'Gold Commodity')
        save_path: Optional path to save the generated figure

    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    # PyTorch Loss
    epochs_pt = range(1, len(pytorch_history["train_loss"]) + 1)
    axes[0].plot(epochs_pt, pytorch_history["train_loss"], label="Train Loss (MSE)", color="#2563eb", lw=2)
    axes[0].plot(epochs_pt, pytorch_history["val_loss"], label="Val Loss (MSE)", color="#dc2626", lw=2, ls="--")
    axes[0].set_title(f"PyTorch Vanilla RNN - {dataset_name}", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=11)
    axes[0].set_ylabel("Loss (MSE, Scaled)", fontsize=11)
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(frameon=True)

    # Keras Loss
    epochs_kr = range(1, len(keras_history["train_loss"]) + 1)
    axes[1].plot(epochs_kr, keras_history["train_loss"], label="Train Loss (MSE)", color="#059669", lw=2)
    axes[1].plot(epochs_kr, keras_history["val_loss"], label="Val Loss (MSE)", color="#ea580c", lw=2, ls="--")
    axes[1].set_title(f"Keras Vanilla RNN - {dataset_name}", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=11)
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend(frameon=True)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_actual_vs_predicted(
    dates: pd.Series,
    y_true: np.ndarray,
    y_pred_pytorch: np.ndarray,
    y_pred_keras: np.ndarray,
    dataset_name: str,
    unit: str = "USD",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot actual vs predicted prices on the test split for both PyTorch and Keras models.

    Args:
        dates: Series of DateTime stamps corresponding to test set points
        y_true: Actual test prices (original scale)
        y_pred_pytorch: PyTorch predictions (original scale)
        y_pred_keras: Keras predictions (original scale)
        dataset_name: Title label
        unit: Price unit string (USD)
        save_path: Optional save destination

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    dates_arr = pd.to_datetime(dates).values
    y_true_flat = np.array(y_true).ravel()
    pt_flat = np.array(y_pred_pytorch).ravel()
    kr_flat = np.array(y_pred_keras).ravel()

    ax.plot(dates_arr, y_true_flat, label="Ground Truth (Actual)", color="#1e293b", lw=2.2, alpha=0.9)
    ax.plot(dates_arr, pt_flat, label="PyTorch Vanilla RNN", color="#2563eb", lw=1.8, ls="--")
    ax.plot(dates_arr, kr_flat, label="Keras Vanilla RNN", color="#10b981", lw=1.8, ls=":")

    ax.set_title(f"Test Set Prediction: Actual vs. Predicted ({dataset_name})", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel(f"Price ({unit})", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True, fontsize=11, loc="upper left")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_residuals(
    dates: pd.Series,
    y_true: np.ndarray,
    y_pred_pytorch: np.ndarray,
    y_pred_keras: np.ndarray,
    dataset_name: str,
    unit: str = "USD",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot prediction residual errors (Actual - Predicted) over time and distribution.

    Args:
        dates: Dates series
        y_true: Ground truth
        y_pred_pytorch: PyTorch predictions
        y_pred_keras: Keras predictions
        dataset_name: Name of dataset
        unit: Unit string
        save_path: Save file path

    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    dates_arr = pd.to_datetime(dates).values
    y_true_flat = np.array(y_true).ravel()
    res_pt = y_true_flat - np.array(y_pred_pytorch).ravel()
    res_kr = y_true_flat - np.array(y_pred_keras).ravel()

    # Time series residuals
    axes[0].plot(dates_arr, res_pt, label="PyTorch Residual", color="#2563eb", alpha=0.7, lw=1.2)
    axes[0].plot(dates_arr, res_kr, label="Keras Residual", color="#10b981", alpha=0.7, lw=1.2)
    axes[0].axhline(0, color="black", linestyle="--", lw=1)
    axes[0].set_title(f"Residual Errors Over Time ({dataset_name})", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Date")
    axes[0].set_ylabel(f"Residual ({unit})")
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend()

    # Residual distribution histogram
    axes[1].hist(res_pt, bins=30, alpha=0.5, color="#2563eb", label=f"PyTorch (std={np.std(res_pt):.2f})", density=True)
    axes[1].hist(res_kr, bins=30, alpha=0.5, color="#10b981", label=f"Keras (std={np.std(res_kr):.2f})", density=True)
    axes[1].axvline(0, color="black", linestyle="--", lw=1)
    axes[1].set_title(f"Residual Distribution ({dataset_name})", fontsize=12, fontweight="bold")
    axes[1].set_xlabel(f"Error ({unit})")
    axes[1].set_ylabel("Density")
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend()

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def build_comparison_dataframe(
    pt_metrics: Dict[str, float],
    kr_metrics: Dict[str, float],
    dataset_name: str,
) -> pd.DataFrame:
    """Create a structured comparison DataFrame between PyTorch and Keras."""
    df = pd.DataFrame(
        {
            "Dataset": [dataset_name, dataset_name],
            "Framework": ["PyTorch", "TensorFlow/Keras"],
            "Model": ["Vanilla RNN (nn.RNN)", "Vanilla RNN (SimpleRNN)"],
            "MAE": [pt_metrics["MAE"], kr_metrics["MAE"]],
            "RMSE": [pt_metrics["RMSE"], kr_metrics["RMSE"]],
            "R2": [pt_metrics["R2"], kr_metrics["R2"]],
            "MAPE (%)": [pt_metrics["MAPE"], kr_metrics["MAPE"]],
        }
    )
    return df
