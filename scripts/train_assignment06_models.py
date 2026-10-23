"""End-to-end training and evaluation script for Assignment 06 (RNN).

Trains 4 models:
1. Stock (AAPL) - PyTorch Vanilla RNN
2. Stock (AAPL) - Keras Vanilla RNN
3. Commodity (Gold) - PyTorch Vanilla RNN
4. Commodity (Gold) - Keras Vanilla RNN

Saves:
- Models to models/assignment06/
- Evaluation metrics to results/assignment06/
- Visualizations to figures/assignment06/
"""

import json
import os
import sys

# Suppress TensorFlow logging before imports
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# Add project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.assignment06.data import (
    prepare_time_series_data,
    create_pytorch_dataloaders,
    get_dataset_metadata,
)
from src.assignment06.models_pytorch import VanillaRNNPyTorch
from src.assignment06.models_keras import build_keras_rnn
from src.assignment06.training import train_pytorch_rnn, train_keras_rnn
from src.assignment06.evaluation import (
    calculate_regression_metrics,
    predict_pytorch,
    predict_keras,
    plot_loss_curves,
    plot_actual_vs_predicted,
    plot_residuals,
    build_comparison_dataframe,
)


def plot_dataset_eda(bundle, title_prefix, save_path):
    """Plot time-series overview, train/val/test splits, and volume."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={"height_ratios": [2.5, 1]})

    df = bundle.df_raw
    dates = pd.to_datetime(df["Date"])
    close = df["Close"]
    volume = df["Volume"]

    # Price subplot with split regions
    axes[0].plot(dates, close, color="#475569", lw=1.2, label="Full History Price (USD)")
    
    # Highlight split zones
    train_end_date = bundle.train_df["Date"].iloc[-1]
    val_end_date = bundle.val_df["Date"].iloc[-1]
    test_end_date = bundle.test_df["Date"].iloc[-1]

    axes[0].axvspan(dates.iloc[0], train_end_date, alpha=0.15, color="#3b82f6", label=f"Train Split (70%, N={len(bundle.train_df)})")
    axes[0].axvspan(train_end_date, val_end_date, alpha=0.18, color="#f59e0b", label=f"Val Split (15%, N={len(bundle.val_df)})")
    axes[0].axvspan(val_end_date, test_end_date, alpha=0.20, color="#10b981", label=f"Test Split (15%, N={len(bundle.test_df)})")

    axes[0].set_title(f"{title_prefix} - Chronological Splits & Price History", fontsize=14, fontweight="bold")
    axes[0].set_ylabel("Close Price (USD)", fontsize=11)
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(loc="upper left", frameon=True)

    # Volume subplot
    axes[1].bar(dates, volume, color="#64748b", width=1.5, alpha=0.7, label="Trading Volume")
    axes[1].set_ylabel("Volume", fontsize=11)
    axes[1].set_xlabel("Date", fontsize=11)
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend(loc="upper left", frameon=True)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def run_experiment_for_dataset(
    dataset_name: str,
    seq_len: int = 30,
    hidden_size: int = 64,
    num_layers: int = 1,
    epochs: int = 40,
    lr: float = 0.001,
    patience: int = 8,
):
    print(f"\n{'='*70}\nSTARTING EXPERIMENT: {dataset_name.upper()}\n{'='*70}")
    
    # Paths
    models_dir = os.path.join(project_root, "models", "assignment06")
    results_dir = os.path.join(project_root, "results", "assignment06")
    figures_dir = os.path.join(project_root, "figures", "assignment06")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # 1. Load and prepare data
    bundle = prepare_time_series_data(
        dataset_name=dataset_name,
        seq_len=seq_len,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
    )
    meta = get_dataset_metadata(dataset_name)
    display_title = meta["name"]
    print(f"Loaded {display_title} | Total rows: {len(bundle.df_raw)}")
    print(f"X_train: {bundle.X_train.shape} | X_val: {bundle.X_val.shape} | X_test: {bundle.X_test.shape}")

    # Plot EDA
    eda_fig_path = os.path.join(figures_dir, f"{dataset_name}_timeseries_eda.png")
    plot_dataset_eda(bundle, display_title, eda_fig_path)
    print(f"Saved EDA chart to: {eda_fig_path}")

    # Dataloaders for PyTorch
    train_loader, val_loader, test_loader = create_pytorch_dataloaders(bundle, batch_size=32, shuffle_train=True)

    # -------------------------------------------------------------
    # 2. PyTorch Vanilla RNN
    # -------------------------------------------------------------
    print(f"\n--- Training PyTorch Vanilla RNN ({dataset_name}) ---")
    pt_save_path = os.path.join(models_dir, f"{dataset_name}_rnn_pytorch.pt")
    pt_model = VanillaRNNPyTorch(
        input_size=len(bundle.feature_cols),
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=1,
    )
    print(f"PyTorch model params: {pt_model.count_parameters():,}")

    pt_model, pt_history = train_pytorch_rnn(
        model=pt_model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=lr,
        patience=patience,
        save_path=pt_save_path,
        verbose=True,
    )

    _, pt_pred_test = predict_pytorch(pt_model, test_loader, target_scaler=bundle.target_scaler)
    y_test_original = bundle.target_scaler.inverse_transform(bundle.y_test)
    pt_metrics = calculate_regression_metrics(y_test_original, pt_pred_test)
    print(f"PyTorch Test Metrics: {pt_metrics}")

    # -------------------------------------------------------------
    # 3. Keras Vanilla RNN
    # -------------------------------------------------------------
    print(f"\n--- Training Keras Vanilla RNN ({dataset_name}) ---")
    kr_save_path = os.path.join(models_dir, f"{dataset_name}_rnn_keras.keras")
    kr_model = build_keras_rnn(
        input_shape=(seq_len, len(bundle.feature_cols)),
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=1,
        learning_rate=lr,
    )
    print(f"Keras model params: {kr_model.count_params():,}")

    kr_model, kr_history = train_keras_rnn(
        model=kr_model,
        X_train=bundle.X_train,
        y_train=bundle.y_train,
        X_val=bundle.X_val,
        y_val=bundle.y_val,
        epochs=epochs,
        batch_size=32,
        patience=patience,
        save_path=kr_save_path,
        verbose=1,
    )

    _, kr_pred_test = predict_keras(kr_model, bundle.X_test, target_scaler=bundle.target_scaler)
    kr_metrics = calculate_regression_metrics(y_test_original, kr_pred_test)
    print(f"Keras Test Metrics: {kr_metrics}")

    # -------------------------------------------------------------
    # 4. Plots and Metric Artifacts
    # -------------------------------------------------------------
    # Loss curves
    loss_fig_path = os.path.join(figures_dir, f"{dataset_name}_loss_curves.png")
    plot_loss_curves(pt_history, kr_history, display_title, save_path=loss_fig_path)
    plt.close()

    # Actual vs Predicted
    pred_fig_path = os.path.join(figures_dir, f"{dataset_name}_actual_vs_predicted.png")
    plot_actual_vs_predicted(
        dates=bundle.test_dates,
        y_true=y_test_original,
        y_pred_pytorch=pt_pred_test,
        y_pred_keras=kr_pred_test,
        dataset_name=display_title,
        save_path=pred_fig_path,
    )
    plt.close()

    # Residuals
    res_fig_path = os.path.join(figures_dir, f"{dataset_name}_residuals.png")
    plot_residuals(
        dates=bundle.test_dates,
        y_true=y_test_original,
        y_pred_pytorch=pt_pred_test,
        y_pred_keras=kr_pred_test,
        dataset_name=display_title,
        save_path=res_fig_path,
    )
    plt.close()

    # Save metrics JSON
    dataset_summary = {
        "dataset_name": dataset_name,
        "metadata": meta,
        "hyperparameters": {
            "seq_len": seq_len,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "epochs": epochs,
            "learning_rate": lr,
            "batch_size": 32,
            "patience": patience,
        },
        "split_sizes": {
            "total_records": len(bundle.df_raw),
            "train_samples": len(bundle.X_train),
            "val_samples": len(bundle.X_val),
            "test_samples": len(bundle.X_test),
        },
        "pytorch": {
            "metrics": pt_metrics,
            "stopped_epoch": pt_history["stopped_epoch"],
            "best_val_loss": pt_history["best_val_loss"],
            "training_time": pt_history["training_time_seconds"],
            "model_path": os.path.relpath(pt_save_path, project_root),
        },
        "keras": {
            "metrics": kr_metrics,
            "stopped_epoch": kr_history["stopped_epoch"],
            "best_val_loss": kr_history["best_val_loss"],
            "training_time": kr_history["training_time_seconds"],
            "model_path": os.path.relpath(kr_save_path, project_root),
        },
    }

    metrics_json_path = os.path.join(results_dir, f"{dataset_name}_metrics.json")
    with open(metrics_json_path, "w") as f:
        json.dump(dataset_summary, f, indent=2)

    comp_df = build_comparison_dataframe(pt_metrics, kr_metrics, display_title)
    print("\nComparison Summary:")
    print(comp_df.to_string(index=False))

    return dataset_summary, comp_df


def main():
    stock_summary, stock_df = run_experiment_for_dataset("stock")
    commodity_summary, commodity_df = run_experiment_for_dataset("commodity")

    # Combine comparison table
    overall_df = pd.concat([stock_df, commodity_df], ignore_index=True)
    results_dir = os.path.join(project_root, "results", "assignment06")
    csv_path = os.path.join(results_dir, "overall_comparison.csv")
    overall_df.to_csv(csv_path, index=False)
    print("\n" + "="*70)
    print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(overall_df.to_string(index=False))
    print(f"\nSaved combined comparison table to: {csv_path}")


if __name__ == "__main__":
    main()
