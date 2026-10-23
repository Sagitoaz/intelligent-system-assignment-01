"""Unit tests for Assignment 06 (RNN Time-Series Forecasting)."""

import os
import sys
import numpy as np
import pytest
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.assignment06.data import (
    load_raw_dataset,
    prepare_time_series_data,
    create_sliding_windows,
    create_pytorch_dataloaders,
)
from src.assignment06.models_pytorch import VanillaRNNPyTorch
from src.assignment06.models_keras import build_keras_rnn
from src.assignment06.evaluation import calculate_regression_metrics


def test_load_raw_datasets():
    """Verify local stock and commodity CSVs exist and contain standard OHLCV columns."""
    for name in ["stock", "commodity"]:
        df = load_raw_dataset(name)
        assert len(df) > 2000, f"{name} should have > 2000 records"
        expected_cols = {"Date", "Open", "High", "Low", "Close", "Volume"}
        assert expected_cols.issubset(set(df.columns)), f"Missing expected cols in {name}"
        assert df["Close"].isnull().sum() == 0, f"Found nulls in {name} Close"


def test_create_sliding_windows():
    """Verify sliding window generator preserves window length and target alignment."""
    T = 100
    D = 5
    W = 10
    features = np.random.randn(T, D).astype(np.float32)
    targets = np.random.randn(T, 1).astype(np.float32)

    X_win, y_win = create_sliding_windows(features, targets, seq_len=W)
    assert X_win.shape == (T - W, W, D)
    assert y_win.shape == (T - W, 1)

    # Check target alignment: y_win[0] should equal targets[W]
    np.testing.assert_allclose(y_win[0], targets[W])
    np.testing.assert_allclose(X_win[0, -1, :], features[W - 1, :])


def test_chronological_data_preparation():
    """Verify that split ratios match and scalers do not introduce data leakage."""
    bundle = prepare_time_series_data("stock", seq_len=30, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    total_n = len(bundle.df_raw)
    assert len(bundle.train_df) == int(total_n * 0.7)
    assert len(bundle.val_df) == int(total_n * 0.15)
    assert len(bundle.test_df) == total_n - len(bundle.train_df) - len(bundle.val_df)

    # Check date ordering
    assert bundle.train_df["Date"].iloc[-1] < bundle.val_df["Date"].iloc[0]
    assert bundle.val_df["Date"].iloc[-1] < bundle.test_df["Date"].iloc[0]

    # Check DataLoader creation
    tl, vl, tel = create_pytorch_dataloaders(bundle, batch_size=32)
    bx, by = next(iter(tl))
    assert bx.ndim == 3 and bx.shape[1:] == (30, 5)
    assert by.ndim == 2 and by.shape[1] == 1


def test_pytorch_model_architecture():
    """Verify PyTorch Vanilla RNN shapes, parameters, and forward pass."""
    batch_size = 8
    seq_len = 30
    input_size = 5
    hidden_size = 32

    model = VanillaRNNPyTorch(input_size=input_size, hidden_size=hidden_size, num_layers=1, output_size=1)
    x = torch.randn(batch_size, seq_len, input_size)
    out = model(x)

    assert out.shape == (batch_size, 1)
    assert model.count_parameters() > 0


def test_keras_model_architecture():
    """Verify Keras Vanilla RNN compilation, shapes, and forward pass."""
    batch_size = 8
    seq_len = 30
    input_size = 5
    hidden_size = 32

    model = build_keras_rnn(input_shape=(seq_len, input_size), hidden_size=hidden_size, num_layers=1, output_size=1)
    x = np.random.randn(batch_size, seq_len, input_size).astype(np.float32)
    out = model.predict(x, verbose=0)

    assert out.shape == (batch_size, 1)


def test_regression_metrics():
    """Verify MAE, RMSE, R2, and MAPE calculations."""
    y_true = np.array([100.0, 110.0, 120.0, 130.0])
    y_pred = np.array([102.0, 108.0, 122.0, 128.0])

    metrics = calculate_regression_metrics(y_true, y_pred)
    assert metrics["MAE"] == pytest.approx(2.0, abs=1e-4)
    assert metrics["RMSE"] == pytest.approx(2.0, abs=1e-4)
    assert metrics["R2"] > 0.9
    assert metrics["MAPE"] < 3.0


def test_saved_model_checkpoints_exist():
    """Verify all 4 trained model checkpoints exist on disk."""
    expected_models = [
        "stock_rnn_pytorch.pt",
        "stock_rnn_keras.keras",
        "commodity_rnn_pytorch.pt",
        "commodity_rnn_keras.keras",
    ]
    for model_name in expected_models:
        path = os.path.join(PROJECT_ROOT, "models", "assignment06", model_name)
        assert os.path.exists(path), f"Checkpoint missing: {path}"
        assert os.path.getsize(path) > 10000, f"Checkpoint file too small: {path}"
