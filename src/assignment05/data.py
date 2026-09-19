"""Data loading and preprocessing pipelines for Assignment 05.

Supports:
- MNIST (28x28 grayscale, 10 classes)
- Fashion-MNIST (28x28 grayscale, 10 classes)
- Diabetes BRFSS 2015 (21 tabular features, binary classification, Conv1D format)
"""

import os
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, TensorDataset

# Label definitions
MNIST_CLASSES = [str(i) for i in range(10)]
FASHION_MNIST_CLASSES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]
DIABETES_CLASSES = ["No Diabetes", "Prediabetes / Diabetes"]


def get_data_dir() -> str:
    """Return the absolute path to the data directory."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # from src/assignment05 -> project root -> data
    project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
    return os.path.join(project_root, "data")


def load_mnist_data(
    val_size: int = 10000,
    max_train_samples: Optional[int] = None,
    max_test_samples: Optional[int] = None,
    seed: int = 42,
) -> Tuple[
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
]:
    """Load and preprocess MNIST dataset from local compressed archive.

    Returns:
        (x_train, y_train), (x_val, y_val), (x_test, y_test)
        Shapes: x is (N, 1, 28, 28) normalized float32, y is (N,) int64.
    """
    data_dir = get_data_dir()
    mnist_path = os.path.join(data_dir, "mnist", "mnist.npz")
    if not os.path.exists(mnist_path):
        raise FileNotFoundError(f"MNIST file not found at {mnist_path}")

    with np.load(mnist_path) as data:
        x_train_raw = data["x_train"]
        y_train_raw = data["y_train"]
        x_test_raw = data["x_test"]
        y_test_raw = data["y_test"]

    # Stratified validation split from training set
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_raw,
        y_train_raw,
        test_size=val_size,
        random_state=seed,
        stratify=y_train_raw,
    )

    x_test = x_test_raw
    y_test = y_test_raw

    if max_train_samples is not None and max_train_samples < len(x_train):
        x_train, _, y_train, _ = train_test_split(
            x_train,
            y_train,
            train_size=max_train_samples,
            random_state=seed,
            stratify=y_train,
        )
    if max_test_samples is not None and max_test_samples < len(x_test):
        x_test, _, y_test, _ = train_test_split(
            x_test,
            y_test,
            train_size=max_test_samples,
            random_state=seed,
            stratify=y_test,
        )

    # Normalize to [0, 1] then standardize using train mean and std
    x_train = (x_train.astype(np.float32) / 255.0)[:, np.newaxis, :, :]
    x_val = (x_val.astype(np.float32) / 255.0)[:, np.newaxis, :, :]
    x_test = (x_test.astype(np.float32) / 255.0)[:, np.newaxis, :, :]

    train_mean = float(x_train.mean())
    train_std = float(x_train.std())

    x_train = (x_train - train_mean) / (train_std + 1e-7)
    x_val = (x_val - train_mean) / (train_std + 1e-7)
    x_test = (x_test - train_mean) / (train_std + 1e-7)

    y_train = y_train.astype(np.int64)
    y_val = y_val.astype(np.int64)
    y_test = y_test.astype(np.int64)

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def load_fashion_mnist_data(
    val_size: int = 10000,
    max_train_samples: Optional[int] = None,
    max_test_samples: Optional[int] = None,
    seed: int = 42,
) -> Tuple[
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
]:
    """Load and preprocess Fashion-MNIST dataset from local compressed archive.

    Returns:
        (x_train, y_train), (x_val, y_val), (x_test, y_test)
        Shapes: x is (N, 1, 28, 28) normalized float32, y is (N,) int64.
    """
    data_dir = get_data_dir()
    fmnist_path = os.path.join(data_dir, "fashion_mnist", "fashion_mnist.npz")
    if not os.path.exists(fmnist_path):
        raise FileNotFoundError(f"Fashion-MNIST file not found at {fmnist_path}")

    with np.load(fmnist_path) as data:
        x_train_raw = data["x_train"]
        y_train_raw = data["y_train"]
        x_test_raw = data["x_test"]
        y_test_raw = data["y_test"]

    x_train, x_val, y_train, y_val = train_test_split(
        x_train_raw,
        y_train_raw,
        test_size=val_size,
        random_state=seed,
        stratify=y_train_raw,
    )

    x_test = x_test_raw
    y_test = y_test_raw

    if max_train_samples is not None and max_train_samples < len(x_train):
        x_train, _, y_train, _ = train_test_split(
            x_train,
            y_train,
            train_size=max_train_samples,
            random_state=seed,
            stratify=y_train,
        )
    if max_test_samples is not None and max_test_samples < len(x_test):
        x_test, _, y_test, _ = train_test_split(
            x_test,
            y_test,
            train_size=max_test_samples,
            random_state=seed,
            stratify=y_test,
        )

    # Normalize to [0, 1] then standardize using train statistics
    x_train = (x_train.astype(np.float32) / 255.0)[:, np.newaxis, :, :]
    x_val = (x_val.astype(np.float32) / 255.0)[:, np.newaxis, :, :]
    x_test = (x_test.astype(np.float32) / 255.0)[:, np.newaxis, :, :]

    train_mean = float(x_train.mean())
    train_std = float(x_train.std())

    x_train = (x_train - train_mean) / (train_std + 1e-7)
    x_val = (x_val - train_mean) / (train_std + 1e-7)
    x_test = (x_test - train_mean) / (train_std + 1e-7)

    y_train = y_train.astype(np.int64)
    y_val = y_val.astype(np.int64)
    y_test = y_test.astype(np.int64)

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def load_diabetes_data(
    test_size: float = 0.15,
    val_size: float = 0.15,
    max_samples: Optional[int] = 60000,
    seed: int = 42,
) -> Tuple[
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
    list,
]:
    """Load and preprocess BRFSS 2015 Diabetes Tabular dataset.

    Note on Conv1D application:
        This is a pedagogical experiment. Tabular features do NOT possess
        natural spatial locality or shift invariance like pixels in an image.
        Applying Conv1D treats the ordered feature sequence as a 1D signal.

    Returns:
        (x_train, y_train), (x_val, y_val), (x_test, y_test), feature_names
        x shape is (N, 1, 21), y shape is (N,).
    """
    data_dir = get_data_dir()
    csv_path = os.path.join(
        data_dir, "diabetes", "diabetes_binary_health_indicators_BRFSS2015.csv"
    )
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Diabetes dataset not found at {csv_path}")

    df = pd.read_csv(csv_path)

    # Subsample if requested for fast pedagogical iteration while preserving class balance
    if max_samples is not None and max_samples < len(df):
        df, _ = train_test_split(
            df,
            train_size=max_samples,
            random_state=seed,
            stratify=df["Diabetes_binary"],
        )

    target_col = "Diabetes_binary"
    feature_cols = [c for c in df.columns if c != target_col]

    x_raw = df[feature_cols].values.astype(np.float32)
    y_raw = df[target_col].values.astype(np.int64)

    # Train / Val / Test split (stratified)
    x_train_val, x_test, y_train_val, y_test = train_test_split(
        x_raw, y_raw, test_size=test_size, random_state=seed, stratify=y_raw
    )

    val_fraction_of_train_val = val_size / (1.0 - test_size)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val,
        y_train_val,
        test_size=val_fraction_of_train_val,
        random_state=seed,
        stratify=y_train_val,
    )

    # Fit scaler ONLY on train set to prevent data leakage
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)

    # Reshape to (N, Channels=1, Length=21) for Conv1D
    x_train = x_train[:, np.newaxis, :].astype(np.float32)
    x_val = x_val[:, np.newaxis, :].astype(np.float32)
    x_test = x_test[:, np.newaxis, :].astype(np.float32)

    return (x_train, y_train), (x_val, y_val), (x_test, y_test), feature_cols


def create_dataloaders(
    train_data: Tuple[np.ndarray, np.ndarray],
    val_data: Tuple[np.ndarray, np.ndarray],
    test_data: Tuple[np.ndarray, np.ndarray],
    batch_size: int = 128,
    num_workers: int = 0,
) -> Dict[str, DataLoader]:
    """Create PyTorch DataLoaders from numpy arrays."""
    x_train, y_train = train_data
    x_val, y_val = val_data
    x_test, y_test = test_data

    train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))
    test_ds = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return {"train": train_loader, "val": val_loader, "test": test_loader}
