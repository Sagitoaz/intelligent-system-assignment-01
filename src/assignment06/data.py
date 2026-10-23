"""Data loading, time-series splitting, scaling, and windowing for Assignment 06 (RNN).

Provides end-to-end reproducible pipelines for:
- Stock Price Time Series (AAPL)
- Commodity Price Time Series (Gold - GC=F)
"""

from dataclasses import dataclass
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import DataLoader, TensorDataset


def get_project_root() -> str:
    """Return the absolute path to the project root directory."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current_dir, "..", ".."))


def get_data_path(dataset_name: str) -> str:
    """Get the CSV path for a given dataset name ('stock' or 'commodity')."""
    root = get_project_root()
    if dataset_name.lower() in ["stock", "aapl", "stock_aapl"]:
        return os.path.join(root, "data", "assignment06", "stock_aapl.csv")
    elif dataset_name.lower() in ["commodity", "gold", "commodity_gold"]:
        return os.path.join(root, "data", "assignment06", "commodity_gold.csv")
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}. Expected 'stock' or 'commodity'.")


def load_raw_dataset(dataset_name: str, csv_path: Optional[str] = None) -> pd.DataFrame:
    """Load raw time-series CSV file, ensure Date datetime index, and sort chronologically.

    Args:
        dataset_name: 'stock' or 'commodity'
        csv_path: Optional custom path to CSV file

    Returns:
        pd.DataFrame sorted by Date
    """
    path = csv_path if csv_path is not None else get_data_path(dataset_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found at {path}")

    df = pd.read_csv(path)

    # Ensure Date column is parsed and sorted
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
    else:
        raise KeyError(f"Expected 'Date' column in {path}")

    # Standardize numeric columns
    numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop any nulls or interpolate
    df = df.dropna().reset_index(drop=True)
    return df


def get_dataset_metadata(dataset_name: str) -> Dict[str, str]:
    """Return descriptive metadata about the chosen dataset."""
    if dataset_name.lower() in ["stock", "aapl", "stock_aapl"]:
        return {
            "name": "Apple Inc. (AAPL) Daily Stock Price",
            "source": "Yahoo Finance (via yfinance API / local archive)",
            "asset_class": "Equities (Tech Sector)",
            "primary_target": "Close (Next-day closing price in USD)",
            "features": "Open, High, Low, Close, Volume",
            "time_horizon": "2014-01-02 to 2023-12-29 (10 years daily trading)",
            "description": (
                "Historical daily stock prices of Apple Inc. (AAPL), representing "
                "one of the most liquid and actively traded technology assets in the world."
            ),
        }
    elif dataset_name.lower() in ["commodity", "gold", "commodity_gold"]:
        return {
            "name": "Gold Futures (GC=F) Daily Price",
            "source": "Yahoo Finance / COMEX (via yfinance API / local archive)",
            "asset_class": "Commodities (Precious Metals)",
            "primary_target": "Close (Next-day settlement price in USD/troy oz)",
            "features": "Open, High, Low, Close, Volume",
            "time_horizon": "2014-01-02 to 2023-12-29 (10 years daily settlement)",
            "description": (
                "Continuous front-month COMEX Gold Futures prices, serving as the benchmark "
                "safe-haven commodity tracking global macroeconomic and inflationary shifts."
            ),
        }
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")


@dataclass
class TimeSeriesDataBundle:
    """Container holding all preprocessed data, scalers, windows, and splits."""

    dataset_name: str
    feature_cols: List[str]
    target_col: str
    seq_len: int
    # Dataframes
    df_raw: pd.DataFrame
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    # Scalers (fit strictly on train)
    feature_scaler: MinMaxScaler
    target_scaler: MinMaxScaler
    # Sliding window numpy arrays: X shape (N, seq_len, D), y shape (N, 1)
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    # Target dates aligned with y
    train_dates: pd.Series
    val_dates: pd.Series
    test_dates: pd.Series


def create_sliding_windows(
    features: np.ndarray,
    targets: np.ndarray,
    seq_len: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Transform sequential feature and target arrays into sliding window sequences.

    For timestep t in range(seq_len, len(features)):
        X[i] = features[t - seq_len : t]  -> shape (seq_len, num_features)
        y[i] = targets[t]                  -> shape (1,)

    Args:
        features: 2D array of shape (T, num_features)
        targets: 2D array of shape (T, 1)
        seq_len: Window length (e.g. 30 days)

    Returns:
        X: 3D array of shape (N, seq_len, num_features)
        y: 2D array of shape (N, 1)
    """
    total_len = len(features)
    if total_len <= seq_len:
        raise ValueError(f"Total length ({total_len}) must be greater than seq_len ({seq_len})")

    X_list = []
    y_list = []
    for t in range(seq_len, total_len):
        X_list.append(features[t - seq_len : t])
        y_list.append(targets[t])

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


def prepare_time_series_data(
    dataset_name: str,
    feature_cols: Optional[List[str]] = None,
    target_col: str = "Close",
    seq_len: int = 30,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    csv_path: Optional[str] = None,
) -> TimeSeriesDataBundle:
    """Load, split chronologically, normalize (fitted ONLY on train), and generate sliding windows.

    Guarantees zero future data leakage:
    1. Chronological split: df[:n_train], df[n_train:n_train+n_val], df[n_train+n_val:]
    2. MinMaxScaler fit on train_df ONLY, transform val_df and test_df.
    3. To ensure test set evaluation starts exactly from the beginning of test_df without
       skipping the initial `seq_len` timesteps, the test window sequence uses the preceding
       `seq_len` rows from val_df as context. Similarly for val_df using train_df.

    Args:
        dataset_name: 'stock' or 'commodity'
        feature_cols: List of column names to use as features. Defaults to Open, High, Low, Close, Volume.
        target_col: Column to predict. Defaults to 'Close'.
        seq_len: Sliding window sequence length (default 30 days).
        train_ratio: Fraction for training (default 0.70).
        val_ratio: Fraction for validation (default 0.15).
        test_ratio: Fraction for testing (default 0.15).
        csv_path: Optional custom path.

    Returns:
        TimeSeriesDataBundle with all arrays, dates, scalers, and dataframes.
    """
    if feature_cols is None:
        feature_cols = ["Open", "High", "Low", "Close", "Volume"]

    df = load_raw_dataset(dataset_name, csv_path=csv_path)
    total_n = len(df)

    # 1. Chronological split
    n_train = int(total_n * train_ratio)
    n_val = int(total_n * val_ratio)
    n_test = total_n - n_train - n_val

    train_df = df.iloc[:n_train].copy().reset_index(drop=True)
    val_df = df.iloc[n_train : n_train + n_val].copy().reset_index(drop=True)
    test_df = df.iloc[n_train + n_val :].copy().reset_index(drop=True)

    # 2. Normalization: Fit ONLY on training split
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler = MinMaxScaler(feature_range=(0, 1))

    feature_scaler.fit(train_df[feature_cols].values)
    target_scaler.fit(train_df[[target_col]].values)

    # Transform splits
    train_feat_scaled = feature_scaler.transform(train_df[feature_cols].values)
    train_tgt_scaled = target_scaler.transform(train_df[[target_col]].values)

    val_feat_scaled = feature_scaler.transform(val_df[feature_cols].values)
    val_tgt_scaled = target_scaler.transform(val_df[[target_col]].values)

    test_feat_scaled = feature_scaler.transform(test_df[feature_cols].values)
    test_tgt_scaled = target_scaler.transform(test_df[[target_col]].values)

    # 3. Create sliding windows
    # For training: windows strictly within train
    X_train, y_train = create_sliding_windows(train_feat_scaled, train_tgt_scaled, seq_len)
    train_dates = train_df["Date"].iloc[seq_len:].reset_index(drop=True)

    # For validation: prepend last seq_len items of train to avoid discarding the first seq_len val targets
    val_feat_full = np.vstack([train_feat_scaled[-seq_len:], val_feat_scaled])
    val_tgt_full = np.vstack([train_tgt_scaled[-seq_len:], val_tgt_scaled])
    X_val, y_val = create_sliding_windows(val_feat_full, val_tgt_full, seq_len)
    val_dates = val_df["Date"].reset_index(drop=True)

    # For testing: prepend last seq_len items of val
    test_feat_full = np.vstack([val_feat_scaled[-seq_len:], test_feat_scaled])
    test_tgt_full = np.vstack([val_tgt_scaled[-seq_len:], test_tgt_scaled])
    X_test, y_test = create_sliding_windows(test_feat_full, test_tgt_full, seq_len)
    test_dates = test_df["Date"].reset_index(drop=True)

    return TimeSeriesDataBundle(
        dataset_name=dataset_name,
        feature_cols=feature_cols,
        target_col=target_col,
        seq_len=seq_len,
        df_raw=df,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        train_dates=train_dates,
        val_dates=val_dates,
        test_dates=test_dates,
    )


def create_pytorch_dataloaders(
    bundle: TimeSeriesDataBundle,
    batch_size: int = 32,
    shuffle_train: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create PyTorch DataLoaders for train, val, and test splits.

    Args:
        bundle: TimeSeriesDataBundle containing numpy arrays
        batch_size: Mini-batch size (default 32)
        shuffle_train: Whether to shuffle training mini-batches (default True)

    Returns:
        (train_loader, val_loader, test_loader)
    """
    train_ds = TensorDataset(
        torch.from_numpy(bundle.X_train),
        torch.from_numpy(bundle.y_train),
    )
    val_ds = TensorDataset(
        torch.from_numpy(bundle.X_val),
        torch.from_numpy(bundle.y_val),
    )
    test_ds = TensorDataset(
        torch.from_numpy(bundle.X_test),
        torch.from_numpy(bundle.y_test),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle_train)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
