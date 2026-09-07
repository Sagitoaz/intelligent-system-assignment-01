"""Standalone PyTorch source for USA house-price regression.

Dataset: ``data/house_price/realtor-data.zip.csv``; modeling sample: 400,000.
Final executed architecture: ``61 -> 64 -> 1`` on standardized log1p(price).
The default CLI only checks saved artifacts and never scans or trains the dataset.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .common import (
    MLP,
    count_trainable_parameters,
    find_repo_root,
    instantiate_checkpoint_model,
    load_checkpoint,
    resolve_device,
    set_seed,
)

RANDOM_STATE = 42
DATA_RELATIVE_PATH = Path("data/house_price/realtor-data.zip.csv")
CHECKPOINT_RELATIVE_PATH = Path("models/assignment03/house_price/final_mlp.pt")
PREPROCESSOR_RELATIVE_PATH = Path("models/assignment03/house_price/preprocessor.joblib")

EXPECTED_RAW_ROWS = 2_226_382
MODEL_SAMPLE_SIZE = 400_000
CHUNK_SIZE = 250_000
TARGET = "price"
NUMERIC_FEATURES = ["bed", "bath", "acre_lot", "house_size"]
CATEGORICAL_FEATURES = ["state", "status"]
MODEL_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]
PROFILE_COLUMNS = MODEL_COLUMNS + ["street", "city", "zip_code", "brokered_by"]

INPUT_DIM = 61
HIDDEN_DIMS = [64]
OUTPUT_DIM = 1
USE_LOG_TARGET = True
LEARNING_RATE = 1e-3
BATCH_SIZE = 512
MAX_EPOCHS = 30
PATIENCE = 6
BEST_EPOCH_REFERENCE = 26
SELECTION_CRITERION = "validation RMSE, rồi validation MAE"
EXPECTED_PARAMETERS = 4_033
CLIP_RATE_WARNING_THRESHOLD = 0.01
EXPERIMENT_CONFIGS = [
    {"name": "Baseline MLP", "hidden_dims": [64], "learning_rate": 1e-3,
     "epochs": MAX_EPOCHS, "patience": PATIENCE},
    {"name": "Deeper MLP / Adam 1e-3", "hidden_dims": [64, 32],
     "learning_rate": 1e-3, "epochs": MAX_EPOCHS, "patience": PATIENCE},
    {"name": "Deeper MLP / Adam 3e-4", "hidden_dims": [64, 32],
     "learning_rate": 3e-4, "epochs": MAX_EPOCHS, "patience": PATIENCE},
]


def select_final_experiment(results: pd.DataFrame) -> str:
    """Select by ascending validation RMSE, then ascending validation MAE."""
    return str(results.sort_values(["rmse", "mae"]).iloc[0]["experiment"])


def build_preprocessor() -> ColumnTransformer:
    """Build the train-fitted numeric and categorical preprocessing pipeline."""
    return ColumnTransformer(
        [
            (
                "numeric",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    (
                        "onehot",
                        OneHotEncoder(
                            handle_unknown="ignore", sparse_output=False,
                            dtype=np.float32,
                        ),
                    ),
                ]),
                CATEGORICAL_FEATURES,
            ),
        ],
        verbose_feature_names_out=False,
    )


def load_modeling_sample(repo_root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Reproduce the notebook's chunked audit and 400,000-row sample.

    This expensive function is available for explicit training workflows only;
    the default check CLI never calls it.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    sampling_fraction = min(1.0, 1.35 * MODEL_SAMPLE_SIZE / EXPECTED_RAW_ROWS)
    raw_rows = 0
    missing_counts = pd.Series(0, index=PROFILE_COLUMNS, dtype="int64")
    invalid_counts = {
        "price_le_0": 0, "house_size_le_0": 0, "bed_lt_0": 0,
        "bath_lt_0": 0, "acre_lot_lt_0": 0,
    }
    candidate_parts: list[pd.DataFrame] = []
    reader = pd.read_csv(
        repo_root / DATA_RELATIVE_PATH,
        usecols=PROFILE_COLUMNS,
        chunksize=CHUNK_SIZE,
    )
    for chunk in reader:
        raw_rows += len(chunk)
        missing_counts = missing_counts.add(chunk.isna().sum(), fill_value=0)
        invalid_counts["price_le_0"] += int((chunk["price"] <= 0).sum())
        invalid_counts["house_size_le_0"] += int((chunk["house_size"] <= 0).sum())
        invalid_counts["bed_lt_0"] += int((chunk["bed"] < 0).sum())
        invalid_counts["bath_lt_0"] += int((chunk["bath"] < 0).sum())
        invalid_counts["acre_lot_lt_0"] += int((chunk["acre_lot"] < 0).sum())
        keep = rng.random(len(chunk)) < sampling_fraction
        candidate_parts.append(chunk.loc[keep, MODEL_COLUMNS])

    candidates = pd.concat(candidate_parts, ignore_index=True)
    valid = candidates["price"].notna() & candidates["price"].gt(0)
    valid &= candidates["house_size"].isna() | candidates["house_size"].gt(0)
    for feature in ["bed", "bath", "acre_lot"]:
        valid &= candidates[feature].isna() | candidates[feature].ge(0)
    candidates = candidates.loc[valid]
    model_df = candidates.sample(
        n=min(MODEL_SAMPLE_SIZE, len(candidates)), random_state=RANDOM_STATE
    ).reset_index(drop=True)
    return model_df, {
        "raw_rows": raw_rows,
        "missing_counts": missing_counts,
        "invalid_counts": invalid_counts,
    }


def split_and_prepare(model_df: pd.DataFrame) -> dict[str, Any]:
    """Split 70/15/15, then fit preprocessing and target statistics on train only."""
    x_raw = model_df.drop(columns=TARGET)
    y_dollar = model_df[TARGET].astype("float64")
    x_train_raw, x_temp_raw, y_train_dollar, y_temp_dollar = train_test_split(
        x_raw, y_dollar, test_size=0.30, random_state=RANDOM_STATE
    )
    x_val_raw, x_test_raw, y_val_dollar, y_test_dollar = train_test_split(
        x_temp_raw, y_temp_dollar, test_size=0.50, random_state=RANDOM_STATE
    )
    y_train_model = transform_target(y_train_dollar)
    y_val_model = transform_target(y_val_dollar)
    y_test_model = transform_target(y_test_dollar)
    preprocessor = build_preprocessor()
    x_train = preprocessor.fit_transform(x_train_raw).astype("float32")
    x_val = preprocessor.transform(x_val_raw).astype("float32")
    x_test = preprocessor.transform(x_test_raw).astype("float32")
    target_mean = float(np.mean(y_train_model))
    target_std = float(np.std(y_train_model) + 1e-8)
    return {
        "preprocessor": preprocessor,
        "x_train": x_train, "x_val": x_val, "x_test": x_test,
        "y_train_dollar": y_train_dollar,
        "y_val_dollar": y_val_dollar,
        "y_test_dollar": y_test_dollar,
        "y_train_model": y_train_model,
        "y_val_model": y_val_model,
        "y_test_model": y_test_model,
        "target_mean": target_mean,
        "target_std": target_std,
    }


def transform_target(values: Any) -> np.ndarray:
    """Apply the executed model-scale target transformation."""
    array = np.asarray(values)
    return np.log1p(array) if USE_LOG_TARGET else array


def inverse_target(values: Any) -> np.ndarray:
    """Invert model-scale values to dollars before metric calculation."""
    array = np.asarray(values)
    return np.expm1(array) if USE_LOG_TARGET else array


def standardize_y(values: Any, target_mean: float, target_std: float) -> np.ndarray:
    return (
        (np.asarray(values) - target_mean) / target_std
    ).astype("float32").reshape(-1, 1)


def unstandardize_prediction(
    values: Any, target_mean: float, target_std: float
) -> np.ndarray:
    return np.asarray(values).reshape(-1) * target_std + target_mean


def make_regression_loader(
    x_array: np.ndarray,
    y_model: np.ndarray,
    target_mean: float,
    target_std: float,
    shuffle: bool,
) -> DataLoader:
    """Create float tensors shaped ``[B,61]`` and ``[B,1]``."""
    dataset = TensorDataset(
        torch.from_numpy(np.asarray(x_array, dtype=np.float32)),
        torch.from_numpy(standardize_y(y_model, target_mean, target_std)),
    )
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=0)


def _regression_metric_values(
    y_true_dollar: Any, prediction_dollar: Any, prefix: str = ""
) -> dict[str, float]:
    mse = mean_squared_error(y_true_dollar, prediction_dollar)
    return {
        f"{prefix}mae": mean_absolute_error(y_true_dollar, prediction_dollar),
        f"{prefix}mse": mse,
        f"{prefix}rmse": np.sqrt(mse),
        f"{prefix}r2": r2_score(y_true_dollar, prediction_dollar),
    }


def inverse_predictions_with_audit(
    prediction_model_scale: Any,
    y_train_model: Any,
    y_true_dollar: Any | None = None,
) -> tuple[np.ndarray | None, np.ndarray, dict[str, Any]]:
    """Return raw/safeguarded USD predictions and the notebook clipping audit.

    Bounds are always derived from the supplied training target on model scale;
    validation or test targets never define them.
    """
    raw_values = np.asarray(prediction_model_scale, dtype=np.float64).reshape(-1)
    train_values = np.asarray(y_train_model, dtype=np.float64).reshape(-1)
    lower, upper = float(np.min(train_values)), float(np.max(train_values))
    finite_mask = np.isfinite(raw_values)
    n_below = int(np.sum(finite_mask & (raw_values < lower)))
    n_above = int(np.sum(finite_mask & (raw_values > upper)))
    n_nonfinite = int(np.sum(~finite_mask))
    n_clipped = n_below + n_above + n_nonfinite
    clip_rate = n_clipped / len(raw_values) if len(raw_values) else 0.0

    raw_prediction_dollar: np.ndarray | None = None
    raw_metrics_valid = False
    if len(raw_values) and n_nonfinite == 0:
        metric_safe_limit = np.sqrt(np.finfo(np.float64).max / len(raw_values)) * 0.25
        inverse_safe = (not USE_LOG_TARGET) or bool(
            np.all(raw_values <= np.log1p(metric_safe_limit))
        )
        if inverse_safe:
            with np.errstate(over="ignore", invalid="ignore"):
                candidate = np.asarray(inverse_target(raw_values), dtype=np.float64)
            residual_safe = y_true_dollar is None or bool(
                np.all(np.isfinite(candidate - np.asarray(y_true_dollar, dtype=np.float64)))
                and np.all(
                    np.abs(candidate - np.asarray(y_true_dollar, dtype=np.float64))
                    <= metric_safe_limit
                )
            )
            if np.all(np.isfinite(candidate)) and residual_safe:
                raw_prediction_dollar = candidate
                raw_metrics_valid = True

    safeguarded_values = np.nan_to_num(
        raw_values,
        nan=float(np.mean(train_values)),
        posinf=upper,
        neginf=lower,
    )
    safeguarded_values = np.clip(safeguarded_values, lower, upper)
    safeguarded_prediction_dollar = np.maximum(
        0.0, np.asarray(inverse_target(safeguarded_values), dtype=np.float64)
    )
    audit = {
        "raw_metrics_valid": raw_metrics_valid,
        "n_below": n_below,
        "n_above": n_above,
        "n_nonfinite": n_nonfinite,
        "n_clipped": n_clipped,
        "clip_rate": clip_rate,
        "clip_lower_train_scale": lower,
        "clip_upper_train_scale": upper,
    }
    return raw_prediction_dollar, safeguarded_prediction_dollar, audit


def regression_metrics(
    y_true_dollar: Any, prediction_model_scale: Any, y_train_model: Any
) -> dict[str, Any]:
    """Compute USD raw/safeguarded metrics plus clipping statistics."""
    raw_prediction, safeguarded_prediction, audit = inverse_predictions_with_audit(
        prediction_model_scale, y_train_model, y_true_dollar=y_true_dollar
    )
    result: dict[str, Any] = _regression_metric_values(
        y_true_dollar, safeguarded_prediction
    )
    if raw_prediction is None:
        result.update({
            "raw_mae": np.nan, "raw_mse": np.nan,
            "raw_rmse": np.nan, "raw_r2": np.nan,
        })
    else:
        result.update(_regression_metric_values(
            y_true_dollar, raw_prediction, prefix="raw_"
        ))
    result.update(audit)
    return result


def clipping_requires_warning(result: dict[str, Any]) -> bool:
    """Match the notebook's strict ``clip_rate > 1%`` warning condition."""
    return float(result["clip_rate"]) > CLIP_RATE_WARNING_THRESHOLD


def evaluate_regressor(
    model: MLP,
    loader: DataLoader,
    y_true_dollar: Any,
    y_train_model: Any,
    target_mean: float,
    target_std: float,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    """Evaluate standardized MSE loss and dollar-scale regression metrics."""
    model.eval()
    losses: list[float] = []
    predictions: list[float] = []
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            output = model(x_batch)
            losses.append(criterion(output, y_batch).item() * len(y_batch))
            predictions.extend(output.cpu().numpy().reshape(-1))
    model_scale = unstandardize_prediction(predictions, target_mean, target_std)
    return {
        "loss": sum(losses) / len(predictions),
        **regression_metrics(y_true_dollar, model_scale, y_train_model),
    }


def train_regressor(
    train_loader: DataLoader,
    val_loader: DataLoader,
    y_train_model: Any,
    y_val_dollar: Any,
    target_mean: float,
    target_std: float,
    device: torch.device,
    *,
    hidden_dims: list[int] = HIDDEN_DIMS,
    learning_rate: float = LEARNING_RATE,
    epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
) -> tuple[MLP, pd.DataFrame, dict[str, Any]]:
    """Train with standardized-target MSE and validation-loss early stopping."""
    model = MLP(INPUT_DIM, hidden_dims, OUTPUT_DIM).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_state = copy.deepcopy(model.state_dict())
    best_loss, best_epoch, wait = float("inf"), 0, 0
    history: list[dict[str, float]] = []
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        total, seen = 0.0, 0
        train_predictions: list[float] = []
        train_true_model: list[float] = []
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(x_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(y_batch)
            seen += len(y_batch)
            train_predictions.extend(output.detach().cpu().numpy().reshape(-1))
            train_true_model.extend(unstandardize_prediction(
                y_batch.detach().cpu().numpy(), target_mean, target_std
            ))

        validation = evaluate_regressor(
            model, val_loader, y_val_dollar, y_train_model,
            target_mean, target_std, criterion, device,
        )
        train_model_predictions = unstandardize_prediction(
            train_predictions, target_mean, target_std
        )
        train_rmse = regression_metrics(
            inverse_target(np.asarray(train_true_model)),
            train_model_predictions,
            y_train_model,
        )["rmse"]
        history.append({
            "epoch": epoch, "train_loss": total / seen,
            "val_loss": validation["loss"], "train_rmse": train_rmse,
            "val_rmse": validation["rmse"],
        })
        if validation["loss"] < best_loss:
            best_state = copy.deepcopy(model.state_dict())
            best_loss, best_epoch, wait = validation["loss"], epoch, 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    elapsed = time.perf_counter() - started
    final_validation = evaluate_regressor(
        model, val_loader, y_val_dollar, y_train_model,
        target_mean, target_std, criterion, device,
    )
    return model, pd.DataFrame(history), {
        "best_epoch": best_epoch,
        "training_seconds": elapsed,
        **final_validation,
    }


def load_final_bundle(
    repo_root: Path | None = None, device: torch.device | None = None
) -> tuple[MLP, ColumnTransformer, dict[str, Any]]:
    """Load and validate the existing model and train-fitted preprocessor."""
    root = repo_root or find_repo_root(DATA_RELATIVE_PATH)
    selected_device = device or resolve_device()
    checkpoint = load_checkpoint(root / CHECKPOINT_RELATIVE_PATH, selected_device)
    expected_metadata = {
        "input_dim": INPUT_DIM,
        "hidden_dims": HIDDEN_DIMS,
        "output_dim": OUTPUT_DIM,
        "use_log_target": USE_LOG_TARGET,
        "experiment": "Baseline MLP",
    }
    for key, expected in expected_metadata.items():
        if checkpoint.get(key) != expected:
            raise ValueError(f"House checkpoint {key}: {checkpoint.get(key)!r} != {expected!r}")
    if not np.isfinite(float(checkpoint["target_mean"])) or not np.isfinite(
        float(checkpoint["target_std"])
    ):
        raise ValueError("House checkpoint target statistics are not finite")
    model = instantiate_checkpoint_model(checkpoint, selected_device)
    preprocessor = joblib.load(root / PREPROCESSOR_RELATIVE_PATH)
    if len(preprocessor.get_feature_names_out()) != INPUT_DIM:
        raise ValueError("House preprocessor output dimension không bằng 61")
    if count_trainable_parameters(model) != EXPECTED_PARAMETERS:
        raise ValueError("House parameter count không bằng 4,033")
    return model, preprocessor, checkpoint


def check_saved_artifacts() -> dict[str, Any]:
    """Run lightweight checkpoint, preprocessor, and synthetic-forward checks."""
    set_seed()
    device = resolve_device()
    model, preprocessor, checkpoint = load_final_bundle(device=device)
    with torch.no_grad():
        output = model(torch.zeros((2, INPUT_DIM), dtype=torch.float32, device=device))
    if tuple(output.shape) != (2, OUTPUT_DIM):
        raise ValueError(f"Unexpected House output shape: {tuple(output.shape)}")
    return {
        "task": "house_price",
        "architecture": f"{INPUT_DIM}→64→{OUTPUT_DIM}",
        "parameters": count_trainable_parameters(model),
        "synthetic_output_shape": list(output.shape),
        "preprocessor_output_dim": len(preprocessor.get_feature_names_out()),
        "checkpoint_experiment": checkpoint["experiment"],
        "target_transform": "log1p" if checkpoint["use_log_target"] else "identity",
        "training_reference": {
            "criterion": "MSELoss on standardized model target",
            "optimizer": "Adam", "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE, "best_epoch": BEST_EPOCH_REFERENCE,
            "selection": SELECTION_CRITERION,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Validate saved artifacts only (default).")
    parser.parse_args()
    print(json.dumps(check_saved_artifacts(), ensure_ascii=True))


if __name__ == "__main__":
    main()
