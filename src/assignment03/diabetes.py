"""Standalone PyTorch source for BRFSS diabetes classification.

Dataset: ``data/diabetes/diabetes_binary_health_indicators_BRFSS2015.csv``.
Final executed architecture: ``21 -> 64 -> 32 -> 2``.
The default CLI only checks saved artifacts; it never starts training.
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
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .common import (
    MLP,
    classification_metrics,
    count_trainable_parameters,
    find_repo_root,
    instantiate_checkpoint_model,
    load_checkpoint,
    resolve_device,
    set_seed,
)

RANDOM_STATE = 42
DATA_RELATIVE_PATH = Path("data/diabetes/diabetes_binary_health_indicators_BRFSS2015.csv")
CHECKPOINT_RELATIVE_PATH = Path("models/assignment03/diabetes/final_mlp.pt")
PREPROCESSOR_RELATIVE_PATH = Path("models/assignment03/diabetes/preprocessor.joblib")

TARGET = "Diabetes_binary"
EXPECTED_RAW_ROWS = 253_680
EXPECTED_RAW_COLUMNS = 22
SCALED_FEATURES = ["BMI", "GenHlth", "MentHlth", "PhysHlth", "Age", "Education", "Income"]
BINARY_FEATURES = [
    "HighBP", "HighChol", "CholCheck", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "DiffWalk", "Sex",
]
FEATURE_COLUMNS = SCALED_FEATURES + BINARY_FEATURES

INPUT_DIM = 21
HIDDEN_DIMS = [64, 32]
OUTPUT_DIM = 2
LEARNING_RATE = 1e-3
BATCH_SIZE = 512
MAX_EPOCHS = 25
PATIENCE = 5
BEST_EPOCH_REFERENCE = 15
SELECTION_CRITERION = "validation F1, rồi validation loss"
EXPECTED_PARAMETERS = 3_554
EXPERIMENT_CONFIGS = [
    {"name": "Baseline MLP", "hidden_dims": [64], "learning_rate": 1e-3,
     "epochs": MAX_EPOCHS, "patience": PATIENCE},
    {"name": "Deeper MLP / Adam 1e-3", "hidden_dims": [64, 32],
     "learning_rate": 1e-3, "epochs": MAX_EPOCHS, "patience": PATIENCE},
    {"name": "Deeper MLP / Adam 3e-4", "hidden_dims": [64, 32],
     "learning_rate": 3e-4, "epochs": MAX_EPOCHS, "patience": PATIENCE},
]


def select_final_experiment(results: pd.DataFrame) -> str:
    """Select by descending validation F1, then ascending validation loss."""
    return str(results.sort_values(["f1", "loss"], ascending=[False, True]).iloc[0]["experiment"])


def build_preprocessor() -> ColumnTransformer:
    """Build the train-fitted scaler plus binary passthrough from the notebook."""
    return ColumnTransformer(
        [
            ("scaled", StandardScaler(), SCALED_FEATURES),
            ("binary", "passthrough", BINARY_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def load_and_prepare_data(repo_root: Path) -> dict[str, Any]:
    """Load the full 253,680-row dataset, split first, then fit preprocessing on train.

    This mirrors the executed pipeline and is not called by the safe check CLI.
    """
    raw_df = pd.read_csv(repo_root / DATA_RELATIVE_PATH)
    if raw_df.shape != (EXPECTED_RAW_ROWS, EXPECTED_RAW_COLUMNS):
        raise ValueError(f"Unexpected Diabetes raw shape: {raw_df.shape}")
    x_raw = raw_df[FEATURE_COLUMNS]
    y = raw_df[TARGET].astype("int64")
    x_train_raw, x_temp_raw, y_train, y_temp = train_test_split(
        x_raw, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
    )
    x_val_raw, x_test_raw, y_val, y_test = train_test_split(
        x_temp_raw, y_temp, test_size=0.50, stratify=y_temp,
        random_state=RANDOM_STATE,
    )
    preprocessor = build_preprocessor()
    x_train = preprocessor.fit_transform(x_train_raw).astype("float32")
    x_val = preprocessor.transform(x_val_raw).astype("float32")
    x_test = preprocessor.transform(x_test_raw).astype("float32")
    return {
        "raw_df": raw_df,
        "preprocessor": preprocessor,
        "x_train": x_train,
        "x_val": x_val,
        "x_test": x_test,
        "y_train": y_train.to_numpy(),
        "y_val": y_val.to_numpy(),
        "y_test": y_test.to_numpy(),
    }


def make_dense_loader(
    x_array: np.ndarray, y_array: np.ndarray, shuffle: bool
) -> DataLoader:
    """Create ``[B,21]`` float inputs and ``[B]`` long targets."""
    dataset = TensorDataset(
        torch.from_numpy(np.asarray(x_array, dtype=np.float32)),
        torch.from_numpy(np.asarray(y_array, dtype=np.int64)),
    )
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=0)


def class_weights_from_training_labels(y_train: np.ndarray) -> torch.Tensor:
    """Compute the balanced two-class weights used by CrossEntropyLoss."""
    counts = np.bincount(np.asarray(y_train, dtype=np.int64), minlength=2)
    return torch.tensor(len(y_train) / (2 * counts), dtype=torch.float32)


def evaluate_classifier(
    model: MLP, loader: DataLoader, criterion: nn.Module, device: torch.device
) -> dict[str, float]:
    """Evaluate loss, Accuracy, Precision, Recall, F1, and ROC-AUC."""
    model.eval()
    losses: list[float] = []
    labels: list[int] = []
    predictions: list[int] = []
    scores: list[float] = []
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            logits = model(x_batch)
            losses.append(criterion(logits, y_batch).item() * len(y_batch))
            labels.extend(y_batch.cpu().numpy())
            predictions.extend(logits.argmax(1).cpu().numpy())
            scores.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
    result = classification_metrics(
        np.asarray(labels), np.asarray(predictions), np.asarray(scores)
    )
    result["loss"] = sum(losses) / len(labels)
    return result


def train_classifier(
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_weights: torch.Tensor,
    device: torch.device,
    *,
    hidden_dims: list[int] = HIDDEN_DIMS,
    learning_rate: float = LEARNING_RATE,
    epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
) -> tuple[MLP, pd.DataFrame, dict[str, float]]:
    """Train with Adam and validation-loss early stopping, as in the notebook."""
    model = MLP(INPUT_DIM, hidden_dims, OUTPUT_DIM).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_state = copy.deepcopy(model.state_dict())
    best_loss, best_epoch, wait = float("inf"), 0, 0
    history: list[dict[str, float]] = []
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, seen = 0.0, 0
        train_labels: list[int] = []
        train_predictions: list[int] = []
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(y_batch)
            seen += len(y_batch)
            train_labels.extend(y_batch.detach().cpu().numpy())
            train_predictions.extend(logits.argmax(1).detach().cpu().numpy())

        validation = evaluate_classifier(model, val_loader, criterion, device)
        train_f1 = f1_score(train_labels, train_predictions)
        history.append({
            "epoch": epoch,
            "train_loss": running_loss / seen,
            "train_f1": train_f1,
            "val_loss": validation["loss"],
            "val_f1": validation["f1"],
            "val_accuracy": validation["accuracy"],
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
    final_validation = evaluate_classifier(model, val_loader, criterion, device)
    return model, pd.DataFrame(history), {
        "best_epoch": best_epoch,
        "training_seconds": elapsed,
        **final_validation,
    }


def load_final_bundle(
    repo_root: Path | None = None, device: torch.device | None = None
) -> tuple[MLP, ColumnTransformer, dict[str, Any]]:
    """Load and validate the existing final model and preprocessing artifact."""
    root = repo_root or find_repo_root(DATA_RELATIVE_PATH)
    selected_device = device or resolve_device()
    checkpoint = load_checkpoint(root / CHECKPOINT_RELATIVE_PATH, selected_device)
    expected_metadata = {
        "input_dim": INPUT_DIM,
        "hidden_dims": HIDDEN_DIMS,
        "output_dim": OUTPUT_DIM,
        "selected_by": "validation_f1",
        "experiment": "Deeper MLP / Adam 1e-3",
    }
    for key, expected in expected_metadata.items():
        if checkpoint.get(key) != expected:
            raise ValueError(f"Diabetes checkpoint {key}: {checkpoint.get(key)!r} != {expected!r}")
    model = instantiate_checkpoint_model(checkpoint, selected_device)
    preprocessor = joblib.load(root / PREPROCESSOR_RELATIVE_PATH)
    if len(preprocessor.get_feature_names_out()) != INPUT_DIM:
        raise ValueError("Diabetes preprocessor output dimension không bằng 21")
    if count_trainable_parameters(model) != EXPECTED_PARAMETERS:
        raise ValueError("Diabetes parameter count không bằng 3,554")
    return model, preprocessor, checkpoint


def check_saved_artifacts() -> dict[str, Any]:
    """Run a lightweight checkpoint, preprocessor, and synthetic-forward check."""
    set_seed()
    device = resolve_device()
    model, preprocessor, checkpoint = load_final_bundle(device=device)
    with torch.no_grad():
        output = model(torch.zeros((2, INPUT_DIM), dtype=torch.float32, device=device))
    if tuple(output.shape) != (2, OUTPUT_DIM):
        raise ValueError(f"Unexpected Diabetes output shape: {tuple(output.shape)}")
    return {
        "task": "diabetes",
        "architecture": f"{INPUT_DIM}→64→32→{OUTPUT_DIM}",
        "parameters": count_trainable_parameters(model),
        "synthetic_output_shape": list(output.shape),
        "preprocessor_output_dim": len(preprocessor.get_feature_names_out()),
        "checkpoint_experiment": checkpoint["experiment"],
        "training_reference": {
            "criterion": "weighted CrossEntropyLoss",
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
