"""Standalone PyTorch source for Amazon review sentiment classification.

Datasets: ``data/ecommerce/train.csv`` and official ``test.csv``.
Final executed architecture: ``20000 -> 64 -> 32 -> 2`` over sparse TF-IDF.
The default CLI only checks saved artifacts; it never scans data or trains.
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
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset

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
TRAIN_RELATIVE_PATH = Path("data/ecommerce/train.csv")
TEST_RELATIVE_PATH = Path("data/ecommerce/test.csv")
CHECKPOINT_RELATIVE_PATH = Path("models/assignment03/comments/final_mlp.pt")
VECTORIZER_RELATIVE_PATH = Path("models/assignment03/comments/tfidf_vectorizer.joblib")

EXPECTED_TRAIN_ROWS = 3_600_000
EXPECTED_TEST_ROWS = 400_000
TRAIN_LIMIT = 800_000
VALIDATION_LIMIT = 100_000
TEST_LIMIT = 200_000
CHUNK_SIZE = 100_000
LABEL_MAP = {1: 0, 2: 1}
LABEL_NAME = {0: "Negative", 1: "Positive"}

TFIDF_MAX_FEATURES = 20_000
TFIDF_MIN_DF = 5
TFIDF_NGRAM_RANGE = (1, 2)
INPUT_DIM = 20_000
HIDDEN_DIMS = [64, 32]
OUTPUT_DIM = 2
LEARNING_RATE = 1e-3
BATCH_SIZE = 256
MAX_EPOCHS = 15
PATIENCE = 4
BEST_EPOCH_REFERENCE = 2
SELECTION_CRITERION = "validation F1, rồi validation loss"
EXPECTED_PARAMETERS = 1_282_210
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


class SparseIndexDataset(Dataset):
    """Return row indices so the collate function can slice CSR by batch."""

    def __init__(self, n_rows: int):
        self.n_rows = n_rows

    def __len__(self) -> int:
        return self.n_rows

    def __getitem__(self, index: int) -> int:
        return index


def normalize_reviews(series: pd.Series) -> pd.Series:
    """Apply only the whitespace normalization used by the notebook."""
    return (
        series.fillna("").astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def reproducible_csv_subset(
    csv_path: Path, desired_total: int, expected_rows: int
) -> pd.DataFrame:
    """Reproduce hash-based chunk sampling followed by balanced class sampling."""
    threshold = np.uint64(min(
        2**64 - 1,
        int((desired_total * 1.25 / expected_rows) * (2**64 - 1)),
    ))
    parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(csv_path, chunksize=CHUNK_SIZE):
        hashes = (
            pd.util.hash_pandas_object(chunk[["Review", "Label"]], index=False)
            .to_numpy(dtype="uint64")
            ^ np.uint64(RANDOM_STATE)
        )
        selected = chunk.loc[hashes <= threshold]
        if len(selected):
            parts.append(selected)
    frame = pd.concat(parts, ignore_index=True)
    frame = frame.dropna(subset=["Label"]).copy()
    frame["label"] = frame["Label"].map(LABEL_MAP)
    frame = frame.dropna(subset=["label"])
    frame["label"] = frame["label"].astype("int64")
    frame["text"] = normalize_reviews(frame["Review"])
    frame = frame.loc[frame["text"].str.len() > 0]
    per_class = desired_total // 2
    balanced = pd.concat(
        [
            group.sample(
                n=min(per_class, len(group)),
                random_state=RANDOM_STATE + int(label),
            )
            for label, group in frame.groupby("label")
        ],
        ignore_index=True,
    )
    return balanced.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)


def build_vectorizer() -> TfidfVectorizer:
    """Build the exact train-only unigram/bigram TF-IDF configuration."""
    return TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=TFIDF_NGRAM_RANGE,
        sublinear_tf=True,
        min_df=TFIDF_MIN_DF,
        dtype=np.float32,
        strip_accents="unicode",
    )


def load_and_prepare_data(repo_root: Path) -> dict[str, Any]:
    """Build the 800K/100K/200K sparse experiment data.

    This expensive function mirrors the notebook but is never called by the
    default check CLI.
    """
    train_pool = reproducible_csv_subset(
        repo_root / TRAIN_RELATIVE_PATH,
        TRAIN_LIMIT + VALIDATION_LIMIT,
        EXPECTED_TRAIN_ROWS,
    )
    test_df = reproducible_csv_subset(
        repo_root / TEST_RELATIVE_PATH, TEST_LIMIT, EXPECTED_TEST_ROWS
    )
    validation_fraction = VALIDATION_LIMIT / (TRAIN_LIMIT + VALIDATION_LIMIT)
    train_df, val_df = train_test_split(
        train_pool,
        test_size=validation_fraction,
        stratify=train_pool["label"],
        random_state=RANDOM_STATE,
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    vectorizer = build_vectorizer()
    x_train = vectorizer.fit_transform(train_df["text"])
    x_val = vectorizer.transform(val_df["text"])
    x_test = vectorizer.transform(test_df["text"])
    return {
        "train_df": train_df, "val_df": val_df, "test_df": test_df,
        "vectorizer": vectorizer,
        "x_train": x_train, "x_val": x_val, "x_test": x_test,
        "y_train": train_df["label"].to_numpy(),
        "y_val": val_df["label"].to_numpy(),
        "y_test": test_df["label"].to_numpy(),
    }


def make_sparse_loader(
    matrix: sparse.spmatrix, labels: np.ndarray, shuffle: bool
) -> DataLoader:
    """Densify only one ``[B,20000]`` CSR slice at a time."""
    label_array = np.asarray(labels, dtype=np.int64)

    def collate(indices: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        index_array = np.asarray(indices, dtype=np.int64)
        x_batch = matrix[index_array].toarray().astype(np.float32, copy=False)
        return torch.from_numpy(x_batch), torch.from_numpy(label_array[index_array])

    return DataLoader(
        SparseIndexDataset(matrix.shape[0]),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=collate,
        pin_memory=torch.cuda.is_available(),
    )


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
    device: torch.device,
    *,
    hidden_dims: list[int] = HIDDEN_DIMS,
    learning_rate: float = LEARNING_RATE,
    epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
) -> tuple[MLP, pd.DataFrame, dict[str, float]]:
    """Train with CrossEntropyLoss, Adam, and validation-loss early stopping."""
    model = MLP(INPUT_DIM, hidden_dims, OUTPUT_DIM).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_state = copy.deepcopy(model.state_dict())
    best_loss, best_epoch, wait = float("inf"), 0, 0
    history: list[dict[str, float]] = []
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        total, seen = 0.0, 0
        train_true: list[int] = []
        train_pred: list[int] = []
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(y_batch)
            seen += len(y_batch)
            train_true.extend(y_batch.cpu().numpy())
            train_pred.extend(logits.argmax(1).detach().cpu().numpy())

        validation = evaluate_classifier(model, val_loader, criterion, device)
        history.append({
            "epoch": epoch, "train_loss": total / seen,
            "val_loss": validation["loss"],
            "train_f1": f1_score(train_true, train_pred),
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
) -> tuple[MLP, TfidfVectorizer, dict[str, Any]]:
    """Load and validate the final checkpoint and fitted TF-IDF vectorizer."""
    root = repo_root or find_repo_root(TRAIN_RELATIVE_PATH)
    selected_device = device or resolve_device()
    checkpoint = load_checkpoint(root / CHECKPOINT_RELATIVE_PATH, selected_device)
    expected_metadata = {
        "input_dim": INPUT_DIM,
        "hidden_dims": HIDDEN_DIMS,
        "output_dim": OUTPUT_DIM,
        "label_map": LABEL_MAP,
        "experiment": "Deeper MLP / Adam 1e-3",
    }
    for key, expected in expected_metadata.items():
        if checkpoint.get(key) != expected:
            raise ValueError(f"Comments checkpoint {key}: {checkpoint.get(key)!r} != {expected!r}")
    model = instantiate_checkpoint_model(checkpoint, selected_device)
    vectorizer = joblib.load(root / VECTORIZER_RELATIVE_PATH)
    if len(vectorizer.vocabulary_) != INPUT_DIM:
        raise ValueError("Comments TF-IDF vocabulary không bằng 20,000")
    if count_trainable_parameters(model) != EXPECTED_PARAMETERS:
        raise ValueError("Comments parameter count không bằng 1,282,210")
    return model, vectorizer, checkpoint


def check_saved_artifacts() -> dict[str, Any]:
    """Run lightweight checkpoint, vectorizer, and synthetic-forward checks."""
    set_seed()
    device = resolve_device()
    model, vectorizer, checkpoint = load_final_bundle(device=device)
    with torch.no_grad():
        output = model(torch.zeros((2, INPUT_DIM), dtype=torch.float32, device=device))
    if tuple(output.shape) != (2, OUTPUT_DIM):
        raise ValueError(f"Unexpected Comments output shape: {tuple(output.shape)}")
    return {
        "task": "comments",
        "architecture": f"{INPUT_DIM}→64→32→{OUTPUT_DIM}",
        "parameters": count_trainable_parameters(model),
        "synthetic_output_shape": list(output.shape),
        "vocabulary_size": len(vectorizer.vocabulary_),
        "checkpoint_experiment": checkpoint["experiment"],
        "training_reference": {
            "criterion": "CrossEntropyLoss", "optimizer": "Adam",
            "learning_rate": LEARNING_RATE, "batch_size": BATCH_SIZE,
            "best_epoch": BEST_EPOCH_REFERENCE, "selection": SELECTION_CRITERION,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Validate saved artifacts only (default).")
    parser.parse_args()
    print(json.dumps(check_saved_artifacts(), ensure_ascii=True))


if __name__ == "__main__":
    main()
