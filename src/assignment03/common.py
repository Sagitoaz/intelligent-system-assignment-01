"""Small utilities shared by the three Assignment 03 PyTorch tasks."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn

RANDOM_STATE = 42


def set_seed(seed: int = RANDOM_STATE) -> None:
    """Set the Python, NumPy, and Torch seeds used by the notebooks."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def find_repo_root(relative_path: Path, start: Path | None = None) -> Path:
    """Find the nearest parent containing a required repository-relative path."""
    origin = (start or Path.cwd()).resolve()
    for candidate in (origin, *origin.parents):
        if (candidate / relative_path).exists():
            return candidate
    raise FileNotFoundError(f"Không tìm thấy {relative_path} từ {origin}")


def resolve_device() -> torch.device:
    """Use CUDA when available, otherwise CPU, matching notebook behavior."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MLP(nn.Module):
    """Dense MLP used by all three notebooks.

    Input is ``[batch, input_dim]``. Each hidden width produces a ReLU
    representation ``[batch, width]``; the final layer returns
    ``[batch, output_dim]``.
    """

    def __init__(self, input_dim: int, hidden_dims: Sequence[int], output_dim: int):
        super().__init__()
        layers: list[nn.Module] = []
        previous = input_dim
        for width in hidden_dims:
            layers.extend([nn.Linear(previous, width), nn.ReLU()])
            previous = width
        layers.append(nn.Linear(previous, output_dim))
        self.network = nn.Sequential(*layers)
        self.hidden_dims = tuple(hidden_dims)

    def forward(
        self, x: torch.Tensor, return_hidden: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        hidden = x
        hidden_states: list[torch.Tensor] = []
        for layer in self.network[:-1]:
            hidden = layer(hidden)
            if isinstance(layer, nn.ReLU):
                hidden_states.append(hidden)
        output = self.network[-1](hidden)
        return (output, hidden_states[-1]) if return_hidden else output


def count_trainable_parameters(model: nn.Module) -> int:
    """Count parameters whose gradients are enabled."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray
) -> dict[str, float]:
    """Return the classification metrics used in the executed notebooks."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": (
            roc_auc_score(y_true, y_score) if np.unique(y_true).size == 2 else np.nan
        ),
    }


def classification_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray
) -> np.ndarray:
    """Return the notebook-compatible confusion matrix with labels 0 and 1."""
    return confusion_matrix(y_true, y_pred, labels=[0, 1])


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    """Load an existing metadata-plus-state-dict checkpoint without modifying it."""
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # Compatibility with older supported Torch releases.
        checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model_state"), dict):
        raise ValueError(f"Checkpoint không có model_state hợp lệ: {path}")
    return checkpoint


def instantiate_checkpoint_model(
    checkpoint: dict[str, Any], device: torch.device
) -> MLP:
    """Instantiate the checkpoint architecture and strictly load its state dict."""
    model = MLP(
        int(checkpoint["input_dim"]),
        list(checkpoint["hidden_dims"]),
        int(checkpoint["output_dim"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    return model
