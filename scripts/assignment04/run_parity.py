"""CLI for the 18 validation runs and post-freeze parity-test evaluation."""

from __future__ import annotations

import os
from pathlib import Path
import sys


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.assignment04.parity_runner import run_all


if __name__ == "__main__":
    run_all(
        REPO_ROOT / "results/assignment04/parity/prepared",
        REPO_ROOT / "results/assignment04/manifests",
        REPO_ROOT / "results/assignment04/parity",
        REPO_ROOT / "models/assignment04",
        max_epochs=20, patience=3, learning_rate=1e-3,
    )
