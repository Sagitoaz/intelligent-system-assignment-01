"""Verify the final A04 full-scale artifacts without retraining."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import nbformat
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
EXPECTED = {
    "diabetes": (177576, 38052, 38052),
    "house": (280000, 60000, 60000),
    "comments": (800000, 100000, 200000),
}


def main():
    results = pd.read_csv(ROOT / "results/assignment04/full_scale/full_results.csv")
    assert len(results) == 6
    assert set(results.implementation) == {"pytorch", "tensorflow"}
    assert set(results.architecture) == {"improved"}
    classification_columns = [
        f"{split}_{metric}" for split in ("validation", "test")
        for metric in ("loss", "accuracy", "precision", "recall", "f1", "roc_auc")
    ]
    regression_columns = [
        f"{split}_{metric}" for split in ("validation", "test")
        for metric in ("loss", "mae", "rmse", "r2")
    ]
    classification_rows = results.loc[results.dataset.isin(["diabetes", "comments"])]
    regression_rows = results.loc[results.dataset.eq("house")]
    assert np.isfinite(classification_rows[classification_columns].to_numpy(dtype=float)).all()
    assert np.isfinite(regression_rows[regression_columns].to_numpy(dtype=float)).all()
    for task, counts in EXPECTED.items():
        rows = results.loc[results.dataset.eq(task)]
        assert len(rows) == 2
        assert (rows[["train_samples", "validation_samples", "test_samples"]].to_numpy() == counts).all()
        for _, row in rows.iterrows():
            assert (ROOT / row.checkpoint_file).exists() if not Path(row.checkpoint_file).is_absolute() else Path(row.checkpoint_file).exists()
            assert (ROOT / row.native_model_file).exists() if not Path(row.native_model_file).is_absolute() else Path(row.native_model_file).exists()
        assert (ROOT / f"figures/assignment04/{task}_full_loss_curves.png").stat().st_size > 0
        assert (ROOT / f"figures/assignment04/{task}_full_metrics.png").stat().st_size > 0
    comments = json.loads((ROOT / "results/assignment04/manifests/comments_preprocessing.json").read_text(encoding="utf-8"))
    assert comments["sequence_length"] == 192 and comments["max_vocabulary"] == 20000

    notebooks = {
        "01_diabetes_cnn.ipynb": ("ScratchCNN", "TorchCNN", "TFCNN"),
        "02_house_price_cnn.ipynb": ("ScratchHouseCNN", "TorchHouseCNN", "TFHouseCNN", "FieldEncoder"),
        "03_comments_cnn.ipynb": ("ScratchTextCNN", "TorchTextCNN", "TFTextCNN", "Embedding", "MAX_LEN=192"),
    }
    notebook_summary = {}
    for filename, markers in notebooks.items():
        notebook = nbformat.read(ROOT / "notebooks/assignment04" / filename, as_version=4)
        code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
        assert code_cells and all(cell.execution_count is not None for cell in code_cells)
        assert not any(output.get("output_type") == "error" for cell in code_cells for output in cell.get("outputs", []))
        text = "\n".join(cell.source for cell in notebook.cells)
        assert all(marker in text for marker in markers)
        assert "Full-scale Improved CNN" in text and "Phân tích kết quả" in text
        notebook_summary[filename] = {"cells": len(notebook.cells), "executed_code_cells": len(code_cells)}

    protected = ["notebooks/assignment03", "src/assignment03", "models/assignment03", "figures/assignment03", "results/assignment03"]
    for cached in (False, True):
        command = ["git", "diff", "--quiet"] + (["--cached"] if cached else []) + ["--", *protected]
        assert subprocess.run(command, cwd=ROOT).returncode == 0
    payload = {"status": "pass", "runs": 6, "notebooks": notebook_summary, "a03_protected_diff": False}
    path = ROOT / "results/assignment04/full_scale/verification.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
