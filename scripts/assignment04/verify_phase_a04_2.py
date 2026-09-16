"""Final integrity checks and auditable metadata for A04-2 artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import nbformat
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.assignment04.data_freeze import verify_frozen_manifest
from src.assignment04.experiment import predict_batches
from src.assignment04.parity import backward_parity_with_torch, finite_difference_checks, forward_parity
from src.assignment04.parity_runner import _canonical_state, _make_model, _prepared_splits


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1_048_576), b""):
            digest.update(block)
    return digest.hexdigest()


def initial_parity(task, metadata, prepared, models_dir):
    splits = _prepared_splits(prepared, task)
    result = {}
    for architecture in ("basic", "improved"):
        state, _ = _canonical_state(task, architecture, metadata, Path(models_dir) / "initializations")
        outputs = {}
        for backend in ("scratch", "pytorch", "tensorflow"):
            model = _make_model(task, architecture, backend, metadata, state)
            outputs[backend] = predict_batches(model, backend, task, {
                key: value[:4] for key, value in splits["validation"].items()
            }, 4)
        result[architecture] = {
            "scratch_pytorch_max_abs": float(np.max(np.abs(outputs["scratch"] - outputs["pytorch"]))),
            "scratch_tensorflow_max_abs": float(np.max(np.abs(outputs["scratch"] - outputs["tensorflow"]))),
        }
    return result


def main():
    parity_dir = ROOT / "results/assignment04/parity"
    manifest_dir = ROOT / "results/assignment04/manifests"
    models_dir = ROOT / "models/assignment04"
    frames = []
    initial = {}
    for task in ("diabetes", "house", "comments"):
        metadata_path = manifest_dir / f"{task}_preprocessing.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        frame = pd.read_csv(parity_dir / task / "parity_results.csv")
        frame["prepared_sha256"] = metadata["prepared_artifact"]["sha256"]
        manifest_path = manifest_dir / ("comments/manifest.json" if task == "comments" else f"{task}.json")
        frame["manifest_file"] = str(manifest_path.relative_to(ROOT))
        frame["timing_valid"] = True
        frame["timing_note"] = "measured wall time"
        anomaly = (frame.architecture.eq("improved") & frame.implementation.eq("tensorflow") & frame.dataset.eq("comments"))
        frame.loc[anomaly, "timing_valid"] = False
        frame.loc[anomaly, "timing_note"] = "wall time includes host/session suspension; exclude from speed comparison"
        frame.to_csv(parity_dir / task / "parity_results.csv", index=False)
        frames.append(frame)
        initial[task] = initial_parity(task, metadata, parity_dir / "prepared" / f"{task}_parity.npz", models_dir)

    all_runs = pd.concat(frames, ignore_index=True)
    all_runs.to_csv(parity_dir / "parity_results.csv", index=False)
    all_runs.loc[all_runs.test_loss.notna()].to_csv(parity_dir / "framework_comparison.csv", index=False)

    completed = all_runs.groupby(["dataset", "architecture", "implementation"]).size()
    numeric = all_runs.select_dtypes("number")
    finite_nonmissing = bool(np.isfinite(numeric.to_numpy()[~np.isnan(numeric.to_numpy())]).all())
    history = pd.read_csv(parity_dir / "training_history.csv")
    history_finite = bool(np.isfinite(history[["train_loss", "val_loss"]].to_numpy()).all())

    tests = subprocess.run(
        [str(ROOT / ".venv/Scripts/python.exe"), "-m", "pytest", "tests/assignment04", "-q"],
        cwd=ROOT, text=True, capture_output=True,
    )
    if tests.returncode:
        raise RuntimeError(tests.stdout + tests.stderr)

    comments_manifest = json.loads((manifest_dir / "comments/manifest.json").read_text(encoding="utf-8"))
    comments_checksums = all(
        sha256(manifest_dir / "comments" / path) == expected
        for path, expected in comments_manifest["artifact_sha256"].items()
    )
    protected = subprocess.run(
        ["git", "diff", "--name-only", "--", "notebooks/assignment03", "src/assignment03", "models/assignment03", "figures/assignment03", "results/assignment03"],
        cwd=ROOT, text=True, capture_output=True,
    ).stdout.strip().splitlines()
    notebook_checks = {}
    for notebook_path in sorted((ROOT / "notebooks/assignment04").glob("*.ipynb")):
        notebook = nbformat.read(notebook_path, 4)
        code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
        notebook_checks[notebook_path.name] = {
            "cells": len(notebook.cells), "code_cells": len(code_cells),
            "executed_code_cells": sum(cell.get("execution_count") is not None for cell in code_cells),
            "error_outputs": sum(output.output_type == "error" for cell in code_cells for output in cell.get("outputs", [])),
        }
    figure_checks = {
        path.name: {"exists": path.exists(), "bytes": path.stat().st_size if path.exists() else 0}
        for path in [ROOT / f"figures/assignment04/{task}_parity_loss_curves.png" for task in ("diabetes", "house", "comments")]
    }

    verification = {
        "tests": {"command": ".venv/Scripts/python.exe -m pytest tests/assignment04 -q", "passed": True,
                  "output": tests.stdout.strip()},
        "finite_difference_relative_errors": finite_difference_checks(),
        "base_forward_parity": {name: forward_parity(improved=(name == "improved")) for name in ("basic", "improved")},
        "backward_parity_with_pytorch": backward_parity_with_torch(),
        "task_initial_forward_parity": initial,
        "artifact_checks": {
            "run_count": int(len(all_runs)), "unique_run_keys": int(len(completed)),
            "all_run_keys_once": bool(len(completed) == 18 and (completed == 1).all()),
            "all_numeric_outputs_finite_where_present": finite_nonmissing,
            "all_history_losses_finite": history_finite,
            "selected_test_rows_per_task": {key: int(value) for key, value in all_runs.groupby("dataset").test_loss.count().items()},
            "nonselected_test_metrics_absent": bool(all_runs.loc[all_runs.architecture == "basic", "test_loss"].isna().all()),
            "parameter_counts_equal_across_frameworks": bool(all_runs.groupby(["dataset", "architecture"]).parameter_count.nunique().eq(1).all()),
            "diabetes_manifest_verified": verify_frozen_manifest(manifest_dir / "diabetes.json"),
            "house_manifest_verified": verify_frozen_manifest(manifest_dir / "house.json"),
            "comments_artifact_checksums_verified": comments_checksums,
            "protected_a03_diff_empty": not protected,
            "protected_a03_diff": protected,
            "notebooks": notebook_checks,
            "all_notebooks_executed_without_errors": all(
                item["code_cells"] == item["executed_code_cells"] and item["error_outputs"] == 0
                for item in notebook_checks.values()
            ),
            "figures": figure_checks,
            "all_figures_nonempty": all(item["exists"] and item["bytes"] > 0 for item in figure_checks.values()),
        },
        "procedure_incident": {
            "dataset": "diabetes", "event": "resume guard initially treated test_samples as a metric and recomputed test for scratch/pytorch once",
            "impact": "no weights, selection, hyperparameters, or stored metrics changed; no tuning followed",
            "fix": "completed-task guard now keys only on three non-null test_loss rows",
        },
        "timing_anomaly": {
            "run": "comments/improved/tensorflow", "recorded_wall_seconds": float(all_runs.loc[
                (all_runs.dataset == "comments") & (all_runs.architecture == "improved") & (all_runs.implementation == "tensorflow"),
                "training_seconds"].iloc[0]),
            "status": "invalid for speed comparison because wall timer included host/session suspension",
        },
    }
    path = parity_dir / "verification.json"
    path.write_text(json.dumps(verification, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(verification, indent=2), flush=True)


if __name__ == "__main__":
    main()
