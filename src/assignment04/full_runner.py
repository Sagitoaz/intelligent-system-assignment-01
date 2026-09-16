"""Final Improved-only PyTorch/TensorFlow training on frozen master splits."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .experiment import (
    classification_class_weights, evaluate, parameter_count, train_pytorch, train_tensorflow,
)
from .full_data import load_full_splits
from .parity_runner import _inverse_target_from_metadata, _make_model


FULL_ARCHITECTURE = "improved"
FULL_BACKENDS = ("pytorch", "tensorflow")
TASK_CONFIG = {
    "diabetes": {"problem": "classification", "batch_size": 256, "batch_seed": 4201},
    "house": {"problem": "regression", "batch_size": 256, "batch_seed": 4202},
    "comments": {"problem": "classification", "batch_size": 128, "batch_seed": 4203, "max_len": 192},
}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1_048_576), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _atomic_csv(frame, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _save_native(model, backend, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if backend == "pytorch":
        import torch
        torch.save(model.state_dict(), path)
    else:
        model.save_weights(path)


def _save_curves(task, history, figures_dir):
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for backend, group in history.groupby("implementation"):
        ax.plot(group.epoch, group.train_loss, label=f"{backend} train")
        ax.plot(group.epoch, group.val_loss, "--", label=f"{backend} validation")
    ax.set(title=f"{task.title()} full-scale Improved CNN", xlabel="Epoch", ylabel="Loss")
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout()
    fig.savefig(figures_dir / f"{task}_full_loss_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_metric_figures(task, rows, figures_dir):
    frame = pd.DataFrame(rows)
    if TASK_CONFIG[task]["problem"] == "classification":
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.7))
        for ax, backend in zip(axes, FULL_BACKENDS):
            row = frame.loc[frame.implementation.eq(backend)].iloc[0]
            matrix = np.asarray(json.loads(row.test_confusion_matrix))
            image = ax.imshow(matrix, cmap="Blues")
            for (i, j), value in np.ndenumerate(matrix):
                ax.text(j, i, f"{value:,}", ha="center", va="center")
            ax.set(title=backend, xlabel="Predicted", ylabel="Actual", xticks=[0, 1], yticks=[0, 1])
        fig.colorbar(image, ax=axes.ravel().tolist(), shrink=.75)
        fig.suptitle(f"{task.title()} full-scale confusion matrices")
    else:
        fig, ax = plt.subplots(figsize=(7, 4))
        x = np.arange(2); width = .36
        ax.bar(x - width / 2, frame.test_mae / 1000, width, label="MAE (thousand USD)")
        ax.bar(x + width / 2, frame.test_rmse / 1000, width, label="RMSE (thousand USD)")
        ax.set(xticks=x, xticklabels=frame.implementation, ylabel="Thousand USD",
               title="House full-scale test errors")
        ax.legend(); ax.grid(axis="y", alpha=.25)
    fig.tight_layout()
    fig.savefig(figures_dir / f"{task}_full_metrics.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_task(task, prepared_dir, metadata_path, parity_selection_path, initialization_path,
             results_root, models_root, figures_dir, *, max_epochs=20, patience=3, learning_rate=1e-3):
    selection = json.loads(Path(parity_selection_path).read_text(encoding="utf-8"))
    if selection["selected_architecture"] != FULL_ARCHITECTURE:
        raise ValueError(f"full-scale architecture was not frozen as improved for {task}")
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    if not Path(initialization_path).exists():
        raise FileNotFoundError(f"canonical parity initialization is required: {initialization_path}")
    canonical = dict(np.load(initialization_path))
    splits = load_full_splits(prepared_dir, task)
    expected = {name: metadata["splits"][name]["master_rows"] for name in ("train", "validation", "test")}
    actual = {name: len(splits[name]["y"]) for name in expected}
    if actual != expected:
        raise ValueError(f"master split counts differ for {task}: {actual} != {expected}")
    config = TASK_CONFIG[task]; problem = config["problem"]
    class_weights = classification_class_weights(splits["train"]["y"]) if task == "diabetes" else None
    inverse_target = _inverse_target_from_metadata(metadata) if problem == "regression" else None
    task_results = Path(results_root) / task; task_models = Path(models_root) / task
    results_path = task_results / "full_results.csv"
    history_path = task_results / "training_history.csv"
    rows = pd.read_csv(results_path).to_dict("records") if results_path.exists() else []
    histories = pd.read_csv(history_path).to_dict("records") if history_path.exists() else []
    completed = {row["implementation"] for row in rows if pd.notna(row.get("test_loss"))}
    for backend in FULL_BACKENDS:
        if backend in completed:
            print(f"SKIP finalized full {task}/{backend}", flush=True)
            continue
        print(f"TRAIN FULL {task}/improved/{backend}", flush=True)
        model = _make_model(task, FULL_ARCHITECTURE, backend, metadata, canonical)
        trainer = train_pytorch if backend == "pytorch" else train_tensorflow
        result = trainer(
            model, task, problem, splits["train"], splits["validation"],
            batch_size=config["batch_size"], max_epochs=max_epochs, patience=patience,
            learning_rate=learning_rate, batch_seed=config["batch_seed"],
            class_weights=class_weights, inverse_target=inverse_target,
        )
        common_path = task_models / f"improved_{backend}.npz"
        common_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(common_path, **result["state"])
        native_path = task_models / (f"improved_{backend}.pt" if backend == "pytorch" else f"improved_{backend}.weights.h5")
        _save_native(result["model"], backend, native_path)
        test_metrics, _ = evaluate(result["model"], backend, task, problem, splits["test"],
                                   config["batch_size"], class_weights=class_weights,
                                   inverse_target=inverse_target)
        row = {
            "dataset": task, "architecture": FULL_ARCHITECTURE, "implementation": backend,
            "train_samples": actual["train"], "validation_samples": actual["validation"],
            "test_samples": actual["test"], "parameter_count": parameter_count(model, backend),
            "best_epoch": result["best_epoch"], "epochs_ran": len(result["history"]),
            "training_seconds": result["seconds"], "learning_rate": learning_rate,
            "batch_size": config["batch_size"], "patience": patience, "batch_seed": config["batch_seed"],
            "initialization_sha256": _sha256(initialization_path), "checkpoint_file": str(common_path),
            "native_model_file": str(native_path), "test_scope": "complete frozen master test (includes prior parity subset; no post-test tuning)",
        }
        for prefix, metrics in (("validation", result["validation"]), ("test", test_metrics)):
            for key, value in metrics.items():
                row[f"{prefix}_{key}"] = json.dumps(value) if key == "confusion_matrix" else _jsonable(value)
        rows.append(row)
        histories.extend({"dataset": task, "architecture": FULL_ARCHITECTURE,
                          "implementation": backend, **entry} for entry in result["history"])
        _atomic_csv(pd.DataFrame(rows), results_path)
        _atomic_csv(pd.DataFrame(histories), history_path)
        print(f"DONE FULL {task}/{backend}: best_epoch={result['best_epoch']} seconds={result['seconds']:.2f} test={test_metrics}", flush=True)
    frame = pd.DataFrame(rows); history = pd.DataFrame(histories)
    _save_curves(task, history, Path(figures_dir))
    _save_metric_figures(task, rows, Path(figures_dir))
    return frame, history


def run_all(root, *, tasks=("diabetes", "house", "comments")):
    root = Path(root)
    prepared = root / "results/assignment04/full_scale/prepared"
    manifests = root / "results/assignment04/manifests"
    parity = root / "results/assignment04/parity"
    results = root / "results/assignment04/full_scale"
    models = root / "models/assignment04/full_scale"
    figures = root / "figures/assignment04"
    all_rows, all_history = [], []
    for task in tasks:
        frame, history = run_task(
            task, prepared, manifests / f"{task}_preprocessing.json", parity / task / "selection.json",
            root / f"models/assignment04/initializations/{task}_improved.npz",
            results, models, figures,
        )
        all_rows.append(frame); all_history.append(history)
    combined = pd.concat(all_rows, ignore_index=True)
    _atomic_csv(combined, results / "full_results.csv")
    _atomic_csv(pd.concat(all_history, ignore_index=True), results / "training_history.csv")
    _atomic_csv(combined.copy(), results / "framework_comparison.csv")
    return combined
