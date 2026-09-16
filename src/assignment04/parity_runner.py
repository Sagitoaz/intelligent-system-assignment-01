"""Run and persist the 18 controlled A04 parity experiments."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .comments_data import SharedTextVectorizer
from .experiment import (
    classification_class_weights,
    evaluate,
    load_scratch_state,
    load_tensorflow_state,
    load_torch_state,
    parameter_count,
    normalize_data_contract,
    scratch_state_dict,
    train_pytorch,
    train_scratch,
    train_tensorflow,
)
from .scratch_models import ScratchCNN, ScratchHouseCNN, ScratchTextCNN


ARCHITECTURES = ("basic", "improved")
BACKENDS = ("scratch", "pytorch", "tensorflow")
TASK_CONFIG = {
    "diabetes": {"problem": "classification", "batch_size": 256, "batch_seed": 4201},
    "house": {"problem": "regression", "batch_size": 256, "batch_seed": 4202},
    "comments": {"problem": "classification", "batch_size": 128, "batch_seed": 4203},
}


def _json_default(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def _write_json(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")


def _prepared_splits(path, task):
    archive = np.load(path)
    splits = {}
    for split in ("train", "validation", "test"):
        if task == "diabetes":
            values = {"x": archive[f"{split}_x"], "y": archive[f"{split}_y"]}
        elif task == "comments":
            values = {"ids": archive[f"{split}_ids"], "y": archive[f"{split}_y"]}
        else:
            values = {
                "numeric": archive[f"{split}_numeric"], "state": archive[f"{split}_state"],
                "status": archive[f"{split}_status"], "y": archive[f"{split}_y"],
            }
        splits[split] = normalize_data_contract(task, values)
    return splits


def _scratch_model(task, improved, metadata):
    rng = np.random.default_rng(42)
    if task == "diabetes":
        return ScratchCNN(1, 2, improved=improved, rng=rng)
    if task == "comments":
        return ScratchTextCNN(int(metadata["vocabulary_size"]), 2, improved=improved, rng=rng)
    return ScratchHouseCNN(int(metadata["state_size"]), int(metadata["status_size"]), improved=improved, rng=rng)


def _torch_model(task, improved, metadata):
    from .torch_models import TorchCNN, TorchHouseCNN, TorchTextCNN
    if task == "diabetes":
        return TorchCNN(1, 2, improved=improved)
    if task == "comments":
        return TorchTextCNN(int(metadata["vocabulary_size"]), 2, improved=improved)
    return TorchHouseCNN(int(metadata["state_size"]), int(metadata["status_size"]), improved=improved)


def _tf_model(task, improved, metadata):
    from .tensorflow_models import build_tf_cnn, build_tf_house_cnn, build_tf_text_cnn
    if task == "diabetes":
        return build_tf_cnn(21, 1, 2, improved=improved)
    if task == "comments":
        return build_tf_text_cnn(int(metadata["vocabulary_size"]), int(metadata["sequence_length"]), 2, improved=improved)
    return build_tf_house_cnn(int(metadata["state_size"]), int(metadata["status_size"]), improved=improved)


def _canonical_state(task, architecture, metadata, initialization_dir):
    path = Path(initialization_dir) / f"{task}_{architecture}.npz"
    reference = _scratch_model(task, architecture == "improved", metadata)
    expected = scratch_state_dict(reference)
    if path.exists():
        stored = dict(np.load(path))
        if stored.keys() != expected.keys() or any(stored[key].shape != expected[key].shape for key in expected):
            raise ValueError(f"Existing canonical initialization is incompatible: {path}")
        return stored, path
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **expected)
    return expected, path


def _make_model(task, architecture, backend, metadata, canonical):
    improved = architecture == "improved"
    if backend == "scratch":
        return load_scratch_state(_scratch_model(task, improved, metadata), canonical)
    if backend == "pytorch":
        return load_torch_state(_torch_model(task, improved, metadata), canonical)
    return load_tensorflow_state(_tf_model(task, improved, metadata), canonical, task)


def _save_checkpoint(path, backend, result):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    state = result["state"]
    np.savez_compressed(path, **state)


def _load_checkpoint(model, backend, task, path):
    state = dict(np.load(path))
    if backend == "scratch":
        return load_scratch_state(model, state)
    if backend == "pytorch":
        return load_torch_state(model, state)
    return load_tensorflow_state(model, state, task)


def _inverse_target_from_metadata(metadata):
    mean, scale = float(metadata["target_mean"]), float(metadata["target_scale"])
    # USD reporting must use float64: a finite standardized prediction can
    # legitimately exceed float32's expm1 range on extreme House rows.
    return lambda values: np.expm1(np.asarray(values, dtype=np.float64).reshape(-1) * scale + mean)


def _selection_table(run_rows, problem):
    frame = pd.DataFrame(run_rows)
    if problem == "classification":
        grouped = frame.groupby("architecture", as_index=False).agg(
            mean_validation_f1=("validation_f1", "mean"),
            mean_validation_loss=("validation_loss", "mean"),
        )
        grouped = grouped.sort_values(["mean_validation_f1", "mean_validation_loss"], ascending=[False, True])
    else:
        grouped = frame.groupby("architecture", as_index=False).agg(
            mean_validation_rmse=("validation_rmse", "mean"),
            mean_validation_mae=("validation_mae", "mean"),
        )
        grouped = grouped.sort_values(["mean_validation_rmse", "mean_validation_mae"], ascending=[True, True])
    return grouped.reset_index(drop=True)


def run_task(task, prepared_path, metadata_path, *, results_dir, models_dir,
             max_epochs=20, patience=3, learning_rate=1e-3):
    """Train six validation runs, freeze one architecture, then open test once."""
    config = TASK_CONFIG[task]
    problem, batch_size = config["problem"], config["batch_size"]
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    splits = _prepared_splits(prepared_path, task)
    class_weights = None
    if problem == "classification" and task == "diabetes":
        class_weights = classification_class_weights(splits["train"]["y"])
    inverse_target = _inverse_target_from_metadata(metadata) if problem == "regression" else None
    task_results = Path(results_dir) / task
    task_models = Path(models_dir) / task
    task_results.mkdir(parents=True, exist_ok=True); task_models.mkdir(parents=True, exist_ok=True)
    final_path = task_results / "parity_results.csv"
    selection_path = task_results / "architecture_comparison.csv"
    history_path = task_results / "training_history.csv"
    if final_path.exists() and selection_path.exists() and history_path.exists():
        existing = pd.read_csv(final_path)
        if len(existing) == 6 and "test_loss" in existing and existing["test_loss"].notna().sum() == 3:
            print(f"SKIP finalized task {task}; parity test will not be reopened", flush=True)
            return existing, pd.read_csv(selection_path), pd.read_csv(history_path)
    validation_path = task_results / "validation_runs.csv"
    run_rows = pd.read_csv(validation_path).to_dict("records") if validation_path.exists() else []
    histories = pd.read_csv(history_path).to_dict("records") if history_path.exists() else []
    completed = {(row["architecture"], row["implementation"]) for row in run_rows}

    # Test is not passed into any training function in this loop.
    for architecture in ARCHITECTURES:
        canonical, initialization_path = _canonical_state(task, architecture, metadata, Path(models_dir) / "initializations")
        for backend in BACKENDS:
            if (architecture, backend) in completed:
                checkpoint = task_models / f"{architecture}_{backend}.npz"
                if not checkpoint.exists():
                    raise FileNotFoundError(f"Result exists without checkpoint: {checkpoint}")
                print(f"SKIP completed {task}/{architecture}/{backend}", flush=True)
                continue
            model = _make_model(task, architecture, backend, metadata, canonical)
            trainer = {"scratch": train_scratch, "pytorch": train_pytorch, "tensorflow": train_tensorflow}[backend]
            result = trainer(
                model, task, problem, splits["train"], splits["validation"], batch_size=batch_size,
                max_epochs=max_epochs, patience=patience, learning_rate=learning_rate,
                batch_seed=config["batch_seed"], class_weights=class_weights, inverse_target=inverse_target,
            )
            checkpoint = task_models / f"{architecture}_{backend}.npz"
            _save_checkpoint(checkpoint, backend, result)
            row = {
                "dataset": task, "architecture": architecture, "implementation": backend,
                "train_samples": len(splits["train"]["y"]), "validation_samples": len(splits["validation"]["y"]),
                "test_samples": len(splits["test"]["y"]), "parameter_count": parameter_count(model, backend),
                "best_epoch": result["best_epoch"], "epochs_ran": len(result["history"]),
                "training_seconds": result["seconds"], "learning_rate": learning_rate,
                "batch_size": batch_size, "patience": patience, "batch_seed": config["batch_seed"],
                "initialization_file": str(initialization_path), "checkpoint_file": str(checkpoint),
            }
            row.update({f"validation_{key}": value for key, value in result["validation"].items()})
            run_rows.append(row)
            for history_row in result["history"]:
                histories.append({"dataset": task, "architecture": architecture, "implementation": backend, **history_row})
            pd.DataFrame(run_rows).to_csv(validation_path, index=False)
            pd.DataFrame(histories).to_csv(history_path, index=False)
            print(f"DONE {task}/{architecture}/{backend}: epoch={result['best_epoch']} "
                  f"seconds={result['seconds']:.2f} validation={result['validation']}", flush=True)

    selection = _selection_table(run_rows, problem)
    selected = str(selection.iloc[0]["architecture"])
    selection.to_csv(task_results / "architecture_comparison.csv", index=False)
    _write_json(task_results / "selection.json", {
        "dataset": task, "selected_architecture": selected,
        "rule": "mean validation F1 desc, then loss asc across frameworks" if problem == "classification"
                else "mean validation RMSE asc, then MAE asc across frameworks",
        "test_opened_after_selection": True,
    })

    # The common architecture is now frozen. Each equivalent implementation is
    # evaluated once on the same held-out parity-test IDs.
    for row in run_rows:
        if row["architecture"] != selected:
            continue
        canonical, _ = _canonical_state(task, selected, metadata, Path(models_dir) / "initializations")
        model = _make_model(task, selected, row["implementation"], metadata, canonical)
        model = _load_checkpoint(model, row["implementation"], task, row["checkpoint_file"])
        test_metrics, _ = evaluate(model, row["implementation"], task, problem, splits["test"], batch_size,
                                   class_weights=class_weights, inverse_target=inverse_target)
        row.update({f"test_{key}": value for key, value in test_metrics.items()})
        print(f"TEST {task}/{selected}/{row['implementation']}: {test_metrics}", flush=True)

    final = pd.DataFrame(run_rows)
    final.to_csv(task_results / "parity_results.csv", index=False)
    pd.DataFrame(histories).to_csv(task_results / "training_history.csv", index=False)
    return final, selection, pd.DataFrame(histories)


def run_all(prepared_dir, metadata_dir, results_dir, models_dir, **kwargs):
    frames, selections, histories = [], [], []
    for task in ("diabetes", "house", "comments"):
        frame, selection, history = run_task(
            task, Path(prepared_dir) / f"{task}_parity.npz", Path(metadata_dir) / f"{task}_preprocessing.json",
            results_dir=results_dir, models_dir=models_dir, **kwargs,
        )
        frames.append(frame); selection.insert(0, "dataset", task); selections.append(selection); histories.append(history)
    all_runs = pd.concat(frames, ignore_index=True)
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    all_runs.to_csv(Path(results_dir) / "parity_results.csv", index=False)
    pd.concat(histories, ignore_index=True).to_csv(Path(results_dir) / "training_history.csv", index=False)
    pd.concat(selections, ignore_index=True).to_csv(Path(results_dir) / "architecture_comparison.csv", index=False)
    selected_rows = all_runs.loc[all_runs["test_loss"].notna()].copy()
    selected_rows.to_csv(Path(results_dir) / "framework_comparison.csv", index=False)
    return all_runs
