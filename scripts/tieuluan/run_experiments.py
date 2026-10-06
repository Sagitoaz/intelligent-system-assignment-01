"""Chạy toàn bộ thực nghiệm Chương 2–4 và lưu kết quả có thể kiểm tra lại.

Mỗi lượt chạy lưu: cấu hình, lịch sử loss, số tham số, thời gian, xác suất dự báo trên
validation/test và mô hình đã huấn luyện. Kết quả chỉ được tái sử dụng khi "dấu vân tay"
(hash của dữ liệu đã xử lý + mã nguồn + môi trường) trùng khớp.

Chạy:  .venv\\Scripts\\python.exe -m scripts.tieuluan.run_experiments
"""

import os

for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ.setdefault(key, "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import argparse
import json
import platform
import re
import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, log_loss, matthews_corrcoef, precision_score, recall_score, roc_auc_score)

from src.tieuluan.config import SEEDS
from src.tieuluan.experiment_data import OUT, ROOT, prepare_datasets
from src.tieuluan.experiment_models import ARCHITECTURES, build_models, count_trainable, fit_model, predict_logits
from src.tieuluan.run_provenance import check_raw_snapshot, ensure_manifest, experiment_manifest

EPOCHS, PATIENCE, BATCH_SIZE = 60, 8, 64


def probabilities(logits):
    return .5 * (1 + np.tanh(np.asarray(logits, dtype=float) / 2))


def select_threshold(y, p):
    """Ngưỡng tối đa hóa balanced accuracy trên VALIDATION (lưới 0,05…0,95); hòa thì chọn gần 0,5 nhất."""
    candidates = np.linspace(.05, .95, 91)
    scores = np.array([balanced_accuracy_score(y, p >= t) for t in candidates])
    best = np.flatnonzero(np.isclose(scores, scores.max()))
    return float(candidates[best[np.argmin(abs(candidates[best] - .5))]])


def metrics(y, p, threshold):
    pred = p >= threshold
    return {"accuracy": float(accuracy_score(y, pred)), "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "roc_auc": float(roc_auc_score(y, p)), "average_precision": float(average_precision_score(y, p)),
            "f1": float(f1_score(y, pred, zero_division=0)), "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)), "mcc": float(matthews_corrcoef(y, pred)),
            "log_loss": float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7))),
            "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(), "threshold": threshold,
            "accuracy_at_05": float(accuracy_score(y, p >= .5))}


def job_list(seeds):
    jobs = []
    for dataset in ["taiwan_bankruptcy", "credit_default"]:
        for seed in seeds:
            for framework in ["scratch", "pytorch", "keras"]:
                jobs.append((dataset, "mlp", framework, seed, "tabular"))
            for kind in ["logistic", "random_forest"]:
                jobs.append((dataset, kind, "sklearn", seed, "tabular"))
        jobs.append((dataset, "majority", "baseline", 0, "tabular"))
    for dataset in ["sp500", "vnindex", "btc"]:
        for seed in seeds:
            for kind, representation in [("cnn4", "image"), ("lstm", "sequence")]:
                for framework in ["scratch", "pytorch", "keras"]:
                    jobs.append((dataset, kind, framework, seed, representation))
            for kind, representation in [("cnn8", "image"), ("cnndeep", "image"), ("rnn", "sequence"), ("gru", "sequence")]:
                jobs.append((dataset, kind, "pytorch", seed, representation))
        jobs.append((dataset, "majority", "baseline", 0, "sequence"))
        jobs.append((dataset, "persistence", "baseline", 0, "sequence"))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--patience", type=int, default=PATIENCE)
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--smoke", action="store_true", help="chạy thử nhanh trên vài trăm mẫu, lưu riêng")
    ap.add_argument("--output-label", default=None, help="thư mục kết quả mới khi cố ý thay đổi thí nghiệm")
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")]
    if not (OUT / "inventory.json").exists():
        prepare_datasets()
    check_raw_snapshot(ROOT)
    label = args.output_label or ("smoke" if args.smoke else "full")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", label):
        raise ValueError("output-label must be a simple directory name")
    result_dir, model_dir = ROOT / "results/tieuluan" / label, ROOT / "models/tieuluan" / label
    result_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    import keras
    import sklearn
    import tensorflow as tf
    import torch

    environment = {"python": platform.python_version(), "platform": platform.platform(), "numpy": np.__version__,
                   "torch": torch.__version__, "tensorflow": tf.__version__, "keras": keras.__version__,
                   "keras_backend": keras.backend.backend(), "sklearn": sklearn.__version__, "device": "CPU",
                   "threads": 2, "epochs": args.epochs, "patience": args.patience, "seeds": seeds,
                   "batch_size": BATCH_SIZE, "learning_rate": .001, "window": 20,
                   "threshold": "maximize validation balanced accuracy on grid .05:.01:.95",
                   "effective_bias": "RNN/LSTM redundant bias_hh frozen at zero in NumPy/PyTorch",
                   "timing": "training + per-epoch validation, first compilation included",
                   "initial_weights": "same NumPy draw mapped into all frameworks; same mini-batch order"}
    manifest = experiment_manifest(ROOT, environment)
    ensure_manifest(result_dir, manifest)
    environment_path = result_dir / "environment.json"
    if not environment_path.exists():
        environment_path.write_text(json.dumps(environment, indent=2), encoding="utf8")

    jobs = job_list(seeds)
    if args.smoke:
        jobs = [("taiwan_bankruptcy", "mlp", f, seeds[0], "tabular") for f in ["scratch", "pytorch", "keras"]] + [
            ("sp500", k, f, seeds[0], r) for k, r in [("cnn4", "image"), ("lstm", "sequence")]
            for f in ["scratch", "pytorch", "keras"]]
    cache = {}
    for number, (dataset, kind, framework, seed, representation) in enumerate(jobs, 1):
        stem = f"{dataset}__{kind}__{framework}__{seed}"
        dest = result_dir / f"{stem}.json"
        if dest.exists():
            old = json.loads(dest.read_text(encoding="utf8"))
            if old.get("run_fingerprint") != manifest["fingerprint"]:
                raise ValueError(f"incompatible or unversioned cached record: {stem}")
            print(f"SKIP {number}/{len(jobs)} {stem}", flush=True)
            continue
        if dataset not in cache:
            with np.load(OUT / f"{dataset}.npz", allow_pickle=False) as archive:
                cache[dataset] = {key: archive[key] for key in archive.files}
        data = cache[dataset]
        x, y, xv, yv, xt, yt = [data[key] for key in [f"train_{representation}", "train_y", f"val_{representation}",
                                                       "val_y", f"test_{representation}", "test_y"]]
        if args.smoke:
            x, y, xv, yv, xt, yt = x[:256], y[:256], xv[:128], yv[:128], xt[:128], yt[:128]
        record = {"dataset": dataset, "kind": kind, "framework": framework, "seed": seed,
                  "epochs_budget": args.epochs, "patience": args.patience, "n_train": len(y), "n_val": len(yv),
                  "n_test": len(yt), "representation": representation, "architecture": ARCHITECTURES.get(kind, kind),
                  "run_fingerprint": manifest["fingerprint"]}
        print(f"RUN {number}/{len(jobs)} {stem}", flush=True)
        if framework == "baseline":
            if kind == "majority":   # luôn dự báo tỷ lệ lớp dương của TRAIN
                pv, pt = np.full(len(yv), y.mean()), np.full(len(yt), y.mean())
            else:                    # persistence: dự báo tăng nếu lợi suất gần nhất dương
                mean, scale = data["scaler_mean"][0], data["scaler_scale"][0]
                pv = (xv[:, -1, 0] * scale + mean > 0).astype(float)
                pt = (xt[:, -1, 0] * scale + mean > 0).astype(float)
            threshold = .5
            record.update({"training_seconds": 0., "parameters": 0, "history": []})
        elif framework == "sklearn":
            model = (LogisticRegression(max_iter=1500, C=1., random_state=seed) if kind == "logistic" else
                     RandomForestClassifier(n_estimators=150, max_depth=8, min_samples_leaf=5, n_jobs=2, random_state=seed))
            start = time.perf_counter()
            model.fit(x, y)
            record["training_seconds"] = time.perf_counter() - start
            pv, pt = model.predict_proba(xv)[:, 1], model.predict_proba(xt)[:, 1]
            threshold = select_threshold(yv, pv)
            import joblib

            joblib.dump(model, model_dir / f"{stem}.joblib", compress=3)
            record.update({"parameters": None, "history": []})
        else:
            model = build_models(kind, x.shape[1:], seed, frameworks=(framework,))[framework]
            record.update(fit_model(model, framework, kind, x, y, xv, yv, seed, epochs=args.epochs,
                                    batch_size=BATCH_SIZE, patience=args.patience))
            pv = probabilities(predict_logits(model, framework, xv))
            pt = probabilities(predict_logits(model, framework, xt))
            threshold = select_threshold(yv, pv)
            record["parameters"] = count_trainable(model, framework, kind)
            if framework == "scratch":
                model.save(model_dir / f"{stem}.npz")
                restored = build_models(kind, x.shape[1:], seed, frameworks=("scratch",))["scratch"].load(model_dir / f"{stem}.npz")
                np.testing.assert_allclose(predict_logits(restored, "scratch", xt[:8]),
                                           predict_logits(model, "scratch", xt[:8]), atol=1e-7)
            elif framework == "pytorch":
                torch.save(model.state_dict(), model_dir / f"{stem}.pt")
            else:
                model.save(model_dir / f"{stem}.keras")
        record["metrics"] = metrics(yt, pt, threshold)
        record["validation_metrics"] = metrics(yv, pv, threshold)
        np.savez_compressed(result_dir / f"{stem}_predictions.npz", y=yt, probability=pt, val_y=yv, val_probability=pv)
        dest.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf8")
        print(f"DONE AUC={record['metrics']['roc_auc']:.4f} BAcc={record['metrics']['balanced_accuracy']:.4f} "
              f"time={record['training_seconds']:.1f}s", flush=True)
        if framework == "keras":
            keras.backend.clear_session()
    records = [json.loads(p.read_text(encoding="utf8")) for p in sorted(result_dir.glob("*__*.json"))]
    (result_dir / "all_results.json").write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf8")
    print(f"COMPLETED {len(records)} records", flush=True)


if __name__ == "__main__":
    main()
