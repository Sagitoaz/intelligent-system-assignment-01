"""Huấn luyện Chương 2 trên bộ Lending Club 150.000 dòng và lưu cùng định dạng với run_experiments.

Cấu hình trung thực (khác run_experiments vì dữ liệu lớn hơn ~5 lần):
  MLP d→16→1 (ReLU), Adam lr 0,001, mini-batch 256, tối đa 25 epoch, dừng sớm sau 5 epoch không cải thiện
  BCE validation; hồi quy logistic (C=1) và rừng ngẫu nhiên (150 cây, độ sâu 8, lá ≥ 20 mẫu); 3 hạt giống 11, 22, 33.
  Ngưỡng quyết định chọn trên validation để tối đa balanced accuracy.

Chạy:  .venv\\Scripts\\python.exe -m scripts.tieuluan.run_loan
"""
import os

for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ.setdefault(key, "4")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import hashlib
import json
import time

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from scripts.tieuluan.run_experiments import metrics, probabilities, select_threshold
from src.tieuluan.analysis import bootstrap_auc, bootstrap_auc_difference
from src.tieuluan.config import SEEDS
from src.tieuluan.experiment_data import ROOT
from src.tieuluan.experiment_models import ARCHITECTURES, build_models, count_trainable, fit_model, predict_logits

DATASET = "loan_default"
EPOCHS, PATIENCE, BATCH = 25, 5, 256
RESULTS = ROOT / "results/tieuluan/full"
MODELS = ROOT / "models/tieuluan/full"


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    npz = ROOT / f"data/tieuluan/processed/{DATASET}.npz"
    fingerprint = hashlib.sha256(npz.read_bytes() + json.dumps([EPOCHS, PATIENCE, BATCH]).encode()).hexdigest()
    with np.load(npz, allow_pickle=False) as a:
        data = {k: a[k] for k in a.files}
    x, y, xv, yv, xt, yt = (data[k] for k in ["train_tabular", "train_y", "val_tabular", "val_y", "test_tabular", "test_y"])
    print("train", x.shape, "val", xv.shape, "test", xt.shape, flush=True)
    import keras
    import torch
    torch.set_num_threads(4)
    jobs = [(kind, fw, seed) for seed in SEEDS for kind, fw in
            [("mlp", "scratch"), ("mlp", "pytorch"), ("mlp", "keras"), ("logistic", "sklearn"), ("random_forest", "sklearn")]]
    jobs.append(("majority", "baseline", 0))
    for number, (kind, framework, seed) in enumerate(jobs, 1):
        stem = f"{DATASET}__{kind}__{framework}__{seed}"
        dest = RESULTS / f"{stem}.json"
        if dest.exists() and json.loads(dest.read_text(encoding="utf8")).get("run_fingerprint") == fingerprint:
            print("SKIP", stem, flush=True)
            continue
        print(f"RUN {number}/{len(jobs)} {stem}", flush=True)
        record = {"dataset": DATASET, "kind": kind, "framework": framework, "seed": seed, "epochs_budget": EPOCHS,
                  "patience": PATIENCE, "batch_size": BATCH, "n_train": len(y), "n_val": len(yv), "n_test": len(yt),
                  "representation": "tabular", "architecture": ARCHITECTURES.get(kind, kind), "run_fingerprint": fingerprint}
        if framework == "baseline":
            pv, pt, threshold = np.full(len(yv), y.mean()), np.full(len(yt), y.mean()), .5
            record.update({"training_seconds": 0., "parameters": 0, "history": []})
        elif framework == "sklearn":
            model = (LogisticRegression(max_iter=2000, C=1., random_state=seed) if kind == "logistic" else
                     RandomForestClassifier(n_estimators=150, max_depth=8, min_samples_leaf=20, n_jobs=4, random_state=seed))
            start = time.perf_counter()
            model.fit(x, y)
            record["training_seconds"] = time.perf_counter() - start
            pv, pt = model.predict_proba(xv)[:, 1], model.predict_proba(xt)[:, 1]
            threshold = select_threshold(yv, pv)
            joblib.dump(model, MODELS / f"{stem}.joblib", compress=3)
            record.update({"parameters": None, "history": []})
        else:
            model = build_models(kind, x.shape[1:], seed, frameworks=(framework,))[framework]
            record.update(fit_model(model, framework, kind, x, y, xv, yv, seed, epochs=EPOCHS, batch_size=BATCH, patience=PATIENCE))
            pv, pt = (probabilities(predict_logits(model, framework, a)) for a in (xv, xt))
            threshold = select_threshold(yv, pv)
            record["parameters"] = count_trainable(model, framework, kind)
            if framework == "scratch":
                model.save(MODELS / f"{stem}.npz")
            elif framework == "pytorch":
                torch.save(model.state_dict(), MODELS / f"{stem}.pt")
            else:
                model.save(MODELS / f"{stem}.keras")
                keras.backend.clear_session()
        record["metrics"] = metrics(yt, pt, threshold)
        record["validation_metrics"] = metrics(yv, pv, threshold)
        np.savez_compressed(RESULTS / f"{stem}_predictions.npz", y=yt, probability=pt, val_y=yv, val_probability=pv)
        dest.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf8")
        print(f"  AUC={record['metrics']['roc_auc']:.4f} BAcc={record['metrics']['balanced_accuracy']:.4f} "
              f"epochs={record.get('epochs_run')} time={record['training_seconds']:.1f}s", flush=True)

    # Tổng hợp: ghi lại all_results.json, khoảng tin cậy AUC, ngưỡng của tổ hợp 3 seed NumPy dùng trong demo.
    records = [json.loads(p.read_text(encoding="utf8")) for p in sorted(RESULTS.glob("*__*.json"))]
    (RESULTS / "all_results.json").write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf8")

    def load(kind, fw, seed):
        with np.load(RESULTS / f"{DATASET}__{kind}__{fw}__{seed}_predictions.npz") as a:
            return {k: a[k] for k in a.files}

    ens = {k: np.mean([load("mlp", "scratch", s)[k] for s in SEEDS], axis=0) for k in ("probability", "val_probability")}
    yt, yv = load("mlp", "scratch", SEEDS[0])["y"], load("mlp", "scratch", SEEDS[0])["val_y"]
    threshold = select_threshold(yv, ens["val_probability"])
    p_rf = np.mean([load("random_forest", "sklearn", s)["probability"] for s in SEEDS], axis=0)
    p_lr = load("logistic", "sklearn", SEEDS[0])["probability"]
    p_torch = np.mean([load("mlp", "pytorch", s)["probability"] for s in SEEDS], axis=0)   # như analyze_results
    analysis_path = RESULTS / "analysis.json"
    analysis = json.loads(analysis_path.read_text(encoding="utf8"))
    analysis.setdefault("uci", {})[DATASET] = {
        "logistic": bootstrap_auc(yt, p_lr), "random_forest": bootstrap_auc(yt, p_rf), "mlp": bootstrap_auc(yt, p_torch),
        "rf_minus_mlp": bootstrap_auc_difference(yt, p_rf, p_torch),
        "mlp_minus_logistic": bootstrap_auc_difference(yt, p_torch, p_lr)}
    analysis_path.write_text(json.dumps(analysis, indent=1, ensure_ascii=False), encoding="utf8")
    from sklearn.metrics import balanced_accuracy_score
    meta = {"dataset": DATASET, "threshold": threshold, "base_rate": float(y.mean()),
            "test_base_rate": float(yt.mean()), "ensemble_seeds": list(SEEDS),
            "ensemble_test_balanced_accuracy": float(balanced_accuracy_score(yt, ens["probability"] >= threshold)),
            "ensemble_test_auc": analysis["uci"][DATASET]["mlp"]["auc"],
            "config": {"epochs_max": EPOCHS, "patience": PATIENCE, "batch_size": BATCH}}
    (ROOT / "data/tieuluan/app/loan_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf8")
    print("COMPLETED", json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
