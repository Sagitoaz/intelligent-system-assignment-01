"""Phân tích độ nhạy của Chương 2 theo cách tiền xử lý.

Bộ phá sản có 24 cột trộn hai thang đo (giá trị trong [0, 1] xen lẫn giá trị tới 1e10); bộ vỡ nợ có
các cột số tiền lệch phải mạnh. Script này giữ NGUYÊN cách chia train/validation/test của thực nghiệm
chính, chỉ thêm phép biến đổi log có dấu  x' = sign(x)·ln(1 + |x|)  trước khi điền khuyết và chuẩn hóa
(vẫn học trên train), rồi huấn luyện lại hồi quy logistic, rừng ngẫu nhiên và MLP (PyTorch, 3 seed).

Chạy:  .venv\\Scripts\\python.exe -m scripts.tieuluan.sensitivity_tabular
Kết quả: results/tieuluan/sensitivity_tabular.json
"""

import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from scripts.tieuluan.run_experiments import select_threshold
from src.tieuluan.analysis import bootstrap_auc
from src.tieuluan.config import RESULTS_DIR
from src.tieuluan.experiment_data import OUT, RAW
from src.tieuluan.experiment_models import build_models, fit_model, predict_logits

SEEDS = (11, 22, 33)


def load_raw(key):
    if key == "taiwan_bankruptcy":
        f = pd.read_csv(RAW / "uci" / key / "data.csv")
        return f.iloc[:, 1:], f.iloc[:, 0].to_numpy(np.float32)
    f = pd.read_excel(next((RAW / "uci" / key).glob("*.xls")), header=1)
    return f.iloc[:, 1:-1], f.iloc[:, -1].to_numpy(np.float32)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    report = {}
    for key in ("taiwan_bankruptcy", "credit_default"):
        x_raw, y = load_raw(key)
        with np.load(OUT / f"{key}.npz", allow_pickle=False) as d:
            idx = {split: d[f"{split}_ids"] for split in ("train", "val", "test")}
        x_log = np.sign(x_raw.to_numpy(float)) * np.log1p(np.abs(x_raw.to_numpy(float)))
        imputer = SimpleImputer(strategy="median").fit(x_log[idx["train"]])
        scaler = StandardScaler().fit(imputer.transform(x_log[idx["train"]]))
        x = scaler.transform(imputer.transform(x_log)).astype(np.float32)
        xs = {s: x[i] for s, i in idx.items()}
        ys = {s: y[i] for s, i in idx.items()}
        result = {}
        lr = LogisticRegression(max_iter=1500, C=1.0, random_state=11).fit(xs["train"], ys["train"])
        result["logistic"] = [lr.predict_proba(xs["test"])[:, 1]]
        result["random_forest"], result["mlp"] = [], []
        for seed in SEEDS:
            rf = RandomForestClassifier(n_estimators=150, max_depth=8, min_samples_leaf=5, n_jobs=2,
                                        random_state=seed).fit(xs["train"], ys["train"])
            result["random_forest"].append(rf.predict_proba(xs["test"])[:, 1])
            model = build_models("mlp", xs["train"].shape[1:], seed, frameworks=("pytorch",))["pytorch"]
            fit_model(model, "pytorch", "mlp", xs["train"], ys["train"], xs["val"], ys["val"], seed)
            pv = 1 / (1 + np.exp(-predict_logits(model, "pytorch", xs["val"])))
            pt = 1 / (1 + np.exp(-predict_logits(model, "pytorch", xs["test"])))
            result["mlp"].append(pt)
            result.setdefault("mlp_bacc", []).append(
                balanced_accuracy_score(ys["test"], pt >= select_threshold(ys["val"], pv)))
        report[key] = {}
        for name in ("logistic", "random_forest", "mlp"):
            aucs = [roc_auc_score(ys["test"], p) for p in result[name]]
            ci = bootstrap_auc(ys["test"], np.mean(result[name], axis=0))
            report[key][name] = {"auc_mean": float(np.mean(aucs)), "auc_sd": float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0,
                                 "ensemble_ci": ci}
        print(key, {k: round(v["auc_mean"], 4) for k, v in report[key].items()})
    (RESULTS_DIR / "sensitivity_tabular.json").write_text(json.dumps(report, indent=1), encoding="utf8")


if __name__ == "__main__":
    main()
