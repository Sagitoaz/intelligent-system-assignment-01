"""Phép thử phụ cho Chương 3: chuẩn hóa GASF về [−1, 1] (mất thông tin chiều giá) so với [0, 1] (dùng chính).

Giữ nguyên mọi thứ khác: cùng cửa sổ, nhãn, cách chia, kiến trúc CNN4 (PyTorch), 3 hạt giống, giao thức huấn luyện.
Chạy:  .venv\\Scripts\\python.exe -m scripts.tieuluan.ablation_gasf
Kết quả: results/tieuluan/ablation_gasf.json
"""

import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.tieuluan.analysis import bootstrap_auc
from src.tieuluan.config import ANALYSIS_START, RESULTS_DIR
from src.tieuluan.experiment_data import OUT, RAW, market_samples, split_by_dates
from src.tieuluan.experiment_models import build_models, fit_model, predict_logits

SEEDS = (11, 22, 33)


def gasf_minus_one_to_one(window):
    z = np.zeros_like(window) if np.ptp(window) < 1e-12 else 2 * (window - window.min()) / np.ptp(window) - 1
    z = np.clip(z, -1, 1)
    s = np.sqrt(np.maximum(0, 1 - z * z))
    return (np.outer(z, z) - np.outer(s, s)).astype(np.float32)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    report = {}
    for key in ("sp500", "vnindex", "btc"):
        frame = pd.read_csv(RAW / "indices" / f"{key}.csv")
        frame = frame[pd.to_datetime(frame.date) >= pd.Timestamp(ANALYSIS_START[key])]
        samples = market_samples(frame)
        prices = frame.sort_values("date").close.to_numpy(float)
        anchors = np.arange(20, len(prices) - 1)
        images = np.array([gasf_minus_one_to_one(prices[i - 19:i + 1])[None] for i in anchors], dtype=np.float32)
        masks = split_by_dates(samples["date"], samples["target_date"])
        x = {s: images[m] for s, m in masks.items()}
        y = {s: samples["y"][m] for s, m in masks.items()}
        with np.load(OUT / f"{key}.npz", allow_pickle=False) as d:   # kiểm tra cùng nhãn với thực nghiệm chính
            assert np.array_equal(d["test_y"], y["test"])
        probs, aucs = [], []
        for seed in SEEDS:
            model = build_models("cnn4", (1, 20, 20), seed, frameworks=("pytorch",))["pytorch"]
            fit_model(model, "pytorch", "cnn4", x["train"], y["train"], x["val"], y["val"], seed)
            p = 1 / (1 + np.exp(-predict_logits(model, "pytorch", x["test"])))
            probs.append(p)
            aucs.append(roc_auc_score(y["test"], p))
        report[key] = {"auc_mean": float(np.mean(aucs)), "auc_sd": float(np.std(aucs, ddof=1)),
                       "ensemble_ci": bootstrap_auc(y["test"], np.mean(probs, axis=0), block=20)}
        print(key, round(report[key]["auc_mean"], 4), report[key]["ensemble_ci"])
    (RESULTS_DIR / "ablation_gasf.json").write_text(json.dumps(report, indent=1), encoding="utf8")


if __name__ == "__main__":
    main()
