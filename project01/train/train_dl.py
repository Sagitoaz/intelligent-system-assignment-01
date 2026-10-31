"""Huấn luyện agent Deep Learning của project01 trên BRFSS 2015 (253.680 hồ sơ, 21 chỉ số).

Nhãn 3 mức: 0 = không tiểu đường, 1 = tiền tiểu đường, 2 = tiểu đường.
Chạy: .venv\\Scripts\\python.exe project01/train/train_dl.py
Kết quả: project01/artifacts/dl_model.npz (trọng số cho suy luận bằng numpy) và dl_metrics.json.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CSV = os.path.join(ROOT, "data", "diabetes", "diabetes_012_health_indicators_BRFSS2015.csv")
OUT = os.path.join(ROOT, "project01", "artifacts")
FEATURES = ["HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke", "HeartDiseaseorAttack", "PhysActivity", "Fruits",
            "Veggies", "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth", "MentHlth", "PhysHlth", "DiffWalk",
            "Sex", "Age", "Education", "Income"]
SEED = 42


def build_model(n_in):
    return torch.nn.Sequential(
        torch.nn.Linear(n_in, 128), torch.nn.ReLU(), torch.nn.Dropout(0.15),
        torch.nn.Linear(128, 64), torch.nn.ReLU(), torch.nn.Dropout(0.1),
        torch.nn.Linear(64, 32), torch.nn.ReLU(),
        torch.nn.Linear(32, 3))


def numpy_forward(params, x):
    h = x
    for i, (w, b) in enumerate(params):
        h = h @ w + b
        if i < len(params) - 1:
            h = np.maximum(h, 0)
    h = h - h.max(axis=1, keepdims=True)
    e = np.exp(h)
    return e / e.sum(axis=1, keepdims=True)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    df = pd.read_csv(CSV)
    y = df["Diabetes_012"].astype(int).values
    X = df[FEATURES].astype("float32").values
    counts = np.bincount(y)
    print("rows", len(df), "class counts", counts.tolist())
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    mean, std = Xtr.mean(axis=0), Xtr.std(axis=0) + 1e-6
    Xtr_n, Xte_n = (Xtr - mean) / std, (Xte - mean) / std
    Xfit, Xval, yfit, yval = train_test_split(Xtr_n, ytr, test_size=0.15, random_state=SEED, stratify=ytr)

    # baseline: hồi quy logistic (cùng dữ liệu, cùng chia)
    t0 = time.time()
    lr = LogisticRegression(max_iter=300, class_weight="balanced")
    lr.fit(Xtr_n, ytr)
    p_lr = lr.predict_proba(Xte_n)
    base = dict(name="Logistic Regression (cân bằng lớp)", accuracy=float(accuracy_score(yte, p_lr.argmax(1))),
                macro_f1=float(f1_score(yte, p_lr.argmax(1), average="macro")),
                auc_ovr=float(roc_auc_score(yte, p_lr, multi_class="ovr")), seconds=round(time.time() - t0, 1))
    print("baseline", base)

    # MLP 3 lớp ẩn
    model = build_model(len(FEATURES))
    weights = torch.tensor((len(yfit) / (3 * np.bincount(yfit))) ** 0.75, dtype=torch.float32)  # cân bằng lớp (mũ 0,75)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-5)
    Xt, yt = torch.tensor(Xfit), torch.tensor(yfit)
    Xv = torch.tensor(Xval)
    Xtest_t = torch.tensor(Xte_n)
    history = []
    best = (-1.0, None)
    t0 = time.time()
    for epoch in range(14):
        model.train()
        perm = torch.randperm(len(Xt))
        total = 0.0
        for i in range(0, len(Xt), 1024):
            idx = perm[i:i + 1024]
            opt.zero_grad()
            loss = loss_fn(model(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
            total += float(loss) * len(idx)
        model.eval()
        with torch.no_grad():
            pv = torch.softmax(model(Xv), 1).numpy()
        val_auc = float(roc_auc_score(yval, pv, multi_class="ovr"))
        history.append(dict(epoch=epoch + 1, train_loss=round(total / len(Xt), 4), val_macro_f1=round(float(f1_score(yval, pv.argmax(1), average="macro")), 4),
                            val_auc_ovr=round(val_auc, 4)))
        print(history[-1])
        if val_auc > best[0]:
            best = (val_auc, {k: v.clone() for k, v in model.state_dict().items()}, epoch + 1)
    model.load_state_dict(best[1])
    model.eval()
    with torch.no_grad():
        pv = torch.softmax(model(Xv), 1).numpy()
        pt = torch.softmax(model(Xtest_t), 1).numpy()
    seconds = round(time.time() - t0, 1)

    params = []
    for layer in model:
        if isinstance(layer, torch.nn.Linear):
            params.append((layer.weight.detach().numpy().T.astype("float32"), layer.bias.detach().numpy().astype("float32")))
    probs = numpy_forward(params, Xte_n)
    assert np.allclose(probs, pt, atol=1e-4)
    # hệ số điều chỉnh ngưỡng quyết định (chỉ dùng để gán nhãn), chọn trên tập validation để cực đại macro-F1
    best_scale, best_f1 = (1.0, 1.0), -1.0
    for s_pre in np.arange(1.0, 14.01, 0.5):
        for s_dia in np.arange(0.6, 3.01, 0.1):
            f = f1_score(yval, (pv * np.array([1.0, s_pre, s_dia])).argmax(1), average="macro")
            if f > best_f1:
                best_f1, best_scale = f, (float(s_pre), float(s_dia))
    scale = np.array([1.0, best_scale[0], best_scale[1]], dtype="float32")
    pred_raw = probs.argmax(1)
    pred = (probs * scale).argmax(1)
    raw_metrics = dict(accuracy=float(accuracy_score(yte, pred_raw)), macro_f1=float(f1_score(yte, pred_raw, average="macro")),
                       recall_per_class=(confusion_matrix(yte, pred_raw).diagonal() / confusion_matrix(yte, pred_raw).sum(1)).tolist())
    cm = confusion_matrix(yte, pred)
    recalls = (cm.diagonal() / cm.sum(axis=1)).tolist()
    roc = {}
    for k, name in enumerate(["none", "pre", "diabetes"]):
        fpr, tpr, _ = roc_curve((yte == k).astype(int), probs[:, k])
        step = max(1, len(fpr) // 150)
        roc[name] = dict(fpr=[round(float(v), 4) for v in fpr[::step]], tpr=[round(float(v), 4) for v in tpr[::step]],
                         auc=round(float(roc_auc_score((yte == k).astype(int), probs[:, k])), 4))

    # độ quan trọng đặc trưng bằng hoán vị (mẫu 6.000 hồ sơ, đo giảm macro-F1)
    rng = np.random.default_rng(SEED)
    sub = rng.choice(len(Xte_n), 6000, replace=False)
    base_f1 = f1_score(yte[sub], numpy_forward(params, Xte_n[sub]).argmax(1), average="macro")
    importance = {}
    for j, name in enumerate(FEATURES):
        drops = []
        for _ in range(3):
            xp = Xte_n[sub].copy()
            xp[:, j] = rng.permutation(xp[:, j])
            drops.append(base_f1 - f1_score(yte[sub], numpy_forward(params, xp).argmax(1), average="macro"))
        importance[name] = round(float(np.mean(drops)), 4)

    os.makedirs(OUT, exist_ok=True)
    flat = {"mean": mean.astype("float32"), "std": std.astype("float32"), "n_layers": np.array(len(params)), "scale": scale}
    for i, (w, b) in enumerate(params):
        flat[f"w{i}"], flat[f"b{i}"] = w, b
    np.savez(os.path.join(OUT, "dl_model.npz"), **flat)
    metrics = dict(
        dataset="BRFSS 2015 (CDC) diabetes_012_health_indicators", rows=int(len(df)), train_rows=int(len(Xtr)), test_rows=int(len(Xte)),
        class_counts={"none": int(counts[0]), "pre": int(counts[1]), "diabetes": int(counts[2])}, features=FEATURES, seed=SEED,
        architecture="21 -> 128 -> 64 -> 32 -> 3 (ReLU, dropout 0,15/0,1)", epochs=14, batch_size=1024, optimizer="Adam lr=2e-3",
        train_seconds=seconds, accuracy=float(accuracy_score(yte, pred)), macro_f1=float(f1_score(yte, pred, average="macro")),
        auc_ovr=float(roc_auc_score(yte, probs, multi_class="ovr")), recall_per_class=recalls, confusion_matrix=cm.tolist(),
        baseline=base, history=history, best_epoch=best[2], decision_scale=[float(v) for v in scale], argmax_without_scale=raw_metrics,
        val_rows=int(len(yval)), class_weight_power=0.75, roc=roc, importance=importance,
        feature_stats={n: dict(mean=float(m), std=float(s)) for n, m, s in zip(FEATURES, mean, std)})
    with open(os.path.join(OUT, "dl_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=1)
    print("MLP", {k: metrics[k] for k in ("accuracy", "macro_f1", "auc_ovr")}, "recalls", [round(r, 3) for r in recalls])


if __name__ == "__main__":
    main()
