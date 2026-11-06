"""Huấn luyện agent Deep Learning của project01 trên BRFSS 2015 (253.680 hồ sơ, 21 chỉ số).

Nhãn 3 mức: 0 = không tiểu đường, 1 = tiền tiểu đường, 2 = tiểu đường.
Chạy: .venv\\Scripts\\python.exe project01/train/train_dl.py
Kết quả: project01/artifacts/dl_model.npz (trọng số cho suy luận bằng numpy) và dl_metrics.json.

======================================================================================================
BẢN ĐỒ FILE
======================================================================================================
Chức năng: huấn luyện agent "Bác sĩ dữ liệu" = mạng nơ-ron MLP (PyTorch) dự đoán 3 mức nguy cơ tiểu đường từ 21 chỉ số khảo sát
BRFSS 2015 của CDC. Chỉ lúc HUẤN LUYỆN mới cần torch; khi chạy demo, mas/dl_agent.py suy luận thuần numpy từ dl_model.npz.
Đọc : data/diabetes/diabetes_012_health_indicators_BRFSS2015.csv (253.680 dòng; nhãn Diabetes_012: 0 không, 1 tiền, 2 tiểu đường).
Ghi : artifacts/dl_model.npz (mean, std, trọng số w0..w3/b0..b3, n_layers, scale) và artifacts/dl_metrics.json (mọi số liệu báo cáo).
Hàm : build_model (kiến trúc), numpy_forward (suy luận numpy, phải khớp PyTorch), main (toàn bộ quy trình).
Quy trình trong main() (theo thứ tự, mỗi bước có comment bên dưới):
  1. Chia dữ liệu: 80% train / 20% test (stratify theo nhãn, seed 42); trong train lại tách 15% làm validation để chọn epoch và ngưỡng.
  2. Chuẩn hóa z-score bằng mean/std của TẬP TRAIN (không rò rỉ thông tin từ test).
  3. Baseline: Logistic Regression cân bằng lớp, để có mốc so sánh.
  4. MLP 21 -> 128 -> 64 -> 32 -> 3, ReLU, Dropout 0,15 và 0,1; loss CrossEntropy có trọng số lớp; Adam lr 2e-3; 14 epoch, batch 1024.
  5. Early stopping thủ công: giữ trọng số của epoch có AUC (one-vs-rest) validation cao nhất.
  6. Kiểm tra bản numpy cho kết quả trùng PyTorch (assert), rồi chọn decision_scale (hệ số nhân xác suất lớp "pre", "diabetes") trên validation.
  7. Đánh giá trên test: accuracy, macro-F1, AUC, recall từng lớp, ma trận nhầm lẫn, ROC từng lớp; độ quan trọng đặc trưng bằng hoán vị.
Kết quả đã lưu (test 50.736 dòng): accuracy 0,752; macro-F1 0,461; AUC 0,782; recall [không 0,797 / tiền 0,175 / tiểu đường 0,554].
Hạn chế cần nói thật khi bảo vệ: lớp "tiền tiểu đường" chỉ ~1,8% dữ liệu nên mô hình gần như không học được (recall 17,5%); mô hình
không dùng xét nghiệm đường huyết nên chỉ ước lượng nguy cơ từ lối sống, không chẩn đoán. Chi tiết độ tin cậy: train/eval_trust.py.
======================================================================================================
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
SEED = 42  # cố định hạt giống để chia dữ liệu và khởi tạo trọng số tái lập được


def build_model(n_in):
    """Kiến trúc MLP 3 lớp ẩn: n_in -> 128 -> 64 -> 32 -> 3 (logit cho 3 lớp; softmax được áp dụng bên ngoài/trong CrossEntropyLoss).

    ReLU làm phi tuyến; Dropout 0,15 và 0,1 ở hai lớp đầu để giảm overfitting (không dropout ở lớp cuối cho tầng nhỏ nhất).
    """
    return torch.nn.Sequential(
        torch.nn.Linear(n_in, 128), torch.nn.ReLU(), torch.nn.Dropout(0.15),
        torch.nn.Linear(128, 64), torch.nn.ReLU(), torch.nn.Dropout(0.1),
        torch.nn.Linear(64, 32), torch.nn.ReLU(),
        torch.nn.Linear(32, 3))


def numpy_forward(params, x):
    """Suy luận thuần numpy: params = [(W, b), ...] (W đã chuyển vị), x = ma trận (N, 21) đã chuẩn hóa. Trả xác suất softmax (N, 3).

    Giống hệt mas/dl_agent.py nên demo không cần cài torch. Dropout bị bỏ khi suy luận (đúng như model.eval()).
    """
    h = x
    for i, (w, b) in enumerate(params):
        h = h @ w + b
        if i < len(params) - 1:
            h = np.maximum(h, 0)  # ReLU cho mọi lớp trừ lớp cuối
    h = h - h.max(axis=1, keepdims=True)  # trừ max từng dòng để softmax ổn định số học (tránh tràn exp)
    e = np.exp(h)
    return e / e.sum(axis=1, keepdims=True)


def main():
    """Toàn bộ quy trình huấn luyện + đánh giá + lưu mô hình (xem BẢN ĐỒ FILE ở đầu file). Không có đầu vào/đầu ra trả về."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    df = pd.read_csv(CSV)
    y = df["Diabetes_012"].astype(int).values
    X = df[FEATURES].astype("float32").values
    counts = np.bincount(y)
    print("rows", len(df), "class counts", counts.tolist())
    # bước 1: chia 80/20 có phân tầng (stratify) để tỉ lệ 3 lớp ở train và test giống nhau (lớp "pre" rất hiếm)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    # bước 2: chuẩn hóa z-score; mean/std CHỈ tính trên train rồi áp cho test (tránh rò rỉ dữ liệu); +1e-6 tránh chia cho 0
    mean, std = Xtr.mean(axis=0), Xtr.std(axis=0) + 1e-6
    Xtr_n, Xte_n = (Xtr - mean) / std, (Xte - mean) / std
    # tách thêm validation (15% của train) để chọn epoch tốt nhất và decision_scale mà không "nhìn trộm" tập test
    Xfit, Xval, yfit, yval = train_test_split(Xtr_n, ytr, test_size=0.15, random_state=SEED, stratify=ytr)

    # bước 3 - baseline: hồi quy logistic (cùng dữ liệu, cùng chia). class_weight="balanced" để mốc so sánh không bị lớp "none" áp đảo.
    # Mục đích: chứng minh MLP có đáng giá hơn một mô hình tuyến tính đơn giản hay không.
    t0 = time.time()
    lr = LogisticRegression(max_iter=300, class_weight="balanced")
    lr.fit(Xtr_n, ytr)
    p_lr = lr.predict_proba(Xte_n)
    base = dict(name="Logistic Regression (cân bằng lớp)", accuracy=float(accuracy_score(yte, p_lr.argmax(1))),
                macro_f1=float(f1_score(yte, p_lr.argmax(1), average="macro")),
                auc_ovr=float(roc_auc_score(yte, p_lr, multi_class="ovr")), seconds=round(time.time() - t0, 1))
    print("baseline", base)

    # bước 4 - MLP 3 lớp ẩn
    model = build_model(len(FEATURES))
    # trọng số lớp = (N / (3 * số mẫu lớp))^0,75: lớp hiếm được phạt nặng hơn khi đoán sai. Mũ 0,75 (thay vì 1,0 = cân bằng hoàn toàn)
    # là mức "nửa vời" để tăng recall lớp hiếm mà không làm accuracy sụp quá nhiều.
    weights = torch.tensor((len(yfit) / (3 * np.bincount(yfit))) ** 0.75, dtype=torch.float32)  # cân bằng lớp (mũ 0,75)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    # Adam lr 2e-3; weight_decay 1e-5 = L2 nhẹ để chống overfitting
    opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-5)
    Xt, yt = torch.tensor(Xfit), torch.tensor(yfit)
    Xv = torch.tensor(Xval)
    Xtest_t = torch.tensor(Xte_n)
    history = []  # nhật ký mỗi epoch (train_loss, val_macro_f1, val_auc_ovr) -> dl_metrics.json, dùng vẽ đường cong học
    best = (-1.0, None)  # (AUC validation tốt nhất, bản sao trọng số, epoch) cho early stopping thủ công
    t0 = time.time()
    for epoch in range(14):
        model.train()  # bật Dropout
        perm = torch.randperm(len(Xt))  # xáo trộn dữ liệu mỗi epoch
        total = 0.0
        for i in range(0, len(Xt), 1024):  # mini-batch 1024 mẫu
            idx = perm[i:i + 1024]
            opt.zero_grad()
            loss = loss_fn(model(Xt[idx]), yt[idx])
            loss.backward()   # lan truyền ngược gradient
            opt.step()        # cập nhật trọng số bằng Adam
            total += float(loss) * len(idx)
        model.eval()  # tắt Dropout để đánh giá validation
        with torch.no_grad():
            pv = torch.softmax(model(Xv), 1).numpy()
        val_auc = float(roc_auc_score(yval, pv, multi_class="ovr"))
        history.append(dict(epoch=epoch + 1, train_loss=round(total / len(Xt), 4), val_macro_f1=round(float(f1_score(yval, pv.argmax(1), average="macro")), 4),
                            val_auc_ovr=round(val_auc, 4)))
        print(history[-1])
        # bước 5 - early stopping thủ công: chạy đủ 14 epoch nhưng chỉ nhớ epoch có AUC validation cao nhất (chọn theo AUC vì
        # accuracy sai lệch khi lớp mất cân bằng); sau đó nạp lại trọng số của epoch đó (best_epoch trong dl_metrics.json)
        if val_auc > best[0]:
            best = (val_auc, {k: v.clone() for k, v in model.state_dict().items()}, epoch + 1)
    model.load_state_dict(best[1])
    model.eval()
    with torch.no_grad():
        pv = torch.softmax(model(Xv), 1).numpy()
        pt = torch.softmax(model(Xtest_t), 1).numpy()
    seconds = round(time.time() - t0, 1)

    # bước 6a - trích trọng số ra numpy (W chuyển vị để dùng h @ W + b) và kiểm tra bản numpy khớp PyTorch (sai số < 1e-4)
    params = []
    for layer in model:
        if isinstance(layer, torch.nn.Linear):
            params.append((layer.weight.detach().numpy().T.astype("float32"), layer.bias.detach().numpy().astype("float32")))
    probs = numpy_forward(params, Xte_n)
    assert np.allclose(probs, pt, atol=1e-4)
    # bước 6b - hệ số điều chỉnh ngưỡng quyết định (chỉ dùng để gán nhãn), chọn trên tập validation để cực đại macro-F1.
    # Vì sao cần: argmax thuần gần như không bao giờ đoán "pre" (recall 0) do lớp này quá hiếm. Nhân xác suất lớp "pre" với s_pre
    # (1..14) và lớp "diabetes" với s_dia (0,6..3) rồi mới argmax sẽ cân bằng lại. Đổi lại: recall tăng nhưng cảnh báo sai cũng tăng
    # (accuracy giảm từ 0,780 xuống 0,752 - xem argmax_without_scale trong dl_metrics.json).
    best_scale, best_f1 = (1.0, 1.0), -1.0
    for s_pre in np.arange(1.0, 14.01, 0.5):
        for s_dia in np.arange(0.6, 3.01, 0.1):
            f = f1_score(yval, (pv * np.array([1.0, s_pre, s_dia])).argmax(1), average="macro")
            if f > best_f1:
                best_f1, best_scale = f, (float(s_pre), float(s_dia))
    scale = np.array([1.0, best_scale[0], best_scale[1]], dtype="float32")
    # bước 7 - đánh giá trên TEST (chỉ chạm tập test một lần ở bước này)
    pred_raw = probs.argmax(1)            # nhãn nếu dùng argmax thuần (để đối chiếu)
    pred = (probs * scale).argmax(1)      # nhãn chính thức sau điều chỉnh ngưỡng
    raw_metrics = dict(accuracy=float(accuracy_score(yte, pred_raw)), macro_f1=float(f1_score(yte, pred_raw, average="macro")),
                       recall_per_class=(confusion_matrix(yte, pred_raw).diagonal() / confusion_matrix(yte, pred_raw).sum(1)).tolist())
    cm = confusion_matrix(yte, pred)
    recalls = (cm.diagonal() / cm.sum(axis=1)).tolist()
    roc = {}  # ROC one-vs-rest cho từng lớp; thưa bớt về ~150 điểm để file JSON nhẹ
    for k, name in enumerate(["none", "pre", "diabetes"]):
        fpr, tpr, _ = roc_curve((yte == k).astype(int), probs[:, k])
        step = max(1, len(fpr) // 150)
        roc[name] = dict(fpr=[round(float(v), 4) for v in fpr[::step]], tpr=[round(float(v), 4) for v in tpr[::step]],
                         auc=round(float(roc_auc_score((yte == k).astype(int), probs[:, k])), 4))

    # độ quan trọng đặc trưng bằng hoán vị (mẫu 6.000 hồ sơ, đo giảm macro-F1): xáo trộn ngẫu nhiên MỘT cột rồi xem macro-F1 giảm bao nhiêu;
    # giảm càng nhiều = mô hình càng dựa vào cột đó. Lặp 3 lần lấy trung bình. Giá trị âm/gần 0 = cột gần như không đóng góp.
    # Chỉ mô tả mô hình này, KHÔNG suy ra quan hệ nhân quả.
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

    # lưu mô hình: mean/std (để chuẩn hóa hồ sơ mới), trọng số từng lớp, scale (decision_scale). mas/dl_agent.py đọc đúng các khóa này
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
