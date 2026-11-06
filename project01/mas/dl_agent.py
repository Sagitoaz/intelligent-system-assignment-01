"""Agent Deep Learning: MLP 21→128→64→32→3 huấn luyện trên BRFSS 2015, suy luận bằng numpy (không cần torch khi chạy demo).

=====================  BẢN ĐỒ FILE (thầy hỏi "phần Deep Learning / mô hình dự đoán ở đâu" thì mở file này)  =====================
Thuộc agent: "Bác sĩ dữ liệu" (id="predictor", vai trò Deep Learning) trong hệ 7 agent của Project 01.

Mô hình: MLP (Multi-Layer Perceptron) 4 lớp tuyến tính, ReLU ở 3 lớp ẩn, softmax ở lớp ra 3 lớp (none / pre / diabetes).
  - Đầu vào: 21 chỉ số BRFSS (danh sách FEATURES trong data.py). Chuẩn hóa z-score bằng mean/std của tập huấn luyện.
  - Dự đoán nhãn: argmax(p · scale) với scale = [1,0; 1,5; 0,9] (lưu trong dl_model.npz) — nhân hệ số để tăng độ nhạy với lớp hiếm "pre".
  - Chỉ số nguy cơ (risk_index) = P(diabetes) + 0,5·P(pre): gộp hai lớp thành một con số liên tục để so sánh / giải thích.
  - Huấn luyện thật bằng PyTorch ở train/train_dl.py; file này CHỈ suy luận (inference) bằng numpy cho nhẹ.

Thành phần chính (tìm theo tên):
  - CLASSES, VI_FEATURE, MODIFIABLE : tên lớp, tên tiếng Việt của 21 chỉ số, danh sách yếu tố có thể thay đổi (BMI, vận động, hút thuốc, rượu, trái cây, rau).
  - class DLAgent
      .__init__()  nạp trọng số dl_model.npz và thống kê dl_metrics.json.
      ._forward()  lan truyền xuôi (forward pass) → xác suất 3 lớp.
      .index()     tính risk_index từ xác suất.
      .predict()   dự đoán cho một hồ sơ: xác suất, nhãn, risk_index, "drivers" (yếu tố nào đẩy nguy cơ) và "what_if" (đổi yếu tố nào giảm nguy cơ nhiều nhất).
      .population_risk_index()  chỉ số nguy cơ của "người trung bình" dân số, dùng làm mốc so sánh.

Dữ liệu đọc: artifacts/dl_model.npz (trọng số w0..w3, b0..b3, mean, std, scale), artifacts/dl_metrics.json (accuracy 0,752; macro-F1 0,461; AUC 0,782; thống kê đặc trưng).
Được gọi từ: mas/orchestrator.py (Clinic.dl = DLAgent(); hội chẩn gọi dl.predict), mas/whatif.py (chạy lại predict với hồ sơ đã sửa),
             mas/critic.py (kiểm tra độ nhạy ±5 kg), server.py (các API gián tiếp qua Clinic).
Giới hạn đã biết: recall lớp "pre" chỉ khoảng 17,5%; mô hình KHÔNG dùng đường huyết làm đầu vào (BRFSS không có) nên critic/luật phải đối chiếu thêm.
=============================================================================================================================
"""
from __future__ import annotations

import json
import os

import numpy as np

from .data import FEATURES, normalize, to_vector

ART = os.path.join(os.path.dirname(__file__), "..", "artifacts")
CLASSES = ["none", "pre", "diabetes"]  # thứ tự 3 đầu ra của MLP: không bệnh / tiền tiểu đường / tiểu đường
# Tên tiếng Việt của 21 chỉ số BRFSS (dùng khi giải thích "yếu tố nào ảnh hưởng"). Khóa phải trùng FEATURES trong data.py.
VI_FEATURE = {"HighBP": "huyết áp cao", "HighChol": "mỡ máu cao", "CholCheck": "đã đo mỡ máu", "BMI": "chỉ số BMI", "Smoker": "hút thuốc",
              "Stroke": "từng đột quỵ", "HeartDiseaseorAttack": "bệnh tim", "PhysActivity": "có vận động", "Fruits": "ăn trái cây",
              "Veggies": "ăn rau", "HvyAlcoholConsump": "uống nhiều rượu bia", "AnyHealthcare": "có chăm sóc y tế", "NoDocbcCost": "ngại đi khám vì chi phí",
              "GenHlth": "tự đánh giá sức khỏe", "MentHlth": "ngày tinh thần kém", "PhysHlth": "ngày thể chất kém", "DiffWalk": "đi lại khó khăn",
              "Sex": "giới tính", "Age": "tuổi", "Education": "học vấn", "Income": "thu nhập"}
# yếu tố có thể thay đổi -> (chỉ số BRFSS, giá trị lành mạnh, nhãn hành động)
MODIFIABLE = [("BMI", 24.5, "đưa BMI về khoảng 24,5"), ("PhysActivity", 1.0, "duy trì vận động đều đặn"), ("Smoker", 0.0, "bỏ thuốc lá"),
              ("HvyAlcoholConsump", 0.0, "hạn chế rượu bia"), ("Fruits", 1.0, "ăn trái cây mỗi ngày"), ("Veggies", 1.0, "ăn rau mỗi ngày")]


class DLAgent:
    """Bác sĩ dữ liệu: bọc mô hình MLP đã huấn luyện, cung cấp dự đoán + giải thích."""

    def __init__(self):
        """Nạp trọng số từ dl_model.npz: mean/std (chuẩn hóa), scale (hệ số quyết định), các cặp (w_i, b_i) của từng lớp;
        nạp dl_metrics.json; pop_mean = trung bình dân số của 21 chỉ số (dùng để "gỡ" từng yếu tố khi giải thích)."""
        z = np.load(os.path.join(ART, "dl_model.npz"))
        self.mean, self.std, self.scale = z["mean"], z["std"], z["scale"]
        self.layers = [(z[f"w{i}"], z[f"b{i}"]) for i in range(int(z["n_layers"]))]
        with open(os.path.join(ART, "dl_metrics.json"), encoding="utf-8") as f:
            self.metrics = json.load(f)
        self.pop_mean = np.array([self.metrics["feature_stats"][n]["mean"] for n in FEATURES], dtype="float32")

    def _forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass. Đầu vào x: mảng (B, 21) giá trị thô. Đầu ra: mảng (B, 3) xác suất [none, pre, diabetes].
        Các bước: chuẩn hóa z-score → với mỗi lớp h = h·W + b → ReLU ở mọi lớp trừ lớp cuối → softmax (trừ max trước khi exp để ổn định số học)."""
        h = (x - self.mean) / self.std
        for i, (w, b) in enumerate(self.layers):
            h = h @ w + b
            if i < len(self.layers) - 1:
                h = np.maximum(h, 0)
        h = h - h.max(axis=1, keepdims=True)
        e = np.exp(h)
        return e / e.sum(axis=1, keepdims=True)

    @staticmethod
    def index(p: np.ndarray) -> np.ndarray:
        return p[:, 2] + 0.5 * p[:, 1]  # chỉ số nguy cơ: tiểu đường + một nửa tiền tiểu đường

    def predict(self, profile: dict) -> dict:
        """Dự đoán cho một hồ sơ thân thiện (dict kiểu data.DEFAULT). Đầu ra: risk (xác suất 3 lớp), label (nhãn sau khi nhân scale),
        risk_index, bmi, drivers (tối đa 6 yếu tố ảnh hưởng nhất) và what_if (thay đổi nào giảm nguy cơ nhiều nhất, sắp theo mức giảm)."""
        prof = normalize(profile)
        x = np.array([to_vector(prof)], dtype="float32")
        p = self._forward(x)[0]
        label = CLASSES[int((p * self.scale).argmax())]
        risk_index = float(self.index(p[None])[0])
        # DRIVERS (giải thích kiểu "occlusion"): thay lần lượt từng yếu tố bằng giá trị trung bình dân số rồi đo risk_index giảm bao nhiêu.
        # effect = base − chỉ số sau khi thay; effect > 0 nghĩa là yếu tố đó đang ĐẨY nguy cơ lên so với người trung bình.
        base = float(self.index(self._forward(x))[0])
        drivers = []
        for j, name in enumerate(FEATURES):
            if name in ("CholCheck", "AnyHealthcare", "Education", "Income", "NoDocbcCost"):
                continue
            x2 = x.copy()
            x2[0, j] = self.pop_mean[j]
            delta = base - float(self.index(self._forward(x2))[0])
            drivers.append(dict(feature=name, vi=VI_FEATURE[name], value=float(x[0, j]), effect=round(delta, 4)))
        drivers.sort(key=lambda d: -abs(d["effect"]))
        # WHAT-IF: với mỗi yếu tố có thể thay đổi (MODIFIABLE) mà người này chưa ở mức lành mạnh, đặt về mức lành mạnh và đo mức nguy cơ giảm (gain).
        what_if = []
        for name, healthy, text in MODIFIABLE:
            j = FEATURES.index(name)
            if (name == "BMI" and x[0, j] <= healthy + 0.5) or (name != "BMI" and x[0, j] == healthy):
                continue
            x2 = x.copy()
            x2[0, j] = healthy
            gain = base - float(self.index(self._forward(x2))[0])
            what_if.append(dict(feature=name, action=text, from_value=float(x[0, j]), to_value=healthy, gain=round(gain, 4)))
        what_if.sort(key=lambda d: -d["gain"])
        return dict(risk=dict(none=float(p[0]), pre=float(p[1]), diabetes=float(p[2])), label=label, risk_index=risk_index,
                    bmi=float(x[0, 3]), drivers=drivers[:6], what_if=what_if)

    def population_risk_index(self) -> float:
        """risk_index của "người trung bình" (mọi chỉ số bằng trung bình dân số BRFSS) — mốc để so với từng bệnh nhân."""
        return float(self.index(self._forward(self.pop_mean[None]))[0])
