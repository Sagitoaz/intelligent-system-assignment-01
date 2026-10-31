"""Agent Deep Learning: MLP 21→128→64→32→3 huấn luyện trên BRFSS 2015, suy luận bằng numpy (không cần torch khi chạy demo)."""
from __future__ import annotations

import json
import os

import numpy as np

from .data import FEATURES, normalize, to_vector

ART = os.path.join(os.path.dirname(__file__), "..", "artifacts")
CLASSES = ["none", "pre", "diabetes"]
VI_FEATURE = {"HighBP": "huyết áp cao", "HighChol": "mỡ máu cao", "CholCheck": "đã đo mỡ máu", "BMI": "chỉ số BMI", "Smoker": "hút thuốc",
              "Stroke": "từng đột quỵ", "HeartDiseaseorAttack": "bệnh tim", "PhysActivity": "có vận động", "Fruits": "ăn trái cây",
              "Veggies": "ăn rau", "HvyAlcoholConsump": "uống nhiều rượu bia", "AnyHealthcare": "có chăm sóc y tế", "NoDocbcCost": "ngại đi khám vì chi phí",
              "GenHlth": "tự đánh giá sức khỏe", "MentHlth": "ngày tinh thần kém", "PhysHlth": "ngày thể chất kém", "DiffWalk": "đi lại khó khăn",
              "Sex": "giới tính", "Age": "tuổi", "Education": "học vấn", "Income": "thu nhập"}
# yếu tố có thể thay đổi -> (chỉ số BRFSS, giá trị lành mạnh, nhãn hành động)
MODIFIABLE = [("BMI", 24.5, "đưa BMI về khoảng 24,5"), ("PhysActivity", 1.0, "duy trì vận động đều đặn"), ("Smoker", 0.0, "bỏ thuốc lá"),
              ("HvyAlcoholConsump", 0.0, "hạn chế rượu bia"), ("Fruits", 1.0, "ăn trái cây mỗi ngày"), ("Veggies", 1.0, "ăn rau mỗi ngày")]


class DLAgent:
    def __init__(self):
        z = np.load(os.path.join(ART, "dl_model.npz"))
        self.mean, self.std, self.scale = z["mean"], z["std"], z["scale"]
        self.layers = [(z[f"w{i}"], z[f"b{i}"]) for i in range(int(z["n_layers"]))]
        with open(os.path.join(ART, "dl_metrics.json"), encoding="utf-8") as f:
            self.metrics = json.load(f)
        self.pop_mean = np.array([self.metrics["feature_stats"][n]["mean"] for n in FEATURES], dtype="float32")

    def _forward(self, x: np.ndarray) -> np.ndarray:
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
        prof = normalize(profile)
        x = np.array([to_vector(prof)], dtype="float32")
        p = self._forward(x)[0]
        label = CLASSES[int((p * self.scale).argmax())]
        risk_index = float(self.index(p[None])[0])
        # tác động của từng yếu tố: thay bằng giá trị trung bình dân số
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
        return float(self.index(self._forward(self.pop_mean[None]))[0])
