"""Bằng chứng tính đúng của bản NumPy tự viết, ghi ra results/tieuluan/verification.json.

1. Kiểm tra gradient: so gradient lan truyền ngược với sai phân trung tâm (số thực 64-bit).
2. Song trùng trước huấn luyện: cùng trọng số, cùng đầu vào → logit của NumPy, PyTorch, Keras.

Chạy: .venv\\Scripts\\python.exe -m scripts.tieuluan.verify_implementations
"""

import json
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np

from src.tieuluan.config import RESULTS_DIR
from src.tieuluan.experiment_models import build_models, predict_logits
from src.tieuluan.scratch.core import Sequential
from src.tieuluan.scratch.core import Module
from src.tieuluan.scratch.gradcheck import check_model
from src.tieuluan.scratch.layers import Conv2d, Dense, Flatten, MaxPool2d, ReLU
from src.tieuluan.scratch.losses import BCEWithLogitsLoss
from src.tieuluan.scratch.recurrent import GRU, LSTM, LastStep, SimpleRNN


def check_model_norm(model: Module, loss_fn, x: np.ndarray, y: np.ndarray, *, eps: float = 1e-6) -> float:
    """Sai số tương đối theo chuẩn của TOÀN BỘ vector gradient: ‖g_số − g_giải tích‖ / (‖g_số‖ + ‖g_giải tích‖).

    Thước đo này không bị khuếch đại bởi các thành phần gradient rất nhỏ (nơi sai số làm tròn của phép
    sai phân chiếm ưu thế), nên phản ánh tốt hơn độ đúng của cả phép lan truyền ngược.
    """
    model.train()
    model.zero_grad()
    loss_fn.forward(model.forward(x), y)
    model.backward(loss_fn.backward())
    analytic = np.concatenate([p.grad.reshape(-1).copy() for p in model.parameters()])
    numeric = []
    for param in model.parameters():
        flat = param.value.reshape(-1)
        for k in range(flat.size):
            old = flat[k]
            flat[k] = old + eps
            plus = loss_fn.forward(model.forward(x), y)
            flat[k] = old - eps
            minus = loss_fn.forward(model.forward(x), y)
            flat[k] = old
            numeric.append((plus - minus) / (2 * eps))
    numeric = np.array(numeric)
    return float(np.linalg.norm(numeric - analytic) / (np.linalg.norm(numeric) + np.linalg.norm(analytic)))


def gradient_checks():
    rng = np.random.default_rng(2026)
    f64 = np.float64
    cases = {
        "mlp": (Sequential(Dense(6, 5, rng=rng, dtype=f64), ReLU(), Dense(5, 1, rng=rng, dtype=f64)), (8, 6)),
        "cnn": (Sequential(Conv2d(1, 3, 3, rng=rng, dtype=f64), ReLU(), MaxPool2d(2), Flatten(),
                           Dense(3 * 3 * 3, 1, rng=rng, dtype=f64)), (4, 1, 8, 8)),
        "rnn": (Sequential(SimpleRNN(1, 4, rng=rng, dtype=f64), LastStep(), Dense(4, 1, rng=rng, dtype=f64)), (4, 6, 1)),
        "lstm": (Sequential(LSTM(1, 4, rng=rng, dtype=f64), LastStep(), Dense(4, 1, rng=rng, dtype=f64)), (4, 6, 1)),
        "gru": (Sequential(GRU(1, 4, rng=rng, dtype=f64), LastStep(), Dense(4, 1, rng=rng, dtype=f64)), (4, 6, 1)),
    }
    out = {}
    for name, (model, shape) in cases.items():
        x = rng.normal(size=shape)
        y = (rng.random((shape[0], 1)) > .5).astype(f64)
        report = check_model(model, BCEWithLogitsLoss(), x, y, eps=1e-6, samples_per_param=25, rng=rng)
        out[name] = {"max_relative_error": float(max(report.values())), "checked_tensors": len(report),
                     "norm_relative_error": check_model_norm(model, BCEWithLogitsLoss(), x, y)}
    return out


def parity_checks():
    rng = np.random.default_rng(7)
    shapes = {"mlp": (16, 23), "cnn4": (16, 1, 20, 20), "cnn8": (16, 1, 20, 20), "cnndeep": (16, 1, 20, 20),
              "rnn": (16, 20, 1), "lstm": (16, 20, 1), "gru": (16, 20, 1)}
    out = {}
    for kind, shape in shapes.items():
        models = build_models(kind, shape[1:], seed=11)
        x = rng.normal(size=shape).astype(np.float32)
        base = predict_logits(models["scratch"], "scratch", x)
        out[kind] = {fw: float(np.max(np.abs(predict_logits(models[fw], fw, x) - base))) for fw in ("pytorch", "keras")}
    return out


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    result = {"gradient_check": gradient_checks(), "forward_parity_max_abs_logit_diff": parity_checks()}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "verification.json").write_text(json.dumps(result, indent=1), encoding="utf8")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
