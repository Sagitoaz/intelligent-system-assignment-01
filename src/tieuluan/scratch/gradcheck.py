"""Kiểm tra gradient bằng sai phân trung tâm: (L(θ+ε) − L(θ−ε)) / 2ε so với gradient giải tích.

Đây là bằng chứng định lượng rằng phần lan truyền ngược tự cài đặt là đúng (sai số tương đối
cỡ 1e-7 hoặc nhỏ hơn với số thực 64-bit được xem là khớp).
"""

from __future__ import annotations

import numpy as np

from .core import Module


def relative_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b) / np.maximum(1e-8, np.abs(a) + np.abs(b))))


def check_model(model: Module, loss_fn, x: np.ndarray, y: np.ndarray, *, eps: float = 1e-6,
                samples_per_param: int = 12, rng=None) -> dict:
    """Trả về sai số tương đối lớn nhất cho từng tham số và cho đầu vào x."""
    rng = rng or np.random.default_rng(0)
    model.train()
    # Dropout phải tắt khi kiểm tra (mặt nạ ngẫu nhiên khác nhau giữa các lần gọi).
    model.zero_grad()
    out = model.forward(x)
    loss_fn.forward(out, y)
    dx = model.backward(loss_fn.backward())
    analytic = {id(p): p.grad.copy() for p in model.parameters()}

    def loss_at() -> float:
        return loss_fn.forward(model.forward(x), y)

    report = {}
    for p_index, param in enumerate(model.parameters()):
        flat = param.value.reshape(-1)
        picks = rng.choice(flat.size, size=min(samples_per_param, flat.size), replace=False)
        numeric, exact = [], []
        for k in picks:
            old = flat[k]
            flat[k] = old + eps; plus = loss_at()
            flat[k] = old - eps; minus = loss_at()
            flat[k] = old
            numeric.append((plus - minus) / (2 * eps))
            exact.append(analytic[id(param)].reshape(-1)[k])
        report[f"{p_index}:{param.name}"] = relative_error(np.array(exact), np.array(numeric))
    flat_x = x.reshape(-1)
    picks = rng.choice(flat_x.size, size=min(samples_per_param, flat_x.size), replace=False)
    numeric, exact = [], []
    for k in picks:
        old = flat_x[k]
        flat_x[k] = old + eps; plus = loss_at()
        flat_x[k] = old - eps; minus = loss_at()
        flat_x[k] = old
        numeric.append((plus - minus) / (2 * eps))
        exact.append(dx.reshape(-1)[k])
    report["input"] = relative_error(np.array(exact), np.array(numeric))
    return report
