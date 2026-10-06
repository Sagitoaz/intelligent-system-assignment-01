"""Hàm mất mát và đạo hàm của chúng theo đầu ra mô hình."""

from __future__ import annotations

import numpy as np


def _softplus(x):
    return np.logaddexp(0.0, x)  # log(1 + e^x) ổn định số học


class BCEWithLogitsLoss:
    """Entropy chéo nhị phân trên logit z (không qua sigmoid trước):

    L = mean( w₊·y·log(1+e^{−z}) + (1−y)·log(1+e^{z}) ),  ∂L/∂z = ((1−y)·σ(z) − w₊·y·(1−σ(z))) / N.
    ``pos_weight`` (w₊) tăng trọng số lớp dương khi dữ liệu mất cân bằng — như PyTorch.
    """

    def __init__(self, pos_weight: float = 1.0):
        self.pos_weight = float(pos_weight)

    def forward(self, logits: np.ndarray, target: np.ndarray) -> float:
        z = logits.reshape(-1).astype(np.float64)
        y = target.reshape(-1).astype(np.float64)
        self._cache = (z, y, logits.shape, logits.dtype)
        loss = self.pos_weight * y * _softplus(-z) + (1.0 - y) * _softplus(z)
        return float(loss.mean())

    def backward(self) -> np.ndarray:
        z, y, shape, dtype = self._cache
        s = 0.5 * (1.0 + np.tanh(0.5 * z))
        grad = ((1.0 - y) * s - self.pos_weight * y * (1.0 - s)) / z.size
        return grad.reshape(shape).astype(dtype)


class MSELoss:
    """Sai số bình phương trung bình: L = mean((ŷ − y)²),  ∂L/∂ŷ = 2(ŷ − y)/N."""

    def forward(self, pred: np.ndarray, target: np.ndarray) -> float:
        diff = pred.reshape(-1).astype(np.float64) - target.reshape(-1).astype(np.float64)
        self._cache = (diff, pred.shape, pred.dtype)
        return float(np.mean(diff ** 2))

    def backward(self) -> np.ndarray:
        diff, shape, dtype = self._cache
        return (2.0 * diff / diff.size).reshape(shape).astype(dtype)
