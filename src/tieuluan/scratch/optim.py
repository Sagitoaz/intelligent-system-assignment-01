"""Thuật toán tối ưu: SGD (có quán tính) và Adam (Kingma & Ba, 2015); cắt chuẩn gradient."""

from __future__ import annotations

import numpy as np

from .core import Parameter


class SGD:
    def __init__(self, params: list[Parameter], lr: float = 0.01, momentum: float = 0.0):
        self.params, self.lr, self.momentum = params, lr, momentum
        self.velocity = [np.zeros_like(p.value) for p in params]

    def step(self) -> None:
        for p, v in zip(self.params, self.velocity):
            v *= self.momentum
            v += p.grad
            p.value -= self.lr * v


class Adam:
    """m ← β₁m + (1−β₁)g;  v ← β₂v + (1−β₂)g²;  θ ← θ − η·m̂/(√v̂ + ε)  (m̂, v̂ đã hiệu chỉnh độ lệch).

    ``weight_decay`` > 0 dùng dạng tách rời (decoupled, như AdamW) để thống nhất với Keras 3.
    """

    def __init__(self, params: list[Parameter], lr: float = 1e-3, betas=(0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 0.0):
        self.params, self.lr, self.eps, self.weight_decay = params, lr, eps, weight_decay
        self.beta1, self.beta2 = betas
        self.m = [np.zeros_like(p.value) for p in params]
        self.v = [np.zeros_like(p.value) for p in params]
        self.t = 0

    def step(self) -> None:
        self.t += 1
        bc1 = 1.0 - self.beta1 ** self.t
        bc2 = 1.0 - self.beta2 ** self.t
        for p, m, v in zip(self.params, self.m, self.v):
            g = p.grad
            m *= self.beta1
            m += (1.0 - self.beta1) * g
            v *= self.beta2
            v += (1.0 - self.beta2) * g * g
            if self.weight_decay:
                p.value -= self.lr * self.weight_decay * p.value
            p.value -= (self.lr * (m / bc1) / (np.sqrt(v / bc2) + self.eps)).astype(p.value.dtype)


def clip_grad_norm(params: list[Parameter], max_norm: float) -> float:
    """Cắt gradient theo chuẩn toàn cục (Pascanu et al., 2013) — chống bùng nổ gradient trong RNN."""
    total = float(np.sqrt(sum(float(np.sum(p.grad.astype(np.float64) ** 2)) for p in params)))
    coef = max_norm / (total + 1e-6)
    if coef < 1.0:
        for p in params:
            p.grad *= coef
    return total
