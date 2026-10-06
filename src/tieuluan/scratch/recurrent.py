"""Mạng hồi quy: SimpleRNN (Elman), LSTM, GRU với lan truyền ngược theo thời gian (BPTT).

Đầu vào dạng [N, T, D] (batch_first), đầu ra là toàn bộ chuỗi trạng thái ẩn [N, T, H];
dùng ``LastStep`` để lấy trạng thái cuối cho bài toán "nhiều-đến-một" (many-to-one).
Bố cục trọng số trùng PyTorch: weight_ih [G·H, D], weight_hh [G·H, H], hai vector bias,
thứ tự cổng LSTM (i, f, g, o) và GRU (r, z, n). Khởi tạo theo mặc định của Keras:
Glorot đều cho trọng số đầu vào, trực giao cho trọng số hồi quy, bias 0 và bias cổng quên = 1.
"""

from __future__ import annotations

import numpy as np

from .core import Module, Parameter, glorot_uniform, orthogonal


def _sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))


class _Recurrent(Module):
    gates = 1

    def __init__(self, input_size: int, hidden_size: int, *, rng=None, dtype=np.float32, forget_bias: float = 0.0):
        rng = rng or np.random.default_rng(0)
        g, h = self.gates, hidden_size
        self.hidden_size = h
        self.weight_ih = Parameter(glorot_uniform(rng, input_size, g * h, (g * h, input_size), dtype), "weight_ih")
        self.weight_hh = Parameter(orthogonal(rng, g * h, h, dtype), "weight_hh")
        bias = np.zeros(g * h, dtype=dtype)
        if forget_bias:
            bias[h:2 * h] = forget_bias
        self.bias_ih = Parameter(bias, "bias_ih")
        self.bias_hh = Parameter(np.zeros(g * h, dtype=dtype), "bias_hh")


class SimpleRNN(_Recurrent):
    """h_t = tanh(W_ih·x_t + b_ih + W_hh·h_{t-1} + b_hh)."""

    gates = 1

    def forward(self, x):
        n, t_len, _ = x.shape
        h = np.zeros((n, self.hidden_size), dtype=x.dtype)
        xw = x @ self.weight_ih.value.T + self.bias_ih.value
        hs = np.empty((n, t_len + 1, self.hidden_size), dtype=x.dtype)
        hs[:, 0] = h
        for t in range(t_len):
            h = np.tanh(xw[:, t] + h @ self.weight_hh.value.T + self.bias_hh.value)
            hs[:, t + 1] = h
        self._cache = (x, hs)
        return hs[:, 1:]

    def backward(self, grad):
        x, hs = self._cache
        n, t_len, d = x.shape
        dxw = np.empty((n, t_len, self.hidden_size), dtype=grad.dtype)
        dh_next = np.zeros((n, self.hidden_size), dtype=grad.dtype)
        w_hh = self.weight_hh.value
        for t in reversed(range(t_len)):
            da = (grad[:, t] + dh_next) * (1.0 - hs[:, t + 1] ** 2)
            self.weight_hh.grad += da.T @ hs[:, t]
            dxw[:, t] = da
            dh_next = da @ w_hh
        flat = dxw.reshape(-1, self.hidden_size)
        self.weight_ih.grad += flat.T @ x.reshape(-1, d)
        self.bias_ih.grad += flat.sum(axis=0)
        self.bias_hh.grad += flat.sum(axis=0)
        return dxw @ self.weight_ih.value


class LSTM(_Recurrent):
    """LSTM (Hochreiter & Schmidhuber, 1997; cổng quên của Gers et al., 2000).

    i = σ(·), f = σ(·), g = tanh(·), o = σ(·);  c_t = f⊙c_{t-1} + i⊙g;  h_t = o⊙tanh(c_t).
    """

    gates = 4

    def __init__(self, input_size, hidden_size, *, rng=None, dtype=np.float32, forget_bias: float = 1.0):
        super().__init__(input_size, hidden_size, rng=rng, dtype=dtype, forget_bias=forget_bias)

    def forward(self, x):
        n, t_len, _ = x.shape
        hd = self.hidden_size
        xw = x @ self.weight_ih.value.T + self.bias_ih.value
        h = np.zeros((n, hd), dtype=x.dtype)
        c = np.zeros((n, hd), dtype=x.dtype)
        hs = np.empty((n, t_len + 1, hd), dtype=x.dtype); hs[:, 0] = h
        cs = np.empty((n, t_len + 1, hd), dtype=x.dtype); cs[:, 0] = c
        acts = np.empty((n, t_len, 4 * hd), dtype=x.dtype)
        tcs = np.empty((n, t_len, hd), dtype=x.dtype)
        for t in range(t_len):
            z = xw[:, t] + h @ self.weight_hh.value.T + self.bias_hh.value
            i = _sigmoid(z[:, :hd]); f = _sigmoid(z[:, hd:2 * hd])
            g = np.tanh(z[:, 2 * hd:3 * hd]); o = _sigmoid(z[:, 3 * hd:])
            c = f * c + i * g
            tc = np.tanh(c)
            h = o * tc
            acts[:, t] = np.concatenate([i, f, g, o], axis=1)
            tcs[:, t], hs[:, t + 1], cs[:, t + 1] = tc, h, c
        self._cache = (x, hs, cs, acts, tcs)
        return hs[:, 1:]

    def backward(self, grad):
        x, hs, cs, acts, tcs = self._cache
        n, t_len, d = x.shape
        hd = self.hidden_size
        dz_all = np.empty((n, t_len, 4 * hd), dtype=grad.dtype)
        dh_next = np.zeros((n, hd), dtype=grad.dtype)
        dc_next = np.zeros((n, hd), dtype=grad.dtype)
        w_hh = self.weight_hh.value
        for t in reversed(range(t_len)):
            i, f, g, o = (acts[:, t, k * hd:(k + 1) * hd] for k in range(4))
            tc = tcs[:, t]
            dh = grad[:, t] + dh_next
            do = dh * tc
            dc = dc_next + dh * o * (1.0 - tc ** 2)
            di, df, dg = dc * g, dc * cs[:, t], dc * i
            dc_next = dc * f
            dz = np.concatenate([di * i * (1 - i), df * f * (1 - f), dg * (1 - g ** 2), do * o * (1 - o)], axis=1)
            self.weight_hh.grad += dz.T @ hs[:, t]
            dz_all[:, t] = dz
            dh_next = dz @ w_hh
        flat = dz_all.reshape(-1, 4 * hd)
        self.weight_ih.grad += flat.T @ x.reshape(-1, d)
        self.bias_ih.grad += flat.sum(axis=0)
        self.bias_hh.grad += flat.sum(axis=0)
        return dz_all @ self.weight_ih.value


class GRU(_Recurrent):
    """GRU (Cho et al., 2014) theo dạng của PyTorch/cuDNN (Keras ``reset_after=True``).

    r = σ(·), z = σ(·), n = tanh(W_in·x + b_in + r⊙(W_hn·h + b_hn)),  h_t = (1−z)⊙n + z⊙h_{t-1}.
    """

    gates = 3

    def forward(self, x):
        n, t_len, _ = x.shape
        hd = self.hidden_size
        gi = x @ self.weight_ih.value.T + self.bias_ih.value
        h = np.zeros((n, hd), dtype=x.dtype)
        hs = np.empty((n, t_len + 1, hd), dtype=x.dtype); hs[:, 0] = h
        cache = np.empty((n, t_len, 4 * hd), dtype=x.dtype)  # r, z, n, gh_n
        for t in range(t_len):
            gh = h @ self.weight_hh.value.T + self.bias_hh.value
            r = _sigmoid(gi[:, t, :hd] + gh[:, :hd])
            z = _sigmoid(gi[:, t, hd:2 * hd] + gh[:, hd:2 * hd])
            nn_ = np.tanh(gi[:, t, 2 * hd:] + r * gh[:, 2 * hd:])
            h = (1.0 - z) * nn_ + z * h
            hs[:, t + 1] = h
            cache[:, t] = np.concatenate([r, z, nn_, gh[:, 2 * hd:]], axis=1)
        self._cache = (x, hs, cache)
        return hs[:, 1:]

    def backward(self, grad):
        x, hs, cache = self._cache
        n, t_len, d = x.shape
        hd = self.hidden_size
        dgi_all = np.empty((n, t_len, 3 * hd), dtype=grad.dtype)
        dh_next = np.zeros((n, hd), dtype=grad.dtype)
        w_hh = self.weight_hh.value
        for t in reversed(range(t_len)):
            r, z, nn_, ghn = (cache[:, t, k * hd:(k + 1) * hd] for k in range(4))
            h_prev = hs[:, t]
            dh = grad[:, t] + dh_next
            dan = dh * (1.0 - z) * (1.0 - nn_ ** 2)
            daz = dh * (h_prev - nn_) * z * (1.0 - z)
            dar = dan * ghn * r * (1.0 - r)
            dgh = np.concatenate([dar, daz, dan * r], axis=1)
            dgi_all[:, t] = np.concatenate([dar, daz, dan], axis=1)
            self.weight_hh.grad += dgh.T @ h_prev
            self.bias_hh.grad += dgh.sum(axis=0)
            dh_next = dh * z + dgh @ w_hh
        flat = dgi_all.reshape(-1, 3 * hd)
        self.weight_ih.grad += flat.T @ x.reshape(-1, d)
        self.bias_ih.grad += flat.sum(axis=0)
        return dgi_all @ self.weight_ih.value


class LastStep(Module):
    """Lấy trạng thái ẩn ở bước thời gian cuối: [N, T, H] → [N, H]."""

    def forward(self, x):
        self._shape = x.shape
        return x[:, -1]

    def backward(self, grad):
        out = np.zeros(self._shape, dtype=grad.dtype)
        out[:, -1] = grad
        return out
