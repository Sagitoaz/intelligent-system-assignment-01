"""Các lớp cơ bản: Dense, hàm kích hoạt, Dropout, BatchNorm, Conv2d (im2col), MaxPool2d.

Quy ước kích thước (giống PyTorch): ảnh có dạng [N, C, H, W]; trọng số Dense có dạng
[out, in]; trọng số Conv2d có dạng [F, C, KH, KW]. Gradient được *cộng dồn* vào
``Parameter.grad`` nên cần gọi ``zero_grad()`` trước mỗi bước cập nhật.
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import as_strided

from .core import Module, Parameter, glorot_uniform


def _pair(value) -> tuple[int, int]:
    return (int(value), int(value)) if np.isscalar(value) else (int(value[0]), int(value[1]))


class Dense(Module):
    """Lớp kết nối đầy đủ: y = x·Wᵀ + b."""

    def __init__(self, in_features: int, out_features: int, *, rng=None, dtype=np.float32):
        rng = rng or np.random.default_rng(0)
        self.weight = Parameter(glorot_uniform(rng, in_features, out_features, (out_features, in_features), dtype), "weight")
        self.bias = Parameter(np.zeros(out_features, dtype=dtype), "bias")
        self._x = None

    def forward(self, x):
        self._x = x
        return x @ self.weight.value.T + self.bias.value

    def backward(self, grad):
        self.weight.grad += grad.T @ self._x
        self.bias.grad += grad.sum(axis=0)
        return grad @ self.weight.value


class ReLU(Module):
    def forward(self, x):
        self._mask = x > 0
        return np.where(self._mask, x, 0).astype(x.dtype, copy=False)

    def backward(self, grad):
        return grad * self._mask


class LeakyReLU(Module):
    """f(x) = x nếu x > 0, ngược lại α·x (Jiang, Kelly & Xiu 2023 dùng α = 0,01)."""

    def __init__(self, negative_slope: float = 0.01):
        self.alpha = negative_slope

    def forward(self, x):
        self._mask = x > 0
        return np.where(self._mask, x, self.alpha * x).astype(x.dtype, copy=False)

    def backward(self, grad):
        return grad * np.where(self._mask, 1.0, self.alpha).astype(grad.dtype, copy=False)


class Sigmoid(Module):
    def forward(self, x):
        self._y = 0.5 * (1.0 + np.tanh(0.5 * x))  # dạng ổn định số học của 1/(1+e^-x)
        return self._y

    def backward(self, grad):
        return grad * self._y * (1.0 - self._y)


class Tanh(Module):
    def forward(self, x):
        self._y = np.tanh(x)
        return self._y

    def backward(self, grad):
        return grad * (1.0 - self._y ** 2)


class Dropout(Module):
    """Dropout "đảo ngược": khi huấn luyện giữ lại mỗi nơ-ron với xác suất 1-p rồi chia cho 1-p."""

    def __init__(self, p: float = 0.5, *, rng=None):
        self.p = float(p)
        self.rng = rng or np.random.default_rng(0)
        self._mask = None

    def forward(self, x):
        if not self.training or self.p == 0.0:
            self._mask = None
            return x
        keep = 1.0 - self.p
        self._mask = (self.rng.random(x.shape) < keep).astype(x.dtype) / keep
        return x * self._mask

    def backward(self, grad):
        return grad if self._mask is None else grad * self._mask


class Flatten(Module):
    def forward(self, x):
        self._shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, grad):
        return grad.reshape(self._shape)


class _BatchNorm(Module):
    """Chuẩn hóa theo lô (Ioffe & Szegedy, 2015). ``momentum`` theo quy ước PyTorch (0,1)."""

    axes: tuple[int, ...] = (0,)

    def __init__(self, num_features: int, *, eps: float = 1e-5, momentum: float = 0.1, dtype=np.float32):
        self.gamma = Parameter(np.ones(num_features, dtype=dtype), "gamma")
        self.beta = Parameter(np.zeros(num_features, dtype=dtype), "beta")
        self.running_mean = np.zeros(num_features, dtype=dtype)
        self.running_var = np.ones(num_features, dtype=dtype)
        self.eps, self.momentum = eps, momentum

    def _shape(self, v, ndim):
        return v.reshape((1, -1) + (1,) * (ndim - 2))

    def forward(self, x):
        if self.training:
            mean = x.mean(axis=self.axes)
            var = x.var(axis=self.axes)
            count = x.size // x.shape[1]
            unbiased = var * count / max(count - 1, 1)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * unbiased
        else:
            mean, var = self.running_mean, self.running_var
        inv_std = 1.0 / np.sqrt(var + self.eps)
        x_hat = (x - self._shape(mean, x.ndim)) * self._shape(inv_std, x.ndim)
        self._cache = (x_hat, inv_std)
        return self._shape(self.gamma.value, x.ndim) * x_hat + self._shape(self.beta.value, x.ndim)

    def backward(self, grad):
        x_hat, inv_std = self._cache
        self.gamma.grad += (grad * x_hat).sum(axis=self.axes)
        self.beta.grad += grad.sum(axis=self.axes)
        g = grad * self._shape(self.gamma.value, grad.ndim)
        if not self.training:  # chế độ suy luận: phép biến đổi tuyến tính cố định
            return g * self._shape(inv_std, grad.ndim)
        m = grad.size // grad.shape[1]
        sum_g = g.sum(axis=self.axes, keepdims=True)
        sum_gx = (g * x_hat).sum(axis=self.axes, keepdims=True)
        return (self._shape(inv_std, grad.ndim) / m) * (m * g - sum_g - x_hat * sum_gx)

    def buffers(self):
        return {"running_mean": self.running_mean, "running_var": self.running_var}


class BatchNorm1d(_BatchNorm):
    axes = (0,)


class BatchNorm2d(_BatchNorm):
    axes = (0, 2, 3)


class Conv2d(Module):
    """Tích chập 2 chiều cài bằng kỹ thuật im2col.

    Mỗi "cửa sổ" KH×KW của ảnh được duỗi thành một hàng của ma trận ``cols``; khi đó
    phép tích chập trở thành một phép nhân ma trận lớn (cols · Wᵀ) — cách các thư viện
    như Caffe từng dùng. Hỗ trợ stride, padding và dilation giống ``torch.nn.Conv2d``.
    """

    def __init__(self, in_channels, out_channels, kernel_size, *, stride=1, padding=0, dilation=1,
                 rng=None, dtype=np.float32):
        rng = rng or np.random.default_rng(0)
        self.kernel = _pair(kernel_size)
        self.stride, self.padding, self.dilation = _pair(stride), _pair(padding), _pair(dilation)
        kh, kw = self.kernel
        fan_in, fan_out = in_channels * kh * kw, out_channels * kh * kw
        self.weight = Parameter(glorot_uniform(rng, fan_in, fan_out, (out_channels, in_channels, kh, kw), dtype), "weight")
        self.bias = Parameter(np.zeros(out_channels, dtype=dtype), "bias")

    def output_shape(self, h: int, w: int) -> tuple[int, int]:
        (kh, kw), (sh, sw), (ph, pw), (dh, dw) = self.kernel, self.stride, self.padding, self.dilation
        return (h + 2 * ph - dh * (kh - 1) - 1) // sh + 1, (w + 2 * pw - dw * (kw - 1) - 1) // sw + 1

    def forward(self, x):
        n, c, h, w = x.shape
        (kh, kw), (sh, sw), (ph, pw), (dh, dw) = self.kernel, self.stride, self.padding, self.dilation
        xp = np.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw))) if (ph or pw) else np.ascontiguousarray(x)
        oh, ow = self.output_shape(h, w)
        s_n, s_c, s_h, s_w = xp.strides
        windows = as_strided(xp, shape=(n, oh, ow, c, kh, kw),
                             strides=(s_n, s_h * sh, s_w * sw, s_c, s_h * dh, s_w * dw), writeable=False)
        cols = windows.reshape(n * oh * ow, c * kh * kw)
        out = cols @ self.weight.value.reshape(self.weight.value.shape[0], -1).T + self.bias.value
        self._cache = (cols, x.shape, xp.shape, oh, ow)
        return np.ascontiguousarray(out.reshape(n, oh, ow, -1).transpose(0, 3, 1, 2))

    def backward(self, grad):
        cols, (n, c, h, w), xp_shape, oh, ow = self._cache
        (kh, kw), (sh, sw), (ph, pw), (dh, dw) = self.kernel, self.stride, self.padding, self.dilation
        f = self.weight.value.shape[0]
        g = grad.transpose(0, 2, 3, 1).reshape(n * oh * ow, f)
        self.weight.grad += (g.T @ cols).reshape(self.weight.value.shape)
        self.bias.grad += g.sum(axis=0)
        dcols = (g @ self.weight.value.reshape(f, -1)).reshape(n, oh, ow, c, kh, kw)
        dxp = np.zeros(xp_shape, dtype=grad.dtype)
        for i in range(kh):  # "col2im": trả gradient của từng vị trí trong nhân về đúng điểm ảnh
            for j in range(kw):
                hs, ws = i * dh, j * dw
                dxp[:, :, hs:hs + sh * (oh - 1) + 1:sh, ws:ws + sw * (ow - 1) + 1:sw] += \
                    dcols[:, :, :, :, i, j].transpose(0, 3, 1, 2)
        return dxp[:, :, ph:ph + h, pw:pw + w]


class MaxPool2d(Module):
    """Gộp cực đại. Khi có nhiều giá trị bằng nhau, chọn phần tử xuất hiện đầu tiên (giống PyTorch)."""

    def __init__(self, kernel_size, stride=None):
        self.kernel = _pair(kernel_size)
        self.stride = _pair(stride) if stride is not None else self.kernel

    def forward(self, x):
        x = np.ascontiguousarray(x)
        n, c, h, w = x.shape
        (kh, kw), (sh, sw) = self.kernel, self.stride
        oh, ow = (h - kh) // sh + 1, (w - kw) // sw + 1
        s_n, s_c, s_h, s_w = x.strides
        windows = as_strided(x, shape=(n, c, oh, ow, kh, kw),
                             strides=(s_n, s_c, s_h * sh, s_w * sw, s_h, s_w), writeable=False)
        flat = windows.reshape(n, c, oh, ow, kh * kw)
        idx = flat.argmax(axis=-1)
        self._cache = (x.shape, idx, oh, ow)
        return np.take_along_axis(flat, idx[..., None], axis=-1)[..., 0]

    def backward(self, grad):
        shape, idx, oh, ow = self._cache
        (kh, kw), (sh, sw) = self.kernel, self.stride
        n, c = shape[:2]
        rows = np.arange(oh)[None, None, :, None] * sh + idx // kw
        cols = np.arange(ow)[None, None, None, :] * sw + idx % kw
        nn_ = np.arange(n)[:, None, None, None]
        cc = np.arange(c)[None, :, None, None]
        dx = np.zeros(shape, dtype=grad.dtype)
        if (sh, sw) == (kh, kw):  # cửa sổ không chồng lấn → mỗi vị trí nhận tối đa một gradient
            dx[nn_, cc, rows, cols] = grad
        else:
            np.add.at(dx, (nn_, cc, rows, cols), grad)
        return dx
