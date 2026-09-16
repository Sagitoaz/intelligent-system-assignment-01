"""Educational NumPy layers; no framework or library convolution is used."""

from __future__ import annotations

import numpy as np


class Conv1D:
    """Cross-correlation layer for NCL tensors with stride one."""

    def __init__(self, in_channels, out_channels, kernel_size=3, *, padding="same", rng=None, dtype=np.float32):
        if padding not in {"same", "valid"}:
            raise ValueError("padding must be 'same' or 'valid'")
        if kernel_size % 2 == 0 and padding == "same":
            raise ValueError("same padding requires an odd kernel size")
        self.padding = padding
        self.kernel_size = int(kernel_size)
        generator = rng or np.random.default_rng(42)
        scale = np.sqrt(2.0 / (in_channels * kernel_size))
        self.weight = generator.normal(0.0, scale, (out_channels, in_channels, kernel_size)).astype(dtype)
        self.bias = np.zeros(out_channels, dtype=dtype)
        self.grad_weight = np.zeros_like(self.weight)
        self.grad_bias = np.zeros_like(self.bias)
        self._cache = None

    @property
    def pad(self):
        return self.kernel_size // 2 if self.padding == "same" else 0

    def forward(self, x):
        x = np.asarray(x)
        if x.ndim != 3 or x.shape[1] != self.weight.shape[1]:
            raise ValueError("Conv1D input must have shape [B, C_in, L]")
        x_pad = np.pad(x, ((0, 0), (0, 0), (self.pad, self.pad)))
        windows = np.lib.stride_tricks.sliding_window_view(x_pad, self.kernel_size, axis=2)
        output = np.einsum("bclk,ock->bol", windows, self.weight) + self.bias[None, :, None]
        self._cache = (x.shape, x_pad, windows)
        return output

    def backward(self, grad_output):
        if self._cache is None:
            raise RuntimeError("forward must be called before backward")
        original_shape, x_pad, windows = self._cache
        grad_output = np.asarray(grad_output)
        self.grad_weight[...] = np.einsum("bol,bclk->ock", grad_output, windows)
        self.grad_bias[...] = grad_output.sum(axis=(0, 2))
        grad_pad = np.zeros_like(x_pad)
        # Each kernel tap contributes to a contiguous slice of the padded
        # input gradient.  Looping over K (three taps here), rather than over
        # every sequence position, keeps the explicit sliding-window math but
        # removes Python overhead that dominates long text sequences.
        length = grad_output.shape[2]
        for tap in range(self.kernel_size):
            grad_pad[:, :, tap : tap + length] += np.einsum(
                "bol,oc->bcl", grad_output, self.weight[:, :, tap]
            )
        return grad_pad[:, :, self.pad : -self.pad] if self.pad else grad_pad.reshape(original_shape)

    def parameters_and_grads(self):
        return [(self.weight, self.grad_weight), (self.bias, self.grad_bias)]


class ReLU:
    def forward(self, x):
        self._mask = np.asarray(x) > 0
        return np.maximum(x, 0)

    def backward(self, grad_output):
        return np.asarray(grad_output) * self._mask

    def parameters_and_grads(self):
        return []


class MaxPool1D:
    def __init__(self, kernel_size=2, stride=2):
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self._cache = None

    def forward(self, x):
        x = np.asarray(x)
        out_length = 1 + (x.shape[2] - self.kernel_size) // self.stride
        starts = np.arange(out_length) * self.stride
        windows = np.stack([x[:, :, start : start + self.kernel_size] for start in starts], axis=2)
        argmax = windows.argmax(axis=3)
        self._cache = (x.shape, starts, argmax)
        return windows.max(axis=3)

    def backward(self, grad_output):
        shape, starts, argmax = self._cache
        grad_input = np.zeros(shape, dtype=np.asarray(grad_output).dtype)
        # Scatter the upstream values to the recorded maxima.  ``np.add.at``
        # preserves the accumulation semantics for overlapping pool windows,
        # while avoiding a Python loop over every batch/channel element.
        batch = np.arange(shape[0])[:, None, None]
        channel = np.arange(shape[1])[None, :, None]
        position = starts[None, None, :] + argmax
        np.add.at(grad_input, (batch, channel, position), np.asarray(grad_output))
        return grad_input

    def downsample_mask(self, mask):
        mask = np.asarray(mask, dtype=bool)
        out_length = 1 + (mask.shape[1] - self.kernel_size) // self.stride
        return np.stack(
            [mask[:, i * self.stride : i * self.stride + self.kernel_size].any(axis=1) for i in range(out_length)],
            axis=1,
        )

    def parameters_and_grads(self):
        return []


class GlobalMaxPool1D:
    def forward(self, x, mask=None):
        x = np.asarray(x)
        valid = np.ones((x.shape[0], x.shape[2]), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
        masked = np.where(valid[:, None, :], x, -np.inf)
        self._argmax = masked.argmax(axis=2)
        self._valid_rows = valid.any(axis=1)
        output = masked.max(axis=2)
        output[~self._valid_rows] = 0.0
        self._input_shape = x.shape
        return output

    def backward(self, grad_output):
        grad_input = np.zeros(self._input_shape, dtype=np.asarray(grad_output).dtype)
        batch = np.arange(self._input_shape[0])[:, None]
        channel = np.arange(self._input_shape[1])[None, :]
        values = np.asarray(grad_output) * self._valid_rows[:, None]
        np.add.at(grad_input, (batch, channel, self._argmax), values)
        return grad_input

    def parameters_and_grads(self):
        return []


class Dense:
    def __init__(self, in_features, out_features, *, rng=None, dtype=np.float32):
        generator = rng or np.random.default_rng(42)
        scale = np.sqrt(2.0 / in_features)
        self.weight = generator.normal(0.0, scale, (out_features, in_features)).astype(dtype)
        self.bias = np.zeros(out_features, dtype=dtype)
        self.grad_weight = np.zeros_like(self.weight)
        self.grad_bias = np.zeros_like(self.bias)
        self._input = None

    def forward(self, x):
        self._input = np.asarray(x)
        return self._input @ self.weight.T + self.bias

    def backward(self, grad_output):
        grad_output = np.asarray(grad_output)
        self.grad_weight[...] = grad_output.T @ self._input
        self.grad_bias[...] = grad_output.sum(axis=0)
        return grad_output @ self.weight

    def parameters_and_grads(self):
        return [(self.weight, self.grad_weight), (self.bias, self.grad_bias)]


class Embedding:
    def __init__(self, vocabulary_size, embedding_dim, *, padding_idx=0, rng=None, dtype=np.float32):
        generator = rng or np.random.default_rng(42)
        self.weight = generator.normal(0.0, 0.05, (vocabulary_size, embedding_dim)).astype(dtype)
        self.grad_weight = np.zeros_like(self.weight)
        self.padding_idx = padding_idx
        self.enforce_padding()
        self._ids = None

    def enforce_padding(self):
        if self.padding_idx is not None:
            self.weight[self.padding_idx] = 0.0

    def forward(self, ids):
        self._ids = np.asarray(ids, dtype=np.int64)
        return self.weight[self._ids]

    def backward(self, grad_output):
        self.grad_weight.fill(0.0)
        np.add.at(self.grad_weight, self._ids, np.asarray(grad_output))
        if self.padding_idx is not None:
            self.grad_weight[self.padding_idx] = 0.0
        return None

    def parameters_and_grads(self):
        return [(self.weight, self.grad_weight)]
