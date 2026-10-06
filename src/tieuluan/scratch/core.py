"""Khối cơ sở: tham số, lớp (module) và mô hình tuần tự."""

from __future__ import annotations

from pathlib import Path

import numpy as np


class Parameter:
    """Một mảng trọng số cùng gradient của nó (``grad`` có cùng kích thước với ``value``)."""

    def __init__(self, value: np.ndarray, name: str = ""):
        self.value = value
        self.grad = np.zeros_like(value)
        self.name = name

    def zero_grad(self) -> None:
        self.grad[...] = 0.0


class Module:
    """Lớp cơ sở. Lớp con cài ``forward`` và ``backward``; ``training`` bật/tắt Dropout, BatchNorm."""

    training: bool = True

    def forward(self, x: np.ndarray) -> np.ndarray:  # pragma: no cover - lớp trừu tượng
        raise NotImplementedError

    def backward(self, grad: np.ndarray) -> np.ndarray:  # pragma: no cover - lớp trừu tượng
        raise NotImplementedError

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def parameters(self) -> list[Parameter]:
        params = []
        for value in self.__dict__.values():
            if isinstance(value, Parameter):
                params.append(value)
            elif isinstance(value, Module):
                params.extend(value.parameters())
            elif isinstance(value, (list, tuple)):
                params.extend(p for item in value if isinstance(item, Module) for p in item.parameters())
        return params

    def buffers(self) -> dict[str, np.ndarray]:
        """Trạng thái không học bằng gradient (ví dụ thống kê chạy của BatchNorm)."""
        return {}

    def train(self, mode: bool = True) -> "Module":
        self.training = mode
        for value in self.__dict__.values():
            if isinstance(value, Module):
                value.train(mode)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, Module):
                        item.train(mode)
        return self

    def eval(self) -> "Module":
        return self.train(False)

    def zero_grad(self) -> None:
        for param in self.parameters():
            param.zero_grad()

    def num_parameters(self) -> int:
        return int(sum(p.value.size for p in self.parameters()))


class Sequential(Module):
    """Xếp chồng các lớp: đầu ra lớp trước là đầu vào lớp sau; lan truyền ngược theo chiều ngược lại."""

    def __init__(self, *layers: Module):
        self.layers = list(layers)

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad: np.ndarray) -> np.ndarray:
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def state_dict(self) -> dict[str, np.ndarray]:
        state = {}
        for index, layer in enumerate(self.layers):
            for p_index, param in enumerate(layer.parameters()):
                state[f"{index}.{param.name or p_index}"] = param.value.copy()
            for b_name, buf in layer.buffers().items():
                state[f"{index}.{b_name}"] = buf.copy()
        return state

    def load_state_dict(self, state: dict[str, np.ndarray]) -> None:
        for index, layer in enumerate(self.layers):
            for p_index, param in enumerate(layer.parameters()):
                param.value[...] = state[f"{index}.{param.name or p_index}"]
            for b_name, buf in layer.buffers().items():
                buf[...] = state[f"{index}.{b_name}"]

    def save(self, path: str | Path) -> None:
        np.savez_compressed(path, **self.state_dict())

    def load(self, path: str | Path) -> "Sequential":
        with np.load(path) as data:
            self.load_state_dict({key: data[key] for key in data.files})
        return self


def glorot_uniform(rng: np.random.Generator, fan_in: int, fan_out: int, shape, dtype) -> np.ndarray:
    """Khởi tạo Glorot/Xavier đều: U(-a, a), a = sqrt(6 / (fan_in + fan_out)). Mặc định của Keras."""
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-limit, limit, size=shape).astype(dtype)


def orthogonal(rng: np.random.Generator, rows: int, cols: int, dtype) -> np.ndarray:
    """Ma trận trực giao (Saxe et al., 2014) — mặc định của Keras cho trọng số hồi quy."""
    flat = rng.normal(0.0, 1.0, size=(max(rows, cols), min(rows, cols)))
    q, r = np.linalg.qr(flat)
    q = q * np.sign(np.diag(r))
    if rows < cols:
        q = q.T
    return q[:rows, :cols].astype(dtype)
