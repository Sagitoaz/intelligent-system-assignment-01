"""Thư viện học sâu tối giản tự cài bằng NumPy ("from scratch").

Không dùng bất kỳ hàm tự động lấy đạo hàm (autograd) nào: mỗi lớp tự cài
``forward`` (lan truyền xuôi) và ``backward`` (lan truyền ngược) theo đúng công
thức toán học trong báo cáo. Bố cục trọng số của các lớp được chọn trùng với
PyTorch để có thể kiểm tra "song trùng" (parity) giữa hai cách cài đặt.
"""

from .core import Module, Parameter, Sequential
from .layers import (
    BatchNorm1d,
    BatchNorm2d,
    Conv2d,
    Dense,
    Dropout,
    Flatten,
    LeakyReLU,
    MaxPool2d,
    ReLU,
    Sigmoid,
    Tanh,
)
from .losses import BCEWithLogitsLoss, MSELoss
from .optim import SGD, Adam, clip_grad_norm
from .recurrent import GRU, LSTM, SimpleRNN, LastStep

__all__ = [
    "Module", "Parameter", "Sequential",
    "Dense", "ReLU", "LeakyReLU", "Sigmoid", "Tanh", "Dropout", "Flatten",
    "BatchNorm1d", "BatchNorm2d", "Conv2d", "MaxPool2d",
    "SimpleRNN", "LSTM", "GRU", "LastStep",
    "BCEWithLogitsLoss", "MSELoss", "SGD", "Adam", "clip_grad_norm",
]
