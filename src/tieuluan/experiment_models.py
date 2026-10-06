"""Một kiến trúc — ba bản cài đặt: NumPy tự viết (scratch), PyTorch và Keras (TensorFlow).

Để phép so sánh giữa ba bản cài đặt công bằng:
1. Cùng đặc tả kiến trúc (``ARCHITECTURES``) và cùng số tham số học.
2. Trọng số khởi tạo được rút MỘT lần bằng NumPy rồi chép sang PyTorch/Keras
   (``copy_weights_to_torch``, ``copy_weights_to_keras``) → cùng điểm xuất phát.
3. Cùng thứ tự mini-batch, cùng hàm mất mát BCE trên logit, Adam(lr = 0,001, ε = 1e-8)
   và cắt chuẩn gradient toàn cục ở mức 1,0.
Các mô hình đều nhỏ, chạy trên CPU với 2 luồng để thời gian đo được so sánh trong cùng điều kiện.
"""

import os

os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import copy
import time

import numpy as np

from .scratch.core import Sequential
from .scratch.layers import Conv2d, Dense, Flatten, MaxPool2d, ReLU
from .scratch.losses import BCEWithLogitsLoss
from .scratch.optim import Adam, clip_grad_norm
from .scratch.recurrent import GRU, LSTM, LastStep, SimpleRNN

HIDDEN = 8  # số phần tử trạng thái của các mạng hồi tiếp

ARCHITECTURES = {
    "mlp": "d → Dense(16) → ReLU → Dense(1)",
    "cnn4": "1×20×20 → Conv(4 bộ lọc 3×3) → ReLU → MaxPool(2×2) → Flatten → Dense(1)",
    "cnn8": "1×20×20 → Conv(8 bộ lọc 3×3) → ReLU → MaxPool(2×2) → Flatten → Dense(1)",
    "cnndeep": "1×20×20 → Conv(4, 3×3) → ReLU → MaxPool(2) → Conv(8, 3×3) → ReLU → MaxPool(2) → Flatten → Dense(1)",
    "rnn": "20×1 → SimpleRNN(h = 8, tanh) → trạng thái cuối → Dense(1)",
    "lstm": "20×1 → LSTM(h = 8) → trạng thái cuối → Dense(1)",
    "gru": "20×1 → GRU(h = 8, reset-after) → trạng thái cuối → Dense(1)",
}


# ---------------------------------------------------------------------------------------------
# Bản 1 — NumPy tự viết: mọi lớp tự cài forward/backward trong src/tieuluan/scratch
# ---------------------------------------------------------------------------------------------
def scratch_model(kind, input_shape, seed):
    rng = np.random.default_rng(seed)
    if kind == "mlp":
        return Sequential(Dense(input_shape[0], 16, rng=rng), ReLU(), Dense(16, 1, rng=rng))
    if kind in ("cnn4", "cnn8"):
        filters = 8 if kind == "cnn8" else 4
        return Sequential(Conv2d(1, filters, 3, rng=rng), ReLU(), MaxPool2d(2),       # 20 → 18 → 9
                          Flatten(), Dense(filters * 9 * 9, 1, rng=rng))
    if kind == "cnndeep":
        return Sequential(Conv2d(1, 4, 3, rng=rng), ReLU(), MaxPool2d(2),             # 20 → 18 → 9
                          Conv2d(4, 8, 3, rng=rng), ReLU(), MaxPool2d(2),             # 9 → 7 → 3
                          Flatten(), Dense(8 * 3 * 3, 1, rng=rng))
    cell = {"rnn": SimpleRNN, "lstm": LSTM, "gru": GRU}[kind]
    return Sequential(cell(input_shape[-1], HIDDEN, rng=rng), LastStep(), Dense(HIDDEN, 1, rng=rng))


# ---------------------------------------------------------------------------------------------
# Bản 2 — PyTorch: autograd tự tính đạo hàm
# ---------------------------------------------------------------------------------------------
def torch_model(kind, input_shape):
    import torch
    from torch import nn

    torch.set_num_threads(2)
    if kind == "mlp":
        return nn.Sequential(nn.Linear(input_shape[0], 16), nn.ReLU(), nn.Linear(16, 1))
    if kind in ("cnn4", "cnn8"):
        filters = 8 if kind == "cnn8" else 4
        return nn.Sequential(nn.Conv2d(1, filters, 3), nn.ReLU(), nn.MaxPool2d(2),
                             nn.Flatten(), nn.Linear(filters * 9 * 9, 1))
    if kind == "cnndeep":
        return nn.Sequential(nn.Conv2d(1, 4, 3), nn.ReLU(), nn.MaxPool2d(2),
                             nn.Conv2d(4, 8, 3), nn.ReLU(), nn.MaxPool2d(2),
                             nn.Flatten(), nn.Linear(8 * 3 * 3, 1))

    class RecurrentNet(nn.Module):
        def __init__(self):
            super().__init__()
            cell = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[kind]
            self.recurrent = cell(input_shape[-1], HIDDEN, batch_first=True)
            self.output = nn.Linear(HIDDEN, 1)

        def forward(self, x):                      # x: [batch, 20, 1]
            sequence, _ = self.recurrent(x)
            return self.output(sequence[:, -1])     # trạng thái ở bước cuối → logit

    return RecurrentNet()


# ---------------------------------------------------------------------------------------------
# Bản 3 — Keras 3 (backend TensorFlow): khai báo mô hình bằng Functional API
# ---------------------------------------------------------------------------------------------
def keras_model(kind, input_shape):
    import keras
    import tensorflow as tf

    try:
        tf.config.threading.set_intra_op_parallelism_threads(2)
        tf.config.threading.set_inter_op_parallelism_threads(2)
    except RuntimeError:  # đã khởi tạo trước đó trong cùng tiến trình
        pass
    inputs = keras.Input(shape=input_shape)
    z = inputs
    if kind == "mlp":
        z = keras.layers.Dense(16, activation="relu")(z)
    elif kind.startswith("cnn"):
        z = keras.layers.Permute((2, 3, 1))(z)      # [C, H, W] → [H, W, C] (quy ước của Keras)
        first = 8 if kind == "cnn8" else 4
        z = keras.layers.Conv2D(first, 3, activation="relu")(z)
        z = keras.layers.MaxPooling2D(2)(z)
        if kind == "cnndeep":
            z = keras.layers.Conv2D(8, 3, activation="relu")(z)
            z = keras.layers.MaxPooling2D(2)(z)
        z = keras.layers.Permute((3, 1, 2))(z)      # trả về [C, H, W] để Flatten cùng thứ tự với PyTorch
        z = keras.layers.Flatten()(z)
    else:
        cell = {"rnn": keras.layers.SimpleRNN, "lstm": keras.layers.LSTM, "gru": keras.layers.GRU}[kind]
        z = cell(HIDDEN)(z)                          # mặc định trả về trạng thái ở bước cuối
    outputs = keras.layers.Dense(1)(z)               # logit (chưa qua sigmoid)
    return keras.Model(inputs, outputs)


# ---------------------------------------------------------------------------------------------
# Chép cùng bộ trọng số khởi tạo từ NumPy sang hai thư viện
# ---------------------------------------------------------------------------------------------
def _scratch_layers_with_weights(model):
    return [layer for layer in model.layers if layer.parameters()]


def copy_weights_to_torch(scratch, model, kind):
    import torch

    with torch.no_grad():
        for target, source in zip(model.parameters(), scratch.parameters(), strict=True):
            target.copy_(torch.from_numpy(source.value))
    if kind in ("rnn", "lstm"):
        # PyTorch có hai bias cộng dồn (b_ih + b_hh), Keras chỉ có một. Giữ b_hh = 0 và không học
        # để số tham số học và phép toán tương đương ở cả ba bản cài đặt.
        model.recurrent.bias_hh_l0.requires_grad_(False)


def copy_weights_to_keras(scratch, model, kind):
    targets = [layer for layer in model.layers if layer.weights]
    for target, source in zip(targets, _scratch_layers_with_weights(scratch), strict=True):
        if isinstance(source, Dense):
            target.set_weights([source.weight.value.T, source.bias.value])
        elif isinstance(source, Conv2d):          # [F, C, KH, KW] → [KH, KW, C, F]
            target.set_weights([source.weight.value.transpose(2, 3, 1, 0), source.bias.value])
        elif kind in ("rnn", "lstm"):              # thứ tự cổng LSTM (i, f, g, o) trùng nhau
            target.set_weights([source.weight_ih.value.T, source.weight_hh.value.T,
                                source.bias_ih.value + source.bias_hh.value])
        else:                                      # GRU: Keras xếp cổng (z, r, n), PyTorch/NumPy (r, z, n)
            h = HIDDEN
            order = np.r_[h:2 * h, 0:h, 2 * h:3 * h]
            target.set_weights([source.weight_ih.value[order].T, source.weight_hh.value[order].T,
                                np.stack([source.bias_ih.value[order], source.bias_hh.value[order]])])


def build_models(kind, input_shape, seed, frameworks=("scratch", "pytorch", "keras")):
    """Dựng các bản cài đặt được yêu cầu, tất cả bắt đầu từ cùng trọng số NumPy (theo ``seed``)."""
    scratch = scratch_model(kind, input_shape, seed)
    models = {"scratch": scratch}
    if "pytorch" in frameworks:
        import torch

        torch.manual_seed(seed)
        models["pytorch"] = torch_model(kind, input_shape)
        copy_weights_to_torch(scratch, models["pytorch"], kind)
    if "keras" in frameworks:
        import keras

        keras.utils.set_random_seed(seed)
        models["keras"] = keras_model(kind, input_shape)
        copy_weights_to_keras(scratch, models["keras"], kind)
    return {key: models[key] for key in frameworks}


def count_trainable(model, framework, kind):
    if framework == "scratch":
        return int(sum(p.value.size for p in model.parameters()
                       if not (kind in ("rnn", "lstm") and p.name == "bias_hh")))
    if framework == "pytorch":
        return int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    return int(sum(np.prod(w.shape) for w in model.trainable_weights))


def predict_logits(model, framework, x, batch_size=512):
    outputs = []
    if framework in ("scratch", "pytorch"):
        model.eval()
    for i in range(0, len(x), batch_size):
        xb = x[i:i + batch_size]
        if framework == "scratch":
            value = model.forward(xb)
        elif framework == "pytorch":
            import torch

            with torch.no_grad():
                value = model(torch.from_numpy(xb)).numpy()
        else:
            value = model(xb, training=False).numpy()
        outputs.append(value.reshape(-1))
    return np.concatenate(outputs)


# ---------------------------------------------------------------------------------------------
# Một bước cập nhật trên một mini-batch cho từng bản cài đặt
# ---------------------------------------------------------------------------------------------
class ScratchTrainer:
    """NumPy: forward → loss → backward (tự viết) → cắt gradient → Adam (tự viết)."""

    def __init__(self, model, kind):
        self.model = model
        self.params = [p for p in model.parameters() if not (kind in ("rnn", "lstm") and p.name == "bias_hh")]
        self.optimizer = Adam(self.params, lr=1e-3, eps=1e-8)
        self.loss_fn = BCEWithLogitsLoss()

    def step(self, xb, yb):
        self.model.train()
        self.model.zero_grad()
        logits = self.model.forward(xb)
        loss = self.loss_fn.forward(logits, yb)
        self.model.backward(self.loss_fn.backward())
        clip_grad_norm(self.params, 1.0)
        self.optimizer.step()
        return loss


class TorchTrainer:
    """PyTorch: loss.backward() do autograd tính; torch.optim.Adam cập nhật."""

    def __init__(self, model, kind):
        import torch

        self.torch = torch
        self.model = model
        self.params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(self.params, lr=1e-3, eps=1e-8)
        self.criterion = torch.nn.BCEWithLogitsLoss()

    def step(self, xb, yb):
        self.model.train()
        self.optimizer.zero_grad()
        loss = self.criterion(self.model(self.torch.from_numpy(xb)), self.torch.from_numpy(yb))
        loss.backward()
        self.torch.nn.utils.clip_grad_norm_(self.params, 1.0)
        self.optimizer.step()
        return float(loss.detach())


class KerasTrainer:
    """Keras: compile (loss + optimizer) rồi train_on_batch cho từng mini-batch."""

    def __init__(self, model, kind):
        import keras

        self.model = model
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3, epsilon=1e-8, global_clipnorm=1.0),
                      loss=keras.losses.BinaryCrossentropy(from_logits=True), jit_compile=False)

    def step(self, xb, yb):
        self.model.reset_metrics()
        return float(self.model.train_on_batch(xb, yb))


TRAINERS = {"scratch": ScratchTrainer, "pytorch": TorchTrainer, "keras": KerasTrainer}


def _snapshot(model, framework):
    if framework == "scratch":
        return model.state_dict()
    if framework == "pytorch":
        return copy.deepcopy(model.state_dict())
    return model.get_weights()


def _restore(model, framework, state):
    if framework == "keras":
        model.set_weights(state)
    else:
        model.load_state_dict(state)


def fit_model(model, framework, kind, x, y, xv, yv, seed, epochs=60, batch_size=64, patience=8):
    """Vòng huấn luyện chung: cùng thứ tự mini-batch cho mọi framework, dừng sớm theo BCE validation,
    khôi phục checkpoint có validation loss nhỏ nhất."""
    rng = np.random.default_rng(seed + 1000)
    trainer = TRAINERS[framework](model, kind)
    val_loss_fn = BCEWithLogitsLoss()
    history, best, wait, state, best_epoch = [], float("inf"), 0, None, 0
    started = time.perf_counter()
    for epoch in range(epochs):
        order = rng.permutation(len(y))
        total = 0.0
        for offset in range(0, len(y), batch_size):
            ids = order[offset:offset + batch_size]
            total += trainer.step(x[ids], y[ids, None]) * len(ids)
        val = val_loss_fn.forward(predict_logits(model, framework, xv), yv)
        if not np.isfinite(val):
            raise RuntimeError("Nonfinite validation loss")
        history.append({"epoch": epoch + 1, "train_loss": total / len(y), "val_loss": val})
        if val < best - 1e-6:
            best, wait, best_epoch = val, 0, epoch + 1
            state = _snapshot(model, framework)
        else:
            wait += 1
            if wait >= patience:
                break
    _restore(model, framework, state)
    return {"history": history, "best_epoch": best_epoch, "epochs_run": len(history),
            "training_seconds": time.perf_counter() - started}
