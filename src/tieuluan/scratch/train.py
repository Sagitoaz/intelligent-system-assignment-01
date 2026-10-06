"""Vòng lặp huấn luyện mini-batch có dừng sớm (early stopping) cho mô hình NumPy."""

from __future__ import annotations

import math
import time

import numpy as np

from .core import Module
from .optim import clip_grad_norm


def predict(model: Module, x: np.ndarray, batch_size: int = 512) -> np.ndarray:
    model.eval()
    outs = [model.forward(x[i:i + batch_size]) for i in range(0, len(x), batch_size)]
    return np.concatenate(outs, axis=0)


def evaluate_loss(model: Module, loss_fn, x: np.ndarray, y: np.ndarray, batch_size: int = 512) -> float:
    model.eval()
    total, count = 0.0, 0
    for i in range(0, len(x), batch_size):
        out = model.forward(x[i:i + batch_size])
        total += loss_fn.forward(out, y[i:i + batch_size]) * len(out)
        count += len(out)
    return total / count


def fit(model: Module, loss_fn, optimizer, x_train, y_train, x_val, y_val, *, epochs: int, batch_size: int,
        rng: np.random.Generator, patience: int = 10, min_delta: float = 0.0, clip_norm: float | None = None,
        verbose: bool = False) -> dict:
    """Huấn luyện tới khi hết ``epochs`` hoặc val_loss không giảm sau ``patience`` epoch; khôi phục trọng số tốt nhất."""
    history = {"train_loss": [], "val_loss": [], "epoch_time": []}
    best_val, best_state, best_epoch, wait = math.inf, None, 0, 0
    params = model.parameters()
    n = len(x_train)
    for epoch in range(1, epochs + 1):
        start = time.perf_counter()
        model.train()
        order = rng.permutation(n)
        running, seen = 0.0, 0
        for i in range(0, n, batch_size):
            idx = order[i:i + batch_size]
            xb, yb = x_train[idx], y_train[idx]
            model.zero_grad()
            out = model.forward(xb)
            loss = loss_fn.forward(out, yb)
            model.backward(loss_fn.backward())
            if clip_norm is not None:
                clip_grad_norm(params, clip_norm)
            optimizer.step()
            running += loss * len(idx)
            seen += len(idx)
        val_loss = evaluate_loss(model, loss_fn, x_val, y_val)
        history["train_loss"].append(running / seen)
        history["val_loss"].append(val_loss)
        history["epoch_time"].append(time.perf_counter() - start)
        if verbose:
            print(f"epoch {epoch:3d} train={running / seen:.5f} val={val_loss:.5f} ({history['epoch_time'][-1]:.1f}s)")
        if val_loss < best_val - min_delta:
            best_val, best_state, best_epoch, wait = val_loss, model.state_dict(), epoch, 0
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    history["best_epoch"] = best_epoch
    history["best_val_loss"] = best_val
    return history
