"""Finite-difference and cross-framework numerical checks."""

from __future__ import annotations

import numpy as np

from .scratch_layers import Conv1D, Dense, Embedding
from .scratch_losses import MSELoss, SoftmaxCrossEntropyLoss
from .scratch_models import ScratchCNN


def relative_error(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    denominator = np.maximum(1e-12, np.abs(actual) + np.abs(expected))
    return float(np.max(np.abs(actual - expected) / denominator))


def _numerical_gradient(array, objective, epsilon=1e-6):
    gradient = np.zeros_like(array, dtype=np.float64)
    iterator = np.nditer(array, flags=["multi_index"], op_flags=["readwrite"])
    while not iterator.finished:
        index = iterator.multi_index
        original = float(array[index])
        array[index] = original + epsilon
        plus = objective()
        array[index] = original - epsilon
        minus = objective()
        array[index] = original
        gradient[index] = (plus - minus) / (2 * epsilon)
        iterator.iternext()
    return gradient


def finite_difference_checks(seed=42):
    rng = np.random.default_rng(seed)
    conv = Conv1D(2, 2, 3, padding="same", rng=rng, dtype=np.float64)
    x = rng.normal(size=(2, 2, 4))
    upstream = rng.normal(size=(2, 2, 4))
    conv.forward(x)
    analytic_x = conv.backward(upstream).copy()
    analytic_w, analytic_b = conv.grad_weight.copy(), conv.grad_bias.copy()
    numeric_x = _numerical_gradient(x, lambda: float(np.sum(conv.forward(x) * upstream)))
    numeric_w = _numerical_gradient(conv.weight, lambda: float(np.sum(conv.forward(x) * upstream)))
    numeric_b = _numerical_gradient(conv.bias, lambda: float(np.sum(conv.forward(x) * upstream)))

    dense = Dense(3, 2, rng=rng, dtype=np.float64)
    dense_x = rng.normal(size=(2, 3)); dense_upstream = rng.normal(size=(2, 2))
    dense.forward(dense_x)
    analytic_dense_x = dense.backward(dense_upstream).copy()
    analytic_dense_w, analytic_dense_b = dense.grad_weight.copy(), dense.grad_bias.copy()
    numeric_dense_x = _numerical_gradient(dense_x, lambda: float(np.sum(dense.forward(dense_x) * dense_upstream)))
    numeric_dense_w = _numerical_gradient(dense.weight, lambda: float(np.sum(dense.forward(dense_x) * dense_upstream)))
    numeric_dense_b = _numerical_gradient(dense.bias, lambda: float(np.sum(dense.forward(dense_x) * dense_upstream)))

    logits = rng.normal(size=(3, 2)); targets = np.array([0, 1, 0])
    ce = SoftmaxCrossEntropyLoss(np.array([1.0, 2.0])); ce.forward(logits, targets)
    analytic_ce = ce.backward().copy()
    numeric_ce = _numerical_gradient(logits, lambda: SoftmaxCrossEntropyLoss(np.array([1.0, 2.0])).forward(logits, targets))

    prediction = rng.normal(size=(3, 1)); target = rng.normal(size=(3, 1))
    mse = MSELoss(); mse.forward(prediction, target); analytic_mse = mse.backward().copy()
    numeric_mse = _numerical_gradient(prediction, lambda: MSELoss().forward(prediction, target))

    embedding = Embedding(5, 2, padding_idx=0, rng=rng, dtype=np.float64)
    ids = np.array([[1, 2, 1]]); embed_upstream = rng.normal(size=(1, 3, 2))
    embedding.forward(ids); embedding.backward(embed_upstream); analytic_embedding = embedding.grad_weight.copy()
    numeric_embedding = _numerical_gradient(embedding.weight, lambda: float(np.sum(embedding.forward(ids) * embed_upstream)))
    numeric_embedding[0] = 0.0

    return {
        "conv_dx": relative_error(analytic_x, numeric_x),
        "conv_dw": relative_error(analytic_w, numeric_w),
        "conv_db": relative_error(analytic_b, numeric_b),
        "dense_dx": relative_error(analytic_dense_x, numeric_dense_x),
        "dense_dw": relative_error(analytic_dense_w, numeric_dense_w),
        "dense_db": relative_error(analytic_dense_b, numeric_dense_b),
        "cross_entropy": relative_error(analytic_ce, numeric_ce),
        "mse": relative_error(analytic_mse, numeric_mse),
        "embedding": relative_error(analytic_embedding, numeric_embedding),
    }


def _copy_scratch_to_torch(scratch, torch_model):
    import torch

    with torch.no_grad():
        for name in ["conv1", "conv2", "conv3", "hidden", "output"]:
            if hasattr(scratch, name):
                source, target = getattr(scratch, name), getattr(torch_model, name)
                target.weight.copy_(torch.from_numpy(source.weight))
                target.bias.copy_(torch.from_numpy(source.bias))


def _copy_scratch_to_tensorflow(scratch, tf_model):
    for name in ["conv1", "conv2", "conv3"]:
        if hasattr(scratch, name):
            source, target = getattr(scratch, name), getattr(tf_model, name)
            target.set_weights([source.weight.transpose(2, 1, 0), source.bias])
    for name, tf_name in [("hidden", "hidden"), ("output", "output_layer")]:
        if hasattr(scratch, name):
            source, target = getattr(scratch, name), getattr(tf_model, tf_name)
            target.set_weights([source.weight.T, source.bias])


def forward_parity(input_channels=1, length=21, output_dim=2, *, improved=False, seed=42):
    import tensorflow as tf
    import torch

    from .tensorflow_models import build_tf_cnn
    from .torch_models import TorchCNN

    rng = np.random.default_rng(seed)
    scratch = ScratchCNN(input_channels, output_dim, improved=improved, rng=rng, dtype=np.float32)
    torch_model = TorchCNN(input_channels, output_dim, improved=improved).eval()
    tf_model = build_tf_cnn(length, input_channels, output_dim, improved=improved)
    _copy_scratch_to_torch(scratch, torch_model)
    _copy_scratch_to_tensorflow(scratch, tf_model)
    x = rng.normal(size=(3, input_channels, length)).astype(np.float32)
    labels = np.array([0, 1, 0], dtype=np.int64)
    scratch_logits = scratch.forward(x)
    with torch.no_grad():
        torch_logits = torch_model(torch.from_numpy(x)).numpy()
    tf_logits = tf_model(np.transpose(x, (0, 2, 1)), training=False).numpy()
    scratch_loss = SoftmaxCrossEntropyLoss().forward(scratch_logits, labels)
    torch_loss = torch.nn.functional.cross_entropy(torch.from_numpy(torch_logits), torch.from_numpy(labels)).item()
    tf_loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels, tf_logits, from_logits=True)).numpy()
    return {
        "scratch_torch_logits": float(np.max(np.abs(scratch_logits - torch_logits))),
        "scratch_tensorflow_logits": float(np.max(np.abs(scratch_logits - tf_logits))),
        "scratch_torch_loss": abs(scratch_loss - torch_loss),
        "scratch_tensorflow_loss": abs(scratch_loss - float(tf_loss)),
    }


def backward_parity_with_torch(seed=42):
    import torch

    rng = np.random.default_rng(seed)
    layer = Conv1D(2, 3, 3, padding="same", rng=rng, dtype=np.float64)
    x = rng.normal(size=(2, 2, 5)); upstream = rng.normal(size=(2, 3, 5))
    layer.forward(x); scratch_dx = layer.backward(upstream)
    torch_layer = torch.nn.Conv1d(2, 3, 3, padding=1, dtype=torch.float64)
    with torch.no_grad():
        torch_layer.weight.copy_(torch.from_numpy(layer.weight)); torch_layer.bias.copy_(torch.from_numpy(layer.bias))
    torch_x = torch.tensor(x, requires_grad=True)
    torch_layer(torch_x).backward(torch.tensor(upstream))
    return {
        "dx": relative_error(scratch_dx, torch_x.grad.numpy()),
        "dw": relative_error(layer.grad_weight, torch_layer.weight.grad.numpy()),
        "db": relative_error(layer.grad_bias, torch_layer.bias.grad.numpy()),
    }
