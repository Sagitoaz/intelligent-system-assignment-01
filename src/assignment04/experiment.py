"""Controlled, framework-neutral parity training for Assignment 04.

The three backends consume the same prepared arrays, canonical NumPy weights,
and deterministic epoch permutations.  Test evaluation is intentionally kept
outside the training functions so orchestration can freeze architecture first.
"""

from __future__ import annotations

import copy
import time

import numpy as np

from .evaluation import classification_metrics, regression_metrics
from .scratch_losses import MSELoss, SoftmaxCrossEntropyLoss
from .scratch_optim import Adam


def epoch_order(size: int, *, seed: int, epoch: int) -> np.ndarray:
    """Return the one canonical sample order consumed by every backend."""
    return np.random.default_rng(seed + epoch).permutation(size)


def classification_class_weights(labels) -> np.ndarray:
    """Balanced binary weights fitted solely from the supplied training labels."""
    labels = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(labels, minlength=2)
    if np.any(counts == 0):
        raise ValueError("Both classes must occur in the parity training split")
    return (len(labels) / (2.0 * counts)).astype(np.float32)


def is_better_checkpoint(candidate: dict, incumbent: dict | None, problem: str) -> bool:
    if incumbent is None:
        return True
    tolerance = 1e-12
    if problem == "classification":
        if candidate["f1"] > incumbent["f1"] + tolerance:
            return True
        return abs(candidate["f1"] - incumbent["f1"]) <= tolerance and candidate["loss"] < incumbent["loss"]
    if candidate["rmse"] < incumbent["rmse"] - tolerance:
        return True
    return abs(candidate["rmse"] - incumbent["rmse"]) <= tolerance and candidate["mae"] < incumbent["mae"]


def _cnn_state(cnn, prefix=""):
    state = {}
    for name in ("conv1", "conv2", "conv3", "hidden", "output"):
        if hasattr(cnn, name):
            layer = getattr(cnn, name)
            state[f"{prefix}{name}.weight"] = layer.weight.copy()
            state[f"{prefix}{name}.bias"] = layer.bias.copy()
    return state


def scratch_state_dict(model) -> dict[str, np.ndarray]:
    """Capture all trainable NumPy arrays with PyTorch-compatible names."""
    if hasattr(model, "embedding") and hasattr(model, "cnn"):
        return {"embedding.weight": model.embedding.weight.copy(), **_cnn_state(model.cnn, "cnn.")}
    if hasattr(model, "encoder") and hasattr(model, "cnn"):
        encoder = model.encoder
        return {
            "encoder.numeric_weight": encoder.numeric_weight.copy(),
            "encoder.numeric_bias": encoder.numeric_bias.copy(),
            "encoder.state_embedding.weight": encoder.state_embedding.weight.copy(),
            "encoder.status_embedding.weight": encoder.status_embedding.weight.copy(),
            **_cnn_state(model.cnn, "cnn."),
        }
    return _cnn_state(model)


def load_scratch_state(model, state: dict[str, np.ndarray]):
    current = scratch_state_dict(model)
    if current.keys() != state.keys():
        raise ValueError(f"Canonical state keys differ: {sorted(current)} != {sorted(state)}")
    if hasattr(model, "embedding") and hasattr(model, "cnn"):
        model.embedding.weight[...] = state["embedding.weight"]
        cnn, prefix = model.cnn, "cnn."
    elif hasattr(model, "encoder") and hasattr(model, "cnn"):
        encoder = model.encoder
        encoder.numeric_weight[...] = state["encoder.numeric_weight"]
        encoder.numeric_bias[...] = state["encoder.numeric_bias"]
        encoder.state_embedding.weight[...] = state["encoder.state_embedding.weight"]
        encoder.status_embedding.weight[...] = state["encoder.status_embedding.weight"]
        cnn, prefix = model.cnn, "cnn."
    else:
        cnn, prefix = model, ""
    for name in ("conv1", "conv2", "conv3", "hidden", "output"):
        if hasattr(cnn, name):
            layer = getattr(cnn, name)
            layer.weight[...] = state[f"{prefix}{name}.weight"]
            layer.bias[...] = state[f"{prefix}{name}.bias"]
    if hasattr(model, "embedding"):
        model.embedding.enforce_padding()
    return model


def load_torch_state(model, state: dict[str, np.ndarray]):
    import torch

    parameters = dict(model.named_parameters())
    if parameters.keys() != state.keys():
        raise ValueError(f"PyTorch state keys differ: {sorted(parameters)} != {sorted(state)}")
    with torch.no_grad():
        for name, parameter in parameters.items():
            parameter.copy_(torch.from_numpy(state[name]))
    return model


def _assign_tf_cnn(cnn, state, prefix):
    for name in ("conv1", "conv2", "conv3"):
        if hasattr(cnn, name):
            getattr(cnn, name).set_weights([
                state[f"{prefix}{name}.weight"].transpose(2, 1, 0),
                state[f"{prefix}{name}.bias"],
            ])
    for scratch_name, tf_name in (("hidden", "hidden"), ("output", "output_layer")):
        if hasattr(cnn, tf_name):
            getattr(cnn, tf_name).set_weights([
                state[f"{prefix}{scratch_name}.weight"].T,
                state[f"{prefix}{scratch_name}.bias"],
            ])


def _tf_cnn_state(cnn, prefix=""):
    state = {}
    for name in ("conv1", "conv2", "conv3"):
        if hasattr(cnn, name):
            kernel, bias = getattr(cnn, name).get_weights()
            state[f"{prefix}{name}.weight"] = kernel.transpose(2, 1, 0).copy()
            state[f"{prefix}{name}.bias"] = bias.copy()
    for canonical, tf_name in (("hidden", "hidden"), ("output", "output_layer")):
        if hasattr(cnn, tf_name):
            kernel, bias = getattr(cnn, tf_name).get_weights()
            state[f"{prefix}{canonical}.weight"] = kernel.T.copy()
            state[f"{prefix}{canonical}.bias"] = bias.copy()
    return state


def tensorflow_state_dict(model, task: str):
    """Extract a named state in the same canonical layout as NumPy/PyTorch."""
    if task == "comments":
        return {"embedding.weight": model.embedding.get_weights()[0].copy(), **_tf_cnn_state(model.cnn, "cnn.")}
    if task == "house":
        return {
            "encoder.numeric_weight": model.encoder.numeric_weight.numpy().copy(),
            "encoder.numeric_bias": model.encoder.numeric_bias.numpy().copy(),
            "encoder.state_embedding.weight": model.encoder.state_embedding.get_weights()[0].copy(),
            "encoder.status_embedding.weight": model.encoder.status_embedding.get_weights()[0].copy(),
            **_tf_cnn_state(model.cnn, "cnn."),
        }
    return _tf_cnn_state(model)


def load_tensorflow_state(model, state: dict[str, np.ndarray], task: str):
    current = tensorflow_state_dict(model, task)
    if current.keys() != state.keys():
        raise ValueError(f"TensorFlow state keys differ: {sorted(current)} != {sorted(state)}")
    for name in current:
        if current[name].shape != np.asarray(state[name]).shape:
            raise ValueError(f"TensorFlow state shape differs for {name}: {current[name].shape} != {np.asarray(state[name]).shape}")
    if task == "comments":
        model.embedding.set_weights([state["embedding.weight"]])
        _assign_tf_cnn(model.cnn, state, "cnn.")
    elif task == "house":
        encoder = model.encoder
        encoder.numeric_weight.assign(state["encoder.numeric_weight"])
        encoder.numeric_bias.assign(state["encoder.numeric_bias"])
        encoder.state_embedding.set_weights([state["encoder.state_embedding.weight"]])
        encoder.status_embedding.set_weights([state["encoder.status_embedding.weight"]])
        _assign_tf_cnn(model.cnn, state, "cnn.")
    else:
        _assign_tf_cnn(model, state, "")
    return model


def parameter_count(model, backend: str) -> int:
    if backend == "scratch":
        return int(sum(parameter.size for parameter, _ in model.parameters_and_grads()))
    if backend == "pytorch":
        return int(sum(parameter.numel() for parameter in model.parameters()))
    return int(model.count_params())


def _slice_inputs(data, indices):
    if "x" in data:
        return data["x"][indices]
    if "ids" in data:
        return data["ids"][indices]
    return tuple(data[name][indices] for name in ("numeric", "state", "status"))


def normalize_data_contract(task, data):
    """Coerce one split to the exact common dtype/shape contract."""
    if task == "diabetes":
        result = {"x": np.ascontiguousarray(data["x"], dtype=np.float32),
                  "y": np.ascontiguousarray(data["y"], dtype=np.int64).reshape(-1)}
    elif task == "comments":
        result = {"ids": np.ascontiguousarray(data["ids"], dtype=np.int64),
                  "y": np.ascontiguousarray(data["y"], dtype=np.int64).reshape(-1)}
    elif task == "house":
        result = {
            "numeric": np.ascontiguousarray(data["numeric"], dtype=np.float32),
            "state": np.ascontiguousarray(data["state"], dtype=np.int64).reshape(-1),
            "status": np.ascontiguousarray(data["status"], dtype=np.int64).reshape(-1),
            "y": np.ascontiguousarray(data["y"], dtype=np.float32).reshape(-1, 1),
        }
    else:
        raise ValueError(f"Unknown task: {task}")
    sizes = {len(value) for value in result.values()}
    if len(sizes) != 1:
        raise ValueError(f"Input/target row counts differ: {sizes}")
    return result


def _scratch_forward(model, inputs, task):
    if task == "house":
        return model.forward(*inputs)
    return model.forward(inputs)


def _torch_forward(model, inputs, task, torch):
    if task == "house":
        numeric, state, status = inputs
        return model(torch.from_numpy(numeric), torch.from_numpy(state), torch.from_numpy(status))
    tensor = torch.from_numpy(inputs)
    return model(tensor)


def _tf_forward(model, inputs, task, tf, training=False):
    if task == "house":
        numeric, state, status = inputs
        return model([numeric, state[:, None], status[:, None]], training=training)
    if task == "diabetes":
        inputs = np.transpose(inputs, (0, 2, 1))
    return model(inputs, training=training)


def _loss_and_metrics(problem, y, logits, class_weights=None, inverse_target=None):
    if problem == "classification":
        loss = SoftmaxCrossEntropyLoss(class_weights).forward(logits, y)
        return {"loss": float(loss), **classification_metrics(y, logits)}
    y = np.asarray(y, dtype=np.float32).reshape(-1, 1)
    logits = np.asarray(logits, dtype=np.float32).reshape(-1, 1)
    loss = MSELoss().forward(logits, y)
    truth = inverse_target(y) if inverse_target else np.asarray(y).reshape(-1)
    prediction = inverse_target(logits) if inverse_target else np.asarray(logits).reshape(-1)
    return {"loss": float(loss), **regression_metrics(truth, prediction)}


def predict_batches(model, backend, task, data, batch_size):
    outputs = []
    if backend == "pytorch":
        import torch
        model.eval()
    for start in range(0, len(data["y"]), batch_size):
        indices = np.arange(start, min(start + batch_size, len(data["y"])))
        inputs = _slice_inputs(data, indices)
        if backend == "scratch":
            output = _scratch_forward(model, inputs, task)
        elif backend == "pytorch":
            with torch.no_grad():
                output = _torch_forward(model, inputs, task, torch).cpu().numpy()
        else:
            import tensorflow as tf
            output = _tf_forward(model, inputs, task, tf, training=False).numpy()
        outputs.append(np.asarray(output))
    return np.concatenate(outputs, axis=0)


def evaluate(model, backend, task, problem, data, batch_size, *, class_weights=None, inverse_target=None):
    logits = predict_batches(model, backend, task, data, batch_size)
    return _loss_and_metrics(problem, data["y"], logits, class_weights, inverse_target), logits


def _batch_denominator(labels, problem, class_weights):
    if problem == "regression" or class_weights is None:
        return float(len(labels))
    return float(np.asarray(class_weights)[np.asarray(labels, dtype=np.int64)].sum())


def train_scratch(model, task, problem, train, validation, *, batch_size, max_epochs, patience,
                  learning_rate, batch_seed, class_weights=None, inverse_target=None):
    loss_fn = SoftmaxCrossEntropyLoss(class_weights) if problem == "classification" else MSELoss()
    optimizer = Adam(model.parameters_and_grads(), learning_rate=learning_rate, epsilon=1e-7)
    history, best, best_state, best_epoch, stale = [], None, None, None, 0
    started = time.perf_counter()
    for epoch in range(max_epochs):
        numerator = denominator = 0.0
        order = epoch_order(len(train["y"]), seed=batch_seed, epoch=epoch)
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            inputs, labels = _slice_inputs(train, indices), train["y"][indices]
            logits = _scratch_forward(model, inputs, task)
            loss = loss_fn.forward(logits, labels)
            model.backward(loss_fn.backward())
            optimizer.step()
            if hasattr(model, "embedding"):
                model.embedding.enforce_padding()
            weight = _batch_denominator(labels, problem, class_weights)
            numerator += loss * weight; denominator += weight
        validation_metrics, _ = evaluate(model, "scratch", task, problem, validation, batch_size,
                                         class_weights=class_weights, inverse_target=inverse_target)
        row = {"epoch": epoch + 1, "train_loss": numerator / denominator,
               **{f"val_{key}": value for key, value in validation_metrics.items() if key != "confusion_matrix"}}
        history.append(row)
        candidate = ({key: validation_metrics[key] for key in ("f1", "loss")}
                     if problem == "classification"
                     else {key: validation_metrics[key] for key in ("rmse", "mae")})
        if is_better_checkpoint(candidate, best, problem):
            best, best_state, best_epoch, stale = candidate, scratch_state_dict(model), epoch + 1, 0
        else:
            stale += 1
            if stale >= patience:
                break
    load_scratch_state(model, best_state)
    metrics, _ = evaluate(model, "scratch", task, problem, validation, batch_size,
                          class_weights=class_weights, inverse_target=inverse_target)
    return {"model": model, "history": history, "best_epoch": best_epoch,
            "validation": metrics, "seconds": time.perf_counter() - started, "state": best_state}


def train_pytorch(model, task, problem, train, validation, *, batch_size, max_epochs, patience,
                  learning_rate, batch_seed, class_weights=None, inverse_target=None):
    import torch

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, eps=1e-7)
    weight_tensor = None if class_weights is None else torch.from_numpy(np.asarray(class_weights, dtype=np.float32))
    history, best, best_state, best_epoch, stale = [], None, None, None, 0
    started = time.perf_counter()
    for epoch in range(max_epochs):
        model.train(); numerator = denominator = 0.0
        order = epoch_order(len(train["y"]), seed=batch_seed, epoch=epoch)
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            inputs, labels_np = _slice_inputs(train, indices), train["y"][indices]
            labels = torch.from_numpy(labels_np)
            optimizer.zero_grad(set_to_none=True)
            logits = _torch_forward(model, inputs, task, torch)
            loss = (torch.nn.functional.cross_entropy(logits, labels, weight=weight_tensor)
                    if problem == "classification" else torch.nn.functional.mse_loss(logits, labels))
            loss.backward(); optimizer.step()
            weight = _batch_denominator(labels_np, problem, class_weights)
            numerator += float(loss.detach()) * weight; denominator += weight
        validation_metrics, _ = evaluate(model, "pytorch", task, problem, validation, batch_size,
                                         class_weights=class_weights, inverse_target=inverse_target)
        history.append({"epoch": epoch + 1, "train_loss": numerator / denominator,
                        **{f"val_{key}": value for key, value in validation_metrics.items() if key != "confusion_matrix"}})
        print(f"EPOCH {task}/pytorch {epoch + 1}: train_loss={numerator / denominator:.6f} "
              f"val_loss={validation_metrics['loss']:.6f}", flush=True)
        candidate = ({key: validation_metrics[key] for key in ("f1", "loss")} if problem == "classification"
                     else {key: validation_metrics[key] for key in ("rmse", "mae")})
        if is_better_checkpoint(candidate, best, problem):
            best, best_epoch, stale = candidate, epoch + 1, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    metrics, _ = evaluate(model, "pytorch", task, problem, validation, batch_size,
                          class_weights=class_weights, inverse_target=inverse_target)
    return {"model": model, "history": history, "best_epoch": best_epoch, "validation": metrics,
            "seconds": time.perf_counter() - started,
            "state": {key: value.detach().cpu().numpy() for key, value in best_state.items()}}


def train_tensorflow(model, task, problem, train, validation, *, batch_size, max_epochs, patience,
                     learning_rate, batch_seed, class_weights=None, inverse_target=None):
    import tensorflow as tf

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=1e-7)
    class_weights_tf = None if class_weights is None else tf.constant(class_weights, dtype=tf.float32)

    @tf.function(reduce_retracing=True)
    def step(inputs, labels):
        with tf.GradientTape() as tape:
            logits = model(inputs, training=True) if task != "house" else model(inputs, training=True)
            if problem == "classification":
                values = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
                if class_weights_tf is not None:
                    weights = tf.gather(class_weights_tf, labels)
                    loss = tf.reduce_sum(values * weights) / tf.reduce_sum(weights)
                else:
                    loss = tf.reduce_mean(values)
            else:
                loss = tf.reduce_mean(tf.square(logits - labels))
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    history, best, best_weights, best_epoch, stale = [], None, None, None, 0
    started = time.perf_counter()
    for epoch in range(max_epochs):
        numerator = denominator = 0.0
        order = epoch_order(len(train["y"]), seed=batch_seed, epoch=epoch)
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            inputs, labels_np = _slice_inputs(train, indices), train["y"][indices]
            if task == "diabetes":
                inputs = np.transpose(inputs, (0, 2, 1))
            elif task == "house":
                numeric, state, status = inputs
                inputs = [numeric, state[:, None], status[:, None]]
            labels = tf.convert_to_tensor(labels_np)
            loss = step(inputs, labels)
            weight = _batch_denominator(labels_np, problem, class_weights)
            numerator += float(loss.numpy()) * weight; denominator += weight
        validation_metrics, _ = evaluate(model, "tensorflow", task, problem, validation, batch_size,
                                         class_weights=class_weights, inverse_target=inverse_target)
        history.append({"epoch": epoch + 1, "train_loss": numerator / denominator,
                        **{f"val_{key}": value for key, value in validation_metrics.items() if key != "confusion_matrix"}})
        print(f"EPOCH {task}/tensorflow {epoch + 1}: train_loss={numerator / denominator:.6f} "
              f"val_loss={validation_metrics['loss']:.6f}", flush=True)
        candidate = ({key: validation_metrics[key] for key in ("f1", "loss")} if problem == "classification"
                     else {key: validation_metrics[key] for key in ("rmse", "mae")})
        if is_better_checkpoint(candidate, best, problem):
            best, best_epoch, stale = candidate, epoch + 1, 0
            best_weights = [value.copy() for value in model.get_weights()]
        else:
            stale += 1
            if stale >= patience:
                break
    model.set_weights(best_weights)
    metrics, _ = evaluate(model, "tensorflow", task, problem, validation, batch_size,
                          class_weights=class_weights, inverse_target=inverse_target)
    return {"model": model, "history": history, "best_epoch": best_epoch, "validation": metrics,
            "seconds": time.perf_counter() - started, "state": tensorflow_state_dict(model, task)}
