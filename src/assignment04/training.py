"""Small training helpers only; full experiment orchestration is deferred."""

import numpy as np

from .scratch_losses import MSELoss, SoftmaxCrossEntropyLoss
from .scratch_optim import Adam


def train_scratch_tiny(model, x, y, *, classification=True, steps=150, learning_rate=0.01, mask=None):
    loss_fn = SoftmaxCrossEntropyLoss() if classification else MSELoss()
    optimizer = Adam(model.parameters_and_grads(), learning_rate=learning_rate)
    losses = []
    for _ in range(steps):
        logits = model.forward(x, mask) if mask is not None else model.forward(x)
        losses.append(loss_fn.forward(logits, y))
        model.backward(loss_fn.backward())
        optimizer.step()
        if hasattr(model, "embedding"):
            model.embedding.enforce_padding()
    return {"initial_loss": losses[0], "final_loss": losses[-1]}


def _fit_tiny(model, forward, targets, *, classification, steps, learning_rate=0.02):
    loss_fn = SoftmaxCrossEntropyLoss() if classification else MSELoss()
    optimizer = Adam(model.parameters_and_grads(), learning_rate=learning_rate)
    initial = None
    final = None
    for _ in range(steps):
        prediction = forward()
        current = loss_fn.forward(prediction, targets)
        if initial is None:
            initial = current
        model.backward(loss_fn.backward())
        optimizer.step()
        if hasattr(model, "embedding"):
            model.embedding.enforce_padding()
        final = current
    return {"initial_loss": float(initial), "final_loss": float(final)}


def tiny_overfit_suite(seed=42, steps=60):
    """Overfit tiny deterministic batches; these are diagnostics, not results."""
    from .scratch_models import ScratchCNN, ScratchHouseCNN, ScratchTextCNN

    rng = np.random.default_rng(seed)
    results = {}

    diabetes_x = rng.normal(size=(16, 1, 21)).astype(np.float32)
    diabetes_y = (diabetes_x.max(axis=(1, 2)) > 1.8).astype(np.int64)
    for improved in (False, True):
        model = ScratchCNN(1, 2, improved=improved, rng=np.random.default_rng(seed))
        key = "diabetes_improved" if improved else "diabetes_basic"
        results[key] = _fit_tiny(model, lambda m=model: m.forward(diabetes_x), diabetes_y, classification=True, steps=steps)

    numeric = rng.normal(size=(16, 4)).astype(np.float32)
    state = rng.integers(1, 4, size=16); status = rng.integers(1, 3, size=16)
    house_y = (0.4 * numeric[:, :1] - 0.2 * numeric[:, 1:2] + state[:, None] * 0.05).astype(np.float32)
    for improved in (False, True):
        model = ScratchHouseCNN(4, 3, improved=improved, rng=np.random.default_rng(seed))
        key = "house_improved" if improved else "house_basic"
        results[key] = _fit_tiny(model, lambda m=model: m.forward(numeric, state, status), house_y, classification=False, steps=steps)

    token_ids = rng.integers(2, 20, size=(16, 12), dtype=np.int64)
    token_ids[:, -3:] = 0
    comments_y = (token_ids[:, 0] % 2).astype(np.int64)
    mask = token_ids != 0
    for improved in (False, True):
        model = ScratchTextCNN(20, 2, improved=improved, rng=np.random.default_rng(seed))
        key = "comments_improved" if improved else "comments_basic"
        results[key] = _fit_tiny(model, lambda m=model: m.forward(token_ids, mask), comments_y, classification=True, steps=steps)

    return results


def framework_tiny_overfit_suite(seed=42, steps=25):
    """Exercise optimizer/backprop paths of both framework model families."""
    import tensorflow as tf
    import torch

    from .tensorflow_models import build_tf_cnn
    from .torch_models import TorchCNN

    rng = np.random.default_rng(seed)
    x_ncl = rng.normal(size=(16, 1, 21)).astype(np.float32)
    labels = (x_ncl.max(axis=(1, 2)) > 1.8).astype(np.int64)
    results = {}
    torch.manual_seed(seed)
    tf.keras.utils.set_random_seed(seed)

    for improved in (False, True):
        model = TorchCNN(1, 2, improved=improved)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
        x = torch.from_numpy(x_ncl); y = torch.from_numpy(labels)
        losses = []
        for _ in range(steps):
            optimizer.zero_grad(); loss = torch.nn.functional.cross_entropy(model(x), y)
            losses.append(float(loss.detach())); loss.backward(); optimizer.step()
        results["pytorch_improved" if improved else "pytorch_basic"] = {
            "initial_loss": losses[0], "final_loss": losses[-1]
        }

    x_nlc = np.transpose(x_ncl, (0, 2, 1))
    for improved in (False, True):
        model = build_tf_cnn(21, 1, 2, improved=improved)
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.02)
        losses = []
        for _ in range(steps):
            with tf.GradientTape() as tape:
                logits = model(x_nlc, training=True)
                loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True))
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            losses.append(float(loss.numpy()))
        results["tensorflow_improved" if improved else "tensorflow_basic"] = {
            "initial_loss": losses[0], "final_loss": losses[-1]
        }
    return results
