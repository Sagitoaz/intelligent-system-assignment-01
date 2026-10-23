"""Six trainable layers for CNN/RNN/LSTM, using existing NumPy primitives."""
import os
os.environ.setdefault('KERAS_BACKEND', 'tensorflow')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import numpy as np
from src.tieuluan.scratch.core import Sequential
from src.tieuluan.scratch.layers import Conv2d, Dense, Flatten, ReLU
from src.tieuluan.scratch.recurrent import SimpleRNN, LSTM, LastStep


def scratch_deep(kind, seed=42):
    rng = np.random.default_rng(seed)
    if kind == 'cnn':
        layers = []
        for cin, cout in [(1, 4), (4, 4), (4, 8), (8, 8)]:
            layers += [Conv2d(cin, cout, 3, rng=rng), ReLU()]
        return Sequential(*layers, Flatten(),
                          Dense(8*12*12, 8, rng=rng), ReLU(),
                          Dense(8, 1, rng=rng))
    cell = {'rnn': SimpleRNN, 'lstm': LSTM}[kind]
    layers = [cell(1, 8, rng=rng)]
    layers += [cell(8, 8, rng=rng) for _ in range(4)]
    return Sequential(*layers, LastStep(), Dense(8, 1, rng=rng))


def keras_deep(kind, seed=42):
    import keras
    keras.utils.set_random_seed(seed)
    if kind == 'cnn':
        x = keras.Input((1, 20, 20))
        z = keras.layers.Permute((2, 3, 1))(x)
        for width in [4, 4, 8, 8]:
            z = keras.layers.Conv2D(width, 3, activation='relu')(z)
        z = keras.layers.Permute((3, 1, 2))(z)
        z = keras.layers.Flatten()(z)
        z = keras.layers.Dense(8, activation='relu')(z)
    else:
        x = keras.Input((20, 1))
        z = x
        cell = {'rnn': keras.layers.SimpleRNN,
                'lstm': keras.layers.LSTM}[kind]
        for depth in range(5):
            z = cell(8, return_sequences=(depth < 4))(z)
    return keras.Model(x, keras.layers.Dense(1)(z))


def torch_deep(kind, seed=42):
    import torch
    from torch import nn
    torch.manual_seed(seed)
    if kind == 'cnn':
        layers = []
        for cin, cout in [(1, 4), (4, 4), (4, 8), (8, 8)]:
            layers += [nn.Conv2d(cin, cout, 3), nn.ReLU()]
        return nn.Sequential(*layers, nn.Flatten(),
                             nn.Linear(8*12*12, 8), nn.ReLU(),
                             nn.Linear(8, 1))

    class Stacked(nn.Module):
        def __init__(self):
            super().__init__()
            cell = {'rnn': nn.RNN, 'lstm': nn.LSTM}[kind]
            self.cells = nn.ModuleList([cell(1, 8, batch_first=True)])
            self.cells.extend([cell(8, 8, batch_first=True)
                               for _ in range(4)])
            self.head = nn.Linear(8, 1)

        def forward(self, x):
            for cell in self.cells:
                x, _ = cell(x)
            return self.head(x[:, -1])
    return Stacked()


def align_weights(source, target, framework, kind):
    """Same effective weights and same trainable parameter counts."""
    weighted = [l for l in source.layers if l.parameters()]
    if framework == 'keras':
        targets = [l for l in target.layers if l.weights]
        for s, t in zip(weighted, targets, strict=True):
            if isinstance(s, Conv2d):
                t.set_weights([s.weight.value.transpose(2, 3, 1, 0),
                               s.bias.value])
            elif isinstance(s, Dense):
                t.set_weights([s.weight.value.T, s.bias.value])
            else:
                t.set_weights([s.weight_ih.value.T,
                               s.weight_hh.value.T,
                               s.bias_ih.value + s.bias_hh.value])
    else:
        import torch
        with torch.no_grad():
            for s, t in zip(source.parameters(), target.parameters(),
                            strict=True):
                t.copy_(torch.from_numpy(s.value))
        if kind != 'cnn':
            for cell in target.cells:
                cell.bias_hh_l0.requires_grad_(False)
    if kind != 'cnn':
        for layer in weighted[:-1]:
            layer.bias_hh.grad[...] = 0


def train_batch(model, framework, x, y, optimizer):
    """One actual optimization step, binary classification on logits."""
    if framework == 'scratch':
        from src.tieuluan.scratch.losses import BCEWithLogitsLoss
        from src.tieuluan.scratch.optim import clip_grad_norm
        model.train()
        model.zero_grad()
        loss = BCEWithLogitsLoss()
        value = loss.forward(model.forward(x), y)
        model.backward(loss.backward())
        # One effective bias, matching Keras and the frozen Torch bias.
        for layer in model.layers:
            if isinstance(layer, (SimpleRNN, LSTM)):
                layer.bias_hh.grad[...] = 0
        clip_grad_norm(model.parameters(), 1.0)
        optimizer.step()
        return value
    if framework == 'pytorch':
        import torch
        model.train()
        optimizer.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            model(torch.from_numpy(x)), torch.from_numpy(y))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        return float(loss.detach())
    import tensorflow as tf
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=y, logits=logits))
    grads = tape.gradient(loss, model.trainable_variables)
    grads, _ = tf.clip_by_global_norm(grads, 1.0)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return float(loss.numpy())
