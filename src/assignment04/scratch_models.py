"""Basic (3-layer) and improved (5-layer) NumPy CNNs."""

from __future__ import annotations

import numpy as np

from .scratch_layers import Conv1D, Dense, Embedding, GlobalMaxPool1D, MaxPool1D, ReLU


class ScratchCNN:
    """Two conv + output dense, optionally conv3 + hidden dense."""

    def __init__(self, input_channels, output_dim, *, improved=False, rng=None, dtype=np.float32):
        generator = rng or np.random.default_rng(42)
        self.improved = improved
        self.conv1 = Conv1D(input_channels, 8, 3, padding="same", rng=generator, dtype=dtype)
        self.relu1 = ReLU()
        self.conv2 = Conv1D(8, 16, 3, padding="same", rng=generator, dtype=dtype)
        self.relu2 = ReLU()
        self.pool = MaxPool1D(2, 2)
        self.global_pool = GlobalMaxPool1D()
        if improved:
            self.conv3 = Conv1D(16, 32, 3, padding="same", rng=generator, dtype=dtype)
            self.relu3 = ReLU()
            self.hidden = Dense(32, 16, rng=generator, dtype=dtype)
            self.relu_hidden = ReLU()
            self.output = Dense(16, output_dim, rng=generator, dtype=dtype)
        else:
            self.output = Dense(16, output_dim, rng=generator, dtype=dtype)
        self._pooled_mask = None

    def forward(self, x, mask=None):
        value = self.relu1.forward(self.conv1.forward(x))
        value = self.relu2.forward(self.conv2.forward(value))
        value = self.pool.forward(value)
        self._pooled_mask = None if mask is None else self.pool.downsample_mask(mask)
        if self.improved:
            value = self.relu3.forward(self.conv3.forward(value))
        value = self.global_pool.forward(value, self._pooled_mask)
        if self.improved:
            value = self.relu_hidden.forward(self.hidden.forward(value))
        return self.output.forward(value)

    def backward(self, grad_logits):
        gradient = self.output.backward(grad_logits)
        if self.improved:
            gradient = self.hidden.backward(self.relu_hidden.backward(gradient))
        gradient = self.global_pool.backward(gradient)
        if self.improved:
            gradient = self.conv3.backward(self.relu3.backward(gradient))
        gradient = self.pool.backward(gradient)
        gradient = self.conv2.backward(self.relu2.backward(gradient))
        return self.conv1.backward(self.relu1.backward(gradient))

    @property
    def trainable_layers(self):
        return 5 if self.improved else 3

    def parameters_and_grads(self):
        layers = [self.conv1, self.conv2]
        if self.improved:
            layers.extend([self.conv3, self.hidden])
        layers.append(self.output)
        return [item for layer in layers for item in layer.parameters_and_grads()]

    def parameter_count(self):
        return sum(parameter.size for parameter, _ in self.parameters_and_grads())


class ScratchTextCNN:
    def __init__(self, vocabulary_size, output_dim=2, *, improved=False, rng=None, dtype=np.float32):
        generator = rng or np.random.default_rng(42)
        self.embedding = Embedding(vocabulary_size, 16, padding_idx=0, rng=generator, dtype=dtype)
        self.cnn = ScratchCNN(16, output_dim, improved=improved, rng=generator, dtype=dtype)

    def forward(self, token_ids, mask=None):
        embedded = self.embedding.forward(token_ids).transpose(0, 2, 1)
        return self.cnn.forward(embedded, token_ids != 0 if mask is None else mask)

    def backward(self, grad_logits):
        grad_embedding = self.cnn.backward(grad_logits).transpose(0, 2, 1)
        self.embedding.backward(grad_embedding)

    def parameters_and_grads(self):
        return self.embedding.parameters_and_grads() + self.cnn.parameters_and_grads()


class ScratchHouseFieldEncoder:
    """Four numeric projections and two independent categorical embeddings."""

    def __init__(self, state_size, status_size, *, rng=None, dtype=np.float32):
        generator = rng or np.random.default_rng(42)
        self.numeric_weight = generator.normal(0.0, 0.1, (4, 8)).astype(dtype)
        self.numeric_bias = np.zeros((4, 8), dtype=dtype)
        self.grad_numeric_weight = np.zeros_like(self.numeric_weight)
        self.grad_numeric_bias = np.zeros_like(self.numeric_bias)
        # ID 0 means missing/OOV for House and therefore needs a learned vector;
        # unlike text PAD, it is not sequence padding.
        self.state_embedding = Embedding(state_size, 8, padding_idx=None, rng=generator, dtype=dtype)
        self.status_embedding = Embedding(status_size, 8, padding_idx=None, rng=generator, dtype=dtype)

    def forward(self, numeric, state_ids, status_ids):
        self._numeric = np.asarray(numeric)
        self._state_ids = np.asarray(state_ids)[:, None]
        self._status_ids = np.asarray(status_ids)[:, None]
        numeric_tokens = self._numeric[:, :, None] * self.numeric_weight[None, :, :] + self.numeric_bias[None, :, :]
        state = self.state_embedding.forward(self._state_ids)
        status = self.status_embedding.forward(self._status_ids)
        return np.concatenate([state, status, numeric_tokens], axis=1).transpose(0, 2, 1)

    def backward(self, grad_output):
        gradient = np.asarray(grad_output).transpose(0, 2, 1)
        self.state_embedding.backward(gradient[:, 0:1])
        self.status_embedding.backward(gradient[:, 1:2])
        numeric_gradient = gradient[:, 2:]
        self.grad_numeric_weight[...] = np.einsum("bf,bfd->fd", self._numeric, numeric_gradient)
        self.grad_numeric_bias[...] = numeric_gradient.sum(axis=0)
        return np.einsum("bfd,fd->bf", numeric_gradient, self.numeric_weight)

    def parameters_and_grads(self):
        return [
            (self.numeric_weight, self.grad_numeric_weight),
            (self.numeric_bias, self.grad_numeric_bias),
            *self.state_embedding.parameters_and_grads(),
            *self.status_embedding.parameters_and_grads(),
        ]

    def parameter_count(self):
        return sum(parameter.size for parameter, _ in self.parameters_and_grads())


class ScratchHouseCNN:
    def __init__(self, state_size, status_size, *, improved=False, rng=None, dtype=np.float32):
        generator = rng or np.random.default_rng(42)
        self.encoder = ScratchHouseFieldEncoder(state_size, status_size, rng=generator, dtype=dtype)
        self.cnn = ScratchCNN(8, 1, improved=improved, rng=generator, dtype=dtype)

    def forward(self, numeric, state_ids, status_ids):
        return self.cnn.forward(self.encoder.forward(numeric, state_ids, status_ids))

    def backward(self, grad_output):
        return self.encoder.backward(self.cnn.backward(grad_output))

    def parameters_and_grads(self):
        return self.encoder.parameters_and_grads() + self.cnn.parameters_and_grads()
