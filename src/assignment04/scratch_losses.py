"""Numerically stable losses and analytical gradients in NumPy."""

import numpy as np


class SoftmaxCrossEntropyLoss:
    def __init__(self, class_weights=None):
        self.class_weights = None if class_weights is None else np.asarray(class_weights)

    def forward(self, logits, targets):
        logits = np.asarray(logits)
        self.targets = np.asarray(targets, dtype=np.int64)
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        self.probabilities = exp / exp.sum(axis=1, keepdims=True)
        self.sample_weights = np.ones(len(targets), dtype=logits.dtype)
        if self.class_weights is not None:
            self.sample_weights = self.class_weights[self.targets].astype(logits.dtype)
        self.normalizer = self.sample_weights.sum()
        # Compute NLL directly from shifted logits.  Taking log of an already
        # underflowed softmax probability would clip extreme losses and diverge
        # from the stable cross-entropy used by PyTorch/TensorFlow.
        log_partition = np.log(exp.sum(axis=1))
        negative_log_likelihood = -shifted[np.arange(len(targets)), self.targets] + log_partition
        return float(np.sum(negative_log_likelihood * self.sample_weights) / self.normalizer)

    def backward(self):
        gradient = self.probabilities.copy()
        gradient[np.arange(len(self.targets)), self.targets] -= 1.0
        gradient *= self.sample_weights[:, None] / self.normalizer
        return gradient


class MSELoss:
    def forward(self, predictions, targets):
        self.difference = np.asarray(predictions) - np.asarray(targets)
        return float(np.mean(self.difference ** 2))

    def backward(self):
        return 2.0 * self.difference / self.difference.size
