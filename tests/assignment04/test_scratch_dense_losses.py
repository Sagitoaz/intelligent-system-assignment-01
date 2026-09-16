import numpy as np

from src.assignment04.scratch_layers import Dense, Embedding
from src.assignment04.scratch_losses import MSELoss, SoftmaxCrossEntropyLoss
from src.assignment04.scratch_optim import Adam


def test_dense_forward_backward_shapes():
    layer = Dense(3, 2, rng=np.random.default_rng(1), dtype=np.float64)
    x = np.arange(12, dtype=np.float64).reshape(4, 3)
    out = layer.forward(x)
    dx = layer.backward(np.ones_like(out))
    assert out.shape == (4, 2)
    assert dx.shape == x.shape


def test_softmax_cross_entropy_is_finite_and_weighted():
    logits = np.array([[1000.0, 999.0], [-1000.0, -999.0]])
    labels = np.array([0, 1])
    loss = SoftmaxCrossEntropyLoss(class_weights=np.array([1.0, 3.0]))
    assert np.isfinite(loss.forward(logits, labels))
    assert loss.backward().shape == logits.shape


def test_softmax_cross_entropy_uses_logsumexp_without_probability_clipping():
    loss = SoftmaxCrossEntropyLoss()
    actual = loss.forward(np.array([[-1000.0, 0.0]]), np.array([0]))
    np.testing.assert_allclose(actual, 1000.0, rtol=1e-12)


def test_mse_forward_backward():
    loss = MSELoss()
    pred = np.array([[1.0], [3.0]])
    target = np.array([[0.0], [1.0]])
    assert loss.forward(pred, target) == 2.5
    np.testing.assert_allclose(loss.backward(), [[1.0], [2.0]])


def test_embedding_accumulates_repeated_ids_and_keeps_pad_zero():
    layer = Embedding(5, 3, padding_idx=0, rng=np.random.default_rng(2), dtype=np.float64)
    ids = np.array([[0, 2, 2]])
    layer.forward(ids)
    layer.backward(np.ones((1, 3, 3)))
    np.testing.assert_array_equal(layer.grad_weight[0], 0.0)
    np.testing.assert_array_equal(layer.grad_weight[2], 2.0)
    layer.weight[0] = 4.0
    layer.enforce_padding()
    np.testing.assert_array_equal(layer.weight[0], 0.0)


def test_manual_adam_applies_bias_corrected_update():
    parameter = np.array([1.0])
    gradient = np.array([0.5])
    optimizer = Adam([(parameter, gradient)], learning_rate=0.1, epsilon=1e-8)
    optimizer.step()
    np.testing.assert_allclose(parameter, [0.9], atol=1e-7)
