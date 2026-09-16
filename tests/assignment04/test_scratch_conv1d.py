import numpy as np

from src.assignment04.scratch_layers import Conv1D


def test_lecture_valid_convolution_example():
    layer = Conv1D(1, 1, 3, padding="valid", dtype=np.float64)
    layer.weight[...] = [[[0.5, -1.0, 0.5]]]
    layer.bias[...] = 0.0
    x = np.array([[[2.0, 1.0, 3.0, 4.0, 2.0]]])
    # First window: 2*0.5 + 1*(-1) + 3*0.5 = 1.5.
    np.testing.assert_allclose(layer.forward(x), [[[1.5, -0.5, -1.5]]])


def test_same_padding_preserves_length_and_backward_shapes():
    rng = np.random.default_rng(42)
    layer = Conv1D(2, 3, 3, padding="same", rng=rng, dtype=np.float64)
    x = rng.normal(size=(4, 2, 7))
    y = layer.forward(x)
    dx = layer.backward(np.ones_like(y))
    assert y.shape == (4, 3, 7)
    assert dx.shape == x.shape
    assert layer.grad_weight.shape == layer.weight.shape
    assert layer.grad_bias.shape == layer.bias.shape
