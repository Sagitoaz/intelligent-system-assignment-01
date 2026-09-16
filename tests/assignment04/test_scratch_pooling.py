import numpy as np

from src.assignment04.scratch_layers import GlobalMaxPool1D, MaxPool1D, ReLU


def test_relu_backward_uses_positive_input_mask():
    layer = ReLU()
    x = np.array([[[-2.0, 0.0, 3.0]]])
    np.testing.assert_array_equal(layer.forward(x), [[[0.0, 0.0, 3.0]]])
    np.testing.assert_array_equal(layer.backward(np.ones_like(x)), [[[0.0, 0.0, 1.0]]])


def test_max_pool_routes_gradient_to_argmax():
    layer = MaxPool1D(kernel_size=2, stride=2)
    x = np.array([[[1.0, 4.0, 3.0, 2.0]]])
    np.testing.assert_array_equal(layer.forward(x), [[[4.0, 3.0]]])
    np.testing.assert_array_equal(layer.backward(np.array([[[2.0, 5.0]]])), [[[0.0, 2.0, 5.0, 0.0]]])


def test_global_pool_mask_prevents_padding_from_winning():
    layer = GlobalMaxPool1D()
    x = np.array([[[-3.0, -2.0, 100.0, 200.0]]])
    mask = np.array([[True, True, False, False]])
    np.testing.assert_array_equal(layer.forward(x, mask), [[-2.0]])
    np.testing.assert_array_equal(layer.backward(np.array([[7.0]])), [[[-0.0, 7.0, 0.0, 0.0]]])
