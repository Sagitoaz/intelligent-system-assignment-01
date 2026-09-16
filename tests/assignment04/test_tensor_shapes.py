import numpy as np
import pytest
import torch

from src.assignment04.scratch_models import ScratchCNN, ScratchHouseCNN, ScratchTextCNN
from src.assignment04.tensorflow_models import build_tf_cnn, build_tf_house_cnn, build_tf_text_cnn
from src.assignment04.torch_models import TorchCNN, TorchHouseCNN, TorchTextCNN


@pytest.mark.parametrize("improved,channels", [(False, 16), (True, 16)])
def test_diabetes_shapes_all_backends(improved, channels):
    x = np.zeros((2, 1, 21), dtype=np.float32)
    assert ScratchCNN(1, 2, improved=improved).forward(x).shape == (2, 2)
    assert tuple(TorchCNN(1, 2, improved=improved)(torch.from_numpy(x)).shape) == (2, 2)
    assert tuple(build_tf_cnn(21, 1, 2, improved=improved)(np.transpose(x, (0, 2, 1))).shape) == (2, 2)


@pytest.mark.parametrize("improved", [False, True])
def test_house_shapes_all_backends(improved):
    numeric = np.ones((2, 4), dtype=np.float32)
    state = np.array([1, 2]); status = np.array([1, 1])
    assert ScratchHouseCNN(4, 3, improved=improved).forward(numeric, state, status).shape == (2, 1)
    assert tuple(TorchHouseCNN(4, 3, improved=improved)(torch.from_numpy(numeric), torch.from_numpy(state), torch.from_numpy(status)).shape) == (2, 1)
    assert tuple(build_tf_house_cnn(4, 3, improved=improved)([numeric, state[:, None], status[:, None]]).shape) == (2, 1)


@pytest.mark.parametrize("improved", [False, True])
def test_comments_shapes_all_backends(improved):
    ids = np.array([[1, 2, 0, 0], [2, 3, 4, 0]], dtype=np.int64)
    mask = ids != 0
    assert ScratchTextCNN(10, 2, improved=improved).forward(ids, mask).shape == (2, 2)
    assert tuple(TorchTextCNN(10, 2, improved=improved)(torch.from_numpy(ids), torch.from_numpy(mask)).shape) == (2, 2)
    assert tuple(build_tf_text_cnn(10, 4, 2, improved=improved)(ids).shape) == (2, 2)


def test_tensorflow_text_pad_embeddings_are_zeroed_in_forward():
    model = build_tf_text_cnn(10, 4, 2)
    embedded = model.embed_tokens(np.array([[2, 0, 0, 0]], dtype=np.int32)).numpy()
    np.testing.assert_array_equal(embedded[:, 1:, :], 0.0)


def test_house_oov_embedding_is_learnable_not_a_padding_vector():
    model = ScratchHouseCNN(4, 3)
    assert np.any(model.encoder.state_embedding.weight[0] != 0.0)
    assert np.any(model.encoder.status_embedding.weight[0] != 0.0)


def test_exact_scratch_parameter_counts_and_layer_convention():
    diabetes_basic = ScratchCNN(1, 2, improved=False)
    diabetes_improved = ScratchCNN(1, 2, improved=True)
    house_basic = ScratchHouseCNN(4, 3, improved=False)
    house_improved = ScratchHouseCNN(4, 3, improved=True)
    comments_basic = ScratchTextCNN(10, 2, improved=False)
    comments_improved = ScratchTextCNN(10, 2, improved=True)
    assert (diabetes_basic.cnn.trainable_layers if hasattr(diabetes_basic, "cnn") else diabetes_basic.trainable_layers) == 3
    assert diabetes_improved.trainable_layers == 5
    assert diabetes_basic.parameter_count() == 466
    assert diabetes_improved.parameter_count() == 2562
    assert house_basic.encoder.parameter_count() == 120
    assert house_basic.cnn.parameter_count() == 617
    assert house_improved.cnn.parameter_count() == 2713
    assert sum(p.size for p, _ in comments_basic.parameters_and_grads()) == 986
    assert sum(p.size for p, _ in comments_improved.parameters_and_grads()) == 3082
