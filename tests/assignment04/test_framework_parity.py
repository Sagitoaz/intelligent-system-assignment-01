import numpy as np
import pytest

from src.assignment04.parity import backward_parity_with_torch, forward_parity


@pytest.mark.parametrize("improved", [False, True])
def test_scratch_torch_tensorflow_forward_parity(improved):
    result = forward_parity(input_channels=1, length=21, output_dim=2, improved=improved, seed=42)
    assert result["scratch_torch_logits"] < 1e-5
    assert result["scratch_tensorflow_logits"] < 1e-5
    assert result["scratch_torch_loss"] < 1e-6
    assert result["scratch_tensorflow_loss"] < 1e-6


def test_scratch_backward_matches_torch_autograd():
    result = backward_parity_with_torch(seed=42)
    assert max(result.values()) < 1e-5, result
