import numpy as np

from src.assignment04.experiment import (
    classification_class_weights,
    epoch_order,
    is_better_checkpoint,
    normalize_data_contract,
    scratch_state_dict,
    train_scratch,
    load_tensorflow_state,
    load_torch_state,
)
from src.assignment04.scratch_models import ScratchCNN, ScratchHouseCNN, ScratchTextCNN


def test_epoch_order_is_backend_independent_and_deterministic():
    first = epoch_order(20, seed=42, epoch=3)
    second = epoch_order(20, seed=42, epoch=3)
    other_epoch = epoch_order(20, seed=42, epoch=4)
    np.testing.assert_array_equal(first, second)
    assert sorted(first.tolist()) == list(range(20))
    assert not np.array_equal(first, other_epoch)


def test_balanced_class_weights_use_training_labels_only():
    weights = classification_class_weights(np.array([0, 0, 0, 1]))
    np.testing.assert_allclose(weights, [2 / 3, 2.0])


def test_checkpoint_rules_follow_assignment_selection_metrics():
    assert is_better_checkpoint({"f1": 0.7, "loss": 2.0}, {"f1": 0.6, "loss": 0.1}, "classification")
    assert is_better_checkpoint({"f1": 0.7, "loss": 0.2}, {"f1": 0.7, "loss": 0.3}, "classification")
    assert is_better_checkpoint({"rmse": 10.0, "mae": 9.0}, {"rmse": 11.0, "mae": 1.0}, "regression")
    assert is_better_checkpoint({"rmse": 10.0, "mae": 8.0}, {"rmse": 10.0, "mae": 9.0}, "regression")


def test_canonical_state_includes_task_specific_encoder_parameters():
    rng = np.random.default_rng(42)
    diabetes = scratch_state_dict(ScratchCNN(1, 2, rng=rng))
    house = scratch_state_dict(ScratchHouseCNN(5, 3, rng=np.random.default_rng(42)))
    text = scratch_state_dict(ScratchTextCNN(30, rng=np.random.default_rng(42)))
    assert "conv1.weight" in diabetes
    assert "encoder.numeric_weight" in house
    assert "encoder.state_embedding.weight" in house
    assert "embedding.weight" in text
    assert "cnn.conv1.weight" in text


def test_house_contract_prevents_broadcasted_regression_targets_and_float64_inputs():
    data = normalize_data_contract("house", {
        "numeric": np.ones((2, 4), dtype=np.float64), "state": np.array([[1], [2]], dtype=np.int32),
        "status": np.array([1, 2]), "y": np.array([10.0, 20.0], dtype=np.float64),
    })
    assert data["numeric"].dtype == np.float32
    assert data["state"].shape == (2,)
    assert data["y"].dtype == np.float32
    assert data["y"].shape == (2, 1)


def test_task_specific_canonical_states_load_strictly_in_both_frameworks():
    from src.assignment04.tensorflow_models import build_tf_house_cnn, build_tf_text_cnn
    from src.assignment04.torch_models import TorchHouseCNN, TorchTextCNN

    for task, scratch, torch_model, tf_model in [
        ("comments", ScratchTextCNN(30, rng=np.random.default_rng(42)), TorchTextCNN(30), build_tf_text_cnn(30, 12)),
        ("house", ScratchHouseCNN(5, 3, rng=np.random.default_rng(42)), TorchHouseCNN(5, 3), build_tf_house_cnn(5, 3)),
    ]:
        state = scratch_state_dict(scratch)
        load_torch_state(torch_model, state)
        load_tensorflow_state(tf_model, state, task)
        broken = dict(state)
        broken.pop(next(iter(broken)))
        import pytest
        with pytest.raises(ValueError, match="state keys differ"):
            load_tensorflow_state(tf_model, broken, task)


def test_scratch_regression_checkpoint_uses_rmse_without_classification_keys():
    model = ScratchHouseCNN(3, 2, rng=np.random.default_rng(42))
    data = normalize_data_contract("house", {
        "numeric": np.zeros((4, 4)), "state": np.array([1, 1, 2, 2]),
        "status": np.array([1, 1, 1, 1]), "y": np.array([0.0, 0.1, 0.2, 0.3]),
    })
    result = train_scratch(
        model, "house", "regression", data, data, batch_size=2, max_epochs=1,
        patience=3, learning_rate=1e-3, batch_seed=42, inverse_target=lambda x: np.asarray(x).reshape(-1),
    )
    assert result["best_epoch"] == 1
    assert np.isfinite(result["validation"]["rmse"])
