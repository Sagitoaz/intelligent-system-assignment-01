from src.assignment04.training import framework_tiny_overfit_suite, tiny_overfit_suite


def test_every_scratch_task_architecture_reduces_tiny_batch_loss():
    results = tiny_overfit_suite(seed=42, steps=60)
    assert set(results) == {
        "diabetes_basic", "diabetes_improved",
        "house_basic", "house_improved",
        "comments_basic", "comments_improved",
    }
    for name, values in results.items():
        assert values["final_loss"] < values["initial_loss"] * 0.8, (name, values)


def test_each_framework_can_reduce_a_tiny_cnn_batch_loss():
    results = framework_tiny_overfit_suite(seed=42, steps=25)
    assert set(results) == {"pytorch_basic", "pytorch_improved", "tensorflow_basic", "tensorflow_improved"}
    for name, values in results.items():
        assert values["final_loss"] < values["initial_loss"] * 0.95, (name, values)
