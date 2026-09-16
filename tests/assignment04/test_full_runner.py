import numpy as np

from src.assignment04.full_runner import FULL_BACKENDS, FULL_ARCHITECTURE, TASK_CONFIG
from src.assignment04.parity_runner import _inverse_target_from_metadata
from src.assignment04.evaluation import regression_metrics


def test_full_runner_is_improved_only_and_excludes_scratch():
    assert FULL_ARCHITECTURE == "improved"
    assert FULL_BACKENDS == ("pytorch", "tensorflow")
    assert TASK_CONFIG["comments"]["batch_size"] == 128
    assert TASK_CONFIG["comments"]["max_len"] == 192


def test_house_inverse_target_uses_float64_for_large_finite_predictions():
    inverse = _inverse_target_from_metadata({"target_mean": 10.0, "target_scale": 1.0})
    result = inverse(np.asarray([100.0], dtype=np.float32))
    assert result.dtype == np.float64
    assert np.isfinite(result).all()


def test_regression_metrics_marks_nonfinite_candidate_as_worst_without_crashing():
    metrics = regression_metrics(np.asarray([100.0]), np.asarray([np.inf]))
    assert metrics == {"mae": np.inf, "rmse": np.inf, "r2": -np.inf}
