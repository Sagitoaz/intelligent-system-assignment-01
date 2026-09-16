import numpy as np

from src.assignment04.parity import finite_difference_checks


def test_all_finite_difference_gradient_checks_pass():
    results = finite_difference_checks(seed=42)
    assert set(results) == {"conv_dx", "conv_dw", "conv_db", "dense_dx", "dense_dw", "dense_db", "cross_entropy", "mse", "embedding"}
    assert all(value < 1e-5 for value in results.values()), results
