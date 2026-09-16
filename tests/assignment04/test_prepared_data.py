import csv
import numpy as np
import pandas as pd

from src.assignment04.prepared_data import DiabetesPreprocessor, HousePreprocessor, load_comment_rows
from src.assignment04.comments_data import normalized_text_sha256
from src.assignment04.diabetes_data import FEATURE_ORDER, SCALED_FEATURES


def test_diabetes_preprocessor_preserves_frozen_order_and_fits_scaled_columns_only_on_train():
    train = pd.DataFrame({name: [0.0, 2.0] for name in FEATURE_ORDER})
    validation = pd.DataFrame({name: [100.0] for name in FEATURE_ORDER})
    fitted = DiabetesPreprocessor().fit(train)
    transformed = fitted.transform(validation)
    assert transformed.shape == (1, 1, 21)
    for position, name in enumerate(FEATURE_ORDER):
        assert transformed[0, 0, position] == (99.0 if name in SCALED_FEATURES else 100.0)


def test_house_preprocessor_has_six_field_positions_and_train_only_unknown_categories():
    train = pd.DataFrame({
        "state": ["A", "B"], "status": ["sale", "sold"],
        "acre_lot": [1.0, np.nan], "house_size": [100.0, 200.0],
        "bed": [2.0, 4.0], "bath": [1.0, 3.0], "price": [100_000.0, 300_000.0],
    })
    validation = pd.DataFrame({
        "state": ["NEW"], "status": [None], "acre_lot": [np.nan],
        "house_size": [300.0], "bed": [6.0], "bath": [5.0], "price": [500_000.0],
    })
    fitted = HousePreprocessor().fit(train)
    data = fitted.transform(validation)
    assert data["numeric"].shape == (1, 4)
    assert data["state"].tolist() == [0]
    assert data["status"].tolist() == [0]
    np.testing.assert_allclose(fitted.inverse_target(data["y"]), [500_000.0], rtol=1e-5)
    assert fitted.field_shape == (8, 6)


def test_comment_rows_are_rehydrated_in_frozen_manifest_order(tmp_path):
    source = tmp_path / "official.csv"
    with source.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["Review", "Label"])
        writer.writeheader()
        writer.writerows([
            {"Review": "zero", "Label": 1},
            {"Review": "one", "Label": 2},
            {"Review": "two", "Label": 1},
        ])
    indices = tmp_path / "indices.csv"
    with indices.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["source", "row_index", "text_sha256"])
        writer.writeheader()
        writer.writerows([
            {"source": "official_test", "row_index": 2, "text_sha256": normalized_text_sha256("two")},
            {"source": "official_test", "row_index": 0, "text_sha256": normalized_text_sha256("zero")},
        ])
    texts, labels = load_comment_rows(source, indices, expected_source="official_test")
    assert texts == ["two", "zero"]
    assert labels.tolist() == [0, 0]
