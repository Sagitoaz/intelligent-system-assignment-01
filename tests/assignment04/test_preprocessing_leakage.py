import numpy as np

from src.assignment04.comments_data import SharedTextVectorizer, duplicate_hash_overlap, normalize_text
from src.assignment04.data_common import TrainOnlyStandardizer
from src.assignment04.house_data import CategoryVocabulary, audit_property_identity_csv, property_identity_quality


def test_standardizer_uses_only_training_statistics():
    scaler = TrainOnlyStandardizer().fit(np.array([[0.0], [2.0]]))
    np.testing.assert_allclose(scaler.mean_, [1.0])
    np.testing.assert_allclose(scaler.transform(np.array([[101.0]])), [[100.0]])


def test_comments_vocabulary_is_train_only_and_has_pad_oov():
    vectorizer = SharedTextVectorizer(max_tokens=5, sequence_length=4).fit(["Alpha beta", "alpha gamma"])
    assert vectorizer.token_to_id["<PAD>"] == 0
    assert vectorizer.token_to_id["<OOV>"] == 1
    encoded = vectorizer.transform(["delta alpha"])
    np.testing.assert_array_equal(encoded, [[1, vectorizer.token_to_id["alpha"], 0, 0]])
    assert "delta" not in vectorizer.token_to_id


def test_unicode_whitespace_normalization_and_duplicate_hashing():
    assert normalize_text("  CAFÉ\n\tTest  ") == "café test"
    result = duplicate_hash_overlap([" Same  text "], ["same text", "different"])
    assert result == {"train_unique": 1, "test_unique": 2, "overlap_unique": 1}


def test_house_category_vocabulary_does_not_learn_validation_category():
    vocab = CategoryVocabulary().fit(["A", "B"])
    np.testing.assert_array_equal(vocab.transform(["B", "NEW", None]), [2, 0, 0])


def test_property_identity_quality_rejects_missing_and_reports_duplicates():
    rows = [
        {"street": 1, "city": "X", "state": "S", "zip_code": "1"},
        {"street": 1, "city": "X", "state": "S", "zip_code": "1"},
        {"street": None, "city": "X", "state": "S", "zip_code": "1"},
    ]
    result = property_identity_quality(rows)
    assert result["complete_rows"] == 2
    assert result["duplicate_rows"] == 1
    assert result["unique_groups"] == 1


def test_property_identity_csv_audit_tracks_conflicting_prices(tmp_path):
    path = tmp_path / "house.csv"
    path.write_text(
        "street,city,state,zip_code,price\n"
        "1,X,S,10,100\n1,X,S,10,120\n2,Y,S,20,200\n,X,S,10,300\n",
        encoding="utf-8",
    )
    result = audit_property_identity_csv(path, chunksize=2)
    assert result == {
        "total_rows": 4,
        "complete_rows": 3,
        "unique_groups": 2,
        "duplicate_rows": 1,
        "groups_with_price_conflicts": 1,
    }
