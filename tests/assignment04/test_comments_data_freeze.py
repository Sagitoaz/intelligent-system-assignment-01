import csv
import json

import numpy as np

from src.assignment04.comments_data import (
    SharedTextVectorizer,
    build_comments_data_freeze,
    max_length_decision,
)


def _write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["Review", "Label"])
        writer.writeheader()
        writer.writerows(rows)


def _balanced_rows(prefix, count_per_label, duplicate=None):
    rows = []
    for label in (1, 2):
        for index in range(count_per_label):
            review = f"{prefix} label {label} row {index}"
            if duplicate is not None and label == 1 and index == 0:
                review = duplicate
            rows.append({"Review": review, "Label": label})
    return rows


def test_data_freeze_is_deterministic_isolated_and_writes_only_indices(tmp_path):
    train_path = tmp_path / "official_train.csv"
    test_path = tmp_path / "official_test.csv"
    train_rows = _balanced_rows("train", 8)
    train_rows[2]["Review"] = " Same duplicate "  # train owns this duplicate.
    train_rows[3]["Review"] = "same   duplicate"
    _write_csv(train_path, train_rows)
    _write_csv(test_path, _balanced_rows("test", 6, duplicate="SAME duplicate"))

    result = build_comments_data_freeze(
        train_path,
        test_path,
        tmp_path / "freeze",
        train_size=8,
        validation_size=4,
        test_size=6,
        seed=42,
    )

    assert result["sizes"] == {"train": 8, "validation": 4, "test": 6}
    assert result["class_counts"] == {
        "train": {"1": 4, "2": 4},
        "validation": {"1": 2, "2": 2},
        "test": {"1": 3, "2": 3},
    }
    assert result["audit"]["after"] == {
        "train_validation": 0,
        "train_test": 0,
        "validation_test": 0,
    }
    assert result["sources"] == {"train": str(train_path), "test": str(test_path)}
    assert train_path.read_text(encoding="utf-8").startswith("Review,Label")
    assert test_path.read_text(encoding="utf-8").startswith("Review,Label")

    manifest = json.loads((tmp_path / "freeze" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["normalization"] == "NFC + lowercase + collapsed whitespace"
    assert manifest["text_hash"] == "sha256(normalized UTF-8 text)"
    assert manifest["vocabulary"]["fit_split"] == "train"
    assert manifest["vocabulary"]["pad_id"] == 0
    assert manifest["vocabulary"]["oov_id"] == 1
    assert set(manifest["artifact_sha256"]) == {
        "train_indices.csv", "validation_indices.csv", "test_indices.csv",
        "parity_train_indices.csv", "parity_validation_indices.csv", "parity_test_indices.csv", "vocabulary.json",
    }
    assert set(next(csv.DictReader((tmp_path / "freeze" / "train_indices.csv").open(encoding="utf-8")))) == {
        "source", "row_index", "text_sha256"
    }
    assert all((tmp_path / "freeze" / f"{split}_indices.csv").exists() for split in ("train", "validation", "test"))


def test_data_freeze_vocabulary_and_token_statistics_are_train_only(tmp_path):
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    _write_csv(train_path, _balanced_rows("known", 5))
    _write_csv(test_path, _balanced_rows("externalonly", 4))

    result = build_comments_data_freeze(
        train_path, test_path, tmp_path / "freeze", train_size=6, validation_size=2, test_size=4, seed=42
    )

    vectorizer = SharedTextVectorizer.load(tmp_path / "freeze" / "vocabulary.json")
    assert "externalonly" not in vectorizer.token_to_id
    assert set(result["token_length_statistics"]) == {"train"}
    assert result["token_length_statistics"]["train"]["count"] == 6
    assert "p75" in result["token_length_statistics"]["train"]
    assert "truncation_rate_at_128" in result["token_length_statistics"]["train"]
    assert "truncation_rate_at_recommended_max_len" in result["max_length_decision"]
    assert "truncation_rate_at_final_max_len" in result["max_length_decision"]
    assert set(result["representation_audit"]) == {"validation", "test"}
    assert set(result["representation_audit"]["validation"]) >= {"oov_token_rate", "truncation_rate"}
    assert result["max_length_decision"]["candidates"] == [128, 192, 256]
    np.testing.assert_array_equal(vectorizer.transform(["externalonly known"])[0, :2], [1, vectorizer.token_to_id["known"]])


def test_data_freeze_deduplicates_hashes_even_when_labels_conflict(tmp_path):
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    import hashlib

    top_label_one = set(sorted(range(12), key=lambda row: hashlib.sha256(f"42:official_train:{row}".encode()).hexdigest())[:4])
    top_label_two = set(sorted(range(12, 24), key=lambda row: hashlib.sha256(f"42:official_train:{row}".encode()).hexdigest())[:4])
    train_rows = [
        {"Review": f"shared {sorted(top_label_one).index(index)}" if index in top_label_one else f"one only {index}", "Label": 1}
        for index in range(12)
    ] + [
        {"Review": f"shared {sorted(top_label_two).index(index)}" if index in top_label_two else f"two only {index}", "Label": 2}
        for index in range(12, 24)
    ]
    _write_csv(train_path, train_rows)
    _write_csv(test_path, _balanced_rows("isolated", 5))

    result = build_comments_data_freeze(
        train_path, test_path, tmp_path / "freeze", train_size=8, validation_size=4, test_size=4, seed=42
    )

    assert result["audit"]["after"]["train_validation"] == 0
    assert result["audit"]["after"]["train_test"] == 0
    assert result["audit"]["after"]["validation_test"] == 0


def test_max_length_decision_chooses_smallest_supported_candidate():
    decision = max_length_decision([2, 4, 130, 180, 260])
    assert decision["candidates"] == [128, 192, 256]
    assert decision["recommended_max_len"] == 256
    assert decision["selection_percentile"] == "p95"


def test_data_freeze_refuses_to_overwrite_a_different_immutable_artifact(tmp_path):
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    _write_csv(train_path, _balanced_rows("known", 5))
    _write_csv(test_path, _balanced_rows("isolated", 5))
    output = tmp_path / "freeze"
    build_comments_data_freeze(train_path, test_path, output, train_size=6, validation_size=2, test_size=4)
    (output / "train_indices.csv").write_text("not the immutable artifact", encoding="utf-8")

    import pytest

    with pytest.raises(FileExistsError, match="immutable artifact"):
        build_comments_data_freeze(train_path, test_path, output, train_size=6, validation_size=2, test_size=4)
