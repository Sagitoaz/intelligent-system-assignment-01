import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from src.assignment04.data_freeze import (
    DIABETES_PARITY_COUNTS,
    HOUSE_PARITY_COUNTS,
    build_diabetes_freeze,
    build_house_freeze,
    load_canonical_house_modeling_sample,
    load_frozen_manifest,
    verify_frozen_manifest,
    write_frozen_manifest,
)
from src.assignment04.diabetes_data import FEATURE_ORDER, TARGET


def _diabetes_rows(size=20):
    values = {feature: np.arange(size, dtype=float) for feature in FEATURE_ORDER}
    values[TARGET] = np.tile([0, 1], size // 2)
    return pd.DataFrame(values)


def _house_rows(size=30):
    rows = []
    for index in range(size):
        group = index // 2
        rows.append(
            {
                "street": f"{group} Main",
                "city": "Hanoi",
                "state": "VN",
                "zip_code": f"{10000 + group}",
                "price": 100_000 + index,
                "bed": 2,
                "bath": 1,
                "acre_lot": 0.1,
                "house_size": 1000,
                "status": "for_sale",
                "_source_row_id": 10_000 + index,
                "brokered_by": 1,
            }
        )
    rows[3]["street"] = None
    rows[11]["zip_code"] = ""
    frame = pd.DataFrame(rows)
    return frame


def test_diabetes_freeze_uses_explicit_feature_order_and_stratified_70_15_15_ids():
    """Fails if the freeze drops a feature, changes the order, or stops stratifying labels."""
    manifest = build_diabetes_freeze(
        _diabetes_rows(), expected_rows=20, parity_counts=None
    )

    assert manifest.feature_order == FEATURE_ORDER
    assert manifest.split_sizes == {"train": 14, "validation": 3, "test": 3}
    assert manifest.metadata["seed"] == 42
    assert manifest.metadata["source_rows"] == 20
    all_ids = set().union(*[set(ids) for ids in manifest.split_ids.values()])
    assert all_ids == set(range(20))
    for split_ids in manifest.split_ids.values():
        labels = _diabetes_rows().iloc[list(split_ids)][TARGET]
        assert labels.sum() in {1, 2, 7}


def test_diabetes_freeze_rejects_any_noncanonical_source_row_count():
    """Fails if a partial BRFSS file can silently become the frozen source."""
    with pytest.raises(ValueError, match="253680"):
        build_diabetes_freeze(_diabetes_rows())


def test_house_freeze_preserves_source_ids_and_never_splits_property_groups():
    """Fails if a property group leaks across partitions or source IDs are regenerated."""
    rows = _house_rows()
    rows.loc[1, "street"] = "  0 MAIN  "
    rows.loc[1, "city"] = " hAnOi "
    manifest = build_house_freeze(
        rows,
        sample_size=30,
        parity_counts={"train": 4, "validation": 2, "test": 2},
        ratio_tolerance=0.10,
    )

    assert manifest.metadata["source_rows"] == 30
    assert set().union(*[set(ids) for ids in manifest.split_ids.values()]) == set(rows["_source_row_id"])
    assert manifest.parity_ids is not None
    assert {name: len(ids) for name, ids in manifest.parity_ids.items()} == {
        "train": 4,
        "validation": 2,
        "test": 2,
    }
    parity_union = set().union(*[set(ids) for ids in manifest.parity_ids.values()])
    assert len(parity_union) == sum(len(ids) for ids in manifest.parity_ids.values())

    group_to_split = {}
    for split_name, ids in manifest.split_ids.items():
        for source_id in ids:
            key = manifest.group_keys[source_id]
            assert group_to_split.setdefault(key, split_name) == split_name
        assert set(manifest.parity_ids[split_name]).issubset(ids)

    assert sum(len(ids) for ids in manifest.split_ids.values()) == len(
        set().union(*[set(ids) for ids in manifest.split_ids.values()])
    )
    assert manifest.group_keys[10_000] == manifest.group_keys[10_001]
    assert manifest.group_keys[10_003] != manifest.group_keys[10_011]


def test_house_freeze_is_deterministic_and_manifest_is_immutable():
    """Fails if a run's frozen IDs/metadata can vary or be mutated after creation."""
    first = build_house_freeze(
        _house_rows(), sample_size=20, parity_counts={"train": 2, "validation": 1, "test": 1}, ratio_tolerance=0.10
    )
    second = build_house_freeze(
        _house_rows(), sample_size=20, parity_counts={"train": 2, "validation": 1, "test": 1}, ratio_tolerance=0.10
    )

    assert first == second
    with pytest.raises(TypeError):
        first.metadata["seed"] = 9
    with pytest.raises(TypeError):
        first.split_ids["train"] = ()
    with pytest.raises(Exception):
        first.metadata = {}


def test_default_house_parity_protocol_is_exact_40k_10k_10k():
    """Fails if the backend-comparison protocol stops requesting equal fixed partitions."""
    assert HOUSE_PARITY_COUNTS == {"train": 40_000, "validation": 10_000, "test": 10_000}


def test_diabetes_freeze_ids_and_stratified_parity_match_a03_split_calls():
    """Fails if A04 changes either the A03 split ordering or stratified parity sampling."""
    rows = _diabetes_rows(100)
    manifest = build_diabetes_freeze(rows, expected_rows=100, parity_counts={"train": 40, "validation": 10, "test": 10})
    x_train, x_temp, y_train, y_temp = train_test_split(
        rows.loc[:, FEATURE_ORDER], rows[TARGET], test_size=0.30, stratify=rows[TARGET], random_state=42
    )
    x_validation, x_test, _, _ = train_test_split(
        x_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )

    assert manifest.split_ids["train"] == tuple(x_train.index)
    assert manifest.split_ids["validation"] == tuple(x_validation.index)
    assert manifest.split_ids["test"] == tuple(x_test.index)
    assert DIABETES_PARITY_COUNTS == {"train": 40_000, "validation": 10_000, "test": 10_000}
    assert {name: len(ids) for name, ids in manifest.parity_ids.items()} == {
        "train": 40,
        "validation": 10,
        "test": 10,
    }
    assert sum(len(set(ids)) for ids in manifest.parity_ids.values()) == len(
        set().union(*[set(ids) for ids in manifest.parity_ids.values()])
    )
    for name, ids in manifest.parity_ids.items():
        assert rows.loc[list(ids), TARGET].sum() == len(ids) // 2
        assert set(ids).issubset(manifest.split_ids[name])


def test_canonical_house_loader_matches_a03_candidate_filter_and_sample_with_source_ids(tmp_path):
    """Fails if canonical loading changes A03's RNG calls, validity filter, or raw row identities."""
    raw = _house_rows(12).drop(columns="_source_row_id")
    raw.loc[1, "price"] = 0
    raw.loc[2, "house_size"] = 0
    raw.loc[3, "bed"] = -1
    path = tmp_path / "house.csv"
    raw.to_csv(path, index=False)

    sample, metadata = load_canonical_house_modeling_sample(
        path, expected_raw_rows=12, sample_size=6, chunksize=5
    )
    rng = np.random.default_rng(42)
    keep = rng.random(len(raw)) < min(1.0, 1.35 * 6 / 12)
    expected = raw.loc[keep].copy()
    valid = expected.price.notna() & expected.price.gt(0)
    valid &= expected.house_size.isna() | expected.house_size.gt(0)
    for column in ("bed", "bath", "acre_lot"):
        valid &= expected[column].isna() | expected[column].ge(0)
    expected = expected.loc[valid].sample(n=min(6, valid.sum()), random_state=42)

    assert sample["_source_row_id"].tolist() == expected.index.tolist()
    assert sample["price"].tolist() == expected.price.tolist()
    assert metadata["raw_rows"] == 12
    assert metadata["source_fingerprint"]["sha256"]


def test_house_freeze_rejects_invalid_or_unproven_modeling_rows():
    """Fails if bad target/features can bypass the loader and enter a frozen split."""
    rows = _house_rows()
    rows.loc[0, "price"] = 0
    with pytest.raises(ValueError, match="price"):
        build_house_freeze(rows, sample_size=30, parity_counts={"train": 4, "validation": 2, "test": 2})
    with pytest.raises(ValueError, match="modeling columns"):
        build_house_freeze(
            _house_rows().drop(columns="bed"), sample_size=30, parity_counts={"train": 4, "validation": 2, "test": 2}
        )


def test_house_freeze_records_ratio_audit_and_rejects_giant_groups():
    """Fails if group-aware allocation silently accepts an infeasible giant property group."""
    manifest = build_house_freeze(
        _house_rows(), sample_size=30, parity_counts={"train": 4, "validation": 2, "test": 2}, ratio_tolerance=0.10
    )
    audit = manifest.metadata["group_audit"]
    assert audit["zero_group_overlap"] is True
    assert set(audit["actual_ratios"]) == {"train", "validation", "test"}

    giant = _house_rows(30)
    giant.loc[:, ["street", "city", "state", "zip_code"]] = ["one", "X", "VN", "1"]
    with pytest.raises(ValueError, match="ratio tolerance"):
        build_house_freeze(giant, sample_size=30, parity_counts={"train": 4, "validation": 2, "test": 2})


def test_durable_manifest_and_index_are_verified_atomic_and_refuse_different_overwrite(tmp_path):
    """Fails if on-disk frozen IDs lose provenance/checksums or overwrite another freeze."""
    manifest = build_diabetes_freeze(
        _diabetes_rows(100), expected_rows=100, parity_counts={"train": 40, "validation": 10, "test": 10}
    )
    path = tmp_path / "diabetes.freeze.json"
    index_path = write_frozen_manifest(manifest, path)

    assert index_path.exists()
    assert load_frozen_manifest(path) == manifest
    assert verify_frozen_manifest(path) is True
    assert "Diabetes_binary" not in index_path.read_text(encoding="utf-8")
    assert write_frozen_manifest(manifest, path) == index_path

    altered = build_diabetes_freeze(
        _diabetes_rows(100).assign(BMI=lambda values: values.BMI + 1),
        expected_rows=100,
        parity_counts={"train": 40, "validation": 10, "test": 10},
    )
    with pytest.raises(FileExistsError, match="different"):
        write_frozen_manifest(altered, path)
    index_path.write_text(index_path.read_text(encoding="utf-8") + "tamper\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        verify_frozen_manifest(path)
