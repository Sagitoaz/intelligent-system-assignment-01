"""Materialize the frozen A04 master splits without creating new splits.

Large Comments token arrays are stored as memory-mapped ``.npy`` files so the
800k/100k/200k design remains practical on a 16 GB CPU-only machine.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .comments_data import SharedTextVectorizer, normalized_text_sha256
from .data_freeze import HOUSE_SOURCE_ROW_ID, load_canonical_house_modeling_sample
from .diabetes_data import FEATURE_ORDER, TARGET


SPLITS = ("train", "validation", "test")


def write_array(path, values):
    """Write once, or verify an existing derived array before reusing it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.asarray(values)
    if path.exists():
        existing = np.load(path, mmap_mode="r")
        if existing.shape != values.shape or existing.dtype != values.dtype or not np.array_equal(existing, values):
            raise FileExistsError(f"existing full-scale array differs: {path}")
        return path
    np.save(path, values)
    return path


def _manifest_ids(index_csv):
    table = pd.read_csv(index_csv, usecols=["source_id_json", "split"])
    table["source_id"] = table["source_id_json"].map(
        lambda value: json.loads(value) if isinstance(value, (str, bytes, bytearray)) else int(value)
    )
    return {name: table.loc[table.split.eq(name), "source_id"].tolist() for name in SPLITS}


def _save_split_arrays(task_dir, split, values):
    for name, value in values.items():
        write_array(Path(task_dir) / f"{split}_{name}.npy", value)


def prepare_diabetes(source_csv, index_csv, preprocessing_json, output_dir):
    metadata = json.loads(Path(preprocessing_json).read_text(encoding="utf-8"))
    frame = pd.read_csv(source_csv)
    ids = _manifest_ids(index_csv)
    positions = [FEATURE_ORDER.index(name) for name in metadata["scaled_features"]]
    means, scales = np.asarray(metadata["means"]), np.asarray(metadata["scales"])
    counts = {}
    for split in SPLITS:
        part = frame.loc[ids[split]]
        x = part.loc[:, FEATURE_ORDER].to_numpy(dtype=np.float32, copy=True)
        x[:, positions] = (x[:, positions] - means) / scales
        values = {"x": x[:, None, :], "y": part[TARGET].to_numpy(dtype=np.int64)}
        _save_split_arrays(Path(output_dir) / "diabetes", split, values)
        counts[split] = len(part)
    return counts


def _encode_categories(values, mapping):
    return np.asarray([0 if pd.isna(value) else mapping.get(str(value), 0) for value in values], dtype=np.int64)


def prepare_house(source_csv, index_csv, preprocessing_json, output_dir):
    metadata = json.loads(Path(preprocessing_json).read_text(encoding="utf-8"))
    frame, loader = load_canonical_house_modeling_sample(source_csv)
    expected = metadata["loader"]["source_fingerprint"]
    if loader["source_fingerprint"]["sha256"] != expected["sha256"]:
        raise ValueError("House source fingerprint differs from the frozen preprocessing metadata")
    indexed = frame.set_index(HOUSE_SOURCE_ROW_ID, drop=False)
    ids = _manifest_ids(index_csv)
    columns = metadata["numeric_columns"]
    medians = np.asarray(metadata["numeric_medians"])
    means, scales = np.asarray(metadata["numeric_means"]), np.asarray(metadata["numeric_scales"])
    counts = {}
    for split in SPLITS:
        part = indexed.loc[ids[split]]
        numeric = part.loc[:, columns].to_numpy(dtype=np.float64)
        numeric = np.where(np.isnan(numeric), medians, numeric)
        numeric = ((numeric - means) / scales).astype(np.float32)
        y = np.log1p(part["price"].to_numpy(dtype=np.float64))
        y = ((y - metadata["target_mean"]) / metadata["target_scale"]).astype(np.float32)[:, None]
        values = {
            "numeric": numeric,
            "state": _encode_categories(part["state"], metadata["state_vocabulary"]),
            "status": _encode_categories(part["status"], metadata["status_vocabulary"]),
            "y": y,
        }
        _save_split_arrays(Path(output_dir) / "house", split, values)
        counts[split] = len(part)
    return counts


def _comment_index(path, expected_source):
    table = pd.read_csv(path, usecols=["source", "row_index", "text_sha256"])
    if not table.source.eq(expected_source).all() or table.row_index.duplicated().any():
        raise ValueError(f"invalid frozen Comments index: {path}")
    return table


def _materialize_comment_source(source_csv, assignments, vectorizer, task_dir):
    max_row = max(int(table.row_index.max()) for _, table in assignments)
    split_code = np.full(max_row + 1, -1, dtype=np.int8)
    offsets = np.full(max_row + 1, -1, dtype=np.int32)
    hashes = {}
    outputs = {}
    for code, (split, table) in enumerate(assignments):
        rows = table.row_index.to_numpy(dtype=np.int64)
        split_code[rows] = code
        offsets[rows] = np.arange(len(table), dtype=np.int32)
        hashes[code] = table.text_sha256.to_numpy()
        ids_path = Path(task_dir) / f"{split}_ids.npy"
        y_path = Path(task_dir) / f"{split}_y.npy"
        if ids_path.exists() or y_path.exists():
            if not (ids_path.exists() and y_path.exists()):
                raise FileExistsError(f"partial Comments artifact exists for {split}")
            ids = np.load(ids_path, mmap_mode="r+")
            labels = np.load(y_path, mmap_mode="r+")
            if ids.shape != (len(table), vectorizer.sequence_length) or ids.dtype != np.int32 or labels.shape != (len(table),):
                raise ValueError(f"existing Comments artifact shape/dtype differs for {split}")
        else:
            ids = np.lib.format.open_memmap(ids_path, mode="w+", dtype=np.int32,
                                            shape=(len(table), vectorizer.sequence_length))
            labels = np.lib.format.open_memmap(y_path, mode="w+", dtype=np.int64, shape=(len(table),))
        outputs[code] = (ids, labels)

    found = np.zeros(len(assignments), dtype=np.int64)
    pending = []

    def flush_pending():
        if not pending:
            return
        encoded = vectorizer.transform([item[2] for item in pending])
        for item, token_ids in zip(pending, encoded, strict=True):
            code, offset, _, raw_label = item
            outputs[code][0][offset] = token_ids
            outputs[code][1][offset] = raw_label - 1
            found[code] += 1
        pending.clear()

    with Path(source_csv).open("r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not {"Review", "Label"}.issubset(reader.fieldnames):
            raise ValueError("Comments CSV must contain Review and Label")
        for row_index, row in enumerate(reader):
            if row_index >= len(split_code):
                break
            code = int(split_code[row_index])
            if code < 0:
                continue
            offset = int(offsets[row_index])
            text = row.get("Review") or ""
            if normalized_text_sha256(text) != hashes[code][offset]:
                raise ValueError(f"Comments frozen text hash mismatch at row {row_index}")
            raw_label = int(row["Label"])
            if raw_label not in (1, 2):
                raise ValueError(f"unsupported Comments label {raw_label}")
            pending.append((code, offset, text, raw_label))
            if len(pending) >= 4096:
                flush_pending()
    flush_pending()
    for code, (split, table) in enumerate(assignments):
        outputs[code][0].flush(); outputs[code][1].flush()
        if found[code] != len(table):
            raise ValueError(f"materialized {found[code]}/{len(table)} Comments rows for {split}")


def prepare_comments(train_csv, test_csv, manifest_dir, preprocessing_json, output_dir):
    metadata = json.loads(Path(preprocessing_json).read_text(encoding="utf-8"))
    freeze = json.loads((Path(manifest_dir) / "manifest.json").read_text(encoding="utf-8"))
    for name, source in (("train", train_csv), ("test", test_csv)):
        digest = hashlib.sha256()
        with Path(source).open("rb") as stream:
            for block in iter(lambda: stream.read(1_048_576), b""):
                digest.update(block)
        if digest.hexdigest() != freeze["source_sha256"][name]:
            raise ValueError(f"Comments {name} source differs from immutable manifest")
    for relative, expected in freeze["artifact_sha256"].items():
        digest = hashlib.sha256((Path(manifest_dir) / relative).read_bytes()).hexdigest()
        if digest != expected:
            raise ValueError(f"Comments manifest artifact differs: {relative}")
    vectorizer = SharedTextVectorizer.load(Path(manifest_dir) / "vocabulary.json")
    if vectorizer.sequence_length != 192 or vectorizer.max_tokens != 20_000:
        raise ValueError("Comments full-scale contract requires MAX_LEN=192 and max vocabulary=20,000")
    train = _comment_index(Path(manifest_dir) / "train_indices.csv", "official_train")
    validation = _comment_index(Path(manifest_dir) / "validation_indices.csv", "official_train")
    test = _comment_index(Path(manifest_dir) / "test_indices.csv", "official_test")
    task_dir = Path(output_dir) / "comments"; task_dir.mkdir(parents=True, exist_ok=True)
    _materialize_comment_source(train_csv, [("train", train), ("validation", validation)], vectorizer, task_dir)
    _materialize_comment_source(test_csv, [("test", test)], vectorizer, task_dir)
    return {"train": len(train), "validation": len(validation), "test": len(test)}


def load_full_splits(prepared_dir, task):
    task_dir = Path(prepared_dir) / task
    names = {"diabetes": ("x", "y"), "house": ("numeric", "state", "status", "y"),
             "comments": ("ids", "y")}[task]
    result = {}
    for split in SPLITS:
        values = {name: np.load(task_dir / f"{split}_{name}.npy", mmap_mode="r") for name in names}
        if len({len(value) for value in values.values()}) != 1:
            raise ValueError(f"full-scale row count mismatch for {task}/{split}")
        result[split] = values
    return result
