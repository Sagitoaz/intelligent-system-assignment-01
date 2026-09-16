"""Leakage-safe, reproducible text preparation for Assignment 04 Comments.

The source CSVs are never rewritten. A freeze contains only row-index files,
a manifest, and a vocabulary fitted exclusively on frozen training text.
"""

from __future__ import annotations

from collections import Counter
import csv
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
import re
import sqlite3
import tempfile
import unicodedata

import numpy as np


TOKEN_PATTERN = re.compile(r"[^\W_]+(?:['â€™][^\W_]+)?", flags=re.UNICODE)
NORMALIZATION_DESCRIPTION = "NFC + lowercase + collapsed whitespace"
HASH_DESCRIPTION = "sha256(normalized UTF-8 text)"
PARITY_SPLIT_SIZES = {"train": 20_000, "validation": 5_000, "test": 10_000}


def normalize_text(text):
    """Canonical normalization shared by tokenization and duplicate identity."""
    normalized = unicodedata.normalize("NFC", "" if text is None else str(text)).lower()
    return " ".join(normalized.split())


def tokenize(text):
    return TOKEN_PATTERN.findall(normalize_text(text))


def normalized_text_sha256(text):
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def _text_hash(text):
    """Backward-compatible name for the normalized text SHA-256."""
    return normalized_text_sha256(text)


class SharedTextVectorizer:
    """Deterministic train-only token vocabulary with PAD=0 and OOV=1."""

    def __init__(self, max_tokens=20_000, sequence_length=128):
        if max_tokens < 2:
            raise ValueError("max_tokens must include PAD and OOV")
        if sequence_length < 1:
            raise ValueError("sequence_length must be positive")
        self.max_tokens = max_tokens
        self.sequence_length = sequence_length

    def fit(self, texts):
        counts = Counter(token for text in texts for token in tokenize(text))
        ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        self.token_to_id = {"<PAD>": 0, "<OOV>": 1}
        self.token_to_id.update({token: index + 2 for index, (token, _) in enumerate(ordered[: self.max_tokens - 2])})
        return self

    def transform(self, texts):
        if not hasattr(self, "token_to_id"):
            raise RuntimeError("fit must be called on training text first")
        result = np.zeros((len(texts), self.sequence_length), dtype=np.int32)
        for row, text in enumerate(texts):
            ids = [self.token_to_id.get(token, 1) for token in tokenize(text)[: self.sequence_length]]
            result[row, : len(ids)] = ids
        return result

    def to_dict(self):
        if not hasattr(self, "token_to_id"):
            raise RuntimeError("fit must be called before serialization")
        return {
            "normalization": NORMALIZATION_DESCRIPTION,
            "tokenizer": TOKEN_PATTERN.pattern,
            "max_tokens": self.max_tokens,
            "sequence_length": self.sequence_length,
            "token_to_id": self.token_to_id,
        }

    @classmethod
    def from_dict(cls, payload):
        vectorizer = cls(int(payload["max_tokens"]), int(payload["sequence_length"]))
        vectorizer.token_to_id = {token: int(index) for token, index in payload["token_to_id"].items()}
        if vectorizer.token_to_id.get("<PAD>") != 0 or vectorizer.token_to_id.get("<OOV>") != 1:
            raise ValueError("serialized vocabulary must reserve PAD=0 and OOV=1")
        return vectorizer

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def duplicate_hash_overlap(train_texts, test_texts):
    train_hashes = {normalized_text_sha256(text) for text in train_texts}
    test_hashes = {normalized_text_sha256(text) for text in test_texts}
    return {"train_unique": len(train_hashes), "test_unique": len(test_hashes), "overlap_unique": len(train_hashes & test_hashes)}


@dataclass(frozen=True)
class _FrozenRow:
    source: str
    row_index: int
    label: str
    text_hash: str


def _priority(seed, source, row_index):
    """Process-stable ordering key, independent of CSV chunking."""
    return hashlib.sha256(f"{seed}:{source}:{row_index}".encode("utf-8")).hexdigest()


def _insert_source(connection, source, path, review_column, label_column, seed):
    with Path(path).open("r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {review_column, label_column}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path} must contain columns {sorted(required)}")
        batch = []
        for row_index, row in enumerate(reader):
            label = str(row[label_column]).strip()
            if label:
                batch.append((source, row_index, label, bytes.fromhex(normalized_text_sha256(row.get(review_column))), _priority(seed, source, row_index)))
            if len(batch) == 10_000:
                connection.executemany("INSERT INTO records VALUES (?, ?, ?, ?, ?)", batch)
                batch.clear()
        if batch:
            connection.executemany("INSERT INTO records VALUES (?, ?, ?, ?, ?)", batch)


def _records_in_priority_order(connection, source, label, limit=None):
    query = "SELECT row_index, label, text_hash FROM records WHERE source = ? AND label = ? ORDER BY priority, row_index"
    parameters = [source, label]
    if limit is not None:
        query += " LIMIT ?"
        parameters.append(limit)
    cursor = connection.execute(query, parameters)
    for row_index, result_label, text_hash in cursor:
        yield _FrozenRow(source, int(row_index), result_label, bytes(text_hash).hex())


def _take_unique(rows, wanted, blocked_hashes=()):
    blocked = set(blocked_hashes)
    chosen = []
    for record in rows:
        if record.text_hash not in blocked:
            chosen.append(record)
            blocked.add(record.text_hash)
            if len(chosen) == wanted:
                return chosen
    raise ValueError(f"not enough unique examples to freeze a balanced split of {wanted} rows")


def _raw_partition(connection, label, train_quota, validation_quota):
    rows = list(_records_in_priority_order(connection, "official_train", label, train_quota + validation_quota))
    if len(rows) != train_quota + validation_quota:
        raise ValueError(f"official train has too few label {label!r} rows")
    return rows[:train_quota], rows[train_quota:]


def _overlap_audit(splits):
    hashes = {name: {row.text_hash for row in rows} for name, rows in splits.items()}
    return {
        "train_validation": len(hashes["train"] & hashes["validation"]),
        "train_test": len(hashes["train"] & hashes["test"]),
        "validation_test": len(hashes["validation"] & hashes["test"]),
    }


def _per_class(size, classes, split_name):
    if size <= 0 or size % len(classes):
        raise ValueError(f"{split_name}_size must be positive and divisible by {len(classes)} classes")
    return size // len(classes)


def _selected_texts(path, row_indices, review_column):
    selected = set(row_indices)
    with Path(path).open("r", newline="", encoding="utf-8") as stream:
        for row_index, row in enumerate(csv.DictReader(stream)):
            if row_index in selected:
                yield row.get(review_column)


def token_length_statistics(texts):
    """Exact token-length summary, keeping only compact integer lengths."""
    lengths = np.fromiter((len(tokenize(text)) for text in texts), dtype=np.int32)
    if not len(lengths):
        raise ValueError("token statistics require at least one training text")
    return {
        "count": int(len(lengths)), "min": int(lengths.min()), "p50": float(np.quantile(lengths, 0.50)),
        "p75": float(np.quantile(lengths, 0.75)),
        "p90": float(np.quantile(lengths, 0.90)), "p95": float(np.quantile(lengths, 0.95)),
        "p99": float(np.quantile(lengths, 0.99)), "max": int(lengths.max()),
        "truncation_rate_at_128": float(np.mean(lengths > 128)),
        "truncation_rates": {str(length): float(np.mean(lengths > length)) for length in (128, 192, 256)},
    }


def max_length_decision_from_statistics(statistics, candidates=(128, 192, 256)):
    choices = sorted({int(candidate) for candidate in candidates})
    if choices != [128, 192, 256]:
        raise ValueError("supported sequence-length choices are exactly 128, 192, and 256")
    recommended = next((choice for choice in choices if choice >= statistics["p95"]), choices[-1])
    return {
        "selection_percentile": "p95", "candidates": choices, "recommended_max_len": recommended,
        "train_token_lengths": statistics, "p95_exceeds_largest_candidate": statistics["p95"] > choices[-1],
        "truncation_rate_at_recommended_max_len": statistics["truncation_rates"][str(recommended)],
        "truncation_rate_at_final_max_len": statistics["truncation_rates"][str(recommended)],
    }


def max_length_decision(lengths, candidates=(128, 192, 256)):
    """Decision helper for token counts; production uses train-only statistics."""
    values = np.asarray(list(lengths), dtype=np.int32)
    if not len(values):
        raise ValueError("token statistics require at least one training text")
    statistics = {
        "count": int(len(values)), "min": int(values.min()), "p50": float(np.quantile(values, .50)), "p75": float(np.quantile(values, .75)),
        "p90": float(np.quantile(values, .90)), "p95": float(np.quantile(values, .95)),
        "p99": float(np.quantile(values, .99)), "max": int(values.max()),
        "truncation_rate_at_128": float(np.mean(values > 128)),
        "truncation_rates": {str(length): float(np.mean(values > length)) for length in (128, 192, 256)},
    }
    return max_length_decision_from_statistics(statistics, candidates)


def _write_immutable(path, content):
    path = Path(path)
    if path.exists():
        if path.read_bytes() != content:
            raise FileExistsError(f"immutable artifact differs: {path}")
        return
    path.write_bytes(content)


def _write_indices(path, rows):
    with io.StringIO(newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["source", "row_index", "text_sha256"])
        writer.writeheader()
        for row in rows:
            writer.writerow({"source": row.source, "row_index": row.row_index, "text_sha256": row.text_hash})
        _write_immutable(path, stream.getvalue().encode("utf-8"))


def _artifact_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _representation_audit(texts, vectorizer):
    examples = tokens = oov = truncated = 0
    for text in texts:
        token_list = tokenize(text)
        examples += 1
        tokens += len(token_list)
        oov += sum(token not in vectorizer.token_to_id for token in token_list)
        truncated += len(token_list) > vectorizer.sequence_length
    return {
        "examples": examples, "tokens": tokens, "max_len": vectorizer.sequence_length,
        "oov_token_rate": float(oov / tokens) if tokens else 0.0,
        "truncation_rate": float(truncated / examples) if examples else 0.0,
    }


def _class_counts(splits, classes):
    return {name: {label: sum(row.label == label for row in rows) for label in classes} for name, rows in splits.items()}


def build_comments_data_freeze(
    official_train_csv, official_test_csv, output_dir, *, train_size=800_000, validation_size=100_000,
    test_size=200_000, seed=42, review_column="Review", label_column="Label", max_tokens=20_000, embedding_dim=16,
    material_duplicate_rate=0.01,
):
    """Build deterministic balanced, index-only train/validation/test manifests.

    The temporary SQLite index makes selection memory-conscious for multi-GB
    CSVs. Train is selected first; validation then excludes train hashes; test
    excludes both. Each later split scans its own official source to refill
    deterministic quotas after duplicate filtering.
    """
    if max_tokens != 20_000 or embedding_dim != 16:
        raise ValueError("Comments freeze fixes vocabulary=20,000 and embedding_dim=16")
    train_path, test_path = Path(official_train_csv), Path(official_test_csv)
    if train_path.resolve() == test_path.resolve():
        raise ValueError("official train and official test must be different files")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="comments-freeze-") as temporary_directory:
        connection = sqlite3.connect(Path(temporary_directory) / "rows.sqlite")
        try:
            connection.execute("CREATE TABLE records (source TEXT, row_index INTEGER, label TEXT, text_hash BLOB, priority TEXT)")
            _insert_source(connection, "official_train", train_path, review_column, label_column, seed)
            _insert_source(connection, "official_test", test_path, review_column, label_column, seed)
            connection.execute("CREATE INDEX records_priority ON records(source, label, priority, row_index)")
            connection.commit()
            classes = [row[0] for row in connection.execute("SELECT DISTINCT label FROM records WHERE source = ? ORDER BY label", ("official_train",))]
            test_classes = [row[0] for row in connection.execute("SELECT DISTINCT label FROM records WHERE source = ? ORDER BY label", ("official_test",))]
            if classes != test_classes or len(classes) != 2:
                raise ValueError("official train/test must contain the same exactly two labels")
            train_quota = _per_class(train_size, classes, "train")
            validation_quota = _per_class(validation_size, classes, "validation")
            test_quota = _per_class(test_size, classes, "test")
            raw = {"train": [], "validation": [], "test": []}
            for label in classes:
                raw_train, raw_validation = _raw_partition(connection, label, train_quota, validation_quota)
                raw["train"].extend(raw_train)
                raw["validation"].extend(raw_validation)
                raw_test = list(_records_in_priority_order(connection, "official_test", label, test_quota))
                if len(raw_test) != test_quota:
                    raise ValueError(f"official test has too few label {label!r} rows")
                raw["test"].extend(raw_test)
            selected = {"train": [], "validation": [], "test": []}
            train_hashes = set()
            for label in classes:
                selected["train"].extend(
                    _take_unique(_records_in_priority_order(connection, "official_train", label), train_quota, train_hashes)
                )
                train_hashes.update(row.text_hash for row in selected["train"] if row.label == label)
            validation_hashes = set(train_hashes)
            for label in classes:
                selected["validation"].extend(
                    _take_unique(_records_in_priority_order(connection, "official_train", label), validation_quota, validation_hashes)
                )
                validation_hashes.update(row.text_hash for row in selected["validation"] if row.label == label)
            blocked_for_test = set(validation_hashes)
            for label in classes:
                selected["test"].extend(
                    _take_unique(_records_in_priority_order(connection, "official_test", label), test_quota, blocked_for_test)
                )
                blocked_for_test.update(row.text_hash for row in selected["test"] if row.label == label)
        finally:
            connection.close()

    for split, rows in selected.items():
        _write_indices(output / f"{split}_indices.csv", rows)
    parity = {}
    for split, size in PARITY_SPLIT_SIZES.items():
        quota = _per_class(size, classes, f"parity_{split}")
        parity[split] = []
        for label in classes:
            parity[split].extend([row for row in selected[split] if row.label == label][:quota])
        _write_indices(output / f"parity_{split}_indices.csv", parity[split])

    train_indices = [row.row_index for row in selected["train"]]
    train_statistics = token_length_statistics(_selected_texts(train_path, train_indices, review_column))
    decision = max_length_decision_from_statistics(train_statistics)
    vectorizer = SharedTextVectorizer(max_tokens=max_tokens, sequence_length=decision["recommended_max_len"])
    vectorizer.fit(_selected_texts(train_path, train_indices, review_column))
    vocabulary_path = output / "vocabulary.json"
    _write_immutable(vocabulary_path, json.dumps(vectorizer.to_dict(), ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8"))
    representation_audit = {
        "validation": _representation_audit(
            _selected_texts(train_path, [row.row_index for row in selected["validation"]], review_column), vectorizer
        ),
        "test": _representation_audit(
            _selected_texts(test_path, [row.row_index for row in selected["test"]], review_column), vectorizer
        ),
    }
    artifact_paths = [
        "train_indices.csv", "validation_indices.csv", "test_indices.csv", "parity_train_indices.csv",
        "parity_validation_indices.csv", "parity_test_indices.csv", "vocabulary.json",
    ]
    before_audit = _overlap_audit(raw)
    before_rates = {
        "train_validation": before_audit["train_validation"] / validation_size,
        "train_test": before_audit["train_test"] / test_size,
        "validation_test": before_audit["validation_test"] / test_size,
    }
    manifest = {
        "seed": seed, "sources": {"train": str(train_path), "test": str(test_path)},
        "source_sha256": {"train": _artifact_sha256(train_path), "test": _artifact_sha256(test_path)},
        "source_roles": {"train": "official_train only", "validation": "official_train only", "test": "official_test only"},
        "normalization": NORMALIZATION_DESCRIPTION, "text_hash": HASH_DESCRIPTION,
        "sizes": {name: len(rows) for name, rows in selected.items()}, "class_counts": _class_counts(selected, classes),
        "audit": {
            "before": before_audit, "before_rates": before_rates, "after": _overlap_audit(selected),
            "material_rate_threshold": material_duplicate_rate,
            "material_design_change": max(before_rates.values()) >= material_duplicate_rate,
        },
        "deduplication": "train-priority; validation excludes train; test excludes train and validation; deterministic refill from own official source",
        "token_length_statistics": {"train": train_statistics}, "max_length_decision": decision,
        "representation_audit": representation_audit,
        "vocabulary": {"fit_split": "train", "max_tokens": max_tokens, "pad_id": 0, "oov_id": 1, "embedding_dim": embedding_dim, "path": "vocabulary.json"},
        "parity": {"balanced_sizes": PARITY_SPLIT_SIZES, "index_files": {name: f"parity_{name}_indices.csv" for name in PARITY_SPLIT_SIZES}},
        "artifact_sha256": {path: _artifact_sha256(output / path) for path in artifact_paths},
    }
    _write_immutable(output / "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8"))
    return manifest
