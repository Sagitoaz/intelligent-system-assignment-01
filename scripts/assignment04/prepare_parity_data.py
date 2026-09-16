"""Freeze A04 master IDs and materialize only the controlled parity arrays."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.assignment04.comments_data import SharedTextVectorizer, build_comments_data_freeze
from src.assignment04.config import COMMENTS_TEST, COMMENTS_TRAIN, DIABETES_DATA, HOUSE_DATA
from src.assignment04.data_freeze import (
    HOUSE_SOURCE_ROW_ID,
    build_diabetes_freeze,
    build_house_freeze,
    load_canonical_house_modeling_sample,
    write_frozen_manifest,
)
from src.assignment04.diabetes_data import TARGET
from src.assignment04.prepared_data import DiabetesPreprocessor, HousePreprocessor, load_comment_rows


MANIFEST_DIR = REPO_ROOT / "results/assignment04/manifests"
PREPARED_DIR = REPO_ROOT / "results/assignment04/parity/prepared"


def _json_bytes(payload):
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")


def _plain(value):
    if hasattr(value, "items"):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_immutable_json(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    content = _json_bytes(payload)
    if path.exists():
        if path.read_bytes() != content:
            raise FileExistsError(f"immutable metadata differs: {path}")
        return
    path.write_bytes(content)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1_048_576), b""):
            digest.update(block)
    return digest.hexdigest()


def _save_npz(path, **arrays):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return {"path": str(path.relative_to(REPO_ROOT)), "sha256": _sha256(path), "bytes": path.stat().st_size}


def prepare_diabetes():
    frame = pd.read_csv(REPO_ROOT / DIABETES_DATA)
    manifest = build_diabetes_freeze(frame)
    write_frozen_manifest(manifest, MANIFEST_DIR / "diabetes.json")
    preprocessor = DiabetesPreprocessor().fit(frame.loc[list(manifest.split_ids["train"])])
    arrays = {}
    split_summary = {}
    for split in ("train", "validation", "test"):
        master_ids = list(manifest.split_ids[split]); parity_ids = list(manifest.parity_ids[split])
        parity = frame.loc[parity_ids]
        arrays[f"{split}_x"] = preprocessor.transform(parity)
        arrays[f"{split}_y"] = parity[TARGET].to_numpy(dtype=np.int64)
        split_summary[split] = {
            "master_rows": len(master_ids), "parity_rows": len(parity_ids),
            "master_class_counts": {str(k): int(v) for k, v in frame.loc[master_ids, TARGET].value_counts().sort_index().items()},
            "parity_class_counts": {str(k): int(v) for k, v in parity[TARGET].value_counts().sort_index().items()},
        }
    artifact = _save_npz(PREPARED_DIR / "diabetes_parity.npz", **arrays)
    metadata = {**preprocessor.metadata(), "dataset": "diabetes", "problem": "classification",
                "vocabulary_size": None, "sequence_length": 21, "splits": split_summary, "prepared_artifact": artifact}
    _write_immutable_json(MANIFEST_DIR / "diabetes_preprocessing.json", metadata)
    print(json.dumps(metadata, indent=2), flush=True)


def prepare_house():
    frame, loader_metadata = load_canonical_house_modeling_sample(REPO_ROOT / HOUSE_DATA)
    manifest = build_house_freeze(frame, loader_metadata=loader_metadata)
    write_frozen_manifest(manifest, MANIFEST_DIR / "house.json")
    indexed = frame.set_index(HOUSE_SOURCE_ROW_ID, drop=False)
    preprocessor = HousePreprocessor().fit(indexed.loc[list(manifest.split_ids["train"])])
    arrays, split_summary = {}, {}
    quantiles = [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]
    for split in ("train", "validation", "test"):
        master_ids = list(manifest.split_ids[split]); parity_ids = list(manifest.parity_ids[split])
        parity = indexed.loc[parity_ids]
        values = preprocessor.transform(parity)
        for name, value in values.items():
            arrays[f"{split}_{name}"] = value
        split_summary[split] = {
            "master_rows": len(master_ids), "parity_rows": len(parity_ids),
            "target_quantiles_usd": {str(q): float(v) for q, v in indexed.loc[master_ids, "price"].quantile(quantiles).items()},
        }
    artifact = _save_npz(PREPARED_DIR / "house_parity.npz", **arrays)
    metadata = {**preprocessor.metadata(), "dataset": "house", "problem": "regression",
                "state_size": preprocessor.state_size, "status_size": preprocessor.status_size,
                "sequence_length": 6, "splits": split_summary,
                "group_audit": _plain(manifest.metadata["group_audit"]), "loader": _plain(loader_metadata),
                "prepared_artifact": artifact}
    _write_immutable_json(MANIFEST_DIR / "house_preprocessing.json", metadata)
    print(json.dumps(metadata, indent=2), flush=True)


def prepare_comments():
    output = MANIFEST_DIR / "comments"
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for relative_path, expected in manifest["artifact_sha256"].items():
            if _sha256(output / relative_path) != expected:
                raise ValueError(f"Comments immutable artifact checksum failed: {relative_path}")
    else:
        manifest = build_comments_data_freeze(REPO_ROOT / COMMENTS_TRAIN, REPO_ROOT / COMMENTS_TEST, output)
    if manifest["audit"]["material_design_change"]:
        raise RuntimeError(f"Comments cross-split duplicate rate is material; stop before training: {manifest['audit']}")
    vectorizer = SharedTextVectorizer.load(output / "vocabulary.json")
    arrays, split_summary = {}, {}
    for split, source, source_name in (
        ("train", REPO_ROOT / COMMENTS_TRAIN, "official_train"),
        ("validation", REPO_ROOT / COMMENTS_TRAIN, "official_train"),
        ("test", REPO_ROOT / COMMENTS_TEST, "official_test"),
    ):
        texts, labels = load_comment_rows(source, output / f"parity_{split}_indices.csv", expected_source=source_name)
        arrays[f"{split}_ids"] = vectorizer.transform(texts)
        arrays[f"{split}_y"] = labels
        split_summary[split] = {
            "master_rows": manifest["sizes"][split], "parity_rows": len(labels),
            "parity_class_counts": {str(k): int(v) for k, v in zip(*np.unique(labels, return_counts=True))},
        }
    artifact = _save_npz(PREPARED_DIR / "comments_parity.npz", **arrays)
    metadata = {
        "dataset": "comments", "problem": "classification", "fit_split": "master_train_only",
        "sequence_length": vectorizer.sequence_length, "vocabulary_size": len(vectorizer.token_to_id),
        "max_vocabulary": vectorizer.max_tokens, "pad_id": 0, "oov_id": 1, "embedding_dim": 16,
        "splits": split_summary, "duplicate_audit": manifest["audit"],
        "token_length_statistics": manifest["token_length_statistics"],
        "max_length_decision": manifest["max_length_decision"],
        "representation_audit": manifest["representation_audit"], "prepared_artifact": artifact,
    }
    _write_immutable_json(MANIFEST_DIR / "comments_preprocessing.json", metadata)
    print(json.dumps(metadata, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("diabetes", "house", "comments", "all"), default="all")
    args = parser.parse_args()
    functions = {"diabetes": prepare_diabetes, "house": prepare_house, "comments": prepare_comments}
    for name, function in functions.items():
        if args.task in (name, "all"):
            print(f"PREPARING {name}", flush=True); function()


if __name__ == "__main__":
    main()
