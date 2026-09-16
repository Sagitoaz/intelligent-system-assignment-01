"""Deterministic, auditable tabular-data freezes for Assignment 04.

The module creates row-ID manifests only: targets and model features never
enter the on-disk index. Preprocessing remains train-fitted by consumers.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Hashable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import RANDOM_STATE
from .diabetes_data import FEATURE_ORDER, TARGET, validate_schema
from .house_data import IDENTITY_COLUMNS

DIABETES_FULL_ROWS = 253_680
HOUSE_SOURCE_SAMPLE_SIZE = 400_000
HOUSE_EXPECTED_RAW_ROWS = 2_226_382
DIABETES_PARITY_COUNTS = MappingProxyType({"train": 40_000, "validation": 10_000, "test": 10_000})
HOUSE_PARITY_COUNTS = MappingProxyType({"train": 40_000, "validation": 10_000, "test": 10_000})
HOUSE_RATIO_TOLERANCE = 0.02
HOUSE_TARGET = "price"
HOUSE_SOURCE_ROW_ID = "_source_row_id"
HOUSE_MODEL_COLUMNS = ("bed", "bath", "acre_lot", "house_size", "state", "status", HOUSE_TARGET)
HOUSE_PROFILE_COLUMNS = (*HOUSE_MODEL_COLUMNS, "street", "city", "zip_code", "brokered_by")
_SPLIT_NAMES = ("train", "validation", "test")
_SPLIT_RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}
_FORMAT_VERSION = 1


def _freeze_mapping(values: Mapping[Hashable, Any]) -> Mapping[Hashable, Any]:
    return MappingProxyType({
        key: _freeze_mapping(value) if isinstance(value, Mapping) else tuple(value) if isinstance(value, list) else value
        for key, value in values.items()
    })


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(_thaw(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_fingerprint(path: Path) -> Mapping[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1_048_576), b""):
            digest.update(block)
    return {"algorithm": "sha256", "sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def _frame_fingerprint(frame: pd.DataFrame) -> Mapping[str, Any]:
    digest = hashlib.sha256()
    digest.update(_canonical_bytes({"columns": list(frame.columns), "dtypes": [str(dtype) for dtype in frame.dtypes]}))
    digest.update(pd.util.hash_pandas_object(frame, index=True).to_numpy(dtype=np.uint64).tobytes())
    return {"algorithm": "sha256", "sha256": digest.hexdigest(), "rows": len(frame)}


@dataclass(frozen=True)
class FrozenSplitManifest:
    """Read-only source identities, split metadata, and group audit evidence."""

    source_ids: tuple[Hashable, ...]
    feature_order: tuple[str, ...]
    split_ids: Mapping[str, tuple[Hashable, ...]]
    metadata: Mapping[str, Any]
    group_keys: Mapping[Hashable, tuple[str, ...]]
    parity_ids: Mapping[str, tuple[Hashable, ...]] | None = None

    @property
    def split_sizes(self) -> dict[str, int]:
        return {name: len(self.split_ids[name]) for name in _SPLIT_NAMES}


def _validated_source_ids(frame: pd.DataFrame, source_column: str | None = None) -> tuple[Hashable, ...]:
    values = frame[source_column] if source_column else frame.index
    source_ids = tuple(values.tolist())
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("source row IDs must be unique")
    return source_ids


def _parity_subsets(split_ids, counts, *, seed, labels_by_id=None):
    if set(counts) != set(_SPLIT_NAMES):
        raise ValueError("parity counts must specify train, validation, and test")
    selected = {}
    for offset, name in enumerate(_SPLIT_NAMES, start=1):
        count, ids = counts[name], split_ids[name]
        if count < 1 or len(ids) < count:
            raise ValueError(f"{name} split has {len(ids)} rows; cannot freeze parity subset of {count}")
        stratify = None if labels_by_id is None else np.asarray([labels_by_id[source_id] for source_id in ids])
        chosen, _ = train_test_split(np.arange(len(ids)), train_size=count, random_state=seed + offset, stratify=stratify)
        selected[name] = tuple(ids[position] for position in chosen)
    return _freeze_mapping(selected)


def build_diabetes_freeze(frame, *, seed=RANDOM_STATE, expected_rows=DIABETES_FULL_ROWS, parity_counts=DIABETES_PARITY_COUNTS):
    """Freeze canonical BRFSS IDs using the exact A03 stratified split calls."""
    validate_schema(frame.columns)
    if len(frame) != expected_rows:
        raise ValueError(f"Diabetes freeze requires exactly {expected_rows} rows; received {len(frame)}")
    source_ids, positions, labels = _validated_source_ids(frame), np.arange(len(frame)), frame[TARGET].to_numpy()
    train_positions, held_out_positions = train_test_split(positions, test_size=0.30, random_state=seed, stratify=labels)
    validation_positions, test_positions = train_test_split(
        held_out_positions, test_size=0.50, random_state=seed, stratify=labels[held_out_positions]
    )
    split_ids = _freeze_mapping({
        "train": tuple(source_ids[position] for position in train_positions),
        "validation": tuple(source_ids[position] for position in validation_positions),
        "test": tuple(source_ids[position] for position in test_positions),
    })
    labels_by_id = dict(zip(source_ids, labels, strict=True))
    return FrozenSplitManifest(
        source_ids=source_ids, feature_order=FEATURE_ORDER, split_ids=split_ids,
        parity_ids=None if parity_counts is None else _parity_subsets(split_ids, parity_counts, seed=seed, labels_by_id=labels_by_id),
        metadata=_freeze_mapping({
            "dataset": "diabetes_brfss2015", "seed": seed, "source_rows": len(frame),
            "split_ratios": _SPLIT_RATIOS, "stratified_target": TARGET,
            "parity_counts": None if parity_counts is None else dict(parity_counts), "source_fingerprint": _frame_fingerprint(frame),
        }), group_keys=MappingProxyType({}),
    )


def load_canonical_house_modeling_sample(csv_path, *, expected_raw_rows=HOUSE_EXPECTED_RAW_ROWS, sample_size=HOUSE_SOURCE_SAMPLE_SIZE, chunksize=250_000, seed=RANDOM_STATE):
    """Reproduce A03 candidate RNG, validity filter, and pandas sample with raw IDs."""
    path, rng, candidate_parts, raw_rows = Path(csv_path), np.random.default_rng(seed), [], 0
    sampling_fraction = min(1.0, 1.35 * sample_size / expected_raw_rows)
    for chunk in pd.read_csv(path, usecols=list(HOUSE_PROFILE_COLUMNS), chunksize=chunksize):
        source_ids = np.arange(raw_rows, raw_rows + len(chunk), dtype=np.int64)
        raw_rows += len(chunk)
        keep = rng.random(len(chunk)) < sampling_fraction
        candidate = chunk.loc[keep, list(HOUSE_PROFILE_COLUMNS)].copy()
        candidate[HOUSE_SOURCE_ROW_ID] = source_ids[keep]
        candidate_parts.append(candidate)
    if raw_rows != expected_raw_rows:
        raise ValueError(f"House source requires exactly {expected_raw_rows} rows; received {raw_rows}")
    candidates = pd.concat(candidate_parts, ignore_index=True)
    valid = candidates[HOUSE_TARGET].notna() & candidates[HOUSE_TARGET].gt(0)
    valid &= candidates["house_size"].isna() | candidates["house_size"].gt(0)
    for feature in ("bed", "bath", "acre_lot"):
        valid &= candidates[feature].isna() | candidates[feature].ge(0)
    valid_candidates = candidates.loc[valid]
    model_frame = valid_candidates.sample(n=min(sample_size, len(valid_candidates)), random_state=seed).reset_index(drop=True)
    return model_frame, _freeze_mapping({
        "canonical_loader": True, "seed": seed, "raw_rows": raw_rows, "candidate_rows": len(candidates),
        "valid_rows": len(valid_candidates), "modeling_rows": len(model_frame), "sample_size": sample_size,
        "sampling_fraction": sampling_fraction, "source_fingerprint": _file_fingerprint(path),
    })


def _validate_house_modeling_frame(frame):
    missing = sorted(set((*HOUSE_MODEL_COLUMNS, *IDENTITY_COLUMNS, HOUSE_SOURCE_ROW_ID)) - set(frame.columns))
    if missing:
        raise ValueError(f"House modeling columns missing: {missing}")
    _validated_source_ids(frame, HOUSE_SOURCE_ROW_ID)
    valid = frame[HOUSE_TARGET].notna() & frame[HOUSE_TARGET].gt(0)
    valid &= frame["house_size"].isna() | frame["house_size"].gt(0)
    for feature in ("bed", "bath", "acre_lot"):
        valid &= frame[feature].isna() | frame[feature].ge(0)
    if not bool(valid.all()):
        raise ValueError(f"House modeling frame has {int((~valid).sum())} invalid price/features rows")


def _property_group_key(row, source_id):
    normalized = []
    for column in IDENTITY_COLUMNS:
        value = row[column]
        if pd.isna(value) or not str(value).strip():
            return ("fallback-source-row", f"{type(source_id).__name__}:{source_id!r}")
        normalized.append(str(value).strip().casefold())
    return ("property", *normalized)


def _group_aware_split(sampled, source_ids: Sequence[Hashable], *, seed, ratio_tolerance):
    group_rows, row_groups = {}, {}
    for source_id, (_, row) in zip(source_ids, sampled.iterrows(), strict=True):
        group = _property_group_key(row, source_id)
        group_rows.setdefault(group, []).append(source_id)
        row_groups[source_id] = group
    groups = list(group_rows)
    ranks = {groups[position]: rank for rank, position in enumerate(np.random.default_rng(seed).permutation(len(groups)))}
    groups.sort(key=lambda group: (-len(group_rows[group]), ranks[group]))
    target_sizes, assigned = ({name: len(sampled) * _SPLIT_RATIOS[name] for name in _SPLIT_NAMES}, {name: [] for name in _SPLIT_NAMES})
    for group in groups:
        split = min(_SPLIT_NAMES, key=lambda name: len(assigned[name]) / target_sizes[name])
        assigned[split].extend(group_rows[group])
    actual = {name: len(assigned[name]) / len(sampled) for name in _SPLIT_NAMES}
    deviations = {name: abs(actual[name] - _SPLIT_RATIOS[name]) for name in _SPLIT_NAMES}
    largest_group = max(map(len, group_rows.values()))
    if max(deviations.values()) > ratio_tolerance:
        raise ValueError(f"group-aware ratio tolerance exceeded (largest_group={largest_group}, deviations={deviations}, tolerance={ratio_tolerance})")
    split_groups = [{row_groups[source_id] for source_id in assigned[name]} for name in _SPLIT_NAMES]
    zero_overlap = all(not (split_groups[left] & split_groups[right]) for left in range(3) for right in range(left + 1, 3))
    if not zero_overlap:
        raise RuntimeError("property group overlap detected")
    audit = _freeze_mapping({"groups": len(group_rows), "largest_group": largest_group, "actual_ratios": actual,
                             "ratio_deviations": deviations, "ratio_tolerance": ratio_tolerance, "zero_group_overlap": zero_overlap})
    return _freeze_mapping({name: tuple(ids) for name, ids in assigned.items()}), _freeze_mapping(row_groups), audit


def build_house_freeze(frame, *, seed=RANDOM_STATE, sample_size=HOUSE_SOURCE_SAMPLE_SIZE, parity_counts=HOUSE_PARITY_COUNTS,
                       ratio_tolerance=HOUSE_RATIO_TOLERANCE, loader_metadata=None):
    """Freeze validated House rows, preserving raw IDs and property-group boundaries."""
    _validate_house_modeling_frame(frame)
    if not 0 <= ratio_tolerance < 1:
        raise ValueError("ratio tolerance must be in [0, 1)")
    if loader_metadata is not None:
        if not loader_metadata.get("canonical_loader") or loader_metadata.get("modeling_rows") != len(frame):
            raise ValueError("loader metadata does not prove a canonical modeling frame")
        sampled, source_fingerprint = frame, loader_metadata["source_fingerprint"]
    else:
        if len(frame) < sample_size:
            raise ValueError(f"House freeze requires at least {sample_size} source rows; received {len(frame)}")
        sampled = frame.iloc[np.random.default_rng(seed).choice(len(frame), size=sample_size, replace=False)]
        source_fingerprint = _frame_fingerprint(frame)
    source_ids = _validated_source_ids(sampled, HOUSE_SOURCE_ROW_ID)
    split_ids, group_keys, audit = _group_aware_split(sampled, source_ids, seed=seed, ratio_tolerance=ratio_tolerance)
    return FrozenSplitManifest(
        source_ids=source_ids, feature_order=(), split_ids=split_ids, group_keys=group_keys,
        parity_ids=_parity_subsets(split_ids, parity_counts, seed=seed),
        metadata=_freeze_mapping({
            "dataset": "realtor_house", "seed": seed, "source_rows": len(frame), "sample_size": len(sampled),
            "split_ratios": _SPLIT_RATIOS, "identity_columns": IDENTITY_COLUMNS,
            "split_sizes": {name: len(split_ids[name]) for name in _SPLIT_NAMES}, "parity_counts": dict(parity_counts),
            "source_fingerprint": source_fingerprint, "group_audit": audit,
        }),
    )


def _index_bytes(manifest):
    rows = [["source_id_json", "split", "parity", "group_sha256"]]
    parity_lookup = {source_id: name for name, ids in (manifest.parity_ids or {}).items() for source_id in ids}
    for split, ids in manifest.split_ids.items():
        for source_id in ids:
            rows.append([json.dumps(_thaw(source_id), ensure_ascii=True), split, parity_lookup.get(source_id, ""),
                         _sha256_bytes(_canonical_bytes(manifest.group_keys.get(source_id, ())))])
    from io import StringIO
    stream = StringIO(newline="")
    csv.writer(stream, lineterminator="\n").writerows(rows)
    return stream.getvalue().encode("utf-8")


def _atomic_write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(content); output.flush(); os.fsync(output.fileno())
        os.replace(temporary, path)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def _manifest_payload(manifest, index_sha256):
    return {
        "format_version": _FORMAT_VERSION, "source_ids": _thaw(manifest.source_ids), "feature_order": _thaw(manifest.feature_order),
        "split_ids": _thaw(manifest.split_ids), "parity_ids": _thaw(manifest.parity_ids), "metadata": _thaw(manifest.metadata),
        "group_keys": {json.dumps(_thaw(key), ensure_ascii=True): _thaw(value) for key, value in manifest.group_keys.items()},
        "index_sha256": index_sha256,
    }


def write_frozen_manifest(manifest, path):
    """Atomically write checksummed manifest plus target-free source-ID index CSV."""
    path = Path(path)
    index_path = path.with_suffix(".index.csv")
    index_content = _index_bytes(manifest)
    payload = _manifest_payload(manifest, _sha256_bytes(index_content))
    payload["manifest_sha256"] = _sha256_bytes(_canonical_bytes(payload))
    manifest_content = _canonical_bytes(payload)
    if path.exists() or index_path.exists():
        if not (path.exists() and index_path.exists()):
            raise FileExistsError("freeze outputs are incomplete; refusing overwrite")
        verify_frozen_manifest(path)
        if path.read_bytes() == manifest_content and index_path.read_bytes() == index_content:
            return index_path
        raise FileExistsError("different frozen manifest already exists; refusing overwrite")
    _atomic_write(index_path, index_content)
    _atomic_write(path, manifest_content)
    return index_path


def load_frozen_manifest(path):
    """Load only a checksum-verified, provenance-bearing freeze manifest."""
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    checksum = payload.pop("manifest_sha256", None)
    if checksum != _sha256_bytes(_canonical_bytes(payload)):
        raise ValueError("manifest checksum verification failed")
    if payload.get("format_version") != _FORMAT_VERSION:
        raise ValueError("unsupported freeze manifest format")
    fingerprint = payload.get("metadata", {}).get("source_fingerprint", {})
    if fingerprint.get("algorithm") != "sha256" or not fingerprint.get("sha256"):
        raise ValueError("manifest lacks source SHA256 fingerprint")
    if _sha256_bytes(path.with_suffix(".index.csv").read_bytes()) != payload.get("index_sha256"):
        raise ValueError("index checksum verification failed")
    group_keys = {json.loads(key): value for key, value in payload["group_keys"].items()}
    return FrozenSplitManifest(
        source_ids=tuple(payload["source_ids"]), feature_order=tuple(payload["feature_order"]),
        split_ids=_freeze_mapping({name: tuple(ids) for name, ids in payload["split_ids"].items()}),
        parity_ids=None if payload["parity_ids"] is None else _freeze_mapping({name: tuple(ids) for name, ids in payload["parity_ids"].items()}),
        metadata=_freeze_mapping(payload["metadata"]), group_keys=_freeze_mapping(group_keys),
    )


def verify_frozen_manifest(path):
    load_frozen_manifest(path)
    return True
