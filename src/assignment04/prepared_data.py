"""Train-only transformations used to materialize the three parity datasets."""

from __future__ import annotations

from dataclasses import dataclass
import csv

import numpy as np
import pandas as pd

from .diabetes_data import FEATURE_ORDER, SCALED_FEATURES
from .house_data import CategoryVocabulary
from .comments_data import normalized_text_sha256


@dataclass
class DiabetesPreprocessor:
    means: np.ndarray | None = None
    scales: np.ndarray | None = None

    def fit(self, train: pd.DataFrame):
        missing = set(FEATURE_ORDER) - set(train.columns)
        if missing:
            raise ValueError(f"Missing Diabetes features: {sorted(missing)}")
        scaled = train.loc[:, SCALED_FEATURES].to_numpy(dtype=np.float64)
        self.means = scaled.mean(axis=0)
        self.scales = scaled.std(axis=0)
        self.scales[self.scales == 0] = 1.0
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if self.means is None:
            raise RuntimeError("fit must be called on master training rows first")
        values = frame.loc[:, FEATURE_ORDER].to_numpy(dtype=np.float32, copy=True)
        positions = [FEATURE_ORDER.index(name) for name in SCALED_FEATURES]
        values[:, positions] = (values[:, positions] - self.means) / self.scales
        return values[:, None, :].astype(np.float32, copy=False)

    def metadata(self):
        return {
            "fit_split": "master_train_only",
            "feature_order": list(FEATURE_ORDER),
            "scaled_features": list(SCALED_FEATURES),
            "means": self.means.tolist(),
            "scales": self.scales.tolist(),
            "output_shape_per_row": [1, 21],
        }


class HousePreprocessor:
    numeric_columns = ("acre_lot", "house_size", "bed", "bath")
    field_order = ("state", "status", *numeric_columns)
    field_shape = (8, 6)

    def fit(self, train: pd.DataFrame):
        numeric = train.loc[:, self.numeric_columns].to_numpy(dtype=np.float64)
        self.numeric_medians = np.nanmedian(numeric, axis=0)
        imputed = np.where(np.isnan(numeric), self.numeric_medians, numeric)
        self.numeric_means = imputed.mean(axis=0)
        self.numeric_scales = imputed.std(axis=0)
        self.numeric_scales[self.numeric_scales == 0] = 1.0
        self.state_vocabulary = CategoryVocabulary().fit(train["state"])
        self.status_vocabulary = CategoryVocabulary().fit(train["status"])
        log_target = np.log1p(train["price"].to_numpy(dtype=np.float64))
        self.target_mean = float(log_target.mean())
        self.target_scale = float(log_target.std())
        if self.target_scale == 0:
            self.target_scale = 1.0
        return self

    def transform(self, frame: pd.DataFrame):
        if not hasattr(self, "target_mean"):
            raise RuntimeError("fit must be called on master training rows first")
        numeric = frame.loc[:, self.numeric_columns].to_numpy(dtype=np.float64)
        numeric = np.where(np.isnan(numeric), self.numeric_medians, numeric)
        numeric = ((numeric - self.numeric_means) / self.numeric_scales).astype(np.float32)
        target = np.log1p(frame["price"].to_numpy(dtype=np.float64))
        target = ((target - self.target_mean) / self.target_scale).astype(np.float32)[:, None]
        return {
            "numeric": numeric,
            "state": self.state_vocabulary.transform(frame["state"]),
            "status": self.status_vocabulary.transform(frame["status"]),
            "y": target,
        }

    def inverse_target(self, standardized):
        model_scale = np.asarray(standardized, dtype=np.float64).reshape(-1) * self.target_scale + self.target_mean
        return np.expm1(model_scale)

    @property
    def state_size(self):
        return len(self.state_vocabulary.mapping) + 1

    @property
    def status_size(self):
        return len(self.status_vocabulary.mapping) + 1

    def metadata(self):
        return {
            "fit_split": "master_train_only",
            "field_order": list(self.field_order),
            "field_representation_dimension": 8,
            "numeric_columns": list(self.numeric_columns),
            "numeric_medians": self.numeric_medians.tolist(),
            "numeric_means": self.numeric_means.tolist(),
            "numeric_scales": self.numeric_scales.tolist(),
            "state_vocabulary": self.state_vocabulary.mapping,
            "status_vocabulary": self.status_vocabulary.mapping,
            "target_transform": "standardize(log1p(price)) with master-train statistics",
            "target_mean": self.target_mean,
            "target_scale": self.target_scale,
            "output": "one unconstrained linear value",
        }


def load_comment_rows(source_csv, index_csv, *, expected_source: str):
    """Rehydrate one frozen index list in manifest order from an official CSV."""
    with open(index_csv, newline="", encoding="utf-8") as stream:
        manifest_rows = list(csv.DictReader(stream))
    if any(row["source"] != expected_source for row in manifest_rows):
        raise ValueError(f"Index file contains rows outside {expected_source}")
    positions = {int(row["row_index"]): offset for offset, row in enumerate(manifest_rows)}
    if len(positions) != len(manifest_rows):
        raise ValueError("Frozen comment row indices must be unique")
    texts = [None] * len(manifest_rows)
    labels = np.empty(len(manifest_rows), dtype=np.int64)
    with open(source_csv, newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not {"Review", "Label"}.issubset(reader.fieldnames):
            raise ValueError("Comments CSV must contain Review and Label")
        for row_index, row in enumerate(reader):
            offset = positions.get(row_index)
            if offset is not None:
                texts[offset] = row.get("Review") or ""
                if normalized_text_sha256(texts[offset]) != manifest_rows[offset]["text_sha256"]:
                    raise ValueError(f"Frozen text hash mismatch at source row {row_index}")
                raw_label = int(row["Label"])
                if raw_label not in (1, 2):
                    raise ValueError(f"Unsupported Comments label {raw_label}")
                labels[offset] = raw_label - 1
    if any(text is None for text in texts):
        missing = sum(text is None for text in texts)
        raise ValueError(f"Could not reconstruct {missing} frozen comment rows")
    return texts, labels
