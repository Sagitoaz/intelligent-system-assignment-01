from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class ReviewTextTransformer(BaseEstimator, TransformerMixin):
    """Combine raw summary and review body into one deployable text document."""

    def fit(self, X: pd.DataFrame, y=None):
        self.feature_names_in_ = np.asarray(list(pd.DataFrame(X).columns), dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        frame = pd.DataFrame(X, columns=["Summary", "Text"])
        summary = frame["Summary"].fillna("").astype(str).str.strip()
        body = frame["Text"].fillna("").astype(str).str.strip()
        return (summary + " " + body).str.strip().to_numpy()


class ReviewTabularTransformer(BaseEstimator, TransformerMixin):
    """Derive the five documented behavioral/length features from raw fields."""

    feature_names = np.array([
        "HelpfulnessNumerator",
        "HelpfulnessDenominator",
        "helpfulness_ratio",
        "review_length",
        "summary_length",
    ])

    def fit(self, X: pd.DataFrame, y=None):
        self.feature_names_in_ = np.asarray(list(pd.DataFrame(X).columns), dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        frame = pd.DataFrame(
            X,
            columns=["Summary", "Text", "HelpfulnessNumerator", "HelpfulnessDenominator"],
        )
        numerator = pd.to_numeric(frame["HelpfulnessNumerator"], errors="coerce").fillna(0).clip(lower=0)
        denominator = pd.to_numeric(frame["HelpfulnessDenominator"], errors="coerce").fillna(0).clip(lower=0)
        summary = frame["Summary"].fillna("").astype(str)
        body = frame["Text"].fillna("").astype(str)
        ratio = numerator / denominator.clip(lower=1)
        return np.column_stack([
            numerator.to_numpy(float),
            denominator.to_numpy(float),
            ratio.to_numpy(float),
            body.str.split().str.len().to_numpy(float),
            summary.str.split().str.len().to_numpy(float),
        ])

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return self.feature_names.copy()
