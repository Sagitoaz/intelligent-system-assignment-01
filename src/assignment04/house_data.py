"""House field definitions and property-identity audit helpers."""

from collections import Counter

import numpy as np
import pandas as pd

FIELD_ORDER = ("state", "status", "acre_lot", "house_size", "bed", "bath")
IDENTITY_COLUMNS = ("street", "city", "state", "zip_code")


class CategoryVocabulary:
    """Train-only category mapping; zero is reserved for missing/OOV."""

    def fit(self, values):
        categories = sorted({str(value) for value in values if value is not None and not _is_nan(value)})
        self.mapping = {category: index + 1 for index, category in enumerate(categories)}
        return self

    def transform(self, values):
        if not hasattr(self, "mapping"):
            raise RuntimeError("fit must be called on training categories first")
        return np.asarray([0 if value is None or _is_nan(value) else self.mapping.get(str(value), 0) for value in values], dtype=np.int64)


def _is_nan(value):
    try:
        return bool(np.isnan(value))
    except TypeError:
        return False


def property_identity_quality(rows):
    keys = []
    total = 0
    for row in rows:
        total += 1
        values = tuple(row.get(column) for column in IDENTITY_COLUMNS)
        if all(value is not None and not _is_nan(value) and str(value).strip() for value in values):
            keys.append(tuple(str(value).strip().lower() for value in values))
    counts = Counter(keys)
    return {
        "total_rows": total,
        "complete_rows": len(keys),
        "unique_groups": len(counts),
        "duplicate_rows": sum(count - 1 for count in counts.values()),
    }


def audit_property_identity_csv(path, chunksize=250_000):
    """Audit a candidate split-group key without loading model features."""
    statistics = {}
    total_rows = 0
    complete_rows = 0
    usecols = [*IDENTITY_COLUMNS, "price"]
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize):
        total_rows += len(chunk)
        complete = chunk.loc[chunk[list(IDENTITY_COLUMNS)].notna().all(axis=1)]
        complete_rows += len(complete)
        for row in complete.itertuples(index=False):
            key = tuple(str(getattr(row, column)).strip().lower() for column in IDENTITY_COLUMNS)
            price = float(row.price) if not _is_nan(row.price) else np.nan
            count, minimum, maximum = statistics.get(key, (0, np.inf, -np.inf))
            if np.isfinite(price):
                minimum, maximum = min(minimum, price), max(maximum, price)
            statistics[key] = (count + 1, minimum, maximum)
    return {
        "total_rows": total_rows,
        "complete_rows": complete_rows,
        "unique_groups": len(statistics),
        "duplicate_rows": sum(count - 1 for count, _, _ in statistics.values()),
        "groups_with_price_conflicts": sum(count > 1 and minimum < maximum for count, minimum, maximum in statistics.values()),
    }
