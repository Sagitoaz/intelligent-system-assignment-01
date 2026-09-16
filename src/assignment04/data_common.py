"""Framework-neutral, train-fitted preprocessing helpers."""

import numpy as np


class TrainOnlyStandardizer:
    def fit(self, values):
        values = np.asarray(values, dtype=np.float64)
        self.mean_ = values.mean(axis=0)
        self.scale_ = values.std(axis=0)
        self.scale_[self.scale_ == 0] = 1.0
        return self

    def transform(self, values):
        if not hasattr(self, "mean_"):
            raise RuntimeError("fit must be called on training data first")
        return (np.asarray(values, dtype=np.float64) - self.mean_) / self.scale_

    def fit_transform(self, values):
        return self.fit(values).transform(values)
