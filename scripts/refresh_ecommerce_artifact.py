from pathlib import Path
import joblib
import numpy as np

root = Path(__file__).resolve().parents[1]
path = root / "models/ecommerce/ecommerce_interest_model.joblib"
pipeline = joblib.load(path)
raw = np.asarray(["Summary", "Text", "HelpfulnessNumerator", "HelpfulnessDenominator"], dtype=object)
for _, branch in pipeline.named_steps["features"].transformer_list:
    branch.steps[0][1].feature_names_in_ = raw.copy()
joblib.dump(pipeline, path, compress=3)
print(list(pipeline.feature_names_in_))
