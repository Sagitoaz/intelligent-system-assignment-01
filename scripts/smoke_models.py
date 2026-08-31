"""Fresh-process smoke test for the three trusted deployment artifacts."""
from pathlib import Path

import joblib
import pandas as pd

root = Path(__file__).resolve().parents[1]
cases = {
    "diabetes": (
        root / "models/diabetes/diabetes_model.joblib",
        {"Pregnancies": 1, "Glucose": 85, "BloodPressure": 66, "BMI": 24.0, "DiabetesPedigreeFunction": 0.2, "Age": 23},
    ),
    "house_price": (
        root / "models/house_price/house_price_model.joblib",
        {"Province": "Hồ Chí Minh", "Area": 45.0, "Frontage": 4.0, "Access Road": 4.0, "House direction": "Đông - Nam", "Balcony direction": "Đông - Nam", "Floors": 3.0, "Bedrooms": 3.0, "Bathrooms": 3.0, "Legal status": "Have certificate", "Furniture state": "Full"},
    ),
    "ecommerce": (
        root / "models/ecommerce/ecommerce_interest_model.joblib",
        {"Summary": "Excellent", "Text": "Fresh, tasty and exactly as described. I would buy it again.", "HelpfulnessNumerator": 1, "HelpfulnessDenominator": 1},
    ),
}

for name, (path, raw) in cases.items():
    pipeline = joblib.load(path)
    frame = pd.DataFrame([raw], columns=pipeline.feature_names_in_)
    prediction = pipeline.predict(frame)[0]
    assert prediction is not None
    print(f"{name}: LOAD_AND_PREDICT_PASS")
