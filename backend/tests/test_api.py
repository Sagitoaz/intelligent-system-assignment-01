from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd

from app.core.config import settings


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "models": {"diabetes": "loaded", "house_price": "loaded", "ecommerce": "loaded"},
        "neo4j": "disabled",
    }


def test_models_metadata(client):
    response = client.get("/api/v1/models")
    assert response.status_code == 200
    models = response.json()["models"]
    assert [model["id"] for model in models] == ["diabetes", "house_price", "ecommerce"]
    assert len(models[0]["expected_raw_features"]) == 6
    assert len(models[1]["expected_raw_features"]) == 11
    assert models[1]["target"]["unit"] == "billion VND"
    assert models[2]["final_transformed_dimension"] == 12005

    for slug in ("diabetes", "house-price", "ecommerce"):
        assert client.get(f"/api/v1/models/{slug}").status_code == 200


def test_diabetes_model_loaded(client):
    assert client.app.state.model_service.models["diabetes"] is not None


def test_house_model_loaded(client):
    assert client.app.state.model_service.models["house_price"] is not None


def test_ecommerce_model_loaded(client):
    assert client.app.state.model_service.models["ecommerce"] is not None


def test_diabetes_prediction_success(client, diabetes_demo_case):
    response = client.post("/api/v1/diabetes/predict", json=diabetes_demo_case)
    assert response.status_code == 200
    result = response.json()

    saved_model = joblib.load(settings.diabetes_model_path)
    frame = pd.DataFrame([diabetes_demo_case], columns=saved_model.feature_names_in_)
    expected_prediction = int(saved_model.predict(frame)[0])
    expected_probability = float(saved_model.predict_proba(frame)[0, 1])

    assert result["prediction"] == expected_prediction == 0
    assert result["label"] == "Non-diabetic"
    assert result["probability"] == pytest.approx(expected_probability)


def test_house_prediction_success(client, house_demo_case):
    response = client.post("/api/v1/house-price/predict", json=house_demo_case)
    assert response.status_code == 200
    result = response.json()

    saved_model = joblib.load(settings.house_price_model_path)
    frame = pd.DataFrame([house_demo_case], columns=saved_model.feature_names_in_)
    expected = float(saved_model.predict(frame)[0])

    assert result["predicted_price"] == pytest.approx(expected, abs=1e-10)
    assert result["predicted_price"] == pytest.approx(5.12666185, abs=1e-7)
    assert result["unit"] == "billion VND"


def test_ecommerce_prediction_success(client, ecommerce_demo_case):
    response = client.post("/api/v1/ecommerce/predict", json=ecommerce_demo_case)
    assert response.status_code == 200
    result = response.json()
    saved_model = joblib.load(settings.ecommerce_model_path)
    frame = pd.DataFrame([ecommerce_demo_case], columns=saved_model.feature_names_in_)
    assert result["prediction"] == int(saved_model.predict(frame)[0]) == 1
    assert result["probability"] == pytest.approx(float(saved_model.predict_proba(frame)[0, 1]))
    assert result["label"] == "Positive"


def test_diabetes_invalid_input(client, diabetes_demo_case):
    invalid = {**diabetes_demo_case, "Glucose": float("inf")}
    response = client.post(
        "/api/v1/diabetes/predict",
        content=json.dumps(invalid, allow_nan=True),
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422

    missing = {key: value for key, value in diabetes_demo_case.items() if key != "Age"}
    assert client.post("/api/v1/diabetes/predict", json=missing).status_code == 422


def test_house_invalid_input(client, house_demo_case):
    invalid = {**house_demo_case, "Area": -1}
    assert client.post("/api/v1/house-price/predict", json=invalid).status_code == 422

    unknown_field = {**house_demo_case, "EncodedProvince": 7}
    assert client.post("/api/v1/house-price/predict", json=unknown_field).status_code == 422

    missing = {key: value for key, value in house_demo_case.items() if key != "Furniture state"}
    assert client.post("/api/v1/house-price/predict", json=missing).status_code == 422


def test_ecommerce_invalid_input(client, ecommerce_demo_case):
    assert client.post("/api/v1/ecommerce/predict", json={**ecommerce_demo_case, "Text": "   "}).status_code == 422
    assert client.post("/api/v1/ecommerce/predict", json={**ecommerce_demo_case, "HelpfulnessNumerator": 2, "HelpfulnessDenominator": 1}).status_code == 422
    assert client.post("/api/v1/ecommerce/predict", json={**ecommerce_demo_case, "Score": 5}).status_code == 422


def test_house_nullable_raw_values_use_saved_pipeline_imputation(client, house_demo_case):
    payload = {
        **house_demo_case,
        "Frontage": None,
        "Access Road": None,
        "House direction": None,
        "Balcony direction": None,
        "Furniture state": None,
    }
    response = client.post("/api/v1/house-price/predict", json=payload)
    assert response.status_code == 200
    assert np.isfinite(response.json()["predicted_price"])


def test_feature_schema_matches_saved_model(client):
    service = client.app.state.model_service
    for key in ("diabetes", "house_price", "ecommerce"):
        assert service.metadata[key]["expected_raw_features"] == list(
            service.models[key].feature_names_in_
        )


def test_model_is_not_refit_during_prediction(client, diabetes_demo_case, house_demo_case, ecommerce_demo_case, monkeypatch):
    service = client.app.state.model_service

    def fail_fit(*_args, **_kwargs):
        raise AssertionError("fit must never be called during inference")

    monkeypatch.setattr(service.models["diabetes"], "fit", fail_fit)
    monkeypatch.setattr(service.models["house_price"], "fit", fail_fit)
    monkeypatch.setattr(service.models["ecommerce"], "fit", fail_fit)

    assert client.post("/api/v1/diabetes/predict", json=diabetes_demo_case).status_code == 200
    assert client.post("/api/v1/house-price/predict", json=house_demo_case).status_code == 200
    assert client.post("/api/v1/ecommerce/predict", json=ecommerce_demo_case).status_code == 200


def test_knowledge_graph_unavailable_does_not_break_predictions(client, diabetes_demo_case, ecommerce_demo_case):
    graph_response = client.get("/api/v1/diabetes/knowledge-graph")
    assert graph_response.status_code == 503
    assert client.post("/api/v1/diabetes/predict", json=diabetes_demo_case).status_code == 200
    assert client.post("/api/v1/ecommerce/predict", json=ecommerce_demo_case).status_code == 200


import pytest  # placed last to keep fixtures/readability above concise
