from __future__ import annotations

import copy

import pytest

from app.services.model_service import ModelContractError, ModelService


def test_verify_model_contract_detects_mismatch(client):
    service = client.app.state.model_service
    metadata = copy.deepcopy(service.metadata["house_price"])
    metadata["expected_raw_features"] = metadata["expected_raw_features"][:6]

    with pytest.raises(ModelContractError, match="metadata fields|saved pipeline features"):
        ModelService.verify_model_contract(
            "house_price",
            service.models["house_price"],
            metadata,
        )


def test_house_encoder_ignores_unknown_category(client, house_demo_case):
    payload = {**house_demo_case, "Province": "Unseen educational province"}
    response = client.post("/api/v1/house-price/predict", json=payload)
    assert response.status_code == 200
